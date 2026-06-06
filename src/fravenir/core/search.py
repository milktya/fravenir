"""Episode search: vector KNN + graph traversal + ACT-R re-ranking."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import structlog

from fravenir.core.activation import base_activation, final_score
from fravenir.core.graph import (
    build_subgraph_from_seeds,
    fan_out_of,
    reach_episodes,
    s_ji,
)
from fravenir.core.self_cue import self_cue_terms
from fravenir.embedding import Embedder
from fravenir.schemas.config import AppConfig
from fravenir.storage import paths
from fravenir.storage.vector import (
    l2_to_cosine,
    search_entities_by_vector,
    search_episodes_by_vector,
)

_ENTITY_TOP_K = 10


def memory_search(
    query: str,
    limit: int = 5,
    kind_filter: list[str] | None = None,
    min_importance: int = 1,
    include_archived: bool = False,
    include_suppressed: bool = False,
    *,
    character_id: str,
    config: AppConfig,
    embedder: Embedder,
) -> list[dict[str, object]]:
    """Search episodes by vector similarity + graph traversal, re-ranked by ACT-R.

    Phase3: vdb_memories で直接ヒットする episode に加え、vdb_entities で近い
    entity を起点に 2ホップ辿り、関連 episode を associative 候補として合流させる。
    activation には連想強度 $S_{ji} = S_{max} - ln(fan_j)$ を加算する。
    """
    actr = config.act_r
    top_k = max(limit * 4, 20)

    q_doc_vec = embedder.encode_query(query)
    q_topic_vec = embedder.encode_topic(query)

    vdb_mem_path = paths.vdb_memories_path(character_id)
    vdb_ent_path = paths.vdb_entities_path(character_id)

    direct_hits: list[tuple[int, float]] = []
    vdb_conn = _open_vdb(str(vdb_mem_path))
    try:
        direct_hits = search_episodes_by_vector(vdb_conn, q_doc_vec, top_k)
    finally:
        vdb_conn.close()

    entity_hits: list[tuple[int, float]] = []
    if vdb_ent_path.exists():
        vdb_conn = _open_vdb(str(vdb_ent_path))
        try:
            entity_hits = search_entities_by_vector(
                vdb_conn, q_topic_vec, _ENTITY_TOP_K
            )
        finally:
            vdb_conn.close()

    kv_conn = sqlite3.connect(str(paths.kv_db_path(character_id)))
    try:
        now = datetime.now(UTC)

        cue_terms = self_cue_terms(kv_conn)
        self_cue_hit = any(t and t in query for t in cue_terms)

        seed_ids = [eid for eid, _ in entity_hits]
        self_entity_ids: list[int] = []
        if self_cue_hit:
            self_entity_ids = _fetch_self_entity_ids(kv_conn)
            seed_ids = list(dict.fromkeys(seed_ids + self_entity_ids))

        assoc_contrib, assoc_episode_ids, boost_episode_ids = _graph_contributions(
            kv_conn,
            seed_ids,
            s_max=actr.s_max,
            boost_seed_ids=set(self_entity_ids),
            boost_beta=actr.self_boost_beta,
        )

        if not direct_hits and not assoc_episode_ids:
            return []

        results = _rank_and_filter(
            kv_conn,
            direct_hits=direct_hits,
            assoc_episode_ids=assoc_episode_ids,
            assoc_contrib=assoc_contrib,
            boost_episode_ids=boost_episode_ids,
            now=now,
            kind_filter=kind_filter,
            min_importance=min_importance,
            include_archived=include_archived,
            include_suppressed=include_suppressed,
            base_decay=actr.base_decay,
            access_history_limit=actr.access_history_limit,
            alpha_sim=actr.alpha_similarity,
            alpha_imp=actr.alpha_importance,
            limit=limit,
        )
        _record_access(kv_conn, results, now)
    finally:
        kv_conn.close()

    return results


def _fetch_self_entity_ids(conn: sqlite3.Connection) -> list[int]:
    """is_self=1 かつ valid_to IS NULL の entity ids を返す。"""
    rows = conn.execute(
        "SELECT id FROM entities WHERE is_self = 1 AND valid_to IS NULL"
    ).fetchall()
    return [int(r[0]) for r in rows]


def _graph_contributions(
    conn: sqlite3.Connection,
    seed_entity_ids: list[int],
    *,
    s_max: float,
    boost_seed_ids: set[int] | None = None,
    boost_beta: float = 0.0,
) -> tuple[dict[int, float], set[int], set[int]]:
    """seed entities から 2ホップ辿り、到達 episode ごとの Σ W_j·S_ji を集計する。

    boost_seed_ids に含まれる seed については、辿った先の episode に対して
    追加で boost_beta を加算する（設計書 §5.4 自己ブースト）。

    Returns:
        (episode_id -> 活性化加算値, reached_episode_ids, boost_reached_episode_ids)
    """
    if not seed_entity_ids:
        return {}, set(), set()

    graph = build_subgraph_from_seeds(conn, seed_entity_ids, max_hops=2)
    if graph.number_of_edges() == 0:
        return {}, set(), set()

    weight = 1.0 / len(seed_entity_ids)
    contrib: dict[int, float] = {}
    reached: set[int] = set()
    boost_reached: set[int] = set()

    for seed_id in seed_entity_ids:
        fan_j = fan_out_of(conn, seed_id)
        strength = s_ji(fan_j, s_max)
        if strength <= 0:
            continue
        episodes = reach_episodes(graph, seed_id)
        reached.update(episodes)
        for ep_id in episodes:
            contrib[ep_id] = contrib.get(ep_id, 0.0) + weight * strength
            if boost_seed_ids and seed_id in boost_seed_ids:
                contrib[ep_id] += boost_beta
                boost_reached.add(ep_id)

    return contrib, reached, boost_reached


def _rank_and_filter(
    kv_conn: sqlite3.Connection,
    *,
    direct_hits: list[tuple[int, float]],
    assoc_episode_ids: set[int],
    assoc_contrib: dict[int, float],
    boost_episode_ids: set[int],
    now: datetime,
    kind_filter: list[str] | None,
    min_importance: int,
    include_archived: bool,
    include_suppressed: bool,
    base_decay: float,
    access_history_limit: int,
    alpha_sim: float,
    alpha_imp: float,
    limit: int,
) -> list[dict[str, object]]:
    l2_by_id = {ep_id: l2 for ep_id, l2 in direct_hits}
    direct_ids = set(l2_by_id.keys())
    all_ids = direct_ids | assoc_episode_ids
    if not all_ids:
        return []

    placeholders = ",".join("?" * len(all_ids))
    rows = kv_conn.execute(
        f"""
        SELECT id, content, kind, importance, valid_from, valid_to,
               supersedes, is_suppressed, created_at
        FROM episodes
        WHERE id IN ({placeholders})
        """,
        tuple(all_ids),
    ).fetchall()

    scored: list[tuple[float, dict[str, object]]] = []
    for row in rows:
        (ep_id, content, kind, importance, valid_from, valid_to,
         supersedes, is_suppressed, created_at) = row

        if not include_archived and valid_to is not None:
            continue
        if is_suppressed:
            if not include_suppressed:
                continue
            if importance < min_importance:
                continue
        if kind_filter and kind not in kind_filter:
            continue

        activation = base_activation(
            kv_conn, "episode", ep_id, base_decay, now, access_history_limit
        )
        sji_bonus = assoc_contrib.get(ep_id, 0.0)
        activation += sji_bonus

        is_direct = ep_id in direct_ids
        cosine_sim = l2_to_cosine(l2_by_id[ep_id]) if is_direct else 0.0

        boost_applied = ep_id in boost_episode_ids
        # boost_beta は既に _graph_contributions で sji_bonus に加算済み

        score = final_score(activation, cosine_sim, importance, alpha_sim, alpha_imp)

        if boost_applied:
            source = "self_boost"
        elif is_direct:
            source = "direct"
        else:
            source = "associative"

        scored.append((
            score,
            {
                "episode_id": ep_id,
                "content": content,
                "kind": kind,
                "importance": importance,
                "activation": activation,
                "score": score,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "supersedes": supersedes,
                "created_at": created_at,
                "source": source,
            },
        ))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


def _record_access(
    kv_conn: sqlite3.Connection,
    results: list[dict[str, object]],
    now: datetime,
) -> None:
    for item in results:
        ep_id = item["episode_id"]
        kv_conn.execute(
            "INSERT INTO access_history(node_type, node_id, accessed_at, source) VALUES (?,?,?,?)",
            ("episode", ep_id, now.isoformat(), item["source"]),
        )
        kv_conn.execute(
            """
            UPDATE episodes
            SET activation_count = activation_count + 1,
                last_activated_at = ?
            WHERE id = ?
            """,
            (now.isoformat(), ep_id),
        )
    kv_conn.commit()


def _open_vdb(db_path: str) -> sqlite3.Connection:
    import sqlite_vec

    try:
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn
    except (sqlite3.Error, OSError) as e:
        _log = structlog.get_logger(__name__)
        _log.exception("vdb_open_error", db_path=db_path, error=str(e))
        raise RuntimeError(f"failed to open vector DB at {db_path}") from e
