"""memory_explore (FEAT-1).

memory_search で当たりをつけたあと、AI が 1 ホップずつ深掘りするためのツール。
詳細仕様は docs/feat1_memory_explore_design.md を参照。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from fravenir.core.activation import base_activation
from fravenir.core.graph import s_ji
from fravenir.schemas.config import AppConfig
from fravenir.schemas.explore import (
    ExploreResult,
    NeighborItem,
    NodeContent,
    PredicateMeta,
)
from fravenir.storage import paths

_TRUNCATE_LEN = 800
_SUMMARY_LEN = 120
_PER_PREDICATE_LIMIT = 5
_TOTAL_LIMIT = 10
_EPISODE_DECAY = 0.5


@dataclass(frozen=True)
class _NeighborRow:
    neighbor_type: Literal["episode", "entity"]
    neighbor_id: int
    direction: Literal["incoming", "outgoing"]
    predicate: str
    strength: float
    name: str | None
    content: str
    neighbor_valid_to: str | None
    neighbor_is_suppressed: bool
    neighbor_decay_rate: float | None


def memory_explore(
    node_type: Literal["episode", "entity"],
    node_id: int,
    depth: int = 1,
    full: bool = False,
    exclude_episode_ids: list[int] | None = None,
    exclude_entity_ids: list[int] | None = None,
    include_archived: bool = False,
    include_suppressed: bool = False,
    *,
    character_id: str,
    config: AppConfig,
) -> ExploreResult:
    """グラフを 1 ホップ深掘り。"""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if depth >= 2:
        raise NotImplementedError(
            "depth >= 2 is reserved for future expansion; "
            "Phase 6 supports depth=1 only"
        )

    exclude_episodes = set(exclude_episode_ids or [])
    exclude_entities = set(exclude_entity_ids or [])
    actr = config.act_r

    kv_conn = sqlite3.connect(str(paths.kv_db_path(character_id)))
    try:
        now = datetime.now(UTC)

        node = _fetch_node(kv_conn, node_type, node_id, full=full)
        if node is None:
            raise ValueError(f"{node_type} with id={node_id} not found")

        # entity 起点の is_suppressed は動的判定（feat1 §6.3）
        if node.type == "entity" and node.decay_rate is not None:
            node_b_i = base_activation(
                kv_conn,
                "entity",
                node.id,
                node.decay_rate,
                now,
                actr.access_history_limit,
            )
            node.is_suppressed = node_b_i < actr.suppress_threshold

        raw_neighbors = _fetch_neighbors(
            kv_conn, node_type, node_id, include_archived=include_archived
        )
        total_unfiltered = len(raw_neighbors)

        parent_fan = _fan_out_of_node(kv_conn, node_type, node_id)

        scored: list[tuple[float, NeighborItem, str]] = []
        for r in raw_neighbors:
            if r.neighbor_type == "episode" and r.neighbor_id in exclude_episodes:
                continue
            if r.neighbor_type == "entity" and r.neighbor_id in exclude_entities:
                continue
            if not include_archived and r.neighbor_valid_to is not None:
                continue

            if (
                not include_suppressed
                and r.neighbor_type == "episode"
                and r.neighbor_is_suppressed
            ):
                continue

            decay = (
                r.neighbor_decay_rate
                if r.neighbor_type == "entity"
                else _EPISODE_DECAY
            )
            if decay is None:
                decay = _EPISODE_DECAY

            b_i = base_activation(
                kv_conn,
                r.neighbor_type,
                r.neighbor_id,
                decay,
                now,
                actr.access_history_limit,
            )

            # entity 側の suppressed は動的判定（v2_design §12.2 #6 で
            # entities.is_suppressed カラム保留のため）。
            if (
                not include_suppressed
                and r.neighbor_type == "entity"
                and b_i < actr.suppress_threshold
            ):
                continue

            # 連想強度: outgoing は parent の fan、incoming は neighbor 側の fan
            fan = (
                parent_fan
                if r.direction == "outgoing"
                else _fan_out_of_node(kv_conn, r.neighbor_type, r.neighbor_id)
            )
            sort_score = b_i + r.strength + s_ji(fan, actr.s_max)

            scored.append(
                (
                    sort_score,
                    NeighborItem(
                        type=r.neighbor_type,
                        id=r.neighbor_id,
                        summary=_make_summary(r),
                        direction=r.direction,
                        sort_score=sort_score,
                    ),
                    r.predicate,
                )
            )

        by_pred: dict[str, list[tuple[float, NeighborItem]]] = {}
        for score, item, pred in scored:
            by_pred.setdefault(pred, []).append((score, item))
        per_pred_total: dict[str, int] = {
            p: len(items) for p, items in by_pred.items()
        }
        for pred_items in by_pred.values():
            pred_items.sort(key=lambda x: x[0], reverse=True)
        per_pred_truncated = {
            p: items[:_PER_PREDICATE_LIMIT] for p, items in by_pred.items()
        }

        flat: list[tuple[float, NeighborItem, str]] = [
            (s, item, p)
            for p, items in per_pred_truncated.items()
            for s, item in items
        ]
        flat.sort(key=lambda x: x[0], reverse=True)
        flat = flat[:_TOTAL_LIMIT]

        neighbors: dict[str, list[NeighborItem]] = {}
        for _, item, pred in flat:
            neighbors.setdefault(pred, []).append(item)

        # 全 predicate を meta に含める（全体上限で落ちた predicate も shown=0 で残す、
        # feat1 §3.3 「見えてないけど存在する」を出力に反映するため）
        meta: dict[str, PredicateMeta] = {
            pred: PredicateMeta(
                shown=len(neighbors.get(pred, [])),
                total=per_pred_total[pred],
            )
            for pred in per_pred_total
        }
        total_shown = sum(len(items) for items in neighbors.values())

        _record_explore_access(kv_conn, node, neighbors, now)

        return ExploreResult(
            node=node,
            neighbors=neighbors,
            meta=meta,
            total_neighbors=total_shown,
            total_neighbors_unfiltered=total_unfiltered,
        )
    finally:
        kv_conn.close()


def _fetch_node(
    conn: sqlite3.Connection,
    node_type: Literal["episode", "entity"],
    node_id: int,
    *,
    full: bool,
) -> NodeContent | None:
    if node_type == "episode":
        row = conn.execute(
            """SELECT id, content, importance, valid_from, valid_to, is_suppressed
            FROM episodes WHERE id = ?""",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        ep_id, content, importance, valid_from, valid_to, is_suppressed = row
        truncated, is_truncated = _truncate(content, full=full)
        return NodeContent(
            type="episode",
            id=int(ep_id),
            name=None,
            content=truncated,
            is_truncated=is_truncated,
            importance=int(importance),
            valid_from=_to_datetime(valid_from),
            valid_to=_to_datetime(valid_to) if valid_to else None,
            is_suppressed=bool(is_suppressed),
        )

    if node_type == "entity":
        row = conn.execute(
            """SELECT id, canonical_name, description, is_self, self_weight,
                      decay_rate, valid_from, valid_to
            FROM entities WHERE id = ?""",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        (
            ent_id,
            name,
            desc,
            is_self,
            self_weight,
            decay_rate,
            valid_from,
            valid_to,
        ) = row
        truncated, is_truncated = _truncate(desc or "", full=full)
        # entities テーブルに importance カラムなし。固定値 1 を返す
        # （feat1 設計書 3.1 の NodeContent.importance を満たすため）。
        # is_suppressed は memory_explore 関数内で動的計算してから代入する。
        return NodeContent(
            type="entity",
            id=int(ent_id),
            name=str(name),
            content=truncated,
            is_truncated=is_truncated,
            importance=1,
            valid_from=_to_datetime(valid_from),
            valid_to=_to_datetime(valid_to) if valid_to else None,
            is_self=bool(is_self),
            self_weight=float(self_weight),
            decay_rate=float(decay_rate),
        )

    raise ValueError(f"invalid node_type: {node_type}")


def _fetch_neighbors(
    conn: sqlite3.Connection,
    node_type: str,
    node_id: int,
    *,
    include_archived: bool,
) -> list[_NeighborRow]:
    valid_clause = "" if include_archived else "AND valid_to IS NULL"
    rows = conn.execute(
        f"""SELECT src_type, src_id, dst_type, dst_id, predicate, strength
        FROM relations
        WHERE ((src_type = ? AND src_id = ?) OR (dst_type = ? AND dst_id = ?))
          {valid_clause}""",  # noqa: S608 — valid_clause は静的リテラルのみ
        (node_type, node_id, node_type, node_id),
    ).fetchall()

    result: list[_NeighborRow] = []
    for src_type, src_id, dst_type, dst_id, predicate, strength in rows:
        if src_type == node_type and src_id == node_id:
            neighbor_type_raw = dst_type
            neighbor_id = int(dst_id)
            direction: Literal["incoming", "outgoing"] = "outgoing"
        else:
            neighbor_type_raw = src_type
            neighbor_id = int(src_id)
            direction = "incoming"

        if neighbor_type_raw == "episode":
            n_row = conn.execute(
                """SELECT content, valid_to, is_suppressed
                FROM episodes WHERE id = ?""",
                (neighbor_id,),
            ).fetchone()
            if n_row is None:
                continue
            n_content, n_valid_to, is_suppressed = n_row
            result.append(
                _NeighborRow(
                    neighbor_type="episode",
                    neighbor_id=neighbor_id,
                    direction=direction,
                    predicate=str(predicate),
                    strength=float(strength),
                    name=None,
                    content=str(n_content),
                    neighbor_valid_to=n_valid_to,
                    neighbor_is_suppressed=bool(is_suppressed),
                    neighbor_decay_rate=None,
                )
            )
        elif neighbor_type_raw == "entity":
            n_row = conn.execute(
                """SELECT canonical_name, description, decay_rate, valid_to
                FROM entities WHERE id = ?""",
                (neighbor_id,),
            ).fetchone()
            if n_row is None:
                continue
            canonical_name, desc, decay_rate, n_valid_to = n_row
            result.append(
                _NeighborRow(
                    neighbor_type="entity",
                    neighbor_id=neighbor_id,
                    direction=direction,
                    predicate=str(predicate),
                    strength=float(strength),
                    name=str(canonical_name),
                    content=str(desc or ""),
                    neighbor_valid_to=n_valid_to,
                    neighbor_is_suppressed=False,
                    neighbor_decay_rate=float(decay_rate),
                )
            )

    return result


def _fan_out_of_node(
    conn: sqlite3.Connection, node_type: str, node_id: int
) -> int:
    row = conn.execute(
        """SELECT COUNT(*) FROM relations
        WHERE valid_to IS NULL AND src_type = ? AND src_id = ?""",
        (node_type, node_id),
    ).fetchone()
    return int(row[0]) if row else 0


def _truncate(text: str, *, full: bool) -> tuple[str, bool]:
    if full or len(text) <= _TRUNCATE_LEN:
        return text, False
    return text[:_TRUNCATE_LEN] + "...", True


def _make_summary(r: _NeighborRow) -> str:
    if r.neighbor_type == "entity":
        base = (
            f"{r.name}: {r.content}"
            if r.name and r.content
            else r.name or r.content
        )
    else:
        base = r.content
    if len(base) > _SUMMARY_LEN:
        return base[: _SUMMARY_LEN - 1] + "…"
    return base


def _to_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _record_explore_access(
    conn: sqlite3.Connection,
    node: NodeContent,
    neighbors: dict[str, list[NeighborItem]],
    now: datetime,
) -> None:
    """起点 + 表示 neighbors の access_history を更新（リハーサル効果）。

    source="explore" を新設。同一 (type, id) は set でユニーク化して
    多重 INSERT を防ぐ（neighbor が複数 predicate に出現するケース対策）。
    """
    targets: set[tuple[str, int]] = {(node.type, node.id)}
    for items in neighbors.values():
        for item in items:
            targets.add((item.type, item.id))

    now_iso = now.isoformat()
    for n_type, n_id in targets:
        conn.execute(
            "INSERT INTO access_history(node_type, node_id, accessed_at, source) "
            "VALUES (?,?,?,?)",
            (n_type, n_id, now_iso, "explore"),
        )
        table = "episodes" if n_type == "episode" else "entities"
        conn.execute(
            f"""UPDATE {table}
            SET activation_count = activation_count + 1,
                last_activated_at = ?
            WHERE id = ?""",
            (now_iso, n_id),
        )
    conn.commit()
