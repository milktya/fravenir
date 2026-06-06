"""Episode write pipeline.

Step 1-2: INSERT episodes + embedding (Phase1).
Step 3-5: LLM extraction → entities/relations (Phase3+, only when extraction_client given).
Step 6: 矛盾検出 + supersede 自動設定 (P5-3 以降)。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog

from fravenir.core.supersede import detect_and_supersede
from fravenir.embedding import Embedder
from fravenir.schemas.config import AppConfig
from fravenir.storage import paths
from fravenir.storage.vector import (
    upsert_entity_vector,
    upsert_episode_vector,
    upsert_relation_vector,
)

if TYPE_CHECKING:
    from fravenir.core.extraction import (
        ExtractedEntity,
        ExtractedRelation,
        ExtractionClient,
        ExtractionResult,
    )

_log = structlog.get_logger(__name__)


def memory_write(
    content: str,
    kind: Literal["facts", "state", "emo"],
    importance: int,
    session_id: str | None,
    *,
    character_id: str,
    config: AppConfig,
    embedder: Embedder,
    extraction_client: ExtractionClient | None = None,
) -> dict[str, object]:
    """Write one episode, its embedding, and (if extraction_client given) entities/relations.

    Returns:
        {"episode_id": int, "created_at": str, "stage": str}
    Raises:
        ValueError: on invalid arguments
    """
    del config  # signature 整合のため受け取るが本関数では未使用

    if not content.strip():
        raise ValueError("content must not be empty")
    if kind not in ("facts", "state", "emo"):
        raise ValueError(f"invalid kind: {kind!r}")
    if not 1 <= importance <= 3:
        raise ValueError(f"importance must be 1-3, got {importance}")

    now = datetime.now(UTC)
    kv_path = paths.kv_db_path(character_id)
    vdb_path = paths.vdb_memories_path(character_id)

    kv_conn = sqlite3.connect(kv_path)
    try:
        episode_id = _insert_episode(kv_conn, content, kind, importance, session_id, now)
        _insert_doc_status(kv_conn, episode_id, "pending")
    finally:
        kv_conn.close()

    vector = embedder.encode_document(content)

    vdb_conn = _open_vdb(vdb_path)
    try:
        upsert_episode_vector(vdb_conn, episode_id, vector)
    finally:
        vdb_conn.close()

    stage = "embedded"
    kv_conn = sqlite3.connect(kv_path)
    try:
        _update_doc_status(kv_conn, episode_id, stage)
    finally:
        kv_conn.close()

    if extraction_client is not None:
        stage = _run_extraction(
            content=content,
            episode_id=episode_id,
            episode_kind=kind,
            character_id=character_id,
            kv_path=kv_path,
            extraction_client=extraction_client,
            embedder=embedder,
            now=now,
        )

    return {
        "episode_id": episode_id,
        "created_at": now.isoformat(),
        "stage": stage,
    }


def _run_extraction(
    *,
    content: str,
    episode_id: int,
    episode_kind: str,
    character_id: str,
    kv_path: Path,
    extraction_client: ExtractionClient,
    embedder: Embedder,
    now: datetime,
) -> str:
    """Run LLM extraction pipeline (Step 3-5). Returns the final stage reached."""
    from fravenir.core.extraction import ExtractionError

    try:
        result = extraction_client.extract(content)
    except ExtractionError as e:
        _log.warning("extraction_failed", episode_id=episode_id, error=str(e))
        kv_conn = sqlite3.connect(kv_path)
        try:
            _update_doc_status(kv_conn, episode_id, "embedded", error=str(e))
        finally:
            kv_conn.close()
        return "embedded"

    try:
        _save_extraction_cache(character_id, episode_id, result)
    except OSError as e:
        _log.warning("extraction_cache_write_failed", episode_id=episode_id, error=str(e))
        kv_conn = sqlite3.connect(kv_path)
        try:
            _update_doc_status(kv_conn, episode_id, "embedded", error=f"cache write: {e}")
        finally:
            kv_conn.close()
        return "embedded"

    kv_conn = sqlite3.connect(kv_path)
    try:
        _update_doc_status(kv_conn, episode_id, "extracted")
        _apply_extraction_to_db(
            kv_conn, episode_id, result, now,
            character_id=character_id, embedder=embedder,
            episode_kind=episode_kind,
        )
        _update_doc_status(kv_conn, episode_id, "linked")
        _update_doc_status(kv_conn, episode_id, "done")
    finally:
        kv_conn.close()
    return "done"


def _save_extraction_cache(
    character_id: str, episode_id: int, result: ExtractionResult
) -> None:
    cache_dir = paths.cache_extractions_dir(character_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{episode_id}.json"
    cache_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def _apply_extraction_to_db(
    conn: sqlite3.Connection,
    episode_id: int,
    result: ExtractionResult,
    now: datetime,
    *,
    character_id: str,
    embedder: Embedder,
    episode_kind: str,
) -> None:
    """entities を照合 or 作成し、relations を書き込む。

    新規 entity / entity→entity relation は vdb にも embedding を書き込む
    (alias match で再利用した既存 entity は vdb 更新しない)。
    LLMが relations.src/dst に entities 外の名前を返した場合、その relation はスキップ。
    """
    now_iso = now.isoformat()
    name_to_id: dict[str, int] = {}
    new_entity_ids: list[tuple[int, ExtractedEntity]] = []
    new_relation_ids: list[tuple[int, ExtractedRelation]] = []

    for ent in result.entities:
        eid, is_new = _find_or_create_entity(
            conn, ent.canonical_name, ent.entity_type, ent.description, now_iso
        )
        name_to_id[ent.canonical_name] = eid
        if is_new:
            new_entity_ids.append((eid, ent))
        _insert_mentions_relation(conn, episode_id, eid, now_iso)

    for rel in result.relations:
        src_id = name_to_id.get(rel.src)
        dst_id = name_to_id.get(rel.dst)
        if src_id is None or dst_id is None:
            _log.debug(
                "extraction_relation_unresolved_endpoint",
                episode_id=episode_id,
                src=rel.src,
                dst=rel.dst,
            )
            continue
        if src_id == dst_id:
            # LLM が「主語が同じ relation」を生成 (例: みるちゃ -performs-> みるちゃ で
            # 本来の目的語が description 側に押し込まれるパターン) を弾く。grouped self-loop は
            # グラフ可視化のノイズになるだけで自己ハブの仕組みとも重複するため捨てる。
            _log.debug(
                "extraction_relation_self_loop_skipped",
                episode_id=episode_id,
                entity=rel.src,
                predicate=rel.predicate,
            )
            continue
        rel_id = _insert_entity_relation(
            conn, src_id, dst_id, rel.predicate, rel.description, now_iso
        )
        new_relation_ids.append((rel_id, rel))

    supersede_stats = detect_and_supersede(
        conn=conn,
        new_episode_id=episode_id,
        new_episode_kind=episode_kind,
        result=result,
        name_to_id=name_to_id,
        now=now,
    )
    if supersede_stats["relations_superseded"] > 0 or supersede_stats["episodes_superseded"] > 0:
        _log.info(
            "supersede_applied",
            episode_id=episode_id,
            **supersede_stats,
        )

    try:
        _write_entity_vectors(character_id, new_entity_ids, embedder)
        _write_relation_vectors(character_id, new_relation_ids, embedder)
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _write_entity_vectors(
    character_id: str,
    new_entities: list[tuple[int, ExtractedEntity]],
    embedder: Embedder,
) -> None:
    if not new_entities:
        return
    vdb_path = paths.vdb_entities_path(character_id)
    conn = _open_vec_db(vdb_path)
    try:
        for eid, ent in new_entities:
            text = _entity_embedding_text(ent.canonical_name, ent.description)
            vec = embedder.encode_topic(text)
            upsert_entity_vector(conn, eid, vec)
    finally:
        conn.close()


def _write_relation_vectors(
    character_id: str,
    new_relations: list[tuple[int, ExtractedRelation]],
    embedder: Embedder,
) -> None:
    if not new_relations:
        return
    vdb_path = paths.vdb_relations_path(character_id)
    conn = _open_vec_db(vdb_path)
    try:
        for rid, rel in new_relations:
            text = _relation_embedding_text(rel.predicate, rel.description)
            vec = embedder.encode_topic(text)
            upsert_relation_vector(conn, rid, vec)
    finally:
        conn.close()


def _entity_embedding_text(canonical_name: str, description: str) -> str:
    if description:
        return f"{canonical_name} {description}"
    return canonical_name


def _relation_embedding_text(predicate: str, description: str) -> str:
    if description:
        return f"{predicate} {description}"
    return predicate


def _find_or_create_entity(
    conn: sqlite3.Connection,
    canonical_name: str,
    entity_type: str,
    description: str,
    now_iso: str,
) -> tuple[int, bool]:
    """Return (entity_id, is_new). is_new=False なら既存を再利用した。"""
    row = conn.execute(
        "SELECT id FROM entities WHERE canonical_name = ? AND valid_to IS NULL",
        (canonical_name,),
    ).fetchone()
    if row is not None:
        return int(row[0]), False

    row = conn.execute(
        """
        SELECT e.id FROM entity_aliases a
        JOIN entities e ON a.entity_id = e.id
        WHERE a.alias = ? AND e.valid_to IS NULL
        """,
        (canonical_name,),
    ).fetchone()
    if row is not None:
        return int(row[0]), False

    cur = conn.execute(
        """
        INSERT INTO entities
            (canonical_name, entity_type, description,
             is_self, self_weight, decay_rate, valid_from)
        VALUES (?, ?, ?, 0, 0.0, 0.5, ?)
        """,
        (canonical_name, entity_type, description, now_iso),
    )
    new_id: int = cur.lastrowid  # type: ignore[assignment]
    return new_id, True


def _insert_mentions_relation(
    conn: sqlite3.Connection, episode_id: int, entity_id: int, now_iso: str
) -> None:
    conn.execute(
        """
        INSERT INTO relations
            (src_type, src_id, dst_type, dst_id, predicate, valid_from)
        VALUES ('episode', ?, 'entity', ?, 'mentions', ?)
        """,
        (episode_id, entity_id, now_iso),
    )


def _insert_entity_relation(
    conn: sqlite3.Connection,
    src_id: int,
    dst_id: int,
    predicate: str,
    description: str,
    now_iso: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO relations
            (src_type, src_id, dst_type, dst_id, predicate, description, valid_from)
        VALUES ('entity', ?, 'entity', ?, ?, ?, ?)
        """,
        (src_id, dst_id, predicate, description, now_iso),
    )
    new_id: int = cur.lastrowid  # type: ignore[assignment]
    return new_id


def _insert_episode(
    conn: sqlite3.Connection,
    content: str,
    kind: str,
    importance: int,
    session_id: str | None,
    now: datetime,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO episodes (content, kind, importance, valid_from, session_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (content, kind, importance, now.isoformat(), session_id),
    )
    conn.commit()
    episode_id: int = cur.lastrowid  # type: ignore[assignment]
    return episode_id


def _insert_doc_status(conn: sqlite3.Connection, episode_id: int, stage: str) -> None:
    now = datetime.now(UTC)
    conn.execute(
        "INSERT INTO doc_status(episode_id, stage, updated_at) VALUES (?, ?, ?)",
        (episode_id, stage, now.isoformat()),
    )
    conn.commit()


def _update_doc_status(
    conn: sqlite3.Connection,
    episode_id: int,
    stage: str,
    error: str | None = None,
) -> None:
    now = datetime.now(UTC)
    conn.execute(
        "UPDATE doc_status SET stage = ?, error = ?, updated_at = ? WHERE episode_id = ?",
        (stage, error, now.isoformat(), episode_id),
    )
    conn.commit()


def _open_vdb(db_path: Path) -> sqlite3.Connection:
    return _open_vec_db(db_path)


def _open_vec_db(db_path: Path) -> sqlite3.Connection:
    import sqlite_vec

    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn
