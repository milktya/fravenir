"""矛盾検出と supersedes 自動設定（Phase 5）。

memory_write 時に「同 src_entity + 同 predicate + 別 dst_entity」の relation を
検出し、古い relation と古い episode の valid_to を立て、新 relation/episode に
supersedes を設定する。

設計書 §6.2 Step 6 (docs/v2_design.md L482-485) と Phase 5 DoD (L1003-1005) の実装。
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from fravenir.core.extraction import ExtractionResult

_log = structlog.get_logger(__name__)

# 単数値前提 predicate のホワイトリスト。
# ここに含まれる predicate でのみ「同 src + 同 predicate + 別 dst」を矛盾と扱う。
# 設計判断: works_as / lives_in / located_at / runs_on のみ。
# is_a (複数並立 OK) や part_of (概念の所属が複数並立) は除外。
SINGLE_VALUE_PREDICATES: frozenset[str] = frozenset({
    "works_as",
    "lives_in",
    "located_at",
    "runs_on",
})


def detect_and_supersede(
    conn: sqlite3.Connection,
    new_episode_id: int,
    new_episode_kind: str,
    result: ExtractionResult,
    name_to_id: dict[str, int],
    now: datetime,
) -> dict[str, int]:
    """新 episode の relation 群を見て、矛盾する古い relation/episode を supersede する。

    Args:
        conn: kv DB 接続（呼び出し側が commit する）
        new_episode_id: 今書いた episode の ID
        new_episode_kind: 今書いた episode の kind ('facts' | 'state' | 'emo')
        result: LLM 抽出結果
        name_to_id: canonical_name -> entity_id のマップ
            （write._apply_extraction_to_db 内で構築済み）
        now: 書き込み時刻（_apply_extraction_to_db に渡された now と同一）

    Returns:
        {"relations_superseded": int, "episodes_superseded": int}
    """
    if new_episode_kind != "facts":
        return {"relations_superseded": 0, "episodes_superseded": 0}

    now_iso = now.isoformat()
    relations_superseded = 0
    episodes_superseded = 0
    superseded_episode_ids: set[int] = set()

    for rel in result.relations:
        if rel.predicate not in SINGLE_VALUE_PREDICATES:
            continue

        new_src_id = name_to_id.get(rel.src)
        new_dst_id = name_to_id.get(rel.dst)
        if new_src_id is None or new_dst_id is None:
            continue

        old_relations = _find_conflicting_relations(
            conn, new_src_id, rel.predicate, new_dst_id
        )
        if not old_relations:
            continue

        if len(old_relations) > 1:
            _log.warning(
                "multiple_conflicting_relations",
                src_id=new_src_id,
                predicate=rel.predicate,
                count=len(old_relations),
                episode_id=new_episode_id,
            )

        new_relation_id = _find_new_relation_id(
            conn, new_src_id, rel.predicate, new_dst_id, now_iso
        )

        for old_rel_id, old_valid_from in old_relations:
            _supersede_relation(conn, old_rel_id, new_relation_id, now_iso)
            relations_superseded += 1

            old_episode_id = _find_source_episode_id(
                conn, new_src_id, old_valid_from
            )
            if old_episode_id is None or old_episode_id in superseded_episode_ids:
                continue
            if old_episode_id == new_episode_id:
                continue
            _supersede_episode(conn, old_episode_id, new_episode_id, now_iso)
            superseded_episode_ids.add(old_episode_id)
            episodes_superseded += 1

    return {
        "relations_superseded": relations_superseded,
        "episodes_superseded": episodes_superseded,
    }


def _find_conflicting_relations(
    conn: sqlite3.Connection,
    src_id: int,
    predicate: str,
    new_dst_id: int,
) -> list[tuple[int, str]]:
    """同 src + 同 predicate + 別 dst + valid_to IS NULL の entity-to-entity relation を返す。

    新規 INSERT 直後でも自分自身（valid_from == now_iso かつ dst == new_dst_id）は除外する。
    """
    rows = conn.execute(
        """
        SELECT id, valid_from
        FROM relations
        WHERE src_type = 'entity' AND src_id = ?
          AND dst_type = 'entity' AND dst_id != ?
          AND predicate = ?
          AND valid_to IS NULL
        """,
        (src_id, new_dst_id, predicate),
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def _find_new_relation_id(
    conn: sqlite3.Connection,
    src_id: int,
    predicate: str,
    dst_id: int,
    now_iso: str,
) -> int | None:
    """今 INSERT したばかりの新 relation の id を引く。

    _apply_extraction_to_db が relation を INSERT した直後にこの関数が呼ばれる前提。
    valid_from == now_iso で一意に絞れる。
    """
    row = conn.execute(
        """
        SELECT id FROM relations
        WHERE src_type = 'entity' AND src_id = ?
          AND dst_type = 'entity' AND dst_id = ?
          AND predicate = ?
          AND valid_from = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (src_id, dst_id, predicate, now_iso),
    ).fetchone()
    return int(row[0]) if row else None


def _find_source_episode_id(
    conn: sqlite3.Connection,
    entity_id: int,
    relation_valid_from: str,
) -> int | None:
    """古い relation を生んだ episode を mentions 経由で逆引きする。

    entity-to-entity relation と mentions relation は同じ _apply_extraction_to_db
    内で同 valid_from で書かれるため、valid_from 完全一致で紐付けできる。
    """
    row = conn.execute(
        """
        SELECT src_id
        FROM relations
        WHERE src_type = 'episode'
          AND dst_type = 'entity' AND dst_id = ?
          AND predicate = 'mentions'
          AND valid_from = ?
        ORDER BY src_id DESC
        LIMIT 1
        """,
        (entity_id, relation_valid_from),
    ).fetchone()
    return int(row[0]) if row else None


def _supersede_relation(
    conn: sqlite3.Connection,
    old_relation_id: int,
    new_relation_id: int | None,
    now_iso: str,
) -> None:
    """古い relation の valid_to を立て、新 relation の supersedes を旧 id に設定。"""
    conn.execute(
        "UPDATE relations SET valid_to = ? WHERE id = ?",
        (now_iso, old_relation_id),
    )
    if new_relation_id is not None:
        conn.execute(
            "UPDATE relations SET supersedes = ? WHERE id = ?",
            (old_relation_id, new_relation_id),
        )


def _supersede_episode(
    conn: sqlite3.Connection,
    old_episode_id: int,
    new_episode_id: int,
    now_iso: str,
) -> None:
    """古い episode の valid_to を立て、新 episode の supersedes を旧 id に設定。"""
    conn.execute(
        "UPDATE episodes SET valid_to = ? WHERE id = ?",
        (now_iso, old_episode_id),
    )
    conn.execute(
        "UPDATE episodes SET supersedes = ? WHERE id = ?",
        (old_episode_id, new_episode_id),
    )
