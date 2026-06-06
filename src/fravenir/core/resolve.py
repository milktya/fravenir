"""merge_candidates の手動 resolve（マージ / 却下）処理。

Phase 5 の P5-1。entity_a と entity_b は事前に互換性検査済（同 entity_type、
canonical_name レーベンシュタイン < 3）として扱う。設計判断:
- keep は id の小さい方が既定。--keep で明示指定可
- relations は valid_to IS NULL の行のみ keep に付け替え、自己ループは論理削除
- drop の canonical_name と aliases は keep の entity_aliases に統合
- drop は valid_to=now, supersedes=keep
- vdb_entities の embedding は valid_to で除外されるためそのまま残す
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: int
    entity_a: int
    entity_b: int
    similarity: float
    a_name: str
    b_name: str
    a_type: str | None
    b_type: str | None
    judge_label: str | None
    judge_confidence: str | None
    judge_reason: str | None
    judge_attempts: int


@dataclass(frozen=True)
class MergeResult:
    candidate_id: int
    keep_id: int
    drop_id: int
    relations_rewired: int
    self_loops_archived: int
    aliases_added: int
    dry_run: bool


@dataclass(frozen=True)
class RejectResult:
    candidate_id: int
    dry_run: bool


class ResolveError(Exception):
    """resolve 処理中の業務エラー（CLI で exit 1 にマップ）。"""


def list_candidates(db_path: Path) -> list[CandidateRow]:
    """resolved=0 の候補を id 昇順で返す。entity 名は valid_to 無視で取得。"""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT mc.id, mc.entity_a, mc.entity_b, mc.similarity,
                   ea.canonical_name, eb.canonical_name,
                   ea.entity_type, eb.entity_type,
                   mc.judge_label, mc.judge_confidence, mc.judge_reason,
                   mc.judge_attempts
            FROM merge_candidates mc
            JOIN entities ea ON ea.id = mc.entity_a
            JOIN entities eb ON eb.id = mc.entity_b
            WHERE mc.resolved = 0
            ORDER BY mc.id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    return [
        CandidateRow(
            candidate_id=r[0],
            entity_a=r[1],
            entity_b=r[2],
            similarity=float(r[3]),
            a_name=r[4],
            b_name=r[5],
            a_type=r[6],
            b_type=r[7],
            judge_label=r[8],
            judge_confidence=r[9],
            judge_reason=r[10],
            judge_attempts=int(r[11]),
        )
        for r in rows
    ]


def _fetch_candidate(conn: sqlite3.Connection, candidate_id: int) -> tuple[int, int, int]:
    """(entity_a, entity_b, resolved) を返す。無ければ KeyError。"""
    row = conn.execute(
        "SELECT entity_a, entity_b, resolved FROM merge_candidates WHERE id = ?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise KeyError(candidate_id)
    return int(row[0]), int(row[1]), int(row[2])


def _merge_with_conn(
    *,
    conn: sqlite3.Connection,
    candidate_id: int,
    entity_a: int,
    entity_b: int,
    keep: int | None,
    now_iso: str,
) -> MergeResult:
    """conn を渡されたら BEGIN/COMMIT/ROLLBACK しない（呼び出し側責任）。

    内部処理: relations 付け替え、自己ループ archive、drop の aliases 統合、
    drop の valid_to + supersedes、merge_candidates.resolved = 1。
    """
    # self-merge ガード: entity_a == entity_b の候補は致命的（自分を自分で
    # supersede して valid_to が刺さり、グラフ参照が壊れる）。通常経路では
    # _detect_merge_candidates が triu(k=1) で対角線を除外しているが、外部
    # 経路で merge_candidates に直接挿入された場合の防御層として早期 raise。
    if entity_a == entity_b:
        raise ResolveError(
            f"candidate {candidate_id}: self-merge is invalid "
            f"(entity_a == entity_b == {entity_a})."
        )

    # is_self ガード: 自己ハブが絡むマージは人間判断必須として拒否。
    # is_self=1 entity は character につき1つ想定 (seed.yaml で初期化)。
    # 並立マージ候補が出る状況自体が異常なので、reject 一択で人間判断に回す。
    a_self_row = conn.execute(
        "SELECT is_self FROM entities WHERE id = ?", (entity_a,)
    ).fetchone()
    b_self_row = conn.execute(
        "SELECT is_self FROM entities WHERE id = ?", (entity_b,)
    ).fetchone()
    if (a_self_row and a_self_row[0]) or (b_self_row and b_self_row[0]):
        raise ResolveError(
            f"candidate {candidate_id}: self-hub entity is involved "
            f"(a={entity_a}, b={entity_b}). reject this candidate manually."
        )

    if keep is None:
        keep_id, drop_id = (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)
    else:
        keep_id = keep
        drop_id = entity_b if keep == entity_a else entity_a

    # 1. relations 付け替え (valid_to IS NULL のみ)
    conn.execute(
        "UPDATE relations SET src_id = ? "
        "WHERE src_type = 'entity' AND src_id = ? AND valid_to IS NULL",
        (keep_id, drop_id),
    )
    conn.execute(
        "UPDATE relations SET dst_id = ? "
        "WHERE dst_type = 'entity' AND dst_id = ? AND valid_to IS NULL",
        (keep_id, drop_id),
    )
    rewired_total = conn.execute(
        "SELECT COUNT(*) FROM relations "
        "WHERE valid_to IS NULL AND ("
        "  (src_type = 'entity' AND src_id = ?) OR "
        "  (dst_type = 'entity' AND dst_id = ?)"
        ")",
        (keep_id, keep_id),
    ).fetchone()[0]

    # 2. 自己ループ relation を論理削除
    cur_loop = conn.execute(
        "UPDATE relations SET valid_to = ? "
        "WHERE valid_to IS NULL AND src_type = 'entity' AND dst_type = 'entity' "
        "AND src_id = dst_id AND src_id = ?",
        (now_iso, keep_id),
    )
    self_loops = cur_loop.rowcount

    # 3. drop の aliases を keep に統合 + drop の canonical_name も alias に
    drop_canonical_row = conn.execute(
        "SELECT canonical_name FROM entities WHERE id = ?",
        (drop_id,),
    ).fetchone()
    drop_canonical = drop_canonical_row[0] if drop_canonical_row else None

    aliases_added = 0
    if drop_canonical is not None:
        cur_a1 = conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
            (drop_canonical, keep_id),
        )
        aliases_added += cur_a1.rowcount

    for (alias,) in conn.execute(
        "SELECT alias FROM entity_aliases WHERE entity_id = ?", (drop_id,)
    ).fetchall():
        cur_a2 = conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
            (alias, keep_id),
        )
        aliases_added += cur_a2.rowcount

    # 4. drop の valid_to / supersedes
    conn.execute(
        "UPDATE entities SET valid_to = ?, supersedes = ? WHERE id = ?",
        (now_iso, keep_id, drop_id),
    )

    # 5. merge_candidates.resolved = 1, resolved_at = now
    conn.execute(
        "UPDATE merge_candidates SET resolved = 1, resolved_at = ? WHERE id = ?",
        (now_iso, candidate_id),
    )

    return MergeResult(
        candidate_id=candidate_id,
        keep_id=keep_id,
        drop_id=drop_id,
        relations_rewired=int(rewired_total),
        self_loops_archived=self_loops,
        aliases_added=aliases_added,
        dry_run=False,
    )


def merge(
    db_path: Path,
    candidate_id: int,
    *,
    keep: int | None = None,
    dry_run: bool = False,
) -> MergeResult:
    """候補をマージ。keep が None なら id 小さい方を採用。

    Raises:
        KeyError: candidate_id が存在しない
        ResolveError: 既に resolved != 0、または keep が候補に含まれない
    """
    conn = sqlite3.connect(db_path)
    conn.execute("BEGIN")
    try:
        entity_a, entity_b, resolved = _fetch_candidate(conn, candidate_id)
        if resolved != 0:
            raise ResolveError(
                f"candidate {candidate_id} already resolved (resolved={resolved})"
            )

        if keep is not None and keep not in (entity_a, entity_b):
            raise ResolveError(
                f"--keep {keep} is not part of candidate {candidate_id} "
                f"(entity_a={entity_a}, entity_b={entity_b})"
            )

        now = datetime.now(UTC).isoformat()
        result = _merge_with_conn(
            conn=conn,
            candidate_id=candidate_id,
            entity_a=entity_a,
            entity_b=entity_b,
            keep=keep,
            now_iso=now,
        )

        if dry_run:
            conn.execute("ROLLBACK")
        else:
            conn.execute("COMMIT")

        return MergeResult(
            candidate_id=result.candidate_id,
            keep_id=result.keep_id,
            drop_id=result.drop_id,
            relations_rewired=result.relations_rewired,
            self_loops_archived=result.self_loops_archived,
            aliases_added=result.aliases_added,
            dry_run=dry_run,
        )
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def reject(
    db_path: Path,
    candidate_id: int,
    *,
    dry_run: bool = False,
) -> RejectResult:
    conn = sqlite3.connect(db_path)
    conn.execute("BEGIN")
    try:
        _, _, resolved = _fetch_candidate(conn, candidate_id)
        if resolved != 0:
            raise ResolveError(
                f"candidate {candidate_id} already resolved (resolved={resolved})"
            )
        now_iso = datetime.now(UTC).isoformat()
        conn.execute(
            "UPDATE merge_candidates SET resolved = 2, resolved_at = ? WHERE id = ?",
            (now_iso, candidate_id),
        )
        if dry_run:
            conn.execute("ROLLBACK")
        else:
            conn.execute("COMMIT")
        return RejectResult(candidate_id=candidate_id, dry_run=dry_run)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
