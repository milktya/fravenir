"""Nightly memory_compact pipeline.

Phase4 の夜バッチ。設計書 §6.3 を実装する。

P4-1 で Step 1 (fan_out 再計算)、P4-2 で Step 2 (strength 共起頻度)、
P4-3 で Step 3 (低活性化 episode の抑制フラグ)、
P4-4 で Step 4 (merge_candidates 検出) を実装済。
P5-4/5/6 で --use-llm semantic judgment pass を追加：
  - merge_candidates 意味判定 (P5-4)
  - relation 方向違い検出 (P5-5)
  - 真逆 claim 検出 (P5-6)
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import sqlite_vec
import structlog

from fravenir.core.activation import base_activation
from fravenir.schemas.config import AppConfig
from fravenir.storage import paths

if TYPE_CHECKING:
    from fravenir.core.semantic_judge import (
        ContradictionBatchResult,
        DirectionBatchResult,
        JudgmentBatchResult,
    )

logger = structlog.get_logger()


@dataclass
class CompactResult:
    fan_out_updated: int
    strength_updated: int
    suppressed: int
    merge_candidates: int
    duration_ms: int
    dry_run: bool
    self_loops_archived: int = 0
    judgment: JudgmentBatchResult | None = None
    direction_judgment: DirectionBatchResult | None = None
    contradiction_judgment: ContradictionBatchResult | None = None
    failed_step: str | None = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "fan_out_updated": self.fan_out_updated,
            "strength_updated": self.strength_updated,
            "suppressed": self.suppressed,
            "merge_candidates": self.merge_candidates,
            "self_loops_archived": self.self_loops_archived,
            "duration_ms": self.duration_ms,
            "dry_run": self.dry_run,
        }
        if self.judgment is not None:
            d["judgment"] = self.judgment.to_summary_dict()
        if self.direction_judgment is not None:
            d["direction_judgment"] = self.direction_judgment.to_summary_dict()
        if self.contradiction_judgment is not None:
            d["contradiction_judgment"] = self.contradiction_judgment.to_summary_dict()
        if self.failed_step is not None:
            d["failed_step"] = self.failed_step
        return d


def run_compact(
    *,
    character_id: str,
    config: AppConfig,
    dry_run: bool = False,
    use_llm: bool = False,
    now: datetime | None = None,
) -> CompactResult:
    """Run the nightly compact pipeline. Returns aggregated counts.

    Args:
        now: 現在時刻の注入点。テスト時に freezegun 不要で時刻を固定するため。
            None なら datetime.now(UTC) を使う。
        use_llm: True かつ config.semantic_judge.enabled=True のとき、
            Step 1〜4 の後に LLM semantic judgment pass を実行。
    """
    if now is None:
        now = datetime.now(UTC)

    started = time.monotonic_ns()
    kv_path = paths.kv_db_path(character_id)
    vdb_path = paths.vdb_entities_path(character_id)
    conn = sqlite3.connect(str(kv_path))
    vdb_conn = _open_vdb_entities(vdb_path)
    fan_out_updated = 0
    strength_updated = 0
    suppressed = 0
    merge_candidates = 0
    self_loops_archived = 0
    failed_step: str | None = None
    try:
        try:
            self_loops_archived = _archive_self_loops(conn, now=now, dry_run=dry_run)
        except Exception as e:
            failed_step = "self_loops"
            logger.exception("compact_step_failed", step="self_loops", error=str(e))
            raise
        try:
            fan_out_updated = _recompute_fan_out(conn, dry_run=dry_run)
        except Exception as e:
            failed_step = "fan_out"
            logger.exception("compact_step_failed", step="fan_out", error=str(e))
            raise
        try:
            strength_updated = _recompute_strength(conn, dry_run=dry_run)
        except Exception as e:
            failed_step = "strength"
            logger.exception("compact_step_failed", step="strength", error=str(e))
            raise
        try:
            suppressed = _mark_suppressed(conn, config=config, now=now, dry_run=dry_run)
        except Exception as e:
            failed_step = "suppressed"
            logger.exception("compact_step_failed", step="suppressed", error=str(e))
            raise
        try:
            merge_candidates = _detect_merge_candidates(
                conn, vdb_conn, dry_run=dry_run
            )
        except Exception as e:
            failed_step = "merge_candidates"
            logger.exception("compact_step_failed", step="merge_candidates", error=str(e))
            raise
        if dry_run:
            conn.rollback()
        else:
            conn.commit()
    finally:
        vdb_conn.close()
        conn.close()

    # LLM 判定フェーズは別 connection で動かす
    judgment: JudgmentBatchResult | None = None
    direction_judgment: DirectionBatchResult | None = None
    contradiction_judgment: ContradictionBatchResult | None = None
    if use_llm and config.semantic_judge.enabled:
        from fravenir.core.semantic_judge import (
            judge_merge_candidates,
            judge_relation_contradictions,
            judge_relation_directions,
        )

        try:
            judgment = judge_merge_candidates(
                db_path=kv_path,
                config=config.semantic_judge,
                dry_run=dry_run,
                now=now,
            )
        except Exception as e:
            logger.exception("compact_judge_merge_error", error=str(e))
        try:
            direction_judgment = judge_relation_directions(
                db_path=kv_path,
                config=config.semantic_judge,
                dry_run=dry_run,
                now=now,
            )
        except Exception as e:
            logger.exception("compact_judge_direction_error", error=str(e))
        try:
            contradiction_judgment = judge_relation_contradictions(
                db_path=kv_path,
                config=config.semantic_judge,
                dry_run=dry_run,
                now=now,
            )
        except Exception as e:
            logger.exception("compact_judge_contradiction_error", error=str(e))

    duration_ms = (time.monotonic_ns() - started) // 1_000_000

    result = CompactResult(
        fan_out_updated=fan_out_updated,
        strength_updated=strength_updated,
        suppressed=suppressed,
        merge_candidates=merge_candidates,
        self_loops_archived=self_loops_archived,
        duration_ms=int(duration_ms),
        dry_run=dry_run,
        judgment=judgment,
        direction_judgment=direction_judgment,
        contradiction_judgment=contradiction_judgment,
        failed_step=failed_step,
    )
    logger.info(
        "memory_compact_done",
        character_id=character_id,
        **result.to_dict(),
    )
    return result


def _archive_self_loops(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    dry_run: bool,
) -> int:
    """active な真の self-loop relation (src_type=dst_type AND src_id=dst_id) を archive する。

    物理削除はせず valid_to に now を打つ。write.py 挿入時ガードと
    extraction prompt のルールが整備される前に投入された既存データの後段防御。
    dry_run=True なら呼び出し側で rollback されるので、件数だけ返して書き戻しは行う。
    """
    cur = conn.execute(
        "UPDATE relations SET valid_to = ? "
        "WHERE valid_to IS NULL "
        "AND src_type = dst_type AND src_id = dst_id",
        (now.isoformat(),),
    )
    return int(cur.rowcount)


def _recompute_fan_out(conn: sqlite3.Connection, *, dry_run: bool) -> int:
    """Step 1: relations.fan_out をライブな src ごとの out-degree で更新。

    SQL は v2_design §6.3 Step 1 に準拠:
        UPDATE relations SET fan_out = (
            SELECT COUNT(*) FROM relations r2
            WHERE r2.src_type = relations.src_type
              AND r2.src_id   = relations.src_id
              AND r2.valid_to IS NULL
        )
        WHERE valid_to IS NULL;

    Returns:
        実際に値が変わった行数（dry_run でも同じ意味の数字を返す）。
    """
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.fan_out,
            (
                SELECT COUNT(*) FROM relations r2
                WHERE r2.src_type = r.src_type
                  AND r2.src_id   = r.src_id
                  AND r2.valid_to IS NULL
            ) AS new_fan_out
        FROM relations r
        WHERE r.valid_to IS NULL
        """
    ).fetchall()

    changed = [(new_fan_out, rid) for rid, current, new_fan_out in rows if current != new_fan_out]

    if not dry_run and changed:
        conn.executemany(
            "UPDATE relations SET fan_out = ? WHERE id = ?",
            changed,
        )

    return len(changed)


def _recompute_strength(conn: sqlite3.Connection, *, dry_run: bool) -> int:
    """Step 2: relations.strength を共起頻度ベースで更新。

    対象: src_type='entity' AND dst_type='entity' のライブな relation。
    共起の定義: src と dst の両方に mentions が張られたライブな episode 件数。
    式: strength = 1.0 + ln(cooccurrence)（cooccurrence >= 1 のときのみ反映）。
    cooccurrence == 0 の relation は既存値を維持（更新カウント対象外）。

    Returns:
        実際に値を変更した行数。
    """
    rows = conn.execute(
        """
        SELECT
            r.id,
            r.strength AS current_strength,
            (
                SELECT COUNT(DISTINCT m1.src_id)
                FROM relations m1
                INNER JOIN relations m2 ON m1.src_id = m2.src_id
                WHERE m1.src_type = 'episode' AND m1.predicate = 'mentions'
                  AND m1.dst_type = 'entity'  AND m1.dst_id = r.src_id
                  AND m1.valid_to IS NULL
                  AND m2.src_type = 'episode' AND m2.predicate = 'mentions'
                  AND m2.dst_type = 'entity'  AND m2.dst_id = r.dst_id
                  AND m2.valid_to IS NULL
            ) AS cooccurrence
        FROM relations r
        WHERE r.src_type = 'entity'
          AND r.dst_type = 'entity'
          AND r.valid_to IS NULL
        """
    ).fetchall()

    updates: list[tuple[float, int]] = []
    for rid, current, cooc in rows:
        if cooc <= 0:
            continue
        new_strength = 1.0 + math.log(cooc)
        if abs(new_strength - current) > 1e-9:
            updates.append((new_strength, rid))

    if not dry_run and updates:
        conn.executemany(
            "UPDATE relations SET strength = ? WHERE id = ?",
            updates,
        )

    return len(updates)


def _mark_suppressed(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    now: datetime,
    dry_run: bool,
) -> int:
    """Step 3: 低活性化 episode に is_suppressed=1 を立てる。

    対象: ライブで未抑制な episodes (valid_to IS NULL AND is_suppressed = 0)。
    判定: A_i < θ_suppress かつ 直近 N 日にアクセス履歴なし。
        - A_i は core.activation.base_activation で B_i のみ使う
          (夜バッチは検索 cue を持たないので S_ji 項は加算しない)。
        - θ_suppress は config.act_r.suppress_threshold。
        - N は config.compact.suppress_recent_access_days。

    entities は対象外 (DDL に is_suppressed カラムなし、
    entity 整理は Phase 5 で merge_candidates resolve 経由で行う方針)。

    Returns:
        新規に抑制した episode 件数。
    """
    threshold = config.act_r.suppress_threshold
    recent_days = config.compact.suppress_recent_access_days
    decay = config.act_r.base_decay
    history_limit = config.act_r.access_history_limit
    cutoff = now - timedelta(days=recent_days)

    rows = conn.execute(
        "SELECT id FROM episodes WHERE valid_to IS NULL AND is_suppressed = 0"
    ).fetchall()

    to_suppress: list[int] = []
    for (ep_id,) in rows:
        recent = conn.execute(
            "SELECT 1 FROM access_history "
            "WHERE node_type = 'episode' AND node_id = ? AND accessed_at >= ? LIMIT 1",
            (ep_id, cutoff.isoformat()),
        ).fetchone()
        if recent:
            continue
        b_i = base_activation(conn, "episode", ep_id, decay, now, limit=history_limit)
        if b_i < threshold:
            to_suppress.append(ep_id)

    if not dry_run and to_suppress:
        conn.executemany(
            "UPDATE episodes SET is_suppressed = 1 WHERE id = ?",
            [(eid,) for eid in to_suppress],
        )

    return len(to_suppress)


def _open_vdb_entities(db_path: object) -> sqlite3.Connection:
    """Open vdb_entities.db with sqlite-vec extension loaded."""
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _levenshtein_distance(a: str, b: str, max_distance: int) -> int:
    """Compute Levenshtein distance with early termination.

    Returns the actual distance if it is <= max_distance,
    or max_distance + 1 as a sentinel when exceeded.

    最小行値が max_distance を超えた段階で打ち切るので、
    距離が遠い文字列ペアでは O(min(len_a, len_b) * max_distance) 程度。
    """
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if abs(len_a - len_b) > max_distance:
        return max_distance + 1
    if not a:
        return len_b
    if not b:
        return len_a

    prev = list(range(len_b + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len_b
        row_min = curr[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                curr[j - 1] + 1,        # insertion
                prev[j] + 1,            # deletion
                prev[j - 1] + cost,     # substitution
            )
            if curr[j] < row_min:
                row_min = curr[j]
        if row_min > max_distance:
            return max_distance + 1
        prev = curr
    return prev[len_b]


def _detect_merge_candidates(
    kv_conn: sqlite3.Connection,
    vdb_conn: sqlite3.Connection,
    *,
    similarity_threshold: float = 0.85,
    levenshtein_threshold: int = 3,
    dry_run: bool,
) -> int:
    """Step 4: 同 entity_type の近似ペアを merge_candidates に登録。

    1. vdb_entities から全ベクトルを取得（embedding は L2 正規化済み前提）
    2. numpy で内積行列 = ペアワイズコサイン類似度を計算
    3. 上三角の cosine > similarity_threshold のペアを抽出
    4. entity_type 一致 + canonical_name レーベンシュタイン距離 < levenshtein_threshold で絞る
    5. resolved=0 で同ペアが merge_candidates に既登録ならスキップ
    6. (entity_a, entity_b) を (小id, 大id) に正規化して insert

    Returns:
        新規 insert した件数。
    """
    rows = vdb_conn.execute(
        "SELECT entity_id, embedding FROM vdb_entities"
    ).fetchall()
    if len(rows) < 2:
        return 0

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    sim = mat @ mat.T  # 正規化済みなら内積 = cosine
    mask = np.triu(sim > similarity_threshold, k=1)
    i_idx, j_idx = np.where(mask)
    if len(i_idx) == 0:
        return 0

    # entity メタ情報（canonical_name / entity_type）をライブのみ一括取得
    placeholders = ",".join("?" for _ in ids)
    meta_rows = kv_conn.execute(
        f"SELECT id, canonical_name, entity_type FROM entities "
        f"WHERE id IN ({placeholders}) AND valid_to IS NULL",
        [int(x) for x in ids],
    ).fetchall()
    meta = {row[0]: (row[1], row[2]) for row in meta_rows}

    inserted = 0
    for i, j in zip(i_idx.tolist(), j_idx.tolist(), strict=True):
        a_id = int(ids[i])
        b_id = int(ids[j])
        a_meta = meta.get(a_id)
        b_meta = meta.get(b_id)
        if a_meta is None or b_meta is None:
            continue  # archived な entity はスキップ
        a_name, a_type = a_meta
        b_name, b_type = b_meta
        if a_type != b_type:
            continue
        if (
            _levenshtein_distance(a_name, b_name, levenshtein_threshold)
            >= levenshtein_threshold
        ):
            continue

        small_id, large_id = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        existing = kv_conn.execute(
            "SELECT 1 FROM merge_candidates "
            "WHERE resolved = 0 "
            "AND ((entity_a = ? AND entity_b = ?) OR (entity_a = ? AND entity_b = ?)) "
            "LIMIT 1",
            (small_id, large_id, large_id, small_id),
        ).fetchone()
        if existing:
            continue

        if not dry_run:
            kv_conn.execute(
                "INSERT INTO merge_candidates (entity_a, entity_b, similarity) "
                "VALUES (?, ?, ?)",
                (small_id, large_id, float(sim[i, j])),
            )
        inserted += 1

    return inserted
