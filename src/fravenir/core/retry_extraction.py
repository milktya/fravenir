"""Retry LLM extraction for episodes that failed at the `embedded` stage.

`memory_write` で extraction が失敗するとエピソードは `doc_status.stage='embedded'`
かつ `error` 付きで停止する。設定変更や LLM 復旧の後にこれらをまとめて再抽出する
ための運用機能。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from fravenir.core.write import _run_extraction
from fravenir.storage import paths

if TYPE_CHECKING:
    from fravenir.core.extraction import ExtractionClient
    from fravenir.embedding import Embedder

_log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FailedEpisode:
    episode_id: int
    content: str
    kind: str
    error: str
    updated_at: str


@dataclass
class RetryResult:
    attempted: list[int] = field(default_factory=list)
    succeeded: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "attempted": len(self.attempted),
                "succeeded": len(self.succeeded),
                "failed": len(self.failed),
            },
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": [
                {"episode_id": eid, "error": err} for eid, err in self.failed
            ],
        }


def list_failed_episodes(
    *,
    character_id: str,
    limit: int | None = None,
    episode_ids: list[int] | None = None,
    include_pending: bool = False,
) -> list[FailedEpisode]:
    """再抽出対象のエピソードを返す。

    - `episode_ids` 指定時はそれらに限定（stage / error は問わない）。手動指定で
      強制的に再抽出したい場合に使う。doc_status 行が欠けていてもヒットさせるため
      LEFT JOIN を使う。
    - 未指定 + `include_pending=False`（既定）: `stage='embedded'` かつ
      `error IS NOT NULL` のもの（従来の Failed のみ）。
    - 未指定 + `include_pending=True`: 上記に加えて「doc_status 行が無い」または
      「stage が 'done' 以外」の episode も対象に含める。`init-character` で
      doc_status を作らずに投入された seed episode などの救済に使う。
    - `limit` は古い順（doc_status が無い場合は episode.id 順）に上限件数を絞る。
    """
    kv_path = paths.kv_db_path(character_id)
    conn = sqlite3.connect(kv_path)
    try:
        if episode_ids:
            placeholders = ",".join("?" * len(episode_ids))
            rows = conn.execute(
                f"""
                SELECT e.id, e.content, e.kind, ds.error, ds.updated_at
                FROM episodes e
                LEFT JOIN doc_status ds ON ds.episode_id = e.id
                WHERE e.id IN ({placeholders})
                ORDER BY e.id
                """,
                episode_ids,
            ).fetchall()
        else:
            if include_pending:
                where_clause = (
                    "e.valid_to IS NULL AND ("
                    " ds.episode_id IS NULL"
                    " OR (ds.stage = 'embedded' AND ds.error IS NOT NULL)"
                    " OR ds.stage IN ('pending', 'extracted', 'linked')"
                    " OR (ds.stage = 'embedded' AND ds.error IS NULL)"
                    ")"
                )
            else:
                where_clause = (
                    "ds.stage = 'embedded' AND ds.error IS NOT NULL"
                )
            sql = f"""
                SELECT e.id, e.content, e.kind, ds.error, ds.updated_at
                FROM episodes e
                LEFT JOIN doc_status ds ON ds.episode_id = e.id
                WHERE {where_clause}
                ORDER BY COALESCE(ds.updated_at, ''), e.id
            """
            params: list[Any] = []
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    return [
        FailedEpisode(
            episode_id=row[0],
            content=row[1],
            kind=row[2],
            error=row[3] or "",
            updated_at=row[4] or "",
        )
        for row in rows
    ]


def retry_extraction(
    targets: list[FailedEpisode],
    *,
    character_id: str,
    extraction_client: ExtractionClient,
    embedder: Embedder,
) -> RetryResult:
    """各エピソードについて `_run_extraction` を再走し、結果サマリーを返す."""
    result = RetryResult()
    kv_path = paths.kv_db_path(character_id)
    for ep in targets:
        result.attempted.append(ep.episode_id)
        _ensure_doc_status_row(kv_path, ep.episode_id)
        now = datetime.now(UTC)
        final_stage = _run_extraction(
            content=ep.content,
            episode_id=ep.episode_id,
            episode_kind=ep.kind,
            character_id=character_id,
            kv_path=kv_path,
            extraction_client=extraction_client,
            embedder=embedder,
            now=now,
        )
        if final_stage == "done":
            result.succeeded.append(ep.episode_id)
        else:
            err = _read_current_error(kv_path, ep.episode_id) or "unknown"
            result.failed.append((ep.episode_id, err))
            _log.warning(
                "retry_extraction_failed",
                episode_id=ep.episode_id,
                final_stage=final_stage,
            )
    return result


def _read_current_error(kv_path: Path, episode_id: int) -> str | None:
    conn = sqlite3.connect(kv_path)
    try:
        row = conn.execute(
            "SELECT error FROM doc_status WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _ensure_doc_status_row(kv_path: Path, episode_id: int) -> None:
    """doc_status 行が無ければ stage='embedded' で作成する。

    `init-character` 等で doc_status エントリ無しに INSERT された episode を
    `_run_extraction` の `_update_doc_status` が UPDATE できる形に揃えるための前処理。
    """
    conn = sqlite3.connect(kv_path)
    try:
        existing = conn.execute(
            "SELECT 1 FROM doc_status WHERE episode_id = ? LIMIT 1",
            (episode_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO doc_status(episode_id, stage) VALUES (?, 'embedded')",
                (episode_id,),
            )
            conn.commit()
    finally:
        conn.close()
