"""Add judge_label/judge_confidence/judge_reason/judge_attempts columns
to merge_candidates (Phase 5 P5-4).

idempotent: 同じ DB に対して何度実行しても安全。
- 列が既に存在すれば ALTER しない
- dry_run=True ならコミットしない
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_COLUMN_SPECS: tuple[tuple[str, str], ...] = (
    (
        "judge_label",
        "ALTER TABLE merge_candidates ADD COLUMN judge_label TEXT",
    ),
    (
        "judge_confidence",
        "ALTER TABLE merge_candidates ADD COLUMN judge_confidence TEXT",
    ),
    (
        "judge_reason",
        "ALTER TABLE merge_candidates ADD COLUMN judge_reason TEXT",
    ),
    (
        "judge_attempts",
        "ALTER TABLE merge_candidates "
        "ADD COLUMN judge_attempts INTEGER NOT NULL DEFAULT 0",
    ),
)


@dataclass(frozen=True)
class MigrationResult:
    added_columns: list[str] = field(default_factory=list)
    dry_run: bool = False


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == col for row in rows)


def migrate(db_path: Path, *, dry_run: bool = False) -> MigrationResult:
    conn = sqlite3.connect(db_path)
    try:
        added: list[str] = []
        for col_name, ddl in _COLUMN_SPECS:
            if _has_column(conn, "merge_candidates", col_name):
                continue
            added.append(col_name)
            if not dry_run:
                conn.execute(ddl)
        if not dry_run:
            conn.commit()
        return MigrationResult(added_columns=added, dry_run=dry_run)
    finally:
        conn.close()
