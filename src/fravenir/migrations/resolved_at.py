"""Add resolved_at column to merge_candidates (Phase 5 P5-7).

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
        "resolved_at",
        "ALTER TABLE merge_candidates ADD COLUMN resolved_at TIMESTAMP",
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
