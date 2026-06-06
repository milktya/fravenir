"""Add curated_at column to entities and create admin_audit_log table
(Phase 6: AdminUI description/aliases edit feature).

idempotent: 同じ DB に対して何度実行しても安全。
- 列・テーブルが既に存在すれば変更しない
- dry_run=True ならコミットしない
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_ENTITY_COLUMN = (
    "curated_at",
    "ALTER TABLE entities ADD COLUMN curated_at TIMESTAMP",
)

_AUDIT_TABLE_DDL = """\
CREATE TABLE IF NOT EXISTS admin_audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT    NOT NULL,
    target_type TEXT    NOT NULL,
    target_id   INTEGER NOT NULL,
    before_json TEXT,
    after_json  TEXT,
    actor       TEXT,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_admin_audit_target
    ON admin_audit_log(target_type, target_id, created_at);
CREATE INDEX IF NOT EXISTS idx_admin_audit_created
    ON admin_audit_log(created_at);
"""


@dataclass(frozen=True)
class MigrationResult:
    added_columns: list[str] = field(default_factory=list)
    created_tables: list[str] = field(default_factory=list)
    dry_run: bool = False


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == col for row in rows)


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def migrate(db_path: Path, *, dry_run: bool = False) -> MigrationResult:
    conn = sqlite3.connect(db_path)
    try:
        added: list[str] = []
        col_name, ddl = _ENTITY_COLUMN
        if not _has_column(conn, "entities", col_name):
            added.append(col_name)
            if not dry_run:
                conn.execute(ddl)

        created: list[str] = []
        if not _has_table(conn, "admin_audit_log"):
            created.append("admin_audit_log")
            if not dry_run:
                conn.executescript(_AUDIT_TABLE_DDL)

        if not dry_run:
            conn.commit()
        return MigrationResult(
            added_columns=added, created_tables=created, dry_run=dry_run
        )
    finally:
        conn.close()
