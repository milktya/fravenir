"""episodes.derived_from に入っていた session_id 文字列を専用カラムへ移送する。

idempotent: 同じ DB に対して何度実行しても安全。
- session_id カラムが無ければ ALTER TABLE で追加
- 必要な INDEX が無ければ CREATE INDEX
- derived_from が文字列のレコードのみ session_id へ移送
  (将来 PROV-O 用途で int を入れた episode は巻き込まない)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_INDEX_SPECS: tuple[tuple[str, str], ...] = (
    (
        "idx_episodes_session_id",
        "CREATE INDEX idx_episodes_session_id ON episodes(session_id)",
    ),
    (
        "idx_merge_candidates_pair",
        "CREATE INDEX idx_merge_candidates_pair "
        "ON merge_candidates(entity_a, entity_b)",
    ),
    (
        "idx_merge_candidates_resolved",
        "CREATE INDEX idx_merge_candidates_resolved "
        "ON merge_candidates(resolved)",
    ),
)


@dataclass(frozen=True)
class MigrationResult:
    added_session_id_column: bool
    added_indexes: list[str] = field(default_factory=list)
    migrated_rows: int = 0
    dry_run: bool = False


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == col for row in rows)


def _has_index(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _count_legacy_derived_from(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM episodes "
        "WHERE derived_from IS NOT NULL AND typeof(derived_from) = 'text'"
    ).fetchone()
    return int(row[0])


def migrate(db_path: Path, *, dry_run: bool = False) -> MigrationResult:
    conn = sqlite3.connect(db_path)
    try:
        added_column = not _has_column(conn, "episodes", "session_id")
        if added_column and not dry_run:
            conn.execute("ALTER TABLE episodes ADD COLUMN session_id TEXT")

        added_indexes: list[str] = []
        for name, ddl in _INDEX_SPECS:
            if _has_index(conn, name):
                continue
            added_indexes.append(name)
            if not dry_run:
                conn.execute(ddl)

        migrated = _count_legacy_derived_from(conn)
        if migrated > 0 and not dry_run:
            conn.execute(
                "UPDATE episodes "
                "SET session_id = derived_from, derived_from = NULL "
                "WHERE typeof(derived_from) = 'text'"
            )

        if not dry_run:
            conn.commit()

        return MigrationResult(
            added_session_id_column=added_column,
            added_indexes=added_indexes,
            migrated_rows=migrated,
            dry_run=dry_run,
        )
    finally:
        conn.close()
