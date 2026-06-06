"""Unit tests for migrations/judge_columns.py (Phase 5 P5-4)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fravenir.migrations.judge_columns import migrate


def _init_phase4_db(db: Path) -> None:
    """Create merge_candidates table with Phase 4 DDL (no judge_* columns)."""
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT    NOT NULL,
                entity_type    TEXT,
                description    TEXT,
                is_self        INTEGER NOT NULL DEFAULT 0,
                valid_from     TIMESTAMP NOT NULL,
                valid_to       TIMESTAMP,
                supersedes     INTEGER REFERENCES entities(id)
            );
            CREATE TABLE IF NOT EXISTS merge_candidates (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a    INTEGER NOT NULL REFERENCES entities(id),
                entity_b    INTEGER NOT NULL REFERENCES entities(id),
                similarity  REAL    NOT NULL,
                detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                resolved    INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_migrate_adds_columns(tmp_path: Path) -> None:
    db = tmp_path / "kv.sqlite"
    _init_phase4_db(db)

    result = migrate(db, dry_run=False)

    assert result.dry_run is False
    assert len(result.added_columns) == 4

    conn = sqlite3.connect(db)
    try:
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(merge_candidates)").fetchall()
        ]
    finally:
        conn.close()
    assert "judge_label" in columns
    assert "judge_confidence" in columns
    assert "judge_reason" in columns
    assert "judge_attempts" in columns


def test_migrate_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "kv.sqlite"
    _init_phase4_db(db)

    migrate(db, dry_run=False)
    result2 = migrate(db, dry_run=False)

    assert result2.added_columns == []


def test_migrate_dry_run(tmp_path: Path) -> None:
    db = tmp_path / "kv.sqlite"
    _init_phase4_db(db)

    result = migrate(db, dry_run=True)

    assert result.dry_run is True
    assert len(result.added_columns) == 4

    conn = sqlite3.connect(db)
    try:
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(merge_candidates)").fetchall()
        ]
    finally:
        conn.close()
    assert "judge_label" not in columns
