"""Tests for resolved_at migration."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fravenir.migrations.resolved_at import migrate
from fravenir.storage.sqlite_init import init_kv


def test_migrate_adds_resolved_at_column(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_kv(db)
    # 既存 DB では resolved_at 列があるはず（init_kv が最新 DDL を使うため）
    # 手動で列を削除して旧 DB をシミュレート
    conn = sqlite3.connect(db)
    try:
        conn.execute("ALTER TABLE merge_candidates DROP COLUMN resolved_at")
        conn.commit()
    finally:
        conn.close()

    result = migrate(db, dry_run=False)
    assert "resolved_at" in result.added_columns

    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(merge_candidates)").fetchall()]
        assert "resolved_at" in cols
    finally:
        conn.close()


def test_migrate_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_kv(db)
    result1 = migrate(db, dry_run=False)
    assert result1.added_columns == []

    result2 = migrate(db, dry_run=False)
    assert result2.added_columns == []


def test_migrate_dry_run(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_kv(db)
    # 列を削除して旧状態に
    conn = sqlite3.connect(db)
    try:
        conn.execute("ALTER TABLE merge_candidates DROP COLUMN resolved_at")
        conn.commit()
    finally:
        conn.close()

    preview = migrate(db, dry_run=True)
    assert "resolved_at" in preview.added_columns

    # DB は変更されていない
    conn = sqlite3.connect(db)
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(merge_candidates)").fetchall()]
        assert "resolved_at" not in cols
    finally:
        conn.close()
