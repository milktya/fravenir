"""Migration test for entities.curated_at + admin_audit_log (Phase 6)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fravenir.migrations.curated_and_audit import migrate
from fravenir.storage.sqlite_init import init_kv


def _has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(row[1] == col for row in conn.execute(f"PRAGMA table_info({table})"))


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _make_legacy_db(path: Path) -> None:
    """curated_at カラムも admin_audit_log テーブルも持たない古いスキーマを再現。"""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                entity_type TEXT,
                description TEXT,
                is_self INTEGER NOT NULL DEFAULT 0,
                self_weight REAL NOT NULL DEFAULT 0.0,
                decay_rate REAL NOT NULL DEFAULT 0.5,
                valid_from TIMESTAMP NOT NULL,
                valid_to TIMESTAMP,
                supersedes INTEGER,
                last_activated_at TIMESTAMP,
                activation_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_migrate_adds_column_and_table(tmp_path: Path) -> None:
    db = tmp_path / "legacy.sqlite"
    _make_legacy_db(db)

    preview = migrate(db, dry_run=True)
    assert preview.added_columns == ["curated_at"]
    assert preview.created_tables == ["admin_audit_log"]
    assert preview.dry_run is True

    # dry_run のあとは変更されていない
    conn = sqlite3.connect(db)
    try:
        assert not _has_column(conn, "entities", "curated_at")
        assert not _has_table(conn, "admin_audit_log")
    finally:
        conn.close()

    result = migrate(db, dry_run=False)
    assert result.added_columns == ["curated_at"]
    assert result.created_tables == ["admin_audit_log"]

    conn = sqlite3.connect(db)
    try:
        assert _has_column(conn, "entities", "curated_at")
        assert _has_table(conn, "admin_audit_log")
        # 既存行は curated_at NULL のまま
        conn.execute(
            "INSERT INTO entities (canonical_name, valid_from) VALUES ('x', '2026-04-01')"
        )
        row = conn.execute(
            "SELECT curated_at FROM entities WHERE canonical_name='x'"
        ).fetchone()
        assert row[0] is None
    finally:
        conn.close()


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "kv.sqlite"
    init_kv(db)  # 新スキーマは最初から curated_at + admin_audit_log を持つ

    preview = migrate(db, dry_run=True)
    assert preview.added_columns == []
    assert preview.created_tables == []

    result = migrate(db, dry_run=False)
    assert result.added_columns == []
    assert result.created_tables == []


def test_fresh_init_includes_new_schema(tmp_path: Path) -> None:
    """init_kv 直後でも curated_at と admin_audit_log が揃っていること。"""
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    conn = sqlite3.connect(db)
    try:
        assert _has_column(conn, "entities", "curated_at")
        assert _has_table(conn, "admin_audit_log")
        # 監査ログテーブルに INSERT できる
        conn.execute(
            """
            INSERT INTO admin_audit_log
                (action, target_type, target_id, before_json, after_json, actor)
            VALUES ('entity.update', 'entity', 1, '{}', '{}', 'admin_ui')
            """
        )
        conn.commit()
    finally:
        conn.close()
