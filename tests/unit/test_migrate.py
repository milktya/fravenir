"""tests for fravenir.migrations.session_id."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fravenir.migrations.session_id import migrate
from fravenir.storage.sqlite_init import init_kv


def _make_old_schema_db(tmp_path: Path) -> Path:
    """旧 DDL (derived_from TEXT, session_id 列なし) の DB を再現する。"""
    db = tmp_path / "kv.sqlite"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE episodes (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                content           TEXT    NOT NULL,
                kind              TEXT    NOT NULL,
                importance        INTEGER NOT NULL DEFAULT 1,
                valid_from        TIMESTAMP NOT NULL,
                valid_to          TIMESTAMP,
                supersedes        INTEGER REFERENCES episodes(id),
                derived_from      TEXT,
                last_activated_at TIMESTAMP,
                activation_count  INTEGER NOT NULL DEFAULT 0,
                is_suppressed     INTEGER NOT NULL DEFAULT 0,
                created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE merge_candidates (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_a    INTEGER NOT NULL,
                entity_b    INTEGER NOT NULL,
                similarity  REAL    NOT NULL,
                detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                resolved    INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO episodes (content, kind, importance, valid_from, derived_from)
            VALUES ('a', 'state', 1, '2026-04-01T00:00:00', 'session-1'),
                   ('b', 'state', 1, '2026-04-02T00:00:00', 'session-2'),
                   ('c', 'state', 1, '2026-04-03T00:00:00', NULL);
            """
        )
        conn.commit()
    finally:
        conn.close()
    return db


class TestMigrateSessionId:
    def test_adds_session_id_column(self, tmp_path: Path) -> None:
        db = _make_old_schema_db(tmp_path)

        result = migrate(db)

        assert result.added_session_id_column is True
        conn = sqlite3.connect(db)
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)")]
            assert "session_id" in cols
        finally:
            conn.close()

    def test_migrates_text_derived_from(self, tmp_path: Path) -> None:
        db = _make_old_schema_db(tmp_path)

        result = migrate(db)

        assert result.migrated_rows == 2
        conn = sqlite3.connect(db)
        try:
            rows = list(
                conn.execute(
                    "SELECT content, derived_from, session_id FROM episodes ORDER BY id"
                )
            )
        finally:
            conn.close()
        assert rows == [
            ("a", None, "session-1"),
            ("b", None, "session-2"),
            ("c", None, None),
        ]

    def test_adds_indexes(self, tmp_path: Path) -> None:
        db = _make_old_schema_db(tmp_path)

        result = migrate(db)

        assert "idx_episodes_session_id" in result.added_indexes
        assert "idx_merge_candidates_pair" in result.added_indexes
        assert "idx_merge_candidates_resolved" in result.added_indexes

    def test_idempotent(self, tmp_path: Path) -> None:
        db = _make_old_schema_db(tmp_path)
        first = migrate(db)
        assert first.migrated_rows == 2

        second = migrate(db)

        assert second.added_session_id_column is False
        assert second.added_indexes == []
        assert second.migrated_rows == 0

    def test_dry_run_no_side_effects(self, tmp_path: Path) -> None:
        db = _make_old_schema_db(tmp_path)

        result = migrate(db, dry_run=True)

        assert result.dry_run is True
        assert result.added_session_id_column is True
        assert result.migrated_rows == 2
        conn = sqlite3.connect(db)
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(episodes)")]
            assert "session_id" not in cols
            row = conn.execute(
                "SELECT derived_from FROM episodes WHERE content = 'a'"
            ).fetchone()
            assert row[0] == "session-1"
        finally:
            conn.close()

    def test_int_derived_from_is_preserved(self, tmp_path: Path) -> None:
        """新スキーマ DB に int derived_from を入れたとき PROV-O 用途として温存される。

        新スキーマでは derived_from が INTEGER REFERENCES なので、typeof は
        'integer' になる。ガード ``typeof = 'text'`` で session_id 移送対象外
        になることを確認する。
        """
        db = tmp_path / "kv.sqlite"
        init_kv(db)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO episodes "
                "(content, kind, importance, valid_from, derived_from) "
                "VALUES ('d', 'state', 1, '2026-04-04T00:00:00', 999)"
            )
            conn.commit()
        finally:
            conn.close()

        result = migrate(db)

        assert result.migrated_rows == 0
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT derived_from, session_id FROM episodes WHERE content = 'd'"
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == 999
        assert row[1] is None

    def test_new_schema_db_is_noop(self, tmp_path: Path) -> None:
        """新 DDL で初期化された DB は既に最終形なので変更は発生しない。"""
        db = tmp_path / "kv.sqlite"
        init_kv(db)

        result = migrate(db)

        assert result.added_session_id_column is False
        assert result.added_indexes == []
        assert result.migrated_rows == 0
