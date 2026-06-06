"""Unit tests for core/self_cue.py."""

import sqlite3
from pathlib import Path

from fravenir.core.self_cue import has_self_cue, self_cue_terms
from fravenir.storage import sqlite_init


def _fresh_kv(tmp_path: Path) -> sqlite3.Connection:
    db = tmp_path / "kv.sqlite"
    sqlite_init.init_kv(db)
    return sqlite3.connect(str(db))


def _insert_self_entity(
    conn: sqlite3.Connection, canonical_name: str, aliases: list[str]
) -> int:
    cur = conn.execute(
        """
        INSERT INTO entities (canonical_name, entity_type, is_self, self_weight,
                              decay_rate, valid_from)
        VALUES (?, 'person', 1, 1.0, 0.2, '2026-01-01T00:00:00+00:00')
        """,
        (canonical_name,),
    )
    eid: int = cur.lastrowid  # type: ignore[assignment]
    for a in aliases:
        conn.execute(
            "INSERT INTO entity_aliases (alias, entity_id) VALUES (?, ?)", (a, eid)
        )
    conn.commit()
    return eid


def _insert_personality(
    conn: sqlite3.Connection, canonical_name: str, self_weight: float
) -> int:
    cur = conn.execute(
        """
        INSERT INTO entities (canonical_name, entity_type, is_self, self_weight,
                              decay_rate, valid_from)
        VALUES (?, 'concept', 0, ?, 0.3, '2026-01-01T00:00:00+00:00')
        """,
        (canonical_name, self_weight),
    )
    conn.commit()
    pid: int = cur.lastrowid  # type: ignore[assignment]
    return pid


class TestSelfCueTerms:
    def test_empty_db(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            assert self_cue_terms(conn) == set()
        finally:
            conn.close()

    def test_identity_and_aliases(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", ["あたし", "ミナ"])
            terms = self_cue_terms(conn)
        finally:
            conn.close()
        assert "mina" in terms
        assert "あたし" in terms
        assert "ミナ" in terms

    def test_strong_personality_included(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", [])
            _insert_personality(conn, "好奇心旺盛", 0.8)
            terms = self_cue_terms(conn)
        finally:
            conn.close()
        assert "好奇心旺盛" in terms

    def test_weak_personality_excluded(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", [])
            _insert_personality(conn, "うっかり屋", 0.3)
            terms = self_cue_terms(conn)
        finally:
            conn.close()
        assert "うっかり屋" not in terms

    def test_threshold_boundary(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", [])
            _insert_personality(conn, "ちょうど", 0.7)
            assert "ちょうど" in self_cue_terms(conn, strong_threshold=0.7)
            assert "ちょうど" not in self_cue_terms(conn, strong_threshold=0.71)
        finally:
            conn.close()

    def test_archived_entity_excluded(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", ["あたし"])
            conn.execute(
                "UPDATE entities SET valid_to = '2026-04-01T00:00:00+00:00' "
                "WHERE canonical_name = 'mina'"
            )
            conn.commit()
            terms = self_cue_terms(conn)
        finally:
            conn.close()
        assert "mina" not in terms
        assert "あたし" not in terms


class TestHasSelfCue:
    def test_query_with_alias(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", ["あたし"])
            assert has_self_cue(conn, "あたしって何が好きだっけ？") is True
        finally:
            conn.close()

    def test_query_without_cue(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", ["あたし"])
            assert has_self_cue(conn, "今日の天気はどう？") is False
        finally:
            conn.close()

    def test_query_with_personality(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            _insert_self_entity(conn, "mina", [])
            _insert_personality(conn, "好奇心旺盛", 0.8)
            assert has_self_cue(conn, "好奇心旺盛って言われたことある") is True
        finally:
            conn.close()

    def test_empty_db_returns_false(self, tmp_path: Path) -> None:
        conn = _fresh_kv(tmp_path)
        try:
            assert has_self_cue(conn, "あたし") is False
        finally:
            conn.close()
