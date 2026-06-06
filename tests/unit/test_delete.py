"""Unit tests for core/delete.py."""

import sqlite3
from pathlib import Path

import pytest

from fravenir.core.delete import memory_delete
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb(data_dir / "vdb_memories.db")
    return char_id


def _make_config(char_id: str = "test_char") -> AppConfig:
    return AppConfig(character=CharacterConfig(id=char_id))


def _insert_episode(tmp_project: Path, char_id: str, content: str = "テスト") -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO episodes (content, kind, importance, valid_from)
            VALUES (?, 'facts', 1, '2026-01-01T00:00:00+00:00')
            """,
            (content,),
        )
        conn.commit()
        ep_id: int = cur.lastrowid  # type: ignore[assignment]
        return ep_id
    finally:
        conn.close()


def _insert_relation(
    tmp_project: Path, char_id: str, src_type: str, src_id: int, dst_type: str, dst_id: int
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from)
            VALUES (?, ?, ?, ?, 'mentions', '2026-01-01T00:00:00+00:00')
            """,
            (src_type, src_id, dst_type, dst_id),
        )
        conn.commit()
        rel_id: int = cur.lastrowid  # type: ignore[assignment]
        return rel_id
    finally:
        conn.close()


class TestMemoryDelete:
    def test_sets_valid_to(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)

        result = memory_delete(
            ep_id, "不要になった", character_id=char_id, config=_make_config(char_id)
        )

        assert result["episode_id"] == ep_id
        assert isinstance(result["valid_to"], str)

        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        row = conn.execute("SELECT valid_to FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        assert row[0] == result["valid_to"]

    def test_is_suppressed_not_touched(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)

        memory_delete(ep_id, "削除理由", character_id=char_id, config=_make_config(char_id))

        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        row = conn.execute("SELECT is_suppressed FROM episodes WHERE id = ?", (ep_id,)).fetchone()
        assert row[0] == 0

    def test_memory_delete_cascades_mentions(self, tmp_project: Path) -> None:
        """memory_delete が episode を src とする relations も archive する"""
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)

        # 別 entity を作って mentions relation を2件作る
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            cur = conn.execute(
                "INSERT INTO entities (canonical_name, entity_type, valid_from) VALUES (?, ?, ?)",
                ("EntityA", "person", "2026-01-01T00:00:00+00:00"),
            )
            entity_a = cur.lastrowid
            cur = conn.execute(
                "INSERT INTO entities (canonical_name, entity_type, valid_from) VALUES (?, ?, ?)",
                ("EntityB", "person", "2026-01-01T00:00:00+00:00"),
            )
            entity_b = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

        rel1 = _insert_relation(tmp_project, char_id, "episode", ep_id, "entity", entity_a)
        rel2 = _insert_relation(tmp_project, char_id, "episode", ep_id, "entity", entity_b)

        memory_delete(ep_id, "不要になった", character_id=char_id, config=_make_config(char_id))

        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            for rel_id in (rel1, rel2):
                row = conn.execute(
                    "SELECT valid_to FROM relations WHERE id = ?", (rel_id,)
                ).fetchone()
                assert row[0] is not None, f"relation {rel_id} should be archived"
        finally:
            conn.close()


class TestMemoryDeleteValidation:
    def test_missing_episode_raises_keyerror(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        with pytest.raises(KeyError):
            memory_delete(9999, "理由", character_id=char_id, config=_make_config(char_id))

    def test_empty_reason_raises_valueerror(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)
        with pytest.raises(ValueError, match="reason"):
            memory_delete(ep_id, "   ", character_id=char_id, config=_make_config(char_id))

    def test_double_delete_raises_valueerror(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)

        memory_delete(ep_id, "1回目", character_id=char_id, config=_make_config(char_id))
        with pytest.raises(ValueError, match="already deleted"):
            memory_delete(ep_id, "2回目", character_id=char_id, config=_make_config(char_id))
