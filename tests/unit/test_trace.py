"""Unit tests for core/trace.py."""

import sqlite3
from pathlib import Path

import pytest

from fravenir.core.trace import memory_trace
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


def _insert_episode(
    tmp_project: Path,
    char_id: str,
    content: str,
    supersedes: int | None = None,
    valid_to: str | None = None,
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO episodes (content, kind, importance, valid_from, valid_to, supersedes)
            VALUES (?, 'facts', 1, '2026-01-01T00:00:00+00:00', ?, ?)
            """,
            (content, valid_to, supersedes),
        )
        conn.commit()
        ep_id: int = cur.lastrowid  # type: ignore[assignment]
        return ep_id
    finally:
        conn.close()


def _force_supersedes(tmp_project: Path, char_id: str, ep_id: int, target: int) -> None:
    """Bypass FK; used for circular-chain test."""
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        conn.execute("UPDATE episodes SET supersedes = ? WHERE id = ?", (target, ep_id))
        conn.commit()
    finally:
        conn.close()


class TestMemoryTrace:
    def test_single_node_no_chain(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id, "唯一")

        result = memory_trace(ep_id, character_id=char_id, config=_make_config(char_id))

        assert result["episode_id"] == ep_id
        assert len(result["chain"]) == 1  # type: ignore[arg-type]
        assert result["chain"][0]["id"] == ep_id  # type: ignore[index]
        assert result["chain"][0]["content"] == "唯一"  # type: ignore[index]

    def test_multi_segment_chain(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        v1 = _insert_episode(tmp_project, char_id, "v1", valid_to="2026-02-01T00:00:00+00:00")
        v2 = _insert_episode(
            tmp_project, char_id, "v2", supersedes=v1, valid_to="2026-03-01T00:00:00+00:00"
        )
        v3 = _insert_episode(tmp_project, char_id, "v3", supersedes=v2)

        result = memory_trace(v3, character_id=char_id, config=_make_config(char_id))

        chain = result["chain"]
        assert [c["id"] for c in chain] == [v3, v2, v1]  # type: ignore[union-attr]
        assert [c["content"] for c in chain] == ["v3", "v2", "v1"]  # type: ignore[union-attr]
        assert chain[0]["valid_to"] is None  # type: ignore[index]
        assert chain[2]["valid_to"] == "2026-02-01T00:00:00+00:00"  # type: ignore[index]

    def test_cycle_guarded(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        a = _insert_episode(tmp_project, char_id, "A")
        b = _insert_episode(tmp_project, char_id, "B", supersedes=a)
        _force_supersedes(tmp_project, char_id, a, b)  # A -> B -> A

        result = memory_trace(b, character_id=char_id, config=_make_config(char_id))

        ids = [c["id"] for c in result["chain"]]  # type: ignore[union-attr]
        assert ids == [b, a]  # stopped at revisit, no infinite loop


    def test_trace_returns_multi_step_chain(self, tmp_project: Path) -> None:
        """3 段の supersedes チェーンが正しい順序で返る。"""
        char_id = _make_character(tmp_project)
        v1 = _insert_episode(
            tmp_project, char_id, "プログラマ", valid_to="2026-02-01T00:00:00+00:00"
        )
        v2 = _insert_episode(
            tmp_project, char_id, "デザイナ",
            supersedes=v1, valid_to="2026-03-01T00:00:00+00:00",
        )
        v3 = _insert_episode(
            tmp_project, char_id, "ライター", supersedes=v2,
        )

        result = memory_trace(v3, character_id=char_id, config=_make_config(char_id))

        chain = result["chain"]
        assert len(chain) == 3  # type: ignore[arg-type]
        assert [c["id"] for c in chain] == [v3, v2, v1]  # type: ignore[union-attr]
        assert [c["content"] for c in chain] == ["ライター", "デザイナ", "プログラマ"]  # type: ignore[union-attr]
        # 最新だけ valid_to=NULL
        assert chain[0]["valid_to"] is None  # type: ignore[index]
        assert chain[1]["valid_to"] == "2026-03-01T00:00:00+00:00"  # type: ignore[index]
        assert chain[2]["valid_to"] == "2026-02-01T00:00:00+00:00"  # type: ignore[index]


class TestMemoryTraceValidation:
    def test_missing_episode_raises_keyerror(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        with pytest.raises(KeyError):
            memory_trace(9999, character_id=char_id, config=_make_config(char_id))
