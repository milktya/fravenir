"""Unit tests for core/explore.py (FEAT-1 memory_explore)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fravenir.core.explore import memory_explore
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    return char_id


def _make_config(char_id: str = "test_char") -> AppConfig:
    return AppConfig(character=CharacterConfig(id=char_id))


def _open_kv(tmp_project: Path, char_id: str) -> sqlite3.Connection:
    return sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))


def _insert_episode(
    conn: sqlite3.Connection,
    content: str = "ep",
    kind: str = "facts",
    importance: int = 1,
    valid_to: str | None = None,
    is_suppressed: int = 0,
) -> int:
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        """INSERT INTO episodes
            (content, kind, importance, valid_from, valid_to, is_suppressed)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (content, kind, importance, now, valid_to, is_suppressed),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def _insert_entity(
    conn: sqlite3.Connection,
    canonical_name: str = "ent",
    entity_type: str = "concept",
    description: str = "",
    is_self: int = 0,
    self_weight: float = 0.0,
    decay_rate: float = 0.5,
    valid_to: str | None = None,
) -> int:
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        """INSERT INTO entities
            (canonical_name, entity_type, description, is_self, self_weight,
             decay_rate, valid_from, valid_to)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            canonical_name, entity_type, description, is_self, self_weight,
            decay_rate, now, valid_to,
        ),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def _insert_relation(
    conn: sqlite3.Connection,
    src_type: str,
    src_id: int,
    dst_type: str,
    dst_id: int,
    predicate: str,
    strength: float = 1.0,
    valid_to: str | None = None,
) -> int:
    now = datetime.now(UTC).isoformat()
    cur = conn.execute(
        """INSERT INTO relations
            (src_type, src_id, dst_type, dst_id, predicate, strength,
             valid_from, valid_to)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (src_type, src_id, dst_type, dst_id, predicate, strength, now, valid_to),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def _insert_access(
    conn: sqlite3.Connection,
    node_type: str,
    node_id: int,
    accessed_at: str,
    source: str = "test",
) -> None:
    conn.execute(
        """INSERT INTO access_history (node_type, node_id, accessed_at, source)
        VALUES (?, ?, ?, ?)""",
        (node_type, node_id, accessed_at, source),
    )
    conn.commit()


# ---------- Errors ----------


class TestErrors:
    def test_node_not_found(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        with pytest.raises(ValueError, match="not found"):
            memory_explore("entity", 999, character_id=char_id, config=config)

    def test_depth_2_raises_not_implemented(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        ent_id = _insert_entity(conn)
        conn.close()
        with pytest.raises(NotImplementedError, match="depth >= 2"):
            memory_explore(
                "entity", ent_id, depth=2,
                character_id=char_id, config=config,
            )

    def test_depth_0_raises_value_error(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        with pytest.raises(ValueError, match="depth must be >= 1"):
            memory_explore(
                "entity", 1, depth=0,
                character_id=char_id, config=config,
            )


# ---------- Node content ----------


class TestNodeContent:
    def test_entity_node_fields(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        ent_id = _insert_entity(
            conn, canonical_name="mina", description="自己ハブ",
            is_self=1, self_weight=1.0, decay_rate=0.2,
        )
        conn.close()
        result = memory_explore(
            "entity", ent_id, character_id=char_id, config=config,
        )
        assert result.node.type == "entity"
        assert result.node.id == ent_id
        assert result.node.name == "mina"
        assert result.node.content == "自己ハブ"
        assert result.node.is_self is True
        assert result.node.self_weight == 1.0
        assert result.node.decay_rate == 0.2
        assert result.node.importance == 1
        # access_history なしなら B_i = 0.0 > -2.0
        assert result.node.is_suppressed is False

    def test_episode_node_fields(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        ep_id = _insert_episode(conn, content="今日の出来事", importance=3)
        conn.close()
        result = memory_explore(
            "episode", ep_id, character_id=char_id, config=config,
        )
        assert result.node.type == "episode"
        assert result.node.name is None
        assert result.node.content == "今日の出来事"
        assert result.node.importance == 3
        assert result.node.is_suppressed is False

    def test_long_content_truncated_by_default(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        long_text = "あ" * 1000
        conn = _open_kv(tmp_project, char_id)
        ep_id = _insert_episode(conn, content=long_text)
        conn.close()
        result = memory_explore(
            "episode", ep_id, character_id=char_id, config=config,
        )
        assert result.node.is_truncated is True
        assert result.node.content.endswith("...")

    def test_full_flag_returns_untruncated(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        long_text = "あ" * 1000
        conn = _open_kv(tmp_project, char_id)
        ep_id = _insert_episode(conn, content=long_text)
        conn.close()
        result = memory_explore(
            "episode", ep_id, full=True, character_id=char_id, config=config,
        )
        assert result.node.is_truncated is False
        assert result.node.content == long_text


# ---------- Neighbor retrieval & direction ----------


class TestNeighbors:
    def test_entity_with_incoming_and_outgoing(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        mina = _insert_entity(conn, "mina")
        trait = _insert_entity(conn, "好奇心旺盛")
        category = _insert_entity(conn, "AIキャラ")
        ep = _insert_episode(conn, "ある日の記録")
        _insert_relation(conn, "entity", trait, "entity", mina, "part_of")
        _insert_relation(conn, "entity", mina, "entity", category, "is_a")
        _insert_relation(conn, "episode", ep, "entity", mina, "mentions")
        conn.close()
        result = memory_explore(
            "entity", mina, character_id=char_id, config=config,
        )
        total = sum(len(v) for v in result.neighbors.values())
        assert total == 3
        assert set(result.neighbors.keys()) == {"part_of", "is_a", "mentions"}
        assert result.neighbors["part_of"][0].direction == "incoming"
        assert result.neighbors["is_a"][0].direction == "outgoing"
        assert result.neighbors["mentions"][0].direction == "incoming"

    def test_episode_outgoing_to_entities(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        ep = _insert_episode(conn, "記録")
        ent1 = _insert_entity(conn, "猫")
        ent2 = _insert_entity(conn, "魚")
        _insert_relation(conn, "episode", ep, "entity", ent1, "mentions")
        _insert_relation(conn, "episode", ep, "entity", ent2, "mentions")
        conn.close()
        result = memory_explore(
            "episode", ep, character_id=char_id, config=config,
        )
        assert "mentions" in result.neighbors
        items = result.neighbors["mentions"]
        assert len(items) == 2
        assert all(i.direction == "outgoing" for i in items)

    def test_same_neighbor_multi_predicate_not_deduped(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        e1 = _insert_entity(conn, "mina")
        e2 = _insert_entity(conn, "猫")
        _insert_relation(conn, "entity", e1, "entity", e2, "likes")
        _insert_relation(conn, "entity", e1, "entity", e2, "has")
        conn.close()
        result = memory_explore(
            "entity", e1, character_id=char_id, config=config,
        )
        assert "likes" in result.neighbors
        assert "has" in result.neighbors
        all_ids = [
            item.id for items in result.neighbors.values() for item in items
        ]
        assert all_ids.count(e2) == 2


# ---------- Exclude ids ----------


class TestExcludeIds:
    def test_exclude_episode_ids_blocks_only_episodes(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ep = _insert_episode(conn, "記録1")
        ent_neighbor = _insert_entity(conn, "好奇心")
        _insert_relation(conn, "episode", ep, "entity", target, "mentions")
        _insert_relation(
            conn, "entity", ent_neighbor, "entity", target, "part_of",
        )
        conn.close()
        result = memory_explore(
            "entity", target, exclude_episode_ids=[ep],
            character_id=char_id, config=config,
        )
        assert "mentions" not in result.neighbors
        assert "part_of" in result.neighbors
        assert result.neighbors["part_of"][0].id == ent_neighbor

    def test_exclude_entity_ids_blocks_only_entities(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ep = _insert_episode(conn, "記録")
        ent = _insert_entity(conn, "好奇心")
        _insert_relation(conn, "episode", ep, "entity", target, "mentions")
        _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        result = memory_explore(
            "entity", target, exclude_entity_ids=[ent],
            character_id=char_id, config=config,
        )
        assert "mentions" in result.neighbors
        assert "part_of" not in result.neighbors

    def test_exclude_does_not_affect_origin(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ent = _insert_entity(conn, "trait")
        _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        result = memory_explore(
            "entity", target, exclude_entity_ids=[target],
            character_id=char_id, config=config,
        )
        # 起点は exclude されても返る
        assert result.node.id == target
        # neighbor も普通に残る
        assert "part_of" in result.neighbors


# ---------- Archived filter ----------


class TestArchivedFilter:
    def test_archived_relation_excluded_by_default(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ent = _insert_entity(conn, "古い特徴")
        archived_ts = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        _insert_relation(
            conn, "entity", ent, "entity", target, "part_of",
            valid_to=archived_ts,
        )
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        assert "part_of" not in result.neighbors
        assert result.total_neighbors == 0
        # SQL レベルで弾かれるので unfiltered にも入らない
        assert result.total_neighbors_unfiltered == 0

    def test_include_archived_returns_archived_relation(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ent = _insert_entity(conn, "古い特徴")
        archived_ts = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        _insert_relation(
            conn, "entity", ent, "entity", target, "part_of",
            valid_to=archived_ts,
        )
        conn.close()
        result = memory_explore(
            "entity", target, include_archived=True,
            character_id=char_id, config=config,
        )
        assert "part_of" in result.neighbors

    def test_archived_neighbor_excluded_by_default(
        self, tmp_project: Path,
    ) -> None:
        """relation は active でも neighbor 本体が archived なら除外。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        archived_ts = (datetime.now(UTC) - timedelta(days=30)).isoformat()
        ent = _insert_entity(conn, "古い特徴", valid_to=archived_ts)
        _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        assert "part_of" not in result.neighbors
        # relation 自体は active なので unfiltered に入る
        assert result.total_neighbors_unfiltered == 1


# ---------- Suppressed filter ----------


class TestSuppressedFilter:
    def test_suppressed_episode_excluded_by_default(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ep = _insert_episode(conn, "古い記録", is_suppressed=1)
        _insert_relation(conn, "episode", ep, "entity", target, "mentions")
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        assert "mentions" not in result.neighbors

    def test_include_suppressed_returns_suppressed_episode(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ep = _insert_episode(conn, "古い記録", is_suppressed=1)
        _insert_relation(conn, "episode", ep, "entity", target, "mentions")
        conn.close()
        result = memory_explore(
            "entity", target, include_suppressed=True,
            character_id=char_id, config=config,
        )
        assert "mentions" in result.neighbors

    def test_suppressed_entity_dynamic(self, tmp_project: Path) -> None:
        """entity 側は base_activation 動的判定（v2_design §12.2 #6 で
        is_suppressed カラム保留中）。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ent = _insert_entity(conn, "忘れかけ特徴", decay_rate=0.5)
        # 1 年前の単発アクセス → B_i ≈ -5.7 で threshold -2.0 未満
        old_ts = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        _insert_access(conn, "entity", ent, old_ts)
        _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        assert "part_of" not in result.neighbors

    def test_origin_entity_is_suppressed_set_when_low_activation(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        ent = _insert_entity(conn, "忘れかけ", decay_rate=0.5)
        old_ts = (datetime.now(UTC) - timedelta(days=365)).isoformat()
        _insert_access(conn, "entity", ent, old_ts)
        conn.close()
        result = memory_explore(
            "entity", ent, character_id=char_id, config=config,
        )
        assert result.node.is_suppressed is True


# ---------- Limits & meta ----------


class TestLimits:
    def test_per_predicate_limit_5(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        for i in range(7):
            ent = _insert_entity(conn, f"trait_{i}")
            _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        assert len(result.neighbors["part_of"]) == 5
        assert result.meta["part_of"].shown == 5
        assert result.meta["part_of"].total == 7

    def test_total_limit_10(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        # 3 predicate × 5 件 = 15、各 predicate 5 で計 15 → 全体 10 で絞り
        for pred in ("part_of", "likes", "is_a"):
            for i in range(5):
                ent = _insert_entity(conn, f"{pred}_ent_{i}")
                _insert_relation(
                    conn, "entity", ent, "entity", target, pred,
                )
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        total = sum(len(v) for v in result.neighbors.values())
        assert total == 10

    def test_meta_keeps_dropped_predicate_with_shown_zero(
        self, tmp_project: Path,
    ) -> None:
        """全体上限で predicate ごと落ちても meta に shown=0 で残る。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        # 強い predicate を 2 種類 × 5 件、弱い predicate 1 種類 × 5 件
        for i in range(5):
            e = _insert_entity(conn, f"strong_a_{i}")
            _insert_relation(
                conn, "entity", e, "entity", target, "strong_a",
                strength=10.0,
            )
        for i in range(5):
            e = _insert_entity(conn, f"strong_b_{i}")
            _insert_relation(
                conn, "entity", e, "entity", target, "strong_b",
                strength=10.0,
            )
        for i in range(5):
            e = _insert_entity(conn, f"weak_{i}")
            _insert_relation(
                conn, "entity", e, "entity", target, "weak",
                strength=0.001,
            )
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        # 全体 10 で strong_a + strong_b が埋め、weak は脱落
        assert "weak" not in result.neighbors
        assert "weak" in result.meta
        assert result.meta["weak"].shown == 0
        assert result.meta["weak"].total == 5

    def test_total_neighbors_unfiltered_counts_all_relations(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        for i in range(15):
            ent = _insert_entity(conn, f"ent_{i}")
            _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        result = memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        assert result.total_neighbors_unfiltered == 15
        assert result.total_neighbors == 5  # part_of 上限 5


# ---------- Rehearsal effect ----------


class TestAccessHistory:
    def test_access_history_records_explore_source(
        self, tmp_project: Path,
    ) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ent = _insert_entity(conn, "trait")
        _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        conn = _open_kv(tmp_project, char_id)
        rows = conn.execute(
            "SELECT node_type, node_id, source FROM access_history "
            "ORDER BY id",
        ).fetchall()
        conn.close()
        sources = {r[2] for r in rows}
        assert "explore" in sources
        target_set = {(r[0], r[1]) for r in rows if r[2] == "explore"}
        assert ("entity", target) in target_set
        assert ("entity", ent) in target_set

    def test_activation_count_incremented(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        target = _insert_entity(conn, "mina")
        ent = _insert_entity(conn, "trait")
        _insert_relation(conn, "entity", ent, "entity", target, "part_of")
        conn.close()
        memory_explore(
            "entity", target, character_id=char_id, config=config,
        )
        conn = _open_kv(tmp_project, char_id)
        target_count = conn.execute(
            "SELECT activation_count FROM entities WHERE id = ?", (target,),
        ).fetchone()[0]
        ent_count = conn.execute(
            "SELECT activation_count FROM entities WHERE id = ?", (ent,),
        ).fetchone()[0]
        conn.close()
        assert target_count >= 1
        assert ent_count >= 1

    def test_no_duplicate_access_for_same_neighbor(
        self, tmp_project: Path,
    ) -> None:
        """同一 neighbor が複数 predicate に出ても access_history は 1 件のみ。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        conn = _open_kv(tmp_project, char_id)
        e1 = _insert_entity(conn, "mina")
        e2 = _insert_entity(conn, "猫")
        _insert_relation(conn, "entity", e1, "entity", e2, "likes")
        _insert_relation(conn, "entity", e1, "entity", e2, "has")
        conn.close()
        memory_explore(
            "entity", e1, character_id=char_id, config=config,
        )
        conn = _open_kv(tmp_project, char_id)
        e2_access_count = conn.execute(
            """SELECT COUNT(*) FROM access_history
            WHERE node_type = 'entity' AND node_id = ? AND source = 'explore'""",
            (e2,),
        ).fetchone()[0]
        conn.close()
        # 2 つの predicate で表示されるが access_history は 1 件
        assert e2_access_count == 1
