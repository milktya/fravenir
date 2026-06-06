"""Unit tests for core/search.py."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import sqlite_vec

from fravenir.core.search import memory_search
from fravenir.core.write import memory_write
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init
from fravenir.storage.vector import upsert_entity_vector


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb(data_dir / "vdb_memories.db")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    return char_id


def _make_config(char_id: str = "test_char") -> AppConfig:
    return AppConfig(character=CharacterConfig(id=char_id))


def _make_embedder(dim: int = 768) -> MagicMock:
    embedder = MagicMock()
    unit = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    embedder.encode_document.return_value = unit
    embedder.encode_query.return_value = unit
    embedder.encode_topic.return_value = unit
    return embedder


class TestMemorySearch:
    def test_returns_list(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        memory_write(
            "記憶1", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        results = memory_search(
            "記憶",
            limit=5,
            character_id=char_id,
            config=config,
            embedder=embedder,
        )
        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_has_required_fields(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        memory_write(
            "詳細テスト", "facts", 2, "s1",
            character_id=char_id, config=config, embedder=embedder,
        )
        results = memory_search(
            "詳細", character_id=char_id, config=config, embedder=embedder,
        )
        assert len(results) == 1
        item = results[0]
        required_keys = (
            "episode_id", "content", "kind", "importance",
            "activation", "score", "valid_from", "source",
        )
        for key in required_keys:
            assert key in item, f"missing key: {key}"

    def test_limit_respected(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        for i in range(10):
            memory_write(
                f"記憶{i}", "facts", 1, None,
                character_id=char_id, config=config, embedder=embedder,
            )
        results = memory_search(
            "記憶", limit=3,
            character_id=char_id, config=config, embedder=embedder,
        )
        assert len(results) <= 3

    def test_access_history_recorded(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        r = memory_write(
            "アクセス確認", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        memory_search(
            "確認", character_id=char_id, config=config, embedder=embedder,
        )

        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        row = kv.execute(
            "SELECT activation_count FROM episodes WHERE id = ?",
            (r["episode_id"],),
        ).fetchone()
        assert row is not None
        assert row[0] >= 1

    def test_kind_filter(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        memory_write(
            "facts記憶", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        memory_write(
            "emo記憶", "emo", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        results = memory_search(
            "記憶", kind_filter=["facts"],
            character_id=char_id, config=config, embedder=embedder,
        )
        assert all(r["kind"] == "facts" for r in results)

    def test_archived_excluded_by_default(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        r = memory_write(
            "アーカイブ記憶", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        # 論理削除
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        kv.execute(
            "UPDATE episodes SET valid_to = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), r["episode_id"]),
        )
        kv.commit()
        results = memory_search(
            "アーカイブ", character_id=char_id, config=config, embedder=embedder,
        )
        ids = [item["episode_id"] for item in results]
        assert r["episode_id"] not in ids

    def test_archived_included_when_flag_true(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        r = memory_write(
            "アーカイブ記憶", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        kv.execute(
            "UPDATE episodes SET valid_to = ? WHERE id = ?",
            (datetime.now(UTC).isoformat(), r["episode_id"]),
        )
        kv.commit()
        results = memory_search(
            "アーカイブ",
            include_archived=True,
            character_id=char_id, config=config, embedder=embedder,
        )
        ids = [item["episode_id"] for item in results]
        assert r["episode_id"] in ids

    def test_empty_db_returns_empty(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        results = memory_search(
            "なにか", character_id=char_id, config=config, embedder=embedder,
        )
        assert results == []

    def test_search_excludes_suppressed_after_compact(self, tmp_project: Path) -> None:
        """Phase 4 P4-3: compact で抑制された episode が memory_search から除外される。"""
        from datetime import timedelta

        from fravenir.core.compact import run_compact

        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        r = memory_write(
            "古い記憶", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        ep_id = r["episode_id"]

        # 1年前の単発アクセスを仕込んで B_i を threshold 未満に振る
        compact_now = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            kv.execute(
                "INSERT INTO access_history (node_type, node_id, accessed_at, source) "
                "VALUES ('episode', ?, ?, 'test')",
                (ep_id, (compact_now - timedelta(days=365)).isoformat()),
            )
            kv.commit()
        finally:
            kv.close()

        result = run_compact(character_id=char_id, config=config, now=compact_now)
        assert result.suppressed == 1

        # デフォルト (include_suppressed=False) では抑制 episode が除外される
        excluded = memory_search(
            "記憶", character_id=char_id, config=config, embedder=embedder,
        )
        assert all(item["episode_id"] != ep_id for item in excluded)

        # include_suppressed=True で復帰
        included = memory_search(
            "記憶",
            character_id=char_id,
            config=config,
            embedder=embedder,
            include_suppressed=True,
        )
        assert any(item["episode_id"] == ep_id for item in included)


def _seed_self_with_link(
    tmp_project: Path, char_id: str, embedder: MagicMock
) -> int:
    """自己 entity を作成し、直前に書き込んだ episode への entity->episode relation を張る。"""
    kv_path = tmp_project / "data" / char_id / "kv.sqlite"
    conn = sqlite3.connect(str(kv_path))
    try:
        cur = conn.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, is_self, self_weight,
                                  decay_rate, valid_from)
            VALUES ('mina', 'person', 1, 1.0, 0.2, '2026-01-01T00:00:00+00:00')
            """
        )
        eid: int = cur.lastrowid  # type: ignore[assignment]
        conn.execute(
            "INSERT INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
            ("あたし", eid),
        )
        row = conn.execute(
            "SELECT id FROM episodes ORDER BY id DESC LIMIT 1"
        ).fetchone()
        ep_id = int(row[0])
        conn.execute(
            """
            INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from)
            VALUES ('entity', ?, 'episode', ?, 'evidences', '2026-01-01T00:00:00+00:00')
            """,
            (eid, ep_id),
        )
        conn.commit()
        return eid
    finally:
        conn.close()


class TestSelfBoost:
    """Phase 2 / P5-7: self cue があると自己 entity を seed に合流し graph 経由で β 伝播。"""

    def test_boost_applied_via_self_entity_seed(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()

        memory_write(
            "あたしは好奇心が強い", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        _seed_self_with_link(tmp_project, char_id, embedder)

        results = memory_search(
            "あたしって何が好きだっけ？",
            character_id=char_id, config=config, embedder=embedder,
        )
        assert len(results) == 1
        assert results[0]["source"] == "self_boost"

    def test_no_boost_without_self_cue(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()

        memory_write(
            "あたしは好奇心が強い", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        _seed_self_with_link(tmp_project, char_id, embedder)

        results = memory_search(
            "好奇心の話",
            character_id=char_id, config=config, embedder=embedder,
        )
        assert len(results) == 1
        assert results[0]["source"] == "direct"

    def test_boost_increases_activation(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()

        memory_write(
            "あたしの好きなこと", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        _seed_self_with_link(tmp_project, char_id, embedder)

        boosted = memory_search(
            "あたしを教えて",
            character_id=char_id, config=config, embedder=embedder,
        )
        non_boost = memory_search(
            "教えて",
            character_id=char_id, config=config, embedder=embedder,
        )
        assert boosted[0]["activation"] > non_boost[0]["activation"]

    def test_self_boost_propagates_to_subjectless_episode(
        self, tmp_project: Path
    ) -> None:
        """content に主語が無いエピソードも自己 entity 経由でブーストされる。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()

        memory_write(
            "好きな食べ物はラーメン", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        _seed_self_with_link(tmp_project, char_id, embedder)

        results = memory_search(
            "あたしの好きなもの",
            character_id=char_id, config=config, embedder=embedder,
        )
        assert len(results) == 1
        assert results[0]["source"] == "self_boost"


def _make_character_phase3(tmp_project: Path, char_id: str = "test_char") -> str:
    """vdb_entities / vdb_relations まで初期化した Phase3 前提のキャラ領域。"""
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb(data_dir / "vdb_memories.db")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    sqlite_init.init_vdb_relations(data_dir / "vdb_relations.db")
    return char_id


def _open_vec(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _insert_assoc_pair(
    tmp_project: Path,
    char_id: str,
    entity_name: str,
    episode_content: str,
    entity_vec: np.ndarray,
    *,
    extra_dst_entities: int = 0,
) -> tuple[int, int]:
    """kv に entity+episode+relation、vdb_entities に entity vec を登録。

    extra_dst_entities>0 なら entity→別 entity の有効エッジを増やして fan_out を水増し。
    episode は vdb_memories には入れない（associative 経由だけで拾わせるため）。
    """
    kv_path = tmp_project / "data" / char_id / "kv.sqlite"
    conn = sqlite3.connect(str(kv_path))
    try:
        cur = conn.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, is_self, self_weight,
                                  decay_rate, valid_from)
            VALUES (?, 'concept', 0, 0.0, 0.5, '2026-01-01T00:00:00+00:00')
            """,
            (entity_name,),
        )
        entity_id = int(cur.lastrowid)  # type: ignore[arg-type]
        cur = conn.execute(
            """
            INSERT INTO episodes (content, kind, valid_from)
            VALUES (?, 'facts', '2026-01-01T00:00:00+00:00')
            """,
            (episode_content,),
        )
        episode_id = int(cur.lastrowid)  # type: ignore[arg-type]
        conn.execute(
            """
            INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate,
                                   valid_from)
            VALUES ('entity', ?, 'episode', ?, 'evidences',
                    '2026-01-01T00:00:00+00:00')
            """,
            (entity_id, episode_id),
        )
        for i in range(extra_dst_entities):
            cur = conn.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, valid_from)
                VALUES (?, 'filler', '2026-01-01T00:00:00+00:00')
                """,
                (f"{entity_name}_filler_{i}",),
            )
            filler_id = int(cur.lastrowid)  # type: ignore[arg-type]
            conn.execute(
                """
                INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate,
                                       valid_from)
                VALUES ('entity', ?, 'entity', ?, 'related',
                        '2026-01-01T00:00:00+00:00')
                """,
                (entity_id, filler_id),
            )
        conn.commit()
    finally:
        conn.close()

    vdb_path = tmp_project / "data" / char_id / "vdb_entities.db"
    vdb_conn = _open_vec(vdb_path)
    try:
        upsert_entity_vector(vdb_conn, entity_id, entity_vec.astype(np.float32))
    finally:
        vdb_conn.close()

    return entity_id, episode_id


class TestMinImportance:
    def test_min_importance_only_filters_suppressed(self, tmp_project: Path) -> None:
        """min_importance は is_suppressed=1 のエピソードのみに適用される。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()

        memory_write(
            "通常の低重要度", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        memory_write(
            "抑制された低重要度", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )

        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            # 2番目のエピソードを抑制
            kv.execute(
                "UPDATE episodes SET is_suppressed = 1 WHERE id = 2"
            )
            kv.commit()
        finally:
            kv.close()

        # min_importance=2: 通常エピソード (importance=1) は通る、抑制エピソードは弾かれる
        results = memory_search(
            "低重要度",
            min_importance=2,
            include_suppressed=True,
            character_id=char_id, config=config, embedder=embedder,
        )
        assert len(results) == 1
        assert results[0]["episode_id"] == 1


class TestGraphTraversal:
    """Phase 3: entity 近傍の 2ホップ BFS で連想ヒットが合流する。"""

    def test_associative_only_episode_surfaces(self, tmp_project: Path) -> None:
        char_id = _make_character_phase3(tmp_project)
        config = _make_config(char_id)
        dim = 768
        entity_vec = np.zeros(dim, dtype=np.float32)
        entity_vec[0] = 1.0

        embedder = MagicMock()
        embedder.encode_document.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_query.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_topic.return_value = entity_vec

        _, ep_id = _insert_assoc_pair(
            tmp_project, char_id,
            entity_name="料理",
            episode_content="カレーを作った",
            entity_vec=entity_vec,
        )
        results = memory_search(
            "料理の話",
            character_id=char_id, config=config, embedder=embedder,
        )
        matched = [r for r in results if r["episode_id"] == ep_id]
        assert len(matched) == 1
        assert matched[0]["source"] == "associative"

    def test_sji_contributes_to_activation(self, tmp_project: Path) -> None:
        char_id = _make_character_phase3(tmp_project)
        config = _make_config(char_id)
        dim = 768
        entity_vec = np.zeros(dim, dtype=np.float32)
        entity_vec[0] = 1.0

        embedder = MagicMock()
        embedder.encode_document.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_query.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_topic.return_value = entity_vec

        _, ep_id = _insert_assoc_pair(
            tmp_project, char_id,
            entity_name="料理",
            episode_content="カレーを作った",
            entity_vec=entity_vec,
        )
        results = memory_search(
            "料理の話",
            character_id=char_id, config=config, embedder=embedder,
        )
        matched = next(r for r in results if r["episode_id"] == ep_id)
        # seed 1件 → W_j=1.0、fan_j=1 → S_ji=s_max、activation ≈ s_max
        assert matched["activation"] == config.act_r.s_max

    def test_high_fan_reduces_sji_contribution(self, tmp_project: Path) -> None:
        char_id = _make_character_phase3(tmp_project)
        config = _make_config(char_id)
        dim = 768
        entity_vec = np.zeros(dim, dtype=np.float32)
        entity_vec[0] = 1.0

        embedder = MagicMock()
        embedder.encode_document.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_query.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_topic.return_value = entity_vec

        _, ep_id = _insert_assoc_pair(
            tmp_project, char_id,
            entity_name="料理",
            episode_content="カレーを作った",
            entity_vec=entity_vec,
            extra_dst_entities=4,  # fan_out = 1(episode) + 4 = 5
        )
        results = memory_search(
            "料理の話",
            character_id=char_id, config=config, embedder=embedder,
        )
        matched = next(r for r in results if r["episode_id"] == ep_id)
        # S_ji = s_max - ln(5) ≈ 0.391
        expected = config.act_r.s_max - float(np.log(5))
        assert abs(matched["activation"] - expected) < 1e-6

    def test_direct_wins_when_both_hit(self, tmp_project: Path) -> None:
        """vdb_memories にも載せると direct でもヒット → source='direct'。"""
        char_id = _make_character_phase3(tmp_project)
        config = _make_config(char_id)
        dim = 768
        entity_vec = np.zeros(dim, dtype=np.float32)
        entity_vec[0] = 1.0

        embedder = MagicMock()
        embedder.encode_document.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_query.return_value = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        embedder.encode_topic.return_value = entity_vec

        # memory_write で episode と embedding を正規経路で登録
        r = memory_write(
            "カレーを作った", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        ep_id = int(r["episode_id"])

        # 同じ episode を関連 entity から辿れるよう relation を張る
        kv_path = tmp_project / "data" / char_id / "kv.sqlite"
        conn = sqlite3.connect(str(kv_path))
        try:
            cur = conn.execute(
                """
                INSERT INTO entities (canonical_name, entity_type, valid_from)
                VALUES ('料理', 'concept', '2026-01-01T00:00:00+00:00')
                """
            )
            entity_id = int(cur.lastrowid)  # type: ignore[arg-type]
            conn.execute(
                """
                INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate,
                                       valid_from)
                VALUES ('entity', ?, 'episode', ?, 'evidences',
                        '2026-01-01T00:00:00+00:00')
                """,
                (entity_id, ep_id),
            )
            conn.commit()
        finally:
            conn.close()

        vdb_conn = _open_vec(tmp_project / "data" / char_id / "vdb_entities.db")
        try:
            upsert_entity_vector(vdb_conn, entity_id, entity_vec)
        finally:
            vdb_conn.close()

        results = memory_search(
            "料理の話",
            character_id=char_id, config=config, embedder=embedder,
        )
        matched = next(r for r in results if r["episode_id"] == ep_id)
        assert matched["source"] == "direct"
