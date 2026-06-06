"""Golden tests: write → search で意図通りの順位になることを保証する。

Embedder をハッシュベースの決定論的実装で置き換え、実モデル不要で動作する。
各テキストに固有の方向ベクトルを割り当て、クエリとの類似度が制御可能。
"""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from fravenir.core.delete import memory_delete
from fravenir.core.search import memory_search
from fravenir.core.write import memory_write
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init
from fravenir.storage.paths import kv_db_path

DIM = 768


def _make_character(tmp_project: Path, char_id: str = "golden_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb(data_dir / "vdb_memories.db")
    return char_id


def _make_config(char_id: str = "golden_char") -> AppConfig:
    return AppConfig(character=CharacterConfig(id=char_id))


def _hash_vec(text: str, dim: int = DIM) -> "np.ndarray[tuple[int], np.dtype[np.float32]]":
    """テキストをハッシュから決定論的な単位ベクトルに変換する。

    同じテキストは同じベクトル、異なるテキストは異なるベクトルになる。
    クエリ==ドキュメントなら cosine_sim=1.0 となり最高スコアを保証できる。
    """
    rng = np.random.default_rng(hash(text) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v


def _make_embedder_for(query_text: str) -> MagicMock:
    """クエリと完全一致するテキストが最高スコアになるようなEmbedderモック。"""
    embedder = MagicMock()
    embedder.encode_document.side_effect = lambda t: _hash_vec(t)
    embedder.encode_query.return_value = _hash_vec(query_text)
    return embedder


class TestGoldenSearch:
    def test_golden_1_relevant_episode_ranks_first(self, tmp_project: Path) -> None:
        """[シナリオA] クエリと同一テキストのエピソードが1位に来る。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        query = "あたしはみるちゃのパートナーとして記憶を共有していく"

        embedder = _make_embedder_for(query)

        # クエリと完全一致する記憶 + 無関係な記憶2件
        target = memory_write(
            query, "facts", 3, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        memory_write(
            "今日の天気は晴れだった", "state", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        memory_write(
            "猫が好き", "emo", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )

        results = memory_search(
            query, limit=3, character_id=char_id, config=config, embedder=embedder,
        )

        assert len(results) >= 1
        assert results[0]["episode_id"] == target["episode_id"], (
            f"期待: episode_id={target['episode_id']} が1位, "
            f"実際: {results[0]['episode_id']}"
        )

    def test_golden_2_keyword_match_ranks_first(self, tmp_project: Path) -> None:
        """[シナリオB] キーワード一致した記憶が無関係な記憶より上位に来る。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)

        # 無関係な記憶を先に仕込む（固有ハッシュのembedderで書く）
        memory_write(
            "数学は面白い", "facts", 1, None,
            character_id=char_id, config=config,
            embedder=_make_embedder_for("数学は面白い"),
        )
        memory_write(
            "山の景色が綺麗", "facts", 1, None,
            character_id=char_id, config=config,
            embedder=_make_embedder_for("山の景色が綺麗"),
        )

        # クエリと同テキストを書き込んで1位になることを確認
        target = memory_write(
            "猫について", "facts", 1, None,
            character_id=char_id, config=config,
            embedder=_make_embedder_for("猫について"),
        )

        results = memory_search(
            "猫について", limit=5,
            character_id=char_id, config=config,
            embedder=_make_embedder_for("猫について"),
        )
        assert len(results) >= 1
        assert results[0]["episode_id"] == target["episode_id"]

    def test_golden_3_importance_breaks_tie(self, tmp_project: Path) -> None:
        """[シナリオC] ベクトル距離が同一の時、importance=3 が importance=1 より上位。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)

        # 同じベクトルを返すEmbedder（距離0 → cosine=1.0 で全件タイ）
        same_vec = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
        embedder = MagicMock()
        embedder.encode_document.return_value = same_vec
        embedder.encode_query.return_value = same_vec

        low = memory_write(
            "重要度低", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        high = memory_write(
            "重要度高", "facts", 3, None,
            character_id=char_id, config=config, embedder=embedder,
        )

        results = memory_search(
            "テスト", limit=5,
            character_id=char_id, config=config, embedder=embedder,
        )

        ids = [r["episode_id"] for r in results]
        assert high["episode_id"] in ids
        assert low["episode_id"] in ids
        high_rank = ids.index(high["episode_id"])
        low_rank = ids.index(low["episode_id"])
        assert high_rank < low_rank, (
            f"importance=3 の記憶({high['episode_id']})が "
            f"importance=1({low['episode_id']})より上位にあるべき"
        )

    def test_golden_4_archived_excluded_by_default(self, tmp_project: Path) -> None:
        """[シナリオD] valid_to が立った記憶は include_archived=False で除外される。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        query = "削除テスト対象"

        embedder = _make_embedder_for(query)

        alive = memory_write(
            "生きている記憶 " + query, "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        dead = memory_write(
            "これから消す記憶 " + query, "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        memory_delete(
            episode_id=int(dead["episode_id"]), reason="golden test",
            character_id=char_id, config=config,
        )

        hidden = memory_search(
            query, limit=10, character_id=char_id, config=config, embedder=embedder,
        )
        hidden_ids = [r["episode_id"] for r in hidden]
        assert dead["episode_id"] not in hidden_ids, "削除済みは既定で出てはいけない"
        assert alive["episode_id"] in hidden_ids

        visible = memory_search(
            query, limit=10, include_archived=True,
            character_id=char_id, config=config, embedder=embedder,
        )
        visible_ids = [r["episode_id"] for r in visible]
        assert dead["episode_id"] in visible_ids, (
            "include_archived=True なら削除済みも検索対象"
        )

    def test_golden_6_self_cue_boosts_personality_mention(self, tmp_project: Path) -> None:
        """[シナリオF/Phase2] 自己キュー付きクエリで personality を mention する記憶が上位。

        identity (mina, aliases=[あたし]) と personality (好奇心旺盛) を仕込み、
        "あたしって何が好きだっけ？" で personality 関連の記憶が self_boost で浮上する。
        """
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        query = "あたしって何が好きだっけ？"
        embedder = _make_embedder_for(query)

        # identity + personality をDBに直接仕込む（CLI 経由ではなく最小セットアップ）
        conn = sqlite3.connect(kv_db_path(char_id))
        try:
            conn.execute(
                """
                INSERT INTO entities
                    (canonical_name, entity_type, is_self, self_weight,
                     decay_rate, valid_from)
                VALUES ('mina', 'person', 1, 1.0, 0.2, '2026-01-01T00:00:00+00:00')
                """
            )
            conn.execute(
                "INSERT INTO entity_aliases (alias, entity_id) VALUES ('あたし', 1)"
            )
            conn.execute(
                """
                INSERT INTO entities
                    (canonical_name, entity_type, is_self, self_weight,
                     decay_rate, valid_from)
                VALUES ('好奇心旺盛', 'concept', 0, 0.8, 0.3,
                        '2026-01-01T00:00:00+00:00')
                """
            )
            conn.commit()
        finally:
            conn.close()

        # personalityを mention する記憶（self-boost 対象）
        target = memory_write(
            "あたしは好奇心旺盛で、裏で何が起きてるか覗き込むのが好き",
            "facts", 2, None,
            character_id=char_id, config=config,
            embedder=_make_embedder_for(
                "あたしは好奇心旺盛で、裏で何が起きてるか覗き込むのが好き"
            ),
        )
        # 自己 entity と target episode を結ぶ relation（graph traversal 用）
        conn = sqlite3.connect(kv_db_path(char_id))
        try:
            conn.execute(
                """
                INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from)
                VALUES ('entity', 1, 'episode', ?, 'evidences', '2026-01-01T00:00:00+00:00')
                """,
                (target["episode_id"],),
            )
            conn.commit()
        finally:
            conn.close()

        # 無関係な記憶（self-boost 非対象）
        memory_write(
            "今日の空は青かった", "state", 1, None,
            character_id=char_id, config=config,
            embedder=_make_embedder_for("今日の空は青かった"),
        )
        memory_write(
            "数学の本を読んだ", "facts", 1, None,
            character_id=char_id, config=config,
            embedder=_make_embedder_for("数学の本を読んだ"),
        )

        results = memory_search(
            query, limit=5,
            character_id=char_id, config=config, embedder=embedder,
        )

        assert len(results) >= 1
        assert results[0]["episode_id"] == target["episode_id"], (
            "self-cue クエリで personality を mention する記憶が1位に来るはず"
        )
        assert results[0]["source"] == "self_boost", (
            "self-boost が発動したことを source で判別できる"
        )

    def test_golden_5_supersedes_new_version_ranks(self, tmp_project: Path) -> None:
        """[シナリオE] supersedes で新版に差し替わった記憶は、新版だけがヒットする。"""
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        query = "みるちゃの好きな色は青"

        embedder = _make_embedder_for(query)

        # v1: 旧版（後で supersedes される）
        v1 = memory_write(
            "みるちゃの好きな色は赤", "facts", 2, None,
            character_id=char_id, config=config, embedder=_make_embedder_for(query),
        )

        # v2: 新版（supersedes=v1, v1 に valid_to を立てる）を直接 DB に書く。
        # memory_write は supersedes 引数を受け取らないので、PROV-O 版管理は
        # Phase1 では DB 直叩き or dedicated API 待ち。ゴールデンは後者想定。
        now = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(kv_db_path(char_id))
        try:
            conn.execute(
                "UPDATE episodes SET valid_to = ? WHERE id = ?", (now, v1["episode_id"]),
            )
            conn.commit()
        finally:
            conn.close()

        v2 = memory_write(
            query, "facts", 2, None,
            character_id=char_id, config=config, embedder=_make_embedder_for(query),
        )
        conn = sqlite3.connect(kv_db_path(char_id))
        try:
            conn.execute(
                "UPDATE episodes SET supersedes = ? WHERE id = ?",
                (v1["episode_id"], v2["episode_id"]),
            )
            conn.commit()
        finally:
            conn.close()

        results = memory_search(
            query, limit=5, character_id=char_id, config=config, embedder=embedder,
        )
        ids = [r["episode_id"] for r in results]
        assert v2["episode_id"] in ids, "新版(v2)は既定検索にヒットするべき"
        assert v1["episode_id"] not in ids, "旧版(v1)は valid_to 済みなので既定では除外"

        # supersedes フィールドが v1 を指している
        v2_row = next(r for r in results if r["episode_id"] == v2["episode_id"])
        assert v2_row["supersedes"] == v1["episode_id"]
