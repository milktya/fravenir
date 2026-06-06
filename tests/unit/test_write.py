"""Unit tests for core/write.py."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from fravenir.core.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionError,
    ExtractionResult,
)
from fravenir.core.write import memory_write
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb(data_dir / "vdb_memories.db")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    sqlite_init.init_vdb_relations(data_dir / "vdb_relations.db")
    return char_id


def _make_config(char_id: str = "test_char") -> AppConfig:
    return AppConfig(character=CharacterConfig(id=char_id))


def _make_embedder(dim: int = 768) -> MagicMock:
    embedder = MagicMock()
    unit = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    embedder.encode_document.return_value = unit
    embedder.encode_topic.return_value = unit
    return embedder


class TestMemoryWrite:
    def test_returns_episode_id(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        result = memory_write(
            "テスト記憶",
            "facts",
            1,
            None,
            character_id=char_id,
            config=_make_config(char_id),
            embedder=_make_embedder(),
        )
        assert isinstance(result["episode_id"], int)
        assert result["episode_id"] >= 1
        assert result["stage"] == "embedded"

    def test_episode_stored_in_kv(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        result = memory_write(
            "覚えておくべきこと",
            "state",
            2,
            "session-abc",
            character_id=char_id,
            config=_make_config(char_id),
            embedder=_make_embedder(),
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        row = kv.execute(
            "SELECT content, kind, importance, session_id FROM episodes WHERE id = ?",
            (result["episode_id"],),
        ).fetchone()
        assert row == ("覚えておくべきこと", "state", 2, "session-abc")

    def test_doc_status_embedded(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        result = memory_write(
            "感情記録",
            "emo",
            3,
            None,
            character_id=char_id,
            config=_make_config(char_id),
            embedder=_make_embedder(),
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        row = kv.execute(
            "SELECT stage FROM doc_status WHERE episode_id = ?",
            (result["episode_id"],),
        ).fetchone()
        assert row is not None
        assert row[0] == "embedded"

    def test_vector_stored_in_vdb(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        result = memory_write(
            "ベクトル記憶",
            "facts",
            1,
            None,
            character_id=char_id,
            config=_make_config(char_id),
            embedder=_make_embedder(),
        )
        import sqlite_vec

        vdb_path = str(tmp_project / "data" / char_id / "vdb_memories.db")
        vdb = sqlite3.connect(vdb_path)
        vdb.enable_load_extension(True)
        sqlite_vec.load(vdb)
        vdb.enable_load_extension(False)
        rows = vdb.execute(
            "SELECT episode_id FROM vdb_memories WHERE episode_id = ?",
            (result["episode_id"],),
        ).fetchall()
        assert len(rows) == 1

    def test_embedder_called_with_content(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        embedder = _make_embedder()
        memory_write(
            "呼ばれた内容",
            "facts",
            1,
            None,
            character_id=char_id,
            config=_make_config(char_id),
            embedder=embedder,
        )
        embedder.encode_document.assert_called_once_with("呼ばれた内容")

    def test_multiple_writes_get_distinct_ids(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        config = _make_config(char_id)
        embedder = _make_embedder()
        r1 = memory_write(
            "記憶1", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        r2 = memory_write(
            "記憶2", "facts", 1, None,
            character_id=char_id, config=config, embedder=embedder,
        )
        assert r1["episode_id"] != r2["episode_id"]


class TestMemoryWriteValidation:
    def test_empty_content_raises(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        with pytest.raises(ValueError, match="content"):
            memory_write(
                "   ", "facts", 1, None,
                character_id=char_id, config=_make_config(char_id),
                embedder=_make_embedder(),
            )

    def test_invalid_kind_raises(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        with pytest.raises(ValueError, match="kind"):
            memory_write(
                "内容", "invalid", 1, None,  # type: ignore[arg-type]
                character_id=char_id, config=_make_config(char_id),
                embedder=_make_embedder(),
            )

    def test_importance_out_of_range_raises(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        with pytest.raises(ValueError, match="importance"):
            memory_write(
                "内容", "facts", 5, None,
                character_id=char_id, config=_make_config(char_id),
                embedder=_make_embedder(),
            )


def _mock_extraction_client(result: ExtractionResult | None = None) -> MagicMock:
    client = MagicMock()
    client.extract.return_value = result if result is not None else ExtractionResult(
        entities=[
            ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
            ExtractedEntity(canonical_name="メモリツール", entity_type="work"),
        ],
        relations=[
            ExtractedRelation(src="みるちゃ", dst="メモリツール", predicate="creates"),
        ],
    )
    return client


class TestMemoryWriteExtraction:
    def test_none_client_behaves_as_phase1(self, tmp_project: Path) -> None:
        """extraction_client=None なら従来通り stage='embedded' で止まる。"""
        char_id = _make_character(tmp_project)
        result = memory_write(
            "そのまま", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(),
        )
        assert result["stage"] == "embedded"

    def test_with_client_reaches_done(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        result = memory_write(
            "みるちゃはメモリツールを作ってる", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        assert result["stage"] == "done"
        client.extract.assert_called_once_with("みるちゃはメモリツールを作ってる")

    def test_cache_json_is_written(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        result = memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        cache_file = (
            tmp_project / "data" / char_id / "cache" / "llm_extractions"
            / f"{result['episode_id']}.json"
        )
        assert cache_file.exists()
        loaded = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "entities" in loaded
        assert "relations" in loaded
        assert len(loaded["entities"]) == 2

    def test_entities_inserted(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        rows = kv.execute(
            "SELECT canonical_name, entity_type FROM entities WHERE valid_to IS NULL "
            "ORDER BY id"
        ).fetchall()
        assert ("みるちゃ", "person") in rows
        assert ("メモリツール", "work") in rows

    def test_reuses_existing_entity_by_canonical_name(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        kv_path = str(tmp_project / "data" / char_id / "kv.sqlite")
        kv = sqlite3.connect(kv_path)
        kv.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, is_self, self_weight,
                                  decay_rate, valid_from)
            VALUES ('みるちゃ', 'person', 0, 0.0, 0.5, '2026-01-01T00:00:00+00:00')
            """
        )
        kv.commit()
        existing_id = kv.execute(
            "SELECT id FROM entities WHERE canonical_name='みるちゃ'"
        ).fetchone()[0]
        kv.close()

        client = _mock_extraction_client()
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )

        kv = sqlite3.connect(kv_path)
        rows = kv.execute(
            "SELECT id FROM entities WHERE canonical_name='みるちゃ' AND valid_to IS NULL"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == existing_id

    def test_reuses_existing_entity_by_alias(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        kv_path = str(tmp_project / "data" / char_id / "kv.sqlite")
        kv = sqlite3.connect(kv_path)
        kv.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, is_self, self_weight,
                                  decay_rate, valid_from)
            VALUES ('milktya', 'person', 0, 0.0, 0.5, '2026-01-01T00:00:00+00:00')
            """
        )
        eid = kv.execute("SELECT last_insert_rowid()").fetchone()[0]
        kv.execute(
            "INSERT INTO entity_aliases (alias, entity_id) VALUES ('みるちゃ', ?)", (eid,)
        )
        kv.commit()
        kv.close()

        client = _mock_extraction_client()
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )

        kv = sqlite3.connect(kv_path)
        rows = kv.execute(
            "SELECT id FROM entities WHERE canonical_name='みるちゃ'"
        ).fetchall()
        # 新規作成されず、既存エンティティが再利用される
        assert len(rows) == 0

    def test_mentions_relation_inserted(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        result = memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        rows = kv.execute(
            """
            SELECT src_type, src_id, dst_type, predicate FROM relations
            WHERE src_type='episode' AND src_id=? AND predicate='mentions'
            """,
            (result["episode_id"],),
        ).fetchall()
        assert len(rows) == 2  # 2 entities → 2 mentions

    def test_entity_to_entity_relation_inserted(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        rows = kv.execute(
            """
            SELECT predicate FROM relations
            WHERE src_type='entity' AND dst_type='entity' AND predicate='creates'
            """
        ).fetchall()
        assert len(rows) == 1

    def test_skips_relation_with_unresolved_endpoint(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client(
            ExtractionResult(
                entities=[ExtractedEntity(canonical_name="A", entity_type="person")],
                relations=[
                    ExtractedRelation(src="A", dst="存在しない", predicate="likes"),
                    ExtractedRelation(src="幽霊", dst="A", predicate="follows"),
                ],
            )
        )
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        rows = kv.execute(
            "SELECT COUNT(*) FROM relations WHERE src_type='entity' AND dst_type='entity'"
        ).fetchone()
        assert rows[0] == 0

    def test_skips_self_loop_relation(self, tmp_project: Path) -> None:
        """src と dst が同じ entity に解決される relation は弾かれる。"""
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client(
            ExtractionResult(
                entities=[
                    ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                    ExtractedEntity(canonical_name="ジャーナル", entity_type="work"),
                ],
                relations=[
                    # 自己ループ: LLM が「みるちゃがレビューを行う」を src=dst で出すパターン
                    ExtractedRelation(
                        src="みるちゃ", dst="みるちゃ", predicate="performs",
                        description="みるちゃがレビューを行う",
                    ),
                    # 正常な relation: こちらは保持される
                    ExtractedRelation(
                        src="みるちゃ", dst="ジャーナル", predicate="creates",
                    ),
                ],
            )
        )
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            self_loop_count = kv.execute(
                "SELECT COUNT(*) FROM relations"
                " WHERE src_type='entity' AND dst_type='entity' AND src_id = dst_id"
            ).fetchone()[0]
            normal_count = kv.execute(
                "SELECT COUNT(*) FROM relations"
                " WHERE src_type='entity' AND dst_type='entity' AND src_id != dst_id"
            ).fetchone()[0]
        finally:
            kv.close()
        assert self_loop_count == 0
        assert normal_count == 1

    def test_extraction_error_keeps_episode_at_embedded(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        client = MagicMock()
        client.extract.side_effect = ExtractionError("LLM unreachable")
        result = memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        assert result["stage"] == "embedded"

        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        row = kv.execute(
            "SELECT stage, error FROM doc_status WHERE episode_id = ?",
            (result["episode_id"],),
        ).fetchone()
        assert row[0] == "embedded"
        assert "LLM unreachable" in row[1]

        # episode は残る
        ep = kv.execute(
            "SELECT content FROM episodes WHERE id = ?", (result["episode_id"],)
        ).fetchone()
        assert ep[0] == "content"

        # entities/relations は一切書かれない
        ent_count = kv.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rel_count = kv.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        assert ent_count == 0
        assert rel_count == 0

    def test_stage_progression_observable_via_doc_status(
        self, tmp_project: Path
    ) -> None:
        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        result = memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        row = kv.execute(
            "SELECT stage FROM doc_status WHERE episode_id = ?",
            (result["episode_id"],),
        ).fetchone()
        assert row[0] == "done"


class TestMemoryWriteVdbVectors:
    def test_new_entities_written_to_vdb_entities(self, tmp_project: Path) -> None:
        import sqlite_vec

        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )

        vdb = sqlite3.connect(str(tmp_project / "data" / char_id / "vdb_entities.db"))
        vdb.enable_load_extension(True)
        sqlite_vec.load(vdb)
        vdb.enable_load_extension(False)
        rows = vdb.execute("SELECT entity_id FROM vdb_entities").fetchall()
        assert len(rows) == 2  # 2 new entities

    def test_new_entity_relation_written_to_vdb_relations(
        self, tmp_project: Path
    ) -> None:
        import sqlite_vec

        char_id = _make_character(tmp_project)
        client = _mock_extraction_client()
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=client,
        )

        vdb = sqlite3.connect(str(tmp_project / "data" / char_id / "vdb_relations.db"))
        vdb.enable_load_extension(True)
        sqlite_vec.load(vdb)
        vdb.enable_load_extension(False)
        rows = vdb.execute("SELECT relation_id FROM vdb_relations").fetchall()
        assert len(rows) == 1  # creates relation のみ (mentions は vdb 不要)

    def test_reused_entity_not_re_embedded(self, tmp_project: Path) -> None:
        """alias match で再利用した既存 entity は vdb_entities に再投入されない."""
        import sqlite_vec

        char_id = _make_character(tmp_project)
        kv_path = str(tmp_project / "data" / char_id / "kv.sqlite")
        # 既存 entity を DB に直書き（vdb_entities には何も入れない）
        kv = sqlite3.connect(kv_path)
        kv.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, is_self, self_weight,
                                  decay_rate, valid_from)
            VALUES ('みるちゃ', 'person', 0, 0.0, 0.5, '2026-01-01T00:00:00+00:00')
            """
        )
        kv.commit()
        kv.close()

        embedder = _make_embedder()
        client = _mock_extraction_client()
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=embedder, extraction_client=client,
        )

        vdb = sqlite3.connect(str(tmp_project / "data" / char_id / "vdb_entities.db"))
        vdb.enable_load_extension(True)
        sqlite_vec.load(vdb)
        vdb.enable_load_extension(False)
        rows = vdb.execute("SELECT entity_id FROM vdb_entities").fetchall()
        # みるちゃは既存再利用なので vdb に入らず、新規の メモリツール のみ入る
        assert len(rows) == 1

    def test_vdb_not_touched_when_extraction_disabled(
        self, tmp_project: Path
    ) -> None:
        import sqlite_vec

        char_id = _make_character(tmp_project)
        memory_write(
            "content", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(),
        )
        vdb_e = sqlite3.connect(str(tmp_project / "data" / char_id / "vdb_entities.db"))
        vdb_e.enable_load_extension(True)
        sqlite_vec.load(vdb_e)
        vdb_e.enable_load_extension(False)
        rows = vdb_e.execute("SELECT entity_id FROM vdb_entities").fetchall()
        assert rows == []
