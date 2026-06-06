"""Integration tests for the memory_write → extraction pipeline.

Mocks the LLM endpoint but runs the full SQLite / cache / stage machinery.
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import sqlite_vec

from fravenir.core.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from fravenir.core.write import memory_write
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init


def _make_character(tmp_project: Path, char_id: str = "integ_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb(data_dir / "vdb_memories.db")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    sqlite_init.init_vdb_relations(data_dir / "vdb_relations.db")
    return char_id


def _embedder(dim: int = 768) -> MagicMock:
    e = MagicMock()
    unit = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    e.encode_document.return_value = unit
    e.encode_topic.return_value = unit
    return e


def _client(result: ExtractionResult) -> MagicMock:
    c = MagicMock()
    c.extract.return_value = result
    return c


def test_full_pipeline_produces_cache_entities_and_relations(
    tmp_project: Path,
) -> None:
    char_id = _make_character(tmp_project)
    config = AppConfig(character=CharacterConfig(id=char_id))
    client = _client(
        ExtractionResult(
            entities=[
                ExtractedEntity(
                    canonical_name="みるちゃ",
                    entity_type="person",
                    description="開発者",
                ),
                ExtractedEntity(
                    canonical_name="記憶基盤",
                    entity_type="work",
                    description="fravenir",
                ),
            ],
            relations=[
                ExtractedRelation(
                    src="みるちゃ",
                    dst="記憶基盤",
                    predicate="creates",
                    description="開発中",
                ),
            ],
        )
    )

    result = memory_write(
        "みるちゃは記憶基盤を開発してる",
        "facts",
        2,
        "sess-1",
        character_id=char_id,
        config=config,
        embedder=_embedder(),
        extraction_client=client,
    )

    assert result["stage"] == "done"
    episode_id = result["episode_id"]

    cache_file = (
        tmp_project / "data" / char_id / "cache" / "llm_extractions" / f"{episode_id}.json"
    )
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert len(payload["entities"]) == 2
    assert payload["relations"][0]["predicate"] == "creates"

    kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    entities = dict(
        kv.execute(
            "SELECT canonical_name, entity_type FROM entities WHERE valid_to IS NULL"
        ).fetchall()
    )
    assert entities == {"みるちゃ": "person", "記憶基盤": "work"}

    mentions = kv.execute(
        "SELECT COUNT(*) FROM relations WHERE predicate='mentions' AND src_id=?",
        (episode_id,),
    ).fetchone()[0]
    assert mentions == 2

    creates = kv.execute(
        "SELECT COUNT(*) FROM relations "
        "WHERE src_type='entity' AND dst_type='entity' AND predicate='creates'"
    ).fetchone()[0]
    assert creates == 1

    final_stage = kv.execute(
        "SELECT stage FROM doc_status WHERE episode_id = ?", (episode_id,)
    ).fetchone()[0]
    assert final_stage == "done"

    # vdb_entities and vdb_relations get populated as well
    vdb_e = sqlite3.connect(
        str(tmp_project / "data" / char_id / "vdb_entities.db")
    )
    vdb_e.enable_load_extension(True)
    sqlite_vec.load(vdb_e)
    vdb_e.enable_load_extension(False)
    assert vdb_e.execute("SELECT COUNT(*) FROM vdb_entities").fetchone()[0] == 2

    vdb_r = sqlite3.connect(
        str(tmp_project / "data" / char_id / "vdb_relations.db")
    )
    vdb_r.enable_load_extension(True)
    sqlite_vec.load(vdb_r)
    vdb_r.enable_load_extension(False)
    assert vdb_r.execute("SELECT COUNT(*) FROM vdb_relations").fetchone()[0] == 1


def test_sequential_writes_reuse_existing_entities(tmp_project: Path) -> None:
    char_id = _make_character(tmp_project)
    config = AppConfig(character=CharacterConfig(id=char_id))
    embedder = _embedder()

    # 1回目: みるちゃ + メモリツール を作る
    client1 = _client(
        ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="メモリツール", entity_type="work"),
            ],
            relations=[
                ExtractedRelation(
                    src="みるちゃ", dst="メモリツール", predicate="creates"
                ),
            ],
        )
    )
    memory_write(
        "1回目", "facts", 1, None,
        character_id=char_id, config=config, embedder=embedder,
        extraction_client=client1,
    )

    # 2回目: みるちゃ + ruri-v3(新規) を出す → みるちゃ は既存を再利用
    client2 = _client(
        ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="ruri-v3", entity_type="work"),
            ],
            relations=[
                ExtractedRelation(
                    src="みるちゃ", dst="ruri-v3", predicate="uses"
                ),
            ],
        )
    )
    memory_write(
        "2回目", "facts", 1, None,
        character_id=char_id, config=config, embedder=embedder,
        extraction_client=client2,
    )

    kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    # みるちゃ は1件のみ（再利用された）
    mirucha_count = kv.execute(
        "SELECT COUNT(*) FROM entities WHERE canonical_name='みるちゃ' AND valid_to IS NULL"
    ).fetchone()[0]
    assert mirucha_count == 1

    # 合計 entity 数は 3（みるちゃ / メモリツール / ruri-v3）
    total = kv.execute(
        "SELECT COUNT(*) FROM entities WHERE valid_to IS NULL"
    ).fetchone()[0]
    assert total == 3

    # entity→entity relations は 2つ（creates, uses）、src は同じ entity
    entity_rels = kv.execute(
        "SELECT predicate, src_id FROM relations "
        "WHERE src_type='entity' AND dst_type='entity' ORDER BY predicate"
    ).fetchall()
    assert [r[0] for r in entity_rels] == ["creates", "uses"]
    assert entity_rels[0][1] == entity_rels[1][1]  # みるちゃ の entity_id 同じ
