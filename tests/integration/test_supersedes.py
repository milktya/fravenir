"""Integration tests for supersedes during memory_write pipeline.

Mocks the LLM endpoint but runs the full SQLite / write / trace / search machinery.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from fravenir.core.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from fravenir.core.search import memory_search
from fravenir.core.trace import memory_trace
from fravenir.core.write import memory_write
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init


def _make_character(tmp_project: Path, char_id: str = "test_sup_integ") -> str:
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
    e.encode_query.return_value = unit
    e.encode_topic.return_value = unit
    return e


def _client(result: ExtractionResult) -> MagicMock:
    c = MagicMock()
    c.extract.return_value = result
    return c


def _works_as_extraction(src: str, dst: str) -> ExtractionResult:
    return ExtractionResult(
        entities=[
            ExtractedEntity(canonical_name=src, entity_type="person"),
            ExtractedEntity(canonical_name=dst, entity_type="work"),
        ],
        relations=[
            ExtractedRelation(
                src=src, dst=dst, predicate="works_as",
                description=f"{src} works as {dst}",
            ),
        ],
    )


def test_write_works_as_supersedes_chain(tmp_project: Path) -> None:
    """3 回連続で works_as を書き換え → trace で 3 段チェーン、search で最新のみ / 全件。"""
    char_id = _make_character(tmp_project)
    config = AppConfig(character=CharacterConfig(id=char_id))

    # 1回目: プログラマ
    r1 = memory_write(
        "みるちゃの仕事はプログラマ", "facts", 1, None,
        character_id=char_id, config=config,
        embedder=_embedder(),
        extraction_client=_client(
            _works_as_extraction("みるちゃ", "プログラマ")
        ),
    )
    assert r1["stage"] == "done"
    ep1_id = r1["episode_id"]

    # 2回目: デザイナ
    r2 = memory_write(
        "みるちゃの仕事はデザイナ", "facts", 1, None,
        character_id=char_id, config=config,
        embedder=_embedder(),
        extraction_client=_client(
            _works_as_extraction("みるちゃ", "デザイナ")
        ),
    )
    assert r2["stage"] == "done"
    ep2_id = r2["episode_id"]

    # 3回目: ライター
    r3 = memory_write(
        "みるちゃの仕事はライター", "facts", 1, None,
        character_id=char_id, config=config,
        embedder=_embedder(),
        extraction_client=_client(
            _works_as_extraction("みるちゃ", "ライター")
        ),
    )
    assert r3["stage"] == "done"
    ep3_id = r3["episode_id"]

    # --- trace で 3 段チェーン ---
    trace_result = memory_trace(ep3_id, character_id=char_id, config=config)
    chain = trace_result["chain"]
    assert len(chain) == 3  # type: ignore[arg-type]
    contents = [c["content"] for c in chain]  # type: ignore[union-attr]
    assert "ライター" in contents[0]
    assert "デザイナ" in contents[1]
    assert "プログラマ" in contents[2]
    # 最新だけ valid_to=NULL
    assert chain[0]["valid_to"] is None  # type: ignore[index]
    assert chain[1]["valid_to"] is not None  # type: ignore[index]
    assert chain[2]["valid_to"] is not None  # type: ignore[index]

    # --- search include_archived=False → 最新だけ ---
    results_active = memory_search(
        "みるちゃ 仕事",
        include_archived=False,
        limit=10,
        character_id=char_id,
        config=config,
        embedder=_embedder(),
    )
    active_ids = {r["episode_id"] for r in results_active}  # type: ignore[index]
    assert ep3_id in active_ids
    assert ep2_id not in active_ids
    assert ep1_id not in active_ids

    # --- search include_archived=True → 3 件全部 ---
    results_all = memory_search(
        "みるちゃ 仕事",
        include_archived=True,
        limit=10,
        character_id=char_id,
        config=config,
        embedder=_embedder(),
    )
    all_ids = {r["episode_id"] for r in results_all}  # type: ignore[index]
    assert ep3_id in all_ids
    assert ep2_id in all_ids
    assert ep1_id in all_ids
