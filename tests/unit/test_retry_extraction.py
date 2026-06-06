"""Unit tests for core/retry_extraction.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from fravenir.core.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionError,
    ExtractionResult,
)
from fravenir.core.retry_extraction import (
    list_failed_episodes,
    retry_extraction,
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


def _success_client() -> MagicMock:
    client = MagicMock()
    client.extract.return_value = ExtractionResult(
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
    return client


def _failing_client(message: str = "extraction failed: test") -> MagicMock:
    client = MagicMock()
    client.extract.side_effect = ExtractionError(message)
    return client


def _write_failed_episode(
    tmp_project: Path,
    char_id: str,
    content: str = "失敗するはずの記憶",
    error_message: str = "extraction failed: test",
) -> int:
    """memory_write を ExtractionError で失敗させ、embedded+error 状態の episode を作る."""
    result = memory_write(
        content,
        "facts",
        1,
        None,
        character_id=char_id,
        config=_make_config(char_id),
        embedder=_make_embedder(),
        extraction_client=_failing_client(error_message),
    )
    assert result["stage"] == "embedded"
    return int(result["episode_id"])


class TestListFailedEpisodes:
    def test_returns_only_embedded_with_error(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)

        # 1件目: 失敗（embedded + error）
        failed_id = _write_failed_episode(tmp_project, char_id, "失敗A")

        # 2件目: 成功（done）
        memory_write(
            "成功するはず", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=_success_client(),
        )

        # 3件目: extraction なし（embedded だが error なし）
        memory_write(
            "抽出未実行", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(),
        )

        targets = list_failed_episodes(character_id=char_id)
        assert len(targets) == 1
        assert targets[0].episode_id == failed_id
        assert targets[0].content == "失敗A"
        assert "extraction failed" in targets[0].error

    def test_episode_ids_filter_ignores_stage(self, tmp_project: Path) -> None:
        """episode_ids 指定時は stage / error の条件を無視して返す."""
        char_id = _make_character(tmp_project)
        result = memory_write(
            "成功記憶", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=_success_client(),
        )
        success_id = int(result["episode_id"])

        targets = list_failed_episodes(
            character_id=char_id, episode_ids=[success_id]
        )
        assert len(targets) == 1
        assert targets[0].episode_id == success_id

    def test_limit_caps_results(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        for i in range(3):
            _write_failed_episode(tmp_project, char_id, f"失敗{i}")

        targets = list_failed_episodes(character_id=char_id, limit=2)
        assert len(targets) == 2

    def test_empty_when_no_failures(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        targets = list_failed_episodes(character_id=char_id)
        assert targets == []

    def test_include_pending_picks_orphan_episodes(
        self, tmp_project: Path
    ) -> None:
        """doc_status 行が無い episode を include_pending=True で拾う。"""
        char_id = _make_character(tmp_project)

        # init-character ライクに直接 INSERT (doc_status を作らない)
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            kv.execute(
                "INSERT INTO episodes (content, kind, importance, valid_from)"
                " VALUES ('orphan ep', 'facts', 2, '2026-05-02T00:00:00+00:00')"
            )
            kv.commit()
            orphan_id = kv.execute("SELECT last_insert_rowid()").fetchone()[0]
        finally:
            kv.close()

        # 既存の Failed (embedded+error) も1件追加
        failed_id = _write_failed_episode(tmp_project, char_id, "failed ep")

        # include_pending=False では Failed のみ
        only_failed = list_failed_episodes(character_id=char_id)
        assert {t.episode_id for t in only_failed} == {failed_id}

        # include_pending=True では orphan も拾う
        with_pending = list_failed_episodes(
            character_id=char_id, include_pending=True
        )
        assert {t.episode_id for t in with_pending} == {failed_id, orphan_id}

    def test_include_pending_excludes_done(self, tmp_project: Path) -> None:
        """include_pending=True でも stage='done' の episode は対象外。"""
        char_id = _make_character(tmp_project)
        memory_write(
            "成功エピソード", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(), extraction_client=_success_client(),
        )
        targets = list_failed_episodes(
            character_id=char_id, include_pending=True
        )
        assert targets == []

    def test_include_pending_picks_embedded_without_error(
        self, tmp_project: Path
    ) -> None:
        """extraction_client なしで投入された stage='embedded' / error=NULL も拾う。"""
        char_id = _make_character(tmp_project)
        result = memory_write(
            "extraction なしエピソード", "facts", 1, None,
            character_id=char_id, config=_make_config(char_id),
            embedder=_make_embedder(),
        )
        ep_id = int(result["episode_id"])

        only_failed = list_failed_episodes(character_id=char_id)
        assert only_failed == []

        with_pending = list_failed_episodes(
            character_id=char_id, include_pending=True
        )
        assert {t.episode_id for t in with_pending} == {ep_id}


class TestRetryExtraction:
    def test_success_updates_stage_to_done(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        failed_id = _write_failed_episode(tmp_project, char_id)

        targets = list_failed_episodes(character_id=char_id)
        assert len(targets) == 1

        result = retry_extraction(
            targets,
            character_id=char_id,
            extraction_client=_success_client(),
            embedder=_make_embedder(),
        )
        assert result.succeeded == [failed_id]
        assert result.failed == []

        # doc_status が done に更新されている
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            row = kv.execute(
                "SELECT stage, error FROM doc_status WHERE episode_id = ?",
                (failed_id,),
            ).fetchone()
        finally:
            kv.close()
        assert row[0] == "done"

    def test_re_failure_keeps_embedded_and_records_error(
        self, tmp_project: Path
    ) -> None:
        char_id = _make_character(tmp_project)
        failed_id = _write_failed_episode(
            tmp_project, char_id, error_message="initial error"
        )

        targets = list_failed_episodes(character_id=char_id)
        result = retry_extraction(
            targets,
            character_id=char_id,
            extraction_client=_failing_client("retry error"),
            embedder=_make_embedder(),
        )
        assert result.succeeded == []
        assert len(result.failed) == 1
        eid, err = result.failed[0]
        assert eid == failed_id
        assert "retry error" in err

        # doc_status は embedded のまま、error が更新されている
        kv = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            row = kv.execute(
                "SELECT stage, error FROM doc_status WHERE episode_id = ?",
                (failed_id,),
            ).fetchone()
        finally:
            kv.close()
        assert row[0] == "embedded"
        assert "retry error" in row[1]

    def test_empty_targets_returns_empty_result(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        result = retry_extraction(
            [],
            character_id=char_id,
            extraction_client=_success_client(),
            embedder=_make_embedder(),
        )
        assert result.attempted == []
        assert result.succeeded == []
        assert result.failed == []

    def test_orphan_episode_gets_doc_status_and_reaches_done(
        self, tmp_project: Path
    ) -> None:
        """doc_status 無し episode に retry を走らせると doc_status が作られて done になる。"""
        char_id = _make_character(tmp_project)

        # doc_status 無しの orphan episode を直接 INSERT
        kv_path = tmp_project / "data" / char_id / "kv.sqlite"
        kv = sqlite3.connect(str(kv_path))
        try:
            kv.execute(
                "INSERT INTO episodes (content, kind, importance, valid_from)"
                " VALUES ('orphan to be saved', 'facts', 2, '2026-05-02T00:00:00+00:00')"
            )
            kv.commit()
            orphan_id = kv.execute("SELECT last_insert_rowid()").fetchone()[0]
        finally:
            kv.close()

        targets = list_failed_episodes(
            character_id=char_id, include_pending=True
        )
        assert {t.episode_id for t in targets} == {orphan_id}

        result = retry_extraction(
            targets,
            character_id=char_id,
            extraction_client=_success_client(),
            embedder=_make_embedder(),
        )
        assert result.succeeded == [orphan_id]

        # doc_status 行ができて stage='done'
        kv = sqlite3.connect(str(kv_path))
        try:
            row = kv.execute(
                "SELECT stage FROM doc_status WHERE episode_id = ?",
                (orphan_id,),
            ).fetchone()
        finally:
            kv.close()
        assert row is not None
        assert row[0] == "done"
