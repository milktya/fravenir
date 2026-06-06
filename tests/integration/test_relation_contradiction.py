"""Integration test for relation contradiction detection in compact pipeline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sqlite_vec

from fravenir.core.compact import run_compact
from fravenir.core.semantic_judge import JudgeClient, _ContradictionJudgment
from fravenir.schemas.config import AppConfig, CharacterConfig, SemanticJudgeConfig
from fravenir.storage import sqlite_init
from fravenir.storage.vector import upsert_entity_vector


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    return char_id


def _insert_entity(
    tmp_project: Path,
    char_id: str,
    name: str,
    entity_type: str = "person",
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    now = datetime.now(UTC).isoformat()
    try:
        cur = conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from)"
            " VALUES (?, ?, ?)",
            (name, entity_type, now),
        )
        conn.commit()
        eid: int = cur.lastrowid  # type: ignore[assignment]
        return eid
    finally:
        conn.close()


def _put_entity_vector(
    tmp_project: Path, char_id: str, entity_id: int, vec: np.ndarray
) -> None:
    db_path = tmp_project / "data" / char_id / "vdb_entities.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        norm = vec / (np.linalg.norm(vec) + 1e-12)
        upsert_entity_vector(conn, entity_id, norm.astype(np.float32))
    finally:
        conn.close()


def _make_config(char_id: str, judge_enabled: bool) -> AppConfig:
    cfg = AppConfig(character=CharacterConfig(id=char_id))
    cfg.semantic_judge = SemanticJudgeConfig(
        enabled=judge_enabled,
        base_url="http://127.0.0.1:8080/v1",
        model="test",
        api_key="dummy",
        timeout=60.0,
        max_retries=0,
        max_attempts=3,
        temperature=0.0,
        min_strength=0.3,
    )
    return cfg


class _FakeContradictionJudgeClient:
    def __init__(self, judgments: list[_ContradictionJudgment]) -> None:
        self._queue = list(judgments)

    def judge_contradiction(self, *args: object, **kwargs: object) -> _ContradictionJudgment:
        return self._queue.pop(0)


class TestCompactContradictionPass:
    def test_compact_use_llm_runs_contradiction_pass(
        self, tmp_project: Path, monkeypatch: object
    ) -> None:
        char_id = _make_character(tmp_project, "contr_test")
        rng = np.random.default_rng(42)

        # Insert two entities with vectors so compact Step 4 can run
        miru_id = _insert_entity(tmp_project, char_id, "miru", "person")
        cats_id = _insert_entity(tmp_project, char_id, "cats", "topic")
        vec_a = rng.normal(size=768).astype(np.float32)
        vec_b = vec_a + 0.005 * rng.normal(size=768).astype(np.float32)
        _put_entity_vector(tmp_project, char_id, miru_id, vec_a)
        _put_entity_vector(tmp_project, char_id, cats_id, vec_b)

        kv = tmp_project / "data" / char_id / "kv.sqlite"
        now = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(kv)
        try:
            # Episodes
            conn.executemany(
                "INSERT INTO episodes (id, content, kind, importance, valid_from)"
                " VALUES (?, ?, 'facts', 2, ?)",
                [
                    (10, "miru likes cats", now),
                    (20, "miru dislikes cats", now),
                ],
            )
            # Relations: contradiction pair (same src/dst, likes vs dislikes)
            conn.executemany(
                "INSERT INTO relations"
                " (id, src_type, src_id, dst_type, dst_id, predicate, strength, valid_from)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (100, "entity", miru_id, "entity", cats_id, "dislikes", 1.0, now),
                    (200, "entity", miru_id, "entity", cats_id, "likes", 1.0, now),
                    # Origin mentions
                    (101, "episode", 10, "entity", miru_id, "mentions", 1.0, now),
                    (102, "episode", 10, "entity", cats_id, "mentions", 1.0, now),
                    (201, "episode", 20, "entity", miru_id, "mentions", 1.0, now),
                    (202, "episode", 20, "entity", cats_id, "mentions", 1.0, now),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        # Monkeypatch JudgeClient so run_compact uses our fake
        fake = _FakeContradictionJudgeClient([
            _ContradictionJudgment(correct="A", confidence="high", reason="Aが正しい"),
        ])
        monkeypatch.setattr(
            JudgeClient, "__init__",
            lambda self, config: setattr(self, "_config", config)
            or setattr(self, "_client", None)
            or None,
        )
        monkeypatch.setattr(JudgeClient, "judge", lambda *a, **k: None)
        monkeypatch.setattr(JudgeClient, "judge_direction", lambda *a, **k: None)
        monkeypatch.setattr(JudgeClient, "judge_contradiction", fake.judge_contradiction)

        result = run_compact(
            character_id=char_id,
            config=_make_config(char_id, judge_enabled=True),
            use_llm=True,
            dry_run=True,
        )

        assert result.contradiction_judgment is not None
        assert result.contradiction_judgment.superseded_b == 1
        assert result.contradiction_judgment.judgments[0].a_predicate == "dislikes"
        assert result.contradiction_judgment.judgments[0].b_predicate == "likes"

        # dry_run: DB must remain unchanged
        conn2 = sqlite3.connect(kv)
        try:
            live_count = conn2.execute(
                "SELECT COUNT(*) FROM relations WHERE id IN (100, 200) AND valid_to IS NULL"
            ).fetchone()[0]
            assert live_count == 2
        finally:
            conn2.close()
