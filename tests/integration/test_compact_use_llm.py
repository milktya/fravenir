"""Integration test for LLM semantic judgment in compact pipeline."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sqlite_vec

from fravenir.core.compact import run_compact
from fravenir.core.semantic_judge import (
    JudgeError,
    _Judgment,
    judge_merge_candidates,
)
from fravenir.schemas.config import AppConfig, CharacterConfig, SemanticJudgeConfig
from fravenir.storage import sqlite_init
from fravenir.storage.paths import kv_db_path
from fravenir.storage.vector import upsert_entity_vector


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    return char_id


def _insert_entity_with_desc(
    tmp_project: Path,
    char_id: str,
    name: str,
    entity_type: str = "concept",
    description: str = "",
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    now = datetime.now(UTC).isoformat()
    try:
        cur = conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, description, valid_from)"
            " VALUES (?, ?, ?, ?)",
            (name, entity_type, description, now),
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
    )
    return cfg


class _FakeJudgeClient:
    def __init__(self, judgments: list[_Judgment | JudgeError]) -> None:
        self._queue = list(judgments)

    def judge(self, *args: object, **kwargs: object) -> _Judgment:
        item = self._queue.pop(0)
        if isinstance(item, JudgeError):
            raise item
        return item


class TestCompactUseLlm:
    def test_compact_use_llm_full_flow(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project, "judge_test")

        rng = np.random.default_rng(42)

        # Use 4 distinct vectors so only intended pairs match
        base_vectors = [rng.normal(size=768).astype(np.float32) for _ in range(4)]

        # Each pair differs by 1 char so levenshtein < 3
        pairs = [
            ("猫A", "猫B", "猫"),       # same/high → auto-resolve
            ("本A", "本B", "本"),       # different/high → auto-reject
            ("車A", "車B", "乗り物"),   # same/medium → queued_for_review
            ("空A", "空B", "概念"),     # unsure/low → deferred
        ]

        eids: list[tuple[int, int]] = []
        for i, (a_name, b_name, _) in enumerate(pairs):
            a_id = _insert_entity_with_desc(
                tmp_project, char_id, a_name, "concept", f"{a_name}の説明",
            )
            b_id = _insert_entity_with_desc(
                tmp_project, char_id, b_name, "concept", f"{b_name}の説明",
            )
            # Both entities in a pair share the same base, slightly perturbed
            v_a = base_vectors[i] + 0.005 * rng.normal(size=768).astype(np.float32)
            v_b = base_vectors[i] + 0.005 * rng.normal(size=768).astype(np.float32)
            _put_entity_vector(tmp_project, char_id, a_id, v_a)
            _put_entity_vector(tmp_project, char_id, b_id, v_b)
            eids.append((a_id, b_id))

        # Give them relations so auto-resolve has something to rewire
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        now = datetime.now(UTC).isoformat()
        try:
            for a_id, b_id in eids:
                conn.execute(
                    "INSERT INTO relations "
                    "(src_type, src_id, dst_type, dst_id, predicate, valid_from)"
                    " VALUES ('entity', ?, 'entity', ?, 'likes', ?)",
                    (a_id, b_id, now),
                )
            conn.commit()
        finally:
            conn.close()

        # Run compact without LLM to populate merge_candidates
        result = run_compact(
            character_id=char_id,
            config=_make_config(char_id, judge_enabled=False),
            use_llm=False,
        )
        assert result.merge_candidates == 4

        # Now run LLM judgment directly with fake client
        db_path = kv_db_path(char_id)
        fake = _FakeJudgeClient([
            _Judgment(label="same", confidence="high", reason="猫の表記揺れ"),
            _Judgment(label="different", confidence="high", reason="本と鉛筆は別物"),
            _Judgment(label="same", confidence="medium", reason="確信はない"),
            _Judgment(label="unsure", confidence="low", reason="空と宙は異なる可能性"),
        ])

        jresult = judge_merge_candidates(
            db_path=db_path,
            config=_make_config(char_id, judge_enabled=True).semantic_judge,
            judge_client=fake,
        )

        assert jresult.auto_resolved == 1
        assert jresult.auto_rejected == 1
        assert jresult.queued_for_review == 1
        assert jresult.deferred == 1

        # Verify DB state
        conn2 = sqlite3.connect(str(db_path))
        try:
            rows = conn2.execute(
                "SELECT id, resolved, judge_label, judge_confidence, judge_attempts "
                "FROM merge_candidates ORDER BY id"
            ).fetchall()

            # Candidate 1: auto-resolved
            assert rows[0][1] == 1
            assert rows[0][2] == "same"
            assert rows[0][3] == "high"
            assert rows[0][4] == 1

            # Candidate 2: auto-rejected
            assert rows[1][1] == 2
            assert rows[1][2] == "different"
            assert rows[1][3] == "high"
            assert rows[1][4] == 1

            # Candidate 3: queued for review
            assert rows[2][1] == 0
            assert rows[2][2] == "same"
            assert rows[2][3] == "medium"
            assert rows[2][4] == 1

            # Candidate 4: deferred
            assert rows[3][1] == 0
            assert rows[3][2] == "unsure"
            assert rows[3][3] == "low"
            assert rows[3][4] == 1

            # Check candidate 1's drop entity got valid_to + supersedes
            drop = conn2.execute(
                "SELECT valid_to, supersedes FROM entities WHERE id = ?",
                (eids[0][1],),
            ).fetchone()
            assert drop[0] is not None
            assert drop[1] == eids[0][0]
        finally:
            conn2.close()
