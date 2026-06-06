"""Unit tests for relation direction conflict detection (Phase 5 P5-5)."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from fravenir.core.semantic_judge import (
    JudgeError,
    _DirectionJudgment,
    _fetch_direction_pairs,
    judge_relation_directions,
)
from fravenir.schemas.config import SemanticJudgeConfig
from fravenir.storage.sqlite_init import init_kv


def _make_db_with_reverse_pair(
    tmp_path: Path, a_strength: float = 1.0, b_strength: float = 1.0
) -> Path:
    """Create test DB with one reverse pair (A hosts B + B hosts A)."""
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    now_a = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()
    now_b = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC).isoformat()
    conn = sqlite3.connect(db)
    try:
        # Entities
        conn.executemany(
            "INSERT INTO entities (id, canonical_name, entity_type, valid_from)"
            " VALUES (?, ?, ?, ?)",
            [
                (1, "serverA", "concept", now_a),
                (2, "serverB", "concept", now_a),
            ],
        )
        # Episodes (origin texts)
        conn.executemany(
            "INSERT INTO episodes (id, content, kind, importance, valid_from)"
            " VALUES (?, ?, 'facts', 2, ?)",
            [
                (10, "serverA hosts serverB", now_a),
                (20, "serverB hosts serverA", now_b),
            ],
        )
        # Relations
        conn.executemany(
            "INSERT INTO relations"
            " (id, src_type, src_id, dst_type, dst_id, predicate, strength, valid_from)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                # Reverse pair
                (100, "entity", 1, "entity", 2, "hosts", a_strength, now_a),
                (200, "entity", 2, "entity", 1, "hosts", b_strength, now_b),
                # Origin mentions (episode 10 -> entities, episode 20 -> entities)
                (101, "episode", 10, "entity", 1, "mentions", 1.0, now_a),
                (102, "episode", 10, "entity", 2, "mentions", 1.0, now_a),
                (201, "episode", 20, "entity", 1, "mentions", 1.0, now_b),
                (202, "episode", 20, "entity", 2, "mentions", 1.0, now_b),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db


class _FakeDirectionJudgeClient:
    """Returns direction judgments in FIFO order from a pre-built queue."""

    def __init__(self, judgments: list[_DirectionJudgment | JudgeError]) -> None:
        self._queue = list(judgments)

    def judge_direction(self, *args: object, **kwargs: object) -> _DirectionJudgment:
        item = self._queue.pop(0)
        if isinstance(item, JudgeError):
            raise item
        return item


def _make_config(**overrides: object) -> SemanticJudgeConfig:
    kwargs: dict[str, object] = {
        "enabled": True,
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "test-model",
        "api_key": "dummy",
        "timeout": 60.0,
        "max_retries": 2,
        "max_attempts": 3,
        "temperature": 0.0,
        "min_strength": 0.3,
    }
    kwargs.update(overrides)
    return SemanticJudgeConfig.model_validate(kwargs)


class TestFetchDirectionPairs:
    def test_fetch_direction_pairs_detects_reverse_pair(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path)
        conn = sqlite3.connect(db)
        try:
            pairs = _fetch_direction_pairs(conn, min_strength=0.3)
        finally:
            conn.close()

        assert len(pairs) == 1
        p = pairs[0]
        assert p.a_id == 100
        assert p.b_id == 200
        assert p.a_src_id == 1
        assert p.a_dst_id == 2
        assert p.predicate == "hosts"
        assert p.a_src_name == "serverA"
        assert p.a_dst_name == "serverB"

    def test_fetch_direction_pairs_skips_below_min_strength(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path, a_strength=1.0, b_strength=0.1)
        conn = sqlite3.connect(db)
        try:
            pairs = _fetch_direction_pairs(conn, min_strength=0.3)
        finally:
            conn.close()

        assert len(pairs) == 0


class TestJudgeRelationDirections:
    NOW = datetime(2026, 4, 27, 12, 0, 0, tzinfo=UTC)

    def test_judge_relation_directions_high_a_supersedes_b(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path)
        fake = _FakeDirectionJudgeClient([
            _DirectionJudgment(correct="A", confidence="high", reason="Aが正しい"),
        ])

        result = judge_relation_directions(
            db_path=db, config=_make_config(), judge_client=fake, now=self.NOW,
        )

        assert result.superseded_b == 1
        assert result.superseded_a == 0
        assert result.kept_both == 0
        assert result.superseded_both == 0
        conn = sqlite3.connect(db)
        try:
            b_row = conn.execute(
                "SELECT valid_to, supersedes FROM relations WHERE id = 200"
            ).fetchone()
            assert b_row[0] is not None
            assert b_row[1] is None
            a_row = conn.execute(
                "SELECT valid_to, supersedes FROM relations WHERE id = 100"
            ).fetchone()
            assert a_row[0] is None
            assert a_row[1] == 200
            # Episode supersede propagation (Commit C)
            # B (loser) episode archived; A (winner) episode untouched
            ep_b = conn.execute(
                "SELECT valid_to, supersedes FROM episodes WHERE id = 20"
            ).fetchone()
            assert ep_b[0] is not None
            assert ep_b[1] == 10
            ep_a = conn.execute(
                "SELECT valid_to, supersedes FROM episodes WHERE id = 10"
            ).fetchone()
            assert ep_a[0] is None
            assert ep_a[1] is None
        finally:
            conn.close()

    def test_judge_relation_directions_high_neither_supersedes_both(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path)
        fake = _FakeDirectionJudgeClient([
            _DirectionJudgment(correct="neither", confidence="high", reason="どちらも誤り"),
        ])

        result = judge_relation_directions(
            db_path=db, config=_make_config(), judge_client=fake, now=self.NOW,
        )

        assert result.superseded_both == 1
        conn = sqlite3.connect(db)
        try:
            a_row = conn.execute(
                "SELECT valid_to, supersedes FROM relations WHERE id = 100"
            ).fetchone()
            b_row = conn.execute(
                "SELECT valid_to, supersedes FROM relations WHERE id = 200"
            ).fetchone()
            assert a_row[0] is not None
            assert b_row[0] is not None
            assert a_row[1] is None
            assert b_row[1] is None
            # Neither: both episodes archived (no keeper)
            ep_a = conn.execute(
                "SELECT valid_to, supersedes FROM episodes WHERE id = 10"
            ).fetchone()
            ep_b = conn.execute(
                "SELECT valid_to, supersedes FROM episodes WHERE id = 20"
            ).fetchone()
            assert ep_a[0] is not None
            assert ep_a[1] is None
            assert ep_b[0] is not None
            assert ep_b[1] is None
        finally:
            conn.close()

    def test_judge_relation_directions_high_both_keeps_both(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path)
        fake = _FakeDirectionJudgeClient([
            _DirectionJudgment(correct="both", confidence="high", reason="双方向関係"),
        ])

        result = judge_relation_directions(
            db_path=db, config=_make_config(), judge_client=fake, now=self.NOW,
        )

        assert result.kept_both == 1
        conn = sqlite3.connect(db)
        try:
            live_count = conn.execute(
                "SELECT COUNT(*) FROM relations WHERE id IN (100, 200) AND valid_to IS NULL"
            ).fetchone()[0]
            assert live_count == 2
        finally:
            conn.close()

    def test_judge_relation_directions_medium_queues_for_review(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path)
        fake = _FakeDirectionJudgeClient([
            _DirectionJudgment(correct="A", confidence="medium", reason="自信なし"),
        ])

        result = judge_relation_directions(
            db_path=db, config=_make_config(), judge_client=fake, now=self.NOW,
        )

        assert result.queued_for_review == 1
        assert result.superseded_a == 0
        assert result.superseded_b == 0
        conn = sqlite3.connect(db)
        try:
            live_count = conn.execute(
                "SELECT COUNT(*) FROM relations WHERE id IN (100, 200) AND valid_to IS NULL"
            ).fetchone()[0]
            assert live_count == 2
        finally:
            conn.close()

    def test_judge_direction_dry_run_rolls_back(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path)
        fake = _FakeDirectionJudgeClient([
            _DirectionJudgment(correct="A", confidence="high", reason="Aが正しい"),
        ])

        result = judge_relation_directions(
            db_path=db, config=_make_config(), judge_client=fake,
            now=self.NOW, dry_run=True,
        )

        assert result.superseded_b == 1
        conn = sqlite3.connect(db)
        try:
            live_count = conn.execute(
                "SELECT COUNT(*) FROM relations WHERE id IN (100, 200) AND valid_to IS NULL"
            ).fetchone()[0]
            assert live_count == 2
        finally:
            conn.close()

    def test_judge_direction_error_counts_as_error(self, tmp_path: Path) -> None:
        db = _make_db_with_reverse_pair(tmp_path)
        fake = _FakeDirectionJudgeClient([JudgeError("connection timeout")])

        result = judge_relation_directions(
            db_path=db, config=_make_config(), judge_client=fake, now=self.NOW,
        )

        assert result.errors == 1
        assert result.judgments[0].action == "error"
        conn = sqlite3.connect(db)
        try:
            live_count = conn.execute(
                "SELECT COUNT(*) FROM relations WHERE id IN (100, 200) AND valid_to IS NULL"
            ).fetchone()[0]
            assert live_count == 2
        finally:
            conn.close()

    def test_skip_self_hub_pair(self, tmp_path: Path) -> None:
        """is_self=1 entity を含む方向違い pair は LLM 判定をスキップ"""
        db = tmp_path / "kv.sqlite"
        init_kv(db)
        now = datetime.now(UTC).isoformat()
        conn = sqlite3.connect(db)
        try:
            # Entities (serverA は is_self=1)
            conn.executemany(
                "INSERT INTO entities (id, canonical_name, entity_type, is_self, valid_from)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (1, "serverA", "concept", 1, now),
                    (2, "serverB", "concept", 0, now),
                ],
            )
            # Episodes (origin texts)
            conn.executemany(
                "INSERT INTO episodes (id, content, kind, importance, valid_from)"
                " VALUES (?, ?, 'facts', 2, ?)",
                [
                    (10, "serverA hosts serverB", now),
                    (20, "serverB hosts serverA", now),
                ],
            )
            # Relations
            conn.executemany(
                "INSERT INTO relations"
                " (id, src_type, src_id, dst_type, dst_id, predicate, strength, valid_from)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    # Reverse pair (serverA is_self=1)
                    (100, "entity", 1, "entity", 2, "hosts", 1.0, now),
                    (200, "entity", 2, "entity", 1, "hosts", 1.0, now),
                    # Origin mentions
                    (101, "episode", 10, "entity", 1, "mentions", 1.0, now),
                    (102, "episode", 10, "entity", 2, "mentions", 1.0, now),
                    (201, "episode", 20, "entity", 1, "mentions", 1.0, now),
                    (202, "episode", 20, "entity", 2, "mentions", 1.0, now),
                ],
            )
            conn.commit()
        finally:
            conn.close()

        # _FakeDirectionJudgeClient should NOT be called (deferred instead of LLM judgment)
        fake = _FakeDirectionJudgeClient([])

        result = judge_relation_directions(
            db_path=db, config=_make_config(), judge_client=fake, now=self.NOW,
        )

        # Both relations should remain untouched, no supersede happened
        assert result.superseded_a == 0
        assert result.superseded_b == 0
        # Deferred should be incremented
        assert result.deferred >= 1
        # Verify both relations are still live
        conn = sqlite3.connect(db)
        try:
            a_row = conn.execute(
                "SELECT valid_to FROM relations WHERE id = 100"
            ).fetchone()
            b_row = conn.execute(
                "SELECT valid_to FROM relations WHERE id = 200"
            ).fetchone()
            assert a_row[0] is None
            assert b_row[0] is None
        finally:
            conn.close()
