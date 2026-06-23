"""Unit tests for core/compact.py (Phase 4 fan_out / strength / suppress / merge)."""

from __future__ import annotations

import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import sqlite_vec

from fravenir.core.compact import CompactResult, run_compact
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init
from fravenir.storage.vector import upsert_entity_vector


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    return char_id


def _make_config(char_id: str = "test_char") -> AppConfig:
    return AppConfig(character=CharacterConfig(id=char_id))


def _insert_relation(
    tmp_project: Path,
    char_id: str,
    *,
    src_id: int,
    dst_id: int,
    predicate: str = "likes",
    valid_to: str | None = None,
    fan_out: int = 1,
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO relations (
                src_type, src_id, dst_type, dst_id, predicate,
                strength, fan_out, valid_from, valid_to
            ) VALUES (?, ?, ?, ?, ?, 1.0, ?, '2026-01-01T00:00:00+00:00', ?)
            """,
            ("entity", src_id, "entity", dst_id, predicate, fan_out, valid_to),
        )
        conn.commit()
        rid: int = cur.lastrowid  # type: ignore[assignment]
        return rid
    finally:
        conn.close()


def _read_fan_out(tmp_project: Path, char_id: str, relation_id: int) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        row = conn.execute(
            "SELECT fan_out FROM relations WHERE id = ?", (relation_id,)
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


class TestArchiveSelfLoops:
    def test_archives_active_self_loop(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        rid = _insert_relation(tmp_project, char_id, src_id=1, dst_id=1)
        result = run_compact(character_id=char_id, config=_make_config(char_id))
        assert result.self_loops_archived == 1

        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            valid_to = conn.execute(
                "SELECT valid_to FROM relations WHERE id = ?", (rid,)
            ).fetchone()[0]
            assert valid_to is not None  # 論理削除されている
        finally:
            conn.close()

    def test_ignores_already_archived(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        _insert_relation(
            tmp_project, char_id, src_id=1, dst_id=1,
            valid_to="2026-01-02T00:00:00+00:00",
        )
        result = run_compact(character_id=char_id, config=_make_config(char_id))
        assert result.self_loops_archived == 0

    def test_ignores_non_self_relations(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        _insert_relation(tmp_project, char_id, src_id=1, dst_id=2)
        result = run_compact(character_id=char_id, config=_make_config(char_id))
        assert result.self_loops_archived == 0

    def test_dry_run_does_not_persist(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        rid = _insert_relation(tmp_project, char_id, src_id=1, dst_id=1)
        result = run_compact(
            character_id=char_id, config=_make_config(char_id), dry_run=True
        )
        assert result.self_loops_archived == 1

        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            valid_to = conn.execute(
                "SELECT valid_to FROM relations WHERE id = ?", (rid,)
            ).fetchone()[0]
            assert valid_to is None  # rollback されている
        finally:
            conn.close()


class TestRecomputeFanOut:
    def test_updates_live_relations(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        # 同じ src(entity, 1) から 3本伸ばす（fan_out 初期値1のまま登録）
        r1 = _insert_relation(tmp_project, char_id, src_id=1, dst_id=10)
        r2 = _insert_relation(tmp_project, char_id, src_id=1, dst_id=11)
        r3 = _insert_relation(tmp_project, char_id, src_id=1, dst_id=12)

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.fan_out_updated == 3
        assert _read_fan_out(tmp_project, char_id, r1) == 3
        assert _read_fan_out(tmp_project, char_id, r2) == 3
        assert _read_fan_out(tmp_project, char_id, r3) == 3

    def test_excludes_archived(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        # 同 src だが 1本は valid_to 立て、ライブ2本のみカウント対象
        live_a = _insert_relation(tmp_project, char_id, src_id=2, dst_id=20)
        live_b = _insert_relation(tmp_project, char_id, src_id=2, dst_id=21)
        archived = _insert_relation(
            tmp_project,
            char_id,
            src_id=2,
            dst_id=22,
            valid_to="2026-04-01T00:00:00+00:00",
            fan_out=99,  # archived は更新されないことの確認用
        )

        run_compact(character_id=char_id, config=_make_config(char_id))

        assert _read_fan_out(tmp_project, char_id, live_a) == 2
        assert _read_fan_out(tmp_project, char_id, live_b) == 2
        # archived は手付かず
        assert _read_fan_out(tmp_project, char_id, archived) == 99


class TestRunCompactBehavior:
    def test_dry_run_does_not_write(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        rid = _insert_relation(tmp_project, char_id, src_id=3, dst_id=30)
        _insert_relation(tmp_project, char_id, src_id=3, dst_id=31)

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), dry_run=True
        )

        # 「変わるはずの行数」は返るが、DB は初期値 1 のまま
        assert result.fan_out_updated == 2
        assert result.dry_run is True
        assert _read_fan_out(tmp_project, char_id, rid) == 1

    def test_returns_zero_for_empty(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.fan_out_updated == 0
        assert result.strength_updated == 0
        assert result.suppressed == 0
        assert result.merge_candidates == 0
        assert result.duration_ms >= 0
        assert result.dry_run is False

    def test_to_dict_keys_match_server_stub_contract(self) -> None:
        result = CompactResult(
            fan_out_updated=1,
            strength_updated=0,
            suppressed=0,
            merge_candidates=0,
            duration_ms=5,
            dry_run=False,
        )
        d = result.to_dict()
        # 旧スタブの後方互換キー + strength_updated（P4-2 で値が入る）
        # + self_loops_archived（BUG-1 / Phase 6 で追加）
        assert set(d.keys()) == {
            "fan_out_updated",
            "strength_updated",
            "suppressed",
            "merge_candidates",
            "self_loops_archived",
            "duration_ms",
            "dry_run",
        }

    def test_compact_use_llm_disabled_no_judgment_phase(self, tmp_project: Path) -> None:
        """use_llm=False or config.semantic_judge.enabled=False → judgment is None."""
        char_id = _make_character(tmp_project)

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), use_llm=False,
        )
        assert result.judgment is None
        assert "judgment" not in result.to_dict()


def _insert_episode(
    tmp_project: Path, char_id: str, *, valid_to: str | None = None
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO episodes (content, kind, importance, valid_from, valid_to)
            VALUES ('テスト本文', 'facts', 1, '2026-01-01T00:00:00+00:00', ?)
            """,
            (valid_to,),
        )
        conn.commit()
        ep_id: int = cur.lastrowid  # type: ignore[assignment]
        return ep_id
    finally:
        conn.close()


def _insert_entity(tmp_project: Path, char_id: str, name: str) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, valid_from)
            VALUES (?, 'concept', '2026-01-01T00:00:00+00:00')
            """,
            (name,),
        )
        conn.commit()
        eid: int = cur.lastrowid  # type: ignore[assignment]
        return eid
    finally:
        conn.close()


def _insert_mention(
    tmp_project: Path, char_id: str, *, episode_id: int, entity_id: int
) -> None:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        conn.execute(
            """
            INSERT INTO relations (
                src_type, src_id, dst_type, dst_id, predicate,
                strength, fan_out, valid_from
            ) VALUES ('episode', ?, 'entity', ?, 'mentions',
                      1.0, 1, '2026-01-01T00:00:00+00:00')
            """,
            (episode_id, entity_id),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_entity_relation(
    tmp_project: Path,
    char_id: str,
    *,
    src_id: int,
    dst_id: int,
    predicate: str = "likes",
    strength: float = 1.0,
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO relations (
                src_type, src_id, dst_type, dst_id, predicate,
                strength, fan_out, valid_from
            ) VALUES ('entity', ?, 'entity', ?, ?,
                      ?, 1, '2026-01-01T00:00:00+00:00')
            """,
            (src_id, dst_id, predicate, strength),
        )
        conn.commit()
        rid: int = cur.lastrowid  # type: ignore[assignment]
        return rid
    finally:
        conn.close()


def _read_strength(tmp_project: Path, char_id: str, relation_id: int) -> float:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        row = conn.execute(
            "SELECT strength FROM relations WHERE id = ?", (relation_id,)
        ).fetchone()
        return float(row[0])
    finally:
        conn.close()


class TestRecomputeStrength:
    def test_uses_cooccurrence(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ent_a = _insert_entity(tmp_project, char_id, "猫")
        ent_b = _insert_entity(tmp_project, char_id, "魚")
        # 2つの episode から両方 mention → cooccurrence=2
        ep1 = _insert_episode(tmp_project, char_id)
        ep2 = _insert_episode(tmp_project, char_id)
        _insert_mention(tmp_project, char_id, episode_id=ep1, entity_id=ent_a)
        _insert_mention(tmp_project, char_id, episode_id=ep1, entity_id=ent_b)
        _insert_mention(tmp_project, char_id, episode_id=ep2, entity_id=ent_a)
        _insert_mention(tmp_project, char_id, episode_id=ep2, entity_id=ent_b)
        rel = _insert_entity_relation(
            tmp_project, char_id, src_id=ent_a, dst_id=ent_b
        )

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.strength_updated == 1
        assert _read_strength(tmp_project, char_id, rel) == 1.0 + math.log(2)

    def test_no_cooccurrence_keeps_strength(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ent_a = _insert_entity(tmp_project, char_id, "A")
        ent_b = _insert_entity(tmp_project, char_id, "B")
        # mentions が一切ない → cooccurrence=0
        rel = _insert_entity_relation(
            tmp_project, char_id, src_id=ent_a, dst_id=ent_b, strength=2.5
        )

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.strength_updated == 0
        # 既存値 2.5 が維持されてる
        assert _read_strength(tmp_project, char_id, rel) == 2.5

    def test_archived_episodes_excluded(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ent_a = _insert_entity(tmp_project, char_id, "A")
        ent_b = _insert_entity(tmp_project, char_id, "B")
        # ライブ ep1 + 削除済み ep2 → cooccurrence=1
        ep_live = _insert_episode(tmp_project, char_id)
        ep_archived = _insert_episode(tmp_project, char_id)
        _insert_mention(tmp_project, char_id, episode_id=ep_live, entity_id=ent_a)
        _insert_mention(tmp_project, char_id, episode_id=ep_live, entity_id=ent_b)
        _insert_mention(tmp_project, char_id, episode_id=ep_archived, entity_id=ent_a)
        _insert_mention(tmp_project, char_id, episode_id=ep_archived, entity_id=ent_b)
        # mentions 側の valid_to を立てて archived エピソード由来の共起を除外
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            conn.execute(
                "UPDATE relations SET valid_to = '2026-04-01T00:00:00+00:00' "
                "WHERE src_type = 'episode' AND src_id = ?",
                (ep_archived,),
            )
            conn.commit()
        finally:
            conn.close()
        rel = _insert_entity_relation(
            tmp_project, char_id, src_id=ent_a, dst_id=ent_b
        )

        run_compact(character_id=char_id, config=_make_config(char_id))

        # cooccurrence=1 → strength = 1.0 + ln(1) = 1.0、初期値と同じなので更新なし
        assert _read_strength(tmp_project, char_id, rel) == 1.0

    def test_dry_run_skips_strength_write(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ent_a = _insert_entity(tmp_project, char_id, "A")
        ent_b = _insert_entity(tmp_project, char_id, "B")
        ep1 = _insert_episode(tmp_project, char_id)
        ep2 = _insert_episode(tmp_project, char_id)
        for ep in (ep1, ep2):
            _insert_mention(tmp_project, char_id, episode_id=ep, entity_id=ent_a)
            _insert_mention(tmp_project, char_id, episode_id=ep, entity_id=ent_b)
        rel = _insert_entity_relation(
            tmp_project, char_id, src_id=ent_a, dst_id=ent_b
        )

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), dry_run=True
        )

        # 件数は返るが DB は不変
        assert result.strength_updated == 1
        assert _read_strength(tmp_project, char_id, rel) == 1.0

    def test_only_targets_entity_to_entity(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ent_a = _insert_entity(tmp_project, char_id, "A")
        ep1 = _insert_episode(tmp_project, char_id)
        # episode → entity の mentions 自体（src_type='episode'）は対象外
        _insert_mention(tmp_project, char_id, episode_id=ep1, entity_id=ent_a)

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        # mentions relation の strength は触らない
        assert result.strength_updated == 0
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            row = conn.execute(
                "SELECT strength FROM relations "
                "WHERE src_type = 'episode' AND predicate = 'mentions'"
            ).fetchone()
            assert row[0] == 1.0
        finally:
            conn.close()


def _insert_access(
    tmp_project: Path,
    char_id: str,
    *,
    node_type: str,
    node_id: int,
    accessed_at: datetime,
    source: str = "test",
) -> None:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        conn.execute(
            "INSERT INTO access_history (node_type, node_id, accessed_at, source) "
            "VALUES (?, ?, ?, ?)",
            (node_type, node_id, accessed_at.isoformat(), source),
        )
        conn.commit()
    finally:
        conn.close()


def _read_is_suppressed(tmp_project: Path, char_id: str, episode_id: int) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        row = conn.execute(
            "SELECT is_suppressed FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


class TestMarkSuppressed:
    NOW = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)

    def test_suppresses_low_activation(self, tmp_project: Path) -> None:
        """access_history が古いだけ → B_i が大きく負に振れて threshold を下回る。"""
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)
        # 1年前の単発アクセス: t≈3.15e7, t^-0.5≈1.78e-4, ln≈-8.6 < -2.0
        _insert_access(
            tmp_project,
            char_id,
            node_type="episode",
            node_id=ep_id,
            accessed_at=self.NOW - timedelta(days=365),
        )

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), now=self.NOW
        )

        assert result.suppressed == 1
        assert _read_is_suppressed(tmp_project, char_id, ep_id) == 1

    def test_recent_access_protects(self, tmp_project: Path) -> None:
        """直近7日以内のアクセスがあれば B_i を計算する前に保護される。"""
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)
        _insert_access(
            tmp_project,
            char_id,
            node_type="episode",
            node_id=ep_id,
            accessed_at=self.NOW - timedelta(days=3),
        )

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), now=self.NOW
        )

        assert result.suppressed == 0
        assert _read_is_suppressed(tmp_project, char_id, ep_id) == 0

    def test_above_threshold_not_suppressed(self, tmp_project: Path) -> None:
        """B_i が threshold を上回るなら（直近アクセスがなくても）抑制されない。"""
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)
        _insert_access(
            tmp_project,
            char_id,
            node_type="episode",
            node_id=ep_id,
            accessed_at=self.NOW - timedelta(days=365),
        )
        # threshold を緩めて、どんな B_i でも上回る設定にする
        config = _make_config(char_id)
        config.act_r.suppress_threshold = -1000.0

        result = run_compact(character_id=char_id, config=config, now=self.NOW)

        assert result.suppressed == 0
        assert _read_is_suppressed(tmp_project, char_id, ep_id) == 0

    def test_already_suppressed_skipped(self, tmp_project: Path) -> None:
        """is_suppressed=1 はスキャンから除外、再カウントされない。"""
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            conn.execute(
                "UPDATE episodes SET is_suppressed = 1 WHERE id = ?", (ep_id,)
            )
            conn.commit()
        finally:
            conn.close()
        _insert_access(
            tmp_project,
            char_id,
            node_type="episode",
            node_id=ep_id,
            accessed_at=self.NOW - timedelta(days=365),
        )

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), now=self.NOW
        )

        # is_suppressed=1 だった分は新規カウントに含まれない
        assert result.suppressed == 0

    def test_archived_episodes_skipped(self, tmp_project: Path) -> None:
        """valid_to が立った episode は対象外。"""
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(
            tmp_project, char_id, valid_to="2026-04-01T00:00:00+00:00"
        )
        _insert_access(
            tmp_project,
            char_id,
            node_type="episode",
            node_id=ep_id,
            accessed_at=self.NOW - timedelta(days=365),
        )

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), now=self.NOW
        )

        assert result.suppressed == 0
        assert _read_is_suppressed(tmp_project, char_id, ep_id) == 0

    def test_dry_run_skips_write(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        ep_id = _insert_episode(tmp_project, char_id)
        _insert_access(
            tmp_project,
            char_id,
            node_type="episode",
            node_id=ep_id,
            accessed_at=self.NOW - timedelta(days=365),
        )

        result = run_compact(
            character_id=char_id,
            config=_make_config(char_id),
            now=self.NOW,
            dry_run=True,
        )

        # 件数は返るが DB は不変
        assert result.suppressed == 1
        assert _read_is_suppressed(tmp_project, char_id, ep_id) == 0


def _insert_entity_with_type(
    tmp_project: Path, char_id: str, name: str, entity_type: str = "concept"
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        cur = conn.execute(
            """
            INSERT INTO entities (canonical_name, entity_type, valid_from)
            VALUES (?, ?, '2026-01-01T00:00:00+00:00')
            """,
            (name, entity_type),
        )
        conn.commit()
        eid: int = cur.lastrowid  # type: ignore[assignment]
        return eid
    finally:
        conn.close()


def _put_entity_vector(
    tmp_project: Path, char_id: str, entity_id: int, vec: np.ndarray
) -> None:
    """正規化したベクトルを vdb_entities に書き込むヘルパ。"""
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


def _count_merge_candidates(
    tmp_project: Path, char_id: str, *, resolved: int | None = None
) -> int:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
    try:
        if resolved is None:
            row = conn.execute(
                "SELECT COUNT(*) FROM merge_candidates"
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM merge_candidates WHERE resolved = ?",
                (resolved,),
            ).fetchone()
        return int(row[0])
    finally:
        conn.close()


class TestDetectMergeCandidates:
    def test_detects_high_similarity_pair(self, tmp_project: Path) -> None:
        """cosine > 0.85、entity_type 一致、レーベンシュタイン < 3 のペアを検出。"""
        char_id = _make_character(tmp_project)
        a_id = _insert_entity_with_type(tmp_project, char_id, "テスト")
        b_id = _insert_entity_with_type(tmp_project, char_id, "テス卜")  # 1文字差
        # ほぼ同じベクトル
        rng = np.random.default_rng(42)
        v = rng.normal(size=768).astype(np.float32)
        _put_entity_vector(tmp_project, char_id, a_id, v)
        _put_entity_vector(tmp_project, char_id, b_id, v + 0.01 * rng.normal(size=768))

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 1
        assert _count_merge_candidates(tmp_project, char_id, resolved=0) == 1

    def test_skips_different_entity_types(self, tmp_project: Path) -> None:
        # 同名は UNIQUE 制約に弾かれるので、レーベンシュタイン1の似た名で別 type を作る
        char_id = _make_character(tmp_project)
        a_id = _insert_entity_with_type(tmp_project, char_id, "テスト", "concept")
        b_id = _insert_entity_with_type(tmp_project, char_id, "テス卜", "person")
        rng = np.random.default_rng(7)
        v = rng.normal(size=768).astype(np.float32)
        _put_entity_vector(tmp_project, char_id, a_id, v)
        _put_entity_vector(tmp_project, char_id, b_id, v)

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        # cosine も name 距離もパスするが entity_type 違いで除外
        assert result.merge_candidates == 0

    def test_skips_far_canonical_names(self, tmp_project: Path) -> None:
        """cosine 高くても canonical_name の距離 >= 3 なら除外。"""
        char_id = _make_character(tmp_project)
        a_id = _insert_entity_with_type(tmp_project, char_id, "アルファベット")
        b_id = _insert_entity_with_type(tmp_project, char_id, "プログラミング")  # 大幅に違う
        rng = np.random.default_rng(3)
        v = rng.normal(size=768).astype(np.float32)
        _put_entity_vector(tmp_project, char_id, a_id, v)
        _put_entity_vector(tmp_project, char_id, b_id, v)

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 0

    def test_skips_low_similarity(self, tmp_project: Path) -> None:
        """cosine <= 0.85 のペアは除外。"""
        char_id = _make_character(tmp_project)
        a_id = _insert_entity_with_type(tmp_project, char_id, "テスト")
        b_id = _insert_entity_with_type(tmp_project, char_id, "テス卜")
        # 直交に近いベクトルを与える
        v_a = np.zeros(768, dtype=np.float32)
        v_a[0] = 1.0
        v_b = np.zeros(768, dtype=np.float32)
        v_b[1] = 1.0
        _put_entity_vector(tmp_project, char_id, a_id, v_a)
        _put_entity_vector(tmp_project, char_id, b_id, v_b)

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 0

    def test_skips_already_pending(self, tmp_project: Path) -> None:
        """resolved=0 で同ペア既登録のスキップ。"""
        char_id = _make_character(tmp_project)
        a_id = _insert_entity_with_type(tmp_project, char_id, "テスト")
        b_id = _insert_entity_with_type(tmp_project, char_id, "テス卜")
        rng = np.random.default_rng(11)
        v = rng.normal(size=768).astype(np.float32)
        _put_entity_vector(tmp_project, char_id, a_id, v)
        _put_entity_vector(tmp_project, char_id, b_id, v)

        # 同ペアを (b, a) の向きで先に登録（正規化されてないペアでも検出されるか）
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            conn.execute(
                "INSERT INTO merge_candidates (entity_a, entity_b, similarity, resolved) "
                "VALUES (?, ?, 0.99, 0)",
                (b_id, a_id),
            )
            conn.commit()
        finally:
            conn.close()

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 0
        assert _count_merge_candidates(tmp_project, char_id, resolved=0) == 1

    def _setup_similar_pair(
        self, tmp_project: Path, *, seed: int
    ) -> tuple[str, int, int]:
        """cosine 高・name 1文字差の検出対象ペアを作って (char_id, a_id, b_id) を返す。"""
        char_id = _make_character(tmp_project)
        a_id = _insert_entity_with_type(tmp_project, char_id, "テスト")
        b_id = _insert_entity_with_type(tmp_project, char_id, "テス卜")
        rng = np.random.default_rng(seed)
        v = rng.normal(size=768).astype(np.float32)
        _put_entity_vector(tmp_project, char_id, a_id, v)
        _put_entity_vector(tmp_project, char_id, b_id, v)
        return char_id, a_id, b_id

    def _insert_resolved_candidate(
        self,
        tmp_project: Path,
        char_id: str,
        a_id: int,
        b_id: int,
        *,
        resolved: int,
        resolved_at: str | None,
    ) -> None:
        small, large = (a_id, b_id) if a_id < b_id else (b_id, a_id)
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            conn.execute(
                "INSERT INTO merge_candidates "
                "(entity_a, entity_b, similarity, resolved, resolved_at) "
                "VALUES (?, ?, 0.99, ?, ?)",
                (small, large, resolved, resolved_at),
            )
            conn.commit()
        finally:
            conn.close()

    def _set_curated_at(
        self, tmp_project: Path, char_id: str, entity_id: int, curated_at: str
    ) -> None:
        conn = sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))
        try:
            conn.execute(
                "UPDATE entities SET curated_at = ? WHERE id = ?",
                (curated_at, entity_id),
            )
            conn.commit()
        finally:
            conn.close()

    def test_skips_rejected_without_curation(self, tmp_project: Path) -> None:
        """却下済み(resolved=2)で curation されていないペアは再検出しない。"""
        char_id, a_id, b_id = self._setup_similar_pair(tmp_project, seed=21)
        self._insert_resolved_candidate(
            tmp_project, char_id, a_id, b_id,
            resolved=2, resolved_at="2026-03-01T00:00:00+00:00",
        )

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 0
        assert _count_merge_candidates(tmp_project, char_id, resolved=0) == 0

    def test_redetects_rejected_after_curation(self, tmp_project: Path) -> None:
        """却下後に entity が curation された(curated_at > resolved_at)なら再判定を許可。"""
        char_id, a_id, b_id = self._setup_similar_pair(tmp_project, seed=22)
        self._insert_resolved_candidate(
            tmp_project, char_id, a_id, b_id,
            resolved=2, resolved_at="2026-03-01T00:00:00+00:00",
        )
        # 却下より後に curation
        self._set_curated_at(
            tmp_project, char_id, a_id, "2026-03-02T00:00:00+00:00"
        )

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 1
        assert _count_merge_candidates(tmp_project, char_id, resolved=0) == 1

    def test_skips_rejected_when_curation_predates_rejection(
        self, tmp_project: Path
    ) -> None:
        """却下より前の curation では再検出しない（比較の向きガード）。"""
        char_id, a_id, b_id = self._setup_similar_pair(tmp_project, seed=23)
        self._insert_resolved_candidate(
            tmp_project, char_id, a_id, b_id,
            resolved=2, resolved_at="2026-03-01T00:00:00+00:00",
        )
        # 却下より前の curation → 新しい情報ではない
        self._set_curated_at(
            tmp_project, char_id, a_id, "2026-02-01T00:00:00+00:00"
        )

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 0

    def test_skips_merged_pair(self, tmp_project: Path) -> None:
        """merge 済み(resolved=1)ペアは恒久スキップ。"""
        char_id, a_id, b_id = self._setup_similar_pair(tmp_project, seed=24)
        self._insert_resolved_candidate(
            tmp_project, char_id, a_id, b_id,
            resolved=1, resolved_at="2026-03-01T00:00:00+00:00",
        )

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 0

    def test_redetects_legacy_rejected_null_resolved_at(
        self, tmp_project: Path
    ) -> None:
        """resolved_at が NULL の旧却下行は初回 1 回だけ再検出される（移行救済）。"""
        char_id, a_id, b_id = self._setup_similar_pair(tmp_project, seed=25)
        self._insert_resolved_candidate(
            tmp_project, char_id, a_id, b_id,
            resolved=2, resolved_at=None,
        )

        result = run_compact(character_id=char_id, config=_make_config(char_id))

        assert result.merge_candidates == 1
        assert _count_merge_candidates(tmp_project, char_id, resolved=0) == 1

    def test_dry_run_skips_insert(self, tmp_project: Path) -> None:
        char_id = _make_character(tmp_project)
        a_id = _insert_entity_with_type(tmp_project, char_id, "テスト")
        b_id = _insert_entity_with_type(tmp_project, char_id, "テス卜")
        rng = np.random.default_rng(5)
        v = rng.normal(size=768).astype(np.float32)
        _put_entity_vector(tmp_project, char_id, a_id, v)
        _put_entity_vector(tmp_project, char_id, b_id, v)

        result = run_compact(
            character_id=char_id, config=_make_config(char_id), dry_run=True
        )

        assert result.merge_candidates == 1
        assert _count_merge_candidates(tmp_project, char_id) == 0
