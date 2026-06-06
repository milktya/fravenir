"""Integration tests for the nightly memory_compact pipeline (Phase 4).

実 SQLite + vdb_entities に episodes / entities / relations / access_history を
直接投入し、run_compact を1回走らせて 4 ステップ全部が期待通り動くことを確認する。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import sqlite_vec

from fravenir.core.compact import run_compact
from fravenir.schemas.config import AppConfig, CharacterConfig
from fravenir.storage import sqlite_init
from fravenir.storage.vector import upsert_entity_vector


def _make_character(tmp_project: Path, char_id: str = "integ_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    sqlite_init.init_vdb_entities(data_dir / "vdb_entities.db")
    return char_id


def _config(char_id: str) -> AppConfig:
    return AppConfig(character=CharacterConfig(id=char_id))


def _kv(tmp_project: Path, char_id: str) -> sqlite3.Connection:
    return sqlite3.connect(str(tmp_project / "data" / char_id / "kv.sqlite"))


def _open_vdb(tmp_project: Path, char_id: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(tmp_project / "data" / char_id / "vdb_entities.db"))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _seed_dataset(
    tmp_project: Path,
    char_id: str,
    *,
    now: datetime,
) -> dict[str, list[int]]:
    """4 ステップ全部に火を入れるミニマルなデータセットを投入する。

    - episodes: 4件（ep_old1/ep_old2 は古いアクセス → 抑制対象、ep_recent はライブ）
    - entities: 4件（cat / fish は同 type で名前1文字差 + ベクトルほぼ同じ → merge 候補）
    - relations: cat→fish の entity-entity を1本（fan_out 計算対象）
    - mentions: ep_old1 / ep_old2 が cat と fish 両方に → strength 共起 2
    - access_history: 1年前1件（ep_old1, ep_old2 を抑制対象に）
    """
    valid_from = "2026-01-01T00:00:00+00:00"
    kv = _kv(tmp_project, char_id)
    try:
        episodes: list[int] = []
        for i in range(3):
            cur = kv.execute(
                "INSERT INTO episodes (content, kind, importance, valid_from) "
                "VALUES (?, 'facts', 1, ?)",
                (f"epi-{i}", valid_from),
            )
            episodes.append(int(cur.lastrowid))  # type: ignore[arg-type]

        entities: list[int] = []
        names_types = [
            ("猫", "concept"),     # cat
            ("狗", "concept"),     # fish-likeness のための漢字1文字差ペア候補
            ("空", "concept"),     # 単独
            ("海", "concept"),     # 単独
        ]
        for name, etype in names_types:
            cur = kv.execute(
                "INSERT INTO entities (canonical_name, entity_type, valid_from) "
                "VALUES (?, ?, ?)",
                (name, etype, valid_from),
            )
            entities.append(int(cur.lastrowid))  # type: ignore[arg-type]

        # cat → fish (entity-entity)
        kv.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, "
            "strength, fan_out, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'likes', 1.0, 1, ?)",
            (entities[0], entities[1], valid_from),
        )

        # mentions: ep_old1 / ep_old2 がそれぞれ cat と fish を両方 mention
        for ep_id in episodes[:2]:
            for ent_id in entities[:2]:
                kv.execute(
                    "INSERT INTO relations (src_type, src_id, dst_type, dst_id, "
                    "predicate, strength, fan_out, valid_from) "
                    "VALUES ('episode', ?, 'entity', ?, 'mentions', 1.0, 1, ?)",
                    (ep_id, ent_id, valid_from),
                )

        # ep_old1 / ep_old2 に1年前のアクセス履歴
        old = (now - timedelta(days=365)).isoformat()
        for ep_id in episodes[:2]:
            kv.execute(
                "INSERT INTO access_history (node_type, node_id, accessed_at, source) "
                "VALUES ('episode', ?, ?, 'integ')",
                (ep_id, old),
            )
        kv.commit()
    finally:
        kv.close()

    # 似ベクトルを投入（cat ≒ 狗 ペアを cosine > 0.85 にする）
    rng = np.random.default_rng(42)
    base = rng.normal(size=768).astype(np.float32)
    base /= np.linalg.norm(base)
    similar = base + 0.01 * rng.normal(size=768).astype(np.float32)
    similar /= np.linalg.norm(similar)
    far_a = rng.normal(size=768).astype(np.float32)
    far_a /= np.linalg.norm(far_a)
    far_b = rng.normal(size=768).astype(np.float32)
    far_b /= np.linalg.norm(far_b)

    vdb = _open_vdb(tmp_project, char_id)
    try:
        upsert_entity_vector(vdb, entities[0], base)
        upsert_entity_vector(vdb, entities[1], similar)
        upsert_entity_vector(vdb, entities[2], far_a)
        upsert_entity_vector(vdb, entities[3], far_b)
    finally:
        vdb.close()

    return {"episodes": episodes, "entities": entities}


def test_full_pipeline_happy_path(tmp_project: Path) -> None:
    char_id = _make_character(tmp_project)
    now = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    seeded = _seed_dataset(tmp_project, char_id, now=now)

    result = run_compact(character_id=char_id, config=_config(char_id), now=now)

    # Step 1: cat→fish の fan_out が src ごとに再計算されてる（cat の out-degree=1）
    # mentions も含めてライブな src ごとに再計算される
    assert result.fan_out_updated >= 1
    # Step 2: cat→fish の strength は cooccurrence=2 → 1 + ln(2) ≈ 1.69 に更新
    assert result.strength_updated == 1
    # Step 3: ep_old1 / ep_old2 が抑制対象、ep_recent は対象外
    assert result.suppressed == 2
    # Step 4: cat ≒ 狗 ペアが merge_candidates に1件登録
    assert result.merge_candidates == 1
    assert result.dry_run is False
    assert result.duration_ms >= 0

    # DB に結果が反映されてること
    kv = _kv(tmp_project, char_id)
    try:
        suppressed_ids = [
            row[0]
            for row in kv.execute(
                "SELECT id FROM episodes WHERE is_suppressed = 1"
            ).fetchall()
        ]
        assert set(suppressed_ids) == set(seeded["episodes"][:2])

        mc_count = kv.execute(
            "SELECT COUNT(*) FROM merge_candidates WHERE resolved = 0"
        ).fetchone()[0]
        assert mc_count == 1
    finally:
        kv.close()


def test_full_pipeline_dry_run(tmp_project: Path) -> None:
    char_id = _make_character(tmp_project)
    now = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    _seed_dataset(tmp_project, char_id, now=now)

    result = run_compact(
        character_id=char_id, config=_config(char_id), now=now, dry_run=True
    )

    # 件数は happy path と同じ
    assert result.fan_out_updated >= 1
    assert result.strength_updated == 1
    assert result.suppressed == 2
    assert result.merge_candidates == 1
    assert result.dry_run is True

    # DB は不変であること
    kv = _kv(tmp_project, char_id)
    try:
        suppressed = kv.execute(
            "SELECT COUNT(*) FROM episodes WHERE is_suppressed = 1"
        ).fetchone()[0]
        mc = kv.execute("SELECT COUNT(*) FROM merge_candidates").fetchone()[0]
        assert suppressed == 0
        assert mc == 0
    finally:
        kv.close()


def test_pipeline_completes_under_one_second(tmp_project: Path) -> None:
    """やや多めのデータでも 1 秒以内（Phase 4 DoD「数秒で完了」のサニティ）。"""
    char_id = _make_character(tmp_project)
    now = datetime(2026, 4, 25, 12, 0, 0, tzinfo=UTC)
    valid_from = "2026-01-01T00:00:00+00:00"

    kv = _kv(tmp_project, char_id)
    try:
        for i in range(50):
            kv.execute(
                "INSERT INTO episodes (content, kind, importance, valid_from) "
                "VALUES (?, 'facts', 1, ?)",
                (f"ep-{i}", valid_from),
            )
        for i in range(50):
            kv.execute(
                "INSERT INTO entities (canonical_name, entity_type, valid_from) "
                "VALUES (?, 'concept', ?)",
                (f"概念-{i:02d}", valid_from),
            )
        kv.commit()
    finally:
        kv.close()

    rng = np.random.default_rng(0)
    vdb = _open_vdb(tmp_project, char_id)
    try:
        for ent_id in range(1, 51):
            v = rng.normal(size=768).astype(np.float32)
            v /= np.linalg.norm(v)
            upsert_entity_vector(vdb, ent_id, v)
    finally:
        vdb.close()

    result = run_compact(character_id=char_id, config=_config(char_id), now=now)

    assert result.duration_ms < 1000
