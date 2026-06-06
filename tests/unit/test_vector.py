"""Unit tests for storage/vector.py."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import sqlite_vec

from fravenir.storage.vector import (
    l2_to_cosine,
    search_episodes_by_vector,
    upsert_entity_vector,
    upsert_episode_vector,
    upsert_relation_vector,
)

DIM = 4  # テスト用の小さい次元数


def _make_vdb(tmp_path: Path) -> sqlite3.Connection:
    db_path = str(tmp_path / "vdb.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vdb_memories USING vec0(
            episode_id INTEGER PRIMARY KEY,
            embedding  FLOAT[{DIM}]
        );
    """)
    return conn


def _unit_vec(values: list[float]) -> "np.ndarray[tuple[int], np.dtype[np.float32]]":
    v = np.array(values, dtype=np.float32)
    v = v / np.linalg.norm(v)
    return v


class TestUpsertEpisodeVector:
    def test_insert_and_retrievable(self, tmp_path: Path) -> None:
        conn = _make_vdb(tmp_path)
        vec = _unit_vec([1.0, 0.0, 0.0, 0.0])
        upsert_episode_vector(conn, episode_id=1, vector=vec)
        rows = conn.execute("SELECT episode_id FROM vdb_memories").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == 1

    def test_replace_on_duplicate_id(self, tmp_path: Path) -> None:
        conn = _make_vdb(tmp_path)
        vec1 = _unit_vec([1.0, 0.0, 0.0, 0.0])
        vec2 = _unit_vec([0.0, 1.0, 0.0, 0.0])
        upsert_episode_vector(conn, episode_id=1, vector=vec1)
        upsert_episode_vector(conn, episode_id=1, vector=vec2)
        rows = conn.execute("SELECT episode_id FROM vdb_memories").fetchall()
        assert len(rows) == 1  # 重複なし


class TestSearchEpisodesByVector:
    def test_returns_nearest(self, tmp_path: Path) -> None:
        conn = _make_vdb(tmp_path)
        # 3件: [1,0,0,0], [0,1,0,0], [0,0,1,0]
        for i, vals in enumerate([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], start=1):
            upsert_episode_vector(conn, i, _unit_vec([float(v) for v in vals]))

        query = _unit_vec([1.0, 0.1, 0.0, 0.0])
        results = search_episodes_by_vector(conn, query, top_k=3)
        assert len(results) == 3
        # episode_id=1 が最も近いはず
        assert results[0][0] == 1

    def test_top_k_limits_results(self, tmp_path: Path) -> None:
        conn = _make_vdb(tmp_path)
        for i in range(5):
            v = np.zeros(DIM, dtype=np.float32)
            v[i % DIM] = 1.0
            upsert_episode_vector(conn, i + 1, v)

        results = search_episodes_by_vector(conn, _unit_vec([1, 0, 0, 0]), top_k=2)
        assert len(results) == 2

    def test_empty_table_returns_empty(self, tmp_path: Path) -> None:
        conn = _make_vdb(tmp_path)
        results = search_episodes_by_vector(conn, _unit_vec([1, 0, 0, 0]), top_k=5)
        assert results == []


def _make_vdb_entities(tmp_path: Path) -> sqlite3.Connection:
    db_path = str(tmp_path / "vdb_entities.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vdb_entities USING vec0(
            entity_id INTEGER PRIMARY KEY,
            embedding FLOAT[{DIM}]
        );
    """)
    return conn


def _make_vdb_relations(tmp_path: Path) -> sqlite3.Connection:
    db_path = str(tmp_path / "vdb_relations.db")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vdb_relations USING vec0(
            relation_id INTEGER PRIMARY KEY,
            embedding   FLOAT[{DIM}]
        );
    """)
    return conn


class TestUpsertEntityVector:
    def test_insert_and_retrievable(self, tmp_path: Path) -> None:
        conn = _make_vdb_entities(tmp_path)
        upsert_entity_vector(conn, entity_id=1, vector=_unit_vec([1, 0, 0, 0]))
        rows = conn.execute("SELECT entity_id FROM vdb_entities").fetchall()
        assert rows == [(1,)]

    def test_replace_on_duplicate(self, tmp_path: Path) -> None:
        conn = _make_vdb_entities(tmp_path)
        upsert_entity_vector(conn, 1, _unit_vec([1, 0, 0, 0]))
        upsert_entity_vector(conn, 1, _unit_vec([0, 1, 0, 0]))
        rows = conn.execute("SELECT entity_id FROM vdb_entities").fetchall()
        assert len(rows) == 1


class TestUpsertRelationVector:
    def test_insert_and_retrievable(self, tmp_path: Path) -> None:
        conn = _make_vdb_relations(tmp_path)
        upsert_relation_vector(conn, relation_id=42, vector=_unit_vec([1, 0, 0, 0]))
        rows = conn.execute("SELECT relation_id FROM vdb_relations").fetchall()
        assert rows == [(42,)]

    def test_replace_on_duplicate(self, tmp_path: Path) -> None:
        conn = _make_vdb_relations(tmp_path)
        upsert_relation_vector(conn, 42, _unit_vec([1, 0, 0, 0]))
        upsert_relation_vector(conn, 42, _unit_vec([0, 1, 0, 0]))
        rows = conn.execute("SELECT relation_id FROM vdb_relations").fetchall()
        assert len(rows) == 1


class TestL2ToCosine:
    def test_identical_vectors_distance_zero(self) -> None:
        assert pytest.approx(l2_to_cosine(0.0), abs=1e-9) == 1.0

    def test_orthogonal_vectors(self) -> None:
        # 単位ベクトル同士の直交: L2 = sqrt(2), cosine = 0
        import math

        l2 = math.sqrt(2.0)
        assert pytest.approx(l2_to_cosine(l2), abs=1e-6) == 0.0

    def test_opposite_vectors(self) -> None:
        # 逆方向: L2 = 2, cosine = -1 → clamp → 0
        assert l2_to_cosine(2.0) == 0.0

    def test_partial_similarity(self) -> None:
        # 45度: cosine = cos(45°) ≈ 0.707, L2 = sqrt(2 - 2*0.707) ≈ 0.765
        import math

        angle = math.pi / 4
        expected_cos = math.cos(angle)
        l2 = math.sqrt(2 - 2 * expected_cos)
        assert pytest.approx(l2_to_cosine(l2), abs=1e-5) == expected_cos
