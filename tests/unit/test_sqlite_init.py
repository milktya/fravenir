"""Unit tests for SQLite DDL execution."""

import sqlite3

import sqlite_vec

from fravenir.storage.sqlite_init import (
    init_kv,
    init_vdb,
    init_vdb_entities,
    init_vdb_relations,
)


def test_kv_tables_created(tmp_path):
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    conn = sqlite3.connect(db)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    expected = {
        "episodes", "access_history", "entities", "entity_aliases",
        "relations", "doc_status", "merge_candidates",
    }
    assert expected <= tables


def test_kv_indexes_created(tmp_path):
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    conn = sqlite3.connect(db)
    indexes = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    conn.close()
    assert "idx_episodes_valid_to" in indexes
    assert "idx_access_history_node" in indexes
    assert "idx_entities_canonical_active" in indexes


def test_kv_idempotent(tmp_path):
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    # calling twice should not raise (IF NOT EXISTS guards)
    init_kv(db)


def test_vdb_virtual_table_created(tmp_path):
    db = tmp_path / "vdb.db"
    init_vdb(db)
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()
    assert "vdb_memories" in tables


def test_vdb_idempotent(tmp_path):
    db = tmp_path / "vdb.db"
    init_vdb(db)
    init_vdb(db)


def test_vdb_dimension_is_768(tmp_path):
    db = tmp_path / "vdb.db"
    init_vdb(db)
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='vdb_memories'"
    ).fetchone()[0]
    conn.close()
    assert "FLOAT[768]" in sql


def test_vdb_entities_virtual_table_created(tmp_path):
    db = tmp_path / "vdb_entities.db"
    init_vdb_entities(db)
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='vdb_entities'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert "FLOAT[768]" in row[0]
    assert "entity_id" in row[0]


def test_vdb_relations_virtual_table_created(tmp_path):
    db = tmp_path / "vdb_relations.db"
    init_vdb_relations(db)
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='vdb_relations'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert "FLOAT[768]" in row[0]
    assert "relation_id" in row[0]


def test_vdb_entities_idempotent(tmp_path):
    db = tmp_path / "vdb_entities.db"
    init_vdb_entities(db)
    init_vdb_entities(db)


def test_vdb_relations_idempotent(tmp_path):
    db = tmp_path / "vdb_relations.db"
    init_vdb_relations(db)
    init_vdb_relations(db)
