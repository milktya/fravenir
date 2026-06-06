"""DDL execution for kv.sqlite and sqlite-vec vdb_memories.db."""

import sqlite3
from pathlib import Path

import sqlite_vec

_KV_DDL = """\
CREATE TABLE IF NOT EXISTS episodes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    content           TEXT    NOT NULL,
    kind              TEXT    NOT NULL,
    importance        INTEGER NOT NULL DEFAULT 1,
    valid_from        TIMESTAMP NOT NULL,
    valid_to          TIMESTAMP,
    supersedes        INTEGER REFERENCES episodes(id),
    -- derived_from: P5-0 で session_id と分離済。memory_compact による
    -- 圧縮エピソードからの派生記録に予約 (Phase 6 以降)。現在は未使用。
    derived_from      INTEGER REFERENCES episodes(id),
    session_id        TEXT,
    last_activated_at TIMESTAMP,
    activation_count  INTEGER NOT NULL DEFAULT 0,
    is_suppressed     INTEGER NOT NULL DEFAULT 0,
    created_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_episodes_valid_to   ON episodes(valid_to);
CREATE INDEX IF NOT EXISTS idx_episodes_kind       ON episodes(kind);
CREATE INDEX IF NOT EXISTS idx_episodes_supersedes ON episodes(supersedes);
CREATE INDEX IF NOT EXISTS idx_episodes_session_id ON episodes(session_id);

CREATE TABLE IF NOT EXISTS access_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_type   TEXT NOT NULL,
    node_id     INTEGER NOT NULL,
    accessed_at TIMESTAMP NOT NULL,
    source      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_access_history_node
    ON access_history(node_type, node_id, accessed_at);

CREATE TABLE IF NOT EXISTS entities (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT    NOT NULL,
    entity_type    TEXT,
    description    TEXT,
    is_self        INTEGER NOT NULL DEFAULT 0,
    self_weight    REAL    NOT NULL DEFAULT 0.0,
    decay_rate     REAL    NOT NULL DEFAULT 0.5,
    valid_from     TIMESTAMP NOT NULL,
    valid_to       TIMESTAMP,
    supersedes     INTEGER REFERENCES entities(id),
    last_activated_at TIMESTAMP,
    activation_count  INTEGER NOT NULL DEFAULT 0,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    -- Phase 6: AdminUI / seed.yaml 経由で人手 curated された description/aliases を
    -- 持つ entity のマーカー (NULL = 自動生成のまま、非NULL = 人手 curated 済み)。
    -- 自動側パイプラインが将来 description を上書きしうる経路を追加した際の保険。
    curated_at     TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_canonical_active
    ON entities(canonical_name) WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS idx_entities_is_self ON entities(is_self) WHERE is_self = 1;

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias     TEXT    NOT NULL,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (alias, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases(alias);

CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    src_type    TEXT    NOT NULL,
    src_id      INTEGER NOT NULL,
    dst_type    TEXT    NOT NULL,
    dst_id      INTEGER NOT NULL,
    predicate   TEXT    NOT NULL,
    strength    REAL    NOT NULL DEFAULT 1.0,
    fan_out     INTEGER NOT NULL DEFAULT 1,
    description TEXT,
    valid_from  TIMESTAMP NOT NULL,
    valid_to    TIMESTAMP,
    supersedes  INTEGER REFERENCES relations(id),
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_relations_src  ON relations(src_type, src_id, valid_to);
CREATE INDEX IF NOT EXISTS idx_relations_dst  ON relations(dst_type, dst_id, valid_to);
CREATE INDEX IF NOT EXISTS idx_relations_pred ON relations(predicate);

CREATE TABLE IF NOT EXISTS doc_status (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id INTEGER REFERENCES episodes(id),
    stage      TEXT NOT NULL,
    error      TEXT,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_doc_status_episode ON doc_status(episode_id);
CREATE INDEX IF NOT EXISTS idx_doc_status_stage   ON doc_status(stage);

CREATE TABLE IF NOT EXISTS merge_candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_a        INTEGER NOT NULL REFERENCES entities(id),
    entity_b        INTEGER NOT NULL REFERENCES entities(id),
    similarity      REAL    NOT NULL,
    detected_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved        INTEGER NOT NULL DEFAULT 0,
    judge_label     TEXT,
    judge_confidence TEXT,
    judge_reason    TEXT,
    judge_attempts  INTEGER NOT NULL DEFAULT 0,
    resolved_at     TIMESTAMP  -- merge() / reject() で記録、未解決は NULL
);
CREATE INDEX IF NOT EXISTS idx_merge_candidates_pair
    ON merge_candidates(entity_a, entity_b);
CREATE INDEX IF NOT EXISTS idx_merge_candidates_resolved
    ON merge_candidates(resolved);

CREATE TABLE IF NOT EXISTS admin_audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT    NOT NULL,
    target_type TEXT    NOT NULL,
    target_id   INTEGER NOT NULL,
    before_json TEXT,
    after_json  TEXT,
    actor       TEXT,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_admin_audit_target
    ON admin_audit_log(target_type, target_id, created_at);
CREATE INDEX IF NOT EXISTS idx_admin_audit_created
    ON admin_audit_log(created_at);
"""

# sqlite-vec virtual tables: 768 dimensions (ruri-v3-310m)
_VDB_DDL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS vdb_memories USING vec0(
    episode_id INTEGER PRIMARY KEY,
    embedding  FLOAT[768]
);
"""

_VDB_ENTITIES_DDL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS vdb_entities USING vec0(
    entity_id INTEGER PRIMARY KEY,
    embedding FLOAT[768]
);
"""

_VDB_RELATIONS_DDL = """\
CREATE VIRTUAL TABLE IF NOT EXISTS vdb_relations USING vec0(
    relation_id INTEGER PRIMARY KEY,
    embedding   FLOAT[768]
);
"""


def init_kv(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_KV_DDL)
        conn.commit()
    finally:
        conn.close()


def _init_vec_db(db_path: Path, ddl: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.executescript(ddl)
        conn.commit()
    finally:
        conn.close()


def init_vdb(db_path: Path) -> None:
    _init_vec_db(db_path, _VDB_DDL)


def init_vdb_entities(db_path: Path) -> None:
    _init_vec_db(db_path, _VDB_ENTITIES_DDL)


def init_vdb_relations(db_path: Path) -> None:
    _init_vec_db(db_path, _VDB_RELATIONS_DDL)
