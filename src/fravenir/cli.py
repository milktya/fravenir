"""CLI entry point for fravenir."""

from __future__ import annotations

import base64
import json
import re
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import click
import structlog
import yaml

from fravenir.embedding import Embedder
from fravenir.schemas.config import AppConfig
from fravenir.schemas.seed import InitialEpisode, SeedConfig
from fravenir.storage.paths import (
    character_dir,
    config_yaml_path,
    data_dir,
    data_root,
    kv_db_path,
    seed_yaml_path,
    vdb_entities_path,
    vdb_memories_path,
    vdb_relations_path,
)
from fravenir.storage.sqlite_init import (
    init_kv,
    init_vdb,
    init_vdb_entities,
    init_vdb_relations,
)
from fravenir.storage.vector import upsert_entity_vector, upsert_episode_vector

log = structlog.get_logger()

_CHARACTER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_character_id(
    ctx: click.Context | None,
    param: click.Parameter | None,
    value: str,
) -> str:
    if not _CHARACTER_ID_PATTERN.match(value):
        raise click.BadParameter(
            f"must match {_CHARACTER_ID_PATTERN.pattern} "
            "(slug characters only, max 64 chars)",
            ctx=ctx,
            param=param,
        )
    return value


def _configure_logging(fmt: str, level: str) -> None:
    import logging

    logging.basicConfig(level=getattr(logging, level, logging.INFO))
    processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    if fmt == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _default_seed_yaml(character_id: str) -> str:
    return f"""\
identity:
  canonical_name: {character_id}
  aliases: []
  description: ""

# personality items are registered as entities with part_of relations to identity (Phase 2+)
personality: []

initial_episodes:
  - content: "あたしは{character_id}。記憶を積み重ねていく。"
    kind: facts
    importance: 3
"""


def _default_config_yaml(character_id: str) -> str:
    return f"""\
character:
  id: {character_id}
  system_prompt_template: |
    あなたは{character_id}です。
    （キャラクター設定と memory_get で得た記憶を組み合わせて使う）

embedding:
  model: cl-nagoya/ruri-v3-310m
  dim: 768
  max_tokens: 8192
  device: auto
  batch_size: 32
  normalize: true
  prefixes:
    general: ""
    topic: "トピック: "
    query: "検索クエリ: "
    document: "検索文書: "

act_r:
  base_decay: 0.5
  self_decay: 0.2
  personality_decay: 0.3
  self_boost_beta: 0.5
  s_max: 2.0
  access_history_limit: 100
  suppress_threshold: -2.0
  alpha_similarity: 1.0
  alpha_importance: 0.3

session:
  auto_timeout_minutes: 15

logging:
  level: INFO
  format: json
  activation_debug: false

compact:
  schedule: "0 3 * * *"
  dry_run_default: false
  suppress_recent_access_days: 7

extraction:
  enabled: true
  base_url: http://127.0.0.1:8080/v1
  model: unsloth/gemma-4-E2B-it-GGUF
  api_key: dummy
  timeout: 30.0
  max_retries: 3
  temperature: 0.0

semantic_judge:
  enabled: false                # P5-4 LLM 意味判定（既定 off）
  base_url: http://127.0.0.1:8080/v1
  model: Gemma4-31B
  api_key: dummy
  timeout: 60.0
  max_retries: 2
  max_attempts: 3
  temperature: 0.0
  min_strength: 0.3             # P5-5: 逆方向 relation 検出の strength 足切り

server:
  transport: stdio              # stdio | streamable-http | sse
  host: 127.0.0.1               # HTTP transports only. Set to Tailscale IP on server.
  port: 8280
"""


def _register_seed(
    conn: sqlite3.Connection,
    seed: SeedConfig,
    *,
    character_id: str | None = None,
    embedder: Embedder | None = None,
) -> dict[str, int]:
    now = datetime.now(UTC).isoformat()
    counts: dict[str, int] = {
        "entities": 0,
        "aliases": 0,
        "episodes": 0,
        "relations": 0,
    }
    new_entities: list[tuple[int, str, str]] = []  # (id, canonical_name, description)
    new_episodes: list[tuple[int, str]] = []  # (id, content)

    # identity entity (is_self=1) — seed.yaml 由来なので curated_at を立てる
    identity = seed.identity
    conn.execute(
        """
        INSERT INTO entities
            (canonical_name, entity_type, description,
             is_self, self_weight, decay_rate, valid_from, curated_at)
        VALUES (?, 'person', ?, 1, 1.0, 0.2, ?, ?)
        """,
        (identity.canonical_name, identity.description, now, now),
    )
    identity_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    counts["entities"] += 1
    new_entities.append((identity_id, identity.canonical_name, identity.description or ""))

    for alias in identity.aliases:
        conn.execute(
            "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
            (alias, identity_id),
        )
        counts["aliases"] += 1

    # personality entities (Phase 2: decay_rate=0.3, part_of -> identity)
    for p in seed.personality:
        conn.execute(
            """
            INSERT INTO entities
                (canonical_name, entity_type, description,
                 is_self, self_weight, decay_rate, valid_from, curated_at)
            VALUES (?, ?, ?, 0, ?, 0.3, ?, ?)
            """,
            (p.canonical_name, p.entity_type, p.description, p.self_weight, now, now),
        )
        p_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        counts["entities"] += 1
        new_entities.append((p_id, p.canonical_name, p.description or ""))
        conn.execute(
            """
            INSERT INTO relations
                (src_type, src_id, dst_type, dst_id, predicate, valid_from)
            VALUES ('entity', ?, 'entity', ?, 'part_of', ?)
            """,
            (p_id, identity_id, now),
        )
        counts["relations"] += 1

    # Phase 6: seed entities (重要固有名詞)。curated_at を立てて投入。
    # personality と違い identity への part_of は張らない (関係の付け方は
    # 利用側に委ねる)。alias もここで展開する。
    for se in seed.seed_entities:
        conn.execute(
            """
            INSERT INTO entities
                (canonical_name, entity_type, description,
                 is_self, self_weight, decay_rate, valid_from, curated_at)
            VALUES (?, ?, ?, 0, ?, ?, ?, ?)
            """,
            (
                se.canonical_name,
                se.entity_type,
                se.description,
                se.self_weight,
                se.decay_rate,
                now,
                now,
            ),
        )
        se_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        counts["entities"] += 1
        new_entities.append((se_id, se.canonical_name, se.description or ""))
        for alias in se.aliases:
            conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
                (alias, se_id),
            )
            counts["aliases"] += 1

    # initial episodes + mentions relation to identity
    for ep in seed.initial_episodes:
        cur = conn.execute(
            """
            INSERT INTO episodes (content, kind, importance, valid_from)
            VALUES (?, ?, ?, ?)
            """,
            (ep.content, ep.kind, ep.importance, now),
        )
        counts["episodes"] += 1
        ep_id: int = cur.lastrowid  # type: ignore[assignment]
        new_episodes.append((ep_id, ep.content))
        conn.execute(
            """
            INSERT INTO relations
                (src_type, src_id, dst_type, dst_id, predicate, valid_from)
            VALUES ('episode', ?, 'entity', ?, 'mentions', ?)
            """,
            (ep_id, identity_id, now),
        )
        counts["relations"] += 1

    if embedder is not None and character_id is not None:
        if new_entities:
            _write_seed_entity_vectors(character_id, new_entities, embedder)
        if new_episodes:
            _write_seed_episode_vectors(character_id, new_episodes, embedder)

    return counts


def _write_seed_entity_vectors(
    character_id: str,
    new_entities: list[tuple[int, str, str]],
    embedder: Embedder,
) -> None:
    import sqlite_vec

    vdb = vdb_entities_path(character_id)
    conn = sqlite3.connect(str(vdb))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        for eid, name, desc in new_entities:
            text = f"{name} {desc}".strip()
            vec = embedder.encode_topic(text)
            upsert_entity_vector(conn, eid, vec)
    finally:
        conn.close()


def _apply_initial_episodes_via_write(
    *,
    character_id: str,
    episodes: list[InitialEpisode],
    app_cfg: AppConfig,
    embedder: Embedder,
) -> tuple[int, int]:
    """initial_episodes を memory_write 経由で投入し extraction も走らせる。

    Returns:
        (episodes_added, relations_added) - 重複スキップ後の実投入件数と、
        memory_write 全体（mentions / entity-entity 双方）の relation 増加分。
    """
    from typing import Literal, cast

    from fravenir.core.extraction import ExtractionClient
    from fravenir.core.write import memory_write

    kv = kv_db_path(character_id)
    extraction_client = ExtractionClient(app_cfg.extraction)

    relations_before = _count_relations(kv)
    episodes_added = 0
    for ep in episodes:
        check_conn = sqlite3.connect(kv)
        try:
            exists = check_conn.execute(
                "SELECT id FROM episodes WHERE content = ? LIMIT 1",
                (ep.content,),
            ).fetchone()
        finally:
            check_conn.close()
        if exists:
            continue
        memory_write(
            content=ep.content,
            kind=cast(Literal["facts", "state", "emo"], ep.kind),
            importance=ep.importance,
            session_id=None,
            character_id=character_id,
            config=app_cfg,
            embedder=embedder,
            extraction_client=extraction_client,
        )
        episodes_added += 1

    relations_after = _count_relations(kv)
    return episodes_added, relations_after - relations_before


def _count_relations(kv_path: Path) -> int:
    conn = sqlite3.connect(kv_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM relations").fetchone()
    finally:
        conn.close()
    return int(row[0]) if row else 0


def _write_seed_episode_vectors(
    character_id: str,
    new_episodes: list[tuple[int, str]],
    embedder: Embedder,
) -> None:
    import sqlite_vec

    vdb = vdb_memories_path(character_id)
    conn = sqlite3.connect(str(vdb))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        for ep_id, content in new_episodes:
            vec = embedder.encode_document(content)
            upsert_episode_vector(conn, ep_id, vec)
    finally:
        conn.close()


@click.group()
def main() -> None:
    """fravenir: character memory MCP server."""


@main.command("create-character")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.option("--seed", "seed_path", default=None, help="Path to seed.yaml")
def create_character(character_id: str, config_path: str | None, seed_path: str | None) -> None:
    """Create a new character and initialize its data directory."""
    _configure_logging("console", "INFO")
    logger = structlog.get_logger().bind(character_id=character_id)

    # 1. Fail if data/<id>/ already exists
    d = data_dir(character_id)
    if d.exists():
        click.echo(f"Error: data/{character_id}/ already exists.", err=True)
        sys.exit(1)

    # 2. Resolve config path and validate
    cfg_path = Path(config_path) if config_path else config_yaml_path(character_id)
    char_dir = character_dir(character_id)
    char_dir.mkdir(parents=True, exist_ok=True)

    if not cfg_path.exists():
        cfg_path.write_text(_default_config_yaml(character_id), encoding="utf-8")
        logger.info("config.yaml generated", path=str(cfg_path))

    with cfg_path.open(encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    # inject character.id in case default template was used without editing
    if isinstance(raw_cfg, dict):
        raw_cfg.setdefault("character", {})["id"] = character_id
    else:
        raw_cfg = {"character": {"id": character_id}}
    app_cfg = AppConfig.model_validate(raw_cfg)

    # 3. Resolve seed path and validate
    s_path = Path(seed_path) if seed_path else seed_yaml_path(character_id)
    if not s_path.exists():
        s_path.write_text(_default_seed_yaml(character_id), encoding="utf-8")
        logger.info("seed.yaml generated", path=str(s_path))

    with s_path.open(encoding="utf-8") as f:
        raw_seed = yaml.safe_load(f)
    seed = SeedConfig.model_validate(raw_seed)

    # 4. Create data/<id>/ and init databases
    d.mkdir(parents=True)
    kv = kv_db_path(character_id)
    vdb = vdb_memories_path(character_id)
    vdb_ent = vdb_entities_path(character_id)
    vdb_rel = vdb_relations_path(character_id)
    init_kv(kv)
    init_vdb(vdb)
    init_vdb_entities(vdb_ent)
    init_vdb_relations(vdb_rel)
    logger.info(
        "databases initialized",
        kv=str(kv), vdb=str(vdb),
        vdb_entities=str(vdb_ent), vdb_relations=str(vdb_rel),
    )

    # 5. Register seed data (+ seed entity embeddings into vdb_entities)
    embedder = Embedder(app_cfg.embedding)
    conn = sqlite3.connect(kv)
    try:
        counts = _register_seed(
            conn, seed, character_id=character_id, embedder=embedder,
        )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "character created",
        entities=counts["entities"],
        aliases=counts["aliases"],
        episodes=counts["episodes"],
        relations=counts["relations"],
    )
    click.echo(
        f"✓ Created character '{character_id}': "
        f"{counts['entities']} entities, "
        f"{counts['aliases']} aliases, "
        f"{counts['episodes']} episodes, "
        f"{counts['relations']} relations"
    )


def _require_data_dir(character_id: str) -> Path:
    d = data_dir(character_id)
    if not d.exists():
        click.echo(f"Error: character '{character_id}' not found (data/{character_id}/ missing).",
                   err=True)
        sys.exit(1)
    return d


def _kv_stats(character_id: str) -> dict[str, Any]:
    kv = kv_db_path(character_id)
    conn = sqlite3.connect(kv)
    try:
        self_row = conn.execute(
            "SELECT canonical_name FROM entities WHERE is_self = 1 LIMIT 1"
        ).fetchone()
        aliases = [
            row[0]
            for row in conn.execute(
                "SELECT ea.alias FROM entity_aliases ea "
                "JOIN entities e ON ea.entity_id = e.id WHERE e.is_self = 1"
            ).fetchall()
        ]
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        episode_count = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE valid_to IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    return {
        "identity": self_row[0] if self_row else "?",
        "aliases": aliases,
        "entities": entity_count,
        "episodes": episode_count,
    }


@main.command("list-characters")
def list_characters() -> None:
    """List all characters with initialized data directories."""
    _configure_logging("console", "WARNING")
    root = data_root()
    if not root.exists():
        click.echo("No characters found.")
        return
    rows = [d for d in sorted(root.iterdir()) if d.is_dir() and (d / "kv.sqlite").exists()]
    if not rows:
        click.echo("No characters found.")
        return
    for d in rows:
        try:
            s = _kv_stats(d.name)
            aliases_str = ", ".join(s["aliases"]) if s["aliases"] else "-"
            click.echo(
                f"  {d.name:<20} identity={s['identity']!r:<16} "
                f"aliases={aliases_str:<16} entities={s['entities']}  episodes={s['episodes']}"
            )
        except (sqlite3.Error, OSError) as exc:
            _log = structlog.get_logger(__name__)
            _log.exception("list_characters_load_error", character_id=d.name, error=str(exc))
            click.echo(f"  {d.name:<20} (error reading stats: {type(exc).__name__}: {exc})")


@main.command("show-character")
@click.argument("character_id", callback=_validate_character_id)
def show_character(character_id: str) -> None:
    """Show details of a character."""
    _configure_logging("console", "WARNING")
    _require_data_dir(character_id)
    s = _kv_stats(character_id)
    aliases_str = ", ".join(s["aliases"]) if s["aliases"] else "(none)"
    click.echo(f"Character : {character_id}")
    click.echo(f"  Identity : {s['identity']}")
    click.echo(f"  Aliases  : {aliases_str}")
    click.echo(f"  Entities : {s['entities']}")
    click.echo(f"  Episodes : {s['episodes']} (active)")
    click.echo(f"  Data dir : {data_dir(character_id)}")


@main.command("delete-character")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--force", is_flag=True, default=False, help="Skip confirmation prompt.")
def delete_character(character_id: str, force: bool) -> None:
    """Delete a character's data directory."""
    _configure_logging("console", "WARNING")
    d = _require_data_dir(character_id)
    if not force:
        click.confirm(
            f"Delete data/{character_id}/ and all its data? This cannot be undone.",
            abort=True,
        )
    shutil.rmtree(d)
    click.echo(f"✓ Deleted data/{character_id}/")


@main.command("init-character")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--seed", "seed_path", default=None, help="Path to seed.yaml")
@click.option("--force", is_flag=True, default=False, help="Apply seed to existing data dir.")
@click.option(
    "--extract-episodes/--no-extract-episodes",
    "extract_episodes",
    default=True,
    help="initial_episodes を memory_write 経由で投入し LLM extraction を走らせるか。"
    " デフォルト ON。OFF の場合は doc_status 行を作らず identity への mentions"
    " 関係のみ手動付与する従来挙動。",
)
def init_character(
    character_id: str,
    seed_path: str | None,
    force: bool,
    extract_episodes: bool,
) -> None:
    """Re-apply seed.yaml to an existing data directory (diff-apply, non-destructive)."""
    _configure_logging("console", "INFO")
    logger = structlog.get_logger().bind(character_id=character_id)
    _require_data_dir(character_id)

    if not force:
        click.confirm(
            f"Re-apply seed to data/{character_id}/? "
            "Existing entities with the same canonical_name will be skipped.",
            abort=True,
        )

    s_path = Path(seed_path) if seed_path else seed_yaml_path(character_id)
    if not s_path.exists():
        click.echo(f"Error: seed.yaml not found at {s_path}", err=True)
        sys.exit(1)

    with s_path.open(encoding="utf-8") as f:
        raw_seed = yaml.safe_load(f)
    seed = SeedConfig.model_validate(raw_seed)

    # Ensure vdb_entities/relations DBs exist (idempotent for existing characters)
    init_vdb_entities(vdb_entities_path(character_id))
    init_vdb_relations(vdb_relations_path(character_id))

    # Load config for embedder (needed for seed entity vectors)
    cfg_path = config_yaml_path(character_id)
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        if isinstance(raw_cfg, dict):
            raw_cfg.setdefault("character", {})["id"] = character_id
        else:
            raw_cfg = {"character": {"id": character_id}}
        app_cfg = AppConfig.model_validate(raw_cfg)
    else:
        app_cfg = AppConfig.model_validate({"character": {"id": character_id}})
    embedder = Embedder(app_cfg.embedding)

    kv = kv_db_path(character_id)
    conn = sqlite3.connect(kv)
    added: dict[str, int] = {
        "entities": 0,
        "aliases": 0,
        "episodes": 0,
        "relations": 0,
    }
    new_entities_for_vdb: list[tuple[int, str, str]] = []
    try:
        now = datetime.now(UTC).isoformat()
        identity = seed.identity
        # INSERT OR IGNORE relies on idx_entities_canonical_active
        # (partial UNIQUE WHERE valid_to IS NULL)
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO entities
                (canonical_name, entity_type, description,
                 is_self, self_weight, decay_rate, valid_from, curated_at)
            VALUES (?, 'person', ?, 1, 1.0, 0.2, ?, ?)
            """,
            (identity.canonical_name, identity.description, now, now),
        )
        if cur.rowcount:
            added["entities"] += 1
            identity_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            new_entities_for_vdb.append(
                (identity_id, identity.canonical_name, identity.description or "")
            )
        else:
            identity_id = conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ? AND valid_to IS NULL",
                (identity.canonical_name,),
            ).fetchone()[0]

        for alias in identity.aliases:
            cur2 = conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
                (alias, identity_id),
            )
            if cur2.rowcount:
                added["aliases"] += 1

        for p in seed.personality:
            cur3 = conn.execute(
                """
                INSERT OR IGNORE INTO entities
                    (canonical_name, entity_type, description,
                     is_self, self_weight, decay_rate, valid_from, curated_at)
                VALUES (?, ?, ?, 0, ?, 0.3, ?, ?)
                """,
                (p.canonical_name, p.entity_type, p.description, p.self_weight, now, now),
            )
            if cur3.rowcount:
                added["entities"] += 1
                p_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                new_entities_for_vdb.append(
                    (p_id, p.canonical_name, p.description or "")
                )
            else:
                p_id = conn.execute(
                    "SELECT id FROM entities WHERE canonical_name = ? AND valid_to IS NULL",
                    (p.canonical_name,),
                ).fetchone()[0]
            rel_exists = conn.execute(
                """
                SELECT 1 FROM relations
                WHERE src_type='entity' AND src_id=?
                  AND dst_type='entity' AND dst_id=?
                  AND predicate='part_of' AND valid_to IS NULL
                LIMIT 1
                """,
                (p_id, identity_id),
            ).fetchone()
            if not rel_exists:
                conn.execute(
                    """
                    INSERT INTO relations
                        (src_type, src_id, dst_type, dst_id, predicate, valid_from)
                    VALUES ('entity', ?, 'entity', ?, 'part_of', ?)
                    """,
                    (p_id, identity_id, now),
                )
                added["relations"] += 1

        # Phase 6: seed entities (重要固有名詞)
        for se in seed.seed_entities:
            cur_se = conn.execute(
                """
                INSERT OR IGNORE INTO entities
                    (canonical_name, entity_type, description,
                     is_self, self_weight, decay_rate, valid_from, curated_at)
                VALUES (?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    se.canonical_name,
                    se.entity_type,
                    se.description,
                    se.self_weight,
                    se.decay_rate,
                    now,
                    now,
                ),
            )
            if cur_se.rowcount:
                added["entities"] += 1
                se_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                new_entities_for_vdb.append(
                    (se_id, se.canonical_name, se.description or "")
                )
            else:
                se_id = conn.execute(
                    "SELECT id FROM entities WHERE canonical_name = ? AND valid_to IS NULL",
                    (se.canonical_name,),
                ).fetchone()[0]
            for alias in se.aliases:
                cur_se_alias = conn.execute(
                    "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
                    (alias, se_id),
                )
                if cur_se_alias.rowcount:
                    added["aliases"] += 1

        if not extract_episodes:
            for ep in seed.initial_episodes:
                exists = conn.execute(
                    "SELECT id FROM episodes WHERE content = ? LIMIT 1",
                    (ep.content,),
                ).fetchone()
                if exists:
                    continue
                cur4 = conn.execute(
                    "INSERT INTO episodes (content, kind, importance, valid_from)"
                    " VALUES (?, ?, ?, ?)",
                    (ep.content, ep.kind, ep.importance, now),
                )
                added["episodes"] += 1
                conn.execute(
                    """
                    INSERT INTO relations
                        (src_type, src_id, dst_type, dst_id, predicate, valid_from)
                    VALUES ('episode', ?, 'entity', ?, 'mentions', ?)
                    """,
                    (cur4.lastrowid, identity_id, now),
                )
                added["relations"] += 1

        conn.commit()
    finally:
        conn.close()

    if new_entities_for_vdb:
        _write_seed_entity_vectors(character_id, new_entities_for_vdb, embedder)

    if extract_episodes and seed.initial_episodes:
        added_episodes, added_via_extract = _apply_initial_episodes_via_write(
            character_id=character_id,
            episodes=list(seed.initial_episodes),
            app_cfg=app_cfg,
            embedder=embedder,
        )
        added["episodes"] += added_episodes
        added["relations"] += added_via_extract

    logger.info("init-character done", **added)
    click.echo(
        f"✓ init-character '{character_id}': "
        f"added {added['entities']} entities, "
        f"{added['aliases']} aliases, {added['episodes']} episodes, "
        f"{added['relations']} relations"
    )


@main.command("compact")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--dry-run", is_flag=True, default=False)
@click.option(
    "--use-llm",
    is_flag=True,
    default=False,
    help="Run LLM-based semantic judgment on merge_candidates after Step 4.",
)
@click.option(
    "--report",
    "report_path",
    default=None,
    help="Write detailed JSON report to this path (use-llm only).",
)
def compact(
    character_id: str,
    dry_run: bool,
    use_llm: bool,
    report_path: str | None,
) -> None:
    """Run the nightly compact pipeline.

    With --use-llm, also runs the Phase 5 P5-4 semantic judgment pass on
    unresolved merge_candidates (gated by config.semantic_judge.enabled).
    """
    _configure_logging("console", "WARNING")
    _require_data_dir(character_id)

    cfg_path = config_yaml_path(character_id)
    if not cfg_path.exists():
        click.echo(f"Error: config.yaml not found at {cfg_path}", err=True)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if isinstance(raw_cfg, dict):
        raw_cfg.setdefault("character", {})["id"] = character_id
    else:
        raw_cfg = {"character": {"id": character_id}}
    config = AppConfig.model_validate(raw_cfg)

    from fravenir.core.compact import run_compact

    result = run_compact(
        character_id=character_id,
        config=config,
        dry_run=dry_run,
        use_llm=use_llm,
    )
    click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

    if use_llm and report_path and result.judgment is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        report: dict[str, Any] = {
            "judgment": result.judgment.to_report_dict(),
        }
        if result.direction_judgment is not None:
            report["direction_conflicts"] = result.direction_judgment.to_report_dict()
        if result.contradiction_judgment is not None:
            report["claim_contradictions"] = result.contradiction_judgment.to_report_dict()
        Path(report_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"✓ Detailed report written to {report_path}")


@main.command("retry-extraction")
@click.argument("character_id", callback=_validate_character_id)
@click.option(
    "--episode-id",
    "-e",
    "episode_ids",
    multiple=True,
    type=int,
    help="特定の episode_id を再抽出（複数指定可）。",
)
@click.option(
    "--all",
    "all_failed",
    is_flag=True,
    default=False,
    help="stage=embedded かつ error 付きの全件を対象。",
)
@click.option(
    "--include-pending",
    is_flag=True,
    default=False,
    help="--all 時、doc_status 行が無い / stage が done 以外の episode も対象に含める。"
    " init-character の --no-extract-episodes 投入分や中断中の救済に使う。",
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="--all 時の最大件数（暴走防止、デフォルト 50）。",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="対象一覧の表示のみ、再抽出は実行しない。",
)
def retry_extraction_cmd(
    character_id: str,
    episode_ids: tuple[int, ...],
    all_failed: bool,
    include_pending: bool,
    limit: int,
    dry_run: bool,
) -> None:
    """エンティティ抽出に失敗したエピソードを再抽出する。

    --episode-id か --all のどちらか必須。--dry-run と組み合わせると
    対象一覧のみ確認できる。--include-pending は --all との併用時のみ有効。
    """
    if not episode_ids and not all_failed:
        click.echo(
            "Error: --episode-id か --all のどちらかを指定してください", err=True
        )
        sys.exit(1)
    if episode_ids and all_failed:
        click.echo(
            "Error: --episode-id と --all は同時指定できません", err=True
        )
        sys.exit(1)
    if include_pending and not all_failed:
        click.echo(
            "Error: --include-pending は --all との併用時のみ有効です", err=True
        )
        sys.exit(1)

    _configure_logging("console", "WARNING")
    _require_data_dir(character_id)

    cfg_path = config_yaml_path(character_id)
    if not cfg_path.exists():
        click.echo(f"Error: config.yaml not found at {cfg_path}", err=True)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if isinstance(raw_cfg, dict):
        raw_cfg.setdefault("character", {})["id"] = character_id
    else:
        raw_cfg = {"character": {"id": character_id}}
    config = AppConfig.model_validate(raw_cfg)

    from fravenir.core.extraction import ExtractionClient
    from fravenir.core.retry_extraction import (
        list_failed_episodes,
        retry_extraction,
    )

    targets = list_failed_episodes(
        character_id=character_id,
        limit=limit if all_failed else None,
        episode_ids=list(episode_ids) if episode_ids else None,
        include_pending=include_pending,
    )
    if not targets:
        click.echo("対象エピソードなし。Failed doc_status は空です。")
        return

    if dry_run:
        click.echo(f"対象 {len(targets)} 件 (dry-run、再抽出なし):")
        for ep in targets:
            click.echo(f"  - id={ep.episode_id} kind={ep.kind}")
            click.echo(f"      error: {ep.error[:120]}")
        return

    embedder = Embedder(config.embedding)
    extraction_client = ExtractionClient(config.extraction)

    click.echo(f"再抽出開始: {len(targets)} 件")
    result = retry_extraction(
        targets,
        character_id=character_id,
        extraction_client=extraction_client,
        embedder=embedder,
    )
    click.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@main.group("migrate")
def migrate_group() -> None:
    """Database migration commands."""


@migrate_group.command("session-id")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def migrate_session_id_cmd(character_id: str, dry_run: bool, yes: bool) -> None:
    """Move legacy derived_from text values into the session_id column."""
    from fravenir.migrations.session_id import migrate

    _require_data_dir(character_id)
    db = kv_db_path(character_id)

    preview = migrate(db, dry_run=True)
    click.echo(f"Character: {character_id}  ({db})")
    if preview.added_session_id_column:
        click.echo("  - episodes.session_id column will be added")
    if preview.added_indexes:
        click.echo(f"  - indexes to add: {', '.join(preview.added_indexes)}")
    click.echo(f"  - rows to migrate: {preview.migrated_rows}")

    nothing_to_do = (
        not preview.added_session_id_column
        and not preview.added_indexes
        and preview.migrated_rows == 0
    )
    if dry_run:
        click.echo("(dry-run; no changes applied)")
        return
    if nothing_to_do:
        click.echo("Nothing to do.")
        return
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return

    result = migrate(db, dry_run=False)
    click.echo(
        f"✓ Migrated {result.migrated_rows} episodes "
        f"(column_added={result.added_session_id_column}, "
        f"indexes_added={len(result.added_indexes)})"
    )


@main.group("resolve")
def resolve_group() -> None:
    """Resolve merge_candidates manually."""


@resolve_group.command("list")
@click.argument("character_id", callback=_validate_character_id)
def resolve_list_cmd(character_id: str) -> None:
    """List unresolved merge_candidates."""
    from fravenir.core.resolve import list_candidates

    _require_data_dir(character_id)
    db = kv_db_path(character_id)
    rows = list_candidates(db)
    if not rows:
        click.echo("No unresolved merge_candidates.")
        return
    click.echo(f"Character: {character_id}  ({db})")
    click.echo(
        f"  {'id':>4}  {'a_id':>5}  {'b_id':>5}  {'sim':>5}  "
        f"{'type':<10}  {'a_name':<20}  {'b_name':<20}  "
        f"{'judge':<10}  {'conf':<6}  {'try':>3}"
    )
    for r in rows:
        type_label = r.a_type or "-"
        judge_label = r.judge_label or "-"
        confidence = r.judge_confidence or "-"
        click.echo(
            f"  {r.candidate_id:>4}  {r.entity_a:>5}  {r.entity_b:>5}  "
            f"{r.similarity:>5.3f}  {type_label:<10}  "
            f"{r.a_name:<20}  {r.b_name:<20}  "
            f"{judge_label:<10}  {confidence:<6}  {r.judge_attempts:>3}"
        )


@resolve_group.command("merge")
@click.argument("character_id", callback=_validate_character_id)
@click.argument("candidate_id", type=int)
@click.option("--keep", "keep_id", type=int, default=None,
              help="Entity id to keep (default: smaller id of the pair).")
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def resolve_merge_cmd(
    character_id: str,
    candidate_id: int,
    keep_id: int | None,
    dry_run: bool,
    yes: bool,
) -> None:
    """Merge a merge_candidate (drop side gets valid_to + supersedes)."""
    from fravenir.core.resolve import ResolveError, merge

    _require_data_dir(character_id)
    db = kv_db_path(character_id)

    # Always show a dry-run preview first
    try:
        preview = merge(db, candidate_id, keep=keep_id, dry_run=True)
    except KeyError:
        click.echo(f"Error: candidate {candidate_id} not found.", err=True)
        sys.exit(1)
    except ResolveError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Character: {character_id}  ({db})")
    click.echo(f"  candidate     : {preview.candidate_id}")
    click.echo(f"  keep          : {preview.keep_id}")
    click.echo(f"  drop          : {preview.drop_id}")
    click.echo(f"  relations(keep after merge): {preview.relations_rewired}")
    click.echo(f"  self-loops archived        : {preview.self_loops_archived}")
    click.echo(f"  aliases added              : {preview.aliases_added}")

    if dry_run:
        click.echo("(dry-run; no changes applied)")
        return
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return

    result = merge(db, candidate_id, keep=keep_id, dry_run=False)
    click.echo(
        f"✓ Merged candidate {result.candidate_id}: "
        f"keep={result.keep_id}, drop={result.drop_id}, "
        f"self_loops={result.self_loops_archived}, "
        f"aliases_added={result.aliases_added}"
    )


@resolve_group.command("reject")
@click.argument("character_id", callback=_validate_character_id)
@click.argument("candidate_id", type=int)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def resolve_reject_cmd(character_id: str, candidate_id: int, yes: bool) -> None:
    """Reject a merge_candidate (resolved=2 only)."""
    from fravenir.core.resolve import ResolveError, reject

    _require_data_dir(character_id)
    db = kv_db_path(character_id)

    if not yes and not click.confirm(
        f"Reject candidate {candidate_id} for '{character_id}'?", default=False
    ):
        click.echo("Aborted.")
        return

    try:
        result = reject(db, candidate_id, dry_run=False)
    except KeyError:
        click.echo(f"Error: candidate {candidate_id} not found.", err=True)
        sys.exit(1)
    except ResolveError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"✓ Rejected candidate {result.candidate_id}")


@migrate_group.command("judge-columns")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def migrate_judge_columns_cmd(character_id: str, dry_run: bool, yes: bool) -> None:
    """Add judge_* columns to merge_candidates (Phase 5 P5-4)."""
    from fravenir.migrations.judge_columns import migrate

    _require_data_dir(character_id)
    db = kv_db_path(character_id)

    preview = migrate(db, dry_run=True)
    click.echo(f"Character: {character_id}  ({db})")
    if preview.added_columns:
        click.echo(f"  - columns to add: {', '.join(preview.added_columns)}")
    else:
        click.echo("  - no columns to add (already migrated)")

    if dry_run:
        click.echo("(dry-run; no changes applied)")
        return
    if not preview.added_columns:
        click.echo("Nothing to do.")
        return
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return

    result = migrate(db, dry_run=False)
    click.echo(f"✓ Added columns: {', '.join(result.added_columns)}")


@migrate_group.command("resolved-at")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def migrate_resolved_at_cmd(character_id: str, dry_run: bool, yes: bool) -> None:
    """Add resolved_at column to merge_candidates (P5-7 audit trail)."""
    from fravenir.migrations.resolved_at import migrate

    _require_data_dir(character_id)
    db = kv_db_path(character_id)

    preview = migrate(db, dry_run=True)
    click.echo(f"Character: {character_id}  ({db})")
    if preview.added_columns:
        click.echo(f"  - columns to add: {', '.join(preview.added_columns)}")
    else:
        click.echo("  - no columns to add (already migrated)")

    if dry_run:
        click.echo("(dry-run; no changes applied)")
        return
    if not preview.added_columns:
        click.echo("Nothing to do.")
        return
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return

    result = migrate(db, dry_run=False)
    click.echo(f"✓ Added columns: {', '.join(result.added_columns)}")



@migrate_group.command("curated-and-audit")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def migrate_curated_and_audit_cmd(character_id: str, dry_run: bool, yes: bool) -> None:
    """Add entities.curated_at + admin_audit_log table (Phase 6 AdminUI edit)."""
    from fravenir.migrations.curated_and_audit import migrate

    _require_data_dir(character_id)
    db = kv_db_path(character_id)

    preview = migrate(db, dry_run=True)
    click.echo(f"Character: {character_id}  ({db})")
    if preview.added_columns:
        click.echo(f"  - columns to add: {', '.join(preview.added_columns)}")
    if preview.created_tables:
        click.echo(f"  - tables to create: {', '.join(preview.created_tables)}")
    if not preview.added_columns and not preview.created_tables:
        click.echo("  - nothing to do (already migrated)")

    if dry_run:
        click.echo("(dry-run; no changes applied)")
        return
    if not preview.added_columns and not preview.created_tables:
        return
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        return

    result = migrate(db, dry_run=False)
    parts: list[str] = []
    if result.added_columns:
        parts.append(f"columns={','.join(result.added_columns)}")
    if result.created_tables:
        parts.append(f"tables={','.join(result.created_tables)}")
    click.echo(f"✓ Applied ({' / '.join(parts)})")


@main.command("serve")
@click.option("--character", "character_id", required=True, help="Character id to serve.")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "streamable-http", "sse"]),
    default=None,
    help="Transport to use. Overrides config.server.transport.",
)
@click.option(
    "--host",
    default=None,
    help="Host to bind (HTTP transports only). Overrides config.server.host.",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind (HTTP transports only). Overrides config.server.port.",
)
def serve(
    character_id: str,
    transport: str | None,
    host: str | None,
    port: int | None,
) -> None:
    """Start the MCP server for a character.

    Transport / host / port resolution order: CLI > config.server > defaults.
    HTTP transports bind via FastMCP settings (host / port).
    """
    from fravenir.server import build_server

    _configure_logging("json", "INFO")
    _require_data_dir(character_id)

    cfg_path = config_yaml_path(character_id)
    if not cfg_path.exists():
        click.echo(f"Error: config.yaml not found at {cfg_path}", err=True)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if isinstance(raw_cfg, dict):
        raw_cfg.setdefault("character", {})["id"] = character_id
    else:
        raw_cfg = {"character": {"id": character_id}}
    config = AppConfig.model_validate(raw_cfg)

    effective_transport = transport or config.server.transport
    effective_host = host or config.server.host
    effective_port = port if port is not None else config.server.port

    if effective_transport == "stdio":
        server = build_server(config)
        log.info("mcp server starting", character_id=character_id, transport="stdio")
        server.run(transport="stdio")
    else:
        # Pass host/port at FastMCP construction so DNS-rebinding protection
        # is auto-selected correctly (on for localhost, off otherwise).
        server = build_server(config, host=effective_host, port=effective_port)
        log.info(
            "mcp server starting",
            character_id=character_id,
            transport=effective_transport,
            host=effective_host,
            port=effective_port,
        )
        server.run(transport=cast(Literal["sse", "streamable-http"], effective_transport))


@main.command("admin-serve")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--host", default="127.0.0.1", help="Bind host.")
@click.option("--port", default=8281, type=int, help="Bind port.")
def admin_serve(character_id: str, host: str, port: int) -> None:
    """Run the admin UI server."""
    _require_data_dir(character_id)

    import uvicorn

    from fravenir.admin.server import create_app

    app = create_app(character_id)
    uvicorn.run(app, host=host, port=port)


@main.command("export")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--out", "out_path", required=True, help="Output .json file path.")
def export_character(character_id: str, out_path: str) -> None:
    """Export character data to a JSON file (embeddings included as base64)."""
    _configure_logging("console", "INFO")
    logger = structlog.get_logger().bind(character_id=character_id)
    _require_data_dir(character_id)

    kv = kv_db_path(character_id)
    vdb = vdb_memories_path(character_id)

    kv_conn = sqlite3.connect(kv)
    kv_conn.row_factory = sqlite3.Row
    try:
        entities = [dict(r) for r in kv_conn.execute("SELECT * FROM entities").fetchall()]
        entity_aliases = [
            dict(r) for r in kv_conn.execute("SELECT * FROM entity_aliases").fetchall()
        ]
        episodes = [dict(r) for r in kv_conn.execute("SELECT * FROM episodes").fetchall()]
        relations = [dict(r) for r in kv_conn.execute("SELECT * FROM relations").fetchall()]
    finally:
        kv_conn.close()

    episode_vectors: dict[str, str] = {}
    if vdb.exists():
        import sqlite_vec

        vdb_conn = sqlite3.connect(vdb)
        try:
            vdb_conn.enable_load_extension(True)
            sqlite_vec.load(vdb_conn)
            vdb_conn.enable_load_extension(False)
            rows = vdb_conn.execute("SELECT episode_id, embedding FROM vdb_memories").fetchall()
            for row in rows:
                episode_vectors[str(row[0])] = base64.b64encode(row[1]).decode()
        finally:
            vdb_conn.close()

    payload = {
        "format_version": "1",
        "model": "cl-nagoya/ruri-v3-310m",
        "dim": 768,
        "character_id": character_id,
        "exported_at": datetime.now(UTC).isoformat(),
        "entities": entities,
        "entity_aliases": entity_aliases,
        "episodes": episodes,
        "episode_vectors": episode_vectors,
        "relations": relations,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("export done", path=str(out))
    click.echo(
        f"✓ Exported '{character_id}' to {out}  "
        f"(episodes={len(episodes)}, vectors={len(episode_vectors)})"
    )


@main.command("import")
@click.argument("json_path")
@click.argument("character_id", callback=_validate_character_id)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing data dir.")
def import_character(json_path: str, character_id: str, overwrite: bool) -> None:
    """Import character data from a JSON export file."""
    _configure_logging("console", "INFO")
    logger = structlog.get_logger().bind(character_id=character_id)

    src = Path(json_path)
    if not src.exists():
        click.echo(f"Error: {json_path} not found.", err=True)
        sys.exit(1)

    payload = json.loads(src.read_text(encoding="utf-8"))
    if payload.get("format_version") != "1":
        click.echo("Error: unsupported format_version.", err=True)
        sys.exit(1)
    export_dim = payload.get("dim")
    if export_dim != 768:
        click.echo(f"Error: dim mismatch (export={export_dim}, expected=768).", err=True)
        sys.exit(1)

    d = data_dir(character_id)
    if d.exists():
        if not overwrite:
            click.echo(
                f"Error: data/{character_id}/ already exists. Use --overwrite to replace.", err=True
            )
            sys.exit(1)
        shutil.rmtree(d)

    d.mkdir(parents=True)
    kv = kv_db_path(character_id)
    vdb = vdb_memories_path(character_id)
    init_kv(kv)
    init_vdb(vdb)

    kv_conn = sqlite3.connect(kv)
    try:
        for row in payload.get("entities", []):
            kv_conn.execute(
                """INSERT INTO entities
                   (id, canonical_name, entity_type, description,
                    is_self, self_weight, decay_rate, valid_from, valid_to,
                    supersedes, last_activated_at, activation_count, created_at)
                   VALUES (:id, :canonical_name, :entity_type, :description,
                    :is_self, :self_weight, :decay_rate, :valid_from, :valid_to,
                    :supersedes, :last_activated_at, :activation_count, :created_at)""",
                row,
            )
        for row in payload.get("entity_aliases", []):
            kv_conn.execute(
                "INSERT OR IGNORE INTO entity_aliases (alias, entity_id)"
                " VALUES (:alias, :entity_id)",
                row,
            )
        for row in payload.get("episodes", []):
            df = row.get("derived_from")
            if isinstance(df, str):
                row.setdefault("session_id", df)
                row["derived_from"] = None
            row.setdefault("session_id", None)
            kv_conn.execute(
                """INSERT INTO episodes
                   (id, content, kind, importance, valid_from, valid_to,
                    supersedes, derived_from, session_id,
                    last_activated_at, activation_count,
                    is_suppressed, created_at)
                   VALUES (:id, :content, :kind, :importance, :valid_from, :valid_to,
                    :supersedes, :derived_from, :session_id,
                    :last_activated_at, :activation_count,
                    :is_suppressed, :created_at)""",
                row,
            )
        for row in payload.get("relations", []):
            # self-ref ガード: 古い export に残っていた src==dst の relation を弾く。
            # write.py 挿入時ガードが整備される前のデータを再投入しないため。
            if (
                row.get("src_type") == row.get("dst_type")
                and row.get("src_id") == row.get("dst_id")
            ):
                logger.warning(
                    "import_skip_self_ref_relation",
                    relation_id=row.get("id"),
                    src_type=row.get("src_type"),
                    src_id=row.get("src_id"),
                    predicate=row.get("predicate"),
                )
                continue
            kv_conn.execute(
                """INSERT INTO relations
                   (id, src_type, src_id, dst_type, dst_id, predicate,
                    strength, fan_out, description, valid_from, valid_to,
                    supersedes, created_at)
                   VALUES (:id, :src_type, :src_id, :dst_type, :dst_id, :predicate,
                    :strength, :fan_out, :description, :valid_from, :valid_to,
                    :supersedes, :created_at)""",
                row,
            )
        kv_conn.commit()
    finally:
        kv_conn.close()

    vectors = payload.get("episode_vectors", {})
    if vectors:
        import sqlite_vec

        vdb_conn = sqlite3.connect(vdb)
        try:
            vdb_conn.enable_load_extension(True)
            sqlite_vec.load(vdb_conn)
            vdb_conn.enable_load_extension(False)
            for ep_id_str, b64 in vectors.items():
                vdb_conn.execute(
                    "INSERT INTO vdb_memories (episode_id, embedding) VALUES (?, ?)",
                    (int(ep_id_str), base64.b64decode(b64)),
                )
            vdb_conn.commit()
        finally:
            vdb_conn.close()

    logger.info("import done", episodes=len(payload.get("episodes", [])), vectors=len(vectors))
    click.echo(
        f"✓ Imported '{character_id}' from {src}  "
        f"(episodes={len(payload.get('episodes', []))}, vectors={len(vectors)})"
    )
