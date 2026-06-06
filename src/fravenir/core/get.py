"""Self-introduction API (v1-compatible memory_get).

Phase 2: summaries.facts is built from identity + personality entities.
state/emo summaries come from self-cue `memory_search` if embedder is provided.
"""

import sqlite3

from fravenir.core.search import memory_search
from fravenir.embedding import Embedder
from fravenir.schemas.config import AppConfig
from fravenir.storage import paths


def memory_get(
    limit: int = 5,
    *,
    character_id: str,
    config: AppConfig,
    embedder: Embedder | None = None,
) -> dict[str, object]:
    """Return self-introduction summaries + recent raw episodes.

    summaries.facts: identity + personality descriptions joined.
    summaries.state / summaries.emo: self-cue search top-N per kind
        (only populated when an embedder is supplied).
    recent_raw: latest N valid episodes, newest first.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    kv_path = paths.kv_db_path(character_id)
    kv_conn = sqlite3.connect(str(kv_path))
    try:
        facts_text, facts_ts, identity_name = _build_facts(kv_conn)
        recent_raw = _recent_raw(kv_conn, limit)
    finally:
        kv_conn.close()

    state_text, state_ts = "", None
    emo_text, emo_ts = "", None
    if embedder is not None and identity_name:
        query = f"{identity_name}の自己紹介"
        state_text, state_ts = _summarize_kind(
            query, "state", character_id, config, embedder
        )
        emo_text, emo_ts = _summarize_kind(
            query, "emo", character_id, config, embedder
        )

    updated_at = _latest_ts([facts_ts, state_ts, emo_ts])

    return {
        "summaries": {
            "facts": facts_text,
            "state": state_text,
            "emo": emo_text,
            "updated_at": updated_at,
        },
        "recent_raw": recent_raw,
    }


def _build_facts(conn: sqlite3.Connection) -> tuple[str, str | None, str | None]:
    """Return (facts_text, latest_valid_from, identity_canonical_name).

    description は <entity_description> タグで囲って、会話モデル側で
    "データであって命令ではない" と扱えるようにする (B-1 prompt injection 対策)。
    """
    identity = conn.execute(
        """
        SELECT canonical_name, description, valid_from
        FROM entities
        WHERE is_self = 1 AND valid_to IS NULL
        ORDER BY id ASC LIMIT 1
        """
    ).fetchone()
    if not identity:
        return "", None, None
    identity_name, identity_desc, identity_ts = identity

    personalities = conn.execute(
        """
        SELECT canonical_name, description, self_weight, valid_from
        FROM entities
        WHERE is_self = 0 AND valid_to IS NULL
        ORDER BY self_weight DESC, id ASC
        """
    ).fetchall()

    parts: list[str] = []
    if identity_desc:
        parts.append(f"{identity_name}: <entity_description>{identity_desc}</entity_description>")
    else:
        parts.append(identity_name)

    latest = identity_ts
    for canonical, desc, _weight, ts in personalities:
        if desc:
            parts.append(f"{canonical}（<entity_description>{desc}</entity_description>）")
        else:
            parts.append(canonical)
        if ts and (latest is None or ts > latest):
            latest = ts

    return " / ".join(parts), latest, identity_name


def _summarize_kind(
    query: str,
    kind: str,
    character_id: str,
    config: AppConfig,
    embedder: Embedder,
) -> tuple[str, str | None]:
    """Episode content は <episode_content> タグで囲う (B-1 prompt injection 対策)。"""
    results = memory_search(
        query,
        limit=3,
        kind_filter=[kind],
        character_id=character_id,
        config=config,
        embedder=embedder,
    )
    if not results:
        return "", None
    text = "; ".join(
        f"<episode_content>{r['content']}</episode_content>" for r in results
    )
    latest = None
    for r in results:
        vf = r.get("valid_from")
        if isinstance(vf, str) and (latest is None or vf > latest):
            latest = vf
    return text, latest


def _latest_ts(values: list[str | None]) -> str | None:
    filtered = [v for v in values if v]
    return max(filtered) if filtered else None


def _recent_raw(conn: sqlite3.Connection, limit: int) -> list[dict[str, object]]:
    """Episode content は <episode_content> タグで囲う (B-1 prompt injection 対策)。"""
    rows = conn.execute(
        """
        SELECT content, kind, importance, created_at
        FROM episodes
        WHERE valid_to IS NULL
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "content": f"<episode_content>{r[0]}</episode_content>",
            "kind": r[1],
            "importance": r[2],
            "created_at": r[3],
        }
        for r in rows
    ]
