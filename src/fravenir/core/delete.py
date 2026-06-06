"""Episode logical deletion via valid_to."""

import sqlite3
from datetime import UTC, datetime

import structlog

from fravenir.schemas.config import AppConfig
from fravenir.storage import paths

logger = structlog.get_logger()


def memory_delete(
    episode_id: int,
    reason: str,
    *,
    character_id: str,
    config: AppConfig,
) -> dict[str, object]:
    """Logically delete an episode by setting valid_to=now.

    Returns:
        {"episode_id": int, "valid_to": str}
    Raises:
        ValueError: reason is empty or episode already deleted
        KeyError: episode_id not found
    """
    del config

    if not reason.strip():
        raise ValueError("reason must not be empty")

    kv_path = paths.kv_db_path(character_id)
    kv_conn = sqlite3.connect(str(kv_path))
    try:
        row = kv_conn.execute(
            "SELECT valid_to FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        if row[0] is not None:
            raise ValueError(f"episode {episode_id} already deleted at {row[0]}")

        now = datetime.now(UTC)
        now_iso = now.isoformat()
        kv_conn.execute(
            "UPDATE episodes SET valid_to = ? WHERE id = ?",
            (now_iso, episode_id),
        )
        kv_conn.execute(
            "UPDATE relations SET valid_to = ? "
            "WHERE src_type = 'episode' AND src_id = ? AND valid_to IS NULL",
            (now_iso, episode_id),
        )
        kv_conn.commit()
    finally:
        kv_conn.close()

    logger.info(
        "episode_deleted",
        episode_id=episode_id,
        reason=reason,
        character_id=character_id,
    )

    return {"episode_id": episode_id, "valid_to": now.isoformat()}
