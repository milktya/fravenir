"""Episode supersedes-chain traversal."""

import sqlite3

from fravenir.schemas.config import AppConfig
from fravenir.storage import paths


def memory_trace(
    episode_id: int,
    *,
    character_id: str,
    config: AppConfig,
) -> dict[str, object]:
    """Traverse the supersedes chain upward from the given episode.

    Chain order: newest (starting point) → older versions.

    Returns:
        {
            "episode_id": int,
            "chain": [
                {"id": int, "content": str, "valid_from": str, "valid_to": str | None},
                ...
            ],
        }
    Raises:
        KeyError: episode_id not found
    """
    del config

    kv_path = paths.kv_db_path(character_id)
    kv_conn = sqlite3.connect(str(kv_path))
    try:
        chain: list[dict[str, object]] = []
        visited: set[int] = set()
        current_id: int | None = episode_id

        while current_id is not None:
            if current_id in visited:
                break
            visited.add(current_id)

            row = kv_conn.execute(
                """
                SELECT id, content, valid_from, valid_to, supersedes
                FROM episodes
                WHERE id = ?
                """,
                (current_id,),
            ).fetchone()

            if row is None:
                if not chain:
                    raise KeyError(episode_id)
                break

            ep_id, content, valid_from, valid_to, supersedes = row
            chain.append({
                "id": ep_id,
                "content": content,
                "valid_from": valid_from,
                "valid_to": valid_to,
            })
            current_id = supersedes
    finally:
        kv_conn.close()

    return {"episode_id": episode_id, "chain": chain}
