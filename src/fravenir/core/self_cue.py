"""Self cue detection for ACT-R self-boost (Phase 2).

Given a query, decide whether it refers to the character itself (via
aliases of is_self=1 entities or canonical_name of strong personality
entities). See design doc §5.4.
"""

import sqlite3


def self_cue_terms(conn: sqlite3.Connection, strong_threshold: float = 0.7) -> set[str]:
    """Return the set of terms that count as a self cue.

    Includes:
    - canonical_name of is_self=1 entities
    - all aliases pointing to is_self=1 entities
    - canonical_name of non-self entities whose self_weight >= strong_threshold
      (i.e. strong personality traits)

    Only entities with valid_to IS NULL are considered.
    """
    rows = conn.execute(
        """
        SELECT canonical_name FROM entities
        WHERE is_self = 1 AND valid_to IS NULL
        """
    ).fetchall()
    terms: set[str] = {r[0] for r in rows}

    rows = conn.execute(
        """
        SELECT ea.alias
        FROM entity_aliases ea
        JOIN entities e ON e.id = ea.entity_id
        WHERE e.is_self = 1 AND e.valid_to IS NULL
        """
    ).fetchall()
    terms.update(r[0] for r in rows)

    rows = conn.execute(
        """
        SELECT canonical_name FROM entities
        WHERE is_self = 0 AND self_weight >= ? AND valid_to IS NULL
        """,
        (strong_threshold,),
    ).fetchall()
    terms.update(r[0] for r in rows)

    return terms


def has_self_cue(
    conn: sqlite3.Connection, query: str, strong_threshold: float = 0.7
) -> bool:
    """Return True if any self-cue term appears in the query string."""
    terms = self_cue_terms(conn, strong_threshold)
    return any(t and t in query for t in terms)
