"""Unit tests for core/graph.py."""

import math
import sqlite3
from pathlib import Path

import pytest

from fravenir.core.graph import (
    build_subgraph_from_seeds,
    fan_out_of,
    reach_episodes,
    s_ji,
)
from fravenir.storage import sqlite_init


@pytest.fixture
def kv_conn(tmp_path: Path) -> sqlite3.Connection:
    kv_path = tmp_path / "kv.sqlite"
    sqlite_init.init_kv(kv_path)
    conn = sqlite3.connect(str(kv_path))
    yield conn
    conn.close()


def _insert_entity(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO entities (canonical_name, valid_from)
        VALUES (?, '2026-01-01T00:00:00+00:00')
        """,
        (name,),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _insert_episode(conn: sqlite3.Connection, content: str) -> int:
    cur = conn.execute(
        """
        INSERT INTO episodes (content, kind, valid_from)
        VALUES (?, 'facts', '2026-01-01T00:00:00+00:00')
        """,
        (content,),
    )
    return int(cur.lastrowid)  # type: ignore[arg-type]


def _add_rel(
    conn: sqlite3.Connection,
    src_type: str,
    src_id: int,
    dst_type: str,
    dst_id: int,
    predicate: str = "related",
    valid_to: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate,
                               valid_from, valid_to)
        VALUES (?, ?, ?, ?, ?, '2026-01-01T00:00:00+00:00', ?)
        """,
        (src_type, src_id, dst_type, dst_id, predicate, valid_to),
    )
    conn.commit()
    return int(cur.lastrowid)  # type: ignore[arg-type]


class TestSji:
    def test_zero_fan_returns_s_max(self) -> None:
        assert s_ji(0, 2.0) == 2.0

    def test_fan_one_returns_s_max(self) -> None:
        assert s_ji(1, 2.0) == 2.0  # ln(1) = 0

    def test_large_fan_decreases(self) -> None:
        small = s_ji(2, 2.0)
        large = s_ji(5, 2.0)
        assert small > large
        assert math.isclose(small, 2.0 - math.log(2))
        assert math.isclose(large, 2.0 - math.log(5))

    def test_clamped_to_zero(self) -> None:
        # fan so large that S_max - ln(fan) is negative → clamped to 0.0
        assert s_ji(1000, 2.0) == 0.0


class TestFanOutOf:
    def test_counts_only_valid(self, kv_conn: sqlite3.Connection) -> None:
        e1 = _insert_entity(kv_conn, "A")
        e2 = _insert_entity(kv_conn, "B")
        e3 = _insert_entity(kv_conn, "C")
        _add_rel(kv_conn, "entity", e1, "entity", e2)
        _add_rel(kv_conn, "entity", e1, "entity", e3)
        _add_rel(
            kv_conn, "entity", e1, "entity", e2,
            valid_to="2026-02-01T00:00:00+00:00",
        )
        assert fan_out_of(kv_conn, e1) == 2

    def test_episode_src_not_counted(self, kv_conn: sqlite3.Connection) -> None:
        e1 = _insert_entity(kv_conn, "A")
        ep = _insert_episode(kv_conn, "x")
        _add_rel(kv_conn, "episode", ep, "entity", e1, predicate="mentions")
        # e1 is dst, not src → fan_out(e1) == 0
        assert fan_out_of(kv_conn, e1) == 0


class TestBuildSubgraph:
    def test_two_hop_reach(self, kv_conn: sqlite3.Connection) -> None:
        # A -> B -> C; 2ホップで C まで到達
        a = _insert_entity(kv_conn, "A")
        b = _insert_entity(kv_conn, "B")
        c = _insert_entity(kv_conn, "C")
        _add_rel(kv_conn, "entity", a, "entity", b)
        _add_rel(kv_conn, "entity", b, "entity", c)

        graph = build_subgraph_from_seeds(kv_conn, [a], max_hops=2)
        assert ("entity", b) in graph.nodes
        assert ("entity", c) in graph.nodes

    def test_three_hop_not_reached(self, kv_conn: sqlite3.Connection) -> None:
        a = _insert_entity(kv_conn, "A")
        b = _insert_entity(kv_conn, "B")
        c = _insert_entity(kv_conn, "C")
        d = _insert_entity(kv_conn, "D")
        _add_rel(kv_conn, "entity", a, "entity", b)
        _add_rel(kv_conn, "entity", b, "entity", c)
        _add_rel(kv_conn, "entity", c, "entity", d)

        graph = build_subgraph_from_seeds(kv_conn, [a], max_hops=2)
        assert ("entity", d) not in graph.nodes

    def test_invalid_edges_skipped(self, kv_conn: sqlite3.Connection) -> None:
        a = _insert_entity(kv_conn, "A")
        b = _insert_entity(kv_conn, "B")
        _add_rel(
            kv_conn, "entity", a, "entity", b,
            valid_to="2026-02-01T00:00:00+00:00",
        )
        graph = build_subgraph_from_seeds(kv_conn, [a], max_hops=2)
        assert ("entity", b) not in graph.nodes

    def test_episode_is_terminal(self, kv_conn: sqlite3.Connection) -> None:
        # A -> ep -> X となっても ep を src とするエッジは辿らない設計
        a = _insert_entity(kv_conn, "A")
        ep = _insert_episode(kv_conn, "x")
        x = _insert_entity(kv_conn, "X")
        _add_rel(kv_conn, "entity", a, "episode", ep, predicate="evidences")
        _add_rel(kv_conn, "episode", ep, "entity", x, predicate="mentions")

        graph = build_subgraph_from_seeds(kv_conn, [a], max_hops=2)
        assert ("episode", ep) in graph.nodes
        assert ("entity", x) not in graph.nodes


class TestReachEpisodes:
    def test_collects_reached_episodes(self, kv_conn: sqlite3.Connection) -> None:
        a = _insert_entity(kv_conn, "A")
        b = _insert_entity(kv_conn, "B")
        ep1 = _insert_episode(kv_conn, "ep1")
        ep2 = _insert_episode(kv_conn, "ep2")
        _add_rel(kv_conn, "entity", a, "entity", b)
        _add_rel(kv_conn, "entity", a, "episode", ep1, predicate="evidences")
        _add_rel(kv_conn, "entity", b, "episode", ep2, predicate="evidences")

        graph = build_subgraph_from_seeds(kv_conn, [a], max_hops=2)
        episodes = reach_episodes(graph, a)
        assert episodes == {ep1, ep2}

    def test_unknown_seed_returns_empty(self, kv_conn: sqlite3.Connection) -> None:
        graph = build_subgraph_from_seeds(kv_conn, [], max_hops=2)
        assert reach_episodes(graph, 999) == set()
