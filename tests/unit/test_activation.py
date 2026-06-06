"""Unit tests for core/activation.py."""

import math
import sqlite3
from datetime import datetime

import pytest

from fravenir.core.activation import base_activation, final_score


@pytest.fixture()
def conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.executescript("""
        CREATE TABLE access_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            node_type   TEXT NOT NULL,
            node_id     INTEGER NOT NULL,
            accessed_at TEXT NOT NULL,
            source      TEXT NOT NULL
        );
    """)
    return c


def insert_access(
    conn: sqlite3.Connection,
    node_type: str,
    node_id: int,
    accessed_at: str,
    source: str = "direct",
) -> None:
    conn.execute(
        "INSERT INTO access_history(node_type, node_id, accessed_at, source) VALUES (?,?,?,?)",
        (node_type, node_id, accessed_at, source),
    )
    conn.commit()


class TestBaseActivation:
    def test_no_history_returns_zero(self, conn: sqlite3.Connection) -> None:
        result = base_activation(conn, "episode", 1, decay=0.5, now=datetime(2026, 4, 21, 12, 0, 0))
        assert result == 0.0

    def test_single_access_one_second_ago(self, conn: sqlite3.Connection) -> None:
        now = datetime(2026, 4, 21, 12, 0, 0)
        insert_access(conn, "episode", 1, "2026-04-21T11:59:59")
        result = base_activation(conn, "episode", 1, decay=0.5, now=now)
        # t=1秒 → max(1, 1)=1, total = 1^(-0.5) = 1.0, log(1.0) = 0.0
        assert pytest.approx(result, abs=1e-9) == 0.0

    def test_single_access_100_seconds_ago(self, conn: sqlite3.Connection) -> None:
        now = datetime(2026, 4, 21, 12, 0, 0)
        insert_access(conn, "episode", 1, "2026-04-21T11:58:20")
        result = base_activation(conn, "episode", 1, decay=0.5, now=now)
        # t=100秒, total = 100^(-0.5) = 0.1, log(0.1) ≈ -2.3026
        expected = math.log(100 ** (-0.5))
        assert pytest.approx(result, abs=1e-6) == expected

    def test_self_decay_lower_than_default(self, conn: sqlite3.Connection) -> None:
        now = datetime(2026, 4, 21, 12, 0, 0)
        insert_access(conn, "entity", 5, "2026-04-21T11:58:20")
        result_self = base_activation(conn, "entity", 5, decay=0.2, now=now)
        result_normal = base_activation(conn, "entity", 5, decay=0.5, now=now)
        # 低decayほど時間減衰が弱い → 過去アクセスが高い
        assert result_self > result_normal

    def test_multiple_accesses_sum(self, conn: sqlite3.Connection) -> None:
        now = datetime(2026, 4, 21, 12, 0, 0)
        insert_access(conn, "episode", 2, "2026-04-21T11:59:59")  # 1秒前
        insert_access(conn, "episode", 2, "2026-04-21T11:58:20")  # 100秒前
        result = base_activation(conn, "episode", 2, decay=0.5, now=now)
        # total = 1^(-0.5) + 100^(-0.5) = 1.0 + 0.1 = 1.1
        expected = math.log(1.0 + 100 ** (-0.5))
        assert pytest.approx(result, abs=1e-6) == expected

    def test_limit_caps_history(self, conn: sqlite3.Connection) -> None:
        now = datetime(2026, 4, 21, 12, 0, 0)
        # 5件挿入して limit=2 で最新2件だけ使われることを確認
        for i in range(5):
            insert_access(conn, "episode", 3, f"2026-04-21T11:59:{59 - i:02d}")
        result_lim2 = base_activation(conn, "episode", 3, decay=0.5, now=now, limit=2)
        result_lim5 = base_activation(conn, "episode", 3, decay=0.5, now=now, limit=5)
        assert result_lim2 < result_lim5

    def test_node_isolation(self, conn: sqlite3.Connection) -> None:
        now = datetime(2026, 4, 21, 12, 0, 0)
        insert_access(conn, "episode", 10, "2026-04-21T11:58:20")
        assert base_activation(conn, "episode", 99, decay=0.5, now=now) == 0.0


class TestFinalScore:
    def test_basic_sum(self) -> None:
        score = final_score(
            activation=1.0,
            cosine_sim=0.8,
            importance=2,
            alpha_sim=1.0,
            alpha_imp=0.3,
        )
        # 1.0 + 1.0*0.8 + 0.3*2 = 1.0 + 0.8 + 0.6 = 2.4
        assert pytest.approx(score, abs=1e-9) == 2.4

    def test_zero_activation(self) -> None:
        score = final_score(0.0, 0.9, 1, 1.0, 0.3)
        # 0 + 0.9 + 0.3 = 1.2
        assert pytest.approx(score, abs=1e-9) == 1.2

    def test_high_importance_boost(self) -> None:
        low = final_score(0.0, 0.5, 1, 1.0, 0.3)
        high = final_score(0.0, 0.5, 3, 1.0, 0.3)
        assert high > low
