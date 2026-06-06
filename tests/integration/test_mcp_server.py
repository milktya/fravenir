"""MCP サーバー統合テスト: FastMCP in-memory client で6本のツール疎通確認。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent

from fravenir.schemas.config import AppConfig, CharacterConfig, ExtractionConfig
from fravenir.server import build_server
from fravenir.storage import sqlite_init

DIM = 768


def _hash_vec(text: str) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    rng = np.random.default_rng(hash(text) % (2**32))
    v = rng.standard_normal(DIM).astype(np.float32)
    v = v / np.linalg.norm(v)
    return v


def _stub_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.encode_document.side_effect = lambda t: _hash_vec(t)
    embedder.encode_query.side_effect = lambda t: _hash_vec(t)
    embedder.encode_topic.side_effect = lambda t: _hash_vec(t)
    return embedder


def _init_character(tmp_project: Path, char_id: str) -> AppConfig:
    d = tmp_project / "data" / char_id
    d.mkdir(parents=True)
    sqlite_init.init_kv(d / "kv.sqlite")
    sqlite_init.init_vdb(d / "vdb_memories.db")
    sqlite_init.init_vdb_entities(d / "vdb_entities.db")
    sqlite_init.init_vdb_relations(d / "vdb_relations.db")
    return AppConfig(
        character=CharacterConfig(id=char_id),
        extraction=ExtractionConfig(enabled=False),
    )


def _parse(result: object) -> object:
    """CallToolResult の structuredContent / テキストを扱いやすい形に整形する。"""
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        # FastMCPは dict 戻り値を {"result": ...} にラップすることがある
        if set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    content = getattr(result, "content", None)
    if content and isinstance(content[0], TextContent):
        return json.loads(content[0].text)
    return None


@pytest.mark.anyio
async def test_mcp_list_tools(tmp_project: Path) -> None:
    config = _init_character(tmp_project, "mcp_smoke")
    server = build_server(config, embedder=_stub_embedder())

    async with create_connected_server_and_client_session(server) as session:
        listed = await session.list_tools()

    names = sorted(t.name for t in listed.tools)
    assert names == [
        "memory_compact",
        "memory_delete",
        "memory_explore",
        "memory_get",
        "memory_search",
        "memory_trace",
        "memory_write",
    ]


@pytest.mark.anyio
async def test_mcp_write_search_roundtrip(tmp_project: Path) -> None:
    config = _init_character(tmp_project, "mcp_rw")
    server = build_server(config, embedder=_stub_embedder())

    async with create_connected_server_and_client_session(server) as session:
        written = _parse(
            await session.call_tool(
                "memory_write",
                {"content": "あたしは記憶を覚えていく", "kind": "facts", "importance": 2},
            )
        )
        assert isinstance(written, dict)
        assert "episode_id" in written

        searched = _parse(
            await session.call_tool(
                "memory_search", {"query": "あたしは記憶を覚えていく", "limit": 3}
            )
        )
        assert isinstance(searched, list)
        assert searched[0]["episode_id"] == written["episode_id"]


@pytest.mark.anyio
async def test_mcp_get_delete_trace(tmp_project: Path) -> None:
    config = _init_character(tmp_project, "mcp_gdt")
    server = build_server(config, embedder=_stub_embedder())

    async with create_connected_server_and_client_session(server) as session:
        written = _parse(
            await session.call_tool(
                "memory_write", {"content": "削除して辿るよ", "kind": "facts"}
            )
        )
        assert isinstance(written, dict)
        ep_id = int(written["episode_id"])

        got = _parse(await session.call_tool("memory_get", {"limit": 5}))
        assert isinstance(got, dict)
        assert "summaries" in got and "recent_raw" in got
        # B-1: content は <episode_content> タグで囲まれる
        assert any(
            r["content"] == "<episode_content>削除して辿るよ</episode_content>"
            for r in got["recent_raw"]
        )

        deleted = _parse(
            await session.call_tool(
                "memory_delete", {"episode_id": ep_id, "reason": "integration test"}
            )
        )
        assert isinstance(deleted, dict)
        assert deleted["episode_id"] == ep_id
        assert deleted["valid_to"] is not None

        traced = _parse(await session.call_tool("memory_trace", {"episode_id": ep_id}))
        assert isinstance(traced, dict)
        assert traced["episode_id"] == ep_id
        assert isinstance(traced["chain"], list) and len(traced["chain"]) == 1


@pytest.mark.anyio
async def test_mcp_search_include_archived_roundtrip(tmp_project: Path) -> None:
    config = _init_character(tmp_project, "mcp_archived")
    server = build_server(config, embedder=_stub_embedder())

    async with create_connected_server_and_client_session(server) as session:
        written = _parse(
            await session.call_tool(
                "memory_write",
                {"content": "アーカイブ対象の記憶", "kind": "facts"},
            )
        )
        assert isinstance(written, dict)
        ep_id = int(written["episode_id"])

        deleted = _parse(
            await session.call_tool(
                "memory_delete",
                {"episode_id": ep_id, "reason": "archive integration test"},
            )
        )
        assert isinstance(deleted, dict)
        assert deleted["valid_to"] is not None

        excluded = _parse(
            await session.call_tool(
                "memory_search", {"query": "アーカイブ対象の記憶", "limit": 5}
            )
        )
        assert isinstance(excluded, list)
        assert all(item["episode_id"] != ep_id for item in excluded)

        included = _parse(
            await session.call_tool(
                "memory_search",
                {
                    "query": "アーカイブ対象の記憶",
                    "limit": 5,
                    "include_archived": True,
                },
            )
        )
        assert isinstance(included, list)
        assert any(item["episode_id"] == ep_id for item in included)


@pytest.mark.anyio
async def test_mcp_compact_stub(tmp_project: Path) -> None:
    config = _init_character(tmp_project, "mcp_compact")
    server = build_server(config, embedder=_stub_embedder())

    async with create_connected_server_and_client_session(server) as session:
        result = _parse(await session.call_tool("memory_compact", {"dry_run": True}))

    assert isinstance(result, dict)
    assert result["fan_out_updated"] == 0
    assert result["suppressed"] == 0
    assert result["dry_run"] is True


@pytest.mark.anyio
async def test_mcp_explore_episode_node(tmp_project: Path) -> None:
    """memory_explore で episode 起点の探索が MCP 経由で動く。"""
    config = _init_character(tmp_project, "mcp_explore_ep")
    server = build_server(config, embedder=_stub_embedder())

    async with create_connected_server_and_client_session(server) as session:
        written = _parse(
            await session.call_tool(
                "memory_write",
                {"content": "今日の探索テスト", "kind": "facts"},
            )
        )
        assert isinstance(written, dict)
        ep_id = int(written["episode_id"])

        explored = _parse(
            await session.call_tool(
                "memory_explore",
                {"node_type": "episode", "node_id": ep_id},
            )
        )

    assert isinstance(explored, dict)
    assert explored["node"]["type"] == "episode"
    assert explored["node"]["id"] == ep_id
    assert "今日の探索テスト" in explored["node"]["content"]
    assert "neighbors" in explored
    assert "meta" in explored
    assert explored["total_neighbors"] == 0


@pytest.mark.anyio
async def test_mcp_explore_entity_with_neighbors(tmp_project: Path) -> None:
    """entity 起点 + 双方向 neighbor が MCP 経由で返る。"""
    import sqlite3
    from datetime import UTC, datetime

    config = _init_character(tmp_project, "mcp_explore_ent")

    kv = sqlite3.connect(
        str(tmp_project / "data" / "mcp_explore_ent" / "kv.sqlite"),
    )
    now = datetime.now(UTC).isoformat()
    cur = kv.execute(
        """INSERT INTO entities
            (canonical_name, entity_type, description, is_self, self_weight,
             decay_rate, valid_from)
        VALUES ('mina', 'person', '自己ハブ', 1, 1.0, 0.2, ?)""",
        (now,),
    )
    mina_id = cur.lastrowid
    cur = kv.execute(
        """INSERT INTO entities
            (canonical_name, entity_type, description, is_self, self_weight,
             decay_rate, valid_from)
        VALUES ('好奇心旺盛', 'concept', '気質', 0, 0.8, 0.3, ?)""",
        (now,),
    )
    trait_id = cur.lastrowid
    kv.execute(
        """INSERT INTO relations
            (src_type, src_id, dst_type, dst_id, predicate, strength, valid_from)
        VALUES ('entity', ?, 'entity', ?, 'part_of', 1.0, ?)""",
        (trait_id, mina_id, now),
    )
    kv.commit()
    kv.close()

    server = build_server(config, embedder=_stub_embedder())
    async with create_connected_server_and_client_session(server) as session:
        explored = _parse(
            await session.call_tool(
                "memory_explore",
                {"node_type": "entity", "node_id": mina_id},
            )
        )

    assert isinstance(explored, dict)
    assert explored["node"]["name"] == "mina"
    assert explored["node"]["is_self"] is True
    assert "part_of" in explored["neighbors"]
    assert explored["neighbors"]["part_of"][0]["direction"] == "incoming"
    assert explored["neighbors"]["part_of"][0]["id"] == trait_id


@pytest.mark.anyio
async def test_mcp_explore_exclude_entity_ids(tmp_project: Path) -> None:
    """exclude_entity_ids が MCP 経由でも効く。"""
    import sqlite3
    from datetime import UTC, datetime

    config = _init_character(tmp_project, "mcp_explore_excl")

    kv = sqlite3.connect(
        str(tmp_project / "data" / "mcp_explore_excl" / "kv.sqlite"),
    )
    now = datetime.now(UTC).isoformat()
    cur = kv.execute(
        """INSERT INTO entities (canonical_name, decay_rate, valid_from)
        VALUES ('mina', 0.2, ?)""",
        (now,),
    )
    mina_id = cur.lastrowid
    cur = kv.execute(
        """INSERT INTO entities (canonical_name, decay_rate, valid_from)
        VALUES ('猫', 0.5, ?)""",
        (now,),
    )
    cat_id = cur.lastrowid
    cur = kv.execute(
        """INSERT INTO entities (canonical_name, decay_rate, valid_from)
        VALUES ('魚', 0.5, ?)""",
        (now,),
    )
    fish_id = cur.lastrowid
    for dst in (cat_id, fish_id):
        kv.execute(
            """INSERT INTO relations
                (src_type, src_id, dst_type, dst_id, predicate, strength, valid_from)
            VALUES ('entity', ?, 'entity', ?, 'likes', 1.0, ?)""",
            (mina_id, dst, now),
        )
    kv.commit()
    kv.close()

    server = build_server(config, embedder=_stub_embedder())
    async with create_connected_server_and_client_session(server) as session:
        explored = _parse(
            await session.call_tool(
                "memory_explore",
                {
                    "node_type": "entity",
                    "node_id": mina_id,
                    "exclude_entity_ids": [cat_id],
                },
            )
        )

    assert isinstance(explored, dict)
    likes_ids = [item["id"] for item in explored["neighbors"].get("likes", [])]
    assert cat_id not in likes_ids
    assert fish_id in likes_ids



@pytest.mark.anyio
async def test_mcp_explore_exclude_episode_ids(tmp_project: Path) -> None:
    """exclude_episode_ids が MCP 経由でも効く。"""
    import sqlite3
    from datetime import UTC, datetime

    config = _init_character(tmp_project, "mcp_explore_excl_ep")

    kv = sqlite3.connect(
        str(tmp_project / "data" / "mcp_explore_excl_ep" / "kv.sqlite"),
    )
    now = datetime.now(UTC).isoformat()
    cur = kv.execute(
        """INSERT INTO entities (canonical_name, decay_rate, valid_from)
        VALUES ('hub', 0.2, ?)""",
        (now,),
    )
    hub_id = cur.lastrowid
    cur = kv.execute(
        """INSERT INTO episodes (content, kind, importance, valid_from)
        VALUES ('記録A', 'facts', 1, ?)""",
        (now,),
    )
    ep_a = cur.lastrowid
    cur = kv.execute(
        """INSERT INTO episodes (content, kind, importance, valid_from)
        VALUES ('記録B', 'facts', 1, ?)""",
        (now,),
    )
    ep_b = cur.lastrowid
    for ep in (ep_a, ep_b):
        kv.execute(
            """INSERT INTO relations
                (src_type, src_id, dst_type, dst_id, predicate, strength, valid_from)
            VALUES ('episode', ?, 'entity', ?, 'mentions', 1.0, ?)""",
            (ep, hub_id, now),
        )
    kv.commit()
    kv.close()

    server = build_server(config, embedder=_stub_embedder())
    async with create_connected_server_and_client_session(server) as session:
        explored = _parse(
            await session.call_tool(
                "memory_explore",
                {
                    "node_type": "entity",
                    "node_id": hub_id,
                    "exclude_episode_ids": [ep_a],
                },
            )
        )

    assert isinstance(explored, dict)
    mentions_ids = [
        item["id"] for item in explored["neighbors"].get("mentions", [])
    ]
    assert ep_a not in mentions_ids
    assert ep_b in mentions_ids


@pytest.mark.anyio
async def test_mcp_explore_include_archived(tmp_project: Path) -> None:
    """include_archived=True で archived relation を含めて取得できる。"""
    import sqlite3
    from datetime import UTC, datetime, timedelta

    config = _init_character(tmp_project, "mcp_explore_archived")

    kv = sqlite3.connect(
        str(tmp_project / "data" / "mcp_explore_archived" / "kv.sqlite"),
    )
    now_dt = datetime.now(UTC)
    now = now_dt.isoformat()
    archived = (now_dt - timedelta(days=30)).isoformat()
    cur = kv.execute(
        """INSERT INTO entities (canonical_name, decay_rate, valid_from)
        VALUES ('hub', 0.2, ?)""",
        (now,),
    )
    hub_id = cur.lastrowid
    cur = kv.execute(
        """INSERT INTO entities (canonical_name, decay_rate, valid_from)
        VALUES ('old_trait', 0.5, ?)""",
        (now,),
    )
    old_id = cur.lastrowid
    kv.execute(
        """INSERT INTO relations
            (src_type, src_id, dst_type, dst_id, predicate, strength,
             valid_from, valid_to)
        VALUES ('entity', ?, 'entity', ?, 'part_of', 1.0, ?, ?)""",
        (old_id, hub_id, now, archived),
    )
    kv.commit()
    kv.close()

    server = build_server(config, embedder=_stub_embedder())
    async with create_connected_server_and_client_session(server) as session:
        excluded = _parse(
            await session.call_tool(
                "memory_explore",
                {"node_type": "entity", "node_id": hub_id},
            )
        )
        included = _parse(
            await session.call_tool(
                "memory_explore",
                {
                    "node_type": "entity",
                    "node_id": hub_id,
                    "include_archived": True,
                },
            )
        )

    assert isinstance(excluded, dict)
    assert excluded["total_neighbors"] == 0
    assert excluded["total_neighbors_unfiltered"] == 0
    assert isinstance(included, dict)
    assert "part_of" in included["neighbors"]
    assert included["total_neighbors_unfiltered"] == 1


@pytest.mark.anyio
async def test_mcp_explore_include_suppressed(tmp_project: Path) -> None:
    """include_suppressed=True で抑制 episode が返る。"""
    import sqlite3
    from datetime import UTC, datetime

    config = _init_character(tmp_project, "mcp_explore_suppressed")

    kv = sqlite3.connect(
        str(tmp_project / "data" / "mcp_explore_suppressed" / "kv.sqlite"),
    )
    now = datetime.now(UTC).isoformat()
    cur = kv.execute(
        """INSERT INTO entities (canonical_name, decay_rate, valid_from)
        VALUES ('hub', 0.2, ?)""",
        (now,),
    )
    hub_id = cur.lastrowid
    cur = kv.execute(
        """INSERT INTO episodes
            (content, kind, importance, valid_from, is_suppressed)
        VALUES ('忘れかけ記録', 'facts', 1, ?, 1)""",
        (now,),
    )
    ep_id = cur.lastrowid
    kv.execute(
        """INSERT INTO relations
            (src_type, src_id, dst_type, dst_id, predicate, strength, valid_from)
        VALUES ('episode', ?, 'entity', ?, 'mentions', 1.0, ?)""",
        (ep_id, hub_id, now),
    )
    kv.commit()
    kv.close()

    server = build_server(config, embedder=_stub_embedder())
    async with create_connected_server_and_client_session(server) as session:
        excluded = _parse(
            await session.call_tool(
                "memory_explore",
                {"node_type": "entity", "node_id": hub_id},
            )
        )
        included = _parse(
            await session.call_tool(
                "memory_explore",
                {
                    "node_type": "entity",
                    "node_id": hub_id,
                    "include_suppressed": True,
                },
            )
        )

    assert isinstance(excluded, dict)
    assert "mentions" not in excluded["neighbors"]
    assert isinstance(included, dict)
    assert "mentions" in included["neighbors"]


@pytest.mark.anyio
async def test_mcp_explore_error_node_not_found(tmp_project: Path) -> None:
    """node not found のエラー内容が MCP 経由で AI 側に伝わる。"""
    config = _init_character(tmp_project, "mcp_explore_err")
    server = build_server(config, embedder=_stub_embedder())

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(
            "memory_explore",
            {"node_type": "entity", "node_id": 999},
        )

    # MCP の tool error として返る
    assert getattr(result, "isError", False) is True
    content = getattr(result, "content", None)
    assert content
    text = content[0].text if hasattr(content[0], "text") else ""
    assert "not found" in text


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
