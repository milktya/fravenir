"""Unit tests for fravenir.admin.queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fravenir.admin.queries import (
    get_doc_status,
    get_entity_detail,
    get_episode_detail,
    get_graph,
    get_merge_candidates,
    get_orphans,
    get_relation_detail,
    get_stats,
)
from fravenir.storage.sqlite_init import init_kv

NOW = "2026-04-29T12:00:00"
THEN = "2026-04-01T00:00:00"

# 30 文字超 (truncation テスト用)
EP1_CONTENT = "Phase 5 全完了レポートの内容はこちら、詳細は別ドキュメントを参照のこと"


def _seed(tmp_path: Path) -> Path:
    """Seed a kv.sqlite with representative data for all query tests.

    episodes:
      ep1 (id=1, active, not suppressed, kind=facts)
      ep2 (id=2, valid_to set → archived, not suppressed, kind=state)
      ep3 (id=3, active, is_suppressed=1, kind=emo)
      ep4 (id=4, active, not suppressed, kind=facts) ← orphan (no relations)

    entities:
      en1 (id=1, "ミナ", is_self=1, active)
      en2 (id=2, "ProjectX", is_self=0, active)
      en3 (id=3, "孤立エンティティ", is_self=0, active) ← orphan
        aliases: ["lonely"]

    relations:
      r1 (id=1): ep1 -[mentions]-> en1  (active)
      r2 (id=2): ep1 -[mentions]-> en2  (archived, valid_to set)
      r3 (id=3): en1 -[hosts]-> en2     (active, strength=0.8)

    merge_candidates:
      mc1 (id=1): en1 vs en2, resolved=0 (pending)
      mc2 (id=2): en1 vs en3, resolved=1 (merged)
      mc3 (id=3): en2 vs en3, resolved=2 (rejected)

    doc_status:
      ds1 (id=1): episode_id=1, stage="done", error=None
      ds2 (id=2): episode_id=2, stage="embedded", error="connection refused"
    """
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            "INSERT INTO episodes (id, content, kind, importance, valid_from, valid_to,"
            " is_suppressed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, EP1_CONTENT, "facts", 3, NOW, None, 0, NOW),
                (2, "古いエピソード", "state", 1, THEN, NOW, 0, THEN),
                (3, "感情エピソード", "emo", 2, NOW, None, 1, NOW),
                (4, "孤立エピソード（リレーションなし）", "facts", 1, NOW, None, 0, NOW),
            ],
        )
        conn.executemany(
            "INSERT INTO entities (id, canonical_name, entity_type, is_self, self_weight,"
            " decay_rate, valid_from, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "ミナ", "person", 1, 1.0, 0.2, NOW, NOW),
                (2, "ProjectX", "concept", 0, 0.0, 0.5, NOW, NOW),
                (3, "孤立エンティティ", "concept", 0, 0.0, 0.5, NOW, NOW),
            ],
        )
        conn.execute(
            "INSERT INTO entity_aliases (alias, entity_id) VALUES (?, ?)", ("lonely", 3)
        )
        conn.executemany(
            "INSERT INTO relations (id, src_type, src_id, dst_type, dst_id, predicate,"
            " strength, valid_from, valid_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "episode", 1, "entity", 1, "mentions", 1.0, NOW, None),
                (2, "episode", 1, "entity", 2, "mentions", 1.0, THEN, NOW),
                (3, "entity", 1, "entity", 2, "hosts", 0.8, NOW, None),
            ],
        )
        conn.executemany(
            "INSERT INTO merge_candidates (id, entity_a, entity_b, similarity,"
            " detected_at, resolved) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 2, 0.92, NOW, 0),
                (2, 1, 3, 0.85, NOW, 1),
                (3, 2, 3, 0.78, NOW, 2),
            ],
        )
        conn.executemany(
            "INSERT INTO doc_status (id, episode_id, stage, error, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            [
                (1, 1, "done", None, NOW),
                (2, 2, "embedded", "connection refused", NOW),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return _seed(tmp_path)


class TestGetStats:
    def test_structure(self, db: Path) -> None:
        result = get_stats(db)
        assert set(result.keys()) == {
            "episodes", "entities", "relations", "merge_candidates",
            "doc_status_failed", "orphans",
        }

    def test_episodes(self, db: Path) -> None:
        ep = get_stats(db)["episodes"]
        assert ep["total"] == 4
        assert ep["active"] == 3   # ep1, ep3, ep4 (ep2 has valid_to)
        assert ep["suppressed"] == 1  # ep3

    def test_entities(self, db: Path) -> None:
        en = get_stats(db)["entities"]
        assert en["total"] == 3
        assert en["active"] == 3
        assert en["is_self"] == 1

    def test_relations(self, db: Path) -> None:
        rel = get_stats(db)["relations"]
        assert rel["total"] == 3
        assert rel["active"] == 2   # r1 and r3 (r2 has valid_to)

    def test_merge_candidates_resolved_values(self, db: Path) -> None:
        mc = get_stats(db)["merge_candidates"]
        assert mc["pending"] == 1
        assert mc["merged"] == 1
        assert mc["rejected"] == 1

    def test_doc_status_failed(self, db: Path) -> None:
        assert get_stats(db)["doc_status_failed"] == 1

    def test_orphans(self, db: Path) -> None:
        orp = get_stats(db)["orphans"]
        # active scope: ep2 (archived) と ep3 (suppressed) は除外、ep4 のみ orphan
        assert orp["episodes"] == 1
        assert orp["entities"] == 1  # en3 (active, neither src nor dst)


class TestGetGraph:
    def test_active_excludes_archived_episode(self, db: Path) -> None:
        result = get_graph(db, "active")
        node_ids = {n["data"]["id"] for n in result["elements"]["nodes"]}
        assert "ep_1" in node_ids
        assert "ep_2" not in node_ids   # valid_to is set

    def test_active_excludes_suppressed(self, db: Path) -> None:
        result = get_graph(db, "active")
        node_ids = {n["data"]["id"] for n in result["elements"]["nodes"]}
        assert "ep_3" not in node_ids   # is_suppressed=1

    def test_archived_includes_valid_to_episode(self, db: Path) -> None:
        result = get_graph(db, "archived")
        node_ids = {n["data"]["id"] for n in result["elements"]["nodes"]}
        assert "ep_2" in node_ids

    def test_archived_excludes_suppressed(self, db: Path) -> None:
        result = get_graph(db, "archived")
        node_ids = {n["data"]["id"] for n in result["elements"]["nodes"]}
        assert "ep_3" not in node_ids

    def test_all_includes_suppressed(self, db: Path) -> None:
        result = get_graph(db, "all")
        node_ids = {n["data"]["id"] for n in result["elements"]["nodes"]}
        assert "ep_3" in node_ids

    def test_node_fields_episode(self, db: Path) -> None:
        result = get_graph(db, "active")
        ep_nodes = [n for n in result["elements"]["nodes"] if n["data"]["type"] == "episode"]
        ep1 = next(n for n in ep_nodes if n["data"]["id"] == "ep_1")
        d = ep1["data"]
        assert d["kind"] == "facts"
        assert d["importance"] == 3
        assert d["is_active"] is True
        assert d["is_suppressed"] is False
        assert "supersedes" in d

    def test_node_label_truncation(self, db: Path) -> None:
        result = get_graph(db, "active")
        ep_nodes = [n for n in result["elements"]["nodes"] if n["data"]["type"] == "episode"]
        ep1 = next(n for n in ep_nodes if n["data"]["id"] == "ep_1")
        label = ep1["data"]["label"]
        assert label.endswith("…")
        assert len(label) == 31  # 30 chars + ellipsis (…)

    def test_node_fields_entity(self, db: Path) -> None:
        result = get_graph(db, "active")
        en_nodes = [n for n in result["elements"]["nodes"] if n["data"]["type"] == "entity"]
        en1 = next(n for n in en_nodes if n["data"]["id"] == "en_1")
        d = en1["data"]
        assert d["label"] == "ミナ"
        assert d["is_self"] is True
        assert d["is_active"] is True

    def test_edge_mentions_id_prefix(self, db: Path) -> None:
        result = get_graph(db, "active")
        edge_ids = {e["data"]["id"] for e in result["elements"]["edges"]}
        assert "men_1" in edge_ids

    def test_edge_relation_id_prefix(self, db: Path) -> None:
        result = get_graph(db, "active")
        edge_ids = {e["data"]["id"] for e in result["elements"]["edges"]}
        assert "rel_3" in edge_ids

    def test_stats_counts(self, db: Path) -> None:
        result = get_graph(db, "active")
        assert result["stats"]["nodes"] == len(result["elements"]["nodes"])
        assert result["stats"]["edges"] == len(result["elements"]["edges"])


class TestGetEpisodeDetail:
    def test_existing_episode(self, db: Path) -> None:
        result = get_episode_detail(db, 1)
        assert result is not None
        assert result["id"] == 1
        assert result["kind"] == "facts"
        assert result["importance"] == 3

    def test_mentions_included(self, db: Path) -> None:
        result = get_episode_detail(db, 1)
        assert result is not None
        entity_ids = {m["entity_id"] for m in result["mentions"]}
        assert 1 in entity_ids   # en1
        assert 2 in entity_ids   # en2

    def test_doc_status_included(self, db: Path) -> None:
        result = get_episode_detail(db, 1)
        assert result is not None
        ds = result["doc_status"]
        assert ds["stage"] == "done"
        assert ds["error"] is None

    def test_nonexistent_returns_none(self, db: Path) -> None:
        assert get_episode_detail(db, 9999) is None


class TestGetEntityDetail:
    def test_existing_entity(self, db: Path) -> None:
        result = get_entity_detail(db, 1)
        assert result is not None
        assert result["id"] == 1
        assert result["canonical_name"] == "ミナ"
        assert result["is_self"] is True

    def test_aliases_included(self, db: Path) -> None:
        result = get_entity_detail(db, 3)
        assert result is not None
        assert "lonely" in result["aliases"]

    def test_in_relations(self, db: Path) -> None:
        result = get_entity_detail(db, 1)
        assert result is not None
        assert len(result["in_relations"]) >= 1
        in_rel = result["in_relations"][0]
        assert "src_type" in in_rel
        assert "predicate" in in_rel

    def test_out_relations(self, db: Path) -> None:
        result = get_entity_detail(db, 1)
        assert result is not None
        assert len(result["out_relations"]) >= 1
        out_rel = result["out_relations"][0]
        assert out_rel["predicate"] == "hosts"
        assert out_rel["strength"] == pytest.approx(0.8)

    def test_nonexistent_returns_none(self, db: Path) -> None:
        assert get_entity_detail(db, 9999) is None


class TestGetRelationDetail:
    def test_existing_relation(self, db: Path) -> None:
        result = get_relation_detail(db, 3)
        assert result is not None
        assert result["id"] == 3
        assert result["src_type"] == "entity"
        assert result["dst_type"] == "entity"
        assert result["predicate"] == "hosts"

    def test_src_label_entity(self, db: Path) -> None:
        result = get_relation_detail(db, 3)
        assert result is not None
        assert result["src_label"] == "ミナ"

    def test_dst_label_entity(self, db: Path) -> None:
        result = get_relation_detail(db, 3)
        assert result is not None
        assert result["dst_label"] == "ProjectX"

    def test_src_label_episode(self, db: Path) -> None:
        result = get_relation_detail(db, 1)
        assert result is not None
        assert result["src_type"] == "episode"
        assert result["src_label"].endswith("…") or len(result["src_label"]) <= 30

    def test_nonexistent_returns_none(self, db: Path) -> None:
        assert get_relation_detail(db, 9999) is None


class TestGetMergeCandidates:
    def test_pending_filter(self, db: Path) -> None:
        result = get_merge_candidates(db, "pending")
        assert result["status_filter"] == "pending"
        assert all(c["resolved"] == 0 for c in result["candidates"])
        assert len(result["candidates"]) == 1

    def test_merged_filter(self, db: Path) -> None:
        result = get_merge_candidates(db, "merged")
        assert all(c["resolved"] == 1 for c in result["candidates"])
        assert len(result["candidates"]) == 1

    def test_rejected_filter(self, db: Path) -> None:
        result = get_merge_candidates(db, "rejected")
        assert all(c["resolved"] == 2 for c in result["candidates"])
        assert len(result["candidates"]) == 1

    def test_all_filter(self, db: Path) -> None:
        result = get_merge_candidates(db, "all")
        assert len(result["candidates"]) == 3

    def test_entity_fields_joined(self, db: Path) -> None:
        result = get_merge_candidates(db, "pending")
        cand = result["candidates"][0]
        assert "id" in cand["entity_a"]
        assert "canonical_name" in cand["entity_a"]
        assert "id" in cand["entity_b"]
        assert "canonical_name" in cand["entity_b"]


class TestGetDocStatus:
    def test_failed_filter(self, db: Path) -> None:
        result = get_doc_status(db, "failed")
        assert result["status_filter"] == "failed"
        assert all(item["error"] is not None for item in result["items"])
        assert len(result["items"]) == 1
        assert result["items"][0]["stage"] == "embedded"

    def test_all_filter(self, db: Path) -> None:
        result = get_doc_status(db, "all")
        assert len(result["items"]) == 2

    def test_episode_label_joined(self, db: Path) -> None:
        result = get_doc_status(db, "failed")
        item = result["items"][0]
        assert "episode_label" in item
        assert isinstance(item["episode_label"], str)


class TestGetOrphans:
    def test_orphan_episode_detected(self, db: Path) -> None:
        result = get_orphans(db, "active")
        ep_ids = {ep["id"] for ep in result["episodes"]}
        assert 4 in ep_ids   # ep4 has no outgoing relation as src

    def test_non_orphan_episode_excluded(self, db: Path) -> None:
        result = get_orphans(db, "active")
        ep_ids = {ep["id"] for ep in result["episodes"]}
        assert 1 not in ep_ids   # ep1 has mentions relations

    def test_orphan_entity_detected(self, db: Path) -> None:
        result = get_orphans(db, "active")
        en_ids = {en["id"] for en in result["entities"]}
        assert 3 in en_ids   # en3 appears in no relation

    def test_non_orphan_entity_excluded(self, db: Path) -> None:
        result = get_orphans(db, "active")
        en_ids = {en["id"] for en in result["entities"]}
        assert 1 not in en_ids   # en1 is in r1 (dst) and r3 (src)
        assert 2 not in en_ids   # en2 is in r2 (dst) and r3 (dst)

    def test_scope_active_excludes_archived(self, db: Path) -> None:
        result = get_orphans(db, "active")
        ep_ids = {ep["id"] for ep in result["episodes"]}
        assert 2 not in ep_ids   # ep2 has valid_to (archived)

    def test_scope_archived_includes_archived(self, db: Path) -> None:
        result = get_orphans(db, "archived")
        ep_ids = {ep["id"] for ep in result["episodes"]}
        assert 4 in ep_ids   # ep4 still orphan in archived scope

    def test_scope_all_includes_suppressed(self, db: Path) -> None:
        result = get_orphans(db, "all")
        ep_ids = {ep["id"] for ep in result["episodes"]}
        assert 4 in ep_ids

    def test_scope_field(self, db: Path) -> None:
        result = get_orphans(db, "active")
        assert result["scope"] == "active"
