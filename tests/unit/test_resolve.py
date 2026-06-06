"""tests for fravenir.core.resolve."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fravenir.core.resolve import (
    ResolveError,
    list_candidates,
    merge,
    reject,
)
from fravenir.storage.sqlite_init import init_kv


def _seed(tmp_path: Path) -> Path:
    """2 entity + relations + 1 merge_candidate を用意する。

    e1 (id=1, "ねこ", concept), e2 (id=2, "ネコ", concept)
    e3 (id=3, "ごはん", concept) — keep の隣接関係用
    relations:
      r1: entity:e2 -[likes]-> entity:e3   (drop が src)
      r2: entity:e3 -[mentions]-> entity:e2 (drop が dst)
      r3: entity:e2 -[part_of]-> entity:e1  (merge 後に self loop になる)
      r4: entity:e2 -[archived]-> entity:e3 (valid_to 既に立ってる、触らない)
    aliases:
      e2.aliases = ["ねこちゃん"]
    merge_candidates:
      mc1: (e1, e2, similarity=0.95, resolved=0)
    """
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            "INSERT INTO entities (id, canonical_name, entity_type, valid_from)"
            " VALUES (?, ?, ?, ?)",
            [
                (1, "ねこ", "concept", now),
                (2, "ネコ", "concept", now),
                (3, "ごはん", "concept", now),
            ],
        )
        conn.execute(
            "INSERT INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
            ("ねこちゃん", 2),
        )
        conn.executemany(
            "INSERT INTO relations "
            "(src_type, src_id, dst_type, dst_id, predicate, valid_from, valid_to)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                ("entity", 2, "entity", 3, "likes", now, None),
                ("entity", 3, "entity", 2, "mentions", now, None),
                ("entity", 2, "entity", 1, "part_of", now, None),
                ("entity", 2, "entity", 3, "archived", now, now),
            ],
        )
        conn.execute(
            "INSERT INTO merge_candidates (entity_a, entity_b, similarity, resolved)"
            " VALUES (1, 2, 0.95, 0)"
        )
        conn.commit()
    finally:
        conn.close()
    return db


class TestListCandidates:
    def test_returns_unresolved_only(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        # 既 resolved の候補を1件追加して、無視されることを確認
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "INSERT INTO merge_candidates (entity_a, entity_b, similarity, resolved)"
                " VALUES (1, 3, 0.91, 1)"
            )
            conn.commit()
        finally:
            conn.close()

        rows = list_candidates(db)

        assert len(rows) == 1
        assert rows[0].candidate_id == 1
        assert rows[0].entity_a == 1
        assert rows[0].entity_b == 2
        assert rows[0].a_name == "ねこ"
        assert rows[0].b_name == "ネコ"
        assert rows[0].a_type == "concept"

    def test_list_candidates_returns_judge_fields(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "UPDATE merge_candidates "
                "SET judge_label = 'same', judge_confidence = 'high', "
                "judge_reason = '同一', judge_attempts = 1 "
                "WHERE id = 1"
            )
            conn.commit()
        finally:
            conn.close()

        rows = list_candidates(db)
        assert len(rows) == 1
        assert rows[0].judge_label == "same"
        assert rows[0].judge_confidence == "high"
        assert rows[0].judge_reason == "同一"
        assert rows[0].judge_attempts == 1


class TestMerge:
    def test_dry_run_no_side_effects(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        result = merge(db, 1, dry_run=True)

        assert result.dry_run is True
        assert result.keep_id == 1
        assert result.drop_id == 2

        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT valid_to, supersedes FROM entities WHERE id = 2"
            ).fetchone()
            mc = conn.execute(
                "SELECT resolved FROM merge_candidates WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
        assert row == (None, None)
        assert mc[0] == 0

    def test_merge_keeps_smaller_id(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        result = merge(db, 1)

        assert result.keep_id == 1
        assert result.drop_id == 2
        assert result.dry_run is False

        conn = sqlite3.connect(db)
        try:
            ent2 = conn.execute(
                "SELECT valid_to, supersedes FROM entities WHERE id = 2"
            ).fetchone()
            ent1 = conn.execute(
                "SELECT valid_to FROM entities WHERE id = 1"
            ).fetchone()
            mc = conn.execute(
                "SELECT resolved, resolved_at FROM merge_candidates WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
        assert ent2[0] is not None  # drop に valid_to が立つ
        assert ent2[1] == 1         # supersedes = keep
        assert ent1[0] is None      # keep は変わらず
        assert mc[0] == 1
        # resolved_at 記録 (Commit F)
        assert mc[1] is not None

    def test_relations_rewired(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        merge(db, 1)

        conn = sqlite3.connect(db)
        try:
            # likes: drop が src だった → keep が src になる
            r_likes = conn.execute(
                "SELECT src_id, dst_id, valid_to FROM relations WHERE predicate = 'likes'"
            ).fetchone()
            # mentions: drop が dst だった → keep が dst になる
            r_mentions = conn.execute(
                "SELECT src_id, dst_id, valid_to FROM relations WHERE predicate = 'mentions'"
            ).fetchone()
            # archived: 元から valid_to 立ってたので touched しない（src/dst が変わらない）
            r_archived = conn.execute(
                "SELECT src_id, dst_id FROM relations WHERE predicate = 'archived'"
            ).fetchone()
        finally:
            conn.close()
        assert r_likes == (1, 3, None)
        assert r_mentions == (3, 1, None)
        assert r_archived == (2, 3)  # valid_to 既存のため変えない

    def test_self_loop_archived(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        result = merge(db, 1)

        assert result.self_loops_archived == 1

        conn = sqlite3.connect(db)
        try:
            # part_of: drop -> keep だったので、付け替えで keep -> keep になる → 論理削除
            r_part_of = conn.execute(
                "SELECT src_id, dst_id, valid_to FROM relations WHERE predicate = 'part_of'"
            ).fetchone()
        finally:
            conn.close()
        assert r_part_of[0] == 1
        assert r_part_of[1] == 1
        assert r_part_of[2] is not None

    def test_aliases_merged(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        result = merge(db, 1)

        # drop の canonical_name "ネコ" + drop の既存 alias "ねこちゃん" の 2件
        assert result.aliases_added == 2

        conn = sqlite3.connect(db)
        try:
            aliases = sorted(
                row[0]
                for row in conn.execute(
                    "SELECT alias FROM entity_aliases WHERE entity_id = 1"
                ).fetchall()
            )
        finally:
            conn.close()
        assert aliases == ["ねこちゃん", "ネコ"]

    def test_explicit_keep(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        result = merge(db, 1, keep=2)

        assert result.keep_id == 2
        assert result.drop_id == 1

    def test_keep_not_in_pair_raises(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        with pytest.raises(ResolveError):
            merge(db, 1, keep=999)

    def test_already_resolved_raises(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        merge(db, 1)
        with pytest.raises(ResolveError):
            merge(db, 1)

    def test_unknown_candidate_raises(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        with pytest.raises(KeyError):
            merge(db, 999)


class TestReject:
    def test_reject_sets_resolved_2(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        result = reject(db, 1)

        assert result.candidate_id == 1
        assert result.dry_run is False

        conn = sqlite3.connect(db)
        try:
            mc = conn.execute(
                "SELECT resolved, resolved_at FROM merge_candidates WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
        assert mc[0] == 2
        assert mc[1] is not None

    def test_reject_dry_run(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        result = reject(db, 1, dry_run=True)
        assert result.dry_run is True

        conn = sqlite3.connect(db)
        try:
            mc = conn.execute(
                "SELECT resolved FROM merge_candidates WHERE id = 1"
            ).fetchone()
        finally:
            conn.close()
        assert mc[0] == 0

    def test_already_resolved_raises(self, tmp_path: Path) -> None:
        db = _seed(tmp_path)
        merge(db, 1)
        with pytest.raises(ResolveError):
            reject(db, 1)


def test_merge_rejects_self_hub(tmp_path: Path) -> None:
    """is_self=1 の entity を含む候補で merge() を呼ぶと ResolveError が raise される。"""
    db = tmp_path / "test.db"
    init_kv(db)
    conn = sqlite3.connect(db)
    now = "2026-04-29T00:00:00+00:00"
    # is_self=1 entity を含むペアを作る
    conn.execute(
        "INSERT INTO entities (id, canonical_name, is_self, valid_from) "
        "VALUES (1, 'mina', 1, ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO entities (id, canonical_name, is_self, valid_from) "
        "VALUES (2, 'mina_clone', 0, ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO merge_candidates (id, entity_a, entity_b, similarity) "
        "VALUES (10, 1, 2, 0.9)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ResolveError, match="self-hub entity is involved"):
        merge(db, candidate_id=10)


def test_merge_rejects_self_pair(tmp_path: Path) -> None:
    """entity_a == entity_b の merge_candidate で merge() を呼ぶと ResolveError が raise される。

    SEC-1 バッチ④ MEDIUM-2: 外部経路で merge_candidates に直接挿入された
    self-pair が、自分を自分で supersede してグラフを壊すのを防ぐ。
    """
    db = tmp_path / "test.db"
    init_kv(db)
    conn = sqlite3.connect(db)
    now = "2026-04-29T00:00:00+00:00"
    conn.execute(
        "INSERT INTO entities (id, canonical_name, is_self, valid_from) "
        "VALUES (1, 'mina', 0, ?)",
        (now,),
    )
    conn.execute(
        "INSERT INTO merge_candidates (id, entity_a, entity_b, similarity) "
        "VALUES (20, 1, 1, 1.0)"
    )
    conn.commit()
    conn.close()

    with pytest.raises(ResolveError, match="self-merge"):
        merge(db, candidate_id=20)
