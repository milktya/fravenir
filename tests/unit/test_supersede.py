"""Unit tests for core/supersede.py."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from fravenir.core.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from fravenir.core.supersede import detect_and_supersede
from fravenir.storage import sqlite_init


def _make_character(tmp_project: Path, char_id: str = "test_char") -> str:
    data_dir = tmp_project / "data" / char_id
    data_dir.mkdir(parents=True)
    sqlite_init.init_kv(data_dir / "kv.sqlite")
    return char_id


def _db_path(tmp_project: Path, char_id: str) -> Path:
    return tmp_project / "data" / char_id / "kv.sqlite"


def _connect(tmp_project: Path, char_id: str) -> sqlite3.Connection:
    return sqlite3.connect(str(_db_path(tmp_project, char_id)))


class TestDetectAndSupersede:
    def test_no_conflict_no_change(self, tmp_project: Path) -> None:
        """矛盾なし（新規 entity-to-entity relation のみ）→ counts == (0, 0)."""
        char_id = _make_character(tmp_project)
        conn = _connect(tmp_project, char_id)
        now = datetime.now(UTC)
        now_iso = now.isoformat()

        # 新 episode
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("新規のみ", now_iso),
        )
        ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # 新 entities
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'person', ?)",
            ("みるちゃ", now_iso),
        )
        src_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'concept', ?)",
            ("猫", now_iso),
        )
        dst_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # mentions + entity relation (likes は allowlist 外)
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (ep_id, src_id, now_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (ep_id, dst_id, now_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'likes', ?)",
            (src_id, dst_id, now_iso),
        )

        result = ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="猫", entity_type="concept"),
            ],
            relations=[
                ExtractedRelation(src="みるちゃ", dst="猫", predicate="likes"),
            ],
        )

        stats = detect_and_supersede(
            conn=conn,
            new_episode_id=ep_id,
            new_episode_kind="facts",
            result=result,
            name_to_id={"みるちゃ": src_id, "猫": dst_id},
            now=now,
        )

        assert stats == {"relations_superseded": 0, "episodes_superseded": 0}
        # 既存 relation の valid_to は変わらない
        rel_valid_to = conn.execute(
            "SELECT valid_to FROM relations WHERE predicate='likes'"
        ).fetchone()[0]
        assert rel_valid_to is None

    def test_single_value_predicate_supersedes_old(self, tmp_project: Path) -> None:
        """works_as の矛盾 → 古い relation/episode の valid_to が立ち、supersedes が設定される。"""
        char_id = _make_character(tmp_project)
        conn = _connect(tmp_project, char_id)
        old_now = datetime(2026, 1, 1, tzinfo=UTC)
        old_iso = old_now.isoformat()
        new_now = datetime(2026, 1, 2, tzinfo=UTC)
        new_iso = new_now.isoformat()

        # -- 古い状態をシード --
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃの仕事はプログラマ", old_iso),
        )
        old_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'person', ?)",
            ("みるちゃ", old_iso),
        )
        mirucha_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'work', ?)",
            ("プログラマ", old_iso),
        )
        programmer_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # mentions (old_iso)
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, mirucha_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, programmer_id, old_iso),
        )
        # entity relation (old_iso)
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, programmer_id, old_iso),
        )
        old_rel_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # -- 新しい状態をシード（すでに DB に書き込まれた後の状態を模擬） --
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃの仕事はデザイナ", new_iso),
        )
        new_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'work', ?)",
            ("デザイナ", new_iso),
        )
        designer_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # mentions (new_iso)
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, mirucha_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, designer_id, new_iso),
        )
        # entity relation (new_iso)
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, designer_id, new_iso),
        )
        new_rel_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        result = ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="デザイナ", entity_type="work"),
            ],
            relations=[
                ExtractedRelation(
                    src="みるちゃ", dst="デザイナ", predicate="works_as"
                ),
            ],
        )

        stats = detect_and_supersede(
            conn=conn,
            new_episode_id=new_ep_id,
            new_episode_kind="facts",
            result=result,
            name_to_id={"みるちゃ": mirucha_id, "デザイナ": designer_id},
            now=new_now,
        )

        assert stats == {"relations_superseded": 1, "episodes_superseded": 1}

        # 旧 relation: valid_to が設定され、supersedes は NULL
        old_rel = conn.execute(
            "SELECT valid_to, supersedes FROM relations WHERE id = ?", (old_rel_id,)
        ).fetchone()
        assert old_rel[0] == new_iso
        assert old_rel[1] is None

        # 新 relation: valid_to は NULL、supersedes = 旧 relation_id
        new_rel = conn.execute(
            "SELECT valid_to, supersedes FROM relations WHERE id = ?", (new_rel_id,)
        ).fetchone()
        assert new_rel[0] is None
        assert new_rel[1] == old_rel_id

        # 旧 episode: valid_to 設定、supersedes は NULL
        old_ep = conn.execute(
            "SELECT valid_to, supersedes FROM episodes WHERE id = ?", (old_ep_id,)
        ).fetchone()
        assert old_ep[0] == new_iso
        assert old_ep[1] is None

        # 新 episode: valid_to NULL、supersedes = 旧 episode_id
        new_ep = conn.execute(
            "SELECT valid_to, supersedes FROM episodes WHERE id = ?", (new_ep_id,)
        ).fetchone()
        assert new_ep[0] is None
        assert new_ep[1] == old_ep_id

    def test_non_allowlist_predicate_skipped(self, tmp_project: Path) -> None:
        """likes は allowlist 外 → 矛盾扱いされない。"""
        char_id = _make_character(tmp_project)
        conn = _connect(tmp_project, char_id)
        old_now = datetime(2026, 1, 1, tzinfo=UTC)
        old_iso = old_now.isoformat()
        new_now = datetime(2026, 1, 2, tzinfo=UTC)
        new_iso = new_now.isoformat()

        # 古い状態: みるちゃ likes 猫
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃは猫が好き", old_iso),
        )
        old_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'person', ?)",
            ("みるちゃ", old_iso),
        )
        mirucha_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'concept', ?)",
            ("猫", old_iso),
        )
        cat_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, mirucha_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, cat_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'likes', ?)",
            (mirucha_id, cat_id, old_iso),
        )

        # 新しい状態: みるちゃ likes 犬
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃは犬が好き", new_iso),
        )
        new_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'concept', ?)",
            ("犬", new_iso),
        )
        dog_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, mirucha_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, dog_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'likes', ?)",
            (mirucha_id, dog_id, new_iso),
        )

        result = ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="犬", entity_type="concept"),
            ],
            relations=[
                ExtractedRelation(src="みるちゃ", dst="犬", predicate="likes"),
            ],
        )

        stats = detect_and_supersede(
            conn=conn,
            new_episode_id=new_ep_id,
            new_episode_kind="facts",
            result=result,
            name_to_id={"みるちゃ": mirucha_id, "犬": dog_id},
            now=new_now,
        )

        assert stats == {"relations_superseded": 0, "episodes_superseded": 0}
        # 両方とも valid_to IS NULL のまま
        active_rel_count = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE predicate='likes' AND valid_to IS NULL"
        ).fetchone()[0]
        assert active_rel_count == 2

    def test_state_kind_skipped(self, tmp_project: Path) -> None:
        """kind='state' では works_as の矛盾でも supersede されない。"""
        char_id = _make_character(tmp_project)
        conn = _connect(tmp_project, char_id)
        old_now = datetime(2026, 1, 1, tzinfo=UTC)
        old_iso = old_now.isoformat()
        new_now = datetime(2026, 1, 2, tzinfo=UTC)
        new_iso = new_now.isoformat()

        # 古い facts 状態
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃはプログラマ", old_iso),
        )
        old_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'person', ?)",
            ("みるちゃ", old_iso),
        )
        mirucha_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'work', ?)",
            ("プログラマ", old_iso),
        )
        prog_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, mirucha_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, prog_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, prog_id, old_iso),
        )

        # 新しい state エピソード（kind='state'）
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'state', 1, ?)",
            ("みるちゃはデザイナ", new_iso),
        )
        new_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'work', ?)",
            ("デザイナ", new_iso),
        )
        designer_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, mirucha_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, designer_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, designer_id, new_iso),
        )

        result = ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="デザイナ", entity_type="work"),
            ],
            relations=[
                ExtractedRelation(
                    src="みるちゃ", dst="デザイナ", predicate="works_as"
                ),
            ],
        )

        stats = detect_and_supersede(
            conn=conn,
            new_episode_id=new_ep_id,
            new_episode_kind="state",  # state → スキップ
            result=result,
            name_to_id={"みるちゃ": mirucha_id, "デザイナ": designer_id},
            now=new_now,
        )

        assert stats == {"relations_superseded": 0, "episodes_superseded": 0}

    def test_same_dst_no_supersede(self, tmp_project: Path) -> None:
        """同 dst なら自己 supersede しない（完全一致の relation は重ねない）。"""
        char_id = _make_character(tmp_project)
        conn = _connect(tmp_project, char_id)
        old_now = datetime(2026, 1, 1, tzinfo=UTC)
        old_iso = old_now.isoformat()
        new_now = datetime(2026, 1, 2, tzinfo=UTC)
        new_iso = new_now.isoformat()

        # 古い状態: みるちゃ works_as プログラマ
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃの仕事はプログラマ", old_iso),
        )
        old_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'person', ?)",
            ("みるちゃ", old_iso),
        )
        mirucha_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'work', ?)",
            ("プログラマ", old_iso),
        )
        prog_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, mirucha_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (old_ep_id, prog_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, prog_id, old_iso),
        )
        old_rel_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # 新しい状態: みるちゃ works_as プログラマ（同じ dst）
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃの仕事はプログラマ（再）", new_iso),
        )
        new_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, mirucha_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, prog_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, prog_id, new_iso),
        )

        result = ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="プログラマ", entity_type="work"),
            ],
            relations=[
                ExtractedRelation(
                    src="みるちゃ", dst="プログラマ", predicate="works_as"
                ),
            ],
        )

        stats = detect_and_supersede(
            conn=conn,
            new_episode_id=new_ep_id,
            new_episode_kind="facts",
            result=result,
            name_to_id={"みるちゃ": mirucha_id, "プログラマ": prog_id},
            now=new_now,
        )

        assert stats == {"relations_superseded": 0, "episodes_superseded": 0}
        # 旧 relation の valid_to は NULL のまま
        old_rel = conn.execute(
            "SELECT valid_to, supersedes FROM relations WHERE id = ?", (old_rel_id,)
        ).fetchone()
        assert old_rel[0] is None

    def test_episode_supersede_only_once_per_write(
        self, tmp_project: Path
    ) -> None:
        """同一 write 内で 2 つの単数値 relation が矛盾→古い episode の supersede は 1 回だけ。"""
        char_id = _make_character(tmp_project)
        conn = _connect(tmp_project, char_id)
        old_now = datetime(2026, 1, 1, tzinfo=UTC)
        old_iso = old_now.isoformat()
        new_now = datetime(2026, 1, 2, tzinfo=UTC)
        new_iso = new_now.isoformat()

        # 古い episode (1 回の write で works_as + lives_in の 2 つがあった)
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃはプログラマで東京住み", old_iso),
        )
        old_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'person', ?)",
            ("みるちゃ", old_iso),
        )
        mirucha_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'work', ?)",
            ("プログラマ", old_iso),
        )
        prog_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'place', ?)",
            ("東京", old_iso),
        )
        tokyo_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        # mentions
        for eid in (mirucha_id, prog_id, tokyo_id):
            conn.execute(
                "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
                "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
                (old_ep_id, eid, old_iso),
            )
        # entity relations
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, prog_id, old_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'lives_in', ?)",
            (mirucha_id, tokyo_id, old_iso),
        )

        # 新しい episode (works_as + lives_in の両方を上書き)
        conn.execute(
            "INSERT INTO episodes (content, kind, importance, valid_from) "
            "VALUES (?, 'facts', 1, ?)",
            ("みるちゃはデザイナで大阪住み", new_iso),
        )
        new_ep_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'work', ?)",
            ("デザイナ", new_iso),
        )
        designer_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO entities (canonical_name, entity_type, valid_from) "
            "VALUES (?, 'place', ?)",
            ("大阪", new_iso),
        )
        osaka_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for eid in (mirucha_id, designer_id, osaka_id):
            if eid == mirucha_id:
                continue  # みるちゃ は既存 → 新たに mentions を貼る
            conn.execute(
                "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
                "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
                (new_ep_id, eid, new_iso),
            )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('episode', ?, 'entity', ?, 'mentions', ?)",
            (new_ep_id, mirucha_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'works_as', ?)",
            (mirucha_id, designer_id, new_iso),
        )
        conn.execute(
            "INSERT INTO relations (src_type, src_id, dst_type, dst_id, predicate, valid_from) "
            "VALUES ('entity', ?, 'entity', ?, 'lives_in', ?)",
            (mirucha_id, osaka_id, new_iso),
        )

        result = ExtractionResult(
            entities=[
                ExtractedEntity(canonical_name="みるちゃ", entity_type="person"),
                ExtractedEntity(canonical_name="デザイナ", entity_type="work"),
                ExtractedEntity(canonical_name="大阪", entity_type="place"),
            ],
            relations=[
                ExtractedRelation(
                    src="みるちゃ", dst="デザイナ", predicate="works_as"
                ),
                ExtractedRelation(
                    src="みるちゃ", dst="大阪", predicate="lives_in"
                ),
            ],
        )

        stats = detect_and_supersede(
            conn=conn,
            new_episode_id=new_ep_id,
            new_episode_kind="facts",
            result=result,
            name_to_id={
                "みるちゃ": mirucha_id,
                "デザイナ": designer_id,
                "大阪": osaka_id,
            },
            now=new_now,
        )

        # relations は 2 件 supersede、episodes は 1 件だけ
        assert stats == {"relations_superseded": 2, "episodes_superseded": 1}
