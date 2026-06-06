"""Unit tests for CLI commands."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
import sqlite_vec
from click.testing import CliRunner

from fravenir.cli import main
from fravenir.embedding import Embedder
from fravenir.storage.paths import (
    data_dir,
    kv_db_path,
    seed_yaml_path,
    vdb_entities_path,
)


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    """Avoid loading ruri-v3 in CLI tests — use a cheap zero-vector stub."""

    def fake_encode(self, texts):
        return np.zeros((len(texts), 768), dtype=np.float32)

    monkeypatch.setattr(Embedder, "_encode", fake_encode)


def _write_seed_with_personality(seed_path, character_id: str = "mina") -> None:
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(
        f"identity:\n"
        f"  canonical_name: {character_id}\n"
        f"  aliases: [あたし, ミナ]\n"
        f"  description: \"技術好きな女の子\"\n"
        f"personality:\n"
        f"  - canonical_name: 好奇心旺盛\n"
        f"    entity_type: concept\n"
        f"    description: \"仕組みの話が好き\"\n"
        f"    self_weight: 0.8\n"
        f"initial_episodes:\n"
        f"  - content: \"あたしは{character_id}。仕組みを覗き込むのが好き。\"\n"
        f"    kind: facts\n"
        f"    importance: 3\n",
        encoding="utf-8",
    )


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mina(tmp_project, runner):
    """Create character 'mina' in tmp_project and return the runner."""
    result = runner.invoke(main, ["create-character", "mina"])
    assert result.exit_code == 0, result.output
    return runner


class TestListCharacters:
    def test_no_characters(self, tmp_project, runner):
        result = runner.invoke(main, ["list-characters"])
        assert result.exit_code == 0
        assert "No characters found" in result.output

    def test_lists_created_character(self, mina, runner):
        result = runner.invoke(main, ["list-characters"])
        assert result.exit_code == 0
        assert "mina" in result.output


class TestShowCharacter:
    def test_not_found(self, tmp_project, runner):
        result = runner.invoke(main, ["show-character", "ghost"])
        assert result.exit_code == 1

    def test_shows_details(self, mina, runner):
        result = runner.invoke(main, ["show-character", "mina"])
        assert result.exit_code == 0
        assert "mina" in result.output
        assert "Entities" in result.output
        assert "Episodes" in result.output


class TestDeleteCharacter:
    def test_requires_confirmation(self, mina, runner):
        result = runner.invoke(main, ["delete-character", "mina"], input="N\n")
        assert result.exit_code != 0
        assert data_dir("mina").exists()

    def test_force_deletes(self, mina, runner):
        result = runner.invoke(main, ["delete-character", "mina", "--force"])
        assert result.exit_code == 0
        assert not data_dir("mina").exists()

    def test_confirms_and_deletes(self, mina, runner):
        result = runner.invoke(main, ["delete-character", "mina"], input="y\n")
        assert result.exit_code == 0
        assert not data_dir("mina").exists()

    def test_not_found(self, tmp_project, runner):
        result = runner.invoke(main, ["delete-character", "ghost", "--force"])
        assert result.exit_code == 1


class TestCharacterIdValidation:
    @pytest.mark.parametrize(
        "bad_id",
        ["../evil", "/abs/path", "foo/bar", "with space"],
    )
    def test_create_rejects_invalid_id(self, tmp_project, runner, bad_id):
        result = runner.invoke(main, ["create-character", bad_id])
        assert result.exit_code != 0
        assert "must match" in result.output

    def test_delete_rejects_path_traversal(self, tmp_project, runner):
        result = runner.invoke(
            main, ["delete-character", "../../../tmp/evil", "--force"]
        )
        assert result.exit_code != 0
        assert "must match" in result.output


class TestInitCharacter:
    def test_not_found(self, tmp_project, runner):
        result = runner.invoke(main, ["init-character", "ghost", "--force"])
        assert result.exit_code == 1

    def test_requires_seed_yaml(self, mina, runner):
        result = runner.invoke(
            main, ["init-character", "mina", "--force", "--seed", "/nonexistent.yaml"]
        )
        assert result.exit_code == 1

    def test_diff_apply(self, mina, runner):
        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        ep_count_before = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()

        # create a seed.yaml with a new episode
        seed_path = seed_yaml_path("mina")
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            "identity:\n  canonical_name: mina\n  aliases: []\n  description: ''\n"
            "initial_episodes:\n"
            "  - content: '新しいエピソード'\n    kind: facts\n    importance: 2\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["init-character", "mina", "--force"])
        assert result.exit_code == 0

        conn = sqlite3.connect(kv)
        ep_count_after = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        assert ep_count_after == ep_count_before + 1

    def test_duplicate_entity_skipped(self, mina, runner):
        seed_path = seed_yaml_path("mina")
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            "identity:\n  canonical_name: mina\n  aliases: []\n  description: ''\n"
            "initial_episodes: []\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["init-character", "mina", "--force"])
        assert result.exit_code == 0
        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        entity_count = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE canonical_name = 'mina'"
        ).fetchone()[0]
        conn.close()
        assert entity_count == 1

    def test_extract_episodes_default_creates_doc_status(
        self, mina, runner, monkeypatch
    ):
        """--extract-episodes は default ON。seed の episode に doc_status 行ができる。"""
        from fravenir.core.extraction import (
            ExtractedEntity,
            ExtractionResult,
        )

        class _MockExtractionClient:
            def __init__(self, _config):
                pass

            def extract(self, _content, entity_types=None, predicates=None):
                return ExtractionResult(
                    entities=[
                        ExtractedEntity(canonical_name="ミナ", entity_type="person"),
                    ],
                    relations=[],
                )

        monkeypatch.setattr(
            "fravenir.core.extraction.ExtractionClient", _MockExtractionClient
        )

        seed_path = seed_yaml_path("mina")
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            "identity:\n  canonical_name: mina\n  aliases: []\n  description: ''\n"
            "initial_episodes:\n"
            "  - content: 'extract モードのエピソード'\n"
            "    kind: facts\n"
            "    importance: 2\n",
            encoding="utf-8",
        )
        result = runner.invoke(main, ["init-character", "mina", "--force"])
        assert result.exit_code == 0, result.output

        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        try:
            row = conn.execute(
                "SELECT ds.stage, ds.error FROM episodes e"
                " JOIN doc_status ds ON ds.episode_id = e.id"
                " WHERE e.content = ?",
                ("extract モードのエピソード",),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "done"
        assert row[1] is None

    def test_no_extract_episodes_skips_doc_status(self, mina, runner):
        """--no-extract-episodes は従来挙動。doc_status 行は作られない。"""
        seed_path = seed_yaml_path("mina")
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_path.write_text(
            "identity:\n  canonical_name: mina\n  aliases: []\n  description: ''\n"
            "initial_episodes:\n"
            "  - content: 'legacy モードのエピソード'\n"
            "    kind: facts\n"
            "    importance: 2\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            main, ["init-character", "mina", "--force", "--no-extract-episodes"]
        )
        assert result.exit_code == 0, result.output

        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        try:
            ep_id = conn.execute(
                "SELECT id FROM episodes WHERE content = ?",
                ("legacy モードのエピソード",),
            ).fetchone()[0]
            ds_row = conn.execute(
                "SELECT 1 FROM doc_status WHERE episode_id = ?",
                (ep_id,),
            ).fetchone()
            # 従来挙動: identity (mina) への mentions が手動で1本張られている
            identity_id = conn.execute(
                "SELECT id FROM entities WHERE canonical_name='mina' AND is_self=1"
            ).fetchone()[0]
            mentions = conn.execute(
                "SELECT 1 FROM relations"
                " WHERE src_type='episode' AND src_id=?"
                " AND dst_type='entity' AND dst_id=? AND predicate='mentions'",
                (ep_id, identity_id),
            ).fetchone()
        finally:
            conn.close()
        assert ds_row is None
        assert mentions is not None


class TestCreateCharacterPersonality:
    """Phase 2: seed.personality is registered as non-self entities + part_of/mentions relations."""

    def test_personality_registered_as_entities(self, tmp_project, runner):
        _write_seed_with_personality(seed_yaml_path("mina"))
        result = runner.invoke(main, ["create-character", "mina"])
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(kv_db_path("mina"))
        try:
            rows = conn.execute(
                "SELECT canonical_name, is_self, self_weight, decay_rate"
                " FROM entities ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

        by_name = {r[0]: r for r in rows}
        assert "mina" in by_name and by_name["mina"][1] == 1  # identity is_self=1
        assert by_name["mina"][3] == 0.2  # self decay
        assert "好奇心旺盛" in by_name
        name, is_self, self_weight, decay_rate = by_name["好奇心旺盛"]
        assert is_self == 0
        assert self_weight == 0.8
        assert decay_rate == 0.3  # personality decay

    def test_part_of_relation_personality_to_identity(self, tmp_project, runner):
        _write_seed_with_personality(seed_yaml_path("mina"))
        runner.invoke(main, ["create-character", "mina"])

        conn = sqlite3.connect(kv_db_path("mina"))
        try:
            identity_id = conn.execute(
                "SELECT id FROM entities WHERE canonical_name='mina' AND is_self=1"
            ).fetchone()[0]
            p_id = conn.execute(
                "SELECT id FROM entities WHERE canonical_name='好奇心旺盛'"
            ).fetchone()[0]
            rel = conn.execute(
                "SELECT predicate FROM relations"
                " WHERE src_type='entity' AND src_id=?"
                "   AND dst_type='entity' AND dst_id=? AND valid_to IS NULL",
                (p_id, identity_id),
            ).fetchone()
        finally:
            conn.close()
        assert rel is not None
        assert rel[0] == "part_of"

    def test_mentions_relation_episode_to_identity(self, tmp_project, runner):
        _write_seed_with_personality(seed_yaml_path("mina"))
        runner.invoke(main, ["create-character", "mina"])

        conn = sqlite3.connect(kv_db_path("mina"))
        try:
            identity_id = conn.execute(
                "SELECT id FROM entities WHERE canonical_name='mina' AND is_self=1"
            ).fetchone()[0]
            ep_id = conn.execute("SELECT id FROM episodes LIMIT 1").fetchone()[0]
            rel = conn.execute(
                "SELECT predicate FROM relations"
                " WHERE src_type='episode' AND src_id=?"
                "   AND dst_type='entity' AND dst_id=? AND valid_to IS NULL",
                (ep_id, identity_id),
            ).fetchone()
        finally:
            conn.close()
        assert rel is not None
        assert rel[0] == "mentions"

    def test_init_character_adds_missing_personality(self, mina, runner):
        # mina was created with default seed (empty personality)
        _write_seed_with_personality(seed_yaml_path("mina"))
        result = runner.invoke(main, ["init-character", "mina", "--force"])
        assert result.exit_code == 0, result.output

        conn = sqlite3.connect(kv_db_path("mina"))
        try:
            p_count = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE canonical_name='好奇心旺盛'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert p_count == 1

    def test_init_character_is_idempotent_for_relations(self, tmp_project, runner):
        _write_seed_with_personality(seed_yaml_path("mina"))
        runner.invoke(main, ["create-character", "mina"])

        conn = sqlite3.connect(kv_db_path("mina"))
        rel_before = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        conn.close()

        result = runner.invoke(main, ["init-character", "mina", "--force"])
        assert result.exit_code == 0

        conn = sqlite3.connect(kv_db_path("mina"))
        rel_after = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        conn.close()
        assert rel_before == rel_after


class TestCompact:
    def test_outputs_json_with_expected_keys(self, mina, runner):
        result = runner.invoke(main, ["compact", "mina"])
        assert result.exit_code == 0
        # ログに混ざる JSON 部分だけ拾う
        json_text = result.output[result.output.index("{") :]
        payload = json.loads(json_text)
        assert set(payload.keys()) == {
            "fan_out_updated",
            "strength_updated",
            "suppressed",
            "merge_candidates",
            "self_loops_archived",
            "duration_ms",
            "dry_run",
        }
        assert payload["dry_run"] is False

    def test_dry_run_flag(self, mina, runner):
        result = runner.invoke(main, ["compact", "mina", "--dry-run"])
        assert result.exit_code == 0
        json_text = result.output[result.output.index("{") :]
        payload = json.loads(json_text)
        assert payload["dry_run"] is True

    def test_not_found(self, tmp_project, runner):
        result = runner.invoke(main, ["compact", "ghost"])
        assert result.exit_code == 1


def _open_vdb_entities(character_id: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(vdb_entities_path(character_id)))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class TestCreateCharacterVdbSeeding:
    def test_vdb_entities_db_created(self, mina):
        assert vdb_entities_path("mina").exists()

    def test_identity_embedded_in_vdb_entities(self, mina):
        conn = _open_vdb_entities("mina")
        try:
            rows = conn.execute("SELECT entity_id FROM vdb_entities").fetchall()
        finally:
            conn.close()
        # default seed: identity 1件のみ（personality 空）
        assert len(rows) == 1

    def test_personality_seed_embedded(self, tmp_project, runner):
        _write_seed_with_personality(seed_yaml_path("mina"))
        result = runner.invoke(main, ["create-character", "mina"])
        assert result.exit_code == 0, result.output

        conn = _open_vdb_entities("mina")
        try:
            rows = conn.execute("SELECT entity_id FROM vdb_entities").fetchall()
        finally:
            conn.close()
        # identity + personality 1件 → 2件
        assert len(rows) == 2


class TestInitCharacterVdbSeeding:
    def test_new_personality_embedded_in_vdb(self, mina, runner):
        # mina はデフォルト seed で作成済み（identity のみ vdb に入ってる）
        conn = _open_vdb_entities("mina")
        before = conn.execute("SELECT COUNT(*) FROM vdb_entities").fetchone()[0]
        conn.close()

        _write_seed_with_personality(seed_yaml_path("mina"))
        result = runner.invoke(main, ["init-character", "mina", "--force"])
        assert result.exit_code == 0, result.output

        conn = _open_vdb_entities("mina")
        try:
            after = conn.execute("SELECT COUNT(*) FROM vdb_entities").fetchone()[0]
        finally:
            conn.close()
        # personality 1件が追加されるはず
        assert after == before + 1


class TestExportImport:
    def test_export_creates_json(self, mina, tmp_project, runner):
        out = tmp_project / "mina_export.json"
        result = runner.invoke(main, ["export", "mina", "--out", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()
        payload = json.loads(out.read_text())
        assert payload["format_version"] == "1"
        assert payload["dim"] == 768
        assert payload["character_id"] == "mina"
        assert isinstance(payload["episodes"], list)

    def test_import_roundtrip(self, mina, tmp_project, runner):
        out = tmp_project / "mina_export.json"
        runner.invoke(main, ["export", "mina", "--out", str(out)])

        runner.invoke(main, ["delete-character", "mina", "--force"])
        assert not data_dir("mina").exists()

        result = runner.invoke(main, ["import", str(out), "mina"])
        assert result.exit_code == 0, result.output
        assert data_dir("mina").exists()

        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        ep_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        assert ep_count > 0

    def test_import_fails_if_exists(self, mina, tmp_project, runner):
        out = tmp_project / "mina_export.json"
        runner.invoke(main, ["export", "mina", "--out", str(out)])
        result = runner.invoke(main, ["import", str(out), "mina"])
        assert result.exit_code == 1

    def test_import_overwrite(self, mina, tmp_project, runner):
        out = tmp_project / "mina_export.json"
        runner.invoke(main, ["export", "mina", "--out", str(out)])
        result = runner.invoke(main, ["import", str(out), "mina", "--overwrite"])
        assert result.exit_code == 0

    def test_import_dim_mismatch(self, tmp_project, runner):
        bad_json = tmp_project / "bad.json"
        bad_json.write_text(
            '{"format_version": "1", "dim": 256, "character_id": "x", "episodes": []}',
            encoding="utf-8",
        )
        result = runner.invoke(main, ["import", str(bad_json), "x"])
        assert result.exit_code == 1
        assert "dim mismatch" in result.output

    def test_import_skips_self_ref_relation(self, mina, tmp_project, runner):
        """古い export に残っていた src==dst の relation は import 時に skip される。"""
        out = tmp_project / "mina_export.json"
        runner.invoke(main, ["export", "mina", "--out", str(out)])

        payload = json.loads(out.read_text(encoding="utf-8"))
        # 既存 entity を一つ拾って self-ref relation を payload に注入
        target_eid = payload["entities"][0]["id"]
        payload["relations"].append({
            "id": 99999,
            "src_type": "entity",
            "src_id": target_eid,
            "dst_type": "entity",
            "dst_id": target_eid,
            "predicate": "is_a",
            "strength": 1.0,
            "fan_out": 1,
            "description": "self-loop legacy",
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_to": None,
            "supersedes": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        })
        out.write_text(json.dumps(payload), encoding="utf-8")

        runner.invoke(main, ["delete-character", "mina", "--force"])
        result = runner.invoke(main, ["import", str(out), "mina"])
        assert result.exit_code == 0, result.output

        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM relations "
                "WHERE src_type = dst_type AND src_id = dst_id"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 0


class TestResolveCommands:
    def test_resolve_list_empty(self, mina, runner):
        result = runner.invoke(main, ["resolve", "list", "mina"])
        assert result.exit_code == 0
        assert "No unresolved merge_candidates" in result.output

    def test_resolve_merge_candidate(self, mina, runner):
        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        try:
            conn.execute(
                "INSERT INTO entities (id, canonical_name, entity_type, valid_from) "
                "VALUES (10, 'A', 'concept', '2026-01-01'), (11, 'B', 'concept', '2026-01-01')"
            )
            conn.execute(
                "INSERT INTO merge_candidates (entity_a, entity_b, similarity, resolved) "
                "VALUES (10, 11, 0.95, 0)"
            )
            conn.commit()
        finally:
            conn.close()

        result = runner.invoke(main, ["resolve", "merge", "mina", "1", "--yes"])
        assert result.exit_code == 0
        assert "Merged candidate 1" in result.output

    def test_resolve_reject_candidate(self, mina, runner):
        kv = kv_db_path("mina")
        conn = sqlite3.connect(kv)
        try:
            conn.execute(
                "INSERT INTO entities (id, canonical_name, entity_type, valid_from) "
                "VALUES (20, 'C', 'concept', '2026-01-01'), (21, 'D', 'concept', '2026-01-01')"
            )
            conn.execute(
                "INSERT INTO merge_candidates (entity_a, entity_b, similarity, resolved) "
                "VALUES (20, 21, 0.95, 0)"
            )
            conn.commit()
        finally:
            conn.close()

        result = runner.invoke(main, ["resolve", "reject", "mina", "1", "--yes"])
        assert result.exit_code == 0
        assert "Rejected candidate 1" in result.output


class TestMigrateCommands:
    def test_migrate_judge_columns_dry_run(self, mina, runner):
        result = runner.invoke(main, ["migrate", "judge-columns", "mina", "--dry-run"])
        assert result.exit_code == 0
        assert "dry-run" in result.output

    def test_migrate_session_id_idempotent(self, mina, runner):
        # 2回実行してもエラーにならない
        result1 = runner.invoke(main, ["migrate", "session-id", "mina"])
        assert result1.exit_code == 0
        result2 = runner.invoke(main, ["migrate", "session-id", "mina"])
        assert result2.exit_code == 0
        assert "already migrated" in result2.output or "Nothing to do" in result2.output
