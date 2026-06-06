"""FastAPI TestClient tests for admin UI endpoints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fravenir.admin.server import create_app
from fravenir.storage.sqlite_init import init_kv

NOW = "2026-04-29T12:00:00"
THEN = "2026-04-01T00:00:00"
EP1_CONTENT = "Phase 5 全完了レポートの内容はこちら、詳細は別ドキュメントを参照のこと"


def _seed(tmp_path: Path) -> Path:
    db = tmp_path / "kv.sqlite"
    init_kv(db)
    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            "INSERT INTO episodes (id, content, kind, importance, valid_from, valid_to,"
            " is_suppressed, last_activated_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, EP1_CONTENT, "facts", 3, NOW, None, 0, NOW, NOW),
                (2, "古いエピソード", "state", 1, THEN, NOW, 0, THEN, THEN),
                (4, "孤立エピソード（リレーションなし）", "facts", 1, NOW, None, 0, NOW, NOW),
            ],
        )
        conn.executemany(
            "INSERT INTO entities"
            " (id, canonical_name, entity_type, is_self, self_weight,"
            " decay_rate, valid_from, last_activated_at, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "ミナ", "person", 1, 1.0, 0.2, NOW, NOW, NOW),
                (2, "ProjectX", "concept", 0, 0.0, 0.5, NOW, NOW, NOW),
            ],
        )
        conn.executemany(
            "INSERT INTO relations (id, src_type, src_id, dst_type, dst_id, predicate,"
            " strength, valid_from, valid_to) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "episode", 1, "entity", 1, "mentions", 1.0, NOW, None),
                (3, "entity", 1, "entity", 2, "hosts", 0.8, NOW, None),
            ],
        )
        conn.executemany(
            "INSERT INTO merge_candidates (id, entity_a, entity_b, similarity,"
            " detected_at, resolved) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 2, 0.92, NOW, 0),
                (2, 1, 2, 0.85, NOW, 1),
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
def client(tmp_path: Path, monkeypatch) -> TestClient:  # type: ignore[type-arg]
    import fravenir.storage.paths as paths_mod

    monkeypatch.setattr(paths_mod, "_project_root", lambda: tmp_path)
    (tmp_path / "data" / "test_char").mkdir(parents=True)
    _seed(tmp_path / "data" / "test_char")
    app = create_app("test_char")
    # lifespan must run; use the context-manager form of TestClient
    with TestClient(app) as c:
        return c


class TestIndexRoute:
    def test_returns_html(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


class TestSecurityHeaders:
    """SEC-1 MEDIUM-5-2: 全レスポンスに付与するセキュリティヘッダの検証。"""

    def test_index_has_security_headers(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["Referrer-Policy"] == "no-referrer"
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "object-src 'none'" in csp
        # style-src は cytoscape の動的 <style> 用に 'unsafe-inline' を許可
        # しているが、inline 属性スタイル (XSS の主経路) は style-src-attr
        # 'self' で依然遮断する。
        assert "style-src-attr 'self'" in csp

    def test_api_route_also_has_security_headers(self, client: TestClient) -> None:
        resp = client.get("/api/stats")
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "Content-Security-Policy" in resp.headers


class TestStats:
    def test_200_with_expected_keys(self, client: TestClient) -> None:
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {
            "episodes", "entities", "relations", "merge_candidates",
            "doc_status_failed", "orphans",
        }

    def test_episode_counts(self, client: TestClient) -> None:
        body = client.get("/api/stats").json()
        assert body["episodes"]["total"] == 3


class TestGraph:
    def test_active_scope_200(self, client: TestClient) -> None:
        resp = client.get("/api/graph?scope=active")
        assert resp.status_code == 200
        body = resp.json()
        assert body["scope"] == "active"
        assert "elements" in body
        assert "nodes" in body["elements"]
        assert "edges" in body["elements"]

    def test_all_scope_200(self, client: TestClient) -> None:
        resp = client.get("/api/graph?scope=all")
        assert resp.status_code == 200

    def test_invalid_scope_422(self, client: TestClient) -> None:
        resp = client.get("/api/graph?scope=invalid")
        assert resp.status_code == 422


class TestEpisodeDetail:
    def test_existing_episode_200(self, client: TestClient) -> None:
        resp = client.get("/api/episodes/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 1
        assert body["kind"] == "facts"
        assert "mentions" in body

    def test_nonexistent_episode_404(self, client: TestClient) -> None:
        resp = client.get("/api/episodes/9999")
        assert resp.status_code == 404


class TestEntityDetail:
    def test_existing_entity_200(self, client: TestClient) -> None:
        resp = client.get("/api/entities/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == 1
        assert body["canonical_name"] == "ミナ"

    def test_nonexistent_entity_404(self, client: TestClient) -> None:
        resp = client.get("/api/entities/9999")
        assert resp.status_code == 404


class TestRelationDetail:
    def test_existing_relation_200(self, client: TestClient) -> None:
        resp = client.get("/api/relations/3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["predicate"] == "hosts"

    def test_nonexistent_relation_404(self, client: TestClient) -> None:
        resp = client.get("/api/relations/9999")
        assert resp.status_code == 404


class TestMergeCandidates:
    def test_pending_200(self, client: TestClient) -> None:
        resp = client.get("/api/merge_candidates?status=pending")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status_filter"] == "pending"
        assert isinstance(body["candidates"], list)

    def test_invalid_status_422(self, client: TestClient) -> None:
        resp = client.get("/api/merge_candidates?status=invalid")
        assert resp.status_code == 422


class TestDocStatus:
    def test_failed_200(self, client: TestClient) -> None:
        resp = client.get("/api/doc_status?status=failed")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status_filter"] == "failed"
        assert len(body["items"]) == 1


class TestOrphans:
    def test_active_scope_200(self, client: TestClient) -> None:
        resp = client.get("/api/orphans?scope=active")
        assert resp.status_code == 200
        body = resp.json()
        assert body["scope"] == "active"
        assert "episodes" in body
        assert "entities" in body


# ─── Phase 6: PATCH /entities/{id} + audit log ──────────────────────────────


@pytest.fixture
def client_no_embed(tmp_path: Path, monkeypatch) -> TestClient:  # type: ignore[type-arg]
    """PATCH 系テスト用。description 変更時の vdb 再エンベディングを no-op に差し替える。

    sentence-transformers モデルのロードを避けて高速化。エンベディングの正しさは
    別ファイル (test_vector.py 等) で検証済。
    """
    import fravenir.admin.api as api_mod
    import fravenir.storage.paths as paths_mod

    monkeypatch.setattr(paths_mod, "_project_root", lambda: tmp_path)
    (tmp_path / "data" / "test_char").mkdir(parents=True)
    _seed(tmp_path / "data" / "test_char")
    monkeypatch.setattr(api_mod, "_reembed_entity", lambda *a, **kw: None)
    app = create_app("test_char")
    with TestClient(app) as c:
        return c


class TestEntityUpdate:
    def test_description_only(self, client_no_embed: TestClient) -> None:
        resp = client_no_embed.patch(
            "/api/entities/2", json={"description": "Project X の説明文"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["changed"] is True
        assert body["after"]["description"] == "Project X の説明文"
        assert body["curated_at"] is not None

        # GET で curated_at が立ち、description が反映されていることを確認
        detail = client_no_embed.get("/api/entities/2").json()
        assert detail["description"] == "Project X の説明文"
        assert detail["curated_at"] is not None

    def test_aliases_only(self, client_no_embed: TestClient) -> None:
        resp = client_no_embed.patch(
            "/api/entities/2", json={"aliases": ["プロX", "Px"]}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["changed"] is True
        assert sorted(body["after"]["aliases"]) == ["Px", "プロX"]

        detail = client_no_embed.get("/api/entities/2").json()
        assert sorted(detail["aliases"]) == ["Px", "プロX"]

    def test_aliases_replace_not_append(self, client_no_embed: TestClient) -> None:
        client_no_embed.patch("/api/entities/2", json={"aliases": ["a", "b"]})
        # 後続の PATCH は完全置換のはず
        resp = client_no_embed.patch("/api/entities/2", json={"aliases": ["c"]})
        assert resp.status_code == 200
        detail = client_no_embed.get("/api/entities/2").json()
        assert detail["aliases"] == ["c"]

    def test_aliases_empty_clears_all(self, client_no_embed: TestClient) -> None:
        client_no_embed.patch("/api/entities/2", json={"aliases": ["x"]})
        resp = client_no_embed.patch("/api/entities/2", json={"aliases": []})
        assert resp.status_code == 200
        detail = client_no_embed.get("/api/entities/2").json()
        assert detail["aliases"] == []

    def test_empty_body_400(self, client_no_embed: TestClient) -> None:
        resp = client_no_embed.patch("/api/entities/2", json={})
        assert resp.status_code == 400

    def test_nonexistent_404(self, client_no_embed: TestClient) -> None:
        resp = client_no_embed.patch(
            "/api/entities/9999", json={"description": "x"}
        )
        assert resp.status_code == 404

    def test_archived_entity_404(self, client_no_embed: TestClient, tmp_path: Path) -> None:
        # entity 2 を archive (valid_to を立てる) してから PATCH すると 404
        db = tmp_path / "data" / "test_char" / "kv.sqlite"
        conn = sqlite3.connect(db)
        try:
            conn.execute(
                "UPDATE entities SET valid_to = ? WHERE id = 2", (NOW,)
            )
            conn.commit()
        finally:
            conn.close()
        resp = client_no_embed.patch(
            "/api/entities/2", json={"description": "x"}
        )
        assert resp.status_code == 404

    def test_noop_when_values_unchanged(self, client_no_embed: TestClient) -> None:
        # 初回 PATCH で description セット
        first = client_no_embed.patch(
            "/api/entities/2", json={"description": "same text"}
        ).json()
        assert first["changed"] is True
        # 同じ値を再 PATCH → changed=False、curated_at 更新されない
        second = client_no_embed.patch(
            "/api/entities/2", json={"description": "same text"}
        ).json()
        assert second["changed"] is False
        assert second["curated_at"] is None

    def test_aliases_dedup_and_trim(self, client_no_embed: TestClient) -> None:
        resp = client_no_embed.patch(
            "/api/entities/2", json={"aliases": [" a ", "a", "b", "", "  "]}
        )
        assert resp.status_code == 200
        detail = client_no_embed.get("/api/entities/2").json()
        assert sorted(detail["aliases"]) == ["a", "b"]

    def test_description_too_long_422(self, client_no_embed: TestClient) -> None:
        big = "x" * 4001
        resp = client_no_embed.patch(
            "/api/entities/2", json={"description": big}
        )
        # Pydantic max_length=4000 で 422
        assert resp.status_code == 422


class TestAuditLog:
    def test_records_entity_update(self, client_no_embed: TestClient) -> None:
        client_no_embed.patch(
            "/api/entities/2", json={"description": "audit テスト用"}
        )
        resp = client_no_embed.get("/api/audit_log")
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert len(entries) >= 1
        first = entries[0]
        assert first["action"] == "entity.update"
        assert first["target_type"] == "entity"
        assert first["target_id"] == 2
        assert first["after"]["description"] == "audit テスト用"
        assert first["actor"] == "admin_ui"

    def test_filter_by_target(self, client_no_embed: TestClient) -> None:
        client_no_embed.patch("/api/entities/2", json={"description": "for 2"})
        # entity 1 への更新は別途、actorは同じ admin_ui
        client_no_embed.patch(
            "/api/entities/1", json={"description": "for 1 (self)"}
        )
        resp = client_no_embed.get(
            "/api/audit_log?target_type=entity&target_id=1"
        )
        assert resp.status_code == 200
        entries = resp.json()["entries"]
        assert all(e["target_id"] == 1 for e in entries)
        assert any(e["after"]["description"] == "for 1 (self)" for e in entries)

    def test_no_op_does_not_audit(self, client_no_embed: TestClient) -> None:
        client_no_embed.patch("/api/entities/2", json={"description": "init"})
        before = len(client_no_embed.get("/api/audit_log").json()["entries"])
        client_no_embed.patch("/api/entities/2", json={"description": "init"})  # no-op
        after = len(client_no_embed.get("/api/audit_log").json()["entries"])
        assert after == before


# ─── Phase 6 B-2: HTTP Basic auth ──────────────────────────────────────────


@pytest.fixture
def client_with_auth(tmp_path: Path, monkeypatch) -> TestClient:  # type: ignore[type-arg]
    """env var をセットして basic auth を有効にしたクライアント。"""
    import fravenir.storage.paths as paths_mod

    monkeypatch.setattr(paths_mod, "_project_root", lambda: tmp_path)
    (tmp_path / "data" / "test_char").mkdir(parents=True)
    _seed(tmp_path / "data" / "test_char")
    monkeypatch.setenv("FRAVENIR_ADMIN_USER", "alice")
    monkeypatch.setenv("FRAVENIR_ADMIN_PASSWORD", "s3cret")
    app = create_app("test_char")
    with TestClient(app) as c:
        return c


class TestBasicAuth:
    def test_no_auth_returns_401(self, client_with_auth: TestClient) -> None:
        resp = client_with_auth.get("/api/stats")
        assert resp.status_code == 401
        assert "Basic" in resp.headers.get("WWW-Authenticate", "")

    def test_correct_credentials_pass(self, client_with_auth: TestClient) -> None:
        resp = client_with_auth.get("/api/stats", auth=("alice", "s3cret"))
        assert resp.status_code == 200

    def test_wrong_password_401(self, client_with_auth: TestClient) -> None:
        resp = client_with_auth.get("/api/stats", auth=("alice", "wrong"))
        assert resp.status_code == 401

    def test_wrong_user_401(self, client_with_auth: TestClient) -> None:
        resp = client_with_auth.get("/api/stats", auth=("bob", "s3cret"))
        assert resp.status_code == 401

    def test_malformed_header_401(self, client_with_auth: TestClient) -> None:
        resp = client_with_auth.get(
            "/api/stats", headers={"Authorization": "Bearer xyz"}
        )
        assert resp.status_code == 401

    def test_static_also_protected(self, client_with_auth: TestClient) -> None:
        # 静的アセットも認証対象 (index page も含む)
        resp = client_with_auth.get("/")
        assert resp.status_code == 401

    def test_patch_requires_auth(self, client_with_auth: TestClient) -> None:
        resp = client_with_auth.patch(
            "/api/entities/2", json={"description": "x"}
        )
        assert resp.status_code == 401

    def test_disabled_when_only_user_set(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """片方だけ set だと無認証 (デフォルト動作) に倒れる。"""
        import fravenir.storage.paths as paths_mod

        monkeypatch.setattr(paths_mod, "_project_root", lambda: tmp_path)
        (tmp_path / "data" / "test_char").mkdir(parents=True)
        _seed(tmp_path / "data" / "test_char")
        monkeypatch.setenv("FRAVENIR_ADMIN_USER", "alice")
        monkeypatch.delenv("FRAVENIR_ADMIN_PASSWORD", raising=False)
        app = create_app("test_char")
        with TestClient(app) as c:
            resp = c.get("/api/stats")
            assert resp.status_code == 200

    def test_disabled_when_both_unset(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """env var なし → 認証なしで動く (既存挙動互換)。"""
        import fravenir.storage.paths as paths_mod

        monkeypatch.setattr(paths_mod, "_project_root", lambda: tmp_path)
        (tmp_path / "data" / "test_char").mkdir(parents=True)
        _seed(tmp_path / "data" / "test_char")
        monkeypatch.delenv("FRAVENIR_ADMIN_USER", raising=False)
        monkeypatch.delenv("FRAVENIR_ADMIN_PASSWORD", raising=False)
        app = create_app("test_char")
        with TestClient(app) as c:
            resp = c.get("/api/stats")
            assert resp.status_code == 200
