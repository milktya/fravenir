"""FastAPI app factory for the admin UI."""

from __future__ import annotations

import base64
import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint

from fravenir.admin.api import router
from fravenir.storage.paths import kv_db_path, vdb_entities_path

_log = structlog.get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"

# SEC-1 MEDIUM-5-2: 全レスポンスに付与するセキュリティヘッダ。
# - script-src 'self': inline script は cytoscape-bootstrap.js に分離済。
# - style-src 'self' 'unsafe-inline': cytoscape が動的に <style> element を
#   挿入するため要許可。
# - style-src-attr 'self': inline 属性スタイル (<span style="...">) は依然
#   遮断。XSS で position:fixed 等を流し込んでクリックジャックする主経路を防御。
_SECURITY_HEADERS: dict[str, str] = {
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "style-src-attr 'self'; "
        "img-src 'self' data:; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}

# Phase 6 B-2: HTTP Basic auth (うっかり防御)。
# 環境変数 FRAVENIR_ADMIN_USER と FRAVENIR_ADMIN_PASSWORD が両方セット
# されている場合のみ有効。片方/両方 unset なら認証なしで起動 (警告ログを出す)。
# Tailscale / LAN 限定運用前提なので、Basic auth で十分。
_AUTH_USER_ENV = "FRAVENIR_ADMIN_USER"
_AUTH_PASS_ENV = "FRAVENIR_ADMIN_PASSWORD"
_AUTH_REALM = "fravenir admin"


def _load_auth_credentials() -> tuple[str, str] | None:
    """Return (user, password) if both env vars set, else None."""
    user = os.environ.get(_AUTH_USER_ENV)
    password = os.environ.get(_AUTH_PASS_ENV)
    if user and password:
        return user, password
    return None


def _check_basic_auth(header: str | None, expected: tuple[str, str]) -> bool:
    """Constant-time check of `Authorization: Basic ...` header."""
    if not header or not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1], validate=True).decode(
            "utf-8", errors="strict"
        )
    except (ValueError, UnicodeDecodeError):
        return False
    if ":" not in decoded:
        return False
    got_user, got_pass = decoded.split(":", 1)
    exp_user, exp_pass = expected
    # 両方比較してから AND を取る (タイミング攻撃避け)
    ok_user = secrets.compare_digest(got_user.encode(), exp_user.encode())
    ok_pass = secrets.compare_digest(got_pass.encode(), exp_pass.encode())
    return ok_user and ok_pass


def create_app(character_id: str) -> FastAPI:
    """Return a FastAPI app bound to the given character's kv.sqlite."""

    kv_path = kv_db_path(character_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        if not kv_path.exists():
            raise RuntimeError(f"kv.sqlite not found: {kv_path}")
        app.state.kv_path = kv_path
        app.state.character_id = character_id
        app.state.vdb_entities_path = vdb_entities_path(character_id)
        # Embedder は遅延初期化 (PATCH /entities/{id} 等 description が変わった時のみ)
        app.state.embedder = None
        _log.info("admin_server_started", character_id=character_id, kv_path=str(kv_path))
        yield

    app = FastAPI(title=f"fravenir admin — {character_id}", lifespan=lifespan)

    auth_creds = _load_auth_credentials()
    if auth_creds is None:
        _log.warning(
            "admin_server_auth_disabled",
            reason=f"either {_AUTH_USER_ENV} or {_AUTH_PASS_ENV} is unset; "
            "running without authentication (LAN/Tailscale-only deployments only)",
        )
    else:
        _log.info("admin_server_auth_enabled")

    @app.middleware("http")
    async def _basic_auth(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if auth_creds is None:
            return await call_next(request)
        if _check_basic_auth(request.headers.get("Authorization"), auth_creds):
            return await call_next(request)
        return Response(
            status_code=401,
            content="Unauthorized",
            headers={"WWW-Authenticate": f'Basic realm="{_AUTH_REALM}"'},
        )

    @app.middleware("http")
    async def _security_headers(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    app.include_router(router, prefix="/api")

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app
