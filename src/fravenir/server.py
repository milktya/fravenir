"""MCP server entry point — FastMCP wrapper exposing memory tools.

サーバー1プロセス = 1キャラ（`data/{character_id}/` を掴む）。
ツール名にキャラ名は含めず、サーバー名（mcp.json の展開名）で分離する。
"""

from __future__ import annotations

from typing import Literal

import structlog
from mcp.server.fastmcp import FastMCP

from fravenir.core.delete import memory_delete as _core_delete
from fravenir.core.explore import memory_explore as _core_explore
from fravenir.core.extraction import ExtractionClient
from fravenir.core.get import memory_get as _core_get
from fravenir.core.search import memory_search as _core_search
from fravenir.core.trace import memory_trace as _core_trace
from fravenir.core.write import memory_write as _core_write
from fravenir.embedding import Embedder
from fravenir.schemas.config import AppConfig

_logger = structlog.get_logger(__name__)


def build_server(
    config: AppConfig,
    embedder: Embedder | None = None,
    extraction_client: ExtractionClient | None = None,
    host: str | None = None,
    port: int | None = None,
) -> FastMCP:
    """Build a FastMCP server bound to a specific character.

    Args:
        config: Validated AppConfig for the target character.
        embedder: Optional Embedder. If None, a new one is created from config.
            Tests inject a stubbed embedder to avoid loading the real model.
        extraction_client: Optional ExtractionClient. If None and
            config.extraction.enabled, a new client is created. Tests inject
            a mock to avoid hitting a real LLM endpoint.
        host: Bind host for HTTP transports. Passed to FastMCP at construction
            time so DNS-rebinding protection is auto-selected (on for localhost,
            off otherwise). If None, FastMCP default (127.0.0.1) is used.
        port: Bind port for HTTP transports. If None, FastMCP default is used.
    """
    character_id = config.character.id
    emb = embedder if embedder is not None else Embedder(config.embedding)

    ext: ExtractionClient | None
    if extraction_client is not None:
        ext = extraction_client
    elif config.extraction.enabled:
        ext = ExtractionClient(config.extraction)
    else:
        ext = None

    fastmcp_kwargs: dict[str, object] = {"name": f"fravenir_{character_id}"}
    if host is not None:
        fastmcp_kwargs["host"] = host
    if port is not None:
        fastmcp_kwargs["port"] = port
    mcp: FastMCP = FastMCP(**fastmcp_kwargs)  # type: ignore[arg-type]

    @mcp.tool()
    def memory_write(
        content: str,
        kind: Literal["facts", "state", "emo"] = "facts",
        importance: int = 1,
        session_id: str | None = None,
    ) -> dict[str, object]:
        """記憶を1件書き込む。"""
        try:
            return _core_write(
                content=content,
                kind=kind,
                importance=importance,
                session_id=session_id,
                character_id=character_id,
                config=config,
                embedder=emb,
                extraction_client=ext,
            )
        except Exception as e:
            _logger.exception("memory_write_error", error=str(e))
            raise RuntimeError("Internal server error in memory_write") from None

    @mcp.tool()
    def memory_search(
        query: str,
        limit: int = 5,
        kind_filter: list[str] | None = None,
        min_importance: int = 1,
        include_archived: bool = False,
        include_suppressed: bool = False,
    ) -> list[dict[str, object]]:
        """関連記憶を検索（ACT-R活性化 + ベクトル類似度）。"""
        try:
            return _core_search(
                query=query,
                limit=limit,
                kind_filter=kind_filter,
                min_importance=min_importance,
                include_archived=include_archived,
                include_suppressed=include_suppressed,
                character_id=character_id,
                config=config,
                embedder=emb,
            )
        except Exception as e:
            _logger.exception("memory_search_error", error=str(e))
            raise RuntimeError("Internal server error in memory_search") from None

    @mcp.tool()
    def memory_get(limit: int = 5) -> dict[str, object]:
        """自己紹介・最近の状態を返す（v1互換API）。"""
        try:
            return _core_get(
                limit=limit,
                character_id=character_id,
                config=config,
                embedder=emb,
            )
        except Exception as e:
            _logger.exception("memory_get_error", error=str(e))
            raise RuntimeError("Internal server error in memory_get") from None

    @mcp.tool()
    def memory_delete(episode_id: int, reason: str) -> dict[str, object]:
        """論理削除（valid_to=now を立てるだけ、行は残る）。"""
        try:
            return _core_delete(
                episode_id=episode_id,
                reason=reason,
                character_id=character_id,
                config=config,
            )
        except Exception as e:
            _logger.exception("memory_delete_error", error=str(e))
            raise RuntimeError("Internal server error in memory_delete") from None

    @mcp.tool()
    def memory_trace(episode_id: int) -> dict[str, object]:
        """supersedes チェーンを遡及する。"""
        try:
            return _core_trace(
                episode_id=episode_id,
                character_id=character_id,
                config=config,
            )
        except Exception as e:
            _logger.exception("memory_trace_error", error=str(e))
            raise RuntimeError("Internal server error in memory_trace") from None

    @mcp.tool()
    def memory_explore(
        node_type: Literal["episode", "entity"],
        node_id: int,
        depth: int = 1,
        full: bool = False,
        exclude_episode_ids: list[int] | None = None,
        exclude_entity_ids: list[int] | None = None,
        include_archived: bool = False,
        include_suppressed: bool = False,
    ) -> dict[str, object]:
        """グラフを 1 ホップ深掘り（memory_search の補完、AI 主導の連想探索）。"""
        try:
            result = _core_explore(
                node_type=node_type,
                node_id=node_id,
                depth=depth,
                full=full,
                exclude_episode_ids=exclude_episode_ids,
                exclude_entity_ids=exclude_entity_ids,
                include_archived=include_archived,
                include_suppressed=include_suppressed,
                character_id=character_id,
                config=config,
            )
            return result.model_dump(mode="json")
        except (ValueError, NotImplementedError):
            # 入力エラー（node not found / depth>=2）は AI 側に内容を伝える
            raise
        except Exception as e:
            _logger.exception("memory_explore_error", error=str(e))
            raise RuntimeError("Internal server error in memory_explore") from None

    @mcp.tool()
    def memory_compact(dry_run: bool = False) -> dict[str, object]:
        """夜バッチを手動起動。"""
        from fravenir.core.compact import run_compact

        try:
            result = run_compact(
                character_id=character_id,
                config=config,
                dry_run=dry_run,
            )
            return result.to_dict()
        except Exception as e:
            _logger.exception("memory_compact_error", error=str(e))
            raise RuntimeError("Internal server error in memory_compact") from None

    return mcp
