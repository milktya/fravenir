"""Admin UI API routes — /api/* endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request

from fravenir.admin import queries, schemas

router = APIRouter()


def _kv_path(request: Request) -> Path:
    return request.app.state.kv_path  # type: ignore[no-any-return]


KvPath = Annotated[Path, Depends(_kv_path)]


@router.get("/stats", response_model=schemas.StatsResponse)
def get_stats(kv_path: KvPath) -> schemas.StatsResponse:
    return schemas.StatsResponse.model_validate(queries.get_stats(kv_path))


@router.get("/graph", response_model=schemas.GraphResponse)
def get_graph(
    kv_path: KvPath,
    scope: Literal["active", "archived", "all"] = "active",
) -> schemas.GraphResponse:
    return schemas.GraphResponse.model_validate(queries.get_graph(kv_path, scope))


@router.get("/episodes/{id}", response_model=schemas.EpisodeDetail)
def get_episode_detail(id: int, kv_path: KvPath) -> schemas.EpisodeDetail:
    result = queries.get_episode_detail(kv_path, id)
    if result is None:
        raise HTTPException(status_code=404)
    return schemas.EpisodeDetail.model_validate(result)


@router.get("/entities/{id}", response_model=schemas.EntityDetail)
def get_entity_detail(id: int, kv_path: KvPath) -> schemas.EntityDetail:
    result = queries.get_entity_detail(kv_path, id)
    if result is None:
        raise HTTPException(status_code=404)
    return schemas.EntityDetail.model_validate(result)


@router.get("/relations/{id}", response_model=schemas.RelationDetail)
def get_relation_detail(id: int, kv_path: KvPath) -> schemas.RelationDetail:
    result = queries.get_relation_detail(kv_path, id)
    if result is None:
        raise HTTPException(status_code=404)
    return schemas.RelationDetail.model_validate(result)


@router.get("/merge_candidates", response_model=schemas.MergeCandidatesResponse)
def get_merge_candidates(
    kv_path: KvPath,
    status: Literal["pending", "merged", "rejected", "all"] = "pending",
) -> schemas.MergeCandidatesResponse:
    return schemas.MergeCandidatesResponse.model_validate(
        queries.get_merge_candidates(kv_path, status)
    )


@router.get("/doc_status", response_model=schemas.DocStatusResponse)
def get_doc_status(
    kv_path: KvPath,
    status: Literal["failed", "all"] = "failed",
) -> schemas.DocStatusResponse:
    return schemas.DocStatusResponse.model_validate(queries.get_doc_status(kv_path, status))


@router.get("/orphans", response_model=schemas.OrphansResponse)
def get_orphans(
    kv_path: KvPath,
    scope: Literal["active", "archived", "all"] = "active",
) -> schemas.OrphansResponse:
    return schemas.OrphansResponse.model_validate(queries.get_orphans(kv_path, scope))


def _reembed_entity(request: Request, entity_id: int, name: str, description: str) -> None:
    """Description が curated 経由で更新された entity の vdb_entities を再エンベディング。

    Embedder は遅延初期化 (admin/server.py で app.state.embedder=None)。
    sentence-transformers のロード/モデルキャッシュが効くので、初回 PATCH のみ重い。
    """
    import sqlite3

    import sqlite_vec
    import structlog
    import yaml

    from fravenir.embedding import Embedder
    from fravenir.schemas.config import AppConfig
    from fravenir.storage.paths import config_yaml_path
    from fravenir.storage.vector import upsert_entity_vector

    log = structlog.get_logger(__name__)
    state = request.app.state
    embedder: Embedder | None = state.embedder
    if embedder is None:
        character_id: str = state.character_id
        cfg_path = config_yaml_path(character_id)
        if cfg_path.exists():
            with cfg_path.open(encoding="utf-8") as f:
                raw_cfg = yaml.safe_load(f) or {}
            if isinstance(raw_cfg, dict):
                raw_cfg.setdefault("character", {})["id"] = character_id
            else:
                raw_cfg = {"character": {"id": character_id}}
        else:
            raw_cfg = {"character": {"id": character_id}}
        app_cfg = AppConfig.model_validate(raw_cfg)
        embedder = Embedder(app_cfg.embedding)
        state.embedder = embedder

    text = f"{name} {description}".strip() if description else name
    vec = embedder.encode_topic(text)

    vdb_path: Path = state.vdb_entities_path
    conn = sqlite3.connect(str(vdb_path))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        upsert_entity_vector(conn, entity_id, vec)
        conn.commit()
    finally:
        conn.close()
    log.info("admin_entity_reembedded", entity_id=entity_id)


@router.patch("/entities/{id}", response_model=schemas.EntityUpdateResponse)
def update_entity(
    id: int,
    payload: schemas.EntityUpdateRequest,
    request: Request,
    kv_path: KvPath,
) -> schemas.EntityUpdateResponse:
    """Entity の description / aliases を更新し curated_at を立てる。

    変更がない (description/aliases ともに現値と一致) 場合は changed=False を返し、
    DB は更新しない。description が変わった場合は vdb_entities も再エンベディング。
    """
    if payload.description is None and payload.aliases is None:
        raise HTTPException(
            status_code=400,
            detail="at least one of description / aliases must be provided",
        )
    try:
        result = queries.update_entity(
            kv_path,
            id,
            description=payload.description,
            aliases=payload.aliases,
        )
    except queries.EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result["changed"] and result["before"]["description"] != result["after"]["description"]:
        # description が変わったので vdb 再エンベディング
        # canonical_name は取得し直す (queries.update_entity 内で保証された active 行)
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(kv_path)
        try:
            row = conn.execute(
                "SELECT canonical_name FROM entities WHERE id = ?", (id,)
            ).fetchone()
        finally:
            conn.close()
        if row is not None:
            _reembed_entity(request, id, row[0], result["after"]["description"] or "")

    return schemas.EntityUpdateResponse.model_validate(result)


@router.get("/audit_log", response_model=schemas.AuditLogResponse)
def get_audit_log(
    kv_path: KvPath,
    target_type: str | None = None,
    target_id: int | None = None,
    limit: int = 100,
) -> schemas.AuditLogResponse:
    try:
        entries = queries.list_audit_log(
            kv_path,
            target_type=target_type,
            target_id=target_id,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.AuditLogResponse.model_validate({"entries": entries})
