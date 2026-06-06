"""Pydantic v2 schemas for admin UI API responses."""

from pydantic import BaseModel, Field


class StatsEpisodes(BaseModel):
    total: int
    active: int
    suppressed: int


class StatsEntities(BaseModel):
    total: int
    active: int
    is_self: int


class StatsRelations(BaseModel):
    total: int
    active: int


class StatsMergeCandidates(BaseModel):
    pending: int
    merged: int
    rejected: int


class StatsOrphans(BaseModel):
    episodes: int
    entities: int


class StatsResponse(BaseModel):
    episodes: StatsEpisodes
    entities: StatsEntities
    relations: StatsRelations
    merge_candidates: StatsMergeCandidates
    doc_status_failed: int
    orphans: StatsOrphans


class GraphNodeEpisodeData(BaseModel):
    id: str
    label: str
    type: str
    kind: str
    importance: int
    is_active: bool
    is_suppressed: bool
    supersedes: int | None


class GraphNodeEntityData(BaseModel):
    id: str
    label: str
    type: str
    entity_type: str
    is_self: bool
    is_active: bool
    supersedes: int | None


class GraphNode(BaseModel):
    data: GraphNodeEpisodeData | GraphNodeEntityData


class GraphEdgeMentionsData(BaseModel):
    id: str
    source: str
    target: str
    type: str
    is_active: bool


class GraphEdgeRelationData(BaseModel):
    id: str
    source: str
    target: str
    type: str
    predicate: str
    strength: float
    is_active: bool


class GraphEdge(BaseModel):
    data: GraphEdgeMentionsData | GraphEdgeRelationData


class GraphElements(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GraphStats(BaseModel):
    nodes: int
    edges: int


class GraphResponse(BaseModel):
    scope: str
    stats: GraphStats
    elements: GraphElements


class DocStatus(BaseModel):
    stage: str
    error: str | None
    updated_at: str | None


class EpisodeMention(BaseModel):
    entity_id: int
    canonical_name: str
    is_self: bool


class EpisodeDetail(BaseModel):
    id: int
    content: str
    kind: str
    importance: int
    valid_from: str
    valid_to: str | None
    supersedes: int | None
    session_id: str | None
    last_activated_at: str | None
    activation_count: int
    is_suppressed: bool
    created_at: str
    doc_status: DocStatus
    mentions: list[EpisodeMention]


class EntityInRelation(BaseModel):
    id: int
    src_type: str
    src_id: int
    predicate: str


class EntityOutRelation(BaseModel):
    id: int
    dst_type: str
    dst_id: int
    predicate: str
    strength: float | None = None


class EntityDetail(BaseModel):
    id: int
    canonical_name: str
    entity_type: str
    description: str | None
    is_self: bool
    self_weight: float
    decay_rate: float
    valid_from: str
    valid_to: str | None
    supersedes: int | None
    last_activated_at: str | None
    activation_count: int
    created_at: str
    curated_at: str | None
    aliases: list[str]
    in_relations: list[EntityInRelation]
    out_relations: list[EntityOutRelation]


class EntityUpdateRequest(BaseModel):
    """AdminUI からの entity 編集リクエスト。

    description / aliases のいずれか一方だけの更新も可 (None を渡せばその項目はスキップ)。
    更新が発生した場合は entities.curated_at が現在時刻に更新され、admin_audit_log に
    before/after が記録される。
    """

    description: str | None = Field(default=None, max_length=4000)
    aliases: list[str] | None = Field(default=None, max_length=64)


class RelationDetail(BaseModel):
    id: int
    src_type: str
    src_id: int
    src_label: str
    dst_type: str
    dst_id: int
    dst_label: str
    predicate: str
    strength: float
    fan_out: int
    description: str | None
    valid_from: str
    valid_to: str | None
    supersedes: int | None
    created_at: str


class MergeCandidateEntity(BaseModel):
    id: int
    canonical_name: str


class MergeCandidate(BaseModel):
    id: int
    entity_a: MergeCandidateEntity
    entity_b: MergeCandidateEntity
    similarity: float
    detected_at: str
    resolved: int
    judge_label: str | None
    judge_confidence: str | None
    judge_reason: str | None
    judge_attempts: int
    resolved_at: str | None


class MergeCandidatesResponse(BaseModel):
    status_filter: str
    candidates: list[MergeCandidate]


class DocStatusItem(BaseModel):
    id: int
    episode_id: int
    stage: str
    error: str | None
    updated_at: str
    episode_label: str


class DocStatusResponse(BaseModel):
    status_filter: str
    items: list[DocStatusItem]


class OrphanEpisode(BaseModel):
    id: int
    label: str
    kind: str
    created_at: str


class OrphanEntity(BaseModel):
    id: int
    canonical_name: str
    is_self: bool
    created_at: str


class OrphansResponse(BaseModel):
    scope: str
    episodes: list[OrphanEpisode]
    entities: list[OrphanEntity]



class EntityUpdateResponse(BaseModel):
    """PATCH /entities/{id} の戻り値。"""

    changed: bool
    before: dict[str, object]
    after: dict[str, object]
    curated_at: str | None


class AuditLogEntry(BaseModel):
    id: int
    action: str
    target_type: str
    target_id: int
    before: dict[str, object] | None
    after: dict[str, object] | None
    actor: str | None
    created_at: str


class AuditLogResponse(BaseModel):
    entries: list[AuditLogEntry]
