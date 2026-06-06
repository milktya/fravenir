"""Pydantic v2 schemas for config.yaml (per-character application config)."""

from pydantic import BaseModel, Field


class CharacterConfig(BaseModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    system_prompt_template: str = ""


class EmbeddingPrefixes(BaseModel):
    general: str = ""
    topic: str = "トピック: "
    query: str = "検索クエリ: "
    document: str = "検索文書: "


class EmbeddingConfig(BaseModel):
    model: str = "cl-nagoya/ruri-v3-310m"
    dim: int = Field(default=768, ge=1)
    max_tokens: int = Field(default=8192, ge=1)
    device: str = "auto"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True
    prefixes: EmbeddingPrefixes = Field(default_factory=EmbeddingPrefixes)


class ActRConfig(BaseModel):
    base_decay: float = Field(default=0.5, gt=0.0)
    self_decay: float = Field(default=0.2, gt=0.0)
    personality_decay: float = Field(default=0.3, gt=0.0)
    self_boost_beta: float = Field(default=0.5, ge=0.0)
    s_max: float = Field(default=2.0, ge=0.0)
    access_history_limit: int = Field(default=100, ge=1)
    suppress_threshold: float = -2.0
    alpha_similarity: float = Field(default=1.0, ge=0.0)
    alpha_importance: float = Field(default=0.3, ge=0.0)


class SessionConfig(BaseModel):
    auto_timeout_minutes: int = Field(default=15, ge=1)


class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")
    format: str = Field(default="json", pattern="^(json|console)$")
    activation_debug: bool = False


class CompactConfig(BaseModel):
    schedule: str = "0 3 * * *"
    dry_run_default: bool = False
    suppress_recent_access_days: int = Field(default=7, ge=1)


class ExtractionConfig(BaseModel):
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "unsloth/gemma-4-E2B-it-GGUF"
    api_key: str = "dummy"
    timeout: float = Field(default=30.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class SemanticJudgeConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "Gemma4-31B"
    api_key: str = "dummy"
    timeout: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    min_strength: float = Field(default=0.3, ge=0.0)


class ServerConfig(BaseModel):
    transport: str = Field(default="stdio", pattern="^(stdio|streamable-http|sse)$")
    host: str = "127.0.0.1"
    port: int = Field(default=8280, ge=1, le=65535)


class AppConfig(BaseModel):
    character: CharacterConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    act_r: ActRConfig = Field(default_factory=ActRConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    compact: CompactConfig = Field(default_factory=CompactConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    semantic_judge: SemanticJudgeConfig = Field(default_factory=SemanticJudgeConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
