"""Pydantic v2 schemas for seed.yaml (character initialization data)."""

from pydantic import BaseModel, Field


class IdentityConfig(BaseModel):
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class PersonalityConfig(BaseModel):
    canonical_name: str
    entity_type: str = "concept"
    description: str = ""
    self_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class InitialEpisode(BaseModel):
    content: str
    kind: str = Field(pattern="^(facts|state|emo)$")
    importance: int = Field(default=1, ge=1, le=3)


class SeedEntityConfig(BaseModel):
    """重要固有名詞を seed として投入するためのエントリ。

    - canonical_name は active 区間内で一意 (sqlite 側で UNIQUE 制約)
    - aliases は entity_aliases テーブルに展開される
    - 投入時に curated_at が現在時刻にセットされる
    """

    canonical_name: str = Field(min_length=1, max_length=200)
    entity_type: str = "person"
    description: str = Field(default="", max_length=4000)
    aliases: list[str] = Field(default_factory=list)
    self_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    decay_rate: float = Field(default=0.5, ge=0.0, le=1.0)


class SeedConfig(BaseModel):
    identity: IdentityConfig
    personality: list[PersonalityConfig] = Field(default_factory=list)
    # Phase 6: 重要固有名詞をエンティティとして seed 段階で投入する枠。
    # ここに入れたものは `curated_at` が立った状態で生成され、AdminUI の手動編集と
    # 同じレールに乗る。
    seed_entities: list["SeedEntityConfig"] = Field(default_factory=list)
    initial_episodes: list[InitialEpisode] = Field(default_factory=list)
