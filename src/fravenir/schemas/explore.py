"""Pydantic v2 schemas for memory_explore (FEAT-1).

詳細仕様は docs/feat1_memory_explore_design.md を参照。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class NeighborItem(BaseModel):
    type: Literal["episode", "entity"]
    id: int
    summary: str                                  # 1 行要約（最大 120 chars）
    direction: Literal["incoming", "outgoing"]
    sort_score: float                             # B_i + S_{parent→i}（debug 用に露出）


class PredicateMeta(BaseModel):
    shown: int
    total: int                                    # フィルタ適用後・絞り込み前の総件数


class NodeContent(BaseModel):
    type: Literal["episode", "entity"]
    id: int
    name: str | None = None                       # entity の canonical_name、episode は None
    content: str                                  # truncate 済（full=False なら 800 chars）
    is_truncated: bool
    importance: int
    valid_from: datetime
    valid_to: datetime | None = None
    is_suppressed: bool | None = None             # 起点ノードの抑制状態（feat1 §6.3）
    # entity 固有
    is_self: bool | None = None
    self_weight: float | None = None
    decay_rate: float | None = None


class ExploreResult(BaseModel):
    node: NodeContent
    neighbors: dict[str, list[NeighborItem]]      # predicate -> neighbor items
    meta: dict[str, PredicateMeta]                # predicate -> shown/total
    total_neighbors: int                          # 表示件数（最大 10）
    total_neighbors_unfiltered: int               # フィルタ前の relation 総数
