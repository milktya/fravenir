r"""Graph traversal helpers (Phase3+).

memory_search の Step 3 で使う 2ホップ探索と連想強度 $S_{ji}$ 計算を提供する。

設計書 §5.3 / §6.1 Step 3 に対応:
- relations は valid_to IS NULL のみを辿る（論理削除されたエッジは無視）
- fan_out は「ノード j から出る有効なエッジ数」
- $S_{ji} = S_{\max} - \ln(\mathrm{fan}_j)$

NetworkX を使う理由は 2ホップ BFS を素直に書けることと fan 計算 API。
全relations をメモリ常駐させず、seed とその近傍だけ SQL で引いて部分グラフを組む。
"""

from __future__ import annotations

import math
import sqlite3

import networkx as nx


def build_subgraph_from_seeds(
    conn: sqlite3.Connection,
    seed_entity_ids: list[int],
    max_hops: int = 2,
) -> nx.DiGraph:
    """Seed entities から最大 `max_hops` ホップ以内の有効エッジを持つ部分グラフを構築。

    ノード識別子は `(type, id)` タプル（type は 'entity' または 'episode'）。
    各エッジには `predicate` 属性のみを付与する（fan_out は後段で `fan_out_of` から取得）。
    """
    graph: nx.DiGraph = nx.DiGraph()
    frontier: set[tuple[str, int]] = {("entity", eid) for eid in seed_entity_ids}
    visited: set[tuple[str, int]] = set()

    for _ in range(max_hops):
        if not frontier:
            break
        entity_frontier = [nid for ntype, nid in frontier if ntype == "entity"]
        visited.update(frontier)
        frontier = set()
        if not entity_frontier:
            continue

        placeholders = ",".join("?" * len(entity_frontier))
        # entity を src とする有効エッジのみを展開する（episode は終端）。
        rows = conn.execute(
            f"""
            SELECT src_type, src_id, dst_type, dst_id, predicate
            FROM relations
            WHERE valid_to IS NULL
              AND src_type = 'entity'
              AND src_id IN ({placeholders})
            """,
            entity_frontier,
        ).fetchall()

        for src_type, src_id, dst_type, dst_id, predicate in rows:
            src_node = (src_type, src_id)
            dst_node = (dst_type, dst_id)
            graph.add_edge(src_node, dst_node, predicate=predicate)
            if dst_node not in visited:
                frontier.add(dst_node)

    return graph


def fan_out_of(conn: sqlite3.Connection, entity_id: int) -> int:
    """Entity から出る `valid_to IS NULL` の relations 件数。

    `relations.fan_out` カラムが存在するが、compact 未実行時はその値が
    古いままになりうるため、ライブカウント (COUNT(*)) で精度を確保する。
    """
    row = conn.execute(
        """
        SELECT COUNT(*) FROM relations
        WHERE valid_to IS NULL
          AND src_type = 'entity'
          AND src_id = ?
        """,
        (entity_id,),
    ).fetchone()
    return int(row[0]) if row is not None else 0


def s_ji(fan_j: int, s_max: float) -> float:
    r"""連想強度 $S_{ji} = S_{\max} - \ln(\mathrm{fan}_j)$。

    fan_j=0 は「到達元として使われていない」状況だが、安全のため 1 に丸めて
    ln を安定化する（$S_{ji} = S_{\max}$ になる）。
    """
    safe_fan = max(fan_j, 1)
    value = s_max - math.log(safe_fan)
    return max(value, 0.0)


def reach_episodes(graph: nx.DiGraph, seed_entity_id: int) -> set[int]:
    """Seed entity から 2ホップ以内で到達できる episode の id 集合。"""
    seed_node = ("entity", seed_entity_id)
    if seed_node not in graph:
        return set()
    episodes: set[int] = set()
    for node in nx.descendants(graph, seed_node):
        node_type, node_id = node
        if node_type == "episode":
            episodes.add(int(node_id))
    return episodes
