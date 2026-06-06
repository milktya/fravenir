"""DB → dict 変換の純粋関数群（admin UI 用）。"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


def get_stats(kv_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        ep_total: int = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        ep_active: int = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE valid_to IS NULL"
        ).fetchone()[0]
        ep_suppressed: int = conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE is_suppressed = 1"
        ).fetchone()[0]

        en_total: int = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        en_active: int = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE valid_to IS NULL"
        ).fetchone()[0]
        en_self: int = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE is_self = 1"
        ).fetchone()[0]

        rel_total: int = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        rel_active: int = conn.execute(
            "SELECT COUNT(*) FROM relations WHERE valid_to IS NULL"
        ).fetchone()[0]

        mc_pending: int = conn.execute(
            "SELECT COUNT(*) FROM merge_candidates WHERE resolved = 0"
        ).fetchone()[0]
        mc_merged: int = conn.execute(
            "SELECT COUNT(*) FROM merge_candidates WHERE resolved = 1"
        ).fetchone()[0]
        mc_rejected: int = conn.execute(
            "SELECT COUNT(*) FROM merge_candidates WHERE resolved = 2"
        ).fetchone()[0]

        ds_failed: int = conn.execute(
            "SELECT COUNT(*) FROM doc_status WHERE error IS NOT NULL"
        ).fetchone()[0]

        # ヘッダー集計バーは「現役の問題検出」が用途のため orphans は active scope に限定
        orp_ep: int = conn.execute(
            """
            SELECT COUNT(*) FROM episodes
            WHERE valid_to IS NULL AND is_suppressed = 0
              AND id NOT IN (
                  SELECT src_id FROM relations WHERE src_type = 'episode'
              )
            """
        ).fetchone()[0]
        orp_en: int = conn.execute(
            """
            SELECT COUNT(*) FROM entities
            WHERE valid_to IS NULL
              AND id NOT IN (
                  SELECT src_id FROM relations WHERE src_type = 'entity'
                  UNION
                  SELECT dst_id FROM relations WHERE dst_type = 'entity'
              )
            """
        ).fetchone()[0]
    finally:
        conn.close()

    return {
        "episodes": {"total": ep_total, "active": ep_active, "suppressed": ep_suppressed},
        "entities": {"total": en_total, "active": en_active, "is_self": en_self},
        "relations": {"total": rel_total, "active": rel_active},
        "merge_candidates": {"pending": mc_pending, "merged": mc_merged, "rejected": mc_rejected},
        "doc_status_failed": ds_failed,
        "orphans": {"episodes": orp_ep, "entities": orp_en},
    }


def _episode_scope_clause(scope: Literal["active", "archived", "all"]) -> str:
    if scope == "active":
        return "valid_to IS NULL AND is_suppressed = 0"
    if scope == "archived":
        return "is_suppressed = 0"
    return "1=1"


def _entity_scope_clause(scope: Literal["active", "archived", "all"]) -> str:
    # entities テーブルに is_suppressed カラムは無いため archived と all は同一
    if scope == "active":
        return "valid_to IS NULL"
    return "1=1"


def get_graph(
    kv_path: Path, scope: Literal["active", "archived", "all"]
) -> dict[str, Any]:
    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        ep_where = _episode_scope_clause(scope)
        ep_rows = conn.execute(
            f"""
            SELECT id, content, kind, importance, valid_to, is_suppressed, supersedes
            FROM episodes
            WHERE {ep_where}
            """
        ).fetchall()

        en_where = _entity_scope_clause(scope)
        en_rows = conn.execute(
            f"""
            SELECT id, canonical_name, entity_type, is_self, valid_to, supersedes
            FROM entities
            WHERE {en_where}
            """
        ).fetchall()

        ep_ids = {r["id"] for r in ep_rows}
        en_ids = {r["id"] for r in en_rows}

        rel_rows = conn.execute(
            """
            SELECT id, src_type, src_id, dst_type, dst_id, predicate, strength, valid_to
            FROM relations
            """
        ).fetchall()
    finally:
        conn.close()

    nodes: list[dict[str, Any]] = []
    for r in ep_rows:
        label = r["content"][:30] + ("…" if len(r["content"]) > 30 else "")
        nodes.append({
            "data": {
                "id": f"ep_{r['id']}",
                "label": label,
                "type": "episode",
                "kind": r["kind"],
                "importance": r["importance"],
                "is_active": r["valid_to"] is None,
                "is_suppressed": bool(r["is_suppressed"]),
                "supersedes": r["supersedes"],
            }
        })

    for r in en_rows:
        nodes.append({
            "data": {
                "id": f"en_{r['id']}",
                "label": r["canonical_name"],
                "type": "entity",
                "entity_type": r["entity_type"],
                "is_self": bool(r["is_self"]),
                "is_active": r["valid_to"] is None,
                "supersedes": r["supersedes"],
            }
        })

    edges: list[dict[str, Any]] = []
    for r in rel_rows:
        src_type: str = r["src_type"]
        dst_type: str = r["dst_type"]
        src_id: int = r["src_id"]
        dst_id: int = r["dst_id"]

        if src_type == "episode" and dst_type == "entity":
            if src_id not in ep_ids or dst_id not in en_ids:
                continue
            edges.append({
                "data": {
                    "id": f"men_{r['id']}",
                    "source": f"ep_{src_id}",
                    "target": f"en_{dst_id}",
                    "type": "mentions",
                    "is_active": r["valid_to"] is None,
                }
            })
        elif src_type == "entity" and dst_type == "entity":
            if src_id not in en_ids or dst_id not in en_ids:
                continue
            edges.append({
                "data": {
                    "id": f"rel_{r['id']}",
                    "source": f"en_{src_id}",
                    "target": f"en_{dst_id}",
                    "type": "relation",
                    "predicate": r["predicate"],
                    "strength": r["strength"],
                    "is_active": r["valid_to"] is None,
                }
            })

    return {
        "scope": scope,
        "stats": {"nodes": len(nodes), "edges": len(edges)},
        "elements": {"nodes": nodes, "edges": edges},
    }


def get_episode_detail(kv_path: Path, episode_id: int) -> dict[str, Any] | None:
    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        ep = conn.execute(
            """
            SELECT id, content, kind, importance, valid_from, valid_to, supersedes,
                   session_id, last_activated_at, activation_count, is_suppressed, created_at
            FROM episodes
            WHERE id = ?
            """,
            (episode_id,),
        ).fetchone()
        if ep is None:
            return None

        ds = conn.execute(
            "SELECT stage, error, updated_at FROM doc_status WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()

        mention_rows = conn.execute(
            """
            SELECT r.dst_id AS entity_id, e.canonical_name, e.is_self
            FROM relations r
            JOIN entities e ON e.id = r.dst_id
            WHERE r.src_type = 'episode' AND r.src_id = ? AND r.dst_type = 'entity'
            """,
            (episode_id,),
        ).fetchall()
    finally:
        conn.close()

    doc_status: dict[str, Any] = (
        {"stage": ds["stage"], "error": ds["error"], "updated_at": ds["updated_at"]}
        if ds is not None
        else {"stage": "unknown", "error": None, "updated_at": None}
    )

    mentions = [
        {
            "entity_id": m["entity_id"],
            "canonical_name": m["canonical_name"],
            "is_self": bool(m["is_self"]),
        }
        for m in mention_rows
    ]

    return {
        "id": ep["id"],
        "content": ep["content"],
        "kind": ep["kind"],
        "importance": ep["importance"],
        "valid_from": ep["valid_from"],
        "valid_to": ep["valid_to"],
        "supersedes": ep["supersedes"],
        "session_id": ep["session_id"],
        "last_activated_at": ep["last_activated_at"],
        "activation_count": ep["activation_count"],
        "is_suppressed": bool(ep["is_suppressed"]),
        "created_at": ep["created_at"],
        "doc_status": doc_status,
        "mentions": mentions,
    }


def get_entity_detail(kv_path: Path, entity_id: int) -> dict[str, Any] | None:
    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        en = conn.execute(
            """
            SELECT id, canonical_name, entity_type, description, is_self, self_weight,
                   decay_rate, valid_from, valid_to, supersedes, last_activated_at,
                   activation_count, created_at, curated_at
            FROM entities
            WHERE id = ?
            """,
            (entity_id,),
        ).fetchone()
        if en is None:
            return None

        alias_rows = conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()

        in_rel_rows = conn.execute(
            """
            SELECT id, src_type, src_id, predicate
            FROM relations
            WHERE dst_type = 'entity' AND dst_id = ?
            """,
            (entity_id,),
        ).fetchall()

        out_rel_rows = conn.execute(
            """
            SELECT id, dst_type, dst_id, predicate, strength
            FROM relations
            WHERE src_type = 'entity' AND src_id = ?
            """,
            (entity_id,),
        ).fetchall()
    finally:
        conn.close()

    return {
        "id": en["id"],
        "canonical_name": en["canonical_name"],
        "entity_type": en["entity_type"],
        "description": en["description"],
        "is_self": bool(en["is_self"]),
        "self_weight": en["self_weight"],
        "decay_rate": en["decay_rate"],
        "valid_from": en["valid_from"],
        "valid_to": en["valid_to"],
        "supersedes": en["supersedes"],
        "last_activated_at": en["last_activated_at"],
        "activation_count": en["activation_count"],
        "created_at": en["created_at"],
        "curated_at": en["curated_at"],
        "aliases": [a["alias"] for a in alias_rows],
        "in_relations": [
            {
                "id": r["id"],
                "src_type": r["src_type"],
                "src_id": r["src_id"],
                "predicate": r["predicate"],
            }
            for r in in_rel_rows
        ],
        "out_relations": [
            {
                "id": r["id"],
                "dst_type": r["dst_type"],
                "dst_id": r["dst_id"],
                "predicate": r["predicate"],
                "strength": r["strength"],
            }
            for r in out_rel_rows
        ],
    }


def _label_for(conn: sqlite3.Connection, node_type: str, node_id: int) -> str:
    if node_type == "episode":
        row = conn.execute("SELECT content FROM episodes WHERE id = ?", (node_id,)).fetchone()
        if row is None:
            return f"ep_{node_id}"
        content: str = row[0]
        return content[:30] + ("…" if len(content) > 30 else "")
    row = conn.execute("SELECT canonical_name FROM entities WHERE id = ?", (node_id,)).fetchone()
    if row is None:
        return f"en_{node_id}"
    return str(row[0])


def get_relation_detail(kv_path: Path, relation_id: int) -> dict[str, Any] | None:
    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        rel = conn.execute(
            """
            SELECT id, src_type, src_id, dst_type, dst_id, predicate, strength, fan_out,
                   description, valid_from, valid_to, supersedes, created_at
            FROM relations
            WHERE id = ?
            """,
            (relation_id,),
        ).fetchone()
        if rel is None:
            return None

        src_label = _label_for(conn, rel["src_type"], rel["src_id"])
        dst_label = _label_for(conn, rel["dst_type"], rel["dst_id"])
    finally:
        conn.close()

    return {
        "id": rel["id"],
        "src_type": rel["src_type"],
        "src_id": rel["src_id"],
        "src_label": src_label,
        "dst_type": rel["dst_type"],
        "dst_id": rel["dst_id"],
        "dst_label": dst_label,
        "predicate": rel["predicate"],
        "strength": rel["strength"],
        "fan_out": rel["fan_out"],
        "description": rel["description"],
        "valid_from": rel["valid_from"],
        "valid_to": rel["valid_to"],
        "supersedes": rel["supersedes"],
        "created_at": rel["created_at"],
    }


def get_merge_candidates(
    kv_path: Path, status: Literal["pending", "merged", "rejected", "all"]
) -> dict[str, Any]:
    where = {
        "pending": "WHERE mc.resolved = 0",
        "merged": "WHERE mc.resolved = 1",
        "rejected": "WHERE mc.resolved = 2",
        "all": "",
    }[status]

    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT mc.id, mc.entity_a, mc.entity_b, mc.similarity, mc.detected_at,
                   mc.resolved, mc.judge_label, mc.judge_confidence, mc.judge_reason,
                   mc.judge_attempts, mc.resolved_at,
                   ea.canonical_name AS a_name, eb.canonical_name AS b_name
            FROM merge_candidates mc
            JOIN entities ea ON ea.id = mc.entity_a
            JOIN entities eb ON eb.id = mc.entity_b
            {where}
            ORDER BY mc.id
            """
        ).fetchall()
    finally:
        conn.close()

    candidates = [
        {
            "id": r["id"],
            "entity_a": {"id": r["entity_a"], "canonical_name": r["a_name"]},
            "entity_b": {"id": r["entity_b"], "canonical_name": r["b_name"]},
            "similarity": r["similarity"],
            "detected_at": r["detected_at"],
            "resolved": r["resolved"],
            "judge_label": r["judge_label"],
            "judge_confidence": r["judge_confidence"],
            "judge_reason": r["judge_reason"],
            "judge_attempts": r["judge_attempts"],
            "resolved_at": r["resolved_at"],
        }
        for r in rows
    ]

    return {"status_filter": status, "candidates": candidates}


def get_doc_status(
    kv_path: Path, status: Literal["failed", "all"]
) -> dict[str, Any]:
    where = "WHERE ds.error IS NOT NULL" if status == "failed" else ""

    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"""
            SELECT ds.id, ds.episode_id, ds.stage, ds.error, ds.updated_at,
                   ep.content AS episode_content
            FROM doc_status ds
            LEFT JOIN episodes ep ON ep.id = ds.episode_id
            {where}
            ORDER BY ds.id
            """
        ).fetchall()
    finally:
        conn.close()

    def _ep_label(content: str | None) -> str:
        if content is None:
            return ""
        return content[:30] + ("…" if len(content) > 30 else "")

    items = [
        {
            "id": r["id"],
            "episode_id": r["episode_id"],
            "stage": r["stage"],
            "error": r["error"],
            "updated_at": r["updated_at"],
            "episode_label": _ep_label(r["episode_content"]),
        }
        for r in rows
    ]

    return {"status_filter": status, "items": items}


def get_orphans(
    kv_path: Path, scope: Literal["active", "archived", "all"]
) -> dict[str, Any]:
    ep_scope = _episode_scope_clause(scope)
    en_scope = _entity_scope_clause(scope)

    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        ep_rows = conn.execute(
            f"""
            SELECT id, content, kind, created_at
            FROM episodes
            WHERE {ep_scope}
              AND id NOT IN (
                  SELECT src_id FROM relations WHERE src_type = 'episode'
              )
            ORDER BY id
            """
        ).fetchall()

        en_rows = conn.execute(
            f"""
            SELECT id, canonical_name, is_self, created_at
            FROM entities
            WHERE {en_scope}
              AND id NOT IN (
                  SELECT src_id FROM relations WHERE src_type = 'entity'
                  UNION
                  SELECT dst_id FROM relations WHERE dst_type = 'entity'
              )
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    def _ep_label(content: str) -> str:
        return content[:30] + ("…" if len(content) > 30 else "")

    episodes = [
        {
            "id": r["id"],
            "label": _ep_label(r["content"]),
            "kind": r["kind"],
            "created_at": r["created_at"],
        }
        for r in ep_rows
    ]

    entities = [
        {
            "id": r["id"],
            "canonical_name": r["canonical_name"],
            "is_self": bool(r["is_self"]),
            "created_at": r["created_at"],
        }
        for r in en_rows
    ]

    return {"scope": scope, "episodes": episodes, "entities": entities}



class EntityNotFoundError(Exception):
    """update_entity 対象の entity が見つからない、または archived。"""


# Phase 6: AdminUI 経由の entity 編集 / 監査ログ用ヘルパー。
# 編集対象は description と aliases のみ。canonical_name / entity_type / supersedes
# 等の構造的フィールドは AdminUI からは触らせない (CLI / merge() 経由のみ)。
_ALIAS_PATTERN_MAX_LEN = 200


def _normalize_aliases(aliases: list[str]) -> list[str]:
    """重複と空白だけのものを除去、前後空白トリム、長さ上限チェック。"""
    seen: set[str] = set()
    out: list[str] = []
    for raw in aliases:
        a = raw.strip()
        if not a:
            continue
        if len(a) > _ALIAS_PATTERN_MAX_LEN:
            raise ValueError(f"alias too long ({len(a)} > {_ALIAS_PATTERN_MAX_LEN})")
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def update_entity(
    kv_path: Path,
    entity_id: int,
    *,
    description: str | None,
    aliases: list[str] | None,
    actor: str = "admin_ui",
) -> dict[str, Any]:
    """Entity の description / aliases を更新し、curated_at を立て、admin_audit_log に記録。

    - description=None / aliases=None ならその項目は変更しない
    - aliases は完全置換 (差分計算は内部で行う)
    - 変更内容が空（before == after）なら curated_at も audit log も更新しない
    - description が変わった場合、vdb_entities の再エンベディングは呼び出し側で行う
      (queries.py は DB のみを扱う方針を維持)
    """
    if aliases is not None:
        aliases = _normalize_aliases(aliases)

    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        before_row = conn.execute(
            """
            SELECT id, description
            FROM entities
            WHERE id = ? AND valid_to IS NULL
            """,
            (entity_id,),
        ).fetchone()
        if before_row is None:
            raise EntityNotFoundError(f"entity {entity_id} not found or archived")

        before_aliases = [
            r["alias"]
            for r in conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias",
                (entity_id,),
            ).fetchall()
        ]

        before = {
            "description": before_row["description"],
            "aliases": before_aliases,
        }
        after: dict[str, Any] = {
            "description": before["description"],
            "aliases": list(before_aliases),
        }

        changed = False
        if description is not None and description != before["description"]:
            after["description"] = description
            changed = True

        if aliases is not None and sorted(aliases) != sorted(before_aliases):
            after["aliases"] = aliases
            changed = True

        if not changed:
            return {
                "changed": False,
                "before": before,
                "after": after,
                "curated_at": None,
            }

        now = datetime.now(UTC).isoformat()
        if description is not None and description != before["description"]:
            conn.execute(
                "UPDATE entities SET description = ?, curated_at = ? WHERE id = ?",
                (description, now, entity_id),
            )
        else:
            conn.execute(
                "UPDATE entities SET curated_at = ? WHERE id = ?",
                (now, entity_id),
            )

        if aliases is not None and sorted(aliases) != sorted(before_aliases):
            conn.execute("DELETE FROM entity_aliases WHERE entity_id = ?", (entity_id,))
            for a in aliases:
                conn.execute(
                    "INSERT OR IGNORE INTO entity_aliases (alias, entity_id) VALUES (?, ?)",
                    (a, entity_id),
                )

        conn.execute(
            """
            INSERT INTO admin_audit_log
                (action, target_type, target_id, before_json, after_json, actor)
            VALUES ('entity.update', 'entity', ?, ?, ?, ?)
            """,
            (
                entity_id,
                json.dumps(before, ensure_ascii=False),
                json.dumps(after, ensure_ascii=False),
                actor,
            ),
        )
        conn.commit()
        return {
            "changed": True,
            "before": before,
            "after": after,
            "curated_at": now,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_audit_log(
    kv_path: Path,
    *,
    target_type: str | None = None,
    target_id: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """admin_audit_log を新しい順に返す。target で絞り込み可。"""
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be in [1, 1000]")
    conn = sqlite3.connect(kv_path)
    conn.row_factory = sqlite3.Row
    try:
        clauses: list[str] = []
        params: list[Any] = []
        if target_type is not None:
            clauses.append("target_type = ?")
            params.append(target_type)
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT id, action, target_type, target_id,
                   before_json, after_json, actor, created_at
            FROM admin_audit_log
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "id": r["id"],
            "action": r["action"],
            "target_type": r["target_type"],
            "target_id": r["target_id"],
            "before": json.loads(r["before_json"]) if r["before_json"] else None,
            "after": json.loads(r["after_json"]) if r["after_json"] else None,
            "actor": r["actor"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
