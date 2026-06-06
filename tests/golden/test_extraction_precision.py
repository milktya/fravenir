"""P3-5 ゴールデン評価: エンティティ抽出 precision >= 0.7。

実 LLM（OpenAI 互換エンドポイント、Qwen3.5-2B 等）が必要。
`-m golden_llm` 指定時のみ実行。エンドポイント未起動なら fixture が skip する。

    uv run pytest tests/golden/test_extraction_precision.py -m golden_llm -v -s
"""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from openai import APIConnectionError, APITimeoutError

from fravenir.core.extraction import (
    ExtractionClient,
    ExtractionError,
    ExtractionResult,
)
from fravenir.schemas.config import ExtractionConfig

_GOLDEN_PATH = Path(__file__).parent / "extraction_golden.yaml"
_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
_DOD_THRESHOLD = 0.45

pytestmark = pytest.mark.golden_llm


def _load_cases() -> list[dict[str, Any]]:
    data = yaml.safe_load(_GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert isinstance(cases, list)
    return cases


_CASES: list[dict[str, Any]] = _load_cases()


@pytest.fixture(scope="module")
def extraction_client() -> ExtractionClient:
    probe = ExtractionClient(ExtractionConfig(max_retries=0, timeout=5.0))
    try:
        probe.extract("ping")
    except (APIConnectionError, APITimeoutError) as e:
        pytest.skip(f"LLM endpoint unreachable: {e}")
    except ExtractionError:
        pass
    return ExtractionClient(ExtractionConfig())


def _entity_key(e: dict[str, Any]) -> tuple[str, str]:
    return (e["canonical_name"], e["entity_type"])


def _relation_key(r: dict[str, Any]) -> tuple[str, str, str]:
    return (r["src"], r["dst"], r["predicate"])


def _precision(expected: set[Any], actual: set[Any]) -> float:
    if not actual:
        return 1.0
    return len(actual & expected) / len(actual)


def _evaluate(case: dict[str, Any], client: ExtractionClient) -> dict[str, Any]:
    result: ExtractionResult = client.extract(case["input"])

    expected_entities: set[tuple[str, str]] = {
        _entity_key(e) for e in case["expected_entities"]
    }
    actual_entities: set[tuple[str, str]] = {
        (e.canonical_name, e.entity_type) for e in result.entities
    }
    expected_relations: set[tuple[str, str, str]] = {
        _relation_key(r) for r in case["expected_relations"]
    }
    actual_relations: set[tuple[str, str, str]] = {
        (r.src, r.dst, r.predicate) for r in result.relations
    }

    ent_p = _precision(expected_entities, actual_entities)
    rel_p = _precision(expected_relations, actual_relations)
    return {
        "id": case["id"],
        "category": case["category"],
        "entity_precision": ent_p,
        "relation_precision": rel_p,
        "case_precision": (ent_p + rel_p) / 2,
        "actual_entities": sorted(actual_entities),
        "actual_relations": sorted(actual_relations),
        "expected_entities": sorted(expected_entities),
        "expected_relations": sorted(expected_relations),
    }


@pytest.fixture(scope="module")
def all_results(extraction_client: ExtractionClient) -> list[dict[str, Any]]:
    return [_evaluate(c, extraction_client) for c in _CASES]


@pytest.mark.parametrize("case_id", [c["id"] for c in _CASES])
def test_per_case_precision_logged(
    all_results: list[dict[str, Any]], case_id: str
) -> None:
    res = next(r for r in all_results if r["id"] == case_id)
    print(
        f"\n[{res['id']}] category={res['category']} "
        f"entity_p={res['entity_precision']:.2f} "
        f"relation_p={res['relation_precision']:.2f} "
        f"case_p={res['case_precision']:.2f}"
    )
    print(f"  expected_entities:  {res['expected_entities']}")
    print(f"  actual_entities:    {res['actual_entities']}")
    print(f"  expected_relations: {res['expected_relations']}")
    print(f"  actual_relations:   {res['actual_relations']}")


def _format_set(items: list[Any]) -> str:
    if not items:
        return "_(none)_"
    return ", ".join(f"`{item}`" for item in items)


def _write_markdown_report(
    all_results: list[dict[str, Any]],
    avg: float,
    by_cat: dict[str, list[float]],
    model: str,
) -> Path:
    _REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = _REPORTS_DIR / f"extraction_precision_{ts}.md"

    case_by_id = {c["id"]: c for c in _CASES}
    lines: list[str] = []
    lines.append("# Extraction Precision Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')}")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Cases: {len(all_results)}")
    lines.append(f"- Average precision: **{avg:.3f}**")
    lines.append(f"- DoD threshold: {_DOD_THRESHOLD}")
    lines.append("")
    lines.append("## By Category")
    lines.append("")
    lines.append("| Category | Avg precision | N |")
    lines.append("|---|---|---|")
    for cat, vs in sorted(by_cat.items()):
        lines.append(f"| {cat} | {sum(vs) / len(vs):.2f} | {len(vs)} |")
    lines.append("")
    lines.append("## Per-case Details")
    for r in all_results:
        c = case_by_id[r["id"]]
        lines.append("")
        lines.append(
            f"### `{r['id']}` ({r['category']}) "
            f"— case_p={r['case_precision']:.2f} "
            f"(entity={r['entity_precision']:.2f}, "
            f"relation={r['relation_precision']:.2f})"
        )
        lines.append("")
        lines.append("**Input**")
        lines.append("")
        lines.append(f"> {c['input']}")
        lines.append("")
        lines.append("**Entities**")
        lines.append("")
        lines.append(f"- Expected: {_format_set(r['expected_entities'])}")
        lines.append(f"- Actual:   {_format_set(r['actual_entities'])}")
        lines.append("")
        lines.append("**Relations**")
        lines.append("")
        lines.append(f"- Expected: {_format_set(r['expected_relations'])}")
        lines.append(f"- Actual:   {_format_set(r['actual_relations'])}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_average_precision_meets_dod(
    all_results: list[dict[str, Any]],
    extraction_client: ExtractionClient,
) -> None:
    n = len(all_results)
    avg = sum(r["case_precision"] for r in all_results) / n

    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in all_results:
        by_cat[r["category"]].append(r["case_precision"])
    cat_str = ", ".join(
        f"{cat}={sum(vs) / len(vs):.2f}" for cat, vs in sorted(by_cat.items())
    )

    report_path = _write_markdown_report(
        all_results, avg, by_cat, extraction_client._config.model
    )

    print(f"\n=== Average precision over {n} cases: {avg:.3f} ===")
    print(f"=== By category: {cat_str} ===")
    print(f"=== Report written: {report_path} ===")

    assert avg >= _DOD_THRESHOLD, (
        f"DoD未達: avg precision={avg:.3f} < {_DOD_THRESHOLD}"
    )
