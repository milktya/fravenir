"""LLM-based entity/relation extraction (Phase3).

OpenAI互換エンドポイント (llama.cpp server 等) に JSON mode で問い合わせ、
episode content からエンティティと関係を抽出する。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import structlog
from openai import APIConnectionError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field, ValidationError

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

    from fravenir.schemas.config import ExtractionConfig

_log = structlog.get_logger(__name__)

DEFAULT_ENTITY_TYPES: list[str] = ["person", "concept", "work", "place", "emotion"]

DEFAULT_PREDICATES: list[str] = [
    "likes", "dislikes", "visits", "drinks", "eats",
    "writes", "creates", "cooks", "hosts", "uses",
    "shares", "performs",
    "works_as", "lives_in",
    "implements", "fixes", "blocks", "configures", "accepts",
    "is_a", "has", "part_of", "includes", "located_at",
    "runs_on", "introduces", "causes",
]


class ExtractedEntity(BaseModel):
    canonical_name: str
    entity_type: str
    description: str = ""


class ExtractedRelation(BaseModel):
    src: str
    dst: str
    predicate: str
    description: str = ""


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)


class ExtractionError(RuntimeError):
    """max_retries を使い切っても抽出に成功しなかった場合に送出."""


def _strip_code_fence(raw: str) -> str:
    # JSON mode 指定を無視して Markdown フェンスで囲むモデルへの保険
    text = raw.strip()
    if not text.startswith("```"):
        return text
    newline = text.find("\n")
    if newline == -1:
        return text
    text = text[newline + 1 :].rstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    # 言語識別子のみの独立行が先頭に残るパターン保険
    # 例: "```\njson\n{...}\n```" → 上記処理後 text = "json\n{...}"
    first_newline = text.find("\n")
    if first_newline != -1:
        first_line = text[:first_newline].strip()
        if (
            first_line
            and first_line.isalpha()
            and first_line.islower()
            and len(first_line) <= 10
        ):
            text = text[first_newline + 1 :].lstrip()
    return text


def _build_schema(entity_types: list[str], predicates: list[str]) -> dict[str, Any]:
    # pydantic の $defs/$ref を使わずフラット化。llama.cpp の GBNF 変換は
    # $ref 越しの enum を強制できないため、この形式でインライン化する。
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_name": {"type": "string"},
                        "entity_type": {"type": "string", "enum": entity_types},
                        "description": {"type": "string"},
                    },
                    "required": ["canonical_name", "entity_type", "description"],
                    "additionalProperties": False,
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "predicate": {"type": "string", "enum": predicates},
                        "description": {"type": "string"},
                    },
                    "required": ["src", "dst", "predicate", "description"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entities", "relations"],
        "additionalProperties": False,
    }


_SYSTEM_PROMPT = """\
あなたはエンティティ抽出システムです。入力テキストから、登場する人物・概念・\
作品・場所・感情などの「エンティティ」と、それらの「関係」を抽出してください。

出力は次の JSON オブジェクトそのものだけを返してください:

{{
  "entities": [
    {{"canonical_name": "エンティティ名", "entity_type": "{entity_types_joined}", \
"description": "簡潔な説明"}}
  ],
  "relations": [
    {{"src": "エンティティ名", "dst": "エンティティ名", \
"predicate": "関係ラベル", \
"description": "簡潔な説明"}}
  ]
}}

ルール:
- canonical_name は表記を1つに正規化（同義語・略称はまとめる）
- canonical_name は所有格・修飾語を剥がした核の語を抽出する\
（例: 「みるちゃの家」→「家」、「近所のカフェ」→「カフェ」、\
「ACT-R活性化」→「ACT-R活性化」は専門語なのでそのまま）
- entity_type は次のいずれか: {entity_types_joined}
- entity_type が emotion の canonical_name は感情の名詞形で書く\
（例: 「嬉しい」→「嬉しさ」、「眠い」→「眠気」、「落ち込む日」→「落ち込み」）
- relations の src / dst は entities リスト内の canonical_name のみ参照
- relations の src と dst は異なる canonical_name を指す\
（同一エンティティに対する relation は出力しない）
- predicate は次のいずれかから 1 つだけ選ぶ（時制は 3 単現で統一、\
意味が遠ければ最も近いものを選ぶ）: {predicates_joined}
- 該当するものがなければ空配列 []
- 応答は {{ で始まり }} で終わる 1 個の JSON オブジェクトのみとする"""


_USER_PROMPT = """\
以下のテキストから entities と relations を抽出し、JSON で返してください:

{content}"""


class ExtractionClient:
    """OpenAI互換エンドポイントに対する抽出クライアント."""

    def __init__(self, config: ExtractionConfig) -> None:
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            max_retries=0,
        )

    def extract(
        self,
        content: str,
        entity_types: list[str] | None = None,
        predicates: list[str] | None = None,
    ) -> ExtractionResult:
        """content からエンティティと関係を抽出する.

        max_retries + 1 回まで試行し、全て失敗したら ExtractionError を送出.
        """
        types = entity_types if entity_types else DEFAULT_ENTITY_TYPES
        preds = predicates if predicates else DEFAULT_PREDICATES
        messages = self._build_messages(content, types, preds)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "ExtractionResult",
                "schema": _build_schema(types, preds),
                "strict": True,
            },
        }

        last_error: Exception | None = None
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model,
                    messages=cast("list[ChatCompletionMessageParam]", messages),
                    temperature=self._config.temperature,
                    response_format=cast("Any", response_format),
                )
                raw = response.choices[0].message.content or ""
                parsed = json.loads(_strip_code_fence(raw))
                return ExtractionResult.model_validate(parsed)
            except (
                json.JSONDecodeError,
                ValidationError,
                APIConnectionError,
                APITimeoutError,
            ) as e:
                last_error = e
                _log.warning(
                    "extraction_attempt_failed",
                    attempt=attempt + 1,
                    attempts=attempts,
                    error_type=type(e).__name__,
                )

        raise ExtractionError(
            f"extraction failed after {attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _build_messages(
        content: str,
        entity_types: list[str],
        predicates: list[str],
    ) -> list[dict[str, str]]:
        system = _SYSTEM_PROMPT.format(
            entity_types_joined=" | ".join(entity_types),
            predicates_joined=" | ".join(predicates),
        )
        user = _USER_PROMPT.format(content=content)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
