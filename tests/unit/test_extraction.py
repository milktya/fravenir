"""Unit tests for core/extraction.py."""

import json
from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APITimeoutError

from fravenir.core.extraction import (
    DEFAULT_ENTITY_TYPES,
    DEFAULT_PREDICATES,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionClient,
    ExtractionError,
    ExtractionResult,
    _strip_code_fence,
)
from fravenir.schemas.config import ExtractionConfig


def _make_config(**overrides) -> ExtractionConfig:
    defaults = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "test-model",
        "api_key": "dummy",
        "timeout": 5.0,
        "max_retries": 2,
        "temperature": 0.0,
    }
    defaults.update(overrides)
    return ExtractionConfig(**defaults)


def _mock_response(json_string: str):
    msg = MagicMock()
    msg.content = json_string
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _valid_payload() -> dict:
    return {
        "entities": [
            {
                "canonical_name": "みるちゃ",
                "entity_type": "person",
                "description": "ユーザー",
            },
            {
                "canonical_name": "メモリツール",
                "entity_type": "work",
                "description": "開発中のツール",
            },
        ],
        "relations": [
            {
                "src": "みるちゃ",
                "dst": "メモリツール",
                "predicate": "creates",
                "description": "",
            },
        ],
    }


class TestPromptBuilding:
    def test_system_prompt_contains_entity_types(self):
        messages = ExtractionClient._build_messages(
            "hello", ["person", "concept"], DEFAULT_PREDICATES
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "person | concept" in messages[0]["content"]

    def test_user_prompt_contains_content(self):
        messages = ExtractionClient._build_messages(
            "今日は雨", DEFAULT_ENTITY_TYPES, DEFAULT_PREDICATES
        )
        assert messages[1]["role"] == "user"
        assert "今日は雨" in messages[1]["content"]

    def test_default_entity_types_included(self):
        messages = ExtractionClient._build_messages(
            "x", DEFAULT_ENTITY_TYPES, DEFAULT_PREDICATES
        )
        system = messages[0]["content"]
        for t in DEFAULT_ENTITY_TYPES:
            assert t in system

    def test_default_predicates_included(self):
        messages = ExtractionClient._build_messages(
            "x", DEFAULT_ENTITY_TYPES, DEFAULT_PREDICATES
        )
        system = messages[0]["content"]
        for p in DEFAULT_PREDICATES:
            assert p in system


class TestExtract:
    def test_parses_valid_json(self):
        client = ExtractionClient(_make_config())
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _mock_response(
            json.dumps(_valid_payload())
        )

        result = client.extract("みるちゃはメモリツールを作ってる")
        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 2
        assert result.entities[0].canonical_name == "みるちゃ"
        assert result.relations[0].predicate == "creates"

    def test_retries_on_invalid_json(self):
        client = ExtractionClient(_make_config(max_retries=2))
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = [
            _mock_response("not json {{{"),
            _mock_response(json.dumps(_valid_payload())),
        ]

        result = client.extract("何か")
        assert len(result.entities) == 2
        assert client._client.chat.completions.create.call_count == 2

    def test_raises_when_all_attempts_fail(self):
        client = ExtractionClient(_make_config(max_retries=1))
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _mock_response("!!!!")

        with pytest.raises(ExtractionError):
            client.extract("何か")
        assert client._client.chat.completions.create.call_count == 2

    def test_max_retries_zero_single_attempt(self):
        client = ExtractionClient(_make_config(max_retries=0))
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _mock_response("bad")

        with pytest.raises(ExtractionError):
            client.extract("何か")
        assert client._client.chat.completions.create.call_count == 1

    def test_uses_custom_entity_types(self):
        client = ExtractionClient(_make_config())
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _mock_response(
            json.dumps({"entities": [], "relations": []})
        )

        client.extract("content", entity_types=["custom_type"])
        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        system_msg = call_kwargs["messages"][0]["content"]
        assert "custom_type" in system_msg

    def test_handles_connection_error_then_succeeds(self):
        client = ExtractionClient(_make_config(max_retries=2))
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = [
            APIConnectionError(request=MagicMock()),
            _mock_response(json.dumps(_valid_payload())),
        ]

        result = client.extract("content")
        assert len(result.entities) == 2

    def test_handles_timeout_error(self):
        client = ExtractionClient(_make_config(max_retries=0))
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = APITimeoutError(
            request=MagicMock()
        )

        with pytest.raises(ExtractionError):
            client.extract("content")

    def test_validation_error_triggers_retry(self):
        client = ExtractionClient(_make_config(max_retries=1))
        bad = {"entities": [{"canonical_name": "x"}], "relations": []}
        client._client = MagicMock()
        client._client.chat.completions.create.side_effect = [
            _mock_response(json.dumps(bad)),
            _mock_response(json.dumps(_valid_payload())),
        ]

        result = client.extract("content")
        assert len(result.entities) == 2

    def test_passes_model_temperature_response_format(self):
        client = ExtractionClient(_make_config(model="my-model", temperature=0.5))
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _mock_response(
            json.dumps({"entities": [], "relations": []})
        )

        client.extract("content")
        kwargs = client._client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "my-model"
        assert kwargs["temperature"] == 0.5
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "ExtractionResult"
        assert rf["json_schema"]["strict"] is True
        schema = rf["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        ent_item = schema["properties"]["entities"]["items"]
        rel_item = schema["properties"]["relations"]["items"]
        assert ent_item["additionalProperties"] is False
        assert rel_item["additionalProperties"] is False
        assert ent_item["properties"]["entity_type"]["enum"]
        assert rel_item["properties"]["predicate"]["enum"] == DEFAULT_PREDICATES

    def test_empty_result_is_valid(self):
        client = ExtractionClient(_make_config())
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _mock_response(
            json.dumps({"entities": [], "relations": []})
        )

        result = client.extract("何もない")
        assert result.entities == []
        assert result.relations == []

    def test_none_message_content_treated_as_invalid(self):
        client = ExtractionClient(_make_config(max_retries=0))
        resp = _mock_response("")
        resp.choices[0].message.content = None
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = resp

        with pytest.raises(ExtractionError):
            client.extract("content")


class TestStripCodeFence:
    def test_strips_json_fence(self):
        raw = '```json\n{"x": 1}\n```'
        assert _strip_code_fence(raw) == '{"x": 1}'

    def test_strips_unlabeled_fence(self):
        raw = '```\n{"x": 1}\n```'
        assert _strip_code_fence(raw) == '{"x": 1}'

    def test_passthrough_when_no_fence(self):
        raw = '{"x": 1}'
        assert _strip_code_fence(raw) == '{"x": 1}'

    def test_handles_surrounding_whitespace(self):
        raw = '  \n```json\n{"x": 1}\n```\n  '
        assert _strip_code_fence(raw) == '{"x": 1}'

    def test_extract_recovers_from_fenced_response(self):
        client = ExtractionClient(_make_config())
        client._client = MagicMock()
        client._client.chat.completions.create.return_value = _mock_response(
            '```json\n' + json.dumps(_valid_payload()) + '\n```'
        )
        result = client.extract("content")
        assert len(result.entities) == 2


class TestSchemas:
    def test_entity_description_defaults_empty(self):
        e = ExtractedEntity(canonical_name="x", entity_type="person")
        assert e.description == ""

    def test_relation_description_defaults_empty(self):
        r = ExtractedRelation(src="a", dst="b", predicate="likes")
        assert r.description == ""

    def test_extraction_result_defaults_empty_lists(self):
        r = ExtractionResult()
        assert r.entities == []
        assert r.relations == []
