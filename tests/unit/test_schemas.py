"""Unit tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from fravenir.schemas.config import AppConfig
from fravenir.schemas.seed import SeedConfig


class TestSeedConfig:
    def test_minimal(self):
        seed = SeedConfig.model_validate({
            "identity": {"canonical_name": "mina"},
        })
        assert seed.identity.canonical_name == "mina"
        assert seed.personality == []
        assert seed.initial_episodes == []

    def test_with_episodes(self):
        seed = SeedConfig.model_validate({
            "identity": {"canonical_name": "mina", "aliases": ["あたし"]},
            "initial_episodes": [
                {"content": "test", "kind": "facts", "importance": 3}
            ],
        })
        assert len(seed.initial_episodes) == 1
        assert seed.initial_episodes[0].kind == "facts"
        assert len(seed.identity.aliases) == 1

    def test_invalid_kind(self):
        with pytest.raises(ValidationError):
            SeedConfig.model_validate({
                "identity": {"canonical_name": "x"},
                "initial_episodes": [{"content": "x", "kind": "invalid"}],
            })

    def test_importance_out_of_range(self):
        with pytest.raises(ValidationError):
            SeedConfig.model_validate({
                "identity": {"canonical_name": "x"},
                "initial_episodes": [{"content": "x", "kind": "facts", "importance": 5}],
            })

    def test_personality_parsed_but_not_required(self):
        seed = SeedConfig.model_validate({
            "identity": {"canonical_name": "mina"},
            "personality": [
                {"canonical_name": "好奇心旺盛", "self_weight": 0.8}
            ],
        })
        assert len(seed.personality) == 1
        assert seed.personality[0].self_weight == 0.8



class TestSeedEntityConfig:
    def test_seed_entities_empty_by_default(self) -> None:
        seed = SeedConfig.model_validate({"identity": {"canonical_name": "mina"}})
        assert seed.seed_entities == []

    def test_seed_entities_parsed(self) -> None:
        seed = SeedConfig.model_validate(
            {
                "identity": {"canonical_name": "mina"},
                "seed_entities": [
                    {
                        "canonical_name": "みるちゃ",
                        "entity_type": "person",
                        "description": "AI と一緒に暮らしたい男性デザイナー",
                        "aliases": ["みるちゃん"],
                    },
                    {
                        "canonical_name": "fravenir",
                        "entity_type": "work",
                        "aliases": ["fravenir"],
                    },
                ],
            }
        )
        assert len(seed.seed_entities) == 2
        assert seed.seed_entities[0].canonical_name == "みるちゃ"
        assert seed.seed_entities[0].aliases == ["みるちゃん"]
        assert seed.seed_entities[1].entity_type == "work"
        # description はデフォルト空
        assert seed.seed_entities[1].description == ""

    def test_seed_entity_description_max_length(self) -> None:
        with pytest.raises(ValidationError):
            SeedConfig.model_validate(
                {
                    "identity": {"canonical_name": "x"},
                    "seed_entities": [
                        {"canonical_name": "y", "description": "z" * 4001}
                    ],
                }
            )

    def test_seed_entity_requires_canonical_name(self) -> None:
        with pytest.raises(ValidationError):
            SeedConfig.model_validate(
                {
                    "identity": {"canonical_name": "x"},
                    "seed_entities": [{"canonical_name": ""}],
                }
            )


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig.model_validate({"character": {"id": "mina"}})
        assert cfg.character.id == "mina"
        assert cfg.act_r.base_decay == 0.5
        assert cfg.act_r.self_decay == 0.2
        assert cfg.act_r.self_boost_beta == 0.5
        assert cfg.embedding.model == "cl-nagoya/ruri-v3-310m"
        assert cfg.embedding.dim == 768
        assert cfg.embedding.max_tokens == 8192
        assert cfg.embedding.normalize is True
        assert cfg.embedding.prefixes.query == "検索クエリ: "
        assert cfg.embedding.prefixes.document == "検索文書: "
        assert cfg.embedding.prefixes.topic == "トピック: "
        assert cfg.embedding.prefixes.general == ""
        assert cfg.session.auto_timeout_minutes == 15
        assert cfg.logging.format == "json"
        assert cfg.server.transport == "stdio"
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.port == 8280

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({
                "character": {"id": "x"},
                "logging": {"level": "VERBOSE"},
            })

    def test_invalid_log_format(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({
                "character": {"id": "x"},
                "logging": {"format": "xml"},
            })

    def test_negative_decay_rejected(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({
                "character": {"id": "x"},
                "act_r": {"base_decay": -0.1},
            })

    def test_server_transport_accepted(self):
        cfg = AppConfig.model_validate({
            "character": {"id": "x"},
            "server": {"transport": "streamable-http", "host": "0.0.0.0", "port": 9000},
        })
        assert cfg.server.transport == "streamable-http"
        assert cfg.server.host == "0.0.0.0"
        assert cfg.server.port == 9000

    def test_server_invalid_transport(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({
                "character": {"id": "x"},
                "server": {"transport": "grpc"},
            })

    def test_server_port_out_of_range(self):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({
                "character": {"id": "x"},
                "server": {"port": 0},
            })
        with pytest.raises(ValidationError):
            AppConfig.model_validate({
                "character": {"id": "x"},
                "server": {"port": 70000},
            })

    @pytest.mark.parametrize(
        "bad_id",
        ["../evil", "foo/bar", "", "with space", "a" * 65, "name.dot"],
    )
    def test_character_id_pattern_rejects_invalid(self, bad_id):
        with pytest.raises(ValidationError):
            AppConfig.model_validate({"character": {"id": bad_id}})
