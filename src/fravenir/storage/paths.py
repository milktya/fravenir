"""Path helpers for data/<id>/ and characters/<id>/ directories."""

import re
from pathlib import Path

_CHARACTER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_character_id(character_id: str) -> None:
    if not _CHARACTER_ID_PATTERN.match(character_id):
        raise ValueError(
            f"Invalid character_id: {character_id!r} "
            f"(must match {_CHARACTER_ID_PATTERN.pattern})"
        )


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def data_root() -> Path:
    return _project_root() / "data"


def data_dir(character_id: str) -> Path:
    _validate_character_id(character_id)
    return _project_root() / "data" / character_id


def kv_db_path(character_id: str) -> Path:
    return data_dir(character_id) / "kv.sqlite"


def vdb_memories_path(character_id: str) -> Path:
    return data_dir(character_id) / "vdb_memories.db"


def vdb_entities_path(character_id: str) -> Path:
    return data_dir(character_id) / "vdb_entities.db"


def vdb_relations_path(character_id: str) -> Path:
    return data_dir(character_id) / "vdb_relations.db"


def cache_dir(character_id: str) -> Path:
    return data_dir(character_id) / "cache"


def cache_extractions_dir(character_id: str) -> Path:
    return cache_dir(character_id) / "llm_extractions"


def character_dir(character_id: str) -> Path:
    _validate_character_id(character_id)
    return _project_root() / "characters" / character_id


def seed_yaml_path(character_id: str) -> Path:
    return character_dir(character_id) / "seed.yaml"


def config_yaml_path(character_id: str) -> Path:
    return character_dir(character_id) / "config.yaml"
