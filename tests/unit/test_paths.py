"""Unit tests for storage.paths character_id validation (SEC-1 HIGH-1)."""

from __future__ import annotations

import pytest

from fravenir.storage.paths import character_dir, data_dir


@pytest.mark.parametrize(
    "bad_id",
    ["../evil", "/abs/path", "foo/bar", "", "with space", "a" * 65, "name.dot"],
)
def test_data_dir_rejects_invalid_character_id(bad_id):
    with pytest.raises(ValueError, match="Invalid character_id"):
        data_dir(bad_id)


@pytest.mark.parametrize(
    "bad_id",
    ["../evil", "/abs/path", "foo/bar", ""],
)
def test_character_dir_rejects_invalid_character_id(bad_id):
    with pytest.raises(ValueError, match="Invalid character_id"):
        character_dir(bad_id)


@pytest.mark.parametrize("good_id", ["mina", "yuki", "test_char", "a-b_c-1", "x"])
def test_data_dir_accepts_valid_slug(good_id):
    assert data_dir(good_id).name == good_id
