"""Shared pytest fixtures."""

import pytest

import fravenir.storage.paths as paths_mod


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Isolated project root with data/ and characters/ directories."""
    (tmp_path / "data").mkdir()
    (tmp_path / "characters").mkdir()
    monkeypatch.setattr(paths_mod, "_project_root", lambda: tmp_path)
    return tmp_path
