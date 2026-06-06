# Repository Guidelines

## Project Structure & Module Organization

This repository implements `fravenir`, a Python 3.12 MCP server for character memory using ACT-R activation, self-hub entities, and lightweight PROV-O metadata.

- `src/fravenir/` contains the package code. Core behavior lives in `core/`, storage in `storage/`, MCP/CLI entry points in `server.py` and `cli.py`, schemas in `schemas/`, and the admin UI in `admin/`.
- `tests/` is split into `unit/`, `integration/`, and `golden/` suites, with shared fixtures in `tests/conftest.py`.
- `docs/` holds design, milestone, and operations notes. Start with `docs/INDEX.md` before editing architecture-sensitive areas.
- `examples/` contains sample config and seed files. `characters/` and `data/` are local runtime areas.

## Build, Test, and Development Commands

- `uv sync` installs runtime and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run fravenir --help` checks the CLI entry point.
- `uv run fravenir serve --character <id>` runs the MCP server locally for a character.
- `uv run pytest` runs the default test set, excluding `slow` and `golden_llm` markers.
- `uv run pytest -m slow` runs model-loading tests when needed.
- `uv run ruff check src tests` runs lint and import-order checks.
- `uv run mypy src` runs strict type checking for package code.

## Coding Style & Naming Conventions

Use Python 3.12, Pydantic v2, and structured logging via `structlog`. Ruff is configured for 100-character lines and rules `E`, `F`, `I`, `UP`, `B`, and `SIM`. Keep modules small and domain-named, such as `core/search.py` or `migrations/resolved_at.py`. Prefer explicit typed functions; `mypy` is strict for `src/`.

## Testing Guidelines

Use `pytest`. Put fast behavior tests in `tests/unit/`, database or end-to-end flows in `tests/integration/`, and curated expected-output checks in `tests/golden/`. Name files `test_<feature>.py` and tests `test_<behavior>()`. Maintain at least 80% coverage when changing source behavior.

## Commit & Pull Request Guidelines

Git history uses concise Conventional Commit-style messages, often with phase scopes, for example `feat(phase6): ...` or `docs(phase7): ...`. Use `feat`, `fix`, `docs`, `test`, or `refactor` where appropriate.

PRs should describe the change, list affected commands or MCP tools, link relevant docs or issues, and include test results. For admin UI changes, include screenshots or note why they are unnecessary.

## Agent-Specific Instructions

Before implementation work, check recent handovers when present and consult `docs/INDEX.md` plus the relevant design section.
