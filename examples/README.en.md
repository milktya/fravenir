# Example Files

[日本語](README.md) | [English](README.en.md)

This directory contains sample files for creating a character.

- `config.yaml`: template for `characters/<id>/config.yaml`.
- `seed.yaml`: template for `characters/<id>/seed.yaml`.

For a step-by-step first run, see [Quick Start](../docs/setup/quickstart.en.md).

## Recommended Use

Copy the files first and edit them:

```bash
mkdir -p characters/mychar
cp examples/config.yaml characters/mychar/config.yaml
cp examples/seed.yaml characters/mychar/seed.yaml

uv run fravenir create-character mychar
```

## Direct Use For Smoke Tests

You can also pass the examples directly:

```bash
uv run fravenir create-character mychar \
  --config examples/config.yaml \
  --seed examples/seed.yaml
```

This is only for quick smoke tests. The sample `seed.yaml` uses `example` as the identity,
so copy and edit the files for real characters.

## What To Edit

In `config.yaml`, update:

- `character.id`
- `extraction.enabled` and `extraction.base_url` if you use an OpenAI-compatible local LLM
- `server.transport`, `server.host`, and `server.port` for HTTP deployments

In `seed.yaml`, update:

- `identity.canonical_name`
- `identity.aliases`
- `identity.description`
- `personality`
- `initial_episodes`