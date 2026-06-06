# Quick Start

[日本語](quickstart.md) | [English](quickstart.en.md)

This guide creates one local character and starts fravenir as an MCP server.

## Prerequisites

- Python 3.12
- uv
- A machine that can download the embedding model on first run

fravenir stores generated data in `data/` and character configuration in `characters/`.

## 1. Install Dependencies

```bash
uv sync
```

The first character creation or search may download the embedding model
`cl-nagoya/ruri-v3-310m`.

## 2. Create Character Files

Copy the sample files:

```bash
mkdir -p characters/mychar
cp examples/config.yaml characters/mychar/config.yaml
cp examples/seed.yaml characters/mychar/seed.yaml
```

Edit both files and replace the sample identity with your character id:

```yaml
character:
  id: mychar
```

```yaml
identity:
  canonical_name: mychar
```

`seed.yaml` is where you put the initial identity, personality traits, and first
memories. `config.yaml` controls storage, embedding, activation, LLM extraction, logging,
and server settings.

## 3. Create The Character

```bash
uv run fravenir create-character mychar
```

This reads the configuration you placed in `characters/mychar/` and creates `data/mychar/` with SQLite databases and other runtime data.

Check the result:

```bash
uv run fravenir show-character mychar
```

## 4. Start The MCP Server

For local MCP clients, use the default stdio transport:

```bash
uv run fravenir serve --character mychar
```

In an MCP client config:

```json
{
  "mcpServers": {
    "fravenir_mychar": {
      "command": "uv",
      "args": ["run", "fravenir", "serve", "--character", "mychar"]
    }
  }
}
```

The server exposes memory tools such as `memory_write`, `memory_search`, `memory_get`,
`memory_explore`, `memory_trace`, `memory_delete`, and `memory_compact`.

## 5. Apply Seed Changes Later

After editing `characters/mychar/seed.yaml`, apply the new seed content with:

```bash
uv run fravenir init-character mychar --force
```

Use this when you add new personality traits or initial episodes after the character
already exists.

## 6. Run Maintenance

Run compaction manually:

```bash
uv run fravenir compact mychar --dry-run
uv run fravenir compact mychar
```

For scheduled server operation, see [Systemd Timer](../operations/systemd_timer.md).

## Optional: LLM Extraction

fravenir can use an OpenAI-compatible endpoint to organize memories.

- During `memory_write`: extract entities and relations from episode text and add them to
  the graph.
- During `compact --use-llm`: semantically judge duplicate candidates, relation direction
  conflicts, and contradiction candidates.

`examples/config.yaml` includes a local endpoint example:

```yaml
extraction:
  enabled: true
  base_url: http://127.0.0.1:8080/v1
  api_key: dummy
```

If you do not have a local extraction endpoint yet, set `extraction.enabled` to `false`
before running a character in production. You can still create characters and use the
basic memory database without that endpoint.

To use `compact --use-llm`, also set `semantic_judge.enabled` to `true` and configure
`semantic_judge.base_url` and `semantic_judge.model` for your environment.

## Next Steps

- Read [Example Files](../../examples/README.en.md) to understand the sample config and seed.
- Read [Operations](../operations/systemd_timer.md) for long-running server setups.
- Read [Technical Design](../design/technical-design.en.md) when you want to modify the internals.
