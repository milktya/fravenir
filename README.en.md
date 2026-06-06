# fravenir

[日本語](README.md) | [English](README.en.md)

> [!NOTE]
> This is a personal hobby project. Responses to issues and PRs are not guaranteed.

fravenir is a character memory MCP server.

It gives an AI character a small long-term memory system: the character can store episodes,
search them later, keep a sense of self, and retire old or contradicted facts without
losing their history.

## What It Does

fravenir combines four ideas:

- **Episode memory**: stores short memories as timestamped episodes.
- **Self hub**: keeps identity and personality as graph entities, so self-related memories
  can be found even when the query is vague.
- **ACT-R style activation**: ranks memories with recency, access history, importance,
  semantic similarity, and graph associations.
- **LLM-assisted organization**: optionally extracts entities/relations from written
  memories and runs semantic checks for duplicate or contradictory candidates during
  compaction.

It is built for MCP clients. Each character runs as a separate MCP server, while the tool
names stay simple: `memory_write`, `memory_search`, `memory_get`, `memory_explore`, and
related maintenance tools.

## Safety Note

The Admin UI is optional. It is intended for local use or closed networks such as VPNs,
not direct exposure to the public internet.

If you use the Admin UI, enable HTTP Basic auth and bind it to `127.0.0.1` or a private
network address. If you need internet access, put it behind your own reverse proxy, TLS,
authentication, and access controls.

## Quick Start

Install dependencies:

```bash
uv sync
```

Copy and edit the sample files:

```bash
mkdir -p characters/mychar
cp examples/config.yaml characters/mychar/config.yaml
cp examples/seed.yaml characters/mychar/seed.yaml
```

Set `character.id` and `identity.canonical_name` to `mychar`, then create the character:

```bash
uv run fravenir create-character mychar
```

Start the MCP server:

```bash
uv run fravenir serve --character mychar
```

See [docs/setup/quickstart.en.md](docs/setup/quickstart.en.md) for the full first-run guide.

## MCP Client Example

In an MCP client config, give each character server a distinct server name. The tools
inside the server keep their `memory_*` names.

```json
{
  "mcpServers": {
    "fravenir_mychar": {
      "command": "fravenir",
      "args": ["serve", "--character", "mychar"]
    }
  }
}
```

## Main Commands

```bash
uv run fravenir list-characters
uv run fravenir show-character <id>
uv run fravenir init-character <id> --force
uv run fravenir compact <id> [--dry-run] [--use-llm]
uv run fravenir resolve list <id>
uv run fravenir export <id> --out file.json
uv run fravenir import file.json <id>
```

## Documentation

- [Quick Start](docs/setup/quickstart.en.md): first character setup and MCP launch.
- [Examples](examples/README.en.md): sample `config.yaml` and `seed.yaml`.
- [Operations](docs/operations/): MCP service, scheduled compaction, Admin UI, and
  prompt-injection notes.
- [DB Upgrade](docs/operations/migrations.md): maintenance reference for upgrading older
  databases to the current schema.
- [Design](docs/design/technical-design.en.md): technical design for the memory model,
  storage, scoring, and MCP interface.

The public documentation starts at [docs/INDEX.en.md](docs/INDEX.en.md).

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run mypy src
```

The package targets Python 3.12 and uses Pydantic v2, SQLite, sqlite-vec,
sentence-transformers, structlog, and FastMCP.

## Runtime Data

Character definitions live under `characters/`, and generated databases live under
`data/`.

## License

[MIT](LICENSE)
