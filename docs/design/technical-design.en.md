# Technical Design

[日本語](technical-design.md) | [English](technical-design.en.md)

This is the public technical design for fravenir. It summarizes the memory model, major
flows, MCP interface, storage layout, and operational assumptions.

## Purpose

fravenir is an MCP server for long-term character memory. It is designed to:

- store conversation events as episodes
- make self-related memories easier to retrieve
- keep historical versions instead of overwriting facts destructively
- support associative recall through entities and relations
- isolate each character in its own MCP server process

## Architecture

```text
MCP client
  |
  v
fravenir MCP server
  |-- memory_write
  |-- memory_search
  |-- memory_get
  |-- memory_explore
  |-- memory_compact
  |
  v
per-character storage
  |-- kv.sqlite
  |-- vdb_memories.db
  |-- vdb_entities.db
  |-- vdb_relations.db
  |-- cache/
```

One MCP server process handles one character. This keeps each character's memory and tool
namespace separate.

## Data Model

### episodes

An episode is one saved memory.

Important fields:

- `content`: memory text
- `kind`: `facts`, `state`, or `emo`
- `importance`: 1 to 3
- `valid_from` / `valid_to`: validity interval
- `supersedes`: previous episode replaced by this one
- `session_id`: optional session identifier
- `is_suppressed`: set by compaction when a memory should be deprioritized

### entities

Entities are people, concepts, places, works, emotions, or other semantic items extracted
from episodes.

Important fields:

- `canonical_name`
- `entity_type`
- `description`
- `is_self`
- `self_weight`
- `decay_rate`
- `curated_at`

Aliases are stored in `entity_aliases`.

### relations

Relations connect episodes and entities, or entities with other entities.

Important fields:

- `src_type`, `src_id`
- `dst_type`, `dst_id`
- `predicate`
- `strength`
- `fan_out`
- `valid_from`, `valid_to`, `supersedes`

### merge_candidates

`merge_candidates` stores entity pairs that may refer to the same thing. When LLM semantic
judgment is enabled, it also stores judgment labels, confidence, reasons, attempts, and
resolution time.

## Scoring

Search ranking combines ACT-R style activation with vector similarity and importance.

```text
score = activation + alpha_similarity * vector_similarity + alpha_importance * importance
```

Activation is influenced by access history, elapsed time, association strength, and
self-related cues.

## Main Flows

### memory_write

1. Insert the episode.
2. Embed the text and store it in `vdb_memories.db`.
3. If extraction is enabled, extract entities and relations with an OpenAI-compatible LLM
   endpoint.
4. Store new entities and relations.
5. Mark older contradictory facts as no longer valid when applicable.

If LLM extraction fails, the episode and embedding are still kept.

### memory_search

1. Embed the query.
2. Search episode vectors.
3. Expand candidates through related entities and relations.
4. Rerank with activation scoring.
5. Update access history.

### memory_get

Returns a compact set of self and recent-state memories for compatibility with simpler
clients.

### memory_explore

Explores one graph hop from an episode or entity. It complements `memory_search` by
letting the caller inspect nearby nodes step by step.

### memory_compact

Maintenance flow for the memory graph:

- recompute `fan_out`
- update relation `strength`
- suppress low-activation episodes
- detect merge candidates
- optionally run LLM semantic judgment with `--use-llm`

## MCP Interface

| Tool | Purpose |
|---|---|
| `memory_write` | Save one memory |
| `memory_search` | Search related memories |
| `memory_get` | Return self/recent state memories |
| `memory_explore` | Explore one graph hop |
| `memory_delete` | Logically delete an episode |
| `memory_trace` | Follow supersedes history |
| `memory_compact` | Maintain the memory graph |

Character IDs are not included in tool names. Use one server per character and distinguish
characters by server name.

## Storage

```text
data/<character_id>/
  kv.sqlite
  vdb_memories.db
  vdb_entities.db
  vdb_relations.db
  cache/
```

Editable character files live under `characters/<character_id>/`.

## LLM Usage

LLM usage is optional. fravenir can use OpenAI-compatible endpoints for:

- entity/relation extraction during `memory_write`
- semantic judgment during `memory_compact --use-llm`

Configure these through `extraction` and `semantic_judge` in `config.yaml`.

## Security And Operations

- Keep `data/` and `characters/` out of public repositories.
- Prefer local or closed-network operation for HTTP transports and the Admin UI.
- The Admin UI is not intended for direct public internet exposure.
- Treat memory text as user-originated data. Apply prompt-injection precautions before
  passing it to language models.

## Extension Points

- replace sqlite-vec with another vector store
- move graph storage to an external graph database
- add compaction rules
- swap LLM models or prompts
- adjust memory presentation per MCP client
