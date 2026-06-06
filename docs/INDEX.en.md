# fravenir Documentation

[日本語](INDEX.md) | [English](INDEX.en.md)

This directory contains the public documentation for fravenir.

## 1. Overview

Start here if you want to understand what fravenir is and whether it fits your use case:

- [README](../README.en.md): short project overview, core ideas, and main commands.

## 2. Setup And Operations

Use these when installing or running fravenir:

- [Quick Start](setup/quickstart.en.md): create your first character and start the MCP server.
- [Example Files](../examples/README.en.md): sample `config.yaml` and `seed.yaml`.
- [Systemd Operations](operations/systemd_timer.md): run the MCP server and scheduled
  compaction.
- [Admin Server](operations/admin_server.md): optional Admin UI for local or closed-network
  use.
- [Migrations](operations/migrations.md): maintenance reference for upgrading older
  databases to the current schema.
- [Prompt Injection Notes](operations/prompt_injection.md): recommended wrapping for
  untrusted memory text before sending it to a language model.

## 3. Design And Technical References

Use these when modifying, extending, or reviewing the internals:

- [Technical Design](design/technical-design.en.md): architecture, storage model, ACT-R activation,
  search/write flows, graph exploration, MCP interface, and design tradeoffs.

## Suggested Reading Paths

- First-time user: README -> Quick Start -> Example Files.
- Server operator: Quick Start -> Systemd Operations -> Admin Server if needed.
- Existing database upgrade: Quick Start -> Migrations -> Systemd Operations.
- Contributor or fork author: README -> Technical Design.

## Public Scope

Internal working notes, local character files, and runtime databases are intentionally not
part of the public documentation set.
