# papers

[![crates.io](https://img.shields.io/crates/v/papers-cli.svg)](https://crates.io/crates/papers-cli)
[![CI](https://github.com/mmgeorge/papers/actions/workflows/ci.yml/badge.svg)](https://github.com/mmgeorge/papers/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Monorepo for `papers` — a CLI and MCP server for searching academic research via OpenAlex, with a Zotero client for personal library management.

[OpenAlex](https://openalex.org) is a fully open catalog of the global research system: over
240 million scholarly works, 90 million author profiles, and comprehensive metadata on journals,
institutions, research topics, publishers, and funders.

[Zotero](https://www.zotero.org) is a personal research library manager for collecting,
organizing, and citing research papers.

## Crates

| Crate | Description |
|-------|-------------|
| [`papers-cli`](crates/papers-cli/) | CLI with human-readable or JSON output. Includes `papers mcp start --stdio` to run the MCP server. |
| [`papers-mcp`](crates/papers-mcp/) | MCP server library (started via `papers mcp start --stdio`). |
| [`papers-openalex`](crates/papers-openalex/) | Typed client for the OpenAlex REST API. |
| [`papers-zotero`](crates/papers-zotero/) | Typed client for the Zotero Web API v3. |
| [`papers-datalab`](crates/papers-datalab/) | Client for the Datalab document processing API. |

