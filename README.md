# papers

[![crates.io](https://img.shields.io/crates/v/papers-cli.svg)](https://crates.io/crates/papers-cli)
[![CI](https://github.com/mmgeorge/papers/actions/workflows/ci.yml/badge.svg)](https://github.com/mmgeorge/papers/actions/workflows/ci.yml)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Search, manage, and explore academic papers from the terminal or LLM. Queries 240M+ works via [OpenAlex](https://openalex.org), integrates with your [Zotero](https://www.zotero.org) library, and builds a local vector index over your papers embeddings (using DirectML and CoreML) with [LanceDB](https://github.com/lancedb/lancedb) so you can semantically search across sections, figures, and chunks locally.

Also runs as an [MCP](https://modelcontextprotocol.io) server, exposing all functionality as tools for LLMs.

> [!NOTE]
> Producing a high quality OCR for technical papers is complex. Even the best PDF-extraction mangle LaTeX and tables. This project uses vision-model-based OCR via [Datalab](https://www.datalab.to/) (requires API key) to produce clean markdown with math and tables preserved. Extracted results (JSON, markdown, images) sync back to your Zotero library.
>
> You can also run [marker](https://github.com/datalab-to/marker) locally if you meet its [license requirements](https://github.com/datalab-to/marker?tab=readme-ov-file#commercial-usage) — just place the output in the cache directory.

## Install

```sh
cargo install --path crates/papers-cli
```

## Commands

| Command | Description |
|---------|-------------|
| `work`, `author`, `source`, `institution`, `topic`, `publisher`, `funder`, `domain`, `field`, `subfield` | Query the OpenAlex catalog |
| `zotero` | Access your personal Zotero library |
| `rag` | Semantic search over locally indexed papers |
| `selection` | Manage named groups of papers |
| `mcp start` | Start the MCP server (stdio transport) |

Commands accepts `--json` for machine-readable output.

## OpenAlex

### Search and filter

```sh
papers work list -s "attention is all you need" -n 3
papers work list --author "Yann LeCun" --year 2020-2024 --open
papers work list --topic "deep learning" --citations ">100" --sort cited_by_count:desc
papers author list --institution harvard --country US --h-index ">50"
```

[Filter aliases](#filter-aliases) (`--author`, `--year`, `--topic`, `--citations`, etc.) resolve names to OpenAlex IDs automatically. You can also use raw [OpenAlex filter syntax](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists) via `--filter`.

### Get by ID or search

The `get` subcommand accepts OpenAlex IDs, DOIs, ORCIDs, ROR IDs, PubMed IDs, ISSNs, or plain search queries:

```sh
papers work get "attention is all you need"
papers work get https://doi.org/10.7717/peerj.4375
papers author get "yoshua bengio"
papers institution get "MIT"
```

## Zotero

Requires `ZOTERO_USER_ID` and `ZOTERO_API_KEY` environment variables ([zotero.org/settings/keys](https://www.zotero.org/settings/keys)).

```sh
papers zotero work list --tag Starred --sort dateModified --direction desc
papers zotero work list --search "rendering" --type conferencePaper -n 5
papers zotero work annotations <key>
papers zotero attachment file <key> --output paper.pdf
papers zotero collection list --top
```

Entities: `work`, `attachment`, `annotation`, `note`, `collection`, `tag`, `search`, `group`.

## RAG

Local semantic search over your papers using [LanceDB](https://github.com/lancedb/lancedb) and [Embedding Gemma 300M](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) (via [FastEmbed](https://github.com/Anush008/fastembed-rs) + [ONNX Runtime](https://onnxruntime.ai)). Inference runs on-device with hardware acceleration via DirectML on Windows and CoreML on macOS.

```sh
papers rag ingest                        # Index papers from marker cache
papers rag search "differentiable rendering" -n 5
papers rag search-figures "neural radiance field architecture"
papers rag get-section <paper> <section>
papers rag outline <paper>
```

## MCP server

Exposes all functionality as MCP tools for AI assistants.

**Claude Code:**

```sh
claude mcp add papers -- papers mcp start --stdio
```

**`.mcp.json` (Claude Desktop, Cursor, etc.):**

```json
{
  "mcpServers": {
    "papers": {
      "command": "papers",
      "args": ["mcp", "start", "--stdio"]
    }
  }
}
```

## Filter aliases

Shorthand flags that resolve to OpenAlex filter expressions. Entity-based aliases accept an OpenAlex ID or a search string (resolved to the top result by citation count).

### `work list`

| Flag | Example | Resolves to |
|------|---------|-------------|
| `--author` | `"einstein"`, `A5108093963` | `authorships.author.id:<id>` |
| `--topic` | `"deep learning"`, `T10320` | `topics.id:<id>` |
| `--domain` | `"physical sciences"`, `3` | `topics.domain.id:<id>` |
| `--field` | `"computer science"`, `17` | `topics.field.id:<id>` |
| `--subfield` | `"artificial intelligence"`, `1702` | `topics.subfield.id:<id>` |
| `--publisher` | `"acm"`, `"acm\|ieee"` | `primary_location.source.publisher_lineage:<id>` |
| `--source` | `"nature"`, `S137773608` | `primary_location.source.id:<id>` |
| `--institution` | `"mit"`, `I136199984` | `authorships.institutions.lineage:<id>` |
| `--year` | `2024`, `>2008`, `2008-2024` | `publication_year:<value>` |
| `--citations` | `">100"`, `"10-50"` | `cited_by_count:<value>` |
| `--country` | `US`, `GB` | `authorships.countries:<value>` |
| `--continent` | `europe`, `asia` | `authorships.continents:<value>` |
| `--type` | `article`, `preprint` | `type:<value>` |
| `--open` | *(flag)* | `is_oa:true` |

### `author list`

| Flag | Example |
|------|---------|
| `--institution` | `"harvard"`, `I136199984` |
| `--country` | `US`, `GB` |
| `--continent` | `europe`, `asia` |
| `--citations` | `">1000"`, `"100-500"` |
| `--works` | `">500"`, `"100-200"` |
| `--h-index` | `">50"`, `"10-20"` |

