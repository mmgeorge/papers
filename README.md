# papers

[![crates.io](https://img.shields.io/crates/v/papers-cli.svg)](https://crates.io/crates/papers-cli)
[![Release](https://github.com/mmgeorge/papers/actions/workflows/release.yml/badge.svg)](https://github.com/mmgeorge/papers/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Search, manage, and explore academic papers from the terminal. Or run as an [MCP](https://github.com/mmgeorge/papers/tree/main?tab=readme-ov-file#mcp-server) to query your papers with an LLM. Queries 240M+ works via [OpenAlex](https://openalex.org), integrates with your [Zotero](https://www.zotero.org) library, and builds a local vector index over your papers with [LanceDB](https://github.com/lancedb/lancedb) so you can semantically search across sections and figures. Embedding accelerated with DirectML and CoreML on Windows and macOS.

> [!NOTE]
> Even the best analytical PDF-extraction methods mangle LaTeX and tables for technical papers. This project uses vision-model-based OCR via [Datalab](https://www.datalab.to/) (requires API key) to produce clean markdown with math and tables preserved. Extracted results (JSON, markdown, images) sync back to your Zotero library. You can also [run marker locally](#using-marker-locally).

## Install

Download a prebuilt binary from [releases](https://github.com/mmgeorge/papers/releases), or build from source:

```sh
cargo install --path crates/papers-cli
```

## Commands

| Command | Description |
|---------|-------------|
| `work`, `author`, `source`, `institution`, `topic`, `publisher`, `funder`, `domain`, `field`, `subfield` | Query the OpenAlex catalog |
| `zotero` | Access your Zotero library |
| `rag` | Semantic search over locally indexed papers |
| `selection` | Manage named groups of papers |
| `mcp` | MCP server integration |

Commands accepts `--json` for machine-readable output.

## MCP server

Exposes CLI commands as MCP tools for LLMs. Currently only --stdio is supported.

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

## OpenAlex

OpenAlex works without authentication but is rate-limited. Set `OPENALEX_KEY` for higher rate limits ([openalex.org/pricing](https://openalex.org/pricing)).

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
papers zotero work annotations <work>
papers zotero attachment file <work> --output paper.pdf
papers zotero collection list --top
```

Entities: `work`, `attachment`, `annotation`, `note`, `collection`, `tag`, `search`, `group`.

## Extraction

PDF extraction uses vision-model-based OCR via marker to produce clean markdown with LaTeX math and tables preserved. Extracted results (markdown, JSON, images) are cached locally and synced back to your Zotero library as attachments.

Requires `DATALAB_API_KEY` ([datalab.to](https://www.datalab.to/)).

```sh
papers zotero work extract <work>                    # Extract a Zotero item (default: balanced)
```

Processing modes: `fast`, `balanced` (default), `accurate`.

### Managing cached extractions

```sh
papers zotero extract list                               # List items with cached extractions
papers zotero extract text <work>                        # Get markdown
papers zotero extract upload [--dry-run]                 # Upload local cache to Zotero
papers zotero extract download [--dry-run]               # Download Zotero cache to local
```

Cache location: `~/.cache/papers/datalab/` (Linux/macOS) or `%APPDATA%\papers\datalab\` (Windows). Override with `PAPERS_DATALAB_CACHE_DIR`.

### Using marker locally

You can run [marker](https://github.com/datalab-to/marker) locally instead of using the Datalab API if you meet its [license requirements](https://github.com/datalab-to/marker?tab=readme-ov-file#commercial-usage). Place the output files in the cache directory:

```
~/.cache/papers/datalab/<item_key>/
├── <item_key>.md
├── <item_key>.json     # optional
└── images/             # optional
```

The extraction will be picked up automatically from the local cache.

## RAG

Local semantic search over your papers using [LanceDB](https://github.com/lancedb/lancedb) and [Embedding Gemma 300M](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) (via [FastEmbed](https://github.com/Anush008/fastembed-rs) + [ONNX Runtime](https://onnxruntime.ai)). Hardware-accelerated with DirectML (Windows) and CoreML (macOS).

```
  Marker OCR          Structural Chunking        Embed (Gemma 300M)
                                                                     ┌─────────┐
 ┌──────────┐    ┌─────────────────────┐    ┌───────────┐            │         │
 │  marker  ├───►│  JSON block tree    ├───►│ 768-d f32 ├───────────►│ LanceDB │
 │  OCR     │    │  per paragraph,     │    │ vectors   │            │         │
 │          │    │  equation, table,   │    └───────────┘            │ chunks  │
 └──────────┘    │  figure, list       │         ▲                   │ figures │
   .md .json     └─────────────────────┘         │                   │         │
                                          ┌──────┴──────┐            └────┬────┘
                                          │ embed query │───── ANN ──────┘
                                          └─────────────┘
```

PDFs are sent to [Datalab Marker](https://www.datalab.to/) for vision-model OCR, which returns a structured JSON block tree alongside markdown. Each block (paragraph, equation, list, table, figure) becomes one chunk — no fixed-size splitting or overlap. Chunks and figure captions are embedded into 768-d vectors and stored in LanceDB. At query time, the query is embedded with the same model and matched via approximate nearest neighbor (ANN) search. Each result includes truncated previews of its neighboring chunks for surrounding context.

```sh
papers rag ingest <work>                 # Index a single paper
papers rag ingest-all                    # Index all cached extractions
papers rag search "differentiable rendering" -n 5
papers rag search-figures "neural radiance field architecture"
papers rag get-chunk <chunk_id>
papers rag get-section <work> <chapter> <section>
papers rag outline <work>
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

