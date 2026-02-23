# papers-openalex

[![crates.io](https://img.shields.io/crates/v/papers-openalex.svg)](https://crates.io/crates/papers-openalex)

> [!WARNING]
> Internal crate for [`papers`](https://crates.io/crates/papers-cli). API may change without notice.

Async Rust client for the [OpenAlex REST API](https://docs.openalex.org). Covers 240M+ works, 110M+ authors, and metadata across sources, institutions, topics, publishers, and funders.

## Quick start

```rust
use papers_openalex::{OpenAlexClient, ListParams};

let client = OpenAlexClient::new();

let params = ListParams::builder()
    .search("machine learning")
    .per_page(5)
    .build();
let response = client.list_works(&params).await?;
```

## Authentication

Optional for most endpoints. Set `OPENALEX_KEY` or pass it explicitly:

```rust
let client = OpenAlexClient::new();                    // reads OPENALEX_KEY from env
let client = OpenAlexClient::with_api_key("your-key"); // explicit
```

## API coverage

| Entity | List | Get | Autocomplete |
|--------|------|-----|--------------|
| Works | `list_works` | `get_work` | `autocomplete_works` |
| Authors | `list_authors` | `get_author` | `autocomplete_authors` |
| Sources | `list_sources` | `get_source` | `autocomplete_sources` |
| Institutions | `list_institutions` | `get_institution` | `autocomplete_institutions` |
| Topics | `list_topics` | `get_topic` | -- |
| Publishers | `list_publishers` | `get_publisher` | `autocomplete_publishers` |
| Funders | `list_funders` | `get_funder` | `autocomplete_funders` |

Plus `find_works` for AI semantic search (requires API key, 1,000 credits per call).

### Parameters

| Struct | Used by | Key fields |
|--------|---------|------------|
| `ListParams` | List endpoints | `filter`, `search`, `sort`, `per_page`, `page`, `cursor`, `sample`, `select`, `group_by` |
| `GetParams` | Get endpoints | `select` |
| `FindWorksParams` | Semantic search | `query`, `count`, `filter` |