# papers

Shared library for querying the [OpenAlex](https://openalex.org) academic research database.
Used by `papers-mcp` (MCP server) and `papers-cli` (CLI tool).

## What's in here

- **`src/api.rs`** — 28 async functions wrapping the OpenAlex REST API (10 list, 10 get, 7 autocomplete, 1 find)
- **`src/summary.rs`** — Slim summary structs returned by list functions; large arrays omitted for conciseness
- **`src/lib.rs`** — Re-exports all `papers-openalex` types that consumers need

## Usage

```toml
[dependencies]
papers = { path = "../papers" }
```

```rust
use papers::{OpenAlexClient, ListParams, api};

let client = OpenAlexClient::new();
let params = ListParams { search: Some("transformer".into()), ..Default::default() };
let results = api::work_list(&client, &params).await?;
println!("{} works found", results.meta.count);
for work in &results.results {
    println!("{} ({})", work.title, work.publication_year.unwrap_or(0));
}
```

## API functions

```rust
// List — returns SlimListResponse<XxxSummary>
api::work_list(client, params)
api::author_list(client, params)
api::source_list(client, params)
api::institution_list(client, params)
api::topic_list(client, params)
api::publisher_list(client, params)
api::funder_list(client, params)

// Get — returns the full entity struct
api::work_get(client, id, params)
api::author_get(client, id, params)
api::source_get(client, id, params)
api::institution_get(client, id, params)
api::topic_get(client, id, params)
api::publisher_get(client, id, params)
api::funder_get(client, id, params)

// Autocomplete — returns AutocompleteResponse
api::work_autocomplete(client, query)
api::author_autocomplete(client, query)
api::source_autocomplete(client, query)
api::institution_autocomplete(client, query)
api::publisher_autocomplete(client, query)
api::funder_autocomplete(client, query)

// Semantic search — auto-selects GET vs POST based on query length
api::work_find(client, params)  // requires OPENALEX_KEY env var
```

## List response slimming

List functions return `SlimListResponse<T>` rather than the full entity. This strips large
arrays that are rarely needed when scanning many results. See [`CHANGES.md`](./CHANGES.md)
for the complete list of omitted fields per entity type.

Use the `_get` functions to retrieve the full record for a specific entity.
