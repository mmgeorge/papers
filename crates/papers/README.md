# papers

Shared library for querying the [OpenAlex](https://openalex.org) academic research database.
Used by `papers-mcp` (MCP server) and `papers-cli` (CLI tool).

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

## List response slimming

List functions return `SlimListResponse<T>` rather than the full entity. This strips large
arrays that are rarely needed when scanning many results. See [`CHANGES.md`](./CHANGES.md)
for the complete list of omitted fields per entity type.

Use the `_get` functions to retrieve the full record for a specific entity.
