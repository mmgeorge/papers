# openalex

Async Rust client for the [OpenAlex REST API](https://docs.openalex.org).

OpenAlex is a free, open catalog of scholarly research: 240M+ works, 110M+ authors, and metadata across sources, institutions, topics, publishers, and funders.

## Quick start

```rust
use openalex::{OpenAlexClient, ListParams};

#[tokio::main]
async fn main() -> openalex::Result<()> {
    let client = OpenAlexClient::new();

    let params = ListParams::builder()
        .search("machine learning")
        .per_page(5)
        .build();
    let response = client.list_works(&params).await?;

    println!("Found {} works", response.meta.count);
    for work in &response.results {
        println!("  {}", work.display_name.as_deref().unwrap_or("untitled"));
    }
    Ok(())
}
```

## Authentication

Set the `OPENALEX_KEY` environment variable, or pass the key explicitly. Optional for most endpoints; recommended for higher rate limits.

```rust
use openalex::OpenAlexClient;

let client = OpenAlexClient::new();                       // reads OPENALEX_KEY from env
let client = OpenAlexClient::with_api_key("your-key");    // explicit key
```

The `/find/works` semantic search endpoint requires an API key (1,000 credits per request).

## Examples

### Filtering and sorting

```rust
let params = ListParams::builder()
    .search("machine learning")
    .filter("publication_year:2024,is_oa:true")
    .sort("cited_by_count:desc")
    .per_page(10)
    .build();

let response = client.list_works(&params).await?;
```

### Single entity by ID

Accepts OpenAlex IDs, DOIs, ORCIDs, ROR IDs, ISSNs, PMIDs, and other formats depending on entity type.

```rust
let work = client.get_work("W2741809807", &GetParams::default()).await?;
let work = client.get_work("https://doi.org/10.7717/peerj.4375", &GetParams::default()).await?;

// Select specific fields
let params = GetParams::builder().select("id,display_name,cited_by_count").build();
let work = client.get_work("W2741809807", &params).await?;
```

### Cursor pagination

Offset pagination caps at 10,000 results. Use cursor pagination for deeper access:

```rust
let mut cursor = Some("*".to_string());

while let Some(c) = cursor {
    let params = ListParams {
        cursor: Some(c),
        per_page: Some(200),
        filter: Some("publication_year:2024".into()),
        ..Default::default()
    };
    let response = client.list_works(&params).await?;
    // process response.results
    cursor = response.meta.next_cursor;
}
```

### Autocomplete

Returns up to 10 results sorted by citation count (~200ms):

```rust
let response = client.autocomplete_authors("einstein").await?;
```

### Group-by aggregation

```rust
let params = ListParams::builder()
    .filter("publication_year:2024")
    .group_by("type")
    .build();
let response = client.list_works(&params).await?;
for group in &response.group_by {
    println!("{}: {} works", group.key_display_name, group.count);
}
```

### Semantic search

Requires an API key.

```rust
let client = OpenAlexClient::with_api_key("your-key");
let params = FindWorksParams::builder()
    .query("machine learning for drug discovery")
    .count(5)
    .build();
let response = client.find_works(&params).await?;
```

## API coverage

| Entity | Struct | List | Get | Autocomplete |
|--------|--------|------|-----|--------------|
| Works | `Work` | `list_works` | `get_work` | `autocomplete_works` |
| Authors | `Author` | `list_authors` | `get_author` | `autocomplete_authors` |
| Sources | `Source` | `list_sources` | `get_source` | `autocomplete_sources` |
| Institutions | `Institution` | `list_institutions` | `get_institution` | `autocomplete_institutions` |
| Topics | `Topic` | `list_topics` | `get_topic` | -- |
| Publishers | `Publisher` | `list_publishers` | `get_publisher` | `autocomplete_publishers` |
| Funders | `Funder` | `list_funders` | `get_funder` | `autocomplete_funders` |
| Concepts (deprecated) | -- | -- | -- | `autocomplete_concepts` |

Plus `find_works` / `find_works_post` for semantic search.

### Parameters

| Struct | Used by | Fields |
|--------|---------|--------|
| `ListParams` | 7 list endpoints | `filter`, `search`, `sort`, `per_page`, `page`, `cursor`, `sample`, `seed`, `select`, `group_by` |
| `GetParams` | 7 get endpoints | `select` |
| `FindWorksParams` | Semantic search | `query` (required), `count`, `filter` |

### Responses

| Struct | Returned by | Contents |
|--------|-------------|----------|
| `ListResponse<T>` | List endpoints | `meta`, `results: Vec<T>`, `group_by` |
| `AutocompleteResponse` | Autocomplete endpoints | `meta`, `results` (up to 10 matches) |
| `FindWorksResponse` | Semantic search | `results` (ranked by `score` 0.0--1.0) |


## Testing

```sh
cargo test -p openalex
cargo test -p openalex -- --ignored  # integration tests
```
