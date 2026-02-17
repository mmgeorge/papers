# papers-zotero

[![crates.io](https://img.shields.io/crates/v/papers-zotero.svg)](https://crates.io/crates/papers-zotero)

Async Rust client for the [Zotero Web API v3](https://www.zotero.org/support/dev/web_api/v3/start).

Zotero is a personal research library manager for collecting, organizing, and citing research papers. This crate provides type-safe access to items, collections, tags, saved searches, and groups.

## Quick start

```rust
use papers_zotero::{ZoteroClient, ItemListParams};

#[tokio::main]
async fn main() -> papers_zotero::Result<()> {
    let client = ZoteroClient::new("16916553", "your-api-key");

    let params = ItemListParams::builder()
        .q("rendering")
        .limit(5)
        .build();
    let response = client.list_items(&params).await?;

    println!("Found {:?} items", response.total_results);
    for item in &response.items {
        println!("  {}", item.data.title.as_deref().unwrap_or("untitled"));
    }
    Ok(())
}
```

## Authentication

A Zotero API key is required. Create one at <https://www.zotero.org/settings/keys>.

```rust
use papers_zotero::ZoteroClient;

// Explicit credentials
let client = ZoteroClient::new("16916553", "your-key");

// Or from ZOTERO_USER_ID and ZOTERO_API_KEY env vars
let client = ZoteroClient::from_env().unwrap();
```

## Examples

### Searching items

```rust
let params = ItemListParams::builder()
    .q("machine learning")
    .item_type("journalArticle")
    .sort("dateModified")
    .direction("desc")
    .limit(10)
    .build();

let response = client.list_items(&params).await?;
```

### Single item by key

```rust
let item = client.get_item("LF4MJWZK").await?;
println!("{}: {}", item.key, item.data.title.as_deref().unwrap_or("?"));
```

### Collections

```rust
use papers_zotero::CollectionListParams;

let collections = client.list_collections(&CollectionListParams::default()).await?;
for coll in &collections.items {
    println!("{} ({} items)", coll.data.name, coll.meta.num_items.unwrap_or(0));
}

// Items in a specific collection
let items = client
    .list_collection_items("BDGZ4NHT", &ItemListParams::default())
    .await?;
```

### Tags

```rust
use papers_zotero::TagListParams;

let tags = client.list_tags(&TagListParams::default()).await?;
for tag in &tags.items {
    println!("{} (type {}, {} items)",
        tag.tag,
        tag.meta.tag_type.unwrap_or(0),
        tag.meta.num_items.unwrap_or(0));
}
```

### Filtering by tag

```rust
let params = ItemListParams::builder()
    .tag("Starred")
    .limit(25)
    .build();
let starred = client.list_items(&params).await?;
```

### Pagination

Zotero uses offset pagination with `start` and `limit`:

```rust
let page1 = client
    .list_items(&ItemListParams::builder().limit(25).start(0).build())
    .await?;
let page2 = client
    .list_items(&ItemListParams::builder().limit(25).start(25).build())
    .await?;
```

### Caching

```rust
use papers_zotero::DiskCache;
use std::time::Duration;

let cache = DiskCache::default_location(Duration::from_secs(600)).unwrap();
let client = ZoteroClient::new("16916553", "your-key").with_cache(cache);
```

## API coverage

| Entity | List | Get |
|--------|------|-----|
| Items | `list_items`, `list_top_items`, `list_trash_items`, `list_item_children`, `list_publication_items`, `list_collection_items`, `list_collection_top_items` | `get_item` |
| Collections | `list_collections`, `list_top_collections`, `list_subcollections` | `get_collection` |
| Tags | `list_tags`, `list_item_tags`, `list_items_tags`, `list_top_items_tags`, `list_trash_tags`, `list_collection_tags`, `list_collection_items_tags`, `list_collection_top_items_tags`, `list_publication_tags` | `get_tag` |
| Searches | `list_searches` | `get_search` |
| Groups | `list_groups` | -- |
| Key | -- | `get_key_info` |

- **Items** — the primary entity: journal articles, books, conference papers, attachments, notes, and 30+ other types. Each item carries creators, tags, collections, and type-specific bibliographic fields (DOI, ISBN, abstract, etc.).

- **Collections** — folders for organizing items into a tree hierarchy. Items can belong to multiple collections.

- **Tags** — labels on items. Two types: user-created (type 0) and automatic/imported (type 1).

- **Saved Searches** — stored search conditions that dynamically match items.

- **Groups** — shared libraries accessible to multiple users.

### Parameters

| Struct | Used by | Fields |
|--------|---------|--------|
| `ItemListParams` | Item endpoints | `q`, `qmode`, `tag`, `item_type`, `item_key`, `since`, `sort`, `direction`, `limit`, `start`, `format`, `include`, `style`, `include_trashed` |
| `CollectionListParams` | Collection endpoints | `sort`, `direction`, `limit`, `start` |
| `TagListParams` | Tag endpoints | `q`, `qmode`, `limit`, `start`, `sort`, `direction` |

### Responses

All list endpoints return `PagedResponse<T>` which combines:
- `items: Vec<T>` — the JSON array body
- `total_results: Option<u64>` — from the `Total-Results` HTTP header
- `last_modified_version: Option<u64>` — from the `Last-Modified-Version` header

Single-entity endpoints (`get_item`, `get_collection`, `get_search`) return the entity directly.

## Testing

```sh
cargo test -p papers-zotero                   # unit + wiremock tests
cargo test -p papers-zotero -- --ignored      # live API tests (requires env vars)
```

Live tests require `ZOTERO_USER_ID` and `ZOTERO_API_KEY` environment variables.
