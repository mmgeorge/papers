# papers-zotero

[![crates.io](https://img.shields.io/crates/v/papers-zotero.svg)](https://crates.io/crates/papers-zotero)

> [!WARNING]
> Internal crate for [`papers`](https://crates.io/crates/papers-cli). API may change without notice.

Async Rust client for the [Zotero Web API v3](https://www.zotero.org/support/dev/web_api/v3/start). Type-safe access to items, collections, tags, saved searches, and groups.

## Quick start

```rust
use papers_zotero::{ZoteroClient, ItemListParams};

let client = ZoteroClient::new("user-id", "api-key");

let params = ItemListParams::builder()
    .q("rendering")
    .limit(5)
    .build();
let response = client.list_items(&params).await?;
```

## Authentication

Requires a Zotero API key from <https://www.zotero.org/settings/keys>.

```rust
let client = ZoteroClient::new("user-id", "api-key");
let client = ZoteroClient::from_env().unwrap(); // ZOTERO_USER_ID + ZOTERO_API_KEY
```

## API coverage

| Entity | List | Get |
|--------|------|-----|
| Items | `list_items`, `list_top_items`, `list_trash_items`, `list_item_children`, `list_collection_items`, `list_collection_top_items` | `get_item` |
| Collections | `list_collections`, `list_top_collections`, `list_subcollections` | `get_collection` |
| Tags | `list_tags`, `list_item_tags`, `list_collection_tags` + others | `get_tag` |
| Searches | `list_searches` | `get_search` |
| Groups | `list_groups` | -- |

### Parameters

| Struct | Used by | Key fields |
|--------|---------|------------|
| `ItemListParams` | Item endpoints | `q`, `qmode`, `tag`, `item_type`, `sort`, `direction`, `limit`, `start` |
| `CollectionListParams` | Collection endpoints | `sort`, `direction`, `limit`, `start` |
| `TagListParams` | Tag endpoints | `q`, `qmode`, `limit`, `start`, `sort`, `direction` |

All list endpoints return `PagedResponse<T>` with pagination headers (`Total-Results`, `Last-Modified-Version`).