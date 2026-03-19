# papers-mcp

MCP server wrapping the `papers` crate (which wraps `papers-openalex`), built with `rmcp` v0.15.
Also exposes Zotero personal library access via `papers-zotero`.

## Architecture

```
src/
  lib.rs       — module declarations
  main.rs      — entry point: create PapersMcp, serve on stdio
  server.rs    — PapersMcp struct + 54 tool methods + ServerHandler impl
  params.rs    — tool parameter structs (schemars + serde)
tests/
  tools.rs     — wiremock integration tests for tool invocation
```

The `papers` crate (at `../papers`) owns all business logic:
- Slim summary structs for list responses
- 28 async API wrapper functions
- Re-exports of all `papers-openalex` types

`papers-mcp` only contains the MCP layer (rmcp macros, parameter structs, JSON serialization).
See `../papers/CHANGES.md` for how responses differ from the raw OpenAlex API.

### server.rs

- `PapersMcp` struct holds an `OpenAlexClient`, `Option<ZoteroClient>`, `Option<String>` (`zotero_check_error`), `Option<DatalabClient>`, and a `ToolRouter<Self>`
- `#[tool_router]` macro on the impl block generates a `tool_router()` constructor
- `#[tool]` on each method registers it as an MCP tool with auto-generated JSON Schema
- `#[tool_handler]` on the `ServerHandler` impl generates `call_tool`, `list_tools`, `get_tool`
- Each tool method takes `Parameters<T>` and returns `Result<String, String>`
- Success: JSON-serialized API response. Error: error message string.
- OpenAlex tools (29) delegate to `papers::api::*` functions (no direct papers-openalex imports)
- Zotero tools (25) call `self.zotero` directly — see Zotero tools section below

#### `zotero_check_error` field

`PapersMcp` has a `zotero_check_error: Option<String>` field. During `new()` or `with_client()`,
if `ZoteroClient::from_env_prefer_local()` returns `Err(ZoteroError::NotRunning { .. })`, the
error message is stored here (and `zotero` is set to `None`). This lets us surface the "Zotero is
installed but not running" error on all Zotero-dependent tools rather than silently omitting Zotero.

#### `require_zotero()` helper

All Zotero tools use this centralized guard instead of inline `ok_or_else`:
```rust
fn require_zotero(&self) -> Result<&ZoteroClient, String> { ... }
```
- Returns `Ok(&ZoteroClient)` when connected
- Returns `Err` with the `zotero_check_error` message (e.g. "Zotero is installed but not running...") if set
- Otherwise returns `Err("Zotero not configured. Set ZOTERO_USER_ID and ZOTERO_API_KEY.")`

`work_get` and `work_text` also guard against `zotero_check_error` at their start (even though they
don't require Zotero, they benefit from early error surfacing when Zotero is expected but not running).

#### Zotero tools (25)

All Zotero tools start with:
```rust
let z = self.require_zotero()?;
```

Multi-step tools chain multiple `ZoteroClient` calls:
- `zotero_work_collections`: `get_item(key)` → `get_collection(ck)` for each key in `data.collections`
- `zotero_work_annotations`: `list_item_children(key, attachment)` → `list_item_children(att_key, annotation)` per attachment
- `zotero_collection_annotations`: `list_collection_items(key, attachment)` → `list_item_children(att_key, annotation)` per attachment

Zotero tools by group:
| Group | Tools |
|-------|-------|
| Work | `zotero_work_list`, `zotero_work_get`, `zotero_work_collections`, `zotero_work_notes`, `zotero_work_attachments`, `zotero_work_annotations`, `zotero_work_tags` |
| Attachment | `zotero_attachment_list`, `zotero_attachment_get` |
| Annotation | `zotero_annotation_list`, `zotero_annotation_get` |
| Note | `zotero_note_list`, `zotero_note_get` |
| Collection | `zotero_collection_list`, `zotero_collection_get`, `zotero_collection_works`, `zotero_collection_attachments`, `zotero_collection_notes`, `zotero_collection_annotations`, `zotero_collection_subcollections`, `zotero_collection_tags` |
| Tag | `zotero_tag_list`, `zotero_tag_get` |
| Other | `zotero_search_list`, `zotero_group_list` |

For testing, use `PapersMcp::with_zotero(ZoteroClient::new("test", "key").with_base_url(mock.uri()))`.

**Critical**: Use struct literal construction for `ItemListParams`, `CollectionListParams`, and
`TagListParams` — do NOT use the builder. `bon`'s type-state changes the generic on each `.field()`
call, making mutable variable reassignment impossible. Example:
```rust
let params = ItemListParams { item_type: Some("note".into()), limit: p.limit, ..Default::default() };
```

### params.rs

Structs with `Deserialize` + `JsonSchema`:
- `WorkListToolParams`, `AuthorListToolParams`, etc. — entity list params with conversion methods
- `GetToolParams` — required `id` + optional `select`
- `AutocompleteToolParams` — required `q`
- `FindWorksToolParams` — required `query`, optional `count` and `filter`
- `WorkTextToolParams` — required `key`
- `ZoteroWorkListToolParams`, `ZoteroWorkChildrenToolParams`, `ZoteroWorkTagsToolParams`
- `ZoteroAttachmentListToolParams`, `ZoteroAnnotationListToolParams`, `ZoteroNoteListToolParams`
- `ZoteroCollectionListToolParams`, `ZoteroCollectionWorksToolParams`, `ZoteroCollectionNotesToolParams`
- `ZoteroCollectionSubcollectionsToolParams`, `ZoteroCollectionTagsToolParams`
- `ZoteroTagListToolParams`, `ZoteroKeyToolParams`, `ZoteroTagGetToolParams`, `ZoteroNoParamsToolParams`
- DB params: `DbChunkSearchParams`, `DbChunkGetParams`, `DbFigureSearchParams`, `DbFigureGetParams`,
  `DbWorkListParams`, `DbWorkOutlineParams`, `DbSectionGetParams`, `DbChapterGetParams`, `DbTagListParams`

## CLI / MCP feature overlap

The MCP server exposes **all read/search operations** from the CLI. Write and
mutation operations remain CLI-only. This is by design — the MCP is for LLM
consumption; indexing, syncing, and bulk ops go through the CLI.

### Selection commands

| CLI subcommand              | MCP tool            | Status    |
|-----------------------------|---------------------|-----------|
| `selection list`            | `selection_list`    | Both      |
| `selection set`             | `selection_get`     | Both (MCP activates on get) |
| `selection create`          | `selection_create`  | Both      |
| `selection delete`          | `selection_delete`  | Both      |
| `selection add`             | `selection_add`     | Both      |
| `selection remove`          | `selection_remove`  | Both      |
| `selection status`          | —                   | CLI only  |
| `selection find`            | —                   | CLI only (OA PDF download) |
| `selection sync`            | —                   | CLI only (Zotero sync) |
| `selection merge`           | —                   | CLI only  |
| `selection rename`          | —                   | CLI only  |
| `selection db add`          | —                   | CLI only (batch ingest) |
| `selection db remove`       | —                   | CLI only (batch remove) |
| `selection collection add`  | —                   | CLI only (import Zotero collection) |

### Database commands

| CLI subcommand       | MCP tool            | Status    |
|----------------------|---------------------|-----------|
| `db chunk search`    | `db_chunk_search`   | Both      |
| `db chunk get`       | `db_chunk_get`      | Both      |
| `db chunk list`      | `db_chunk_list`     | Both      |
| `db exhibit search`  | `db_exhibit_search` | Both      |
| `db exhibit get`     | `db_exhibit_get`    | Both      |
| `db work list`       | `db_work_list`      | Both      |
| `db work get`        | `db_work_get`       | Both      |
| `db work search`     | `db_work_search`    | Both      |
| `db work outline`    | `db_work_outline`   | Both      |
| `db work add`        | —                   | CLI only (index paper; `--embed-only` re-embeds without re-extracting) |
| `db work remove`     | —                   | CLI only (remove from index) |
| `db work extract`    | —                   | CLI only (print cached extraction) |
| `db section search`  | `db_section_search` | Both      |
| `db section list`    | `db_section_list`   | Both      |
| `db section get`     | `db_section_get`    | Both      |
| `db chapter search`  | `db_chapter_search` | Both (pending merge into `db_section_*` with depth param) |
| `db chapter list`    | `db_chapter_list`   | Both (pending merge) |
| `db chapter get`     | `db_chapter_get`    | Both (pending merge) |
| `db tag list`        | `db_tag_list`       | Both      |

### CLI-only gaps (not exposed via MCP)

**Selection**: status, find, sync, merge, rename, db add/remove, collection add
**Database**: work add/remove, work extract

When adding new MCP tools that close these gaps, update this table.

### Extraction pipeline

The DB ingestion pipeline uses `papers-extract` (local ONNX-based extraction)
instead of the DataLab cloud API. When `reflow.json` is present in the cache,
the reflow-based chunking pipeline is used; otherwise falls back to the legacy
DataLab Marker JSON pipeline.

Cache locations:
- **New (preferred)**: `<cache_dir>/papers/extracts/{zotero_id}/` (env: `PAPERS_EXTRACT_CACHE_DIR`)
- **Legacy**: `<cache_dir>/papers/datalab/{zotero_id}/` (env: `PAPERS_DATALAB_CACHE_DIR`)

## How to update

When the `papers` crate adds or changes endpoints:
1. Add a new `#[tool]` method to `server.rs`
2. Use the appropriate params struct (or create a new one in `params.rs`)
3. Call the corresponding `papers::api::*` function and wrap with `json_result()`
4. Add a wiremock test in `tests/tools.rs`
5. Run `cargo test -p papers-mcp` to verify

When `rmcp` updates:
1. Build docs locally: `cargo doc -p rmcp --no-deps`
2. Check for breaking changes in `ServerHandler`, `tool_router`, `tool` macros

## Key gotchas

- `rmcp` requires `Clone` on the service struct (PapersMcp)
- `rmcp` uses `schemars` v1 (not v0.8) — must match versions
- All papers-openalex types come from `papers::*` — do NOT add `papers-openalex` as a direct dep
- The `#[tool]` macro transforms async fns — they return `Pin<Box<dyn Future>>`, not regular futures
- `tool_router` visibility must be set via `#[tool_router(vis = "pub")]` for external access
- Tool methods need `pub` visibility to be testable from integration tests
- **Never use `panic!`/`expect`/`unwrap` in code reachable from tool handlers.** The `rmcp`
  framework catches panics from tool handlers and returns a generic MCP error to the client.
  The actual panic message only appears in server stderr — the client never sees the real
  error, resulting in silently failing tools or opaque timeouts. Always use `Result` and `?`
  to propagate errors so they surface as readable MCP error responses.
