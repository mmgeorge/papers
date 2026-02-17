# papers-mcp

MCP server wrapping the `papers` crate (which wraps `papers-openalex`), built with `rmcp` v0.15.

## Architecture

```
src/
  lib.rs       — module declarations
  main.rs      — entry point: create PapersMcp, serve on stdio
  server.rs    — PapersMcp struct + 28 tool methods + ServerHandler impl
  params.rs    — 4 tool parameter structs (schemars + serde)
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

- `PapersMcp` struct holds an `OpenAlexClient` (via `papers::OpenAlexClient`) and a `ToolRouter<Self>`
- `#[tool_router]` macro on the impl block generates a `tool_router()` constructor
- `#[tool]` on each method registers it as an MCP tool with auto-generated JSON Schema
- `#[tool_handler]` on the `ServerHandler` impl generates `call_tool`, `list_tools`, `get_tool`
- Each tool method takes `Parameters<T>` and returns `Result<String, String>`
- Success: JSON-serialized API response. Error: error message string.
- All 28 tools delegate to `papers::api::*` functions (no direct papers-openalex imports)
- List tools return slim `SlimListResponse<XxxSummary>` via `papers::api::*_list()`
- Get/autocomplete/find tools return full entities via `papers::api::*_get/autocomplete/find()`

### params.rs

4 structs with `Deserialize` + `JsonSchema`:
- `ListToolParams` — 10 optional fields for list endpoints
- `GetToolParams` — required `id` + optional `select`
- `AutocompleteToolParams` — required `q`
- `FindWorksToolParams` — required `query`, optional `count` and `filter`

Each has a conversion method to the corresponding `papers`/`papers-openalex` params type.

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
