# papers-cli

CLI binary over the `papers` shared crate. Provides a `papers` command for querying
the OpenAlex academic research database and your personal Zotero reference library.

## Architecture

```
src/
  main.rs      — tokio main; parse Cli; dispatch to papers::api::* functions
  cli.rs       — all clap structs (Cli, EntityCommand, WorkCommand, ZoteroCommand, etc.)
  format.rs    — human-readable text formatters for each entity/response type
tests/
  cli.rs       — wiremock integration tests (format output + slim JSON assertions)
```

Imports only `papers::*` for OpenAlex — no direct dependency on `papers-openalex`.
Imports `papers_zotero::ZoteroClient` directly for Zotero commands.

## CLI commands

### OpenAlex commands
```
papers work list   [-s <query>] [-f <filter>] [--sort <field>] [-n <per_page=10>]
                   [--page <n>] [--cursor <c>] [--sample <n>] [--seed <n>] [--json]
papers work get    <id> [--json]
papers work autocomplete <query> [--json]
papers work find   <query> [-n <count>] [-f <filter>] [--json]

papers author list / get / autocomplete
papers source list / get / autocomplete
papers institution list / get / autocomplete
papers topic list / get
papers publisher list / get / autocomplete
papers funder list / get / autocomplete
```

### Zotero commands

Requires `ZOTERO_USER_ID` and `ZOTERO_API_KEY` env vars. Exits with error if not set.

```
papers zotero work list        [-s <q>] [--tag <t>] [--type <t>] [--sort <f>] [-n <n>] [--json]
papers zotero work get         <key> [--json]
papers zotero work collections <key> [--json]
papers zotero work notes       <key> [-n <n>] [--json]
papers zotero work attachments <key> [-n <n>] [--json]
papers zotero work annotations <key> [--json]
papers zotero work tags        <key> [-q <q>] [--json]
papers zotero attachment list  [-s <q>] [--sort <f>] [-n <n>] [--json]
papers zotero attachment get   <key> [--json]
papers zotero attachment file  <key> -o <output-path>

papers zotero annotation list  [-n <n>] [--json]
papers zotero annotation get   <key> [--json]

papers zotero note list        [-s <q>] [-n <n>] [--json]
papers zotero note get         <key> [--json]

papers zotero collection list           [--sort <f>] [-n <n>] [--top] [--json]
papers zotero collection get    <key>   [--json]
papers zotero collection works  <key>   [-s <q>] [--type <t>] [--sort <f>] [-n <n>] [--json]
papers zotero collection attachments <key> [-n <n>] [--json]
papers zotero collection notes  <key>   [-s <q>] [-n <n>] [--json]
papers zotero collection annotations <key> [--json]
papers zotero collection subcollections <key> [--sort <f>] [-n <n>] [--json]
papers zotero collection tags   <key>   [-q <q>] [--top] [--json]

papers zotero tag list         [-q <q>] [--sort <f>] [-n <n>] [--top] [--trash] [--json]
papers zotero tag get          <name>   [--json]

papers zotero search list      [--json]
papers zotero search get       <key>    [--json]

papers zotero group list       [--json]

papers zotero extract list     [-s <q>] [-n <n>] [--json]
papers zotero extract text     <key|doi|title>
papers zotero extract json     <key|doi|title>
papers zotero extract get      <key|doi|title>
```

`extract list` shows all items that have a DataLab extraction in **either** the
local cache or Zotero (`papers_extract_{key}.zip` backup), with two checkmarks per item:
`[✓ local] [✓ zotero]`. Items with neither are omitted.

`extract text` / `extract json` / `extract get` are **read-only** — they never
invoke DataLab. Use `papers rag work add <key>` to run extraction and index the paper.

Default output is human-readable text. Add `--json` for raw JSON.
Default `--per-page` is 10 (vs API default of 25).

## Output modes

- **Text (default)**: formatted for human reading — titles, authors, key stats
- **JSON (`--json`)**: pretty-printed JSON from `papers::api::*` — list tools return
  slim `SlimListResponse<XxxSummary>`; get tools return full entity

## `work find` auth guard

Before calling the API, `main.rs` checks `std::env::var("OPENALEX_KEY")`.
If absent, exits with an error message and non-zero status.
Do not call the API if the key is missing.

## How to add a new OpenAlex command

1. Add the clap variant to the appropriate `*Command` enum in `cli.rs`
2. Add a formatter in `format.rs` (text output)
3. Wire the new variant into the `match` in `main.rs`, calling `papers::api::*`
4. Add tests in `tests/cli.rs` — at least text + JSON modes

## How to add a new Zotero command

1. Add the variant to the appropriate `Zotero*Command` enum in `cli.rs`
2. Add a formatter in `format.rs` using `papers_zotero` types
3. Wire into the `EntityCommand::Zotero { cmd }` arm in `main.rs`, calling `ZoteroClient` directly
4. Add tests in `tests/cli.rs` — Zotero commands call client methods directly (not `papers::api::*`)

## Zotero client initialization

Two helpers in `main.rs`:

- **`zotero_client()`** — required Zotero client; used by the `papers zotero *` subcommands. Returns `Err` on any failure (not configured, not running, etc.). The Zotero arm calls it and exits early on error.

- **`optional_zotero()`** — optional Zotero enrichment; used by `work get` and `work text` where Zotero provides extra context but is not strictly required.
  - Returns `Ok(Some(client))` when Zotero is configured and reachable
  - Returns `Ok(None)` when Zotero env vars are absent (silently omit Zotero info)
  - Returns `Err(NotRunning)` when Zotero is installed on disk but not running → caller calls `exit_err()` to surface the actionable message
  - Set `ZOTERO_CHECK_LAUNCHED=0` to make `optional_zotero()` return `Ok(None)` instead of `Err` when Zotero is installed but not running

## Key notes

- `format.rs` formatters take references to response types from `papers::*` or `papers_zotero::*`
- Formatters for slim types (`SlimListResponse<XxxSummary>`) for OpenAlex list commands
- Formatters for full entity types (`Work`, `Author`, etc.) for OpenAlex get commands
- `format_autocomplete` and `format_find_works` are shared across all entities
- All list commands default to `per_page = 10`
- `exit_err` in main.rs prints to stderr and exits with code 1

## Running the CLI for testing

Always run `target/debug/papers.exe` directly — never pass API keys on the command line or via
`powershell.exe` env var lookups. All required keys (`ZOTERO_API_KEY`, `ZOTERO_USER_ID`,
`DATALAB_API_KEY`, etc.) are already present in the shell environment.

```
# correct
target/debug/papers.exe rag work add U9PRIZJ7

# wrong — don't do this
ZOTERO_API_KEY=$(powershell.exe -Command "...") target/debug/papers.exe ...
```

## Integration test cache directory

Tests that write fake DataLab cache files must **NOT** write to the production
`papers/datalab` directory. Instead, redirect the cache by setting the
`PAPERS_DATALAB_CACHE_DIR` environment variable before any test code runs:

```rust
static INIT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

fn test_cache_base() -> &'static std::path::PathBuf {
    INIT.get_or_init(|| {
        let dir = dirs::cache_dir().unwrap().join("papers").join("test");
        std::fs::create_dir_all(&dir).unwrap();
        unsafe { std::env::set_var("PAPERS_DATALAB_CACHE_DIR", &dir) };
        dir
    })
}
```

- Write to: `{cache_dir}/papers/test/{key}/`  (NOT `papers/datalab`)
- `PAPERS_DATALAB_CACHE_DIR` is read by `datalab_cache_dir()` and `datalab_cached_item_keys()` in `papers-core`
- Use per-test unique key namespaces (e.g., `EXT00101`) to prevent parallel test interference
- Use a drop-guard (`CacheCleanup`) to remove test dirs even on panic
- See `tests/extract.rs` for the canonical pattern

## Key gotchas

- Do NOT add `papers-openalex` as a direct dependency — use `papers::*` only
- `FindWorksParams` builder uses consumed-builder pattern for optional fields:
  use intermediate `let mut builder = ...` + `if let Some(v) = opt { builder = builder.field(v); }`
- `ZoteroClient` params (`ItemListParams`, `CollectionListParams`, `TagListParams`) use struct literal
  construction: `ItemListParams { item_type: Some("note".into()), limit, start, ..Default::default() }`
  Do NOT use the builder — bon's type-state changes the generic on each `.field()` call, making
  mutable variable reassignment impossible.
- Zotero commands call `ZoteroClient::from_env()` at the top of the `Zotero` arm; exit early if Err
