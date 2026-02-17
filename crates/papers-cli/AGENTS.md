# papers-cli

CLI binary over the `papers` shared crate. Provides a `papers` command for querying
the OpenAlex academic research database from the terminal.

## Architecture

```
src/
  main.rs      — tokio main; parse Cli; dispatch to papers::api::* functions
  cli.rs       — all clap structs (Cli, EntityCommand, WorkCommand, etc.)
  format.rs    — human-readable text formatters for each entity/response type
tests/
  cli.rs       — wiremock integration tests (format output + slim JSON assertions)
```

Imports only `papers::*` — no direct dependency on `openalex`.

## CLI commands (22 total)

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
papers concept autocomplete <query> [--json]
```

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

## How to add a new command

1. Add the clap variant to the appropriate `*Command` enum in `cli.rs`
2. Add a formatter in `format.rs` (text output)
3. Wire the new variant into the `match` in `main.rs`, calling `papers::api::*`
4. Add tests in `tests/cli.rs` — at least text + JSON modes

## Key notes

- `format.rs` formatters take references to response types from `papers::*`
- Formatters for slim types (`SlimListResponse<XxxSummary>`) for list commands
- Formatters for full entity types (`Work`, `Author`, etc.) for get commands
- `format_autocomplete` and `format_find_works` are shared across all entities
- All list commands default to `per_page = 10`
- `exit_err` in main.rs prints to stderr and exits with code 1

## Key gotchas

- Do NOT add `openalex` as a direct dependency — use `papers::*` only
- `FindWorksParams` builder uses consumed-builder pattern for optional fields:
  use intermediate `let mut builder = ...` + `if let Some(v) = opt { builder = builder.field(v); }`
