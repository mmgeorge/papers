# papers

Shared Rust library that wraps the `openalex` crate with business logic: slim summary
structs for list responses, 22 async API wrapper functions, and re-exports of all
`openalex` types that consumers need.

Neither `papers-mcp` nor `papers-cli` import `openalex` directly — they get everything
via `papers::*`.

## Architecture

```
src/
  lib.rs       — pub mod declarations + re-exports from openalex
  summary.rs   — 7 slim summary structs + From<FullEntity> impls + SlimListResponse
  api.rs       — 22 async wrapper functions (7 list, 7 get, 7 autocomplete, 1 find)
tests/
  api.rs       — 21 wiremock tests covering all api functions and CHANGES.md rules
CHANGES.md     — documents every intentional difference vs the raw OpenAlex API
```

### summary.rs

Contains 7 slim structs (`WorkSummary`, `AuthorSummary`, `SourceSummary`,
`InstitutionSummary`, `TopicSummary`, `PublisherSummary`, `FunderSummary`) plus:

- `SlimListResponse<S>` — wraps `ListMeta` + `Vec<S>`, omitting `group_by`
- `summary_list_result()` — internal helper used by `api.rs` list functions:
  takes a `Result<ListResponse<T>, OpenAlexError>` and a mapping fn, returns
  `Result<SlimListResponse<S>, OpenAlexError>`

Each summary struct implements `From<FullEntity>` to extract the relevant fields.
See `CHANGES.md` for exactly which fields are kept and why.

### api.rs

22 public async functions organized by verb:

| Group | Count | Return type |
|-------|-------|-------------|
| `work_list`, `author_list`, ..., `funder_list` | 7 | `Result<SlimListResponse<XxxSummary>, OpenAlexError>` |
| `work_get`, `author_get`, ..., `funder_get` | 7 | `Result<FullEntity, OpenAlexError>` |
| `work_autocomplete`, ..., `funder_autocomplete` + `concept_autocomplete` | 7 | `Result<AutocompleteResponse, OpenAlexError>` |
| `work_find` | 1 | `Result<FindWorksResponse, OpenAlexError>` |

`work_find` automatically selects POST when `params.query.len() > 2048`.

### lib.rs re-exports

Re-exports everything consumers need from `openalex` so neither `papers-mcp` nor
`papers-cli` need to declare a direct dependency on `openalex`:

```rust
pub use openalex::{
    Author, Funder, Institution, Publisher, Source, Topic, Work,
    OpenAlexClient, OpenAlexError, Result,
    ListParams, GetParams, FindWorksParams,
    ListMeta, ListResponse,
    AutocompleteResponse, AutocompleteResult,
    FindWorksResponse, FindWorksResult,
    GroupByResult,
};
pub use summary::SlimListResponse;
```

## How to add a new entity

1. Add a slim struct in `summary.rs` with `From<FullEntity>` impl
2. Add list/get/autocomplete functions in `api.rs`
3. Re-export the full entity type and any new params from `lib.rs`
4. Add wiremock tests in `tests/api.rs`
5. Update `CHANGES.md` if the list response differs from the raw API

## Key notes

- The `openalex` crate is not re-exported as a module — only specific items are
  re-exported. Consumers should use `papers::OpenAlexClient`, not
  `papers::openalex::OpenAlexClient`.
- All `api.rs` functions take `&OpenAlexClient` (not `self`) so they are plain
  async functions usable from any context.
- `summary_list_result` is `pub(crate)` within `summary.rs` — it is an internal
  helper. The public API is `SlimListResponse<S>` (the output type).

## When to update CHANGES.md

Any time you intentionally change what `api::*_list()` returns relative to the raw
OpenAlex API — added fields, removed fields, renamed fields — update `CHANGES.md`
to document what changed and why. Also update `tests/api.rs` to assert the new shape.
