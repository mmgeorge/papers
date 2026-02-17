# papers

Shared Rust library that wraps the `papers-openalex` crate with business logic: slim summary
structs for list responses, 22 async API wrapper functions, and re-exports of all
`papers-openalex` types that consumers need.

Neither `papers-mcp` nor `papers-cli` import `papers-openalex` directly — they get everything
via `papers::*`.

## Architecture

```
src/
  lib.rs       — pub mod declarations + re-exports from papers-openalex
  summary.rs   — 10 slim summary structs + From<FullEntity> impls + SlimListResponse
  api.rs       — 28 async wrapper functions (10 list, 10 get, 7 autocomplete, 1 find)
  filter.rs    — work filter alias resolution (search strings → entity IDs)
tests/
  api.rs       — 27 wiremock tests covering all api functions and CHANGES.md rules
  filter.rs    — 11 wiremock tests for filter alias resolution
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

28 public async functions organized by verb:

| Group | Count | Return type |
|-------|-------|-------------|
| `work_list`, `author_list`, ..., `subfield_list` | 10 | `Result<SlimListResponse<XxxSummary>, FilterError>` |
| `work_get`, `author_get`, ..., `subfield_get` | 10 | `Result<FullEntity, OpenAlexError>` |
| `work_autocomplete`, ..., `funder_autocomplete`, `subfield_autocomplete` | 7 | `Result<AutocompleteResponse, OpenAlexError>` |
| `work_find` | 1 | `Result<FindWorksResponse, OpenAlexError>` |

`work_find` automatically selects POST when `params.query.len() > 2048`.

### filter.rs

Contains the multi-step filter resolution logic used by `work_list` in both MCP
and CLI. This is the only module in the `papers` crate that makes API calls
itself (to resolve search strings to entity IDs).

**Resolution flow for ID-based aliases (author, topic, domain, field, subfield, publisher, source):**

1. Split the alias value on `|` to get individual segments
2. For each segment, detect if it's an OpenAlex ID or a search string:
   - IDs: full URLs (`https://openalex.org/...`), prefixed short IDs (`A123`, `P456`, `T789`, `S012`),
     hierarchy paths (`domains/3`, `fields/17`, `subfields/1702`), bare digits for hierarchy entities
   - Everything else: treated as a search string
3. For search strings, call the corresponding entity's list endpoint with
   `filter=display_name.search:{query}&sort=cited_by_count:desc&per_page=1&select=id`
   (exception: publishers use the `search` query param to match alternate titles)
4. Extract the entity ID from the top result; error if no results
5. Normalize the ID to short form (strip `https://openalex.org/` prefix)
6. Join all resolved IDs with `|`
7. Produce the final filter condition: `{openalex_filter_key}:{joined_ids}`

**Direct-value aliases (year, citations):** passed through as-is to the
corresponding OpenAlex filter key. No API calls needed.

**Overlap detection:** before combining aliases with the raw `filter` param,
parse the raw filter's comma-separated conditions, extract each key (before `:`),
and check for conflicts with alias filter keys. Error if any overlap.

**Final combination:** all resolved alias conditions are joined with `,` (AND)
alongside any raw filter conditions.

### lib.rs re-exports

Re-exports everything consumers need from `papers-openalex` so neither `papers-mcp` nor
`papers-cli` need to declare a direct dependency on `papers-openalex`:

```rust
pub use papers_openalex::{
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

- The `papers-openalex` crate is not re-exported as a module — only specific items are
  re-exported. Consumers should use `papers::OpenAlexClient`, not
  `papers::papers_openalex::OpenAlexClient`.
- All `api.rs` functions take `&OpenAlexClient` (not `self`) so they are plain
  async functions usable from any context.
- `summary_list_result` is `pub(crate)` within `summary.rs` — it is an internal
  helper. The public API is `SlimListResponse<S>` (the output type).

## When to update CHANGES.md

Any time you intentionally change what `api::*_list()` returns relative to the raw
OpenAlex API — added fields, removed fields, renamed fields — update `CHANGES.md`
to document what changed and why. Also update `tests/api.rs` to assert the new shape.
