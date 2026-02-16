# OpenAlex Rust Client — Agent Guide

## Origin

This crate is a Rust API client for the [OpenAlex REST API](https://docs.openalex.org). OpenAlex is a free, open catalog of the world's scholarly research, containing 240M+ works, 110M+ authors, and related metadata.

## How This Crate Was Derived

1. The API specification was extracted from https://docs.openalex.org/api-guide-for-llms (the LLM-optimized docs page)
2. Individual entity documentation pages were consulted for detailed object schemas:
   - https://docs.openalex.org/api-entities/works/work-object
   - https://docs.openalex.org/api-entities/authors/author-object
   - https://docs.openalex.org/api-entities/sources/source-object
   - https://docs.openalex.org/api-entities/institutions/institution-object
   - https://docs.openalex.org/api-entities/topics/topic-object
   - https://docs.openalex.org/api-entities/publishers/publisher-object
   - https://docs.openalex.org/api-entities/funders/funder-object
3. Every endpoint was called against the live API to verify actual response shapes
4. The results were codified into `api-spec.toml` as the single source of truth

## Architecture

- `api-spec.toml` — Machine-readable API specification: endpoints, parameters, response types, enum values
- `src/client.rs` — `OpenAlexClient` struct with 23 public methods (one per endpoint)
- `src/types/` — Serde-deserializable Rust structs for every entity and nested object
- `src/params.rs` — Parameter structs with `#[derive(Default, bon::Builder)]` for both struct-update and builder patterns
- `src/response.rs` — Generic response wrappers: `ListResponse<T>`, `AutocompleteResponse`, `FindWorksResponse`
- `src/error.rs` — Error types for HTTP, JSON, and API errors
- `tests/fixtures/` — JSON response fixtures captured from the live API

## How to Update When the API Changes

### Step 1: Check for changes
Visit https://docs.openalex.org/api-guide-for-llms and compare against `api-spec.toml`. Look for:
- New endpoints
- New query parameters
- New entity fields
- Changed enum values
- Deprecated features

### Step 2: Verify against live API
Call the changed endpoints directly to confirm actual response shapes:
```
# List endpoint (use per_page=1 for minimal response)
curl "https://api.openalex.org/works?per_page=1" | python -m json.tool

# Single entity (full object)
curl "https://api.openalex.org/works/W2741809807" | python -m json.tool

# Autocomplete
curl "https://api.openalex.org/autocomplete/works?q=test" | python -m json.tool
```

### Step 3: Update api-spec.toml
Add/modify endpoints, parameters, entity fields, or enum values in the TOML spec.

### Step 4: Update Rust code
- New/changed entity fields → update structs in `src/types/*.rs`
- New endpoints → add methods to `src/client.rs`
- New parameters → update structs in `src/params.rs`
- New enum values → update `api-spec.toml` (enum values are strings in Rust, not Rust enums)

### Step 5: Update tests
- Add/update fixtures in `tests/fixtures/`
- Add integration tests in `tests/integration.rs`
- Run `cargo test -p openalex` (unit + mock tests)
- Run `cargo test -p openalex -- --ignored` (live API tests)

## Key Gotchas

- **`type` keyword:** Works, sources, institutions all have a `type` field. Use Rust raw identifier `r#type`
- **`2yr_mean_citedness`:** Not a valid Rust identifier. Mapped via `#[serde(rename = "2yr_mean_citedness")] pub two_yr_mean_citedness`
- **`per-page` vs `per_page`:** API query key is hyphenated `per-page`, Rust field is `per_page`
- **Entity IDs are URIs:** `id` values are full URIs like `https://openalex.org/W2741809807`, not just `W2741809807`
- **Abstract format:** `abstract_inverted_index` is `HashMap<String, Vec<u32>>` (word→positions), not plain text
- **TopicHierarchyLevel.id:** Can be integer (in Topic entity) or string (in Work.topics). Deserialized as `serde_json::Value`
- **Nullable vs missing:** All entity fields except `id` are `Option<T>` because the API may omit them or return null
- **API key:** Read from `OPENALEX_KEY` env var. Required for `/find/works` (semantic search). Optional but recommended for other endpoints (higher rate limits)
- **`mag` fields are strings:** `WorkIds.mag`, `SourceIds.mag`, `InstitutionIds.mag` are returned as strings (e.g. `"2741809807"`), not integers. Use `Option<String>`, not `Option<i64>`
- **Null elements in arrays:** `host_organization_lineage` can contain null elements (e.g. `[null]`). Use `Option<Vec<Option<String>>>` instead of `Option<Vec<String>>`
