# MCP Response Differences vs OpenAlex API

This file documents every place where the MCP server intentionally returns
**different or reduced data** compared to the raw OpenAlex REST API.

The guiding principle: list endpoints return slim summary structs to reduce
context window consumption for LLM callers. Single-entity (`get_*`) endpoints
pass the full API response through unchanged.

---

## List endpoints — slim summary structs

**Implemented in:** `src/summary.rs`
**Applied in:** `src/server.rs` — all 7 `list_*` tools use `summary_list_result()`

### Response shape change

The raw API returns:
```json
{ "meta": {...}, "results": [...full entities...], "group_by": [...] }
```

The MCP returns:
```json
{ "meta": {...}, "results": [...summary structs...] }
```

`group_by` is always dropped from list responses. It is always empty unless
the caller uses the `group_by` parameter, in which case `results` is empty
anyway — so dropping it causes no loss for typical usage.

---

### `list_works` → `WorkSummary`

**Reason:** Full `Work` objects include `referenced_works` (100+ IDs),
`counts_by_year` (20+ entries), `locations` (multiple objects), `mesh`,
`concepts`, `keywords`, `sdgs`, `awards`, and many rarely-needed fields.
These can easily consume 10–50 KB per result page.

| Kept | Dropped |
|------|---------|
| `id`, `title` (from `display_name`), `doi`, `publication_year`, `type` | `referenced_works`, `related_works` |
| `authors` (display_name strings only, from `authorships`) | `locations`, `best_oa_location`, `locations_count` |
| `journal` (from `primary_location.source.display_name`) | `counts_by_year`, `biblio` |
| `is_oa`, `oa_url` (from `open_access`) | `concepts`, `keywords`, `mesh`, `sustainable_development_goals` |
| `cited_by_count`, `primary_topic` (display_name only) | `topics`, `funders`, `awards`, `ids` |
| `abstract_text` | `fwci`, `citation_normalized_percentile`, `cited_by_percentile_year` |
| | `apc_list`, `apc_paid`, `has_fulltext`, `has_content`, `content_urls` |
| | `is_retracted`, `is_paratext`, `is_xpac`, `indexed_in`, `language` |
| | `type_crossref`, `corresponding_author_ids`, `countries_distinct_count`, etc. |

`abstract_text` is kept because it is critical for relevance judgement by
LLM callers.

---

### `list_authors` → `AuthorSummary`

**Reason:** Full `Author` includes `affiliations` (full institutional history
with years), `counts_by_year`, `x_concepts` (deprecated), and `topic_share`.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `orcid` | `affiliations` (full history) |
| `works_count`, `cited_by_count` | `topic_share`, `x_concepts` |
| `h_index` (from `summary_stats.h_index`) | `counts_by_year`, `ids`, `works_api_url` |
| `last_known_institutions` (display_name strings only) | `display_name_alternatives` |
| `top_topics` (first 3 display_names from `topics`) | remaining `summary_stats` fields |

Only the first 3 topics are kept (same as the number a work can have), since
beyond that the marginal value drops sharply.

---

### `list_sources` → `SourceSummary`

**Reason:** Full `Source` includes `apc_prices` (multi-currency price lists),
`topics` (25 entries with full hierarchy), `topic_share`, `counts_by_year`,
`societies`, `lineage`, and many boolean flags.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `issn_l`, `type` | `issn` (all), `alternate_titles`, `abbreviated_title` |
| `is_oa`, `is_in_doaj` | `apc_prices`, `apc_usd` |
| `works_count`, `cited_by_count` | `topics`, `topic_share` |
| `h_index` (from `summary_stats.h_index`) | `counts_by_year`, `ids`, `works_api_url` |
| `host_organization_name` | `societies`, `lineage`, `host_organization_lineage` |
| | `is_core`, `is_high_oa_rate`, `is_in_scielo`, `is_ojs`, `oa_flip_year` |
| | `first_publication_year`, `last_publication_year`, `homepage_url`, `country_code` |

---

### `list_institutions` → `InstitutionSummary`

**Reason:** Full `Institution` includes `associated_institutions` (can be
50+ related organizations), `topics` (25 entries), `topic_share` (25 entries),
`counts_by_year`, `repositories`, `international` (translations), and lineage.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `ror`, `country_code`, `type` | `associated_institutions` |
| `city` (from `geo.city`) | `topics`, `topic_share` |
| `works_count`, `cited_by_count` | `counts_by_year`, `ids`, `works_api_url` |
| `h_index` (from `summary_stats.h_index`) | `repositories`, `international`, `lineage` |
| | `homepage_url`, `image_url`, `image_thumbnail_url` |
| | `display_name_acronyms`, `display_name_alternatives`, `roles`, `type_id` |

---

### `list_topics` → `TopicSummary`

**Reason:** Full `Topic` includes `keywords` (10+ terms, redundant given
`description`), `siblings` (all topics at the same level — can be 50+), and
`ids`.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `description` | `keywords` |
| `subfield`, `field`, `domain` (display_names only, hierarchy flattened) | `siblings` |
| `works_count`, `cited_by_count` | `ids`, `works_api_url`, `counts_by_year` |

The nested `TopicHierarchyLevel` objects are flattened to plain display_name
strings to reduce nesting depth.

---

### `list_publishers` → `PublisherSummary`

**Reason:** Full `Publisher` includes `lineage` (parent chain, can be
objects or IDs), `alternate_titles`, `counts_by_year`, `roles`, `summary_stats`,
`image_url`, and `parent_publisher`.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `hierarchy_level`, `country_codes` | `parent_publisher`, `lineage` |
| `works_count`, `cited_by_count` | `alternate_titles`, `counts_by_year` |
| | `summary_stats`, `ids`, `roles`, `sources_api_url` |
| | `homepage_url`, `image_url`, `image_thumbnail_url` |

---

### `list_funders` → `FunderSummary`

**Reason:** Full `Funder` includes `alternate_titles`, `counts_by_year`,
`roles`, `summary_stats`, `ids`, `image_url`, and `homepage_url`.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `country_code`, `description` | `alternate_titles` |
| `awards_count`, `works_count`, `cited_by_count` | `counts_by_year`, `roles` |
| | `summary_stats`, `ids`, `homepage_url`, `image_url`, `image_thumbnail_url` |

---

## Get endpoints — no changes

All 7 `get_*` tools (`get_work`, `get_author`, `get_source`, `get_institution`,
`get_topic`, `get_publisher`, `get_funder`) return the full deserialized API
response. Use these when full entity data is needed after identifying items
via a list tool.

## Autocomplete endpoints — no changes

All 7 `autocomplete_*` tools return the full `AutocompleteResponse`. These
already return compact 10-result lists, so no slimming is needed.

## `find_works` — no changes

Returns the full `FindWorksResponse` including similarity scores.

---

## How to update this file

When you intentionally change what the MCP returns relative to the raw API:

1. Add or update the relevant section above
2. State clearly: what was changed, what was dropped/added, and why
3. Update `src/summary.rs` for list-endpoint changes
4. Update tests in `tests/tools.rs` to assert the new shape
