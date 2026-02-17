# MCP Response Differences vs OpenAlex API

This file documents every place where the MCP server intentionally differs
from the raw OpenAlex REST API — whether in tool naming, response shape, or
returned fields.

The guiding principles:
- Tool names follow `{entity}_{verb}` (e.g. `work_list`, `work_get`) rather than the API's URL structure.
- List tools return slim summary structs to reduce context window consumption for LLM callers.
- Single-entity get tools pass the full API response through unchanged.

---

## Tool naming convention

**Implemented in:** `src/server.rs` — method names

MCP tool names follow `{entity_singular}_{verb}` rather than the
`{verb}_{entity_plural}` pattern used by the raw OpenAlex API URL paths.

| MCP tool name | OpenAlex API path |
|---|---|
| `work_list` | `GET /works` |
| `author_list` | `GET /authors` |
| `source_list` | `GET /sources` |
| `institution_list` | `GET /institutions` |
| `topic_list` | `GET /topics` |
| `publisher_list` | `GET /publishers` |
| `funder_list` | `GET /funders` |
| `domain_list` | `GET /domains` |
| `field_list` | `GET /fields` |
| `subfield_list` | `GET /subfields` |
| `work_get` | `GET /works/{id}` |
| `author_get` | `GET /authors/{id}` |
| `source_get` | `GET /sources/{id}` |
| `institution_get` | `GET /institutions/{id}` |
| `topic_get` | `GET /topics/{id}` |
| `publisher_get` | `GET /publishers/{id}` |
| `funder_get` | `GET /funders/{id}` |
| `domain_get` | `GET /domains/{id}` |
| `field_get` | `GET /fields/{id}` |
| `subfield_get` | `GET /subfields/{id}` |
| `work_autocomplete` | `GET /autocomplete/works` |
| `author_autocomplete` | `GET /autocomplete/authors` |
| `source_autocomplete` | `GET /autocomplete/sources` |
| `institution_autocomplete` | `GET /autocomplete/institutions` |
| `concept_autocomplete` | `GET /autocomplete/concepts` |
| `publisher_autocomplete` | `GET /autocomplete/publishers` |
| `funder_autocomplete` | `GET /autocomplete/funders` |
| `subfield_autocomplete` | `GET /autocomplete/subfields` |
| `work_find` | `GET /find/works` (or `POST` for long queries) |

**Reason:** Grouping by entity first makes the tool list sort and scan naturally
by subject — all `work_*` tools appear together, all `author_*` tools together,
etc. — rather than by verb, which clusters unrelated entities.

---

## List endpoints — slim summary structs

**Implemented in:** `src/summary.rs`
**Applied in:** `src/server.rs` — all 10 `*_list` tools use `summary_list_result()`

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

### `list_domains` → `DomainSummary`

**Reason:** Full `Domain` includes `ids`, `display_name_alternatives`, `siblings`
(other 3 domains), `works_api_url`, `updated_date`, and `created_date`. These
are not needed for discovery. The `fields` list is flattened to display_name
strings for quick scanning.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `description` | `ids`, `display_name_alternatives` |
| `fields` (display_name strings only) | `siblings` |
| `works_count`, `cited_by_count` | `works_api_url`, `updated_date`, `created_date` |

---

### `list_fields` → `FieldSummary`

**Reason:** Full `Field` includes `ids`, `display_name_alternatives`, `subfields`
(full list of child entities), `siblings` (all 25 other fields), `works_api_url`,
`updated_date`, and `created_date`. The subfields list is summarized as a count.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `description` | `ids`, `display_name_alternatives` |
| `domain` (display_name only) | `subfields` (replaced by `subfield_count`) |
| `subfield_count` (number of child subfields) | `siblings` |
| `works_count`, `cited_by_count` | `works_api_url`, `updated_date`, `created_date` |

---

### `list_subfields` → `SubfieldSummary`

**Reason:** Full `Subfield` includes `ids`, `display_name_alternatives`, `topics`
(can be 50+ child topics), `siblings` (all subfields in the same field),
`works_api_url`, `updated_date`, and `created_date`. The topics list is dropped
entirely since it can be very large.

| Kept | Dropped |
|------|---------|
| `id`, `display_name`, `description` | `ids`, `display_name_alternatives` |
| `field`, `domain` (display_names only, hierarchy flattened) | `topics` |
| `works_count`, `cited_by_count` | `siblings` |
| | `works_api_url`, `updated_date`, `created_date` |

---

## Get tools — no response changes

All 10 `*_get` tools (`work_get`, `author_get`, `source_get`, `institution_get`,
`topic_get`, `publisher_get`, `funder_get`, `domain_get`, `field_get`,
`subfield_get`) return the full deserialized API response. Use these when full
entity data is needed after identifying items via a `*_list` tool.

## Autocomplete tools — no response changes

All 8 `*_autocomplete` tools return the full `AutocompleteResponse`. These
already return compact 10-result lists, so no slimming is needed.

Note: only subfields support autocomplete among the hierarchy entities.
Domains and fields do not have autocomplete endpoints (the API returns 404).

## `work_find` — no response changes

Returns the full `FindWorksResponse` including similarity scores.

---

## Filter aliases — all list endpoints

**Implemented in:** `src/filter.rs`
**Applied in:** `papers-mcp/src/server.rs` (all `*_list` tools), `papers-cli/src/main.rs` (all `* list` commands)

All 10 list endpoints accept shorthand filter aliases that resolve to real
OpenAlex filter expressions. There are three alias kinds:

- **Entity** — accepts an OpenAlex entity ID or a search string (resolved via
  the API to the top result by citation count). One extra API call per alias.
- **Direct** — passes the value through as-is to the filter key.
- **Boolean** — when set to true, emits `filter_key:true`.

**Reason:** Raw OpenAlex filter keys are long and require knowing entity IDs
upfront. Aliases let callers use natural names (e.g. `--publisher acm` instead
of `--filter "primary_location.source.publisher_lineage:P4310319798"`).

If an alias conflicts with a key already present in the raw `filter` param,
an error is returned rather than silently overwriting.

### `work_list` / `work list` — 14 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `author` | `authorships.author.id` | Entity |
| `topic` | `primary_topic.id` | Entity |
| `domain` | `primary_topic.domain.id` | Entity |
| `field` | `primary_topic.field.id` | Entity |
| `subfield` | `primary_topic.subfield.id` | Entity |
| `publisher` | `primary_location.source.publisher_lineage` | Entity |
| `source` | `primary_location.source.id` | Entity |
| `institution` | `authorships.institutions.lineage` | Entity |
| `year` | `publication_year` | Direct |
| `citations` | `cited_by_count` | Direct |
| `country` | `authorships.institutions.country_code` | Direct |
| `continent` | `authorships.institutions.continent` | Direct |
| `type` | `type` | Direct |
| `open` | `is_oa` | Boolean |

### `author_list` / `author list` — 6 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `institution` | `last_known_institutions.id` | Entity |
| `country` | `last_known_institutions.country_code` | Direct |
| `continent` | `last_known_institutions.continent` | Direct |
| `citations` | `cited_by_count` | Direct |
| `works` | `works_count` | Direct |
| `h_index` | `summary_stats.h_index` | Direct |

### `source_list` / `source list` — 7 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `publisher` | `host_organization_lineage` | Entity |
| `country` | `country_code` | Direct |
| `continent` | `continent` | Direct |
| `type` | `type` | Direct |
| `open` | `is_oa` | Boolean |
| `citations` | `cited_by_count` | Direct |
| `works` | `works_count` | Direct |

### `institution_list` / `institution list` — 5 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `country` | `country_code` | Direct |
| `continent` | `continent` | Direct |
| `type` | `type` | Direct |
| `citations` | `cited_by_count` | Direct |
| `works` | `works_count` | Direct |

### `topic_list` / `topic list` — 5 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `domain` | `domain.id` | Entity |
| `field` | `field.id` | Entity |
| `subfield` | `subfield.id` | Entity |
| `citations` | `cited_by_count` | Direct |
| `works` | `works_count` | Direct |

### `publisher_list` / `publisher list` — 4 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `country` | `country_codes` | Direct |
| `continent` | `continent` | Direct |
| `citations` | `cited_by_count` | Direct |
| `works` | `works_count` | Direct |

### `funder_list` / `funder list` — 4 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `country` | `country_code` | Direct |
| `continent` | `continent` | Direct |
| `citations` | `cited_by_count` | Direct |
| `works` | `works_count` | Direct |

### `domain_list` / `domain list` — 1 alias

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `works` | `works_count` | Direct |

### `field_list` / `field list` — 2 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `domain` | `domain.id` | Entity |
| `works` | `works_count` | Direct |

### `subfield_list` / `subfield list` — 3 aliases

| Alias | OpenAlex filter key | Kind |
|---|---|---|
| `domain` | `domain.id` | Entity |
| `field` | `field.id` | Entity |
| `works` | `works_count` | Direct |

---

## How to update this file

When you intentionally change what the MCP returns relative to the raw API:

1. Add or update the relevant section above
2. State clearly: what was changed, what was dropped/added, and why
3. Update `src/summary.rs` for list-endpoint changes
4. Update tests in `tests/tools.rs` to assert the new shape
