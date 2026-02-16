use serde::Deserialize;

// ── List response ──────────────────────────────────────────────────────

/// Paginated list response returned by all 7 list endpoints.
///
/// Contains metadata about the query, a page of results, and optional group-by
/// aggregations.
///
/// # Example response shape
///
/// ```json
/// {
///   "meta": {"count": 288286684, "db_response_time_ms": 109, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null},
///   "results": [{"id": "https://openalex.org/W3038568908", "display_name": "..."}],
///   "group_by": []
/// }
/// ```
///
/// When cursor pagination is used, `meta.page` is `None` and `meta.next_cursor`
/// contains the cursor for the next page (or `None` if no more results).
///
/// When `group_by` is used, the `group_by` array is populated and
/// `meta.groups_count` is non-null.
#[derive(Debug, Clone, Deserialize)]
pub struct ListResponse<T> {
    /// Metadata about the query: total count, pagination state, timing.
    pub meta: ListMeta,

    /// The page of entity results.
    pub results: Vec<T>,

    /// Group-by aggregation results. Empty unless `group_by` was specified in
    /// the request.
    #[serde(default)]
    pub group_by: Vec<GroupByResult>,
}

/// Metadata returned with every list response.
///
/// ```json
/// {"count": 288286684, "db_response_time_ms": 109, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ListMeta {
    /// Total number of entities matching the query (before pagination).
    pub count: i64,

    /// Server-side query execution time in milliseconds.
    pub db_response_time_ms: i64,

    /// Current page number. `None` when cursor pagination is used.
    pub page: Option<i32>,

    /// Number of results per page.
    pub per_page: Option<i32>,

    /// Cursor for the next page of results. `None` when there are no more
    /// results or when offset pagination is used. Pass this value as
    /// [`ListParams::cursor`](crate::ListParams::cursor) to fetch the next
    /// page.
    pub next_cursor: Option<String>,

    /// Number of distinct groups when `group_by` is used. `None` otherwise.
    pub groups_count: Option<i64>,
}

/// A single group in a `group_by` aggregation result.
///
/// ```json
/// {"key": "https://openalex.org/types/article", "key_display_name": "article", "count": 209055572}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct GroupByResult {
    /// The raw key value for this group (often an OpenAlex URI).
    pub key: String,

    /// Human-readable display name for this group.
    pub key_display_name: String,

    /// Number of entities in this group.
    pub count: i64,
}

// ── Autocomplete response ──────────────────────────────────────────────

/// Response from any of the 7 autocomplete endpoints. Returns up to 10 results
/// sorted by citation count.
///
/// Fast type-ahead search (~200ms). Each result includes an entity ID, display
/// name, contextual hint (e.g. institution name for authors, host organization
/// for sources), and a `filter_key` for use in subsequent list queries.
///
/// # Example
///
/// ```json
/// {
///   "meta": {"count": 955, "db_response_time_ms": 30, "page": 1, "per_page": 10},
///   "results": [{
///     "id": "https://openalex.org/A5024159082",
///     "short_id": "authors/A5024159082",
///     "display_name": "Einstein",
///     "hint": "Helios Hospital Berlin-Buch, Germany",
///     "cited_by_count": 1,
///     "works_count": 2,
///     "entity_type": "author",
///     "external_id": null,
///     "filter_key": "authorships.author.id"
///   }]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AutocompleteResponse {
    /// Metadata about the autocomplete query.
    pub meta: AutocompleteMeta,

    /// Up to 10 autocomplete results, sorted by citation count.
    pub results: Vec<AutocompleteResult>,
}

/// Metadata for an autocomplete response.
#[derive(Debug, Clone, Deserialize)]
pub struct AutocompleteMeta {
    /// Total number of entities matching the query prefix.
    pub count: i64,

    /// Server-side query execution time in milliseconds.
    pub db_response_time_ms: i64,

    /// Always 1 (autocomplete does not support pagination).
    pub page: i32,

    /// Always 10 (autocomplete returns at most 10 results).
    pub per_page: i32,
}

/// A single autocomplete result.
///
/// The `hint` field contains contextual information that varies by entity type:
/// - **works:** first author name
/// - **authors:** last known institution and country
/// - **sources:** host organization
/// - **institutions:** city and country
/// - **concepts:** hierarchy level
/// - **publishers:** country
/// - **funders:** country and description
#[derive(Debug, Clone, Deserialize)]
pub struct AutocompleteResult {
    /// Full OpenAlex URI (e.g. `"https://openalex.org/A5024159082"`).
    pub id: String,

    /// Short ID path (e.g. `"authors/A5024159082"`).
    pub short_id: Option<String>,

    /// Human-readable entity name.
    pub display_name: String,

    /// Contextual hint whose meaning varies by entity type (see struct docs).
    pub hint: Option<String>,

    /// Total citation count for this entity.
    pub cited_by_count: Option<i64>,

    /// Total works count for this entity.
    pub works_count: Option<i64>,

    /// Entity type: one of `"work"`, `"author"`, `"source"`, `"institution"`,
    /// `"concept"`, `"publisher"`, `"funder"`.
    pub entity_type: String,

    /// External identifier (e.g. ISSN for sources, ROR for institutions,
    /// Wikidata for concepts). `None` if not available.
    pub external_id: Option<String>,

    /// The filter field name to use this result in subsequent list queries. For
    /// example, `"authorships.author.id"` for authors, or
    /// `"primary_location.source.id"` for sources.
    pub filter_key: Option<String>,
}

// ── Find works response ────────────────────────────────────────────────

/// Response from the `/find/works` semantic search endpoint.
///
/// Returns works ranked by AI similarity score (0-1). Requires an API key.
///
/// # Example
///
/// ```json
/// {
///   "results": [
///     {"score": 0.92, "id": "https://openalex.org/W...", "display_name": "Machine Learning for Drug Discovery", ...},
///     {"score": 0.87, "id": "https://openalex.org/W...", "display_name": "Deep Learning in Drug Design", ...}
///   ]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct FindWorksResponse {
    /// Optional metadata (structure varies).
    pub meta: Option<serde_json::Value>,

    /// Works ranked by similarity score. Each result contains a `score` field
    /// and the remaining Work fields flattened into `work`.
    pub results: Vec<FindWorksResult>,
}

/// A single result from semantic search, containing a similarity score and the
/// work data.
#[derive(Debug, Clone, Deserialize)]
pub struct FindWorksResult {
    /// AI similarity score between 0.0 and 1.0, where 1.0 is most similar to
    /// the query.
    pub score: f64,

    /// The work entity data as a JSON value. Contains the same fields as
    /// [`Work`](crate::Work) but flattened alongside `score`.
    #[serde(flatten)]
    pub work: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_list_response() {
        let json = r#"{
            "meta": {"count": 288286684, "db_response_time_ms": 109, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null},
            "results": [{"id": "https://openalex.org/W3038568908", "display_name": "Test Work"}],
            "group_by": []
        }"#;
        let resp: ListResponse<serde_json::Value> =
            serde_json::from_str(json).expect("Failed to deserialize ListResponse");
        assert_eq!(resp.meta.count, 288286684);
        assert_eq!(resp.results.len(), 1);
        assert!(resp.group_by.is_empty());
    }

    #[test]
    fn test_deserialize_list_with_cursor() {
        let json = r#"{
            "meta": {"count": 288286684, "db_response_time_ms": 125, "page": null, "per_page": 1, "next_cursor": "IlsxMDAuMC...", "groups_count": null},
            "results": [{"id": "test"}],
            "group_by": []
        }"#;
        let resp: ListResponse<serde_json::Value> =
            serde_json::from_str(json).expect("Failed to deserialize cursor response");
        assert!(resp.meta.page.is_none());
        assert_eq!(
            resp.meta.next_cursor.as_deref(),
            Some("IlsxMDAuMC...")
        );
    }

    #[test]
    fn test_deserialize_list_with_group_by() {
        let json = r#"{
            "meta": {"count": 288286684, "db_response_time_ms": 85, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": 1},
            "results": [],
            "group_by": [{"key": "https://openalex.org/types/article", "key_display_name": "article", "count": 209055572}]
        }"#;
        let resp: ListResponse<serde_json::Value> =
            serde_json::from_str(json).expect("Failed to deserialize group_by response");
        assert_eq!(resp.meta.groups_count, Some(1));
        assert_eq!(resp.group_by.len(), 1);
        assert_eq!(resp.group_by[0].key_display_name, "article");
        assert_eq!(resp.group_by[0].count, 209055572);
    }

    #[test]
    fn test_deserialize_autocomplete_response() {
        let json = r#"{
            "meta": {"count": 955, "db_response_time_ms": 30, "page": 1, "per_page": 10},
            "results": [{
                "id": "https://openalex.org/A5024159082",
                "short_id": "authors/A5024159082",
                "display_name": "Einstein",
                "hint": "Helios Hospital Berlin-Buch, Germany",
                "cited_by_count": 1,
                "works_count": 2,
                "entity_type": "author",
                "external_id": null,
                "filter_key": "authorships.author.id"
            }]
        }"#;
        let resp: AutocompleteResponse =
            serde_json::from_str(json).expect("Failed to deserialize AutocompleteResponse");
        assert_eq!(resp.meta.count, 955);
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].display_name, "Einstein");
        assert_eq!(resp.results[0].entity_type, "author");
        assert_eq!(
            resp.results[0].filter_key.as_deref(),
            Some("authorships.author.id")
        );
        assert_eq!(
            resp.results[0].short_id.as_deref(),
            Some("authors/A5024159082")
        );
    }
}
