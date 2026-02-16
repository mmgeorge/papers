/// Query parameters shared by all 7 list endpoints (works, authors, sources,
/// institutions, topics, publishers, funders). All fields are optional.
///
/// Supports both struct-update syntax and the bon builder pattern:
///
/// ```
/// use openalex::ListParams;
///
/// // Struct-update syntax
/// let params = ListParams {
///     search: Some("machine learning".into()),
///     per_page: Some(10),
///     ..Default::default()
/// };
///
/// // Builder syntax
/// let params = ListParams::builder()
///     .search("machine learning")
///     .filter("publication_year:2024,is_oa:true")
///     .sort("cited_by_count:desc")
///     .per_page(10)
///     .page(1)
///     .build();
/// ```
///
/// # Pagination
///
/// Two pagination modes are available (mutually exclusive):
///
/// - **Offset pagination:** set `page` (max `page * per_page <= 10,000`)
/// - **Cursor pagination:** set `cursor` to `"*"` for the first page, then pass
///   [`ListMeta::next_cursor`](crate::ListMeta::next_cursor) from the previous
///   response. When `next_cursor` is `None`, there are no more results.
///
/// # Sampling
///
/// Set `sample` to get a random sample instead of paginated results. Use `seed`
/// for reproducibility.
///
/// # Grouping
///
/// Set `group_by` to aggregate results by a field. The response will include a
/// [`group_by`](crate::ListResponse::group_by) array with key, display name,
/// and count for each group.
#[derive(Debug, Default, Clone, bon::Builder)]
#[builder(on(String, into))]
pub struct ListParams {
    /// Filter expression. Comma-separated AND conditions, pipe (`|`) for OR
    /// (max 50 alternatives). Supports negation (`!`), comparison (`>`, `<`),
    /// and ranges (`2020-2023`).
    ///
    /// Example: `"publication_year:2024,is_oa:true"` or `"type:article|preprint"`
    pub filter: Option<String>,

    /// Full-text search query. For works, searches across title, abstract, and
    /// fulltext. For other entities, searches `display_name`. Supports stemming
    /// and stop-word removal.
    pub search: Option<String>,

    /// Sort field with optional direction suffix. Append `:desc` for descending
    /// order. Multiple fields can be comma-separated.
    ///
    /// Available fields: `display_name`, `cited_by_count`, `works_count`,
    /// `publication_date`, `relevance_score` (only with active search).
    ///
    /// Example: `"cited_by_count:desc"`
    pub sort: Option<String>,

    /// Results per page (1-200, default 25).
    ///
    /// Note: the API query key is `per-page` (hyphenated); this field handles
    /// the mapping automatically.
    pub per_page: Option<u32>,

    /// Page number for offset pagination. Maximum accessible:
    /// `page * per_page <= 10,000`. For deeper results, use cursor pagination.
    pub page: Option<u32>,

    /// Cursor for cursor-based pagination. Start with `"*"`, then pass
    /// `meta.next_cursor` from the previous response. When `next_cursor` is
    /// `None`, there are no more results. Mutually exclusive with `page`.
    pub cursor: Option<String>,

    /// Return a random sample of this many results instead of paginated results.
    /// Use with `seed` for reproducibility.
    pub sample: Option<u32>,

    /// Seed value for reproducible random sampling. Only meaningful when
    /// `sample` is set.
    pub seed: Option<u32>,

    /// Comma-separated list of fields to include in the response. Reduces
    /// payload size. Unselected fields will be omitted.
    ///
    /// Example: `"id,display_name,cited_by_count"`
    pub select: Option<String>,

    /// Aggregate results by a field and return counts. The response will include
    /// a `group_by` array with `key`, `key_display_name`, and `count` for each
    /// group.
    ///
    /// Example: `"type"` groups works by article/preprint/etc.
    pub group_by: Option<String>,
}

impl ListParams {
    pub(crate) fn to_query_pairs(&self) -> Vec<(&str, String)> {
        let mut pairs = Vec::new();
        if let Some(v) = &self.filter {
            pairs.push(("filter", v.clone()));
        }
        if let Some(v) = &self.search {
            pairs.push(("search", v.clone()));
        }
        if let Some(v) = &self.sort {
            pairs.push(("sort", v.clone()));
        }
        if let Some(v) = self.per_page {
            pairs.push(("per-page", v.to_string()));
        }
        if let Some(v) = self.page {
            pairs.push(("page", v.to_string()));
        }
        if let Some(v) = &self.cursor {
            pairs.push(("cursor", v.clone()));
        }
        if let Some(v) = self.sample {
            pairs.push(("sample", v.to_string()));
        }
        if let Some(v) = self.seed {
            pairs.push(("seed", v.to_string()));
        }
        if let Some(v) = &self.select {
            pairs.push(("select", v.clone()));
        }
        if let Some(v) = &self.group_by {
            pairs.push(("group_by", v.clone()));
        }
        pairs
    }
}

/// Query parameters for single-entity GET endpoints. Only field selection is
/// supported.
///
/// ```
/// use openalex::GetParams;
///
/// // No field selection (return full entity)
/// let params = GetParams::default();
///
/// // Select specific fields to reduce payload
/// let params = GetParams::builder()
///     .select("id,display_name,cited_by_count")
///     .build();
/// ```
#[derive(Debug, Default, Clone, bon::Builder)]
#[builder(on(String, into))]
pub struct GetParams {
    /// Comma-separated list of fields to include in the response.
    ///
    /// Example: `"id,display_name,cited_by_count"`
    pub select: Option<String>,
}

impl GetParams {
    pub(crate) fn to_query_pairs(&self) -> Vec<(&str, String)> {
        let mut pairs = Vec::new();
        if let Some(v) = &self.select {
            pairs.push(("select", v.clone()));
        }
        pairs
    }
}

/// Parameters for semantic search (`/find/works`). Uses AI embeddings to find
/// conceptually similar works. Requires an API key. Costs 1,000 credits per
/// request.
///
/// ```
/// use openalex::FindWorksParams;
///
/// let params = FindWorksParams::builder()
///     .query("machine learning for drug discovery")
///     .count(10)
///     .filter("publication_year:>2020")
///     .build();
/// ```
#[derive(Debug, Clone, bon::Builder)]
#[builder(on(String, into))]
pub struct FindWorksParams {
    /// Text to find similar works for. Can be a title, abstract, or research
    /// question. Maximum 10,000 characters. For POST requests, this is sent in
    /// the JSON body as `{"query": "..."}`.
    pub query: String,

    /// Number of results to return (1-100, default 25). Results are ranked by
    /// similarity score.
    pub count: Option<u32>,

    /// Filter expression to constrain results (same syntax as list endpoints).
    /// Applied after semantic ranking.
    pub filter: Option<String>,
}

impl FindWorksParams {
    pub(crate) fn to_query_pairs(&self) -> Vec<(&str, String)> {
        let mut pairs = Vec::new();
        pairs.push(("query", self.query.clone()));
        if let Some(v) = self.count {
            pairs.push(("count", v.to_string()));
        }
        if let Some(v) = &self.filter {
            pairs.push(("filter", v.clone()));
        }
        pairs
    }

    pub(crate) fn to_post_query_pairs(&self) -> Vec<(&str, String)> {
        let mut pairs = Vec::new();
        if let Some(v) = self.count {
            pairs.push(("count", v.to_string()));
        }
        if let Some(v) = &self.filter {
            pairs.push(("filter", v.clone()));
        }
        pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_params_default() {
        let params = ListParams::default();
        assert!(params.filter.is_none());
        assert!(params.search.is_none());
        assert!(params.sort.is_none());
        assert!(params.per_page.is_none());
        assert!(params.page.is_none());
        assert!(params.cursor.is_none());
        assert!(params.sample.is_none());
        assert!(params.seed.is_none());
        assert!(params.select.is_none());
        assert!(params.group_by.is_none());
    }

    #[test]
    fn test_list_params_builder() {
        let params = ListParams::builder()
            .search("machine learning")
            .per_page(10)
            .sort("cited_by_count:desc")
            .build();
        assert_eq!(params.search.as_deref(), Some("machine learning"));
        assert_eq!(params.per_page, Some(10));
        assert_eq!(params.sort.as_deref(), Some("cited_by_count:desc"));
    }

    #[test]
    fn test_list_params_struct_update() {
        let params = ListParams {
            search: Some("test".into()),
            per_page: Some(5),
            ..Default::default()
        };
        assert_eq!(params.search.as_deref(), Some("test"));
        assert_eq!(params.per_page, Some(5));
        assert!(params.filter.is_none());
    }

    #[test]
    fn test_get_params_default() {
        let params = GetParams::default();
        assert!(params.select.is_none());
    }

    #[test]
    fn test_find_works_params_builder() {
        let params = FindWorksParams::builder()
            .query("drug discovery")
            .count(10)
            .build();
        assert_eq!(params.query, "drug discovery");
        assert_eq!(params.count, Some(10));
        assert!(params.filter.is_none());
    }
}
