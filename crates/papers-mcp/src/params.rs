use schemars::JsonSchema;
use serde::Deserialize;

/// Parameters for list endpoints (works, authors, sources, institutions, topics,
/// publishers, funders).
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ListToolParams {
    /// Filter expression. Comma-separated AND conditions, pipe (`|`) for OR.
    /// Example: `"publication_year:2024,is_oa:true"`
    pub filter: Option<String>,

    /// Full-text search query. Searches title, abstract, and fulltext for works;
    /// display_name for other entities.
    pub search: Option<String>,

    /// Sort field with optional `:desc` suffix.
    /// Example: `"cited_by_count:desc"`
    pub sort: Option<String>,

    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,

    /// Page number for offset pagination (max page * per_page <= 10,000).
    pub page: Option<u32>,

    /// Cursor for cursor-based pagination. Use `"*"` for the first page, then
    /// pass `meta.next_cursor` from the previous response.
    pub cursor: Option<String>,

    /// Return a random sample of this many results.
    pub sample: Option<u32>,

    /// Seed for reproducible random sampling. Only meaningful with `sample`.
    pub seed: Option<u32>,

    /// Comma-separated list of fields to include in the response.
    /// Example: `"id,display_name,cited_by_count"`
    pub select: Option<String>,

    /// Aggregate results by a field.
    /// Example: `"type"` groups works by article/preprint/etc.
    pub group_by: Option<String>,
}

impl ListToolParams {
    pub fn into_list_params(self) -> papers::ListParams {
        papers::ListParams {
            filter: self.filter,
            search: self.search,
            sort: self.sort,
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor,
            sample: self.sample,
            seed: self.seed,
            select: self.select,
            group_by: self.group_by,
        }
    }
}

/// Parameters for single-entity GET endpoints.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct GetToolParams {
    /// Entity ID. Accepts OpenAlex IDs (e.g. `W2741809807`), DOIs, ORCIDs,
    /// ROR IDs, ISSNs, PMIDs, etc.
    pub id: String,

    /// Comma-separated list of fields to include in the response.
    pub select: Option<String>,
}

impl GetToolParams {
    pub fn into_get_params(&self) -> papers::GetParams {
        papers::GetParams {
            select: self.select.clone(),
        }
    }
}

/// Parameters for autocomplete endpoints.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AutocompleteToolParams {
    /// Search query for type-ahead matching.
    pub q: String,
}

/// Parameters for the find_works semantic search endpoint.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FindWorksToolParams {
    /// Text to find similar works for. Can be a title, abstract, or research
    /// question. Maximum 10,000 characters.
    pub query: String,

    /// Number of results to return (1-100, default 25).
    pub count: Option<u32>,

    /// Filter expression to constrain results (same syntax as list endpoints).
    pub filter: Option<String>,
}

impl FindWorksToolParams {
    pub fn into_find_params(self) -> papers::FindWorksParams {
        papers::FindWorksParams {
            query: self.query,
            count: self.count,
            filter: self.filter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_params_conversion() {
        let tool_params = ListToolParams {
            filter: Some("is_oa:true".into()),
            search: Some("machine learning".into()),
            sort: Some("cited_by_count:desc".into()),
            per_page: Some(10),
            page: Some(2),
            cursor: None,
            sample: None,
            seed: None,
            select: Some("id,display_name".into()),
            group_by: Some("type".into()),
        };
        let params = tool_params.into_list_params();
        assert_eq!(params.filter.as_deref(), Some("is_oa:true"));
        assert_eq!(params.search.as_deref(), Some("machine learning"));
        assert_eq!(params.sort.as_deref(), Some("cited_by_count:desc"));
        assert_eq!(params.per_page, Some(10));
        assert_eq!(params.page, Some(2));
        assert!(params.cursor.is_none());
        assert_eq!(params.select.as_deref(), Some("id,display_name"));
        assert_eq!(params.group_by.as_deref(), Some("type"));
    }

    #[test]
    fn test_get_params_conversion() {
        let tool_params = GetToolParams {
            id: "W2741809807".into(),
            select: Some("id,title".into()),
        };
        let params = tool_params.into_get_params();
        assert_eq!(params.select.as_deref(), Some("id,title"));
    }

    #[test]
    fn test_find_params_conversion() {
        let tool_params = FindWorksToolParams {
            query: "drug discovery".into(),
            count: Some(10),
            filter: Some("publication_year:>2020".into()),
        };
        let params = tool_params.into_find_params();
        assert_eq!(params.query, "drug discovery");
        assert_eq!(params.count, Some(10));
        assert_eq!(params.filter.as_deref(), Some("publication_year:>2020"));
    }

    #[test]
    fn test_default_values() {
        let json = r#"{}"#;
        let params: ListToolParams = serde_json::from_str(json).unwrap();
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
    fn test_list_params_schema() {
        let schema = schemars::schema_for!(ListToolParams);
        let json = serde_json::to_value(&schema).unwrap();
        assert_eq!(json["type"], "object");
        let props = json["properties"].as_object().unwrap();
        assert!(props.contains_key("filter"));
        assert!(props.contains_key("search"));
        assert!(props.contains_key("sort"));
        assert!(props.contains_key("per_page"));
    }
}
