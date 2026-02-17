use schemars::JsonSchema;
use serde::Deserialize;

/// Parameters for list endpoints that don't have filter aliases.
/// Currently unused but kept for potential future use.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct ListToolParams {
    /// Filter expression. Comma-separated AND conditions, pipe (`|`) for OR.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
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
    pub select: Option<String>,
    /// Aggregate results by a field.
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

/// Parameters for the `work_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct WorkListToolParams {
    /// Filter expression. Comma-separated AND conditions, pipe (`|`) for OR.
    pub filter: Option<String>,
    /// Full-text search query. Searches title, abstract, and fulltext.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix. Example: `"cited_by_count:desc"`
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination (max page * per_page <= 10,000).
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination. Use `"*"` for the first page.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include in the response.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by author name or OpenAlex author ID (e.g. "einstein", "Albert Einstein", or "A5108093963")
    pub author: Option<String>,
    /// Filter by topic name or OpenAlex topic ID (e.g. "deep learning", or "T10320")
    pub topic: Option<String>,
    /// Filter by domain name or ID. The 4 domains are: 1 Life Sciences, 2 Social Sciences,
    /// 3 Physical Sciences, 4 Health Sciences (e.g. "physical sciences" or "3")
    pub domain: Option<String>,
    /// Filter by field name or ID (e.g. "computer science" or "17")
    pub field: Option<String>,
    /// Filter by subfield name or ID (e.g. "artificial intelligence" or "1702")
    pub subfield: Option<String>,
    /// Filter by publisher name or ID (e.g. "acm", "acm|ieee", or "P4310319798")
    pub publisher: Option<String>,
    /// Filter by source (journal/conference) name or ID (e.g. "siggraph" or "S131921510")
    pub source: Option<String>,
    /// Filter by institution name or ID. Uses lineage for broad matching (e.g. "mit" or "I136199984")
    pub institution: Option<String>,
    /// Filter by publication year (e.g. "2024", ">2008", "2008-2024")
    pub year: Option<String>,
    /// Filter by citation count (e.g. ">100", "10-50")
    pub citations: Option<String>,
    /// Filter by country code of author institutions (e.g. "US", "GB")
    pub country: Option<String>,
    /// Filter by continent of author institutions (e.g. "europe", "asia")
    pub continent: Option<String>,
    /// Filter by work type (e.g. "article", "preprint", "dataset")
    pub r#type: Option<String>,
    /// Filter for open access works only. Set to true to include only OA works.
    pub open: Option<bool>,
}

impl WorkListToolParams {
    pub fn into_work_list_params(&self) -> papers::WorkListParams {
        papers::WorkListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            author: self.author.clone(),
            topic: self.topic.clone(),
            domain: self.domain.clone(),
            field: self.field.clone(),
            subfield: self.subfield.clone(),
            publisher: self.publisher.clone(),
            source: self.source.clone(),
            institution: self.institution.clone(),
            year: self.year.clone(),
            citations: self.citations.clone(),
            country: self.country.clone(),
            continent: self.continent.clone(),
            r#type: self.r#type.clone(),
            open: self.open,
        }
    }
}

/// Parameters for the `author_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct AuthorListToolParams {
    /// Filter expression. Comma-separated AND conditions, pipe (`|`) for OR.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by institution name or ID (e.g. "harvard", "mit", or "I136199984")
    pub institution: Option<String>,
    /// Filter by country code of last known institution (e.g. "US", "GB")
    pub country: Option<String>,
    /// Filter by continent of last known institution (e.g. "europe", "asia")
    pub continent: Option<String>,
    /// Filter by citation count (e.g. ">1000", "100-500")
    pub citations: Option<String>,
    /// Filter by works count (e.g. ">500", "100-200")
    pub works: Option<String>,
    /// Filter by h-index (e.g. ">50", "10-20"). The h-index measures sustained
    /// research impact: an author with h-index *h* has *h* works each cited at
    /// least *h* times. Unlike raw citation count, it isn't inflated by a single
    /// highly-cited paper.
    pub h_index: Option<String>,
}

impl AuthorListToolParams {
    pub fn into_entity_params(&self) -> papers::AuthorListParams {
        papers::AuthorListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            institution: self.institution.clone(),
            country: self.country.clone(),
            continent: self.continent.clone(),
            citations: self.citations.clone(),
            works: self.works.clone(),
            h_index: self.h_index.clone(),
        }
    }
}

/// Parameters for the `source_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SourceListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by publisher name or ID (e.g. "springer", "P4310319798")
    pub publisher: Option<String>,
    /// Filter by country code (e.g. "US", "GB")
    pub country: Option<String>,
    /// Filter by continent (e.g. "europe")
    pub continent: Option<String>,
    /// Filter by source type (e.g. "journal", "repository", "conference")
    pub r#type: Option<String>,
    /// Filter for open access sources only.
    pub open: Option<bool>,
    /// Filter by citation count (e.g. ">10000")
    pub citations: Option<String>,
    /// Filter by works count (e.g. ">100000")
    pub works: Option<String>,
}

impl SourceListToolParams {
    pub fn into_entity_params(&self) -> papers::SourceListParams {
        papers::SourceListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            publisher: self.publisher.clone(),
            country: self.country.clone(),
            continent: self.continent.clone(),
            r#type: self.r#type.clone(),
            open: self.open,
            citations: self.citations.clone(),
            works: self.works.clone(),
        }
    }
}

/// Parameters for the `institution_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct InstitutionListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by country code (e.g. "US", "GB")
    pub country: Option<String>,
    /// Filter by continent (e.g. "europe", "asia")
    pub continent: Option<String>,
    /// Filter by institution type (e.g. "education", "healthcare", "company")
    pub r#type: Option<String>,
    /// Filter by citation count (e.g. ">100000")
    pub citations: Option<String>,
    /// Filter by works count (e.g. ">100000")
    pub works: Option<String>,
}

impl InstitutionListToolParams {
    pub fn into_entity_params(&self) -> papers::InstitutionListParams {
        papers::InstitutionListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            country: self.country.clone(),
            continent: self.continent.clone(),
            r#type: self.r#type.clone(),
            citations: self.citations.clone(),
            works: self.works.clone(),
        }
    }
}

/// Parameters for the `topic_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct TopicListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by domain name or ID (e.g. "life sciences", "3")
    pub domain: Option<String>,
    /// Filter by field name or ID (e.g. "computer science", "17")
    pub field: Option<String>,
    /// Filter by subfield name or ID (e.g. "artificial intelligence", "1702")
    pub subfield: Option<String>,
    /// Filter by citation count (e.g. ">1000")
    pub citations: Option<String>,
    /// Filter by works count (e.g. ">1000")
    pub works: Option<String>,
}

impl TopicListToolParams {
    pub fn into_entity_params(&self) -> papers::TopicListParams {
        papers::TopicListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            domain: self.domain.clone(),
            field: self.field.clone(),
            subfield: self.subfield.clone(),
            citations: self.citations.clone(),
            works: self.works.clone(),
        }
    }
}

/// Parameters for the `publisher_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct PublisherListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by country code (e.g. "US", "GB"). Note: uses `country_codes` (plural).
    pub country: Option<String>,
    /// Filter by continent (e.g. "europe")
    pub continent: Option<String>,
    /// Filter by citation count (e.g. ">10000")
    pub citations: Option<String>,
    /// Filter by works count (e.g. ">1000000")
    pub works: Option<String>,
}

impl PublisherListToolParams {
    pub fn into_entity_params(&self) -> papers::PublisherListParams {
        papers::PublisherListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            country: self.country.clone(),
            continent: self.continent.clone(),
            citations: self.citations.clone(),
            works: self.works.clone(),
        }
    }
}

/// Parameters for the `funder_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FunderListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by country code (e.g. "US", "GB")
    pub country: Option<String>,
    /// Filter by continent (e.g. "europe")
    pub continent: Option<String>,
    /// Filter by citation count (e.g. ">10000")
    pub citations: Option<String>,
    /// Filter by works count (e.g. ">100000")
    pub works: Option<String>,
}

impl FunderListToolParams {
    pub fn into_entity_params(&self) -> papers::FunderListParams {
        papers::FunderListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            country: self.country.clone(),
            continent: self.continent.clone(),
            citations: self.citations.clone(),
            works: self.works.clone(),
        }
    }
}

/// Parameters for the `domain_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct DomainListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by works count (e.g. ">100000000")
    pub works: Option<String>,
}

impl DomainListToolParams {
    pub fn into_entity_params(&self) -> papers::DomainListParams {
        papers::DomainListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            works: self.works.clone(),
        }
    }
}

/// Parameters for the `field_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct FieldListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by domain name or ID (e.g. "life sciences", "3")
    pub domain: Option<String>,
    /// Filter by works count (e.g. ">1000000")
    pub works: Option<String>,
}

impl FieldListToolParams {
    pub fn into_entity_params(&self) -> papers::FieldListParams {
        papers::FieldListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            domain: self.domain.clone(),
            works: self.works.clone(),
        }
    }
}

/// Parameters for the `subfield_list` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct SubfieldListToolParams {
    /// Filter expression.
    pub filter: Option<String>,
    /// Full-text search query.
    pub search: Option<String>,
    /// Sort field with optional `:desc` suffix.
    pub sort: Option<String>,
    /// Results per page (1-200, default 25).
    pub per_page: Option<u32>,
    /// Page number for offset pagination.
    pub page: Option<u32>,
    /// Cursor for cursor-based pagination.
    pub cursor: Option<String>,
    /// Return a random sample of this many results.
    pub sample: Option<u32>,
    /// Seed for reproducible random sampling.
    pub seed: Option<u32>,
    /// Comma-separated list of fields to include.
    pub select: Option<String>,
    /// Aggregate results by a field.
    pub group_by: Option<String>,
    /// Filter by domain name or ID (e.g. "physical sciences", "3")
    pub domain: Option<String>,
    /// Filter by field name or ID (e.g. "computer science", "17")
    pub field: Option<String>,
    /// Filter by works count (e.g. ">1000000")
    pub works: Option<String>,
}

impl SubfieldListToolParams {
    pub fn into_entity_params(&self) -> papers::SubfieldListParams {
        papers::SubfieldListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
            domain: self.domain.clone(),
            field: self.field.clone(),
            works: self.works.clone(),
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

    #[test]
    fn test_work_list_params_conversion_with_new_fields() {
        let tool_params: WorkListToolParams = serde_json::from_value(serde_json::json!({
            "institution": "mit",
            "country": "US",
            "continent": "north america",
            "type": "article",
            "open": true,
            "year": "2024"
        })).unwrap();
        let params = tool_params.into_work_list_params();
        assert_eq!(params.institution.as_deref(), Some("mit"));
        assert_eq!(params.country.as_deref(), Some("US"));
        assert_eq!(params.continent.as_deref(), Some("north america"));
        assert_eq!(params.r#type.as_deref(), Some("article"));
        assert_eq!(params.open, Some(true));
        assert_eq!(params.year.as_deref(), Some("2024"));
    }

    #[test]
    fn test_author_list_params_conversion() {
        let tool_params: AuthorListToolParams = serde_json::from_value(serde_json::json!({
            "institution": "harvard",
            "country": "US",
            "citations": ">1000",
            "h_index": ">50"
        })).unwrap();
        let params = tool_params.into_entity_params();
        assert_eq!(params.institution.as_deref(), Some("harvard"));
        assert_eq!(params.country.as_deref(), Some("US"));
        assert_eq!(params.citations.as_deref(), Some(">1000"));
        assert_eq!(params.h_index.as_deref(), Some(">50"));
    }

    #[test]
    fn test_source_list_params_conversion() {
        let tool_params: SourceListToolParams = serde_json::from_value(serde_json::json!({
            "publisher": "springer",
            "type": "journal",
            "open": true
        })).unwrap();
        let params = tool_params.into_entity_params();
        assert_eq!(params.publisher.as_deref(), Some("springer"));
        assert_eq!(params.r#type.as_deref(), Some("journal"));
        assert_eq!(params.open, Some(true));
    }

    #[test]
    fn test_topic_list_params_conversion() {
        let tool_params: TopicListToolParams = serde_json::from_value(serde_json::json!({
            "domain": "3",
            "field": "17"
        })).unwrap();
        let params = tool_params.into_entity_params();
        assert_eq!(params.domain.as_deref(), Some("3"));
        assert_eq!(params.field.as_deref(), Some("17"));
    }

    #[test]
    fn test_institution_list_params_conversion() {
        let tool_params: InstitutionListToolParams = serde_json::from_value(serde_json::json!({
            "country": "US",
            "type": "education",
            "works": ">100000"
        })).unwrap();
        let params = tool_params.into_entity_params();
        assert_eq!(params.country.as_deref(), Some("US"));
        assert_eq!(params.r#type.as_deref(), Some("education"));
        assert_eq!(params.works.as_deref(), Some(">100000"));
    }

    #[test]
    fn test_subfield_list_params_conversion() {
        let tool_params: SubfieldListToolParams = serde_json::from_value(serde_json::json!({
            "domain": "3",
            "field": "17",
            "works": ">1000000"
        })).unwrap();
        let params = tool_params.into_entity_params();
        assert_eq!(params.domain.as_deref(), Some("3"));
        assert_eq!(params.field.as_deref(), Some("17"));
        assert_eq!(params.works.as_deref(), Some(">1000000"));
    }
}
