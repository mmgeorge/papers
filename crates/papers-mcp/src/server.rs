use papers::{DiskCache, OpenAlexClient};
use std::time::Duration;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use serde::Serialize;

use crate::params::{
    AutocompleteToolParams, AuthorListToolParams, DomainListToolParams, FieldListToolParams,
    FindWorksToolParams, FunderListToolParams, GetToolParams, InstitutionListToolParams,
    PublisherListToolParams, SourceListToolParams, SubfieldListToolParams, TopicListToolParams,
    WorkListToolParams,
};

#[derive(Clone)]
pub struct PapersMcp {
    client: OpenAlexClient,
    tool_router: ToolRouter<Self>,
}

impl Default for PapersMcp {
    fn default() -> Self {
        Self::new()
    }
}

impl PapersMcp {
    pub fn new() -> Self {
        let mut client = OpenAlexClient::new();
        if let Ok(cache) = DiskCache::default_location(Duration::from_secs(600)) {
            client = client.with_cache(cache);
        }
        Self {
            client,
            tool_router: Self::tool_router(),
        }
    }

    pub fn with_client(client: OpenAlexClient) -> Self {
        Self {
            client,
            tool_router: Self::tool_router(),
        }
    }
}

fn json_result<T: Serialize, E: std::fmt::Display>(result: Result<T, E>) -> Result<String, String> {
    match result {
        Ok(response) => {
            serde_json::to_string_pretty(&response).map_err(|e| format!("JSON serialization error: {e}"))
        }
        Err(e) => Err(e.to_string()),
    }
}

#[tool_router(vis = "pub")]
impl PapersMcp {
    // ── List tools ───────────────────────────────────────────────────────

    /// Search, filter, and paginate scholarly works (articles, preprints, datasets, etc.). 240M+ records.
    /// Accepts shorthand filter aliases (author, topic, year, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/works/filter-works
    #[tool]
    pub async fn work_list(&self, Parameters(params): Parameters<WorkListToolParams>) -> Result<String, String> {
        json_result(papers::api::work_list(&self.client, &params.into_work_list_params()).await)
    }

    /// Search, filter, and paginate author profiles. 110M+ records.
    /// Accepts shorthand filter aliases (institution, country, citations, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/authors/filter-authors
    #[tool]
    pub async fn author_list(&self, Parameters(params): Parameters<AuthorListToolParams>) -> Result<String, String> {
        json_result(papers::api::author_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate publishing venues (journals, repositories, conferences).
    /// Accepts shorthand filter aliases (publisher, country, type, open, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/sources/filter-sources
    #[tool]
    pub async fn source_list(&self, Parameters(params): Parameters<SourceListToolParams>) -> Result<String, String> {
        json_result(papers::api::source_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research institutions and organizations.
    /// Accepts shorthand filter aliases (country, continent, type, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/institutions/filter-institutions
    #[tool]
    pub async fn institution_list(&self, Parameters(params): Parameters<InstitutionListToolParams>) -> Result<String, String> {
        json_result(papers::api::institution_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research topics (3-level hierarchy: domain > field > subfield > topic).
    /// Accepts shorthand filter aliases (domain, field, subfield, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/topics/filter-topics
    #[tool]
    pub async fn topic_list(&self, Parameters(params): Parameters<TopicListToolParams>) -> Result<String, String> {
        json_result(papers::api::topic_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate publishing organizations (e.g. Elsevier, Springer Nature).
    /// Accepts shorthand filter aliases (country, continent, citations, works) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/publishers/filter-publishers
    #[tool]
    pub async fn publisher_list(&self, Parameters(params): Parameters<PublisherListToolParams>) -> Result<String, String> {
        json_result(papers::api::publisher_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate funding organizations (e.g. NIH, NSF, ERC).
    /// Accepts shorthand filter aliases (country, continent, citations, works) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/funders/filter-funders
    #[tool]
    pub async fn funder_list(&self, Parameters(params): Parameters<FunderListToolParams>) -> Result<String, String> {
        json_result(papers::api::funder_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research domains (broadest level of topic hierarchy). 4 domains total.
    /// Accepts shorthand filter alias (works) that resolves to OpenAlex filter expression.
    /// Filtering: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists
    #[tool]
    pub async fn domain_list(&self, Parameters(params): Parameters<DomainListToolParams>) -> Result<String, String> {
        json_result(papers::api::domain_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate academic fields (second level of topic hierarchy). 26 fields total.
    /// Accepts shorthand filter aliases (domain, works) that resolve to OpenAlex filter expressions.
    /// Filtering: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists
    #[tool]
    pub async fn field_list(&self, Parameters(params): Parameters<FieldListToolParams>) -> Result<String, String> {
        json_result(papers::api::field_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research subfields (third level of topic hierarchy). ~252 subfields total.
    /// Accepts shorthand filter aliases (domain, field, works) that resolve to OpenAlex filter expressions.
    /// Filtering: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists
    #[tool]
    pub async fn subfield_list(&self, Parameters(params): Parameters<SubfieldListToolParams>) -> Result<String, String> {
        json_result(papers::api::subfield_list(&self.client, &params.into_entity_params()).await)
    }

    // ── Get tools ────────────────────────────────────────────────────────

    /// Get a single work by ID (OpenAlex ID, DOI, PMID, or PMCID).
    #[tool]
    pub async fn work_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::work_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single author by ID (OpenAlex ID or ORCID).
    #[tool]
    pub async fn author_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::author_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single source by ID (OpenAlex ID or ISSN).
    #[tool]
    pub async fn source_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::source_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single institution by ID (OpenAlex ID or ROR).
    #[tool]
    pub async fn institution_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::institution_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single topic by OpenAlex ID.
    #[tool]
    pub async fn topic_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::topic_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single publisher by OpenAlex ID.
    #[tool]
    pub async fn publisher_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::publisher_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single funder by OpenAlex ID.
    #[tool]
    pub async fn funder_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::funder_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single domain by numeric ID (1-4).
    #[tool]
    pub async fn domain_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::domain_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single academic field by numeric ID (e.g. 17 for Computer Science).
    #[tool]
    pub async fn field_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::field_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single research subfield by numeric ID (e.g. 1702 for Artificial Intelligence).
    #[tool]
    pub async fn subfield_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers::api::subfield_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    // ── Autocomplete tools ───────────────────────────────────────────────

    /// Type-ahead search for works by title. Returns up to 10 results sorted by citation count.
    #[tool]
    pub async fn work_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::work_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for authors. Returns up to 10 results sorted by citation count.
    #[tool]
    pub async fn author_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::author_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for sources (journals, repositories). Returns up to 10 results.
    #[tool]
    pub async fn source_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::source_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for institutions. Returns up to 10 results.
    #[tool]
    pub async fn institution_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::institution_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for concepts (deprecated but functional). Returns up to 10 results.
    #[tool]
    pub async fn concept_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::concept_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for publishers. Returns up to 10 results.
    #[tool]
    pub async fn publisher_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::publisher_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for funders. Returns up to 10 results.
    #[tool]
    pub async fn funder_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::funder_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for subfields. Returns up to 10 results.
    #[tool]
    pub async fn subfield_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers::api::subfield_autocomplete(&self.client, &params.q).await)
    }

    // ── Semantic search ──────────────────────────────────────────────────

    /// AI semantic search for works by conceptual similarity. Requires API key. Uses POST for queries > 2048 chars.
    #[tool]
    pub async fn work_find(&self, Parameters(params): Parameters<FindWorksToolParams>) -> Result<String, String> {
        json_result(papers::api::work_find(&self.client, &params.into_find_params()).await)
    }
}

#[tool_handler]
impl ServerHandler for PapersMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: rmcp::model::Implementation {
                name: "papers-mcp".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: None,
                description: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "MCP server for querying the OpenAlex academic research database. \
                 Provides tools to search, filter, and retrieve scholarly works, \
                 authors, sources, institutions, topics, publishers, and funders."
                    .into(),
            ),
        }
    }
}
