use papers::OpenAlexClient;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use serde::Serialize;

use crate::params::{AutocompleteToolParams, FindWorksToolParams, GetToolParams, ListToolParams};

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
        let client = OpenAlexClient::new();
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

fn json_result<T: Serialize>(result: Result<T, papers::OpenAlexError>) -> Result<String, String> {
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
    #[tool]
    pub async fn work_list(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        json_result(papers::api::work_list(&self.client, &params.into_list_params()).await)
    }

    /// Search, filter, and paginate author profiles. 110M+ records.
    #[tool]
    pub async fn author_list(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        json_result(papers::api::author_list(&self.client, &params.into_list_params()).await)
    }

    /// Search, filter, and paginate publishing venues (journals, repositories, conferences).
    #[tool]
    pub async fn source_list(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        json_result(papers::api::source_list(&self.client, &params.into_list_params()).await)
    }

    /// Search, filter, and paginate research institutions and organizations.
    #[tool]
    pub async fn institution_list(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        json_result(papers::api::institution_list(&self.client, &params.into_list_params()).await)
    }

    /// Search, filter, and paginate research topics (3-level hierarchy: domain > field > subfield > topic).
    #[tool]
    pub async fn topic_list(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        json_result(papers::api::topic_list(&self.client, &params.into_list_params()).await)
    }

    /// Search, filter, and paginate publishing organizations (e.g. Elsevier, Springer Nature).
    #[tool]
    pub async fn publisher_list(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        json_result(papers::api::publisher_list(&self.client, &params.into_list_params()).await)
    }

    /// Search, filter, and paginate funding organizations (e.g. NIH, NSF, ERC).
    #[tool]
    pub async fn funder_list(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        json_result(papers::api::funder_list(&self.client, &params.into_list_params()).await)
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
