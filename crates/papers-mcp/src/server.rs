use openalex::OpenAlexClient;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::{ServerHandler, tool, tool_handler, tool_router};
use serde::Serialize;

use crate::params::{AutocompleteToolParams, FindWorksToolParams, GetToolParams, ListToolParams};
use crate::summary::{
    AuthorSummary, FunderSummary, InstitutionSummary, PublisherSummary, SourceSummary,
    TopicSummary, WorkSummary, summary_list_result,
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

fn json_result<T: Serialize>(result: Result<T, openalex::OpenAlexError>) -> Result<String, String> {
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
    pub async fn list_works(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        summary_list_result(self.client.list_works(&params.into_list_params()).await, WorkSummary::from)
    }

    /// Search, filter, and paginate author profiles. 110M+ records.
    #[tool]
    pub async fn list_authors(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        summary_list_result(self.client.list_authors(&params.into_list_params()).await, AuthorSummary::from)
    }

    /// Search, filter, and paginate publishing venues (journals, repositories, conferences).
    #[tool]
    pub async fn list_sources(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        summary_list_result(self.client.list_sources(&params.into_list_params()).await, SourceSummary::from)
    }

    /// Search, filter, and paginate research institutions and organizations.
    #[tool]
    pub async fn list_institutions(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        summary_list_result(self.client.list_institutions(&params.into_list_params()).await, InstitutionSummary::from)
    }

    /// Search, filter, and paginate research topics (3-level hierarchy: domain > field > subfield > topic).
    #[tool]
    pub async fn list_topics(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        summary_list_result(self.client.list_topics(&params.into_list_params()).await, TopicSummary::from)
    }

    /// Search, filter, and paginate publishing organizations (e.g. Elsevier, Springer Nature).
    #[tool]
    pub async fn list_publishers(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        summary_list_result(self.client.list_publishers(&params.into_list_params()).await, PublisherSummary::from)
    }

    /// Search, filter, and paginate funding organizations (e.g. NIH, NSF, ERC).
    #[tool]
    pub async fn list_funders(&self, Parameters(params): Parameters<ListToolParams>) -> Result<String, String> {
        summary_list_result(self.client.list_funders(&params.into_list_params()).await, FunderSummary::from)
    }

    // ── Get tools ────────────────────────────────────────────────────────

    /// Get a single work by ID (OpenAlex ID, DOI, PMID, or PMCID).
    #[tool]
    pub async fn get_work(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(self.client.get_work(&params.id, &params.into_get_params()).await)
    }

    /// Get a single author by ID (OpenAlex ID or ORCID).
    #[tool]
    pub async fn get_author(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(self.client.get_author(&params.id, &params.into_get_params()).await)
    }

    /// Get a single source by ID (OpenAlex ID or ISSN).
    #[tool]
    pub async fn get_source(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(self.client.get_source(&params.id, &params.into_get_params()).await)
    }

    /// Get a single institution by ID (OpenAlex ID or ROR).
    #[tool]
    pub async fn get_institution(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(self.client.get_institution(&params.id, &params.into_get_params()).await)
    }

    /// Get a single topic by OpenAlex ID.
    #[tool]
    pub async fn get_topic(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(self.client.get_topic(&params.id, &params.into_get_params()).await)
    }

    /// Get a single publisher by OpenAlex ID.
    #[tool]
    pub async fn get_publisher(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(self.client.get_publisher(&params.id, &params.into_get_params()).await)
    }

    /// Get a single funder by OpenAlex ID.
    #[tool]
    pub async fn get_funder(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(self.client.get_funder(&params.id, &params.into_get_params()).await)
    }

    // ── Autocomplete tools ───────────────────────────────────────────────

    /// Type-ahead search for works by title. Returns up to 10 results sorted by citation count.
    #[tool]
    pub async fn autocomplete_works(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(self.client.autocomplete_works(&params.q).await)
    }

    /// Type-ahead search for authors. Returns up to 10 results sorted by citation count.
    #[tool]
    pub async fn autocomplete_authors(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(self.client.autocomplete_authors(&params.q).await)
    }

    /// Type-ahead search for sources (journals, repositories). Returns up to 10 results.
    #[tool]
    pub async fn autocomplete_sources(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(self.client.autocomplete_sources(&params.q).await)
    }

    /// Type-ahead search for institutions. Returns up to 10 results.
    #[tool]
    pub async fn autocomplete_institutions(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(self.client.autocomplete_institutions(&params.q).await)
    }

    /// Type-ahead search for concepts (deprecated but functional). Returns up to 10 results.
    #[tool]
    pub async fn autocomplete_concepts(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(self.client.autocomplete_concepts(&params.q).await)
    }

    /// Type-ahead search for publishers. Returns up to 10 results.
    #[tool]
    pub async fn autocomplete_publishers(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(self.client.autocomplete_publishers(&params.q).await)
    }

    /// Type-ahead search for funders. Returns up to 10 results.
    #[tool]
    pub async fn autocomplete_funders(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(self.client.autocomplete_funders(&params.q).await)
    }

    // ── Semantic search ──────────────────────────────────────────────────

    /// AI semantic search for works by conceptual similarity. Requires API key. Uses POST for queries > 2048 chars.
    #[tool]
    pub async fn find_works(&self, Parameters(params): Parameters<FindWorksToolParams>) -> Result<String, String> {
        let use_post = params.query.len() > 2048;
        let find_params = params.into_find_params();
        let result = if use_post {
            self.client.find_works_post(&find_params).await
        } else {
            self.client.find_works(&find_params).await
        };
        json_result(result)
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
