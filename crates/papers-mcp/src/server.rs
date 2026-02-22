use papers_core::{filter::FilterError, zotero as zotero_resolve, DiskCache, OpenAlexClient};
use papers_datalab::DatalabClient;
use papers_zotero::ZoteroClient;
use std::sync::Arc;
use std::time::Duration;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{ServerCapabilities, ServerInfo};
use rmcp::service::RoleServer;
use rmcp::{Peer, ServerHandler, tool, tool_handler, tool_router};
use serde::Serialize;

use crate::params::{
    AutocompleteToolParams, AuthorListToolParams, DomainListToolParams, FieldListToolParams,
    FindWorksToolParams, FunderListToolParams, GetToolParams, GetChapterToolParams,
    GetChunkToolParams, GetFigureToolParams, GetPaperOutlineToolParams, GetSectionToolParams,
    InstitutionListToolParams, ListPapersToolParams, ListTagsToolParams,
    PublisherListToolParams, SearchFiguresToolParams, SearchToolParams,
    SelectionAddToolParams, SelectionCreateToolParams,
    SelectionDeleteToolParams, SelectionGetToolParams, SelectionListToolParams,
    SelectionRemoveToolParams, SourceListToolParams, SubfieldListToolParams, TopicListToolParams,
    WorkListToolParams, WorkTextToolParams,
    ZoteroAnnotationListToolParams, ZoteroAttachmentListToolParams, ZoteroCollectionListToolParams,
    ZoteroCollectionNotesToolParams, ZoteroCollectionSubcollectionsToolParams,
    ZoteroCollectionTagsToolParams, ZoteroCollectionWorksToolParams, ZoteroDeletedListToolParams,
    ZoteroKeyToolParams, ZoteroNoParamsToolParams, ZoteroNoteListToolParams,
    ZoteroSettingGetToolParams, ZoteroTagGetToolParams, ZoteroTagListToolParams,
    ZoteroWorkChildrenToolParams, ZoteroWorkListToolParams, ZoteroWorkTagsToolParams,
};

#[derive(Clone)]
pub struct PapersMcp {
    client: OpenAlexClient,
    zotero: Arc<tokio::sync::Mutex<Option<ZoteroClient>>>,
    datalab: Option<DatalabClient>,
    rag: Option<Arc<papers_rag::RagStore>>,
    tool_router: ToolRouter<Self>,
}

impl PapersMcp {
    pub async fn new() -> Self {
        let mut client = OpenAlexClient::new();
        if let Ok(cache) = DiskCache::default_location(Duration::from_secs(600)) {
            client = client.with_cache(cache);
        }
        let datalab = DatalabClient::from_env().ok();
        let rag = Self::open_rag_store().await;
        Self {
            client,
            zotero: Arc::new(tokio::sync::Mutex::new(None)),
            datalab,
            rag,
            tool_router: Self::tool_router(),
        }
    }

    pub async fn with_client(client: OpenAlexClient) -> Self {
        let rag = Self::open_rag_store().await;
        Self {
            client,
            zotero: Arc::new(tokio::sync::Mutex::new(None)),
            datalab: DatalabClient::from_env().ok(),
            rag,
            tool_router: Self::tool_router(),
        }
    }

    /// Create a server with an explicit Zotero client (for testing).
    pub fn with_zotero(zotero: ZoteroClient) -> Self {
        Self {
            client: OpenAlexClient::new(),
            zotero: Arc::new(tokio::sync::Mutex::new(Some(zotero))),
            datalab: None,
            rag: None,
            tool_router: Self::tool_router(),
        }
    }

    async fn open_rag_store() -> Option<Arc<papers_rag::RagStore>> {
        let path = papers_rag::RagStore::default_path();
        match papers_rag::RagStore::open(&path).await {
            Ok(store) => Some(Arc::new(store)),
            Err(e) => {
                eprintln!("RAG store unavailable: {e}");
                None
            }
        }
    }

    fn resolve_selection_paper_ids(selection: &str) -> Result<Vec<String>, String> {
        let sel = papers_core::selection::load_selection(selection)
            .map_err(|e| e.to_string())?;
        let ids: Vec<String> = sel
            .entries
            .iter()
            .flat_map(|e| {
                e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
            })
            .collect();
        Ok(ids)
    }

    /// Try to get the Zotero client, probing if not yet connected.
    /// Caches a successful connection for future calls. Returns:
    /// - `Ok(Some(client))` — connected
    /// - `Ok(None)` — not configured (env vars absent)
    /// - `Err(msg)` — installed but not running (`NotRunning` error)
    /// Try to get a Zotero client for optional enrichment.
    ///
    /// Returns `Ok(None)` for any error — including "installed but not running" — so that
    /// OpenAlex tools like `work_get` can still serve results without Zotero enrichment.
    async fn get_optional_zotero(&self) -> Result<Option<ZoteroClient>, String> {
        let mut lock = self.zotero.lock().await;
        if let Some(z) = lock.as_ref() {
            return Ok(Some(z.clone()));
        }
        match ZoteroClient::from_env_prefer_local().await {
            Ok(z) => {
                *lock = Some(z.clone());
                Ok(Some(z))
            }
            Err(_) => Ok(None),
        }
    }

    /// Require a Zotero client; returns an error (including the "not running" hint) if unavailable.
    ///
    /// Used by dedicated Zotero tools where Zotero is mandatory.
    async fn require_zotero(&self) -> Result<ZoteroClient, String> {
        {
            let lock = self.zotero.lock().await;
            if let Some(z) = lock.as_ref() {
                return Ok(z.clone());
            }
        }
        match ZoteroClient::from_env_prefer_local().await {
            Ok(z) => {
                let mut lock = self.zotero.lock().await;
                *lock = Some(z.clone());
                Ok(z)
            }
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Returns true if this attachment supports annotation children (PDF, EPUB, or HTML snapshot).
fn is_annotatable_attachment(att: &papers_zotero::Item) -> bool {
    matches!(
        att.data.content_type.as_deref(),
        Some("application/pdf") | Some("application/epub+zip") | Some("text/html")
    )
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
        json_result(papers_core::api::work_list(&self.client, &params.into_work_list_params()).await)
    }

    /// Search, filter, and paginate author profiles. 110M+ records.
    /// Accepts shorthand filter aliases (institution, country, citations, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/authors/filter-authors
    #[tool]
    pub async fn author_list(&self, Parameters(params): Parameters<AuthorListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::author_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate publishing venues (journals, repositories, conferences).
    /// Accepts shorthand filter aliases (publisher, country, type, open, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/sources/filter-sources
    #[tool]
    pub async fn source_list(&self, Parameters(params): Parameters<SourceListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::source_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research institutions and organizations.
    /// Accepts shorthand filter aliases (country, continent, type, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/institutions/filter-institutions
    #[tool]
    pub async fn institution_list(&self, Parameters(params): Parameters<InstitutionListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::institution_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research topics (3-level hierarchy: domain > field > subfield > topic).
    /// Accepts shorthand filter aliases (domain, field, subfield, etc.) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/topics/filter-topics
    #[tool]
    pub async fn topic_list(&self, Parameters(params): Parameters<TopicListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::topic_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate publishing organizations (e.g. Elsevier, Springer Nature).
    /// Accepts shorthand filter aliases (country, continent, citations, works) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/publishers/filter-publishers
    #[tool]
    pub async fn publisher_list(&self, Parameters(params): Parameters<PublisherListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::publisher_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate funding organizations (e.g. NIH, NSF, ERC).
    /// Accepts shorthand filter aliases (country, continent, citations, works) that resolve to OpenAlex filter expressions.
    /// Advanced filtering: https://docs.openalex.org/api-entities/funders/filter-funders
    #[tool]
    pub async fn funder_list(&self, Parameters(params): Parameters<FunderListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::funder_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research domains (broadest level of topic hierarchy). 4 domains total.
    /// Accepts shorthand filter alias (works) that resolves to OpenAlex filter expression.
    /// Filtering: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists
    #[tool]
    pub async fn domain_list(&self, Parameters(params): Parameters<DomainListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::domain_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate academic fields (second level of topic hierarchy). 26 fields total.
    /// Accepts shorthand filter aliases (domain, works) that resolve to OpenAlex filter expressions.
    /// Filtering: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists
    #[tool]
    pub async fn field_list(&self, Parameters(params): Parameters<FieldListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::field_list(&self.client, &params.into_entity_params()).await)
    }

    /// Search, filter, and paginate research subfields (third level of topic hierarchy). ~252 subfields total.
    /// Accepts shorthand filter aliases (domain, field, works) that resolve to OpenAlex filter expressions.
    /// Filtering: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists
    #[tool]
    pub async fn subfield_list(&self, Parameters(params): Parameters<SubfieldListToolParams>) -> Result<String, String> {
        json_result(papers_core::api::subfield_list(&self.client, &params.into_entity_params()).await)
    }

    // ── Get tools ────────────────────────────────────────────────────────

    /// Get a single work by ID (OpenAlex ID, DOI, PMID, or PMCID).
    /// Response includes `in_zotero` (bool) and `zotero` (object or null) with brief Zotero library info.
    #[tool]
    pub async fn work_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        let zotero = self.get_optional_zotero().await?;
        match papers_core::api::work_get_response(&self.client, zotero.as_ref(), &params.id, &params.into_get_params()).await {
            Ok(response) => serde_json::to_string_pretty(&response).map_err(|e| format!("JSON serialization error: {e}")),
            Err(FilterError::Suggestions { query, suggestions }) => {
                let candidates: Vec<_> = suggestions
                    .into_iter()
                    .map(|(name, citations)| serde_json::json!({"name": name, "citations": citations}))
                    .collect();
                Ok(serde_json::json!({
                    "message": "no_exact_match",
                    "query": query,
                    "candidates": candidates,
                }).to_string())
            }
            Err(e) => Err(e.to_string()),
        }
    }

    /// Get a single author by ID (OpenAlex ID or ORCID).
    #[tool]
    pub async fn author_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::author_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single source by ID (OpenAlex ID or ISSN).
    #[tool]
    pub async fn source_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::source_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single institution by ID (OpenAlex ID or ROR).
    #[tool]
    pub async fn institution_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::institution_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single topic by OpenAlex ID.
    #[tool]
    pub async fn topic_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::topic_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single publisher by OpenAlex ID.
    #[tool]
    pub async fn publisher_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::publisher_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single funder by OpenAlex ID.
    #[tool]
    pub async fn funder_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::funder_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single domain by numeric ID (1-4).
    #[tool]
    pub async fn domain_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::domain_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single academic field by numeric ID (e.g. 17 for Computer Science).
    #[tool]
    pub async fn field_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::field_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    /// Get a single research subfield by numeric ID (e.g. 1702 for Artificial Intelligence).
    #[tool]
    pub async fn subfield_get(&self, Parameters(params): Parameters<GetToolParams>) -> Result<String, String> {
        json_result(papers_core::api::subfield_get(&self.client, &params.id, &params.into_get_params()).await)
    }

    // ── Autocomplete tools ───────────────────────────────────────────────

    /// Type-ahead search for works by title. Returns up to 10 results sorted by citation count.
    #[tool]
    pub async fn work_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers_core::api::work_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for authors. Returns up to 10 results sorted by citation count.
    #[tool]
    pub async fn author_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers_core::api::author_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for sources (journals, repositories). Returns up to 10 results.
    #[tool]
    pub async fn source_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers_core::api::source_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for institutions. Returns up to 10 results.
    #[tool]
    pub async fn institution_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers_core::api::institution_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for publishers. Returns up to 10 results.
    #[tool]
    pub async fn publisher_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers_core::api::publisher_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for funders. Returns up to 10 results.
    #[tool]
    pub async fn funder_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers_core::api::funder_autocomplete(&self.client, &params.q).await)
    }

    /// Type-ahead search for subfields. Returns up to 10 results.
    #[tool]
    pub async fn subfield_autocomplete(&self, Parameters(params): Parameters<AutocompleteToolParams>) -> Result<String, String> {
        json_result(papers_core::api::subfield_autocomplete(&self.client, &params.q).await)
    }

    // ── Semantic search ──────────────────────────────────────────────────

    /// AI semantic search for works by conceptual similarity. Requires API key. Uses POST for queries > 2048 chars.
    #[tool]
    pub async fn work_find(&self, Parameters(params): Parameters<FindWorksToolParams>) -> Result<String, String> {
        json_result(papers_core::api::work_find(&self.client, &params.into_find_params()).await)
    }

    // ── Zotero tools ─────────────────────────────────────────────────────

    /// List bibliographic items in your Zotero library (journalArticle, book, conferencePaper, etc.).
    /// Excludes notes, attachments, and annotations. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_list(&self, Parameters(p): Parameters<ZoteroWorkListToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let params = papers_zotero::ItemListParams {
            item_type: p.item_type,
            q: p.search,
            qmode: p.everything.then(|| "everything".to_string()),
            tag: p.tag,
            item_key: p.item_key,
            since: p.since,
            sort: p.sort,
            direction: p.direction,
            limit: p.limit,
            start: p.start,
            ..Default::default()
        };
        json_result(z.list_top_items(&params).await)
    }

    /// Get a single bibliographic item by Zotero key or title search. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_get(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let item = z.get_item(&key).await.map_err(|e| e.to_string())?;
        let mut value = serde_json::to_value(&item).map_err(|e| e.to_string())?;
        value["zotero_uri"] = serde_json::Value::String(format!("zotero://select/library/items/{key}"));
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())
    }

    /// List the collections a work belongs to. Multi-step: reads item record then resolves collection names.
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_collections(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let item = z.get_item(&key).await.map_err(|e| e.to_string())?;
        let col_keys = item.data.collections.clone();
        let mut collections = Vec::new();
        for ck in &col_keys {
            match z.get_collection(ck).await {
                Ok(c) => collections.push(c),
                Err(e) => return Err(e.to_string()),
            }
        }
        json_result::<Vec<_>, String>(Ok(collections))
    }

    /// List notes attached to a specific work. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_notes(&self, Parameters(p): Parameters<ZoteroWorkChildrenToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::ItemListParams { item_type: Some("note".into()), limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_item_children(&key, &params).await)
    }

    /// List file attachments of a specific work. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_attachments(&self, Parameters(p): Parameters<ZoteroWorkChildrenToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_item_children(&key, &params).await)
    }

    /// List all PDF annotations across all attachments of a work. Multi-step: fetches attachments
    /// then annotations per attachment. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_annotations(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let att_params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
        let attachments = z.list_item_children(&key, &att_params).await.map_err(|e| e.to_string())?;
        let ann_params = papers_zotero::ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
        let mut all_annotations = Vec::new();
        for att in &attachments.items {
            if !is_annotatable_attachment(att) { continue; }
            if let Ok(r) = z.list_item_children(&att.key, &ann_params).await {
                all_annotations.extend(r.items);
            }
        }
        json_result::<Vec<_>, String>(Ok(all_annotations))
    }

    /// List tags attached to a specific work. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_tags(&self, Parameters(p): Parameters<ZoteroWorkTagsToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::TagListParams { q: p.search, qmode: Some("contains".to_string()), limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_item_tags(&key, &params).await)
    }

    /// List all attachment items in the library (PDFs, snapshots, links).
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_attachment_list(&self, Parameters(p): Parameters<ZoteroAttachmentListToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), q: p.search, sort: p.sort, direction: p.direction, limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_items(&params).await)
    }

    /// Get a single attachment item by key or title search. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_attachment_get(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        json_result(z.get_item(&key).await)
    }

    /// List all annotation items in the library (highlights, comments from the PDF reader).
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_annotation_list(&self, Parameters(p): Parameters<ZoteroAnnotationListToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let params = papers_zotero::ItemListParams { item_type: Some("annotation".into()), limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_items(&params).await)
    }

    /// Get a single annotation by key or search string. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_annotation_get(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        json_result(z.get_item(&key).await)
    }

    /// List all note items in the library (user-written text notes, child or standalone).
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_note_list(&self, Parameters(p): Parameters<ZoteroNoteListToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let params = papers_zotero::ItemListParams { item_type: Some("note".into()), q: p.search, limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_items(&params).await)
    }

    /// Get a single note by key or search string. Full HTML content is in the `data.note` field.
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_note_get(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        json_result(z.get_item(&key).await)
    }

    /// List collections in the Zotero library. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_list(&self, Parameters(p): Parameters<ZoteroCollectionListToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let params = papers_zotero::CollectionListParams { sort: p.sort, direction: p.direction, limit: p.limit, start: p.start };
        let result = if p.scope.as_deref() == Some("top") {
            z.list_top_collections(&params).await
        } else {
            z.list_collections(&params).await
        };
        json_result(result)
    }

    /// Get a single collection by key or name search. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_get(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_collection_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        json_result(z.get_collection(&key).await)
    }

    /// List bibliographic works within a collection (excludes notes, attachments, annotations).
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_works(&self, Parameters(p): Parameters<ZoteroCollectionWorksToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_collection_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::ItemListParams {
            item_type: p.item_type,
            q: p.search,
            qmode: p.everything.then(|| "everything".to_string()),
            tag: p.tag,
            sort: p.sort,
            direction: p.direction,
            limit: p.limit,
            start: p.start,
            ..Default::default()
        };
        json_result(z.list_collection_top_items(&key, &params).await)
    }

    /// List attachment items within a collection. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_attachments(&self, Parameters(p): Parameters<ZoteroWorkChildrenToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_collection_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_collection_items(&key, &params).await)
    }

    /// List note items within a collection. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_notes(&self, Parameters(p): Parameters<ZoteroCollectionNotesToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_collection_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::ItemListParams { item_type: Some("note".into()), q: p.search, limit: p.limit, start: p.start, ..Default::default() };
        json_result(z.list_collection_items(&key, &params).await)
    }

    /// List annotations on PDFs within a collection. Multi-step: fetches attachments then
    /// annotations per attachment. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_annotations(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_collection_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let att_params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
        let attachments = z.list_collection_items(&key, &att_params).await.map_err(|e| e.to_string())?;
        let ann_params = papers_zotero::ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
        let mut all_annotations = Vec::new();
        for att in &attachments.items {
            if !is_annotatable_attachment(att) { continue; }
            if let Ok(r) = z.list_item_children(&att.key, &ann_params).await {
                all_annotations.extend(r.items);
            }
        }
        json_result::<Vec<_>, String>(Ok(all_annotations))
    }

    /// List sub-collections of a collection. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_subcollections(&self, Parameters(p): Parameters<ZoteroCollectionSubcollectionsToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_collection_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::CollectionListParams { sort: p.sort, direction: p.direction, limit: p.limit, start: p.start };
        json_result(z.list_subcollections(&key, &params).await)
    }

    /// List tags on items within a collection. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_collection_tags(&self, Parameters(p): Parameters<ZoteroCollectionTagsToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_collection_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let params = papers_zotero::TagListParams { q: p.search, qmode: Some("contains".to_string()), limit: p.limit, start: p.start, ..Default::default() };
        let result = if p.top == Some(true) {
            z.list_collection_top_items_tags(&key, &params).await
        } else {
            z.list_collection_items_tags(&key, &params).await
        };
        json_result(result)
    }

    /// List tags from the global library tag index (with per-tag item counts).
    /// Scope: `"all"` (default), `"top"` (top-level items only), or `"trash"`.
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_tag_list(&self, Parameters(p): Parameters<ZoteroTagListToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let params = papers_zotero::TagListParams { q: p.search, qmode: Some("contains".to_string()), sort: p.sort, direction: p.direction, limit: p.limit, start: p.start };
        let result = match p.scope.as_deref() {
            Some("trash") => z.list_trash_tags(&params).await,
            Some("top") => z.list_top_items_tags(&params).await,
            _ => z.list_tags(&params).await,
        };
        json_result(result)
    }

    /// Get a specific tag by name. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_tag_get(&self, Parameters(p): Parameters<ZoteroTagGetToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        json_result(z.get_tag(&p.name).await)
    }

    /// List all saved searches in the library. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_search_list(&self, Parameters(_p): Parameters<ZoteroNoParamsToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        json_result(z.list_searches().await)
    }

    /// List all Zotero groups accessible to the current user. Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_group_list(&self, Parameters(_p): Parameters<ZoteroNoParamsToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        json_result(z.list_groups().await)
    }

    /// Get Zotero's indexed full-text content for a work's primary PDF attachment.
    /// Resolves the work key, finds its first PDF child attachment, and returns the indexed text
    /// (content, page count, character count). Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_fulltext(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let att_params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
        let children = z.list_item_children(&key, &att_params).await.map_err(|e| e.to_string())?;
        let pdf = children.items.iter()
            .find(|a| a.data.content_type.as_deref() == Some("application/pdf"))
            .ok_or_else(|| format!("No PDF attachment found for item {key}"))?;
        json_result(z.get_item_fulltext(&pdf.key).await)
    }

    /// Get the CDN view URL for a work's primary PDF attachment.
    /// Resolves the work key, finds its first PDF child, and returns the URL.
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_view_url(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let att_params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
        let children = z.list_item_children(&key, &att_params).await.map_err(|e| e.to_string())?;
        let pdf = children.items.iter()
            .find(|a| a.data.content_type.as_deref() == Some("application/pdf"))
            .ok_or_else(|| format!("No PDF attachment found for item {key}"))?;
        z.get_item_file_view_url(&pdf.key).await.map_err(|e| e.to_string())
    }

    /// Download the PDF for a work's primary attachment and return its size in bytes.
    /// Useful to confirm a PDF is accessible. Use `zotero_work_view_url` to get the URL.
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_work_view(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        let att_params = papers_zotero::ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
        let children = z.list_item_children(&key, &att_params).await.map_err(|e| e.to_string())?;
        let pdf = children.items.iter()
            .find(|a| a.data.content_type.as_deref() == Some("application/pdf"))
            .ok_or_else(|| format!("No PDF attachment found for item {key}"))?;
        let bytes = z.get_item_file_view(&pdf.key).await.map_err(|e| e.to_string())?;
        Ok(format!("{{\"size_bytes\": {}}}", bytes.len()))
    }

    /// Get the CDN view URL for a specific attachment item by key.
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_attachment_url(&self, Parameters(p): Parameters<ZoteroKeyToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let key = zotero_resolve::resolve_item_key(&z, &p.key).await.map_err(|e| e.to_string())?;
        z.get_item_file_view_url(&key).await.map_err(|e| e.to_string())
    }

    /// Get permissions and identity for the current API key (user info, access scopes).
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_permission_list(&self, Parameters(_p): Parameters<ZoteroNoParamsToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        json_result(z.get_current_key_info().await)
    }

    /// List all library settings (tagColors, lastPageIndex, feeds, etc.).
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_setting_list(&self, Parameters(_p): Parameters<ZoteroNoParamsToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        json_result(z.get_settings().await)
    }

    /// Get a single library setting by key (e.g. `"tagColors"`).
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_setting_get(&self, Parameters(p): Parameters<ZoteroSettingGetToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        json_result(z.get_setting(&p.key).await)
    }

    /// List objects deleted from the library since a given version.
    /// Returns keys of deleted collections, items, searches, tags, and settings.
    /// Requires ZOTERO_USER_ID and ZOTERO_API_KEY.
    #[tool]
    pub async fn zotero_deleted_list(&self, Parameters(p): Parameters<ZoteroDeletedListToolParams>) -> Result<String, String> {
        let z = self.require_zotero().await?;
        let params = papers_zotero::DeletedParams { since: p.since.unwrap_or(0) };
        json_result(z.get_deleted(&params).await)
    }

    /// Get the full text content of a scholarly work by downloading and extracting its PDF.
    /// Tries multiple sources: local Zotero library, remote Zotero API,
    /// direct open-access URLs, and the OpenAlex content API.
    /// If no PDF is found, may ask the LLM for help finding one, or prompt the user
    /// to add the paper to Zotero via its DOI page.
    /// Accepts OpenAlex IDs, DOIs, or other work identifiers.
    #[tool]
    pub async fn work_text(
        &self,
        peer: Peer<RoleServer>,
        Parameters(params): Parameters<WorkTextToolParams>,
    ) -> Result<String, String> {
        let zotero = self.get_optional_zotero().await?;
        let datalab = self.datalab.as_ref().and_then(|dl| {
            let mode = match params.advanced.as_deref() {
                Some("fast")     => papers_core::text::ProcessingMode::Fast,
                Some("accurate") => papers_core::text::ProcessingMode::Accurate,
                Some(_)          => papers_core::text::ProcessingMode::Balanced,
                None             => return None,
            };
            Some((dl, mode))
        });
        match papers_core::text::work_text(&self.client, zotero.as_ref(), datalab, &params.id).await {
            Ok(result) => json_result::<_, String>(Ok(result)),
            Err(papers_core::text::WorkTextError::NoPdfFound { work_id, title, doi }) => {
                // Try the fallback chain: sampling → elicitation → error
                if let Some(result) = self.work_text_fallback(&peer, &work_id, title.as_deref(), doi.as_deref(), zotero.as_ref()).await {
                    return result;
                }
                let display = title.as_deref().unwrap_or(&work_id);
                let mut msg = format!("No PDF found for \"{display}\".");
                if let Some(doi) = &doi {
                    let bare = doi.strip_prefix("https://doi.org/").unwrap_or(doi);
                    msg.push_str(&format!(
                        "\n\nTo get this paper, ask the user to open https://doi.org/{bare} \
                         and save it to their Zotero library using the Zotero browser connector. \
                         Then call work_text again with the same ID."
                    ));
                }
                if zotero.is_none() {
                    msg.push_str(
                        "\n\nNote: Zotero integration is not configured. \
                         Set ZOTERO_USER_ID and ZOTERO_API_KEY environment variables to enable it."
                    );
                }
                Err(msg)
            }
            Err(e) => Err(e.to_string()),
        }
    }

    // ── Selection tools ───────────────────────────────────────────────────

    /// List all named paper selections with item counts.
    /// Marks the currently active selection.
    #[tool]
    pub async fn selection_list(&self, Parameters(_p): Parameters<SelectionListToolParams>) -> Result<String, String> {
        use papers_core::selection::{list_selection_names, load_selection, load_state};
        let names = list_selection_names();
        let state = load_state();
        let active = state.active.as_deref();
        let items: Vec<_> = names
            .iter()
            .map(|name| {
                let count = load_selection(name).map(|s| s.entries.len()).unwrap_or(0);
                serde_json::json!({
                    "name": name,
                    "item_count": count,
                    "is_active": Some(name.as_str()) == active,
                })
            })
            .collect();
        serde_json::to_string_pretty(&items).map_err(|e| e.to_string())
    }

    /// Get a selection's info and all its entries; activates the selection.
    /// Defaults to the active selection if name is omitted.
    #[tool]
    pub async fn selection_get(&self, Parameters(p): Parameters<SelectionGetToolParams>) -> Result<String, String> {
        use papers_core::selection::{active_selection_name, load_selection, load_state, resolve_selection, save_state};
        let name = match p.name {
            Some(n) => resolve_selection(&n).map_err(|e| e.to_string())?,
            None => active_selection_name().ok_or_else(|| "no active selection; run selection_list".to_string())?,
        };
        let sel = load_selection(&name).map_err(|e| e.to_string())?;
        let mut state = load_state();
        state.active = Some(name.clone());
        let _ = save_state(&state);
        json_result::<_, String>(Ok(serde_json::json!({
            "name": sel.name,
            "is_active": true,
            "entries": sel.entries,
        })))
    }

    /// Create a new named selection and activate it.
    /// Returns an error if a selection with that name already exists.
    #[tool]
    pub async fn selection_create(&self, Parameters(p): Parameters<SelectionCreateToolParams>) -> Result<String, String> {
        use papers_core::selection::{load_selection, load_state, save_selection, save_state, validate_name, Selection};
        validate_name(&p.name).map_err(|e| e.to_string())?;
        if load_selection(&p.name).is_ok() {
            return Err(format!("selection {:?} already exists", p.name));
        }
        let sel = Selection { name: p.name.clone(), entries: Vec::new() };
        save_selection(&sel).map_err(|e| e.to_string())?;
        let mut state = load_state();
        state.active = Some(p.name.clone());
        save_state(&state).map_err(|e| e.to_string())?;
        json_result::<_, String>(Ok(serde_json::json!({ "name": p.name, "is_active": true, "entries": [] })))
    }

    /// Delete a named selection. Deactivates it if it was the active selection.
    #[tool]
    pub async fn selection_delete(&self, Parameters(p): Parameters<SelectionDeleteToolParams>) -> Result<String, String> {
        use papers_core::selection::{delete_selection, load_state, resolve_selection, save_state};
        let name = resolve_selection(&p.name).map_err(|e| e.to_string())?;
        let mut state = load_state();
        let was_active = state.active.as_deref() == Some(&name);
        delete_selection(&name).map_err(|e| e.to_string())?;
        if was_active {
            state.active = None;
            let _ = save_state(&state);
        }
        json_result::<_, String>(Ok(serde_json::json!({ "name": name, "was_active": was_active })))
    }

    /// Add a paper to a selection using smart resolution.
    /// Input can be a Zotero key, DOI, OpenAlex Work ID (e.g. W2741809807), or title text.
    /// Zotero is optional; falls back to OpenAlex-only metadata if not configured.
    /// Skips duplicates silently. Defaults to the active selection.
    #[tool]
    pub async fn selection_add(&self, Parameters(p): Parameters<SelectionAddToolParams>) -> Result<String, String> {
        use papers_core::selection::{
            active_selection_name, entry_matches_doi, entry_matches_key, entry_matches_openalex,
            load_selection, resolve_paper, resolve_selection, save_selection,
        };
        let sel_name = match p.selection {
            Some(s) => resolve_selection(&s).map_err(|e| e.to_string())?,
            None => active_selection_name().ok_or_else(|| "no active selection; use selection param or create one first".to_string())?,
        };
        let zotero = self.get_optional_zotero().await?;
        let entry = resolve_paper(&p.paper, &self.client, zotero.as_ref()).await.map_err(|e| e.to_string())?;
        let mut sel = load_selection(&sel_name).map_err(|e| e.to_string())?;
        let is_dup = sel.entries.iter().any(|e| {
            entry.zotero_key.as_deref().map(|k| entry_matches_key(e, k)).unwrap_or(false)
                || entry.openalex_id.as_deref().map(|id| entry_matches_openalex(e, id)).unwrap_or(false)
                || entry.doi.as_deref().map(|d| entry_matches_doi(e, d)).unwrap_or(false)
        });
        if !is_dup {
            sel.entries.push(entry.clone());
            save_selection(&sel).map_err(|e| e.to_string())?;
        }
        json_result::<_, String>(Ok(entry))
    }

    // ── RAG tools ─────────────────────────────────────────────────────────────

    /// Semantic search across indexed paper chunks. Scope with selection, paper, chapter, or section.
    /// Returns matched chunks with immediate neighbors (prev/next) for reading context.
    /// Requires papers to be indexed first via `papers rag ingest`.
    #[tool]
    pub async fn rag_search(&self, Parameters(p): Parameters<SearchToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured. Run: papers rag ingest <ITEM_KEY>".to_string())?;
        let paper_ids = match p.selection.as_deref() {
            Some(sel) => Some(Self::resolve_selection_paper_ids(sel)?),
            None => p.paper_id.map(|id| vec![id]),
        };
        let params = papers_rag::SearchParams {
            query: p.query,
            paper_ids,
            chapter_idx: p.chapter_idx,
            section_idx: p.section_idx,
            filter_year_min: p.filter_year_min,
            filter_year_max: p.filter_year_max,
            filter_venue: p.filter_venue,
            filter_tags: p.filter_tags,
            filter_depth: p.filter_depth,
            limit: p.limit.unwrap_or(5),
        };
        json_result(papers_rag::query::search(rag, params).await)
    }

    /// Search for figures, tables, and diagrams by description.
    /// Use when the user asks about a specific visualization, comparison table, or diagram.
    #[tool]
    pub async fn rag_search_figures(&self, Parameters(p): Parameters<SearchFiguresToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured. Run: papers rag ingest <ITEM_KEY>".to_string())?;
        let paper_ids = match p.selection.as_deref() {
            Some(sel) => Some(Self::resolve_selection_paper_ids(sel)?),
            None => p.paper_id.map(|id| vec![id]),
        };
        let params = papers_rag::SearchFiguresParams {
            query: p.query,
            paper_ids,
            filter_figure_type: p.filter_figure_type,
            limit: p.limit.unwrap_or(5),
        };
        json_result(papers_rag::query::search_figures(rag, params).await)
    }

    /// Retrieve a specific chunk by ID with its prev/next neighbors for sequential reading.
    /// Use after rag_search to follow prev/next chunk references.
    #[tool]
    pub async fn rag_get_chunk(&self, Parameters(p): Parameters<GetChunkToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured.".to_string())?;
        json_result(papers_rag::query::get_chunk(rag, &p.chunk_id).await)
    }

    /// Fetch all chunks in a specific section in reading order.
    /// Use when you need complete section content after finding a relevant chunk.
    #[tool]
    pub async fn rag_get_section(&self, Parameters(p): Parameters<GetSectionToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured.".to_string())?;
        json_result(papers_rag::query::get_section(rag, &p.paper_id, p.chapter_idx, p.section_idx).await)
    }

    /// Fetch the full content of an entire chapter, grouped by section.
    /// Use when the user asks about a broad topic within a paper.
    #[tool]
    pub async fn rag_get_chapter(&self, Parameters(p): Parameters<GetChapterToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured.".to_string())?;
        json_result(papers_rag::query::get_chapter(rag, &p.paper_id, p.chapter_idx).await)
    }

    /// Retrieve full details for a figure by ID, including the image file path.
    /// Use when you need the image path to display it, or to see cross-references.
    #[tool]
    pub async fn rag_get_figure(&self, Parameters(p): Parameters<GetFigureToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured.".to_string())?;
        json_result(papers_rag::query::get_figure(rag, &p.figure_id).await)
    }

    /// Get the table of contents for a paper (all chapters and sections with chunk counts).
    /// Use to understand paper structure before searching within it.
    #[tool]
    pub async fn rag_get_paper_outline(&self, Parameters(p): Parameters<GetPaperOutlineToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured.".to_string())?;
        json_result(papers_rag::query::get_paper_outline(rag, &p.paper_id).await)
    }

    /// Browse indexed papers with optional metadata filters.
    /// Use when the user asks what papers are available, or to find a paper by metadata.
    #[tool]
    pub async fn rag_list_papers(&self, Parameters(p): Parameters<ListPapersToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured.".to_string())?;
        let paper_ids = match p.selection.as_deref() {
            Some(sel) => Some(Self::resolve_selection_paper_ids(sel)?),
            None => None,
        };
        let params = papers_rag::ListPapersParams {
            paper_ids,
            filter_year_min: p.filter_year_min,
            filter_year_max: p.filter_year_max,
            filter_venue: p.filter_venue,
            filter_tags: p.filter_tags,
            filter_authors: p.filter_authors,
            sort_by: p.sort_by,
            limit: p.limit.unwrap_or(50),
        };
        json_result(papers_rag::query::list_papers(rag, params).await)
    }

    /// List all tags and per-tag paper counts. Use to discover available filter categories.
    #[tool]
    pub async fn rag_list_tags(&self, Parameters(p): Parameters<ListTagsToolParams>) -> Result<String, String> {
        let rag = self.rag.as_ref().ok_or_else(|| "RAG database not configured.".to_string())?;
        let paper_ids = match p.selection.as_deref() {
            Some(sel) => Some(Self::resolve_selection_paper_ids(sel)?),
            None => None,
        };
        let params = papers_rag::ListTagsParams { paper_ids };
        json_result(papers_rag::query::list_tags(rag, params).await)
    }

    /// Remove a paper from a selection.
    /// Matches by Zotero key, DOI, OpenAlex ID, or title substring.
    /// Defaults to the active selection.
    #[tool]
    pub async fn selection_remove(&self, Parameters(p): Parameters<SelectionRemoveToolParams>) -> Result<String, String> {
        use papers_core::selection::{
            active_selection_name, entry_matches_remove_input,
            load_selection, resolve_selection, save_selection, SelectionError,
        };
        let sel_name = match p.selection {
            Some(s) => resolve_selection(&s).map_err(|e| e.to_string())?,
            None => active_selection_name().ok_or_else(|| "no active selection".to_string())?,
        };
        let mut sel = load_selection(&sel_name).map_err(|e| e.to_string())?;
        let removed = sel.entries.iter().find(|e| entry_matches_remove_input(e, &p.paper)).cloned();
        let before = sel.entries.len();
        sel.entries.retain(|e| !entry_matches_remove_input(e, &p.paper));
        if sel.entries.len() == before {
            return Err(SelectionError::ItemNotFound.to_string());
        }
        save_selection(&sel).map_err(|e| e.to_string())?;
        let title = removed.and_then(|e| e.title).unwrap_or_else(|| p.paper.clone());
        json_result::<_, String>(Ok(serde_json::json!({ "removed": title, "selection": sel_name })))
    }
}

impl PapersMcp {
    /// Fallback chain when no PDF is found: sampling → elicitation + polling → None.
    async fn work_text_fallback(
        &self,
        peer: &Peer<RoleServer>,
        work_id: &str,
        title: Option<&str>,
        doi: Option<&str>,
        zotero: Option<&ZoteroClient>,
    ) -> Option<Result<String, String>> {
        let display = title.unwrap_or(work_id);

        // Step A: Try sampling — ask the LLM to find a PDF URL
        if let Some(doi) = doi {
            if peer.supports_sampling_tools() {
                if let Some(result) = self.try_sampling_pdf(peer, work_id, title, doi).await {
                    return Some(result);
                }
            }
        }

        // Step B: Try elicitation — ask the user to add to Zotero via DOI
        if let (Some(doi), Some(zotero)) = (doi, zotero) {
            let modes = peer.supported_elicitation_modes();
            if modes.contains(&rmcp::service::ElicitationMode::Url) {
                let bare_doi = doi.strip_prefix("https://doi.org/").unwrap_or(doi);
                let url = format!("https://doi.org/{bare_doi}");
                let message = format!(
                    "No PDF found for \"{display}\". Open the DOI page to save this paper to your Zotero library?"
                );

                match peer.elicit_url(&message, url::Url::parse(&url).unwrap(), format!("work_text_{work_id}")).await {
                    Ok(rmcp::model::ElicitationAction::Accept) => {
                        // Poll Zotero with progress notifications
                        return Some(self.poll_with_progress(peer, zotero, work_id, title, bare_doi).await);
                    }
                    Ok(_) => {
                        // User declined or cancelled
                        return None;
                    }
                    Err(_) => {}
                }
            }
        }

        None
    }

    /// Ask the LLM to find a PDF URL via sampling, then try to download it.
    async fn try_sampling_pdf(
        &self,
        peer: &Peer<RoleServer>,
        work_id: &str,
        title: Option<&str>,
        doi: &str,
    ) -> Option<Result<String, String>> {
        use rmcp::model::{CreateMessageRequestParams, SamplingMessage};

        let bare_doi = doi.strip_prefix("https://doi.org/").unwrap_or(doi);
        let display = title.unwrap_or(work_id);
        let prompt = format!(
            "Find a direct PDF download URL for the academic paper: \"{display}\" (DOI: {bare_doi}). \
             Reply with ONLY the URL or the word 'none' if you cannot find one."
        );

        let result = peer.create_message(CreateMessageRequestParams {
            meta: None,
            task: None,
            messages: vec![SamplingMessage::user_text(&prompt)],
            model_preferences: None,
            system_prompt: None,
            temperature: None,
            max_tokens: 200,
            stop_sequences: None,
            include_context: None,
            metadata: None,
            tools: None,
            tool_choice: None,
        }).await;

        let result = match result {
            Ok(r) => r,
            Err(_) => return None,
        };

        // Extract text from response
        let text = match result.message.content.first() {
            Some(rmcp::model::SamplingMessageContent::Text(t)) => t.text.clone(),
            _ => return None,
        };
        let text = text.trim();

        if text.eq_ignore_ascii_case("none") || text.is_empty() || !text.starts_with("http") {
            return None;
        }

        // Try downloading the URL
        let http = reqwest::Client::new();
        let resp = match http.get(text)
            .header("User-Agent", "papers-mcp/0.1")
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r,
            _ => return None,
        };

        let is_pdf = resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("application/pdf"));

        if !is_pdf {
            return None;
        }

        let bytes = match resp.bytes().await {
            Ok(b) if !b.is_empty() => b,
            _ => return None,
        };

        let extracted = match papers_core::text::extract_text_bytes(&bytes) {
            Ok(t) => t,
            Err(_) => return None,
        };

        Some(json_result::<_, String>(Ok(papers_core::text::WorkTextResult {
            text: extracted,
            source: papers_core::text::PdfSource::DirectUrl { url: text.to_string() },
            work_id: work_id.to_string(),
            title: title.map(String::from),
            doi: Some(doi.to_string()),
        })))
    }

    /// Poll Zotero for a work, sending progress notifications to the client.
    async fn poll_with_progress(
        &self,
        peer: &Peer<RoleServer>,
        zotero: &ZoteroClient,
        work_id: &str,
        title: Option<&str>,
        doi: &str,
    ) -> Result<String, String> {
        use rmcp::model::ProgressNotificationParam;

        let token = rmcp::model::ProgressToken(rmcp::model::NumberOrString::String(format!("poll_{work_id}").into()));
        let total_steps = 56i64; // 1 initial + 55 polls

        // Notify start
        let _ = peer.notify_progress(ProgressNotificationParam {
            progress_token: token.clone(),
            progress: 0.0,
            total: Some(total_steps as f64),
            message: Some("Waiting for paper to appear in Zotero...".into()),
        }).await;

        // Initial wait
        tokio::time::sleep(Duration::from_secs(5)).await;
        let _ = peer.notify_progress(ProgressNotificationParam {
            progress_token: token.clone(),
            progress: 1.0,
            total: Some(total_steps as f64),
            message: Some("Polling Zotero...".into()),
        }).await;

        for i in 0..55 {
            match papers_core::text::try_zotero(zotero, doi, title).await {
                Ok(Some((bytes, source, _zotero_key))) => {
                    let _ = peer.notify_progress(ProgressNotificationParam {
                        progress_token: token.clone(),
                        progress: total_steps as f64,
                        total: Some(total_steps as f64),
                        message: Some("PDF found!".into()),
                    }).await;

                    let text = match papers_core::text::extract_text_bytes(&bytes) {
                        Ok(t) => t,
                        Err(e) => return Err(format!("PDF extraction error: {e}")),
                    };

                    return json_result::<_, String>(Ok(papers_core::text::WorkTextResult {
                        text,
                        source,
                        work_id: work_id.to_string(),
                        title: title.map(String::from),
                        doi: Some(doi.to_string()),
                    }));
                }
                Ok(None) => {}
                Err(e) => return Err(e.to_string()),
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
            let _ = peer.notify_progress(ProgressNotificationParam {
                progress_token: token.clone(),
                progress: (i + 2) as f64,
                total: Some(total_steps as f64),
                message: Some(format!("Polling Zotero... ({}/55)", i + 1)),
            }).await;
        }

        Err(format!(
            "Timed out waiting for paper in Zotero: {}", title.unwrap_or(work_id)
        ))
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
                 authors, sources, institutions, topics, publishers, and funders. \
                 Also supports full-text extraction from PDFs via the work_text tool, \
                 which can download papers from Zotero, open-access repositories, \
                 or the OpenAlex content API."
                    .into(),
            ),
        }
    }
}
