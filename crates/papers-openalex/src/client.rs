use crate::error::{OpenAlexError, Result};
use crate::params::{FindWorksParams, GetParams, ListParams};
use crate::response::{AutocompleteResponse, FindWorksResponse, ListResponse};
use crate::types::*;
use serde::de::DeserializeOwned;

const DEFAULT_BASE_URL: &str = "https://api.openalex.org";

/// Async client for the [OpenAlex REST API](https://docs.openalex.org).
///
/// Provides 30 methods covering all OpenAlex endpoints: 10 list, 10 get,
/// 8 autocomplete, and 2 semantic search.
///
/// # Creating a client
///
/// ```no_run
/// use papers_openalex::OpenAlexClient;
///
/// // Reads API key from OPENALEX_KEY env var (optional but recommended)
/// let client = OpenAlexClient::new();
///
/// // Or pass an explicit API key
/// let client = OpenAlexClient::with_api_key("your-key-here");
/// ```
///
/// # Example: search and paginate
///
/// ```no_run
/// # async fn example() -> papers_openalex::Result<()> {
/// use papers_openalex::{OpenAlexClient, ListParams};
///
/// let client = OpenAlexClient::new();
/// let params = ListParams::builder()
///     .search("machine learning")
///     .filter("publication_year:2024,is_oa:true")
///     .sort("cited_by_count:desc")
///     .per_page(5)
///     .build();
///
/// let response = client.list_works(&params).await?;
/// for work in &response.results {
///     println!("{}: {} cites",
///         work.display_name.as_deref().unwrap_or("?"),
///         work.cited_by_count.unwrap_or(0));
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Example: cursor pagination
///
/// ```no_run
/// # async fn example() -> papers_openalex::Result<()> {
/// use papers_openalex::{OpenAlexClient, ListParams};
///
/// let client = OpenAlexClient::new();
/// let mut cursor = Some("*".to_string());
///
/// while let Some(c) = cursor {
///     let params = ListParams {
///         cursor: Some(c),
///         per_page: Some(200),
///         filter: Some("publication_year:2024".into()),
///         ..Default::default()
///     };
///     let response = client.list_works(&params).await?;
///     for work in &response.results {
///         // process each work
///     }
///     cursor = response.meta.next_cursor;
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OpenAlexClient {
    http: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
}

impl Default for OpenAlexClient {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenAlexClient {
    /// Create a new client, reading the API key from the `OPENALEX_KEY`
    /// environment variable. The key is optional for most endpoints but
    /// recommended for higher rate limits.
    ///
    /// ```no_run
    /// use papers_openalex::OpenAlexClient;
    /// let client = OpenAlexClient::new();
    /// ```
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: std::env::var("OPENALEX_KEY").ok(),
        }
    }

    /// Create a new client with an explicit API key.
    ///
    /// ```no_run
    /// use papers_openalex::OpenAlexClient;
    /// let client = OpenAlexClient::with_api_key("your-key-here");
    /// ```
    pub fn with_api_key(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: Some(api_key.into()),
        }
    }

    /// Override the base URL. Useful for testing with a mock server.
    ///
    /// ```no_run
    /// use papers_openalex::OpenAlexClient;
    /// let client = OpenAlexClient::new()
    ///     .with_base_url("http://localhost:8080");
    /// ```
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    // ── Private helpers ────────────────────────────────────────────────

    fn append_api_key(&self, pairs: &mut Vec<(&str, String)>) {
        if let Some(key) = &self.api_key {
            pairs.push(("api_key", key.clone()));
        }
    }

    async fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        mut query: Vec<(&str, String)>,
    ) -> Result<T> {
        self.append_api_key(&mut query);
        let url = format!("{}{}", self.base_url, path);
        let resp = self.http.get(&url).query(&query).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let message = resp.text().await.unwrap_or_default();
            return Err(OpenAlexError::Api {
                status: status.as_u16(),
                message,
            });
        }
        let text = resp.text().await?;
        serde_json::from_str(&text).map_err(OpenAlexError::Json)
    }

    async fn post_json<T: DeserializeOwned>(
        &self,
        path: &str,
        mut query: Vec<(&str, String)>,
        body: serde_json::Value,
    ) -> Result<T> {
        self.append_api_key(&mut query);
        let url = format!("{}{}", self.base_url, path);
        let resp = self.http.post(&url).query(&query).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let message = resp.text().await.unwrap_or_default();
            return Err(OpenAlexError::Api {
                status: status.as_u16(),
                message,
            });
        }
        let text = resp.text().await?;
        serde_json::from_str(&text).map_err(OpenAlexError::Json)
    }

    async fn list_entities<T: DeserializeOwned>(
        &self,
        path: &str,
        params: &ListParams,
    ) -> Result<ListResponse<T>> {
        self.get_json(path, params.to_query_pairs()).await
    }

    async fn get_entity<T: DeserializeOwned>(
        &self,
        entity_path: &str,
        id: &str,
        params: &GetParams,
    ) -> Result<T> {
        let path = format!("{}/{}", entity_path, id);
        self.get_json(&path, params.to_query_pairs()).await
    }

    async fn autocomplete_entity(
        &self,
        entity: &str,
        q: &str,
    ) -> Result<AutocompleteResponse> {
        let path = format!("/autocomplete/{}", entity);
        self.get_json(&path, vec![("q", q.to_string())]).await
    }

    // ── List endpoints ─────────────────────────────────────────────────

    /// List scholarly works (articles, books, datasets, preprints). 240M+
    /// records. Supports full-text search across titles, abstracts, and full
    /// text. Filter by publication year, OA status, type, citations, author,
    /// institution, topic, funder, and 130+ other fields.
    ///
    /// `GET /works`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let params = ListParams::builder()
    ///     .filter("publication_year:2024,is_oa:true")
    ///     .sort("cited_by_count:desc")
    ///     .per_page(5)
    ///     .build();
    /// let response = client.list_works(&params).await?;
    /// // response.meta.count => total matching works
    /// // response.results    => Vec<Work>
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_works(&self, params: &ListParams) -> Result<ListResponse<Work>> {
        self.list_entities("/works", params).await
    }

    /// List disambiguated author profiles. 110M+ records. Each author has a
    /// unified identity across name variants, with linked ORCID, institutional
    /// affiliations, publication history, and citation metrics.
    ///
    /// `GET /authors`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let params = ListParams::builder()
    ///     .search("einstein")
    ///     .per_page(3)
    ///     .build();
    /// let response = client.list_authors(&params).await?;
    /// for author in &response.results {
    ///     println!("{}: {} works",
    ///         author.display_name.as_deref().unwrap_or("?"),
    ///         author.works_count.unwrap_or(0));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_authors(&self, params: &ListParams) -> Result<ListResponse<Author>> {
        self.list_entities("/authors", params).await
    }

    /// List publishing venues: journals, repositories, conferences, ebook
    /// platforms, and book series. Includes ISSN identifiers, OA status, APC
    /// pricing, host organization, and impact metrics.
    ///
    /// `GET /sources`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let params = ListParams::builder()
    ///     .filter("is_oa:true,type:journal")
    ///     .sort("cited_by_count:desc")
    ///     .per_page(5)
    ///     .build();
    /// let response = client.list_sources(&params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_sources(&self, params: &ListParams) -> Result<ListResponse<Source>> {
        self.list_entities("/sources", params).await
    }

    /// List research organizations: universities, hospitals, companies,
    /// government agencies. Linked to ROR identifiers. Includes geographic
    /// location, parent/child relationships, and affiliated repositories.
    ///
    /// `GET /institutions`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let params = ListParams::builder()
    ///     .filter("country_code:US,type:education")
    ///     .sort("cited_by_count:desc")
    ///     .per_page(10)
    ///     .build();
    /// let response = client.list_institutions(&params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_institutions(
        &self,
        params: &ListParams,
    ) -> Result<ListResponse<Institution>> {
        self.list_entities("/institutions", params).await
    }

    /// List research topics organized in a 3-level hierarchy: domain > field >
    /// subfield > topic. AI-generated descriptions and keywords. Each work is
    /// assigned up to 3 topics with relevance scores.
    ///
    /// `GET /topics`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let params = ListParams::builder()
    ///     .search("machine learning")
    ///     .per_page(5)
    ///     .build();
    /// let response = client.list_topics(&params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_topics(&self, params: &ListParams) -> Result<ListResponse<Topic>> {
        self.list_entities("/topics", params).await
    }

    /// List publishing organizations (e.g. Elsevier, Springer Nature). Includes
    /// parent/child hierarchy, country of origin, and linked sources. Some
    /// publishers also act as funders or institutions (see `roles`).
    ///
    /// `GET /publishers`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let params = ListParams::builder()
    ///     .sort("works_count:desc")
    ///     .per_page(10)
    ///     .build();
    /// let response = client.list_publishers(&params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_publishers(&self, params: &ListParams) -> Result<ListResponse<Publisher>> {
        self.list_entities("/publishers", params).await
    }

    /// List research domains (broadest level of topic hierarchy). 4 domains
    /// total: Life Sciences, Social Sciences, Physical Sciences, Health
    /// Sciences. Each domain contains multiple academic fields.
    ///
    /// `GET /domains`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.list_domains(&ListParams::default()).await?;
    /// for domain in &response.results {
    ///     println!("{}: {} works",
    ///         domain.display_name.as_deref().unwrap_or("?"),
    ///         domain.works_count.unwrap_or(0));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_domains(&self, params: &ListParams) -> Result<ListResponse<Domain>> {
        self.list_entities("/domains", params).await
    }

    /// List academic fields (second level of topic hierarchy). 26 fields total
    /// (e.g. Computer Science, Medicine, Mathematics). Each field has a parent
    /// domain and contains multiple subfields.
    ///
    /// `GET /fields`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.list_fields(&ListParams::default()).await?;
    /// for field in &response.results {
    ///     println!("{}: {} works",
    ///         field.display_name.as_deref().unwrap_or("?"),
    ///         field.works_count.unwrap_or(0));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_fields(&self, params: &ListParams) -> Result<ListResponse<Field>> {
        self.list_entities("/fields", params).await
    }

    /// List research subfields (third level of topic hierarchy). ~252 subfields
    /// total (e.g. Artificial Intelligence, Organic Chemistry). Each subfield
    /// has a parent field and domain, and contains multiple topics.
    ///
    /// `GET /subfields`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.list_subfields(&ListParams::default()).await?;
    /// for subfield in &response.results {
    ///     println!("{}: {} works",
    ///         subfield.display_name.as_deref().unwrap_or("?"),
    ///         subfield.works_count.unwrap_or(0));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_subfields(&self, params: &ListParams) -> Result<ListResponse<Subfield>> {
        self.list_entities("/subfields", params).await
    }

    /// List research funding organizations (e.g. NIH, NSF, ERC). Linked to
    /// Crossref funder registry. Includes grant counts, funded works, and impact
    /// metrics.
    ///
    /// `GET /funders`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, ListParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let params = ListParams::builder()
    ///     .filter("country_code:US")
    ///     .sort("works_count:desc")
    ///     .per_page(5)
    ///     .build();
    /// let response = client.list_funders(&params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn list_funders(&self, params: &ListParams) -> Result<ListResponse<Funder>> {
        self.list_entities("/funders", params).await
    }

    // ── Single entity endpoints ────────────────────────────────────────

    /// Get a single scholarly work by ID. Returns full metadata including title,
    /// authors, abstract (as inverted index), citations, topics, OA status,
    /// locations, funding, and bibliographic data.
    ///
    /// `GET /works/{id}`
    ///
    /// Accepts: OpenAlex ID (`W...`), DOI, PMID (`pmid:...`), PMCID
    /// (`pmcid:...`), MAG (`mag:...`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    ///
    /// // By OpenAlex ID
    /// let work = client.get_work("W2741809807", &GetParams::default()).await?;
    ///
    /// // By DOI
    /// let work = client.get_work("https://doi.org/10.7717/peerj.4375", &GetParams::default()).await?;
    ///
    /// // With field selection
    /// let params = GetParams::builder().select("id,display_name,cited_by_count").build();
    /// let work = client.get_work("W2741809807", &params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_work(&self, id: &str, params: &GetParams) -> Result<Work> {
        self.get_entity("/works", id, params).await
    }

    /// Get a single author profile. Returns disambiguated identity with name
    /// variants, institutional affiliations over time, publication/citation
    /// counts, h-index, and topic expertise.
    ///
    /// `GET /authors/{id}`
    ///
    /// Accepts: OpenAlex ID (`A...`), ORCID.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let author = client.get_author("A5023888391", &GetParams::default()).await?;
    /// println!("{}: h-index {}",
    ///     author.display_name.as_deref().unwrap_or("?"),
    ///     author.summary_stats.as_ref().and_then(|s| s.h_index).unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_author(&self, id: &str, params: &GetParams) -> Result<Author> {
        self.get_entity("/authors", id, params).await
    }

    /// Get a single publishing venue. Returns ISSNs, OA status, DOAJ membership,
    /// APC pricing, host publisher hierarchy, impact metrics, and publication
    /// year range.
    ///
    /// `GET /sources/{id}`
    ///
    /// Accepts: OpenAlex ID (`S...`), ISSN.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let source = client.get_source("S137773608", &GetParams::default()).await?;
    /// println!("{} ({})", source.display_name.as_deref().unwrap_or("?"),
    ///     source.r#type.as_deref().unwrap_or("unknown"));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_source(&self, id: &str, params: &GetParams) -> Result<Source> {
        self.get_entity("/sources", id, params).await
    }

    /// Get a single research institution. Returns ROR ID, geographic
    /// coordinates, parent/child institution relationships, hosted repositories,
    /// and research output metrics.
    ///
    /// `GET /institutions/{id}`
    ///
    /// Accepts: OpenAlex ID (`I...`), ROR.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let inst = client.get_institution("I136199984", &GetParams::default()).await?;
    /// // Also accepts ROR:
    /// let inst = client.get_institution("https://ror.org/03vek6s52", &GetParams::default()).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_institution(&self, id: &str, params: &GetParams) -> Result<Institution> {
        self.get_entity("/institutions", id, params).await
    }

    /// Get a single research topic. Returns AI-generated description, keywords,
    /// position in domain > field > subfield hierarchy, sibling topics, and work
    /// counts.
    ///
    /// `GET /topics/{id}`
    ///
    /// Accepts: OpenAlex ID (`T...`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let topic = client.get_topic("T10001", &GetParams::default()).await?;
    /// println!("{}: {} works",
    ///     topic.display_name.as_deref().unwrap_or("?"),
    ///     topic.works_count.unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_topic(&self, id: &str, params: &GetParams) -> Result<Topic> {
        self.get_entity("/topics", id, params).await
    }

    /// Get a single publisher. Returns hierarchy level, parent publisher,
    /// country codes, linked sources, and citation metrics.
    ///
    /// `GET /publishers/{id}`
    ///
    /// Accepts: OpenAlex ID (`P...`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let publisher = client.get_publisher("P4310319965", &GetParams::default()).await?;
    /// println!("{}: {} works",
    ///     publisher.display_name.as_deref().unwrap_or("?"),
    ///     publisher.works_count.unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_publisher(&self, id: &str, params: &GetParams) -> Result<Publisher> {
        self.get_entity("/publishers", id, params).await
    }

    /// Get a single research domain. Returns description, child fields,
    /// sibling domains, and work/citation counts.
    ///
    /// `GET /domains/{id}`
    ///
    /// Accepts: numeric ID (e.g. `"3"` for Physical Sciences).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let domain = client.get_domain("3", &GetParams::default()).await?;
    /// println!("{}: {} fields",
    ///     domain.display_name.as_deref().unwrap_or("?"),
    ///     domain.fields.as_ref().map(|f| f.len()).unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_domain(&self, id: &str, params: &GetParams) -> Result<Domain> {
        self.get_entity("/domains", id, params).await
    }

    /// Get a single academic field. Returns parent domain, child subfields,
    /// sibling fields, and work/citation counts.
    ///
    /// `GET /fields/{id}`
    ///
    /// Accepts: numeric ID (e.g. `"17"` for Computer Science).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let field = client.get_field("17", &GetParams::default()).await?;
    /// println!("{}: {} subfields",
    ///     field.display_name.as_deref().unwrap_or("?"),
    ///     field.subfields.as_ref().map(|s| s.len()).unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_field(&self, id: &str, params: &GetParams) -> Result<Field> {
        self.get_entity("/fields", id, params).await
    }

    /// Get a single research subfield. Returns parent field and domain, child
    /// topics, sibling subfields, and work/citation counts.
    ///
    /// `GET /subfields/{id}`
    ///
    /// Accepts: numeric ID (e.g. `"1702"` for Artificial Intelligence).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let subfield = client.get_subfield("1702", &GetParams::default()).await?;
    /// println!("{}: {} topics",
    ///     subfield.display_name.as_deref().unwrap_or("?"),
    ///     subfield.topics.as_ref().map(|t| t.len()).unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_subfield(&self, id: &str, params: &GetParams) -> Result<Subfield> {
        self.get_entity("/subfields", id, params).await
    }

    /// Get a single funder. Returns Wikidata description, Crossref funder ID,
    /// grant/award counts, country, and research impact metrics.
    ///
    /// `GET /funders/{id}`
    ///
    /// Accepts: OpenAlex ID (`F...`).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, GetParams};
    ///
    /// let client = OpenAlexClient::new();
    /// let funder = client.get_funder("F4320332161", &GetParams::default()).await?;
    /// println!("{} ({}): {} awards",
    ///     funder.display_name.as_deref().unwrap_or("?"),
    ///     funder.country_code.as_deref().unwrap_or("?"),
    ///     funder.awards_count.unwrap_or(0));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn get_funder(&self, id: &str, params: &GetParams) -> Result<Funder> {
        self.get_entity("/funders", id, params).await
    }

    // ── Autocomplete endpoints ─────────────────────────────────────────

    /// Autocomplete for works. Searches titles. Returns up to 10 results sorted
    /// by citation count. Hint shows first author name.
    ///
    /// `GET /autocomplete/works?q=...`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_works("machine learning").await?;
    /// for result in &response.results {
    ///     println!("{} (by {})", result.display_name,
    ///         result.hint.as_deref().unwrap_or("unknown author"));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_works(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("works", q).await
    }

    /// Autocomplete for authors. Searches display names. Returns up to 10
    /// results sorted by citation count. Hint shows last known institution and
    /// country.
    ///
    /// `GET /autocomplete/authors?q=...`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_authors("einstein").await?;
    /// for result in &response.results {
    ///     println!("{} — {}", result.display_name,
    ///         result.hint.as_deref().unwrap_or(""));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_authors(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("authors", q).await
    }

    /// Autocomplete for sources (journals, repositories). Searches display
    /// names. Returns up to 10 results sorted by citation count. Hint shows
    /// host organization. External ID is ISSN.
    ///
    /// `GET /autocomplete/sources?q=...`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_sources("nature").await?;
    /// for result in &response.results {
    ///     println!("{} ({})", result.display_name,
    ///         result.hint.as_deref().unwrap_or(""));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_sources(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("sources", q).await
    }

    /// Autocomplete for institutions. Searches display names. Returns up to 10
    /// results sorted by citation count. Hint shows city and country. External
    /// ID is ROR.
    ///
    /// `GET /autocomplete/institutions?q=...`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_institutions("harvard").await?;
    /// for result in &response.results {
    ///     println!("{} — {}", result.display_name,
    ///         result.hint.as_deref().unwrap_or(""));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_institutions(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("institutions", q).await
    }

    /// Autocomplete for concepts (deprecated entity type, but autocomplete still
    /// works). Hint shows concept hierarchy level.
    ///
    /// `GET /autocomplete/concepts?q=...`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_concepts("physics").await?;
    /// for result in &response.results {
    ///     println!("{} (level {})", result.display_name,
    ///         result.hint.as_deref().unwrap_or("?"));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_concepts(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("concepts", q).await
    }

    /// Autocomplete for publishers. Searches display names. Returns up to 10
    /// results sorted by citation count. Hint shows country.
    ///
    /// `GET /autocomplete/publishers?q=...`
    ///
    /// Note: this endpoint has been observed returning HTTP 500 errors
    /// intermittently (server-side issue).
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_publishers("elsevier").await?;
    /// for result in &response.results {
    ///     println!("{} ({})", result.display_name,
    ///         result.hint.as_deref().unwrap_or(""));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_publishers(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("publishers", q).await
    }

    /// Autocomplete for subfields. Searches display names. Returns up to 10
    /// results sorted by citation count. Hint shows description.
    ///
    /// `GET /autocomplete/subfields?q=...`
    ///
    /// Note: autocomplete is only available for subfields, not domains or
    /// fields (`/autocomplete/domains` and `/autocomplete/fields` return 404).
    ///
    /// # Quirks
    ///
    /// The subfield autocomplete endpoint returns `entity_type: null` and
    /// `short_id: "Nones/..."` — these are known API quirks. The
    /// [`AutocompleteResult`](crate::AutocompleteResult) fields are
    /// `Option<String>` to handle this.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_subfields("artificial").await?;
    /// for result in &response.results {
    ///     println!("{} — {}", result.display_name,
    ///         result.hint.as_deref().unwrap_or(""));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_subfields(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("subfields", q).await
    }

    /// Autocomplete for funders. Searches display names. Returns up to 10
    /// results sorted by citation count. Hint shows country and description.
    ///
    /// `GET /autocomplete/funders?q=...`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::OpenAlexClient;
    ///
    /// let client = OpenAlexClient::new();
    /// let response = client.autocomplete_funders("national science").await?;
    /// for result in &response.results {
    ///     println!("{} ({})", result.display_name,
    ///         result.hint.as_deref().unwrap_or(""));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn autocomplete_funders(&self, q: &str) -> Result<AutocompleteResponse> {
        self.autocomplete_entity("funders", q).await
    }

    // ── Semantic search endpoints ──────────────────────────────────────

    /// Semantic search for works via GET. Sends query as a query parameter.
    /// Returns works ranked by AI similarity score (0-1). Maximum 10,000
    /// character query. **Requires API key. Costs 1,000 credits per request.**
    ///
    /// `GET /find/works?query=...`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, FindWorksParams};
    ///
    /// let client = OpenAlexClient::with_api_key("your-key");
    /// let params = FindWorksParams::builder()
    ///     .query("machine learning for drug discovery")
    ///     .count(5)
    ///     .build();
    /// let response = client.find_works(&params).await?;
    /// for result in &response.results {
    ///     println!("Score {:.2}: {}", result.score,
    ///         result.work.get("display_name")
    ///             .and_then(|v| v.as_str())
    ///             .unwrap_or("?"));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn find_works(&self, params: &FindWorksParams) -> Result<FindWorksResponse> {
        self.get_json("/find/works", params.to_query_pairs()).await
    }

    /// Semantic search for works via POST. Sends query in JSON body as
    /// `{"query": "..."}`, useful for long queries exceeding URL length limits.
    /// Same response format as [`find_works`](Self::find_works). **Requires API
    /// key. Costs 1,000 credits per request.**
    ///
    /// `POST /find/works`
    ///
    /// # Example
    ///
    /// ```no_run
    /// # async fn example() -> papers_openalex::Result<()> {
    /// use papers_openalex::{OpenAlexClient, FindWorksParams};
    ///
    /// let client = OpenAlexClient::with_api_key("your-key");
    /// let long_query = "A very long research question or abstract text...";
    /// let params = FindWorksParams::builder()
    ///     .query(long_query)
    ///     .count(10)
    ///     .filter("publication_year:>2020")
    ///     .build();
    /// let response = client.find_works_post(&params).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn find_works_post(&self, params: &FindWorksParams) -> Result<FindWorksResponse> {
        let body = serde_json::json!({ "query": params.query });
        self.post_json("/find/works", params.to_post_query_pairs(), body)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn minimal_list_json() -> String {
        r#"{
            "meta": {"count": 1, "db_response_time_ms": 10, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null},
            "results": [],
            "group_by": []
        }"#
        .to_string()
    }

    fn minimal_autocomplete_json() -> String {
        r#"{
            "meta": {"count": 1, "db_response_time_ms": 10, "page": 1, "per_page": 10},
            "results": [{
                "id": "https://openalex.org/test",
                "short_id": "test/T1",
                "display_name": "Test",
                "hint": "hint",
                "cited_by_count": 0,
                "works_count": 0,
                "entity_type": "work",
                "external_id": null,
                "filter_key": "openalex"
            }]
        }"#
        .to_string()
    }

    fn minimal_find_json() -> String {
        r#"{
            "meta": null,
            "results": []
        }"#
        .to_string()
    }

    async fn setup_client(server: &MockServer) -> OpenAlexClient {
        OpenAlexClient::new().with_base_url(server.uri())
    }

    // ── List endpoint tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_works() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.list_works(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_authors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/authors"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.list_authors(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_sources() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/sources"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.list_sources(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_institutions() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/institutions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client
            .list_institutions(&ListParams::default())
            .await
            .unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_topics() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/topics"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.list_topics(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_publishers() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/publishers"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client
            .list_publishers(&ListParams::default())
            .await
            .unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_funders() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/funders"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.list_funders(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_domains() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/domains"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.list_domains(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_fields() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/fields"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.list_fields(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_subfields() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/subfields"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client
            .list_subfields(&ListParams::default())
            .await
            .unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_with_all_params() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works"))
            .and(query_param("filter", "publication_year:2024"))
            .and(query_param("search", "machine learning"))
            .and(query_param("sort", "cited_by_count:desc"))
            .and(query_param("per-page", "10"))
            .and(query_param("page", "2"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let params = ListParams::builder()
            .filter("publication_year:2024")
            .search("machine learning")
            .sort("cited_by_count:desc")
            .per_page(10)
            .page(2)
            .build();
        let resp = client.list_works(&params).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_list_with_cursor() {
        let server = MockServer::start().await;
        let cursor_json = r#"{
            "meta": {"count": 1000, "db_response_time_ms": 10, "page": null, "per_page": 1, "next_cursor": "abc123", "groups_count": null},
            "results": [],
            "group_by": []
        }"#;
        Mock::given(method("GET"))
            .and(path("/works"))
            .and(query_param("cursor", "*"))
            .respond_with(ResponseTemplate::new(200).set_body_string(cursor_json))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let params = ListParams {
            cursor: Some("*".into()),
            ..Default::default()
        };
        let resp = client.list_works(&params).await.unwrap();
        assert!(resp.meta.page.is_none());
        assert_eq!(resp.meta.next_cursor.as_deref(), Some("abc123"));
    }

    #[tokio::test]
    async fn test_list_with_group_by() {
        let server = MockServer::start().await;
        let group_json = r#"{
            "meta": {"count": 1000, "db_response_time_ms": 10, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": 2},
            "results": [],
            "group_by": [
                {"key": "article", "key_display_name": "article", "count": 500},
                {"key": "preprint", "key_display_name": "preprint", "count": 300}
            ]
        }"#;
        Mock::given(method("GET"))
            .and(path("/works"))
            .and(query_param("group_by", "type"))
            .respond_with(ResponseTemplate::new(200).set_body_string(group_json))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let params = ListParams {
            group_by: Some("type".into()),
            ..Default::default()
        };
        let resp = client.list_works(&params).await.unwrap();
        assert_eq!(resp.meta.groups_count, Some(2));
        assert_eq!(resp.group_by.len(), 2);
        assert_eq!(resp.group_by[0].key, "article");
    }

    #[tokio::test]
    async fn test_list_with_sample_seed() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works"))
            .and(query_param("sample", "50"))
            .and(query_param("seed", "42"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let params = ListParams {
            sample: Some(50),
            seed: Some(42),
            ..Default::default()
        };
        client.list_works(&params).await.unwrap();
    }

    // ── Get endpoint tests ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_work() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works/W123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/W123"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let work = client.get_work("W123", &GetParams::default()).await.unwrap();
        assert_eq!(work.id, "https://openalex.org/W123");
    }

    #[tokio::test]
    async fn test_get_author() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/authors/A123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/A123"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let author = client
            .get_author("A123", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(author.id, "https://openalex.org/A123");
    }

    #[tokio::test]
    async fn test_get_source() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/sources/S123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/S123"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let source = client
            .get_source("S123", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(source.id, "https://openalex.org/S123");
    }

    #[tokio::test]
    async fn test_get_institution() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/institutions/I123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/I123"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let inst = client
            .get_institution("I123", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(inst.id, "https://openalex.org/I123");
    }

    #[tokio::test]
    async fn test_get_topic() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/topics/T123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/T123"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let topic = client
            .get_topic("T123", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(topic.id, "https://openalex.org/T123");
    }

    #[tokio::test]
    async fn test_get_publisher() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/publishers/P123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/P123"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let publisher = client
            .get_publisher("P123", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(publisher.id, "https://openalex.org/P123");
    }

    #[tokio::test]
    async fn test_get_funder() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/funders/F123"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/F123"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let funder = client
            .get_funder("F123", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(funder.id, "https://openalex.org/F123");
    }

    #[tokio::test]
    async fn test_get_domain() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/domains/3"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/domains/3"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let domain = client
            .get_domain("3", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(domain.id, "https://openalex.org/domains/3");
    }

    #[tokio::test]
    async fn test_get_field() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/fields/17"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/fields/17"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let field = client
            .get_field("17", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(field.id, "https://openalex.org/fields/17");
    }

    #[tokio::test]
    async fn test_get_subfield() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/subfields/1702"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/subfields/1702"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let subfield = client
            .get_subfield("1702", &GetParams::default())
            .await
            .unwrap();
        assert_eq!(subfield.id, "https://openalex.org/subfields/1702");
    }

    #[tokio::test]
    async fn test_get_with_select() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works/W123"))
            .and(query_param("select", "id,display_name"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(r#"{"id":"https://openalex.org/W123","display_name":"Test"}"#),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let params = GetParams::builder().select("id,display_name").build();
        let work = client.get_work("W123", &params).await.unwrap();
        assert_eq!(work.id, "https://openalex.org/W123");
    }

    // ── Autocomplete endpoint tests ────────────────────────────────────

    #[tokio::test]
    async fn test_autocomplete_works() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/works"))
            .and(query_param("q", "machine"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_works("machine").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_autocomplete_authors() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/authors"))
            .and(query_param("q", "einstein"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_authors("einstein").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_autocomplete_sources() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/sources"))
            .and(query_param("q", "nature"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_sources("nature").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_autocomplete_institutions() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/institutions"))
            .and(query_param("q", "harvard"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_institutions("harvard").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_autocomplete_concepts() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/concepts"))
            .and(query_param("q", "physics"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_concepts("physics").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_autocomplete_publishers() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/publishers"))
            .and(query_param("q", "elsevier"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_publishers("elsevier").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_autocomplete_funders() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/funders"))
            .and(query_param("q", "nsf"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_funders("nsf").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    #[tokio::test]
    async fn test_autocomplete_subfields() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/autocomplete/subfields"))
            .and(query_param("q", "artificial"))
            .respond_with(
                ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()),
            )
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let resp = client.autocomplete_subfields("artificial").await.unwrap();
        assert_eq!(resp.results.len(), 1);
    }

    // ── Find works tests ───────────────────────────────────────────────

    #[tokio::test]
    async fn test_find_works_get() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/find/works"))
            .and(query_param("query", "drug discovery"))
            .and(query_param("count", "5"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_find_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let params = FindWorksParams::builder()
            .query("drug discovery")
            .count(5)
            .build();
        let resp = client.find_works(&params).await.unwrap();
        assert!(resp.results.is_empty());
    }

    #[tokio::test]
    async fn test_find_works_post() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/find/works"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_find_json()))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let params = FindWorksParams::builder()
            .query("long query text")
            .build();
        let resp = client.find_works_post(&params).await.unwrap();
        assert!(resp.results.is_empty());
    }

    // ── API key and error tests ────────────────────────────────────────

    #[tokio::test]
    async fn test_api_key_sent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works"))
            .and(query_param("api_key", "test-key-123"))
            .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
            .mount(&server)
            .await;
        let client = OpenAlexClient::with_api_key("test-key-123").with_base_url(server.uri());
        let resp = client.list_works(&ListParams::default()).await.unwrap();
        assert_eq!(resp.meta.count, 1);
    }

    #[tokio::test]
    async fn test_error_404() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works/invalid"))
            .respond_with(ResponseTemplate::new(404).set_body_string("Not found"))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let err = client
            .get_work("invalid", &GetParams::default())
            .await
            .unwrap_err();
        match err {
            OpenAlexError::Api { status, message } => {
                assert_eq!(status, 404);
                assert_eq!(message, "Not found");
            }
            _ => panic!("Expected Api error, got {:?}", err),
        }
    }

    #[tokio::test]
    async fn test_error_403() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/works"))
            .respond_with(ResponseTemplate::new(403).set_body_string("Forbidden"))
            .mount(&server)
            .await;
        let client = setup_client(&server).await;
        let err = client
            .list_works(&ListParams::default())
            .await
            .unwrap_err();
        match err {
            OpenAlexError::Api { status, .. } => assert_eq!(status, 403),
            _ => panic!("Expected Api error"),
        }
    }
}
