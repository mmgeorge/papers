pub use papers_datalab::ProcessingMode;
use base64::Engine as _;
use papers_datalab::{DatalabClient, MarkerRequest, OutputFormat};
use papers_openalex::{GetParams, OpenAlexClient, Work};
use papers_zotero::{ItemListParams, ZoteroClient};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Where the PDF was obtained from.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PdfSource {
    ZoteroLocal { path: String },
    ZoteroRemote { item_key: String },
    DirectUrl { url: String },
    OpenAlexContent,
    DataLab,
}

/// Result of extracting text from a work's PDF.
#[derive(Debug, Clone, Serialize)]
pub struct WorkTextResult {
    pub text: String,
    pub source: PdfSource,
    pub work_id: String,
    pub title: Option<String>,
    pub doi: Option<String>,
}

/// Errors from the work_text pipeline.
#[derive(Debug, thiserror::Error)]
pub enum WorkTextError {
    #[error("OpenAlex error: {0}")]
    OpenAlex(#[from] papers_openalex::OpenAlexError),

    #[error("Filter error: {0}")]
    Filter(#[from] crate::filter::FilterError),

    #[error("Zotero error: {0}")]
    Zotero(#[from] papers_zotero::ZoteroError),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("PDF extraction error: {0}")]
    PdfExtract(String),

    #[error(transparent)]
    DataLab(#[from] papers_datalab::DatalabError),

    #[error("No PDF found for work {work_id}{}", title.as_ref().map(|t| format!(" ({})", t)).unwrap_or_default())]
    NoPdfFound {
        work_id: String,
        title: Option<String>,
        doi: Option<String>,
    },

    #[error("Invalid Zotero item key: {0}")]
    InvalidZoteroKey(String),
}

/// Whitelisted domains for direct PDF download.
const DIRECT_PDF_DOMAINS: &[&str] = &[
    "arxiv.org",
    "europepmc.org",
    "biorxiv.org",
    "medrxiv.org",
    "ncbi.nlm.nih.gov",
    "peerj.com",
    "mdpi.com",
    "frontiersin.org",
    "plos.org",
];

/// Extract text from PDF bytes using pdf-extract.
pub fn extract_text_bytes(pdf_bytes: &[u8]) -> Result<String, WorkTextError> {
    extract_text(pdf_bytes)
}

fn extract_text(pdf_bytes: &[u8]) -> Result<String, WorkTextError> {
    pdf_extract::extract_text_from_mem(pdf_bytes)
        .map_err(|e| WorkTextError::PdfExtract(e.to_string()))
}

/// Strip the `https://doi.org/` prefix from a DOI URL, returning the bare DOI.
fn bare_doi(doi: &str) -> &str {
    doi.strip_prefix("https://doi.org/").unwrap_or(doi)
}

/// Extract the short OpenAlex ID (e.g. `W12345`) from a full URL.
fn short_openalex_id(full_id: &str) -> &str {
    full_id
        .strip_prefix("https://openalex.org/")
        .unwrap_or(full_id)
}

/// Check if a URL's host matches one of the whitelisted domains.
fn is_whitelisted_url(url: &str) -> bool {
    DIRECT_PDF_DOMAINS
        .iter()
        .any(|domain| url.contains(domain))
}

/// Get the Zotero data directory path.
fn zotero_data_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("ZOTERO_DATA_DIR") {
        return Some(PathBuf::from(dir));
    }
    dirs::home_dir().map(|h| h.join("Zotero"))
}

fn datalab_cache_dir(short_id: &str) -> Option<PathBuf> {
    if let Ok(base) = std::env::var("PAPERS_DATALAB_CACHE_DIR") {
        return Some(PathBuf::from(base).join(short_id));
    }
    dirs::cache_dir().map(|d| d.join("papers").join("datalab").join(short_id))
}

/// Upload the local DataLab cache for `item_key` to Zotero as
/// `papers_extract_{item_key}.zip` attached to that same item.
///
/// The caller is responsible for ensuring the Zotero item exists before calling
/// this function (callers should check item existence and skip if absent).
///
/// Returns an error if there is no local cache for the key or the upload fails.
pub async fn upload_extraction_to_zotero(
    zc: &ZoteroClient,
    item_key: &str,
) -> Result<(), WorkTextError> {
    let dir = datalab_cache_dir(item_key)
        .ok_or_else(|| WorkTextError::PdfExtract("cannot determine cache directory".into()))?;
    if !dir.join(format!("{item_key}.md")).exists() {
        return Err(WorkTextError::PdfExtract(format!("no local cache for {item_key}")));
    }
    upload_papers_zip(zc, item_key, &dir, item_key).await
}

/// Download `papers_extract_{item_key}.zip` from Zotero (identified by `att_key`)
/// and restore it to the local cache directory.
///
/// `att_key` is the Zotero key of the attachment item itself (not the parent).
pub async fn download_extraction_from_zotero(
    zc: &ZoteroClient,
    att_key: &str,
    item_key: &str,
) -> Result<(), WorkTextError> {
    let zip_bytes = zc.download_item_file(att_key).await?;
    if zip_bytes.is_empty() {
        return Err(WorkTextError::PdfExtract(format!("empty download for {item_key}")));
    }
    let dir = datalab_cache_dir(item_key)
        .ok_or_else(|| WorkTextError::PdfExtract("cannot determine cache directory".into()))?;
    unzip_to_cache_dir(&zip_bytes, &dir).map_err(|e| WorkTextError::PdfExtract(e.to_string()))
}

/// Return the cached markdown for `cache_id` if it exists, otherwise `None`.
pub fn datalab_cached_markdown(cache_id: &str) -> Option<String> {
    let dir = datalab_cache_dir(cache_id)?;
    std::fs::read_to_string(dir.join(format!("{cache_id}.md"))).ok()
}

/// Return the keys of all locally cached DataLab extractions.
///
/// Scans the DataLab cache base directory and returns the name of every
/// subdirectory that contains a `{key}.md` file.
pub fn datalab_cached_item_keys() -> Vec<String> {
    let base = if let Ok(base_str) = std::env::var("PAPERS_DATALAB_CACHE_DIR") {
        PathBuf::from(base_str)
    } else {
        match dirs::cache_dir() {
            Some(d) => d.join("papers").join("datalab"),
            None => return vec![],
        }
    };
    if !base.is_dir() {
        return vec![];
    }
    let mut keys = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            let key = match entry.file_name().to_str() {
                Some(k) => k.to_string(),
                None => continue,
            };
            if entry.path().join(format!("{key}.md")).exists() {
                keys.push(key);
            }
        }
    }
    keys
}

/// Return the cached JSON for `cache_id` if it exists, otherwise `None`.
pub fn datalab_cached_json(cache_id: &str) -> Option<String> {
    let dir = datalab_cache_dir(cache_id)?;
    std::fs::read_to_string(dir.join(format!("{cache_id}.json"))).ok()
}

/// Return the local cache directory path for `cache_id` if determinable.
pub fn datalab_cache_dir_path(cache_id: &str) -> Option<std::path::PathBuf> {
    datalab_cache_dir(cache_id)
}

/// Return the directory where the open-access PDF for `doi` would be cached.
///
/// Uses `PAPERS_DATALAB_CACHE_DIR` override when set, otherwise
/// `{cache_dir}/papers/doi/{sanitized_doi}/`.
/// Slashes in the DOI are replaced with `_` to form a valid directory name.
pub fn doi_pdf_cache_dir(doi: &str) -> Option<PathBuf> {
    // Strip any doi: / https://doi.org/ prefix first
    let bare = doi
        .strip_prefix("https://doi.org/")
        .or_else(|| doi.strip_prefix("http://doi.org/"))
        .or_else(|| doi.strip_prefix("doi:"))
        .unwrap_or(doi);
    let safe = bare.replace('/', "_");
    if let Ok(base) = std::env::var("PAPERS_DATALAB_CACHE_DIR") {
        return Some(PathBuf::from(base).join("doi").join(safe));
    }
    dirs::cache_dir().map(|d| d.join("papers").join("doi").join(safe))
}

/// Return `true` if a PDF file exists in the DOI cache for `doi`.
pub fn doi_pdf_cached(doi: &str) -> bool {
    let dir = match doi_pdf_cache_dir(doi) {
        Some(d) => d,
        None => return false,
    };
    if !dir.is_dir() {
        return false;
    }
    std::fs::read_dir(&dir)
        .ok()
        .into_iter()
        .flatten()
        .any(|e| {
            e.ok()
                .map(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .map(|x| x.eq_ignore_ascii_case("pdf"))
                        .unwrap_or(false)
                })
                .unwrap_or(false)
        })
}

/// Try to download an open-access PDF for `work` from direct URLs or OpenAlex
/// Content API, without Zotero involvement.
///
/// Returns `Ok(Some((bytes, source)))` if a PDF was found, `Ok(None)` if none
/// of the known OA locations yielded a PDF, or an error on network failure.
pub async fn try_download_open_access_pdf(
    http: &reqwest::Client,
    work: &Work,
) -> Result<Option<(Vec<u8>, PdfSource)>, WorkTextError> {
    let urls = collect_pdf_urls(work);
    if let Some(result) = try_direct_urls(http, &urls).await? {
        return Ok(Some(result));
    }
    try_openalex_content(http, work).await
}

/// Metadata written alongside each DataLab extraction cache entry as `meta.json`.
///
/// All fields except `item_key` are `Option` so that the struct can be read
/// from older cache entries that may be missing fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionMeta {
    pub item_key: String,
    pub zotero_user_id: Option<String>,
    pub title: Option<String>,
    pub authors: Option<Vec<String>>,
    pub item_type: Option<String>,
    pub date: Option<String>,
    pub doi: Option<String>,
    pub url: Option<String>,
    pub publication_title: Option<String>,
    pub extracted_at: Option<String>,
    pub processing_mode: Option<String>,
    pub pdf_source: Option<serde_json::Value>,
}

/// Read the `meta.json` for `cache_id` from the local DataLab cache, if present.
pub fn read_extraction_meta(cache_id: &str) -> Option<ExtractionMeta> {
    let dir = datalab_cache_dir(cache_id)?;
    let bytes = std::fs::read(dir.join("meta.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Return an ISO 8601 UTC timestamp for the current moment (no external deps).
fn iso_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Civil date from unix epoch seconds using Hinnant's algorithm.
    let days = (secs / 86400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    let hh = (secs % 86_400) / 3_600;
    let mm = (secs % 3_600) / 60;
    let ss = secs % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Write `meta.json` into `dir` for the given extraction.
///
/// Fetches Zotero item metadata when `zotero` is provided (best-effort; never
/// returns an error — failures are silently ignored).
async fn write_extraction_meta(
    dir: &std::path::Path,
    item_key: &str,
    zotero: Option<&ZoteroClient>,
    mode_str: Option<&str>,
    pdf_source: Option<&PdfSource>,
) {
    let mut meta = ExtractionMeta {
        item_key: item_key.to_string(),
        zotero_user_id: std::env::var("ZOTERO_USER_ID").ok(),
        title: None,
        authors: None,
        item_type: None,
        date: None,
        doi: None,
        url: None,
        publication_title: None,
        extracted_at: Some(iso_now()),
        processing_mode: mode_str.map(String::from),
        pdf_source: pdf_source.and_then(|s| serde_json::to_value(s).ok()),
    };

    if let Some(zc) = zotero {
        if let Ok(item) = zc.get_item(item_key).await {
            meta.title = item.data.title;
            meta.item_type = Some(item.data.item_type);
            meta.date = item.data.date;
            meta.doi = item.data.doi;
            meta.url = item.data.url;
            meta.publication_title = item.data.publication_title;
            let authors: Vec<String> = item
                .data
                .creators
                .iter()
                .filter(|c| c.creator_type == "author")
                .map(|c| match (&c.first_name, &c.last_name, &c.name) {
                    (Some(f), Some(l), _) if !l.is_empty() => format!("{f} {l}"),
                    (_, Some(l), _) if !l.is_empty() => l.clone(),
                    (_, _, Some(n)) if !n.is_empty() => n.clone(),
                    _ => String::new(),
                })
                .filter(|s| !s.is_empty())
                .collect();
            if !authors.is_empty() {
                meta.authors = Some(authors);
            }
        }
    }

    if let Ok(json) = serde_json::to_string_pretty(&meta) {
        let _ = std::fs::write(dir.join("meta.json"), json);
    }
}

/// Collect all pdf_url values from an OpenAlex Work's locations.
fn collect_pdf_urls(work: &Work) -> Vec<String> {
    let mut urls = Vec::new();

    if let Some(loc) = &work.best_oa_location {
        if let Some(url) = &loc.pdf_url {
            urls.push(url.clone());
        }
    }
    if let Some(loc) = &work.primary_location {
        if let Some(url) = &loc.pdf_url {
            if !urls.contains(url) {
                urls.push(url.clone());
            }
        }
    }
    if let Some(locations) = &work.locations {
        for loc in locations {
            if let Some(url) = &loc.pdf_url {
                if !urls.contains(url) {
                    urls.push(url.clone());
                }
            }
        }
    }

    urls
}

/// Brief Zotero library info for a work matched by DOI.
#[derive(Debug, Clone, Serialize)]
pub struct ZoteroItemInfo {
    pub key: String,
    pub item_type: String,
    pub tags: Vec<String>,
    pub has_pdf: bool,
    pub date_added: Option<String>,
    pub uri: String,
}

/// Check if a work exists in the Zotero library, matched by DOI.
///
/// Returns `Ok(Some(...))` with brief metadata if found, `Ok(None)` if the
/// work has no DOI or is not in the library, or an error on API failure.
pub async fn find_work_in_zotero(
    zotero: &ZoteroClient,
    work: &papers_openalex::Work,
) -> Result<Option<ZoteroItemInfo>, papers_zotero::ZoteroError> {
    let doi = match &work.doi {
        Some(d) => bare_doi(d),
        None => return Ok(None),
    };
    let title = work.display_name.as_deref().or(work.title.as_deref());

    // Search by title using the default q mode (title/creator/year only — fast).
    // qmode("everything") would search full-text of attached PDFs, which is very slow.
    // DOI validation is done below on the returned item's metadata, not via full-text search.
    let t_search = std::time::Instant::now();
    let items: Vec<papers_zotero::Item> = if let Some(t) = title {
        let title_params = ItemListParams::builder().q(t).build();
        let res = zotero.list_top_items(&title_params).await?;
        eprintln!("[timing] zotero title search ({} results): {:?}", res.items.len(), t_search.elapsed());
        res.items
    } else {
        eprintln!("[timing] zotero: no title, skipping search");
        return Ok(None);
    };

    for item in &items {
        let item_doi = match &item.data.doi {
            Some(d) => d,
            None => continue,
        };
        if !item_doi.eq_ignore_ascii_case(doi) {
            continue;
        }

        let t_children = std::time::Instant::now();
        let children = zotero
            .list_item_children(&item.key, &ItemListParams::default())
            .await?;
        eprintln!("[timing] zotero list_item_children: {:?}", t_children.elapsed());
        let has_pdf = children.items.iter().any(|child| {
            child.data.content_type.as_deref() == Some("application/pdf")
                && matches!(
                    child.data.link_mode.as_deref(),
                    Some("imported_file" | "imported_url")
                )
        });

        let tags: Vec<String> = item.data.tags.iter().map(|t| t.tag.clone()).collect();
        let uri = format!("zotero://select/library/items/{}", item.key);
        return Ok(Some(ZoteroItemInfo {
            key: item.key.clone(),
            item_type: item.data.item_type.clone(),
            tags,
            has_pdf,
            date_added: item.data.date_added.clone(),
            uri,
        }));
    }

    Ok(None)
}

/// Try to find and download a PDF from Zotero (local storage first, then remote API).
///
/// Returns `(pdf_bytes, source, zotero_item_key)` where `zotero_item_key` is the
/// parent bibliographic item key (e.g. `U9PRIZJ7`), suitable for use as a cache ID.
pub async fn try_zotero(
    zotero: &ZoteroClient,
    doi: &str,
    title: Option<&str>,
) -> Result<Option<(Vec<u8>, PdfSource, String)>, WorkTextError> {
    // Zotero API's `q` parameter only searches title, creator, year, and full-text
    // content — it does NOT search metadata fields like DOI (per Zotero docs:
    // "Searching of other fields will be possible in the future").
    // Search by title first, then fall back to DOI (which may match full-text content).
    let mut candidate_queries: Vec<String> = Vec::new();
    if let Some(t) = title {
        candidate_queries.push(t.to_string());
    }
    candidate_queries.push(doi.to_string());

    for query in &candidate_queries {
        let params = ItemListParams::builder()
            .q(query.as_str())
            .qmode("everything")
            .build();

        let results = zotero.list_top_items(&params).await?;
        if results.items.is_empty() {
            continue;
        }

        for item in &results.items {
            // Check that this item's DOI actually matches
            let item_doi = match &item.data.doi {
                Some(d) => d,
                None => continue,
            };
            if !item_doi.eq_ignore_ascii_case(doi) {
                continue;
            }

        // Get children to find PDF attachment
        let children = zotero
            .list_item_children(&item.key, &ItemListParams::default())
            .await?;

        for child in &children.items {
            let is_pdf = child
                .data
                .content_type
                .as_deref()
                == Some("application/pdf");
            let has_local_file = matches!(
                child.data.link_mode.as_deref(),
                Some("imported_file" | "imported_url")
            );

            if !is_pdf || !has_local_file {
                continue;
            }

            // Try local file first
            if let Some(filename) = &child.data.filename {
                if let Some(data_dir) = zotero_data_dir() {
                    let local_path = data_dir
                        .join("storage")
                        .join(&child.key)
                        .join(filename);
                    if local_path.exists() {
                        let bytes = tokio::fs::read(&local_path)
                            .await
                            .map_err(|e| WorkTextError::PdfExtract(format!("Failed to read local file: {e}")))?;
                        return Ok(Some((
                            bytes,
                            PdfSource::ZoteroLocal {
                                path: local_path.to_string_lossy().into_owned(),
                            },
                            item.key.clone(),
                        )));
                    }
                }
            }

            // Try remote download
            match zotero.download_item_file(&child.key).await {
                Ok(bytes) if !bytes.is_empty() => {
                    return Ok(Some((
                        bytes,
                        PdfSource::ZoteroRemote {
                            item_key: child.key.clone(),
                        },
                        item.key.clone(),
                    )));
                }
                _ => continue,
            }
        }
        }
    }

    Ok(None)
}

/// Try downloading a PDF from direct URLs (whitelisted domains only).
async fn try_direct_urls(
    http: &reqwest::Client,
    urls: &[String],
) -> Result<Option<(Vec<u8>, PdfSource)>, WorkTextError> {
    for url in urls {
        if !is_whitelisted_url(url) {
            continue;
        }

        let resp = http
            .get(url)
            .header(
                "User-Agent",
                "papers-mcp/0.1 (https://github.com/mmgeorge/papers; mailto:papers@example.com)",
            )
            .send()
            .await;

        let resp = match resp {
            Ok(r) if r.status().is_success() => r,
            _ => continue,
        };

        // Verify content type
        let is_pdf = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("application/pdf"));

        if !is_pdf {
            continue;
        }

        let bytes = resp.bytes().await?.to_vec();
        if !bytes.is_empty() {
            return Ok(Some((
                bytes,
                PdfSource::DirectUrl { url: url.clone() },
            )));
        }
    }

    Ok(None)
}

/// Try downloading from the OpenAlex Content API.
async fn try_openalex_content(
    http: &reqwest::Client,
    work: &Work,
) -> Result<Option<(Vec<u8>, PdfSource)>, WorkTextError> {
    let has_pdf = work
        .has_content
        .as_ref()
        .and_then(|hc| hc.pdf)
        .unwrap_or(false);

    if !has_pdf {
        return Ok(None);
    }

    let api_key = match std::env::var("OPENALEX_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => return Ok(None),
    };

    let short_id = short_openalex_id(&work.id);
    let url = format!(
        "https://content.openalex.org/works/{}.pdf?api_key={}",
        short_id, api_key
    );

    let resp = http.get(&url).send().await;

    let resp = match resp {
        Ok(r) if r.status().is_success() => r,
        _ => return Ok(None),
    };

    let bytes = resp.bytes().await?.to_vec();
    if !bytes.is_empty() {
        return Ok(Some((bytes, PdfSource::OpenAlexContent)));
    }

    Ok(None)
}

/// Returns true if `key` is a valid Zotero item key (8 ASCII uppercase letters or digits).
fn is_valid_zotero_key(key: &str) -> bool {
    key.len() == 8 && key.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

/// Returns true if the error is a 403 Forbidden from the Zotero API.
/// Used to silently skip upload attempts when only a read-only API key is available.
fn is_zotero_write_denied(e: &WorkTextError) -> bool {
    matches!(
        e,
        WorkTextError::Zotero(papers_zotero::ZoteroError::Api { status: 403, .. })
    )
}

/// The filename used for DataLab extraction backups stored in Zotero.
/// The parent item key is embedded so `extract list` can find all extractions
/// with a single `q=papers_extract` query instead of paginating all attachments.
fn papers_extract_filename(parent_key: &str) -> String {
    format!("papers_extract_{parent_key}.zip")
}

/// Find the attachment key of the `papers_extract_{parent_key}.zip` child on `parent_key`.
pub async fn find_papers_zip_key(
    zc: &ZoteroClient,
    parent_key: &str,
) -> Result<Option<String>, WorkTextError> {
    let expected = papers_extract_filename(parent_key);
    let children = zc
        .list_item_children(parent_key, &ItemListParams::default())
        .await?;
    for child in &children.items {
        if child.data.filename.as_deref() == Some(&expected)
            && child.data.link_mode.as_deref() == Some("imported_file")
        {
            return Ok(Some(child.key.clone()));
        }
    }
    Ok(None)
}

/// Create an in-memory ZIP of the DataLab cache directory for `id`.
fn zip_cache_dir(dir: &std::path::Path, id: &str) -> std::io::Result<Vec<u8>> {
    use std::io::Write as _;
    let buf = Vec::new();
    let cursor = std::io::Cursor::new(buf);
    let mut zip = zip::ZipWriter::new(cursor);
    let opts = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // Add {id}.md
    let md_path = dir.join(format!("{id}.md"));
    if md_path.exists() {
        zip.start_file(format!("{id}.md"), opts)?;
        zip.write_all(&std::fs::read(&md_path)?)?;
    }

    // Add {id}.json (if present)
    let json_path = dir.join(format!("{id}.json"));
    if json_path.exists() {
        zip.start_file(format!("{id}.json"), opts)?;
        zip.write_all(&std::fs::read(&json_path)?)?;
    }

    // Add meta.json (if present)
    let meta_path = dir.join("meta.json");
    if meta_path.exists() {
        zip.start_file("meta.json", opts)?;
        zip.write_all(&std::fs::read(&meta_path)?)?;
    }

    // Add images/ (if present)
    let img_dir = dir.join("images");
    if img_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&img_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        zip.start_file(format!("images/{name}"), opts)?;
                        zip.write_all(&std::fs::read(&path)?)?;
                    }
                }
            }
        }
    }

    let cursor = zip.finish()?;
    Ok(cursor.into_inner())
}

/// Extract a ZIP archive into `dir`, creating it first if needed.
fn unzip_to_cache_dir(zip_bytes: &[u8], dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let cursor = std::io::Cursor::new(zip_bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let out_path = dir.join(file.name());
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = std::fs::File::create(&out_path)?;
        std::io::copy(&mut file, &mut out)?;
    }
    Ok(())
}

/// Best-effort: zip the cache dir for `id` and upload it as `papers_extract_{id}.zip` under `parent_key`.
async fn upload_papers_zip(
    zc: &ZoteroClient,
    parent_key: &str,
    dir: &std::path::Path,
    id: &str,
) -> Result<(), WorkTextError> {
    let filename = papers_extract_filename(id);
    let zip_bytes = zip_cache_dir(dir, id).map_err(|e| WorkTextError::PdfExtract(e.to_string()))?;
    let att_key = zc
        .create_imported_attachment(parent_key, &filename, "application/zip")
        .await?;
    zc.upload_attachment_file(&att_key, &filename, zip_bytes)
        .await?;
    Ok(())
}

/// Extract text from PDF bytes, routing through DataLab if `datalab` is `Some`.
///
/// `zotero_id` is the Zotero parent item key (or OpenAlex short ID for non-Zotero sources)
/// used as the on-disk cache ID. When `zotero` is `Some`, the DataLab result is also
/// backed up to/restored from a `papers_extract_{key}.zip` attachment on the parent Zotero item.
pub async fn do_extract(
    pdf_bytes: Vec<u8>,
    zotero_id: &str,
    zotero: Option<&ZoteroClient>,
    datalab: Option<(&DatalabClient, ProcessingMode)>,
    source: &mut PdfSource,
) -> Result<String, WorkTextError> {
    if let Some((dl, mode)) = datalab {
        // Validate key if Zotero sync is requested
        if let Some(zc) = zotero {
            if !is_valid_zotero_key(zotero_id) {
                return Err(WorkTextError::InvalidZoteroKey(zotero_id.to_string()));
            }
            let _ = zc; // used below
        }

        let cache_dir = datalab_cache_dir(zotero_id);

        // --- local cache check ---
        if let Some(ref dir) = cache_dir {
            let md_path = dir.join(format!("{zotero_id}.md"));
            if let Ok(text) = std::fs::read_to_string(&md_path) {
                *source = PdfSource::DataLab;
                // Best-effort: upload to Zotero if no papers_extract_*.zip exists yet
                if let Some(zc) = zotero {
                    let zc = zc.clone();
                    let dir = dir.clone();
                    let id = zotero_id.to_string();
                    tokio::spawn(async move {
                        match find_papers_zip_key(&zc, &id).await {
                            Ok(None) => {
                                if let Err(e) = upload_papers_zip(&zc, &id, &dir, &id).await {
                                    if !is_zotero_write_denied(&e) {
                                        eprintln!("[papers] Zotero backup upload failed: {e}");
                                    }
                                }
                            }
                            Ok(Some(_)) => {} // already present
                            Err(e) => {
                                if !is_zotero_write_denied(&e) {
                                    eprintln!("[papers] Zotero children check failed: {e}");
                                }
                            }
                        }
                    });
                }
                return Ok(text);
            }
        }

        // --- Zotero cache check (papers_extract_*.zip) ---
        if let Some(zc) = zotero {
            if let Ok(Some(att_key)) = find_papers_zip_key(zc, zotero_id).await {
                match zc.download_item_file(&att_key).await {
                    Ok(zip_bytes) if !zip_bytes.is_empty() => {
                        if let Some(ref dir) = cache_dir {
                            if unzip_to_cache_dir(&zip_bytes, dir).is_ok() {
                                let md_path = dir.join(format!("{zotero_id}.md"));
                                if let Ok(text) = std::fs::read_to_string(&md_path) {
                                    *source = PdfSource::DataLab;
                                    return Ok(text);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        return Err(WorkTextError::Zotero(e));
                    }
                    _ => {}
                }
            }
        }

        // --- DataLab API call ---
        // Capture mode string and original source before they are moved/overwritten.
        let mode_str_opt = serde_json::to_value(&mode)
            .ok()
            .and_then(|v| v.as_str().map(String::from));
        let original_source = source.clone();
        let dl_result = dl
            .convert_document(MarkerRequest {
                file: Some(pdf_bytes),
                filename: Some(format!("{zotero_id}.pdf")),
                output_format: vec![OutputFormat::Markdown, OutputFormat::Json],
                mode,
                ..Default::default()
            })
            .await?;

        *source = PdfSource::DataLab;
        let markdown = dl_result.markdown.clone().unwrap_or_default();

        // --- write local cache (best-effort) ---
        if let Some(ref dir) = cache_dir {
            let _ = std::fs::create_dir_all(dir);

            let md_path = dir.join(format!("{zotero_id}.md"));
            let _ = std::fs::write(&md_path, &markdown);

            if let Some(ref json_val) = dl_result.json {
                let json_path = dir.join(format!("{zotero_id}.json"));
                let _ = std::fs::write(&json_path, json_val.to_string());
            }

            if let Some(ref images) = dl_result.images {
                if !images.is_empty() {
                    let img_dir = dir.join("images");
                    let _ = std::fs::create_dir_all(&img_dir);
                    for (name, data) in images {
                        let b64 = if let Some(pos) = data.find(";base64,") {
                            &data[pos + 8..]
                        } else {
                            data.as_str()
                        };
                        if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                            let img_path = img_dir.join(name);
                            let _ = std::fs::write(&img_path, bytes);
                        }
                    }
                }
            }

            // Write meta.json (best-effort, before upload so it's included in the ZIP)
            write_extraction_meta(
                dir,
                zotero_id,
                zotero,
                mode_str_opt.as_deref(),
                Some(&original_source),
            )
            .await;

            // Best-effort: upload papers_extract_*.zip to Zotero (silently skip on 403)
            if let Some(zc) = zotero {
                if let Err(e) = upload_papers_zip(zc, zotero_id, dir, zotero_id).await {
                    if !is_zotero_write_denied(&e) {
                        eprintln!("[papers] Zotero backup upload failed: {e}");
                    }
                }
            }
        }

        Ok(markdown)
    } else {
        extract_text(&pdf_bytes)
    }
}

/// Download and extract the full text of a scholarly work.
///
/// Tries multiple sources in priority order:
/// 1. Local Zotero storage (filesystem)
/// 2. Remote Zotero API (if credentials available)
/// 3. Direct PDF URLs from OpenAlex locations (whitelisted domains)
/// 4. OpenAlex Content API (requires `OPENALEX_API_KEY`)
///
/// When `datalab` is `Some`, the final extraction step uses the DataLab Marker
/// API instead of local pdfium extraction, producing higher-quality markdown.
/// The `ProcessingMode` controls quality vs. speed: `Fast` < `Balanced` < `Accurate`.
pub async fn work_text(
    openalex: &OpenAlexClient,
    zotero: Option<&ZoteroClient>,
    datalab: Option<(&DatalabClient, ProcessingMode)>,
    work_id: &str,
) -> Result<WorkTextResult, WorkTextError> {
    // 1. Fetch work metadata from OpenAlex
    let work = crate::api::work_get(openalex, work_id, &GetParams::default()).await?;

    let title = work.title.clone().or_else(|| work.display_name.clone());
    let doi_raw = work.doi.as_deref();
    let doi = doi_raw.map(bare_doi);
    let short_id = short_openalex_id(&work.id);

    let http = reqwest::Client::new();

    // 2. Try Zotero (local then remote)
    if let (Some(zotero), Some(doi)) = (zotero, doi) {
        if let Some((bytes, mut source, zotero_key)) = try_zotero(zotero, doi, title.as_deref()).await? {
            let text = do_extract(bytes, &zotero_key, Some(zotero), datalab, &mut source).await?;
            return Ok(WorkTextResult {
                text,
                source,
                work_id: work.id.clone(),
                title,
                doi: doi_raw.map(String::from),
            });
        }
    }

    // 3. Try direct PDF URLs from OpenAlex locations
    let pdf_urls = collect_pdf_urls(&work);
    if let Some((bytes, mut source)) = try_direct_urls(&http, &pdf_urls).await? {
        let text = do_extract(bytes, short_id, None, datalab, &mut source).await?;
        return Ok(WorkTextResult {
            text,
            source,
            work_id: work.id.clone(),
            title,
            doi: doi_raw.map(String::from),
        });
    }

    // 4. Try OpenAlex Content API
    if let Some((bytes, mut source)) = try_openalex_content(&http, &work).await? {
        let text = do_extract(bytes, short_id, None, datalab, &mut source).await?;
        return Ok(WorkTextResult {
            text,
            source,
            work_id: work.id.clone(),
            title,
            doi: doi_raw.map(String::from),
        });
    }

    // 5. No PDF found
    Err(WorkTextError::NoPdfFound {
        work_id: work.id.clone(),
        title,
        doi: doi_raw.map(String::from),
    })
}

/// Poll Zotero for a work by DOI. Waits 5s initially, then polls every 2s for up to ~2 min.
///
/// This is used by callers (CLI prompt, MCP elicitation) after asking the user to add a paper
/// to Zotero. Returns the extracted text if the paper appears in Zotero within the timeout.
pub async fn poll_zotero_for_work(
    zotero: &ZoteroClient,
    work_id: &str,
    title: Option<&str>,
    doi: &str,
) -> Result<WorkTextResult, WorkTextError> {
    // Initial wait to give user time to save
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    for _ in 0..55 {
        if let Some((bytes, source, _zotero_key)) = try_zotero(zotero, doi, title).await? {
            let text = extract_text(&bytes)?;
            return Ok(WorkTextResult {
                text,
                source,
                work_id: work_id.to_string(),
                title: title.map(String::from),
                doi: Some(doi.to_string()),
            });
        }
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }

    Err(WorkTextError::NoPdfFound {
        work_id: work_id.to_string(),
        title: title.map(String::from),
        doi: Some(doi.to_string()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bare_doi() {
        assert_eq!(bare_doi("https://doi.org/10.1234/test"), "10.1234/test");
        assert_eq!(bare_doi("10.1234/test"), "10.1234/test");
    }

    #[test]
    fn test_short_openalex_id() {
        assert_eq!(
            short_openalex_id("https://openalex.org/W2741809807"),
            "W2741809807"
        );
        assert_eq!(short_openalex_id("W2741809807"), "W2741809807");
    }

    #[test]
    fn test_is_whitelisted_url() {
        assert!(is_whitelisted_url("https://arxiv.org/pdf/2301.12345"));
        assert!(is_whitelisted_url(
            "https://europepmc.org/articles/PMC123/pdf"
        ));
        assert!(is_whitelisted_url("https://www.biorxiv.org/content/pdf"));
        assert!(is_whitelisted_url("https://www.mdpi.com/some/pdf"));
        assert!(!is_whitelisted_url("https://evil.com/pdf"));
        assert!(!is_whitelisted_url("https://publisher.com/paper.pdf"));
    }

    #[test]
    fn test_collect_pdf_urls_empty() {
        let work: Work = serde_json::from_str(r#"{"id": "https://openalex.org/W1"}"#).unwrap();
        assert!(collect_pdf_urls(&work).is_empty());
    }

    #[test]
    fn test_collect_pdf_urls_deduplicates() {
        let work: Work = serde_json::from_value(serde_json::json!({
            "id": "https://openalex.org/W1",
            "best_oa_location": { "pdf_url": "https://arxiv.org/pdf/1234" },
            "primary_location": { "pdf_url": "https://arxiv.org/pdf/1234" },
            "locations": [
                { "pdf_url": "https://arxiv.org/pdf/1234" },
                { "pdf_url": "https://europepmc.org/pdf/5678" }
            ]
        }))
        .unwrap();
        let urls = collect_pdf_urls(&work);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0], "https://arxiv.org/pdf/1234");
        assert_eq!(urls[1], "https://europepmc.org/pdf/5678");
    }
}
