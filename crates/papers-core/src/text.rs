pub use papers_datalab::ProcessingMode;
use papers_datalab::{DatalabClient, MarkerRequest, OutputFormat};
use papers_openalex::{GetParams, OpenAlexClient, Work};
use papers_zotero::{ItemListParams, ZoteroClient};
use serde::Serialize;
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

/// Try to find and download a PDF from Zotero (local storage first, then remote API).
pub async fn try_zotero(
    zotero: &ZoteroClient,
    doi: &str,
    title: Option<&str>,
) -> Result<Option<(Vec<u8>, PdfSource)>, WorkTextError> {
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

/// Extract text from PDF bytes, routing through DataLab if `datalab` is `Some`.
async fn do_extract(
    pdf_bytes: Vec<u8>,
    short_id: &str,
    datalab: Option<(&DatalabClient, ProcessingMode)>,
    source: &mut PdfSource,
) -> Result<String, WorkTextError> {
    if let Some((dl, mode)) = datalab {
        let dl_result = dl
            .convert_document(MarkerRequest {
                file: Some(pdf_bytes),
                filename: Some(format!("{}.pdf", short_id)),
                output_format: OutputFormat::Markdown,
                mode,
                ..Default::default()
            })
            .await?;
        *source = PdfSource::DataLab;
        Ok(dl_result.markdown.unwrap_or_default())
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
        if let Some((bytes, mut source)) = try_zotero(zotero, doi, title.as_deref()).await? {
            let text = do_extract(bytes, short_id, datalab, &mut source).await?;
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
        let text = do_extract(bytes, short_id, datalab, &mut source).await?;
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
        let text = do_extract(bytes, short_id, datalab, &mut source).await?;
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
        if let Some((bytes, source)) = try_zotero(zotero, doi, title).await? {
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
