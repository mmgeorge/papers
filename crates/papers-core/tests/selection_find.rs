//! Tests for DOI PDF cache utilities and open-access PDF download.
//!
//! These tests cover:
//! - `doi_pdf_cache_dir`: DOI sanitization and base directory resolution
//! - `doi_pdf_cached`: filesystem check for cached PDFs
//! - `try_download_open_access_pdf`: HTTP download via whitelisted direct URLs

use papers_core::text::{doi_pdf_cache_dir, doi_pdf_cached, try_download_open_access_pdf};
use serial_test::serial;
use std::fs;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Redirect `PAPERS_DATALAB_CACHE_DIR` to an isolated temp dir.
fn isolated_cache(dir: &TempDir) {
    unsafe { std::env::set_var("PAPERS_DATALAB_CACHE_DIR", dir.path()) };
}

/// Construct a minimal `papers_openalex::Work` with a `best_oa_location.pdf_url`.
fn work_with_pdf_url(id: &str, pdf_url: &str) -> papers_openalex::Work {
    serde_json::from_value(serde_json::json!({
        "id": format!("https://openalex.org/{id}"),
        "best_oa_location": {
            "pdf_url": pdf_url,
            "is_oa": true
        }
    }))
    .expect("valid work json")
}

/// Construct a minimal `papers_openalex::Work` with no OA locations.
fn work_no_oa(id: &str) -> papers_openalex::Work {
    serde_json::from_value(serde_json::json!({
        "id": format!("https://openalex.org/{id}")
    }))
    .expect("valid work json")
}

// ── doi_pdf_cache_dir ────────────────────────────────────────────────────────

#[test]
#[serial]
fn doi_pdf_cache_dir_sanitizes_slash() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let result = doi_pdf_cache_dir("10.1145/3290605.3300357").unwrap();
    let last = result.file_name().unwrap().to_str().unwrap();
    // Slashes replaced with underscores
    assert_eq!(last, "10.1145_3290605.3300357");
    assert!(result.starts_with(dir.path().join("doi")));
}

#[test]
#[serial]
fn doi_pdf_cache_dir_strips_https_prefix() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let with_prefix = doi_pdf_cache_dir("https://doi.org/10.1234/foo").unwrap();
    let bare = doi_pdf_cache_dir("10.1234/foo").unwrap();
    assert_eq!(with_prefix, bare);
}

#[test]
#[serial]
fn doi_pdf_cache_dir_strips_doi_colon_prefix() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let with_prefix = doi_pdf_cache_dir("doi:10.5678/bar").unwrap();
    let bare = doi_pdf_cache_dir("10.5678/bar").unwrap();
    assert_eq!(with_prefix, bare);
}

#[test]
#[serial]
fn doi_pdf_cache_dir_uses_env_override() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let result = doi_pdf_cache_dir("10.1/x").unwrap();
    // Should be under the overridden dir, not the system cache
    assert!(result.starts_with(dir.path()));
}

// ── doi_pdf_cached ────────────────────────────────────────────────────────────

#[test]
#[serial]
fn doi_pdf_cached_returns_true_when_pdf_exists() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let doi = "10.9999/test";
    let cache_dir = doi_pdf_cache_dir(doi).unwrap();
    fs::create_dir_all(&cache_dir).unwrap();
    fs::write(cache_dir.join("paper.pdf"), b"%PDF-1.4 fake content").unwrap();
    assert!(doi_pdf_cached(doi));
}

#[test]
#[serial]
fn doi_pdf_cached_returns_false_when_dir_empty() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let doi = "10.9998/empty";
    let cache_dir = doi_pdf_cache_dir(doi).unwrap();
    fs::create_dir_all(&cache_dir).unwrap();
    // Directory exists but no PDF
    assert!(!doi_pdf_cached(doi));
}

#[test]
#[serial]
fn doi_pdf_cached_returns_false_when_dir_missing() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    // Never created the directory
    assert!(!doi_pdf_cached("10.9997/missing"));
}

#[test]
#[serial]
fn doi_pdf_cached_returns_false_for_non_pdf_files() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let doi = "10.9996/txt-only";
    let cache_dir = doi_pdf_cache_dir(doi).unwrap();
    fs::create_dir_all(&cache_dir).unwrap();
    // A .txt file does not count
    fs::write(cache_dir.join("paper.txt"), b"not a pdf").unwrap();
    assert!(!doi_pdf_cached(doi));
}

#[test]
#[serial]
fn doi_pdf_cached_case_insensitive_extension() {
    let dir = TempDir::new().unwrap();
    isolated_cache(&dir);
    let doi = "10.9995/caps-ext";
    let cache_dir = doi_pdf_cache_dir(doi).unwrap();
    fs::create_dir_all(&cache_dir).unwrap();
    // .PDF (uppercase) should also count
    fs::write(cache_dir.join("paper.PDF"), b"%PDF-1.4 fake").unwrap();
    assert!(doi_pdf_cached(doi));
}

// ── try_download_open_access_pdf ─────────────────────────────────────────────

/// The whitelist check is `url.contains(domain)`, so a mock URL whose path
/// includes a whitelisted domain string (e.g. "arxiv.org") passes the check.
/// This lets us point the Work at our wiremock server for testing.

#[tokio::test]
#[serial]
async fn try_download_oa_pdf_returns_bytes_on_success() {
    let mock = MockServer::start().await;
    let fake_pdf = b"%PDF-1.4 fake content";

    // Path includes "arxiv.org" so the whitelist check passes
    Mock::given(method("GET"))
        .and(path("/arxiv.org/paper.pdf"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/pdf")
                .set_body_bytes(fake_pdf.as_ref()),
        )
        .mount(&mock)
        .await;

    let pdf_url = format!("{}/arxiv.org/paper.pdf", mock.uri());
    let work = work_with_pdf_url("W1", &pdf_url);
    let http = reqwest::Client::new();

    let result = try_download_open_access_pdf(&http, &work).await.unwrap();
    let (bytes, _source) = result.expect("should find OA PDF");
    assert_eq!(bytes, fake_pdf);
}

#[tokio::test]
async fn try_download_oa_pdf_returns_none_when_no_oa_locations() {
    let work = work_no_oa("W2");
    let http = reqwest::Client::new();
    let result = try_download_open_access_pdf(&http, &work).await.unwrap();
    assert!(result.is_none(), "no OA location should return None");
}

#[tokio::test]
async fn try_download_oa_pdf_returns_none_when_pdf_url_is_null() {
    let work: papers_openalex::Work = serde_json::from_value(serde_json::json!({
        "id": "https://openalex.org/W3",
        "best_oa_location": {
            "pdf_url": null,
            "is_oa": false
        }
    }))
    .unwrap();
    let http = reqwest::Client::new();
    let result = try_download_open_access_pdf(&http, &work).await.unwrap();
    assert!(result.is_none());
}

#[tokio::test]
#[serial]
async fn try_download_oa_pdf_skips_non_whitelisted_url() {
    // A URL that doesn't contain any whitelisted domain should be skipped;
    // with no fallback content API, the result is None.
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/no-whitelist-domain/paper.pdf"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "application/pdf")
                .set_body_bytes(b"%PDF-1.4 skipped".as_ref()),
        )
        .mount(&mock)
        .await;

    let pdf_url = format!("{}/no-whitelist-domain/paper.pdf", mock.uri());
    let work = work_with_pdf_url("W4", &pdf_url);
    let http = reqwest::Client::new();

    let result = try_download_open_access_pdf(&http, &work).await.unwrap();
    // URL doesn't match any whitelisted domain → skipped → None
    assert!(result.is_none());
}

#[tokio::test]
#[serial]
async fn try_download_oa_pdf_skips_non_pdf_content_type() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/arxiv.org/html-page"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html")
                .set_body_bytes(b"<html>not a pdf</html>".as_ref()),
        )
        .mount(&mock)
        .await;

    let pdf_url = format!("{}/arxiv.org/html-page", mock.uri());
    let work = work_with_pdf_url("W5", &pdf_url);
    let http = reqwest::Client::new();

    let result = try_download_open_access_pdf(&http, &work).await.unwrap();
    // content-type is text/html, not application/pdf → skipped
    assert!(result.is_none());
}
