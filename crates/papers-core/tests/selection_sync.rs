//! Tests for Zotero sync helper functions used by `selection sync`.
//!
//! Covers:
//! - `find_papers_zip_key`: locates the `papers_extract_{key}.zip` attachment

use papers_core::text::find_papers_zip_key;
use papers_zotero::ZoteroClient;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Test helpers ──────────────────────────────────────────────────────────────

fn make_zotero_client(mock: &MockServer) -> ZoteroClient {
    ZoteroClient::new("testuser", "test-api-key").with_base_url(mock.uri())
}

/// Build a JSON array response suitable for `list_item_children`.
fn children_response(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("Total-Results", "1")
        .insert_header("Last-Modified-Version", "100")
        .set_body_string(body)
}

fn empty_children_response() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("Total-Results", "0")
        .insert_header("Last-Modified-Version", "100")
        .set_body_string("[]")
}

/// Build a Zotero attachment item JSON.
fn attachment_item(key: &str, parent_key: &str, filename: &str, link_mode: &str) -> String {
    format!(
        r#"{{
            "key": "{key}",
            "version": 1,
            "library": {{"type": "user", "id": 1, "name": "testuser", "links": {{}}}},
            "links": {{}},
            "meta": {{}},
            "data": {{
                "key": "{key}",
                "version": 1,
                "itemType": "attachment",
                "parentItem": "{parent_key}",
                "title": "{filename}",
                "filename": "{filename}",
                "linkMode": "{link_mode}",
                "contentType": "application/zip",
                "creators": [],
                "tags": [],
                "collections": [],
                "dateAdded": "2024-01-01T00:00:00Z",
                "dateModified": "2024-01-01T00:00:00Z"
            }}
        }}"#
    )
}

// ── find_papers_zip_key ───────────────────────────────────────────────────────

#[tokio::test]
async fn find_papers_zip_key_found() {
    let mock = MockServer::start().await;
    let parent = "LF4MJWZK";
    let att_key = "ATT12345";
    let filename = format!("papers_extract_{parent}.zip");

    let body = format!("[{}]", attachment_item(att_key, parent, &filename, "imported_file"));
    Mock::given(method("GET"))
        .and(path(format!("/users/testuser/items/{parent}/children")))
        .respond_with(children_response(&body))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let found = find_papers_zip_key(&zc, parent).await.unwrap();
    assert_eq!(found.as_deref(), Some(att_key));
}

#[tokio::test]
async fn find_papers_zip_key_not_found_empty_children() {
    let mock = MockServer::start().await;
    let parent = "XY2Z3ABC";

    Mock::given(method("GET"))
        .and(path(format!("/users/testuser/items/{parent}/children")))
        .respond_with(empty_children_response())
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let found = find_papers_zip_key(&zc, parent).await.unwrap();
    assert!(found.is_none());
}

#[tokio::test]
async fn find_papers_zip_key_not_found_wrong_filename() {
    let mock = MockServer::start().await;
    let parent = "AAAA1111";
    let att_key = "BBBZ2222";

    // Attachment exists but with wrong filename
    let body = format!(
        "[{}]",
        attachment_item(att_key, parent, "some_other_attachment.zip", "imported_file")
    );
    Mock::given(method("GET"))
        .and(path(format!("/users/testuser/items/{parent}/children")))
        .respond_with(children_response(&body))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let found = find_papers_zip_key(&zc, parent).await.unwrap();
    assert!(found.is_none());
}

#[tokio::test]
async fn find_papers_zip_key_skips_wrong_link_mode() {
    let mock = MockServer::start().await;
    let parent = "CCCC3333";
    let att_key = "DDDD4444";
    let filename = format!("papers_extract_{parent}.zip");

    // Correct filename but link_mode is "linked_file" instead of "imported_file"
    let body = format!("[{}]", attachment_item(att_key, parent, &filename, "linked_file"));
    Mock::given(method("GET"))
        .and(path(format!("/users/testuser/items/{parent}/children")))
        .respond_with(children_response(&body))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let found = find_papers_zip_key(&zc, parent).await.unwrap();
    assert!(found.is_none());
}

#[tokio::test]
async fn find_papers_zip_key_found_among_multiple_children() {
    let mock = MockServer::start().await;
    let parent = "EEEE5555";
    let pdf_att = "FFFF6666";
    let zip_att = "GGGG7777";
    let zip_filename = format!("papers_extract_{parent}.zip");

    // Multiple children: a PDF attachment and the extract ZIP
    let pdf_item = attachment_item(pdf_att, parent, "paper.pdf", "imported_file");
    let zip_item = attachment_item(zip_att, parent, &zip_filename, "imported_file");
    let body = format!("[{pdf_item}, {zip_item}]");

    Mock::given(method("GET"))
        .and(path(format!("/users/testuser/items/{parent}/children")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "2")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(body),
        )
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let found = find_papers_zip_key(&zc, parent).await.unwrap();
    assert_eq!(found.as_deref(), Some(zip_att));
}
