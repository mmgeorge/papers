use papers_core::selection::*;
use serial_test::serial;
use std::path::PathBuf;
use tempfile::TempDir;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ── Test helpers ───────────────────────────────────────────────────────────

/// Sets `PAPERS_DATA_DIR` to an isolated temp dir for the duration of the
/// returned `TempDir`. The caller must keep the `TempDir` alive.
fn isolated_dir() -> (TempDir, PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_path_buf();
    // SAFETY: tests are single-threaded (cargo test runs each test in its own process
    // for integration tests, and set_var is safe when no other threads exist).
    unsafe { std::env::set_var("PAPERS_DATA_DIR", &path) };
    (dir, path)
}

fn make_oa_client(mock: &MockServer) -> papers_openalex::OpenAlexClient {
    papers_openalex::OpenAlexClient::new().with_base_url(mock.uri())
}

fn work_json(id: &str, doi: Option<&str>, title: &str, authors: &[&str], year: i32) -> String {
    let doi_str = doi.map_or("null".into(), |d| format!("\"https://doi.org/{d}\""));
    let authors_str: Vec<String> = authors
        .iter()
        .map(|a| {
            format!(
                r#"{{"author_position":"first","author":{{"id":"https://openalex.org/A1","display_name":"{a}"}}}}"#
            )
        })
        .collect();
    format!(
        r#"{{
          "id": "https://openalex.org/{id}",
          "doi": {doi_str},
          "display_name": "{title}",
          "title": "{title}",
          "publication_year": {year},
          "type": "article",
          "cited_by_count": 10,
          "authorships": [{authors}],
          "primary_location": {{"source": {{"id":"https://openalex.org/S1","display_name":"Nature","issn":["0028-0836"],"issn_l":"0028-0836","is_oa":false,"is_in_doaj":false,"host_organization":null,"host_organization_name":null,"host_organization_lineage":null,"host_organization_lineage_names":null,"type":"journal"}}, "is_oa": false, "landing_page_url": null, "pdf_url": null, "license": null, "license_id": null, "version": null, "is_accepted": false, "is_published": false}},
          "open_access": {{"is_oa": false, "oa_status": "closed", "oa_url": null, "any_repository_has_fulltext": false}},
          "primary_topic": null,
          "referenced_works": [],
          "counts_by_year": []
        }}"#,
        authors = authors_str.join(",")
    )
}

fn list_response(result: &str) -> String {
    format!(
        r#"{{"meta":{{"count":1,"db_response_time_ms":5,"page":1,"per_page":25,"next_cursor":null,"groups_count":null}},"results":[{result}],"group_by":[]}}"#
    )
}

fn empty_list_response() -> String {
    r#"{"meta":{"count":0,"db_response_time_ms":5,"page":1,"per_page":25,"next_cursor":null,"groups_count":null},"results":[],"group_by":[]}"#.to_string()
}

// ── selection list ─────────────────────────────────────────────────────────

#[test]
#[serial]
fn list_empty_dir() {
    let (_dir, _) = isolated_dir();
    let names = list_selection_names();
    assert!(names.is_empty());
}

#[test]
#[serial]
fn list_multiple_no_active() {
    let (_dir, _) = isolated_dir();
    for name in &["alpha", "beta", "gamma"] {
        save_selection(&Selection { name: name.to_string(), entries: vec![] }).unwrap();
    }
    let names = list_selection_names();
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    // No active
    let state = load_state();
    assert!(state.active.is_none());
}

#[test]
#[serial]
fn list_active_marked() {
    let (_dir, _) = isolated_dir();
    for name in &["a", "b", "c"] {
        save_selection(&Selection { name: name.to_string(), entries: vec![] }).unwrap();
    }
    save_state(&SelectionState { active: Some("b".into()) }).unwrap();
    assert_eq!(active_selection_name().as_deref(), Some("b"));
    let state = load_state();
    assert_eq!(state.active.as_deref(), Some("b"));
}

#[test]
#[serial]
fn list_counts_match_entries() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W1".into()),
        doi: None,
        title: Some("Test".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    save_selection(&Selection { name: "zero".into(), entries: vec![] }).unwrap();
    save_selection(&Selection {
        name: "two".into(),
        entries: vec![entry.clone(), entry.clone()],
    }).unwrap();
    save_selection(&Selection {
        name: "five".into(),
        entries: vec![entry.clone(); 5],
    }).unwrap();

    assert_eq!(load_selection("zero").unwrap().entries.len(), 0);
    assert_eq!(load_selection("two").unwrap().entries.len(), 2);
    assert_eq!(load_selection("five").unwrap().entries.len(), 5);
}

#[test]
#[serial]
fn list_sorted_alphabetically() {
    let (_dir, _) = isolated_dir();
    for name in &["zebra", "alpha", "middle"] {
        save_selection(&Selection { name: name.to_string(), entries: vec![] }).unwrap();
    }
    let names = list_selection_names();
    assert_eq!(names, vec!["alpha", "middle", "zebra"]);
}

// ── selection get ──────────────────────────────────────────────────────────

#[test]
#[serial]
fn get_by_name_activates() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "foo".into(), entries: vec![] }).unwrap();
    save_selection(&Selection { name: "bar".into(), entries: vec![] }).unwrap();

    // simulate get("foo"): resolve then set active
    let name = resolve_selection("foo").unwrap();
    let mut state = load_state();
    state.active = Some(name.clone());
    save_state(&state).unwrap();

    assert_eq!(active_selection_name().as_deref(), Some("foo"));
}

#[test]
#[serial]
fn get_by_index_activates() {
    let (_dir, _) = isolated_dir();
    // sorted: alpha, beta, gamma → index 2 = beta
    for name in &["gamma", "alpha", "beta"] {
        save_selection(&Selection { name: name.to_string(), entries: vec![] }).unwrap();
    }
    let name = resolve_selection("2").unwrap();
    assert_eq!(name, "beta");
}

#[test]
#[serial]
fn get_no_arg_uses_active() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "bar".into(), entries: vec![] }).unwrap();
    save_state(&SelectionState { active: Some("bar".into()) }).unwrap();
    let active = active_selection_name().unwrap();
    let sel = load_selection(&active).unwrap();
    assert_eq!(sel.name, "bar");
}

#[test]
#[serial]
fn get_no_arg_no_active_errors() {
    let (_dir, _) = isolated_dir();
    // No state file → no active
    let result = active_selection_name();
    assert!(result.is_none());
}

#[test]
#[serial]
fn get_not_found_errors() {
    let (_dir, _) = isolated_dir();
    let err = load_selection("nonexistent").unwrap_err();
    assert!(matches!(err, SelectionError::NotFound(_)));
}

#[test]
#[serial]
fn get_lists_all_entries() {
    let (_dir, _) = isolated_dir();
    let entries: Vec<SelectionEntry> = (0..3)
        .map(|i| SelectionEntry {
            zotero_key: None,
            openalex_id: Some(format!("W{i}")),
            doi: None,
            title: Some(format!("Paper {i}")),
            authors: None,
            year: Some(2020 + i),
            issn: None,
            isbn: None,
        work_type: None,
        })
        .collect();
    save_selection(&Selection { name: "mysel".into(), entries: entries.clone() }).unwrap();
    let sel = load_selection("mysel").unwrap();
    assert_eq!(sel.entries.len(), 3);
}

// ── selection create ───────────────────────────────────────────────────────

#[test]
#[serial]
fn create_makes_file() {
    let (_dir, data_path) = isolated_dir();
    let sel = Selection { name: "newsel".into(), entries: vec![] };
    save_selection(&sel).unwrap();
    let file = data_path.join("papers").join("selections").join("newsel.json");
    assert!(file.exists());
}

#[test]
#[serial]
fn create_activates() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "mysel".into(), entries: vec![] }).unwrap();
    save_state(&SelectionState { active: Some("mysel".into()) }).unwrap();
    assert_eq!(active_selection_name().as_deref(), Some("mysel"));
}

#[test]
#[serial]
fn create_invalid_name() {
    let err = validate_name("bad name").unwrap_err();
    assert!(matches!(err, SelectionError::InvalidName(_)));
    let err2 = validate_name("bad/name").unwrap_err();
    assert!(matches!(err2, SelectionError::InvalidName(_)));
}

#[test]
#[serial]
fn create_duplicate_errors() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "dup".into(), entries: vec![] }).unwrap();
    // Simulate the "already exists" check used by CLI/MCP
    let exists = load_selection("dup").is_ok();
    assert!(exists);
}

#[test]
#[serial]
fn create_replaces_previous_active() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "a".into(), entries: vec![] }).unwrap();
    save_state(&SelectionState { active: Some("a".into()) }).unwrap();

    save_selection(&Selection { name: "b".into(), entries: vec![] }).unwrap();
    save_state(&SelectionState { active: Some("b".into()) }).unwrap();

    assert_eq!(active_selection_name().as_deref(), Some("b"));
    // "a" still exists
    assert!(load_selection("a").is_ok());
}

// ── selection delete ───────────────────────────────────────────────────────

#[test]
#[serial]
fn delete_removes_file() {
    let (_dir, data_path) = isolated_dir();
    save_selection(&Selection { name: "todel".into(), entries: vec![] }).unwrap();
    let file = data_path.join("papers").join("selections").join("todel.json");
    assert!(file.exists());
    delete_selection("todel").unwrap();
    assert!(!file.exists());
}

#[test]
#[serial]
fn delete_active_clears_state() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "active_one".into(), entries: vec![] }).unwrap();
    save_state(&SelectionState { active: Some("active_one".into()) }).unwrap();
    delete_selection("active_one").unwrap();
    // caller (CLI/MCP) is responsible for clearing state; test the mechanism:
    let mut state = load_state();
    if state.active.as_deref() == Some("active_one") {
        state.active = None;
        save_state(&state).unwrap();
    }
    assert!(active_selection_name().is_none());
}

#[test]
#[serial]
fn delete_non_active_preserves_state() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "keep_active".into(), entries: vec![] }).unwrap();
    save_selection(&Selection { name: "other".into(), entries: vec![] }).unwrap();
    save_state(&SelectionState { active: Some("keep_active".into()) }).unwrap();
    delete_selection("other").unwrap();
    assert_eq!(active_selection_name().as_deref(), Some("keep_active"));
}

#[test]
#[serial]
fn delete_by_index() {
    let (_dir, _) = isolated_dir();
    for name in &["a", "b", "c"] {
        save_selection(&Selection { name: name.to_string(), entries: vec![] }).unwrap();
    }
    let name = resolve_selection("1").unwrap();
    assert_eq!(name, "a");
    delete_selection(&name).unwrap();
    let remaining = list_selection_names();
    assert!(!remaining.contains(&"a".to_string()));
}

#[test]
#[serial]
fn delete_nonexistent_errors() {
    let (_dir, _) = isolated_dir();
    let err = delete_selection("ghost").unwrap_err();
    assert!(matches!(err, SelectionError::NotFound(_)));
}

// ── selection add ──────────────────────────────────────────────────────────

#[tokio::test]
#[serial]
async fn add_by_openalex_id_not_in_zotero() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W2741809807", Some("10.1234/test"), "Attention Is All You Need", &["Vaswani"], 2017);

    Mock::given(method("GET"))
        .and(path("/works/W2741809807"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    save_selection(&Selection { name: "mysel".into(), entries: vec![] }).unwrap();

    let entry = resolve_paper("W2741809807", &client, None).await.unwrap();
    assert_eq!(entry.openalex_id.as_deref(), Some("W2741809807"));
    assert_eq!(entry.doi.as_deref(), Some("10.1234/test"));
    assert!(entry.title.is_some());
    assert!(entry.authors.is_some());
    assert_eq!(entry.year, Some(2017));
    assert!(entry.zotero_key.is_none());
}

#[tokio::test]
#[serial]
async fn add_by_doi_not_in_zotero() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W1", Some("10.1234/test"), "A Great Paper", &["Author A"], 2020);
    Mock::given(method("GET"))
        .and(path("/works/doi:10.1234/test"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let entry = resolve_paper("10.1234/test", &client, None).await.unwrap();
    assert_eq!(entry.doi.as_deref(), Some("10.1234/test"));
    assert!(entry.openalex_id.is_some());
}

#[tokio::test]
#[serial]
async fn add_title_zotero_miss_openalex_hit() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W999", Some("10.9999/x"), "My Unique Title", &["Someone"], 2021);
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("search", "My Unique Title"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&list_response(&work)))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let entry = resolve_paper("My Unique Title", &client, None).await.unwrap();
    assert_eq!(entry.openalex_id.as_deref(), Some("W999"));
    assert_eq!(entry.title.as_deref(), Some("My Unique Title"));
    assert!(entry.zotero_key.is_none());
}

#[tokio::test]
#[serial]
async fn add_title_zotero_miss_openalex_miss() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&empty_list_response()))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let err = resolve_paper("xyzzy completely unknown title xyz", &client, None).await.unwrap_err();
    assert!(matches!(err, SelectionError::CannotResolve(_)));
}

#[tokio::test]
#[serial]
async fn add_populates_all_fields() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W100", Some("10.100/foo"), "Full Fields Paper", &["Alice", "Bob"], 2022);
    Mock::given(method("GET"))
        .and(path("/works/W100"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let entry = resolve_paper("W100", &client, None).await.unwrap();
    assert_eq!(entry.openalex_id.as_deref(), Some("W100"));
    assert_eq!(entry.doi.as_deref(), Some("10.100/foo"));
    assert_eq!(entry.title.as_deref(), Some("Full Fields Paper"));
    assert!(entry.authors.as_ref().map(|a| a.len()).unwrap_or(0) >= 1);
    assert_eq!(entry.year, Some(2022));
}

#[tokio::test]
#[serial]
async fn add_issn_from_journal() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W200", Some("10.200/journal"), "Journal Paper", &["J Author"], 2023);
    Mock::given(method("GET"))
        .and(path("/works/W200"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let entry = resolve_paper("W200", &client, None).await.unwrap();
    // The mock work has ISSN 0028-0836 in primary_location.source.issn
    assert!(entry.issn.is_some());
    let issns = entry.issn.unwrap();
    assert!(!issns.is_empty());
    assert!(issns.contains(&"0028-0836".to_string()));
}

#[tokio::test]
#[serial]
async fn add_duplicate_by_doi_noop() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W300", Some("10.300/dup"), "Dup Paper", &["Auth"], 2020);
    Mock::given(method("GET"))
        .and(path("/works/doi:10.300/dup"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    save_selection(&Selection { name: "s".into(), entries: vec![] }).unwrap();

    let entry1 = resolve_paper("10.300/dup", &client, None).await.unwrap();
    let mut sel = load_selection("s").unwrap();
    sel.entries.push(entry1.clone());
    save_selection(&sel).unwrap();

    // Second add — same DOI
    let entry2 = resolve_paper("10.300/dup", &client, None).await.unwrap();
    let mut sel = load_selection("s").unwrap();
    let is_dup = sel.entries.iter().any(|e| {
        entry2.doi.as_deref().map(|d| entry_matches_doi(e, d)).unwrap_or(false)
    });
    if !is_dup {
        sel.entries.push(entry2);
        save_selection(&sel).unwrap();
    }

    let sel = load_selection("s").unwrap();
    assert_eq!(sel.entries.len(), 1);
}

#[tokio::test]
#[serial]
async fn add_duplicate_by_openalex_id_noop() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W400", Some("10.400/x"), "OA Dup", &["Auth"], 2021);
    Mock::given(method("GET"))
        .and(path("/works/W400"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    save_selection(&Selection { name: "t".into(), entries: vec![] }).unwrap();

    let entry1 = resolve_paper("W400", &client, None).await.unwrap();
    let mut sel = load_selection("t").unwrap();
    sel.entries.push(entry1.clone());
    save_selection(&sel).unwrap();

    let entry2 = resolve_paper("W400", &client, None).await.unwrap();
    let mut sel = load_selection("t").unwrap();
    let is_dup = sel.entries.iter().any(|e| {
        entry2.openalex_id.as_deref().map(|id| entry_matches_openalex(e, id)).unwrap_or(false)
    });
    if !is_dup {
        sel.entries.push(entry2);
        save_selection(&sel).unwrap();
    }

    assert_eq!(load_selection("t").unwrap().entries.len(), 1);
}

// ── validate_name ──────────────────────────────────────────────────────────

#[test]
fn validate_name_valid_with_hyphen() {
    assert!(validate_name("my-selection").is_ok());
    assert!(validate_name("alpha-beta-gamma").is_ok());
}

#[test]
fn validate_name_valid_with_underscore() {
    assert!(validate_name("my_selection").is_ok());
    assert!(validate_name("gpu_papers_2024").is_ok());
}

#[test]
fn validate_name_empty_invalid() {
    let err = validate_name("").unwrap_err();
    assert!(matches!(err, SelectionError::InvalidName(_)));
}

#[test]
fn validate_name_dot_invalid() {
    let err = validate_name("my.paper").unwrap_err();
    assert!(matches!(err, SelectionError::InvalidName(_)));
}

// ── resolve_selection edge cases ────────────────────────────────────────────

#[test]
#[serial]
fn resolve_selection_zero_index_errors() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "alpha".into(), entries: vec![] }).unwrap();
    let err = resolve_selection("0").unwrap_err();
    assert!(matches!(err, SelectionError::NotFound(_)));
}

#[test]
#[serial]
fn resolve_selection_overflow_index_errors() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "alpha".into(), entries: vec![] }).unwrap();
    let err = resolve_selection("99").unwrap_err();
    assert!(matches!(err, SelectionError::NotFound(_)));
}

#[test]
#[serial]
fn resolve_selection_case_insensitive() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "alpha".into(), entries: vec![] }).unwrap();
    let result = resolve_selection("ALPHA").unwrap();
    assert_eq!(result, "alpha");
}

#[test]
#[serial]
fn resolve_selection_not_found_name() {
    let (_dir, _) = isolated_dir();
    let err = resolve_selection("doesnotexist").unwrap_err();
    assert!(matches!(err, SelectionError::NotFound(_)));
}

// ── looks_like_doi ──────────────────────────────────────────────────────────

#[test]
fn looks_like_doi_variants() {
    assert!(looks_like_doi("10.1145/123456.789"));
    assert!(looks_like_doi("https://doi.org/10.1145/123456.789"));
    assert!(looks_like_doi("http://doi.org/10.1145/123456.789"));
    assert!(looks_like_doi("doi:10.1145/123456.789"));
}

#[test]
fn looks_like_doi_rejects_non_dois() {
    assert!(!looks_like_doi("W2741809807"));
    assert!(!looks_like_doi("attention is all you need"));
    assert!(!looks_like_doi("ABCD1234"));
    assert!(!looks_like_doi("10.1234")); // starts with 10. but no slash
}

// ── looks_like_openalex_work_id ─────────────────────────────────────────────

#[test]
fn looks_like_openalex_work_id_valid() {
    assert!(looks_like_openalex_work_id("W2741809807"));
    assert!(looks_like_openalex_work_id("W1"));
    assert!(looks_like_openalex_work_id("https://openalex.org/W2741809807"));
}

#[test]
fn looks_like_openalex_work_id_invalid() {
    assert!(!looks_like_openalex_work_id("W")); // just W, no digits
    assert!(!looks_like_openalex_work_id("author123")); // not W-prefixed
    assert!(!looks_like_openalex_work_id("ABCD1234")); // Zotero key format
    assert!(!looks_like_openalex_work_id("10.1234/foo")); // DOI
}

// ── entry_matches_doi normalization ────────────────────────────────────────

#[test]
fn entry_matches_doi_normalized() {
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: None,
        doi: Some("10.1234/foo".into()),
        title: None,
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    assert!(entry_matches_doi(&entry, "10.1234/foo"));
    assert!(entry_matches_doi(&entry, "https://doi.org/10.1234/foo"));
    assert!(entry_matches_doi(&entry, "http://doi.org/10.1234/foo"));
    assert!(entry_matches_doi(&entry, "doi:10.1234/foo"));
    assert!(!entry_matches_doi(&entry, "10.9999/other"));
}

// ── entry_matches_remove_input ─────────────────────────────────────────────

#[test]
fn entry_matches_remove_title_case_insensitive() {
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: None,
        doi: None,
        title: Some("Attention Is All You Need".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    assert!(entry_matches_remove_input(&entry, "attention"));
    assert!(entry_matches_remove_input(&entry, "ATTENTION IS ALL"));
    assert!(!entry_matches_remove_input(&entry, "transformers"));
}

#[test]
fn entry_matches_remove_oa_full_url() {
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W99999".into()),
        doi: None,
        title: None,
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    assert!(entry_matches_remove_input(&entry, "https://openalex.org/W99999"));
    assert!(entry_matches_remove_input(&entry, "W99999"));
}

// ── save/load round-trip ───────────────────────────────────────────────────

#[test]
#[serial]
fn save_load_roundtrip_all_fields() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: Some("LF4MJWZK".into()),
        openalex_id: Some("W2741809807".into()),
        doi: Some("10.48550/arxiv.1706.03762".into()),
        title: Some("Attention Is All You Need".into()),
        authors: Some(vec!["Vaswani".into(), "Shazeer".into()]),
        year: Some(2017),
        issn: Some(vec!["0028-0836".into()]),
        isbn: Some(vec!["978-3-16-148410-0".into()]),
        work_type: None,
    };
    let sel = Selection { name: "roundtrip".into(), entries: vec![entry] };
    save_selection(&sel).unwrap();

    let loaded = load_selection("roundtrip").unwrap();
    let e = &loaded.entries[0];
    assert_eq!(e.zotero_key.as_deref(), Some("LF4MJWZK"));
    assert_eq!(e.openalex_id.as_deref(), Some("W2741809807"));
    assert_eq!(e.doi.as_deref(), Some("10.48550/arxiv.1706.03762"));
    assert_eq!(e.title.as_deref(), Some("Attention Is All You Need"));
    assert_eq!(e.authors.as_ref().unwrap(), &["Vaswani", "Shazeer"]);
    assert_eq!(e.year, Some(2017));
    assert_eq!(e.issn.as_ref().unwrap(), &["0028-0836"]);
    assert_eq!(e.isbn.as_ref().unwrap(), &["978-3-16-148410-0"]);
}

// ── add with DOI/OA URL prefix variants ────────────────────────────────────

#[tokio::test]
#[serial]
async fn add_doi_https_prefix_resolves() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W500", Some("10.500/test"), "HTTPS DOI Paper", &["Auth"], 2022);
    Mock::given(method("GET"))
        .and(path("/works/doi:10.500/test"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let entry = resolve_paper("https://doi.org/10.500/test", &client, None).await.unwrap();
    assert_eq!(entry.doi.as_deref(), Some("10.500/test"));
    assert_eq!(entry.openalex_id.as_deref(), Some("W500"));
}

#[tokio::test]
#[serial]
async fn add_doi_colon_prefix_resolves() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W501", Some("10.501/test"), "Colon DOI Paper", &["Auth"], 2022);
    Mock::given(method("GET"))
        .and(path("/works/doi:10.501/test"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let entry = resolve_paper("doi:10.501/test", &client, None).await.unwrap();
    assert_eq!(entry.doi.as_deref(), Some("10.501/test"));
    assert_eq!(entry.openalex_id.as_deref(), Some("W501"));
}

#[tokio::test]
#[serial]
async fn add_oa_full_url_resolves() {
    let (_dir, _) = isolated_dir();
    let mock = MockServer::start().await;

    let work = work_json("W600", Some("10.600/x"), "Full URL Paper", &["Auth"], 2023);
    Mock::given(method("GET"))
        .and(path("/works/W600"))
        .respond_with(ResponseTemplate::new(200).set_body_string(&work))
        .mount(&mock)
        .await;

    let client = make_oa_client(&mock);
    let entry = resolve_paper("https://openalex.org/W600", &client, None).await.unwrap();
    assert_eq!(entry.openalex_id.as_deref(), Some("W600"));
    assert_eq!(entry.title.as_deref(), Some("Full URL Paper"));
}

// ── selection remove ───────────────────────────────────────────────────────

#[test]
#[serial]
fn remove_by_zotero_key() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: Some("LF4MJWZK".into()),
        openalex_id: None,
        doi: None,
        title: Some("My Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let mut sel = Selection { name: "r".into(), entries: vec![entry] };
    save_selection(&sel).unwrap();

    sel.entries.retain(|e| !entry_matches_remove_input(e, "LF4MJWZK"));
    save_selection(&sel).unwrap();

    assert!(load_selection("r").unwrap().entries.is_empty());
}

#[test]
#[serial]
fn remove_by_doi() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W1".into()),
        doi: Some("10.1234/foo".into()),
        title: Some("DOI Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let mut sel = Selection { name: "s".into(), entries: vec![entry] };
    save_selection(&sel).unwrap();

    sel.entries.retain(|e| !entry_matches_remove_input(e, "10.1234/foo"));
    save_selection(&sel).unwrap();

    assert!(load_selection("s").unwrap().entries.is_empty());
}

#[test]
#[serial]
fn remove_by_openalex_id() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W99999".into()),
        doi: None,
        title: Some("OA Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let mut sel = Selection { name: "u".into(), entries: vec![entry] };
    save_selection(&sel).unwrap();

    sel.entries.retain(|e| !entry_matches_remove_input(e, "W99999"));
    save_selection(&sel).unwrap();

    assert!(load_selection("u").unwrap().entries.is_empty());
}

#[test]
#[serial]
fn remove_nonexistent_item() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W1".into()),
        doi: None,
        title: Some("A Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let sel = Selection { name: "v".into(), entries: vec![entry] };
    save_selection(&sel).unwrap();

    // Try removing something that isn't there
    let sel = load_selection("v").unwrap();
    let before = sel.entries.len();
    let mut entries = sel.entries;
    entries.retain(|e| !entry_matches_remove_input(e, "NOTFOUND"));
    assert_eq!(entries.len(), before); // nothing removed
}

#[test]
#[serial]
fn remove_from_explicit_selection() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: Some("ABCD1234".into()),
        openalex_id: None,
        doi: None,
        title: Some("Other Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    // Create two selections; only "other" has the entry
    save_selection(&Selection { name: "active".into(), entries: vec![] }).unwrap();
    save_selection(&Selection { name: "other".into(), entries: vec![entry] }).unwrap();
    save_state(&SelectionState { active: Some("active".into()) }).unwrap();

    // Remove from "other" explicitly
    let name = resolve_selection("other").unwrap();
    let mut sel = load_selection(&name).unwrap();
    sel.entries.retain(|e| !entry_matches_remove_input(e, "ABCD1234"));
    save_selection(&sel).unwrap();

    assert!(load_selection("other").unwrap().entries.is_empty());
    // Active selection unchanged
    assert_eq!(active_selection_name().as_deref(), Some("active"));
}

#[test]
#[serial]
fn remove_by_title_substring_partial() {
    let (_dir, _) = isolated_dir();
    let entry1 = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W1".into()),
        doi: None,
        title: Some("Graph Neural Networks Survey".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let entry2 = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W2".into()),
        doi: None,
        title: Some("Attention Is All You Need".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let mut sel = Selection { name: "multi".into(), entries: vec![entry1, entry2] };
    save_selection(&sel).unwrap();

    // Remove only the "neural" matching entry
    sel.entries.retain(|e| !entry_matches_remove_input(e, "neural"));
    save_selection(&sel).unwrap();

    let loaded = load_selection("multi").unwrap();
    assert_eq!(loaded.entries.len(), 1);
    assert_eq!(loaded.entries[0].openalex_id.as_deref(), Some("W2"));
}

// ── work_type field ─────────────────────────────────────────────────────────

fn make_work_with_type(id: &str, oa_type: &str) -> papers_openalex::Work {
    serde_json::from_value(serde_json::json!({
        "id": format!("https://openalex.org/{id}"),
        "type": oa_type
    }))
    .expect("valid work json")
}

fn make_work_with_crossref_type(id: &str, oa_type: &str, crossref: &str) -> papers_openalex::Work {
    serde_json::from_value(serde_json::json!({
        "id": format!("https://openalex.org/{id}"),
        "type": oa_type,
        "type_crossref": crossref
    }))
    .expect("valid work json")
}

#[test]
fn fill_from_oa_work_captures_type() {
    let work = make_work_with_type("W1", "preprint");
    let mut entry = SelectionEntry {
        zotero_key: None,
        openalex_id: None,
        doi: None,
        title: None,
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    fill_from_oa_work(&mut entry, &work);
    assert_eq!(entry.work_type.as_deref(), Some("preprint"));
}

#[test]
fn fill_from_oa_work_prefers_crossref_type() {
    // type_crossref is more granular; should be preferred over type
    let work = make_work_with_crossref_type("W2", "article", "proceedings-article");
    let mut entry = SelectionEntry {
        zotero_key: None,
        openalex_id: None,
        doi: None,
        title: None,
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    fill_from_oa_work(&mut entry, &work);
    assert_eq!(entry.work_type.as_deref(), Some("proceedings-article"));
}

#[test]
fn fill_from_oa_work_does_not_overwrite_existing_work_type() {
    let work = make_work_with_type("W3", "book");
    let mut entry = SelectionEntry {
        zotero_key: None,
        openalex_id: None,
        doi: None,
        title: None,
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: Some("preprint".into()), // already set
    };
    fill_from_oa_work(&mut entry, &work);
    // Should not overwrite existing value
    assert_eq!(entry.work_type.as_deref(), Some("preprint"));
}

#[test]
fn openalex_type_to_zotero_article() {
    assert_eq!(openalex_type_to_zotero("article"), "journalArticle");
    assert_eq!(openalex_type_to_zotero("journal-article"), "journalArticle");
    assert_eq!(openalex_type_to_zotero("review"), "journalArticle");
}

#[test]
fn openalex_type_to_zotero_preprint() {
    assert_eq!(openalex_type_to_zotero("preprint"), "preprint");
    assert_eq!(openalex_type_to_zotero("posted-content"), "preprint");
}

#[test]
fn openalex_type_to_zotero_book() {
    assert_eq!(openalex_type_to_zotero("book"), "book");
    assert_eq!(openalex_type_to_zotero("book-chapter"), "bookSection");
    assert_eq!(openalex_type_to_zotero("book-section"), "bookSection");
}

#[test]
fn openalex_type_to_zotero_dissertation() {
    assert_eq!(openalex_type_to_zotero("dissertation"), "thesis");
    assert_eq!(openalex_type_to_zotero("thesis"), "thesis");
}

#[test]
fn openalex_type_to_zotero_conference() {
    assert_eq!(openalex_type_to_zotero("proceedings-article"), "conferencePaper");
    assert_eq!(openalex_type_to_zotero("conference-paper"), "conferencePaper");
}

#[test]
fn openalex_type_to_zotero_unknown_defaults_to_document() {
    assert_eq!(openalex_type_to_zotero("unknown-type"), "document");
    assert_eq!(openalex_type_to_zotero(""), "document");
    assert_eq!(openalex_type_to_zotero("dataset"), "document");
}

#[test]
#[serial]
fn roundtrip_with_work_type() {
    let (_dir, _) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W100".into()),
        doi: Some("10.100/test".into()),
        title: Some("Type Test Paper".into()),
        authors: None,
        year: Some(2024),
        issn: None,
        isbn: None,
        work_type: Some("proceedings-article".into()),
    };
    let sel = Selection { name: "type-test".into(), entries: vec![entry] };
    save_selection(&sel).unwrap();
    let loaded = load_selection("type-test").unwrap();
    assert_eq!(
        loaded.entries[0].work_type.as_deref(),
        Some("proceedings-article")
    );
}

#[test]
#[serial]
fn roundtrip_work_type_none_omitted_from_json() {
    // work_type: None should be omitted from serialized JSON (skip_serializing_if)
    let (_dir, data_path) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W200".into()),
        doi: None,
        title: Some("No Type Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let sel = Selection { name: "no-type".into(), entries: vec![entry] };
    save_selection(&sel).unwrap();
    let json_path = data_path.join("papers/selections/no-type.json");
    let json = std::fs::read_to_string(json_path).unwrap();
    // work_type: None should be omitted from the file
    assert!(!json.contains("work_type"));
    // But loading still works (defaults to None)
    let loaded = load_selection("no-type").unwrap();
    assert!(loaded.entries[0].work_type.is_none());
}

// ── selection rename (library-level) ───────────────────────────────────────

#[test]
#[serial]
fn rename_changes_file_and_updates_state() {
    let (_dir, data_path) = isolated_dir();
    let entry = SelectionEntry {
        zotero_key: Some("LF4MJWZK".into()),
        openalex_id: None,
        doi: None,
        title: Some("Rename Test".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    save_selection(&Selection { name: "old-name".into(), entries: vec![entry] }).unwrap();
    save_state(&SelectionState { active: Some("old-name".into()) }).unwrap();

    // Perform rename: save under new name, copy entries, delete old, update state
    let old_sel = load_selection("old-name").unwrap();
    let new_sel = Selection { name: "new-name".into(), entries: old_sel.entries };
    save_selection(&new_sel).unwrap();
    delete_selection("old-name").unwrap();
    save_state(&SelectionState { active: Some("new-name".into()) }).unwrap();

    let old_file = data_path.join("papers/selections/old-name.json");
    let new_file = data_path.join("papers/selections/new-name.json");
    assert!(!old_file.exists(), "old file should be gone");
    assert!(new_file.exists(), "new file should exist");
    assert_eq!(active_selection_name().as_deref(), Some("new-name"));
    // entries are preserved
    let loaded = load_selection("new-name").unwrap();
    assert_eq!(loaded.entries.len(), 1);
    assert_eq!(loaded.entries[0].zotero_key.as_deref(), Some("LF4MJWZK"));
}

#[test]
#[serial]
fn rename_to_existing_name_detected_by_caller() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "alpha".into(), entries: vec![] }).unwrap();
    save_selection(&Selection { name: "beta".into(), entries: vec![] }).unwrap();

    // Caller checks: if load_selection(new_name).is_ok() → refuse
    let conflict = load_selection("beta").is_ok();
    assert!(conflict, "should detect existing name before renaming");
}

#[test]
fn rename_invalid_new_name_caught_by_validate_name() {
    let err = validate_name("bad name").unwrap_err();
    assert!(matches!(err, SelectionError::InvalidName(_)));
    let err2 = validate_name("bad/name").unwrap_err();
    assert!(matches!(err2, SelectionError::InvalidName(_)));
}

// ── selection merge (library-level) ────────────────────────────────────────

#[test]
#[serial]
fn merge_adds_new_entries() {
    let (_dir, _) = isolated_dir();
    let e1 = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W1".into()),
        doi: None,
        title: Some("Paper One".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let e2 = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W2".into()),
        doi: None,
        title: Some("Paper Two".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };

    let mut target = Selection { name: "target".into(), entries: vec![e1.clone()] };
    let source = Selection { name: "source".into(), entries: vec![e2.clone()] };
    save_selection(&target).unwrap();
    save_selection(&source).unwrap();

    // Merge: add entries from source that aren't already in target
    for entry in &source.entries {
        let already = target.entries.iter().any(|e| {
            entry.openalex_id.as_deref().map(|id| entry_matches_openalex(e, id)).unwrap_or(false)
                || entry.doi.as_deref().map(|d| entry_matches_doi(e, d)).unwrap_or(false)
        });
        if !already {
            target.entries.push(entry.clone());
        }
    }
    save_selection(&target).unwrap();

    let loaded = load_selection("target").unwrap();
    assert_eq!(loaded.entries.len(), 2);
}

#[test]
#[serial]
fn merge_deduplicates_by_openalex_id() {
    let (_dir, _) = isolated_dir();
    let e = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W999".into()),
        doi: None,
        title: Some("Same Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let mut target = Selection { name: "t".into(), entries: vec![e.clone()] };
    let source = Selection { name: "s".into(), entries: vec![e.clone()] };
    save_selection(&target).unwrap();
    save_selection(&source).unwrap();

    for entry in &source.entries {
        let already = target.entries.iter().any(|t| {
            entry.openalex_id.as_deref().map(|id| entry_matches_openalex(t, id)).unwrap_or(false)
        });
        if !already {
            target.entries.push(entry.clone());
        }
    }
    save_selection(&target).unwrap();

    assert_eq!(load_selection("t").unwrap().entries.len(), 1);
}

#[test]
#[serial]
fn merge_deduplicates_by_doi() {
    let (_dir, _) = isolated_dir();
    let e = SelectionEntry {
        zotero_key: None,
        openalex_id: None,
        doi: Some("10.1234/dup".into()),
        title: Some("DOI Dup".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let mut target = Selection { name: "td".into(), entries: vec![e.clone()] };
    let source = Selection { name: "sd".into(), entries: vec![e.clone()] };
    save_selection(&target).unwrap();
    save_selection(&source).unwrap();

    for entry in &source.entries {
        let already = target.entries.iter().any(|t| {
            entry.doi.as_deref().map(|d| entry_matches_doi(t, d)).unwrap_or(false)
        });
        if !already {
            target.entries.push(entry.clone());
        }
    }
    save_selection(&target).unwrap();

    assert_eq!(load_selection("td").unwrap().entries.len(), 1);
}

#[test]
#[serial]
fn merge_preserves_source_selection() {
    let (_dir, _) = isolated_dir();
    let e = SelectionEntry {
        zotero_key: None,
        openalex_id: Some("W555".into()),
        doi: None,
        title: Some("Preserved Paper".into()),
        authors: None,
        year: None,
        issn: None,
        isbn: None,
        work_type: None,
    };
    let mut target = Selection { name: "main".into(), entries: vec![] };
    let source = Selection { name: "side".into(), entries: vec![e.clone()] };
    save_selection(&target).unwrap();
    save_selection(&source).unwrap();

    for entry in &source.entries {
        target.entries.push(entry.clone());
    }
    save_selection(&target).unwrap();
    // Source should still exist unchanged
    let src = load_selection("side").unwrap();
    assert_eq!(src.entries.len(), 1);
    assert_eq!(src.entries[0].openalex_id.as_deref(), Some("W555"));
}

#[test]
#[serial]
fn merge_nonexistent_source_errors() {
    let (_dir, _) = isolated_dir();
    save_selection(&Selection { name: "target".into(), entries: vec![] }).unwrap();
    let err = load_selection("nonexistent-source").unwrap_err();
    assert!(matches!(err, SelectionError::NotFound(_)));
}
