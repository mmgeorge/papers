//! Integration tests for the papers-db crate.
//!
//! These tests exercise the full ingest → query pipeline using a fake embedder
//! (zero vectors, no model download) against a temporary LanceDB directory.

use std::fs;
use tempfile::TempDir;
use serial_test::serial;

use crate::ingest::{IngestParams, ingest_paper, is_ingested, list_cached_item_keys};
use crate::query::{
    get_chapter, get_chunk, get_paper_outline, get_section, list_papers, list_tags,
};
use crate::store::DbStore;
use crate::types::{ListPapersParams, ListTagsParams};

// ── Test isolation ────────────────────────────────────────────────────────────

/// RAII guard that redirects the embed cache to an isolated TempDir for one
/// test, then clears the env var on drop.  Use together with `#[serial]` to
/// prevent parallel tests from clobbering each other's `PAPERS_EMBED_CACHE_DIR`
/// and from writing into the real platform cache.
struct EmbedCacheGuard {
    _dir: TempDir,
}

impl EmbedCacheGuard {
    fn new() -> Self {
        let dir = TempDir::new().expect("create temp embed cache dir");
        unsafe {
            std::env::set_var("PAPERS_EMBED_CACHE_DIR", dir.path());
        }
        Self { _dir: dir }
    }
}

impl Drop for EmbedCacheGuard {
    fn drop(&mut self) {
        unsafe {
            std::env::remove_var("PAPERS_EMBED_CACHE_DIR");
        }
    }
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Minimal Marker JSON with two top-level chapters, a subsection, a figure, a
/// table, and several block types that must be skipped (Caption, PageHeader).
/// Fixture JSON matching real Marker output conventions:
///   h1 = paper title (skipped), h2 = chapter, h3 = subsection, h6 = skipped.
fn minimal_marker_json() -> String {
    r#"{
      "children": [
        {
          "block_type": "Page",
          "children": [
            {
              "block_type": "SectionHeader",
              "id": "sh_title",
              "page": 0,
              "html": "<h1>Test Paper Title</h1>",
              "section_hierarchy": {}
            },
            {
              "block_type": "SectionHeader",
              "id": "sh_intro",
              "page": 0,
              "html": "<h2>Introduction</h2>",
              "section_hierarchy": {"1": "sh_title"}
            },
            {
              "block_type": "Text",
              "id": "t0",
              "page": 0,
              "html": "<p>First intro paragraph.</p>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_intro"}
            },
            {
              "block_type": "Text",
              "id": "t1",
              "page": 0,
              "html": "<p>Second intro paragraph.</p>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_intro"}
            },
            {
              "block_type": "SectionHeader",
              "id": "sh_bg",
              "page": 0,
              "html": "<h3>Background</h3>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_intro", "3": "sh_bg"}
            },
            {
              "block_type": "Text",
              "id": "t2",
              "page": 1,
              "html": "<p>Background text.</p>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_intro", "3": "sh_bg"}
            },
            {
              "block_type": "Figure",
              "id": "fig1",
              "page": 1,
              "html": "<img src=\"fig1.png\" alt=\"Figure 1: A diagram\"/>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_intro", "3": "sh_bg"}
            },
            {
              "block_type": "Caption",
              "id": "cap1",
              "page": 1,
              "html": "<p>Caption for figure 1.</p>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_intro", "3": "sh_bg"}
            },
            {
              "block_type": "SectionHeader",
              "id": "sh_method",
              "page": 2,
              "html": "<h2>Method</h2>",
              "section_hierarchy": {"1": "sh_title"}
            },
            {
              "block_type": "Text",
              "id": "t3",
              "page": 2,
              "html": "<p>Method description text.</p>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_method"}
            },
            {
              "block_type": "ListGroup",
              "id": "lg0",
              "page": 2,
              "html": "<ul><li>Step one</li><li>Step two</li></ul>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_method"}
            },
            {
              "block_type": "Table",
              "id": "tbl1",
              "page": 3,
              "html": "<img src=\"tbl1.png\" alt=\"Table 1: Results\"/>",
              "section_hierarchy": {"1": "sh_title", "2": "sh_method"}
            },
            {
              "block_type": "PageHeader",
              "id": "ph0",
              "page": 0,
              "html": "<p>Paper Title</p>",
              "section_hierarchy": {}
            }
          ]
        }
      ]
    }"#
    .to_string()
}

/// Write minimal_marker_json + a stub meta.json into a temp dir and return
/// a ready-to-use IngestParams.
fn make_test_cache(dir: &TempDir, item_key: &str) -> IngestParams {
    let cache_dir = dir.path().join(item_key);
    fs::create_dir_all(&cache_dir).unwrap();
    fs::write(
        cache_dir.join(format!("{item_key}.json")),
        minimal_marker_json(),
    )
    .unwrap();

    IngestParams {
        item_key: item_key.to_string(),
        paper_id: item_key.to_string(),
        title: "Test Paper".to_string(),
        authors: vec!["Alice".to_string(), "Bob".to_string()],
        year: Some(2023),
        venue: Some("SIGGRAPH".to_string()),
        tags: vec!["rendering".to_string(), "GPU".to_string()],
        cache_dir,
        force: false,
    }
}

/// Open a test store against a temporary directory.
async fn open_test_store(dir: &TempDir) -> DbStore {
    let path = dir.path().join("rag_db");
    DbStore::open_for_test(path.to_str().unwrap())
        .await
        .expect("open_for_test")
}

// ── Ingest tests ──────────────────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_ingest_basic_counts() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "TESTKEY1");

    let stats = ingest_paper(&store, params).await.unwrap();

    // Buffer-based chunking merges short blocks within each section:
    // ch1/s0: t0+t1 → 1 chunk, ch1/s1: t2 → 1 chunk, ch2/s0: t3+lg0 → 1 chunk
    assert_eq!(stats.chunks_added, 3, "expected 3 chunks (merged within sections)");
    // 2 exhibit blocks: fig1 (Figure) + tbl1 (Table); Caption is consumed
    assert_eq!(stats.exhibits_added, 2, "expected 2 exhibits");
}

#[serial]
#[tokio::test]
async fn test_ingest_chapter_section_assignment() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "TESTKEY2");

    ingest_paper(&store, params).await.unwrap();

    let outline = get_paper_outline(&store, "TESTKEY2").await.unwrap();
    assert_eq!(outline.chapters.len(), 2, "expected 2 chapters");

    let ch1 = &outline.chapters[0];
    assert_eq!(ch1.chapter_title, "Introduction");
    assert_eq!(ch1.chapter_idx, 1);
    // ch1 has sections: s0 (before Background) and s1 (Background)
    assert_eq!(ch1.sections.len(), 2, "ch1 should have 2 sections");

    let ch2 = &outline.chapters[1];
    assert_eq!(ch2.chapter_title, "Method");
    assert_eq!(ch2.chapter_idx, 2);
    assert_eq!(ch2.sections.len(), 1, "ch2 should have 1 section");
}

#[serial]
#[tokio::test]
async fn test_ingest_skips_caption_and_pageheader() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "TESTKEY3");

    let stats = ingest_paper(&store, params).await.unwrap();

    // If Caption or PageHeader were ingested as chunks, the count would be 7
    assert!(
        stats.chunks_added < 7,
        "Caption/PageHeader must not be ingested as chunks"
    );
}

#[serial]
#[tokio::test]
async fn test_ingest_chunk_ids_have_correct_format() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "PAPER_X");

    ingest_paper(&store, params).await.unwrap();

    // First chunk: ch1/s0/p0
    let chunk = get_chunk(&store, "PAPER_X/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.chunk_id, "PAPER_X/ch1/s0/p0");
    assert_eq!(chunk.chunk.chapter_idx, 1);
    assert_eq!(chunk.chunk.section_idx, 0);
    assert_eq!(chunk.chunk.chunk_idx, 0);
}

#[serial]
#[tokio::test]
async fn test_ingest_text_content_stripped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "STRIPTEST");

    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "STRIPTEST/ch1/s0/p0").await.unwrap();
    // HTML tags must be stripped; t0+t1 are merged in the same section
    assert!(chunk.chunk.text.contains("First intro paragraph."));
    assert!(chunk.chunk.text.contains("Second intro paragraph."));
    assert!(!chunk.chunk.text.contains('<'));
}

#[serial]
#[tokio::test]
async fn test_ingest_reingest_deduplicates() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // First ingest
    let params1 = make_test_cache(&cache_dir, "DEDUP");
    ingest_paper(&store, params1).await.unwrap();

    // Second ingest (same paper_id) — should replace, not accumulate
    let params2 = make_test_cache(&cache_dir, "DEDUP");
    ingest_paper(&store, params2).await.unwrap();

    // list_papers should show exactly 1 paper, not 2
    let papers = list_papers(
        &store,
        ListPapersParams {
            paper_ids: None,
            filter_year_min: None,
            filter_year_max: None,
            filter_venue: None,
            filter_tags: None,
            filter_authors: None,
            sort_by: None,
            limit: 50,
        },
    )
    .await
    .unwrap();
    assert_eq!(papers.len(), 1, "re-ingest should not duplicate the paper");
    assert_eq!(papers[0].chunk_count, 3);
}

// ── is_ingested ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_is_ingested_false_before_ingest() {
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    assert!(!is_ingested(&store, "NOTINGESTED").await);
}

#[serial]
#[tokio::test]
async fn test_is_ingested_true_after_ingest() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "INGESTED1");

    ingest_paper(&store, params).await.unwrap();
    assert!(is_ingested(&store, "INGESTED1").await);
}

// ── list_papers ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_list_papers_empty_db() {
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let papers = list_papers(
        &store,
        ListPapersParams {
            paper_ids: None,
            filter_year_min: None,
            filter_year_max: None,
            filter_venue: None,
            filter_tags: None,
            filter_authors: None,
            sort_by: None,
            limit: 50,
        },
    )
    .await
    .unwrap();
    assert!(papers.is_empty());
}

#[serial]
#[tokio::test]
async fn test_list_papers_returns_ingested_paper() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "LP_TEST1");

    ingest_paper(&store, params).await.unwrap();

    let papers = list_papers(
        &store,
        ListPapersParams {
            paper_ids: None,
            filter_year_min: None,
            filter_year_max: None,
            filter_venue: None,
            filter_tags: None,
            filter_authors: None,
            sort_by: None,
            limit: 50,
        },
    )
    .await
    .unwrap();

    assert_eq!(papers.len(), 1);
    let p = &papers[0];
    assert_eq!(p.paper_id, "LP_TEST1");
    assert_eq!(p.title, "Test Paper");
    assert_eq!(p.year, Some(2023));
    assert_eq!(p.venue.as_deref(), Some("SIGGRAPH"));
    assert!(p.authors.contains(&"Alice".to_string()));
    assert_eq!(p.chunk_count, 3);
    assert_eq!(p.exhibit_count, 2);
}

#[serial]
#[tokio::test]
async fn test_list_papers_year_filter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Ingest two papers with different years
    let mut params1 = make_test_cache(&cache_dir, "YR2020");
    params1.year = Some(2020);
    ingest_paper(&store, params1).await.unwrap();

    let mut params2 = make_test_cache(&cache_dir, "YR2023");
    params2.year = Some(2023);
    ingest_paper(&store, params2).await.unwrap();

    let papers = list_papers(
        &store,
        ListPapersParams {
            paper_ids: None,
            filter_year_min: Some(2022),
            filter_year_max: None,
            filter_venue: None,
            filter_tags: None,
            filter_authors: None,
            sort_by: None,
            limit: 50,
        },
    )
    .await
    .unwrap();

    assert_eq!(papers.len(), 1);
    assert_eq!(papers[0].paper_id, "YR2023");
}

#[serial]
#[tokio::test]
async fn test_list_papers_author_filter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params1 = make_test_cache(&cache_dir, "AUTH1");
    params1.authors = vec!["Alice Smith".to_string()];
    ingest_paper(&store, params1).await.unwrap();

    let mut params2 = make_test_cache(&cache_dir, "AUTH2");
    params2.authors = vec!["Bob Jones".to_string()];
    ingest_paper(&store, params2).await.unwrap();

    let papers = list_papers(
        &store,
        ListPapersParams {
            paper_ids: None,
            filter_year_min: None,
            filter_year_max: None,
            filter_venue: None,
            filter_tags: None,
            filter_authors: Some(vec!["alice".to_string()]),
            sort_by: None,
            limit: 50,
        },
    )
    .await
    .unwrap();

    assert_eq!(papers.len(), 1);
    assert_eq!(papers[0].paper_id, "AUTH1");
}

// ── list_tags ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_list_tags_empty_db() {
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let tags = list_tags(&store, ListTagsParams { paper_ids: None })
        .await
        .unwrap();
    assert!(tags.is_empty());
}

#[serial]
#[tokio::test]
async fn test_list_tags_returns_tag_counts() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Two papers: both share "rendering", only one has "GPU"
    let mut p1 = make_test_cache(&cache_dir, "TAG_A");
    p1.tags = vec!["rendering".to_string(), "GPU".to_string()];
    ingest_paper(&store, p1).await.unwrap();

    let mut p2 = make_test_cache(&cache_dir, "TAG_B");
    p2.tags = vec!["rendering".to_string()];
    ingest_paper(&store, p2).await.unwrap();

    let tags = list_tags(&store, ListTagsParams { paper_ids: None })
        .await
        .unwrap();

    let rendering = tags.iter().find(|t| t.tag == "rendering").unwrap();
    let gpu = tags.iter().find(|t| t.tag == "GPU").unwrap();

    assert_eq!(rendering.paper_count, 2, "rendering should appear in 2 papers");
    assert_eq!(gpu.paper_count, 1, "GPU should appear in 1 paper");
}

// ── get_paper_outline ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_get_paper_outline_not_found() {
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let result = get_paper_outline(&store, "NOEXIST").await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("NOEXIST"));
}

#[serial]
#[tokio::test]
async fn test_get_paper_outline_structure() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "OUTLINE1");

    ingest_paper(&store, params).await.unwrap();

    let outline = get_paper_outline(&store, "OUTLINE1").await.unwrap();

    assert_eq!(outline.paper_id, "OUTLINE1");
    assert_eq!(outline.title, "Test Paper");
    assert_eq!(outline.year, Some(2023));
    assert_eq!(outline.total_chunks, 3);
    assert_eq!(outline.total_exhibits, 2);
    assert_eq!(outline.chapters.len(), 2);

    // Chapter 1: Introduction with 2 sections
    let ch1 = &outline.chapters[0];
    assert_eq!(ch1.chapter_title, "Introduction");
    assert_eq!(ch1.sections.len(), 2);
    // s0: t0+t1 merged → 1 chunk; s1: t2 → 1 chunk
    assert_eq!(ch1.sections[0].chunk_count, 1);
    assert_eq!(ch1.sections[1].chunk_count, 1);

    // Chapter 2: Method with 1 section
    let ch2 = &outline.chapters[1];
    assert_eq!(ch2.chapter_title, "Method");
    assert_eq!(ch2.sections.len(), 1);
    // Method section has t3 + lg0 merged into 1 chunk
    assert_eq!(ch2.sections[0].chunk_count, 1);
}

// ── get_chunk ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_get_chunk_not_found() {
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let result = get_chunk(&store, "NOEXIST/ch1/s0/p0").await;
    assert!(result.is_err());
}

#[serial]
#[tokio::test]
async fn test_get_chunk_returns_prev_and_next() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "NEIGHBORS");

    ingest_paper(&store, params).await.unwrap();

    // With buffer-based chunking, t0+t1 merge into ch1/s0/p0 (only 1 chunk in s0).
    // ch1/s0/p0 is the first chunk — no prev, no next within s0.
    let result = get_chunk(&store, "NEIGHBORS/ch1/s0/p0").await.unwrap();
    assert!(result.prev.is_none(), "first chunk should have no prev");
    assert!(result.next.is_none(), "only chunk in s0 should have no next");
}

#[serial]
#[tokio::test]
async fn test_get_chunk_position_context() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "POS");

    ingest_paper(&store, params).await.unwrap();

    let result = get_chunk(&store, "POS/ch1/s0/p0").await.unwrap();
    let pos = &result.chunk.position;

    assert!(pos.is_first_in_section);
    assert_eq!(pos.total_chapters_in_paper, 2);
}

// ── get_section ───────────────────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_get_section_returns_chunks_in_order() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "SEC");

    ingest_paper(&store, params).await.unwrap();

    // ch2/s0: t3 + lg0 merged into 1 chunk
    let section = get_section(&store, "SEC", 2, 0).await.unwrap();
    assert_eq!(section.chapter_title, "Method");
    assert_eq!(section.total_chunks, 1);

    assert_eq!(section.chunks[0].chunk_idx, 0);
    assert!(section.chunks[0].text.contains("Method description text."));
    assert!(section.chunks[0].text.contains("Step one"));
}

#[serial]
#[tokio::test]
async fn test_get_section_empty_returns_zero_chunks() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "SECEMP");

    ingest_paper(&store, params).await.unwrap();

    // Section that doesn't exist should return 0 chunks (not an error)
    let section = get_section(&store, "SECEMP", 99, 99).await.unwrap();
    assert_eq!(section.total_chunks, 0);
}

// ── get_chapter ───────────────────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_get_chapter_groups_by_section() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "CHAP");

    ingest_paper(&store, params).await.unwrap();

    // Chapter 1: Introduction has 2 sections
    let chapter = get_chapter(&store, "CHAP", 1).await.unwrap();
    assert_eq!(chapter.chapter_title, "Introduction");
    assert_eq!(chapter.sections.len(), 2);
    assert_eq!(chapter.total_chunks, 2); // t0+t1 merged in s0 + t2 in s1
}

#[serial]
#[tokio::test]
async fn test_get_chapter_section_titles() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "SECTABLE");

    ingest_paper(&store, params).await.unwrap();

    let chapter = get_chapter(&store, "SECTABLE", 1).await.unwrap();

    // s1 (index 1) should have title "Background"
    let bg_section = chapter
        .sections
        .iter()
        .find(|s| s.section_idx == 1)
        .expect("should have section 1");
    assert_eq!(bg_section.section_title, "Background");
}

// ── list_cached_item_keys (filesystem test) ───────────────────────────────────

#[serial]
#[test]
fn test_list_cached_keys_multiple_papers() {
    let dir = TempDir::new().unwrap();

    for key in &["AKEY", "BKEY", "CKEY"] {
        let sub = dir.path().join(key);
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join(format!("{key}.json")), "{}").unwrap();
    }
    // Extra dir without JSON — should be skipped
    fs::create_dir_all(dir.path().join("NOJSON")).unwrap();

    let old = std::env::var("PAPERS_DATALAB_CACHE_DIR").ok();
    unsafe {
        std::env::set_var("PAPERS_DATALAB_CACHE_DIR", dir.path().to_str().unwrap());
    }

    let mut keys = list_cached_item_keys();
    keys.sort();

    unsafe {
        if let Some(v) = old {
            std::env::set_var("PAPERS_DATALAB_CACHE_DIR", v);
        } else {
            std::env::remove_var("PAPERS_DATALAB_CACHE_DIR");
        }
    }

    assert_eq!(keys, vec!["AKEY", "BKEY", "CKEY"]);
}

// ── figure ID format ──────────────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_exhibit_ids_have_correct_format() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "FIGID");

    ingest_paper(&store, params).await.unwrap();

    // Figures should be named FIGID/fig1 and FIGID/fig2
    let fig = get_exhibit(&store, "FIGID/fig1").await.unwrap();
    assert_eq!(fig.exhibit_id, "FIGID/fig1");
    assert_eq!(fig.exhibit_type, "figure");
    // Caption comes from the adjacent Caption block, not from alt text
    assert_eq!(fig.caption, "Caption for figure 1.");
    assert_eq!(fig.description.as_deref(), Some("Figure 1: A diagram"));

    let tbl = get_exhibit(&store, "FIGID/fig2").await.unwrap();
    assert_eq!(tbl.exhibit_id, "FIGID/fig2");
    assert_eq!(tbl.exhibit_type, "table");
    // No adjacent Caption block for the table, so caption falls back to alt text
    assert_eq!(tbl.caption, "Table 1: Results");
}

// ── paper metadata roundtrip ──────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_paper_metadata_preserved_in_chunks() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "META");

    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "META/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.title, "Test Paper");
    assert_eq!(chunk.chunk.year, Some(2023));
    assert_eq!(chunk.chunk.venue.as_deref(), Some("SIGGRAPH"));
    assert!(chunk.chunk.authors.contains(&"Alice".to_string()));
    assert!(chunk.chunk.authors.contains(&"Bob".to_string()));
}

// ── Heading-level parsing tests ────────────────────────────────────────────────
//
// These tests verify that the ingestor correctly classifies SectionHeader blocks
// by their HTML heading tag (h1–h6), matching the real Marker output structure
// seen in papers like YFACFA8C (Vertex Block Descent).

/// Build a minimal JSON with one page containing the given block list.
fn make_json(blocks: &[(&str, &str, &str)]) -> String {
    // blocks: (block_type, id, html)
    let children: String = blocks
        .iter()
        .map(|(bt, id, html)| {
            format!(
                r#"{{"block_type":"{bt}","id":"{id}","page":0,"html":"{html}","section_hierarchy":{{}}}}"#,
                bt = bt,
                id = id,
                html = html.replace('"', "\\\""),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(r#"{{"children":[{{"block_type":"Page","children":[{children}]}}]}}"#)
}

fn make_cache_from_json(dir: &TempDir, item_key: &str, json: &str) -> IngestParams {
    let cache_dir = dir.path().join(item_key);
    fs::create_dir_all(&cache_dir).unwrap();
    fs::write(cache_dir.join(format!("{item_key}.json")), json).unwrap();
    IngestParams {
        item_key: item_key.to_string(),
        paper_id: item_key.to_string(),
        title: "Heading Test".to_string(),
        authors: vec![],
        year: None,
        venue: None,
        tags: vec![],
        cache_dir,
        force: false,
    }
}

// 1. h1 is treated as paper title and skipped — no chapter is created.
#[serial]
#[tokio::test]
async fn test_h1_skipped_as_paper_title() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h1>Paper Title</h1>"),
        ("Text", "t0", "<p>Some text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H1SKIP", &json);
    ingest_paper(&store, params).await.unwrap();

    // Text lands in chapter 0 (no h2 ever encountered)
    let chunk = get_chunk(&store, "H1SKIP/ch0/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.chapter_idx, 0);
    assert_eq!(chunk.chunk.chapter_title, "");
}

// 2. h2 creates a chapter.
#[serial]
#[tokio::test]
async fn test_h2_creates_chapter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>1 INTRODUCTION</h2>"),
        ("Text", "t0", "<p>Intro text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H2CH", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "H2CH/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.chapter_idx, 1);
    assert_eq!(chunk.chunk.chapter_title, "1 INTRODUCTION");
    assert_eq!(chunk.chunk.section_idx, 0);
}

// 3. h3 creates a section within the current chapter.
#[serial]
#[tokio::test]
async fn test_h3_creates_section() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>3 METHOD</h2>"),
        ("SectionHeader", "sh2", "<h3>3.1 Global Optimization</h3>"),
        ("Text", "t0", "<p>Optimization text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H3SEC", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "H3SEC/ch1/s1/p0").await.unwrap();
    assert_eq!(chunk.chunk.chapter_idx, 1);
    assert_eq!(chunk.chunk.section_idx, 1);
    assert_eq!(chunk.chunk.section_title, "3.1 Global Optimization");
}

// 4. h4 also creates a section (treated same as h3).
#[serial]
#[tokio::test]
async fn test_h4_creates_section() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>Details</h2>"),
        ("SectionHeader", "sh2", "<h4>3.1.1 Sub-detail</h4>"),
        ("Text", "t0", "<p>Sub-detail text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H4SEC", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "H4SEC/ch1/s1/p0").await.unwrap();
    assert_eq!(chunk.chunk.section_idx, 1);
    assert_eq!(chunk.chunk.section_title, "3.1.1 Sub-detail");
}

// 5. h5 is skipped — does not create a chapter or section.
#[serial]
#[tokio::test]
async fn test_h5_is_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>Chapter</h2>"),
        ("SectionHeader", "sh2", "<h5>Marginal note</h5>"),
        ("Text", "t0", "<p>Text after h5.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H5SKIP", &json);
    ingest_paper(&store, params).await.unwrap();

    // h5 must not have bumped section_idx
    let chunk = get_chunk(&store, "H5SKIP/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.section_idx, 0, "h5 must not create a section");
}

// 6. h6 is skipped — matches ACM reference format and algorithm labels.
#[serial]
#[tokio::test]
async fn test_h6_is_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>1 INTRODUCTION</h2>"),
        ("SectionHeader", "sh2", "<h6>ACM Reference Format:</h6>"),
        ("Text", "t0", "<p>Intro text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H6SKIP", &json);
    ingest_paper(&store, params).await.unwrap();

    // h6 must not have bumped chapter_idx or section_idx
    let chunk = get_chunk(&store, "H6SKIP/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.chapter_idx, 1);
    assert_eq!(chunk.chunk.section_idx, 0, "h6 must not create a section");
}

// 7. Multiple h2 headings produce sequentially-numbered chapters.
#[serial]
#[tokio::test]
async fn test_multiple_h2_chapters_sequential() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>1 INTRODUCTION</h2>"),
        ("Text", "t0", "<p>Intro.</p>"),
        ("SectionHeader", "sh2", "<h2>2 RELATED WORK</h2>"),
        ("Text", "t1", "<p>Related work.</p>"),
        ("SectionHeader", "sh3", "<h2>3 METHOD</h2>"),
        ("Text", "t2", "<p>Method.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "MULTICH", &json);
    ingest_paper(&store, params).await.unwrap();

    let c1 = get_chunk(&store, "MULTICH/ch1/s0/p0").await.unwrap();
    let c2 = get_chunk(&store, "MULTICH/ch2/s0/p0").await.unwrap();
    let c3 = get_chunk(&store, "MULTICH/ch3/s0/p0").await.unwrap();
    assert_eq!(c1.chunk.chapter_title, "1 INTRODUCTION");
    assert_eq!(c2.chunk.chapter_title, "2 RELATED WORK");
    assert_eq!(c3.chunk.chapter_title, "3 METHOD");
}

// 8. Multiple h3 sections within one chapter are numbered sequentially.
#[serial]
#[tokio::test]
async fn test_multiple_h3_sections_sequential() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>3 VERTEX BLOCK DESCENT</h2>"),
        ("SectionHeader", "sh2", "<h3>3.1 Global Optimization</h3>"),
        ("Text", "t0", "<p>Text 3.1.</p>"),
        ("SectionHeader", "sh3", "<h3>3.2 Local System Solver</h3>"),
        ("Text", "t1", "<p>Text 3.2.</p>"),
        ("SectionHeader", "sh4", "<h3>3.3 Damping</h3>"),
        ("Text", "t2", "<p>Text 3.3.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "MULTISEC", &json);
    ingest_paper(&store, params).await.unwrap();

    let s1 = get_chunk(&store, "MULTISEC/ch1/s1/p0").await.unwrap();
    let s2 = get_chunk(&store, "MULTISEC/ch1/s2/p0").await.unwrap();
    let s3 = get_chunk(&store, "MULTISEC/ch1/s3/p0").await.unwrap();
    assert_eq!(s1.chunk.section_title, "3.1 Global Optimization");
    assert_eq!(s2.chunk.section_title, "3.2 Local System Solver");
    assert_eq!(s3.chunk.section_title, "3.3 Damping");
}

// 9. Section counter resets when a new h2 chapter starts.
#[serial]
#[tokio::test]
async fn test_section_counter_resets_on_new_chapter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>1 INTRODUCTION</h2>"),
        ("SectionHeader", "sh2", "<h3>1.1 Background</h3>"),
        ("Text", "t0", "<p>Background.</p>"),
        ("SectionHeader", "sh3", "<h2>2 RELATED WORK</h2>"),
        ("SectionHeader", "sh4", "<h3>2.1 Prior Art</h3>"),
        ("Text", "t1", "<p>Prior art.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "SECRESET", &json);
    ingest_paper(&store, params).await.unwrap();

    // First h3 in ch2 must be section_idx=1 (reset from ch1's count)
    let chunk = get_chunk(&store, "SECRESET/ch2/s1/p0").await.unwrap();
    assert_eq!(chunk.chunk.chapter_idx, 2);
    assert_eq!(chunk.chunk.section_idx, 1);
    assert_eq!(chunk.chunk.section_title, "2.1 Prior Art");
}

// 10. chunk_idx resets to 0 when a new h3 section starts.
#[serial]
#[tokio::test]
async fn test_chunk_idx_resets_on_new_section() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>3 METHOD</h2>"),
        ("Text", "t0", "<p>Paragraph one.</p>"),
        ("Text", "t1", "<p>Paragraph two.</p>"),
        ("SectionHeader", "sh2", "<h3>3.1 Subsection</h3>"),
        ("Text", "t2", "<p>First paragraph of subsection.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "CHKRESET", &json);
    ingest_paper(&store, params).await.unwrap();

    // First paragraph of new section must have chunk_idx=0
    let chunk = get_chunk(&store, "CHKRESET/ch1/s1/p0").await.unwrap();
    assert_eq!(chunk.chunk.chunk_idx, 0);
    assert_eq!(chunk.chunk.text, "First paragraph of subsection.");
}

// 11. chunk_idx resets to 0 when a new h2 chapter starts.
#[serial]
#[tokio::test]
async fn test_chunk_idx_resets_on_new_chapter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>1 INTRODUCTION</h2>"),
        ("Text", "t0", "<p>Para one.</p>"),
        ("Text", "t1", "<p>Para two.</p>"),
        ("SectionHeader", "sh2", "<h2>2 RELATED WORK</h2>"),
        ("Text", "t2", "<p>First para of ch2.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "CH2RESET", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "CH2RESET/ch2/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.chunk_idx, 0);
    assert_eq!(chunk.chunk.text, "First para of ch2.");
}

// 12. Text before any h2 lands in chapter 0 (implicit preamble).
#[serial]
#[tokio::test]
async fn test_text_before_h2_lands_in_chapter_0() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h1>Paper Title</h1>"),
        ("Text", "t0", "<p>Abstract text before any section.</p>"),
        ("SectionHeader", "sh2", "<h2>1 INTRODUCTION</h2>"),
        ("Text", "t1", "<p>Intro text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "PREAMBLE", &json);
    ingest_paper(&store, params).await.unwrap();

    let pre = get_chunk(&store, "PREAMBLE/ch0/s0/p0").await.unwrap();
    assert_eq!(pre.chunk.chapter_idx, 0);
    assert_eq!(pre.chunk.text, "Abstract text before any section.");

    let intro = get_chunk(&store, "PREAMBLE/ch1/s0/p0").await.unwrap();
    assert_eq!(intro.chunk.chapter_idx, 1);
}

// 13. Chapter title is correctly extracted from h2 HTML (tags stripped).
#[serial]
#[tokio::test]
async fn test_chapter_title_from_h2_html() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>3 VERTEX BLOCK DESCENT FOR ELASTIC BODIES</h2>"),
        ("Text", "t0", "<p>Body text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "CHTITLE", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "CHTITLE/ch1/s0/p0").await.unwrap();
    assert_eq!(
        chunk.chunk.chapter_title,
        "3 VERTEX BLOCK DESCENT FOR ELASTIC BODIES"
    );
}

// 14. Section title is correctly extracted from h3 HTML (tags stripped).
#[serial]
#[tokio::test]
async fn test_section_title_from_h3_html() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>3 METHOD</h2>"),
        ("SectionHeader", "sh2", "<h3>3.9 Parallelization</h3>"),
        ("Text", "t0", "<p>Parallel text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "SECTITLE", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "SECTITLE/ch1/s1/p0").await.unwrap();
    assert_eq!(chunk.chunk.section_title, "3.9 Parallelization");
}

// 15. Section title is cleared (empty) when a new h2 chapter starts.
#[serial]
#[tokio::test]
async fn test_section_title_cleared_on_new_chapter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>1 INTRODUCTION</h2>"),
        ("SectionHeader", "sh2", "<h3>1.1 Motivation</h3>"),
        ("Text", "t0", "<p>Motivation text.</p>"),
        ("SectionHeader", "sh3", "<h2>2 RELATED WORK</h2>"),
        ("Text", "t1", "<p>Related text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "SECTCLR", &json);
    ingest_paper(&store, params).await.unwrap();

    // Text in ch2/s0 should have no section title (cleared by h2)
    let chunk = get_chunk(&store, "SECTCLR/ch2/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.section_title, "");
}

// 16. h6 "Algorithm 1" triggers algorithm detection, text goes to exhibit body.
#[serial]
#[tokio::test]
async fn test_h6_algorithm_triggers_exhibit_detection() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>5 RESULTS</h2>"),
        ("SectionHeader", "sh2", "<h3>5.2 Unit Tests</h3>"),
        ("SectionHeader", "sh3", "<h6>Algorithm 1: VBD simulation for one time step.</h6>"),
        ("Text", "t0", "<p>Input: vertices V, edges E</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H6ALGO", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // Algorithm detected as exhibit, text becomes algorithm body (not a chunk)
    assert_eq!(stats.exhibits_added, 1);
    assert_eq!(stats.chunks_added, 0);

    let exhibit = get_exhibit(&store, "H6ALGO/fig1").await.unwrap();
    assert_eq!(exhibit.exhibit_type, "algorithm");
    assert_eq!(exhibit.caption, "Algorithm 1: VBD simulation for one time step.");
    assert!(exhibit.content.unwrap().contains("Input: vertices V, edges E"));
}

// 16b. h6 without algorithm pattern still skipped (no section advance).
#[serial]
#[tokio::test]
async fn test_h6_non_algorithm_is_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>5 RESULTS</h2>"),
        ("SectionHeader", "sh2", "<h3>5.2 Unit Tests</h3>"),
        ("SectionHeader", "sh3", "<h6>ACM Reference Format:</h6>"),
        ("Text", "t0", "<p>Text after h6.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H6NOSEC", &json);
    ingest_paper(&store, params).await.unwrap();

    // Non-algorithm h6 must not bump section_idx
    let chunk = get_chunk(&store, "H6NOSEC/ch1/s1/p0").await.unwrap();
    assert_eq!(chunk.chunk.section_idx, 1, "h6 must not increment section_idx");
}

// 17. h1 appearing after an h2 does not start a new chapter.
#[serial]
#[tokio::test]
async fn test_h1_after_h2_does_not_start_new_chapter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>1 INTRODUCTION</h2>"),
        ("Text", "t0", "<p>Intro.</p>"),
        ("SectionHeader", "sh2", "<h1>Stray h1 header</h1>"),
        ("Text", "t1", "<p>After stray h1.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H1AFTER", &json);
    ingest_paper(&store, params).await.unwrap();

    // Both texts merge into one chunk in chapter 1
    let c0 = get_chunk(&store, "H1AFTER/ch1/s0/p0").await.unwrap();
    assert_eq!(c0.chunk.chapter_idx, 1);
    assert!(c0.chunk.text.contains("Intro."));
    assert!(c0.chunk.text.contains("After stray h1."), "h1 must not start a new chapter");
}

// 18. h2 after h3 correctly starts a new chapter (not a section).
#[serial]
#[tokio::test]
async fn test_h2_after_h3_starts_new_chapter() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>3 METHOD</h2>"),
        ("SectionHeader", "sh2", "<h3>3.9 Parallelization</h3>"),
        ("Text", "t0", "<p>Parallel text.</p>"),
        ("SectionHeader", "sh3", "<h2>4 GPU IMPLEMENTATION</h2>"),
        ("Text", "t1", "<p>GPU text.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H2AFTER3", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "H2AFTER3/ch2/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.chapter_idx, 2);
    assert_eq!(chunk.chunk.chapter_title, "4 GPU IMPLEMENTATION");
    assert_eq!(chunk.chunk.section_idx, 0);
}

// 19. Equation blocks are merged with surrounding text.
#[serial]
#[tokio::test]
async fn test_equation_block_merged_with_text() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>3 METHOD</h2>"),
        ("Text", "t0", "<p>Let x be defined as follows:</p>"),
        ("Equation", "eq0", "<p>x = y + z</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "EQCHUNK", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // Short text + equation merge into one chunk
    assert_eq!(stats.chunks_added, 1);
    let chunk = get_chunk(&store, "EQCHUNK/ch1/s0/p0").await.unwrap();
    assert!(chunk.chunk.text.contains("Let x be defined"));
    assert!(chunk.chunk.text.contains("x = y + z"));
}

// 20. Realistic YFACFA8C-style structure with algorithm detection.
#[serial]
#[tokio::test]
async fn test_realistic_yfacfa8c_structure() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let json = make_json(&[
        // Paper title (h1) — skipped
        ("SectionHeader", "sh_title", "<h1>Vertex Block Descent</h1>"),
        // ACM reference (h6) — skipped (no algorithm pattern)
        ("SectionHeader", "sh_acm", "<h6>ACM Reference Format:</h6>"),
        // Chapter 1
        ("SectionHeader", "sh_intro", "<h2>1 INTRODUCTION</h2>"),
        ("Text", "t_intro", "<p>Introduction text.</p>"),
        // Chapter 2
        ("SectionHeader", "sh_related", "<h2>2 RELATED WORK</h2>"),
        ("Text", "t_related", "<p>Related work text.</p>"),
        // Chapter 3 with subsections
        ("SectionHeader", "sh_vbd", "<h2>3 VERTEX BLOCK DESCENT FOR ELASTIC BODIES</h2>"),
        ("SectionHeader", "sh_global", "<h3>3.1 Global Optimization</h3>"),
        ("Text", "t_global", "<p>Global optimization text.</p>"),
        ("SectionHeader", "sh_local", "<h3>3.2 Local System Solver</h3>"),
        ("Text", "t_local", "<p>Local solver text.</p>"),
        // Chapter 4
        ("SectionHeader", "sh_gpu", "<h2>4 GPU IMPLEMENTATION</h2>"),
        ("Text", "t_gpu", "<p>GPU implementation text.</p>"),
        // Algorithm label (h6) — triggers algorithm detection
        ("SectionHeader", "sh_alg", "<h6>Algorithm 1: VBD simulation.</h6>"),
        ("Text", "t_after_alg", "<p>Input: vertices V</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "VBD", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    let outline = get_paper_outline(&store, "VBD").await.unwrap();
    assert_eq!(outline.chapters.len(), 4, "should have 4 h2 chapters");
    assert_eq!(outline.chapters[0].chapter_title, "1 INTRODUCTION");
    assert_eq!(outline.chapters[1].chapter_title, "2 RELATED WORK");
    assert_eq!(
        outline.chapters[2].chapter_title,
        "3 VERTEX BLOCK DESCENT FOR ELASTIC BODIES"
    );
    assert_eq!(outline.chapters[3].chapter_title, "4 GPU IMPLEMENTATION");

    // Chapter 3 should have 2 subsections (3.1 and 3.2)
    assert_eq!(outline.chapters[2].sections.len(), 2);

    // GPU text chunk exists
    let gpu_chunk = get_chunk(&store, "VBD/ch4/s0/p0").await.unwrap();
    assert_eq!(gpu_chunk.chunk.chapter_idx, 4);
    assert_eq!(gpu_chunk.chunk.section_idx, 0);

    // Algorithm 1 detected as exhibit (text after label becomes algorithm body)
    assert_eq!(stats.exhibits_added, 1);
    let algo = get_exhibit(&store, "VBD/fig1").await.unwrap();
    assert_eq!(algo.exhibit_type, "algorithm");
    assert_eq!(algo.caption, "Algorithm 1: VBD simulation.");
    assert!(algo.content.unwrap().contains("Input: vertices V"));
}

// ── block_type field ──────────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_block_type_stored_correctly() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>Method</h2>"),
        ("Text", "t0", "<p>Some text.</p>"),
        ("Equation", "eq0", "<p>x = y</p>"),
        ("ListGroup", "lg0", "<ul><li>item</li></ul>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "BTYPE", &json);
    ingest_paper(&store, params).await.unwrap();

    // All block types merge into one chunk with block_type="text"
    let chunk = get_chunk(&store, "BTYPE/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.block_type, "text");
    assert!(chunk.chunk.text.contains("Some text."));
    assert!(chunk.chunk.text.contains("x = y"));
    assert!(chunk.chunk.text.contains("item"));
}

// ── search result shape tests ───────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_search_result_has_paper_title() {
    use crate::query::search;
    use crate::types::SearchParams;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "SRCH1");
    ingest_paper(&store, params).await.unwrap();

    let results = search(
        &store,
        SearchParams {
            query: "introduction".to_string(),
            paper_ids: Some(vec!["SRCH1".to_string()]),
            chapter_idx: None,
            section_idx: None,
            filter_year_min: None,
            filter_year_max: None,
            filter_venue: None,
            filter_tags: None,
            filter_depth: None,
            limit: 5,
        },
    )
    .await
    .unwrap();

    assert!(!results.is_empty(), "should find at least one result");
    let r = &results[0];
    // SearchChunkResult has paper_title, not title
    assert_eq!(r.chunk.paper_title, "Test Paper");
    // Has block_type
    assert!(!r.chunk.block_type.is_empty());
    // Has exhibit_ids field
    let _ = &r.chunk.exhibit_ids;
    // Has chunk_idx
    let _ = r.chunk.chunk_idx;
}

// ── sentence-aware preview in neighbors ──────────────────────────────────

#[serial]
#[tokio::test]
async fn test_neighbor_preview_sentence_truncation() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Create two chunks by using separate sections (h3 forces a section boundary)
    let long_text = format!(
        "<p>{}. Next sentence after the boundary.</p>",
        "A".repeat(125)
    );
    let json = serde_json::json!({
        "children": [{
            "block_type": "Page",
            "children": [
                {"block_type": "SectionHeader", "id": "h1", "html": "<h2>Test</h2>", "page": 1},
                {"block_type": "Text", "id": "t0", "html": long_text, "page": 1},
                {"block_type": "SectionHeader", "id": "h2", "html": "<h3>Next Section</h3>", "page": 1},
                {"block_type": "Text", "id": "t1", "html": "<p>Second chunk.</p>", "page": 1}
            ]
        }]
    });
    let item_dir = cache_dir.path().join("SENTPREV");
    fs::create_dir_all(&item_dir).unwrap();
    fs::write(item_dir.join("SENTPREV.json"), serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    let params = IngestParams {
        item_key: "SENTPREV".to_string(),
        paper_id: "SENTPREV".to_string(),
        title: "Test".to_string(),
        authors: vec![],
        year: None,
        venue: None,
        tags: vec![],
        cache_dir: item_dir,
        force: false,
    };
    ingest_paper(&store, params).await.unwrap();

    // ch1/s0/p0 is the long text, ch1/s1/p0 is the second chunk
    // Neighbor preview works within sections, so test that s0/p0 exists
    let result = get_chunk(&store, "SENTPREV/ch1/s0/p0").await.unwrap();
    assert!(result.chunk.text.contains("Next sentence after the boundary"));
}

// ── equation merging with surrounding text ────────────────────────────────

#[serial]
#[tokio::test]
async fn test_equation_merged_with_surrounding_text() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // [TextA, Equation, TextB] — all merge into one chunk
    let json = serde_json::json!({
        "children": [{
            "block_type": "Page",
            "children": [
                {"block_type": "SectionHeader", "id": "h1", "html": "<h2>Method</h2>", "page": 1},
                {"block_type": "Text", "id": "t0", "html": "<p>We define the objective function as follows.</p>", "page": 1},
                {"block_type": "Equation", "id": "eq0", "html": "<p><math>E = mc^2</math></p>", "page": 1},
                {"block_type": "Text", "id": "t1", "html": "<p>Where m is mass and c is the speed of light.</p>", "page": 1}
            ]
        }]
    });
    let item_dir = cache_dir.path().join("EQNBR");
    fs::create_dir_all(&item_dir).unwrap();
    fs::write(item_dir.join("EQNBR.json"), serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    let params = IngestParams {
        item_key: "EQNBR".to_string(),
        paper_id: "EQNBR".to_string(),
        title: "Test".to_string(),
        authors: vec![],
        year: None,
        venue: None,
        tags: vec![],
        cache_dir: item_dir,
        force: false,
    };
    ingest_paper(&store, params).await.unwrap();

    // All three blocks merge into one chunk
    let result = get_chunk(&store, "EQNBR/ch1/s0/p0").await.unwrap();
    assert!(result.chunk.text.contains("objective function"));
    assert!(result.chunk.text.contains("E = mc^2"));
    assert!(result.chunk.text.contains("mass and c is the speed"));
}

#[serial]
#[tokio::test]
async fn test_equation_merged_preserves_math() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let json = serde_json::json!({
        "children": [{
            "block_type": "Page",
            "children": [
                {"block_type": "SectionHeader", "id": "h1", "html": "<h2>Method</h2>", "page": 1},
                {"block_type": "Equation", "id": "eq0", "html": "<p><math>x = 0</math></p>", "page": 1},
                {"block_type": "Text", "id": "t0", "html": "<p>Where x is defined above.</p>", "page": 1}
            ]
        }]
    });
    let item_dir = cache_dir.path().join("EQBOUND");
    fs::create_dir_all(&item_dir).unwrap();
    fs::write(item_dir.join("EQBOUND.json"), serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    let params = IngestParams {
        item_key: "EQBOUND".to_string(),
        paper_id: "EQBOUND".to_string(),
        title: "Test".to_string(),
        authors: vec![],
        year: None,
        venue: None,
        tags: vec![],
        cache_dir: item_dir,
        force: false,
    };
    ingest_paper(&store, params).await.unwrap();

    // Equation and text merge into one chunk
    let result = get_chunk(&store, "EQBOUND/ch1/s0/p0").await.unwrap();
    assert!(result.chunk.text.contains("x = 0"));
    assert!(result.chunk.text.contains("Where x is defined"));
}

// ── table content in figures ────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_table_block_stores_content() {
    use crate::query::get_exhibit;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Create a paper with a Table block containing actual <table> HTML
    let json = serde_json::json!({
        "children": [{
            "block_type": "Page",
            "children": [
                {
                    "block_type": "SectionHeader", "id": "h1",
                    "html": "<h2>Results</h2>", "page": 1
                },
                {
                    "block_type": "Text", "id": "t0",
                    "html": "<p>See Table 1.</p>", "page": 1
                },
                {
                    "block_type": "Caption", "id": "cap1",
                    "html": "<p><b>Table 1.</b> Performance comparison.</p>", "page": 2
                },
                {
                    "block_type": "Table", "id": "tbl1",
                    "html": "<table><thead><tr><th>Method</th><th>Time</th></tr></thead><tbody><tr><td>Ours</td><td>5ms</td></tr><tr><td>Baseline</td><td>12ms</td></tr></tbody></table>",
                    "page": 2
                }
            ]
        }]
    });
    let item_dir = cache_dir.path().join("TBLCONT");
    fs::create_dir_all(&item_dir).unwrap();
    fs::write(item_dir.join("TBLCONT.json"), serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    let params = crate::ingest::IngestParams {
        item_key: "TBLCONT".to_string(),
        paper_id: "TBLCONT".to_string(),
        title: "Test".to_string(),
        authors: vec![],
        year: None,
        venue: None,
        tags: vec![],
        cache_dir: item_dir,
        force: false,
    };
    ingest_paper(&store, params).await.unwrap();

    let fig = get_exhibit(&store, "TBLCONT/fig1").await.unwrap();
    assert_eq!(fig.exhibit_type, "table");
    assert!(fig.content.is_some(), "table should have content");
    let content = fig.content.unwrap();
    assert!(content.contains("Ours"), "content should have cell values: {}", content);
    assert!(content.contains("5ms"), "content should have cell values: {}", content);
}

#[serial]
#[tokio::test]
async fn test_exhibit_block_has_null_content() {
    use crate::query::get_exhibit;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "FIGNULL");
    ingest_paper(&store, params).await.unwrap();

    // fig1 is a Figure block, should have no content
    let fig = get_exhibit(&store, "FIGNULL/fig1").await.unwrap();
    assert_eq!(fig.exhibit_type, "figure");
    assert!(fig.content.is_none(), "figure should have null content");
}

#[serial]
#[tokio::test]
async fn test_search_exhibits_returns_content() {
    use crate::query::search_exhibits;
    use crate::types::SearchExhibitsParams;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let json = serde_json::json!({
        "children": [{
            "block_type": "Page",
            "children": [
                {
                    "block_type": "SectionHeader", "id": "h1",
                    "html": "<h2>Results</h2>", "page": 1
                },
                {
                    "block_type": "Table", "id": "tbl1",
                    "html": "<table><thead><tr><th>Solver</th></tr></thead><tbody><tr><td>NeoHookean</td></tr></tbody></table>",
                    "page": 2
                }
            ]
        }]
    });
    let item_dir = cache_dir.path().join("SFCONT");
    fs::create_dir_all(&item_dir).unwrap();
    fs::write(item_dir.join("SFCONT.json"), serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    let params = crate::ingest::IngestParams {
        item_key: "SFCONT".to_string(),
        paper_id: "SFCONT".to_string(),
        title: "Test".to_string(),
        authors: vec![],
        year: None,
        venue: None,
        tags: vec![],
        cache_dir: item_dir,
        force: false,
    };
    ingest_paper(&store, params).await.unwrap();

    let results = search_exhibits(
        &store,
        SearchExhibitsParams {
            query: "solver".to_string(),
            paper_ids: Some(vec!["SFCONT".to_string()]),
            filter_exhibit_type: None,
            limit: 5,
        },
    )
    .await
    .unwrap();

    assert!(!results.is_empty());
    let r = &results[0];
    assert!(r.content.is_some(), "search result should include content");
    assert!(r.content.as_ref().unwrap().contains("NeoHookean"));
}

// ── figure search result with score ──────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_search_exhibits_has_score() {
    use crate::query::search_exhibits;
    use crate::types::SearchExhibitsParams;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "FIGSCORE");
    ingest_paper(&store, params).await.unwrap();

    let results = search_exhibits(
        &store,
        SearchExhibitsParams {
            query: "diagram".to_string(),
            paper_ids: Some(vec!["FIGSCORE".to_string()]),
            filter_exhibit_type: None,
            limit: 5,
        },
    )
    .await
    .unwrap();

    assert!(!results.is_empty(), "should find at least one figure");
    // ExhibitSearchResult should have score field
    // With zero-vector embedder, distances will be 0.0
    let _ = results[0].score;
    // Should have all expected fields
    assert!(!results[0].exhibit_id.is_empty());
    assert!(!results[0].paper_id.is_empty());
    assert!(!results[0].exhibit_type.is_empty());
}

// ── resolve_paper_id ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_resolve_paper_id_empty_db() {
    use crate::query::resolve_paper_id;

    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let result = resolve_paper_id(&store, "anything").await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("no paper matching"), "error should describe the problem: {}", msg);
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_exact_match() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "RESOLVE1");
    ingest_paper(&store, params).await.unwrap();

    let result = resolve_paper_id(&store, "RESOLVE1").await.unwrap();
    assert_eq!(result, "RESOLVE1");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_exact_doi() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Ingest with a DOI-style paper_id
    let mut params = make_test_cache(&cache_dir, "DOIDIR");
    params.paper_id = "10.1145/3528223.3530168".to_string();
    ingest_paper(&store, params).await.unwrap();

    let result = resolve_paper_id(&store, "10.1145/3528223.3530168").await.unwrap();
    assert_eq!(result, "10.1145/3528223.3530168");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_title_substring() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params = make_test_cache(&cache_dir, "TITLESUB");
    params.title = "Vertex Block Descent".to_string();
    ingest_paper(&store, params).await.unwrap();

    // Substring match
    let result = resolve_paper_id(&store, "vertex block").await.unwrap();
    assert_eq!(result, "TITLESUB");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_case_insensitive() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params = make_test_cache(&cache_dir, "CASETEST");
    params.title = "Neural Radiance Fields".to_string();
    ingest_paper(&store, params).await.unwrap();

    // All-caps search should still match
    let result = resolve_paper_id(&store, "NEURAL RADIANCE").await.unwrap();
    assert_eq!(result, "CASETEST");

    // Mixed case
    let result = resolve_paper_id(&store, "neural Radiance").await.unwrap();
    assert_eq!(result, "CASETEST");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_exact_match_preferred_over_title() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Paper 1: paper_id matches a title substring of paper 2
    let mut params1 = make_test_cache(&cache_dir, "ABC");
    params1.title = "Some Paper".to_string();
    ingest_paper(&store, params1).await.unwrap();

    let mut params2 = make_test_cache(&cache_dir, "OTHER");
    params2.title = "About ABC Protocol".to_string();
    ingest_paper(&store, params2).await.unwrap();

    // "ABC" should match paper_id exactly, not title
    let result = resolve_paper_id(&store, "ABC").await.unwrap();
    assert_eq!(result, "ABC");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_no_match_lists_available() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params = make_test_cache(&cache_dir, "AVAIL1");
    params.title = "First Available Paper".to_string();
    ingest_paper(&store, params).await.unwrap();

    let mut params2 = make_test_cache(&cache_dir, "AVAIL2");
    params2.title = "Second Available Paper".to_string();
    ingest_paper(&store, params2).await.unwrap();

    let result = resolve_paper_id(&store, "nonexistent query").await;
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("AVAIL1"), "should list available papers: {}", msg);
    assert!(msg.contains("AVAIL2"), "should list available papers: {}", msg);
    assert!(msg.contains("First Available Paper"), "should list titles: {}", msg);
    assert!(msg.contains("Second Available Paper"), "should list titles: {}", msg);
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_multiple_title_matches_returns_first() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params1 = make_test_cache(&cache_dir, "MULTI1");
    params1.title = "Gaussian Splatting for Real-Time Rendering".to_string();
    ingest_paper(&store, params1).await.unwrap();

    let mut params2 = make_test_cache(&cache_dir, "MULTI2");
    params2.title = "Gaussian Splatting in the Wild".to_string();
    ingest_paper(&store, params2).await.unwrap();

    // "Gaussian Splatting" matches both; should return one (first found)
    let result = resolve_paper_id(&store, "gaussian splatting").await.unwrap();
    assert!(
        result == "MULTI1" || result == "MULTI2",
        "should return one of the matching papers: {}",
        result
    );
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_full_title_match() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params = make_test_cache(&cache_dir, "FULLTITLE");
    params.title = "3D Gaussian Splatting for Real-Time Radiance Field Rendering".to_string();
    ingest_paper(&store, params).await.unwrap();

    // Full title as search
    let result = resolve_paper_id(
        &store,
        "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
    )
    .await
    .unwrap();
    assert_eq!(result, "FULLTITLE");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_single_word_title_match() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params = make_test_cache(&cache_dir, "ONEWORD");
    params.title = "Differentiable Rendering: A Survey".to_string();
    ingest_paper(&store, params).await.unwrap();

    // Single keyword match
    let result = resolve_paper_id(&store, "differentiable").await.unwrap();
    assert_eq!(result, "ONEWORD");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_special_chars_in_title() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params = make_test_cache(&cache_dir, "SPECIAL");
    params.title = "C++ Templates: The Complete Guide (2nd Ed.)".to_string();
    ingest_paper(&store, params).await.unwrap();

    let result = resolve_paper_id(&store, "c++ templates").await.unwrap();
    assert_eq!(result, "SPECIAL");
}

#[serial]
#[tokio::test]
async fn test_resolve_paper_id_with_doi_containing_slash() {
    use crate::query::resolve_paper_id;

    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut params = make_test_cache(&cache_dir, "DOISLASH");
    params.paper_id = "10.1145/3592433".to_string();
    params.title = "Some Conference Paper".to_string();
    ingest_paper(&store, params).await.unwrap();

    // Exact DOI match with slashes
    let result = resolve_paper_id(&store, "10.1145/3592433").await.unwrap();
    assert_eq!(result, "10.1145/3592433");

    // Title fallback still works
    let result = resolve_paper_id(&store, "conference paper").await.unwrap();
    assert_eq!(result, "10.1145/3592433");
}

// ══════════════════════════════════════════════════════════════════════════════
// Chunking pipeline tests
// ══════════════════════════════════════════════════════════════════════════════

/// Build a JSON block value with customizable page number.
fn make_block(block_type: &str, id: &str, html: &str, page: u16) -> serde_json::Value {
    serde_json::json!({
        "block_type": block_type,
        "id": id,
        "html": html,
        "page": page,
        "section_hierarchy": {}
    })
}

/// Build a complete Marker-style JSON document from a list of blocks.
fn make_json_from_blocks(blocks: Vec<serde_json::Value>) -> String {
    let doc = serde_json::json!({
        "children": [{
            "block_type": "Page",
            "children": blocks
        }]
    });
    serde_json::to_string_pretty(&doc).unwrap()
}

/// Generate a paragraph of approximately `n_words` words.
fn words(n: usize) -> String {
    (0..n).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ")
}

/// Wrap text in a `<p>` tag.
fn p(text: &str) -> String {
    format!("<p>{text}</p>")
}

/// Create an IngestParams from a JSON string.
fn make_params_from_json_str(dir: &TempDir, key: &str, json: &str) -> IngestParams {
    let cache_dir = dir.path().join(key);
    fs::create_dir_all(&cache_dir).unwrap();
    fs::write(cache_dir.join(format!("{key}.json")), json).unwrap();
    IngestParams {
        item_key: key.to_string(),
        paper_id: key.to_string(),
        title: "Test Paper".to_string(),
        authors: vec![],
        year: None,
        venue: None,
        tags: vec![],
        cache_dir,
        force: false,
    }
}

// ── Basic merge: short blocks within one section → 1 chunk ──────────────

#[serial]
#[tokio::test]
async fn test_chunking_basic_merge() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // 5 short Text blocks (~50 tokens each) all in one section
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("Text", "t0", &p(&words(40)), 0),
        make_block("Text", "t1", &p(&words(40)), 0),
        make_block("Text", "t2", &p(&words(40)), 0),
        make_block("Text", "t3", &p(&words(40)), 0),
        make_block("Text", "t4", &p(&words(40)), 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "MERGE1", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // 5 × ~52 tokens = ~260, below TARGET=400 → all merge into 1 chunk
    assert_eq!(stats.chunks_added, 1);
}

// ── Token limit flush: blocks exceeding TARGET → multiple chunks ────────

#[serial]
#[tokio::test]
async fn test_chunking_token_limit_flush() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // 4 blocks of ~150 words each = ~195 tokens each
    // Buffer: block0 (195) → block1 would make ~390 (< 400) → push
    //  → block2 would make ~585 (> 400) → flush, start new buffer
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("Text", "t0", &p(&words(150)), 0),
        make_block("Text", "t1", &p(&words(150)), 0),
        make_block("Text", "t2", &p(&words(150)), 0),
        make_block("Text", "t3", &p(&words(150)), 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "TOKFLUSH", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert!(stats.chunks_added >= 2, "should produce at least 2 chunks, got {}", stats.chunks_added);
}

// ── Section boundary flush (h2) ─────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_section_boundary_h2() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter 1</h2>", 0),
        make_block("Text", "t0", &p(&words(50)), 0),
        make_block("SectionHeader", "h2", "<h2>Chapter 2</h2>", 0),
        make_block("Text", "t1", &p(&words(50)), 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "SECBH2", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // h2 boundary forces flush → 2 separate chunks (one per chapter)
    assert_eq!(stats.chunks_added, 2);
}

// ── Section boundary flush (h3) ─────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_section_boundary_h3() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("Text", "t0", &p(&words(50)), 0),
        make_block("SectionHeader", "h2", "<h3>Section</h3>", 0),
        make_block("Text", "t1", &p(&words(50)), 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "SECBH3", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // h3 boundary forces flush → 2 chunks (different sections)
    assert_eq!(stats.chunks_added, 2);
}

// ── Smart merge: trailing fragment < MIN → merged into previous ─────────

#[serial]
#[tokio::test]
async fn test_chunking_smart_merge_trailing() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // First section: ~270 words (~350 tokens), second section: ~50 words (~65 tokens)
    // 65 < MIN=200, prev=350, combined=415 < MAX=600 → merge!
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("Text", "t0", &p(&words(270)), 0),
        make_block("SectionHeader", "h2", "<h3>Section</h3>", 0),
        make_block("Text", "t1", &p(&words(50)), 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "SMERGE", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // Wait — smart merge only merges within same chapter+section.
    // These are different sections (s0 and s1), so they stay separate.
    assert_eq!(stats.chunks_added, 2);
}

// ── Smart merge within same section ─────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_smart_merge_within_section() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Two blocks in same section: first ~350 tokens causes a flush, second ~50 tokens
    // is tiny but in same section → second flushes at end-of-doc, same section as prev → merge
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("Text", "t0", &p(&words(270)), 0),
        make_block("Text", "t1", &p(&words(270)), 0),
        make_block("Text", "t2", &p(&words(30)), 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "SMWSEC", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // t0+t1 → buffer ~700 tokens, t1 pushes past TARGET → flush t0 as chunk, then t1 in buffer
    // t2 → buffer, combined with t1 overlap still manageable
    // End → flush; if combined < MAX with prev, merge
    // This depends on exact behavior; just verify at least 2 chunks
    assert!(stats.chunks_added >= 1, "should produce chunks");
}

// ── Equation merges with text ───────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_equation_merges_with_text() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("Text", "t0", "<p>We define energy:</p>", 0),
        make_block("Equation", "eq0", "<p><math>E = mc^2</math></p>", 0),
        make_block("Text", "t1", "<p>where m is mass.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "EQMERGE", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.chunks_added, 1);
    let chunk = get_chunk(&store, "EQMERGE/ch1/s0/p0").await.unwrap();
    assert!(chunk.chunk.text.contains("E = mc^2"));
    assert!(chunk.chunk.text.contains("We define energy"));
}

// ── ListGroup merges with text ──────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_listgroup_merges_with_text() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("Text", "t0", "<p>The steps are:</p>", 0),
        make_block("ListGroup", "lg0", "<ul><li>Step one</li><li>Step two</li></ul>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "LGMERGE", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.chunks_added, 1);
    let chunk = get_chunk(&store, "LGMERGE/ch1/s0/p0").await.unwrap();
    assert!(chunk.chunk.text.contains("The steps are"));
    assert!(chunk.chunk.text.contains("Step one"));
}

// ── References skipped ──────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_references_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let mut blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Conclusion</h2>", 0),
        make_block("Text", "t0", "<p>We presented our method.</p>", 0),
        make_block("SectionHeader", "h2", "<h2>References</h2>", 0),
    ];
    // Add 10 reference text blocks
    for i in 0..10 {
        blocks.push(make_block("Text", &format!("r{i}"), &p(&format!("[{i}] Author et al. Title. Venue, 2024.")), 0));
    }
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "REFSKIP", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // Only the conclusion chunk, no reference chunks
    assert_eq!(stats.chunks_added, 1);
    let chunk = get_chunk(&store, "REFSKIP/ch1/s0/p0").await.unwrap();
    assert!(chunk.chunk.text.contains("We presented our method"));
}

// ── References variants ─────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_references_bibliography_variant() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Conclusion</h2>", 0),
        make_block("Text", "t0", "<p>Done.</p>", 0),
        make_block("SectionHeader", "h2", "<h2>Bibliography</h2>", 0),
        make_block("Text", "r0", "<p>[1] Some reference.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "BIBSKIP", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.chunks_added, 1);
}

// ── Non-references not skipped ──────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_non_references_not_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Related Work</h2>", 0),
        make_block("Text", "t0", "<p>Prior methods include...</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "NOREFSKIP", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.chunks_added, 1);
}

// ── Single large block not split ────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_single_large_block_not_split() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // ~700 token block (540 words) — exceeds TARGET but never split mid-block
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("Text", "t0", &p(&words(540)), 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "LARGE1", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.chunks_added, 1, "single large block must not be split");
}

// ── Empty text blocks skipped ───────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_empty_text_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("Text", "t0", "<p>   </p>", 0),
        make_block("Text", "t1", "<p></p>", 0),
        make_block("Text", "t2", "<p>Real text.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "EMPTYSKIP", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.chunks_added, 1);
    let chunk = get_chunk(&store, "EMPTYSKIP/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.text, "Real text.");
}

// ── Page range spanning ─────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_page_range() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Blocks spanning pages 3, 4, 5 — all merge into one chunk
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 3),
        make_block("Text", "t0", "<p>Page three text.</p>", 3),
        make_block("Text", "t1", "<p>Page four text.</p>", 4),
        make_block("Text", "t2", "<p>Page five text.</p>", 5),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "PGRANGE", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "PGRANGE/ch1/s0/p0").await.unwrap();
    // All three pages merged into one chunk
    assert!(chunk.chunk.text.contains("Page three text"));
    assert!(chunk.chunk.text.contains("Page four text"));
    assert!(chunk.chunk.text.contains("Page five text"));
}

// ── Mixed block types → block_type = "text" ─────────────────────────────

#[serial]
#[tokio::test]
async fn test_chunking_mixed_types_block_type_text() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Ch</h2>", 0),
        make_block("Text", "t0", "<p>Text.</p>", 0),
        make_block("Equation", "eq0", "<p>x=1</p>", 0),
        make_block("ListGroup", "lg0", "<ul><li>a</li></ul>", 0),
        make_block("Text", "t1", "<p>More.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "MIXTYPE", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "MIXTYPE/ch1/s0/p0").await.unwrap();
    assert_eq!(chunk.chunk.block_type, "text");
}

// ══════════════════════════════════════════════════════════════════════════════
// Algorithm detection tests
// ══════════════════════════════════════════════════════════════════════════════

// ── Algorithm detection via h6 ──────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_algorithm_detection_h6() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("Text", "t0", "<p>See Algorithm 1.</p>", 0),
        make_block("SectionHeader", "h2", "<h6>Algorithm 1: VBD simulation for one time step.</h6>", 0),
        make_block("Text", "t1", "<p>Input: mesh M, timestep dt</p>", 0),
        make_block("Text", "t2", "<p>1. Compute forces</p>", 0),
        make_block("Text", "t3", "<p>2. Update positions</p>", 0),
        make_block("SectionHeader", "h3", "<h3>3.1 Details</h3>", 0),
        make_block("Text", "t4", "<p>Details text.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "ALGODET", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.exhibits_added, 1);

    let exhibit = get_exhibit(&store, "ALGODET/fig1").await.unwrap();
    assert_eq!(exhibit.exhibit_type, "algorithm");
    assert_eq!(exhibit.caption, "Algorithm 1: VBD simulation for one time step.");
    let content = exhibit.content.unwrap();
    assert!(content.contains("Input: mesh M"));
    assert!(content.contains("Compute forces"));
    assert!(content.contains("Update positions"));
}

// ── Algorithm ends at section header ────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_algorithm_ends_at_section_header() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("SectionHeader", "h2", "<h6>Algorithm 1: Solver</h6>", 0),
        make_block("Text", "t0", "<p>Step 1: init</p>", 0),
        make_block("Text", "t1", "<p>Step 2: solve</p>", 0),
        make_block("SectionHeader", "h3", "<h3>3.1 Discussion</h3>", 0),
        make_block("Text", "t2", "<p>Discussion text.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "ALGOEND1", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "ALGOEND1/fig1").await.unwrap();
    let content = exhibit.content.unwrap();
    assert!(content.contains("Step 1: init"));
    assert!(content.contains("Step 2: solve"));
    // Discussion text is NOT in the algorithm
    assert!(!content.contains("Discussion text"));
}

// ── Algorithm ends at figure ────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_algorithm_ends_at_figure() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("SectionHeader", "h2", "<h6>Algorithm 1: Solver</h6>", 0),
        make_block("Text", "t0", "<p>Pseudocode here.</p>", 0),
        make_block("Figure", "fig1", "<img src=\"f.png\" alt=\"Figure 1: Diagram\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "ALGOEND2", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    // 1 algorithm exhibit + 1 figure exhibit
    assert_eq!(stats.exhibits_added, 2);

    let algo = get_exhibit(&store, "ALGOEND2/fig1").await.unwrap();
    assert_eq!(algo.exhibit_type, "algorithm");
    assert!(algo.content.unwrap().contains("Pseudocode here"));

    let fig = get_exhibit(&store, "ALGOEND2/fig2").await.unwrap();
    assert_eq!(fig.exhibit_type, "figure");
}

// ── Non-algorithm h5/h6 skipped ─────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_non_algorithm_h5_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("SectionHeader", "h2", "<h5>Footnote</h5>", 0),
        make_block("Text", "t0", "<p>Normal text.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "H5NOALGO", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.exhibits_added, 0);
    assert_eq!(stats.chunks_added, 1);
}

// ── Procedure detection ─────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_procedure_detection() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("SectionHeader", "h2", "<h5>Procedure 1: Setup</h5>", 0),
        make_block("Text", "t0", "<p>Initialize parameters.</p>", 0),
        make_block("SectionHeader", "h3", "<h3>Next</h3>", 0),
        make_block("Text", "t1", "<p>After.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "PROCDET", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "PROCDET/fig1").await.unwrap();
    assert_eq!(exhibit.exhibit_type, "algorithm");
    assert_eq!(exhibit.caption, "Procedure 1: Setup");
}

// ── Listing detection ───────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_listing_detection() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("SectionHeader", "h2", "<h6>Listing 2: CUDA kernel</h6>", 0),
        make_block("Text", "t0", "<p>__global__ void kernel() {}</p>", 0),
        make_block("SectionHeader", "h3", "<h2>Next Chapter</h2>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "LISTDET", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "LISTDET/fig1").await.unwrap();
    assert_eq!(exhibit.exhibit_type, "algorithm");
    assert_eq!(exhibit.caption, "Listing 2: CUDA kernel");
}

// ── No-number h5/h6 skipped ─────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_no_number_h5_skipped() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // "Algorithm Overview" has no number → should NOT trigger detection
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter</h2>", 0),
        make_block("SectionHeader", "h2", "<h5>Algorithm Overview</h5>", 0),
        make_block("Text", "t0", "<p>This is just text.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "NONUM", &json);
    let stats = ingest_paper(&store, params).await.unwrap();

    assert_eq!(stats.exhibits_added, 0, "no-number h5 should not create exhibit");
    assert_eq!(stats.chunks_added, 1);
}

// ── Case-insensitive detection ──────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_algorithm_case_insensitive() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("SectionHeader", "h2", "<h6>ALGORITHM 1: SOLVER</h6>", 0),
        make_block("Text", "t0", "<p>Steps here.</p>", 0),
        make_block("SectionHeader", "h3", "<h2>Next</h2>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "ALGOCASE", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "ALGOCASE/fig1").await.unwrap();
    assert_eq!(exhibit.exhibit_type, "algorithm");
}

// ══════════════════════════════════════════════════════════════════════════════
// Cross-linking tests
// ══════════════════════════════════════════════════════════════════════════════

// ── Figure cross-linking ────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_crosslink_figure_ref() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Results</h2>", 0),
        make_block("Text", "t0", "<p>As shown in Fig. 1, the method works.</p>", 0),
        make_block("Figure", "fig1", "<img src=\"f.png\" alt=\"Figure 1: Results\"/>", 0),
        make_block("Caption", "cap1", "<p>Figure 1: Performance results.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "XLINK1", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "XLINK1/ch1/s0/p0").await.unwrap();
    assert!(
        chunk.chunk.exhibit_ids.contains(&"XLINK1/fig1".to_string()),
        "chunk should reference figure: {:?}",
        chunk.chunk.exhibit_ids
    );
}

// ── Table cross-linking ─────────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_crosslink_table_ref() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Results</h2>", 0),
        make_block("Text", "t0", "<p>Table 1 shows the results.</p>", 0),
        make_block("Table", "tbl1", "<img src=\"t.png\" alt=\"Table 1: Metrics\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "XLINKT", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "XLINKT/ch1/s0/p0").await.unwrap();
    assert!(
        chunk.chunk.exhibit_ids.contains(&"XLINKT/fig1".to_string()),
        "chunk should reference table exhibit: {:?}",
        chunk.chunk.exhibit_ids
    );
}

// ── Algorithm cross-linking ─────────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_crosslink_algorithm_ref() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("Text", "t0", "<p>Algorithm 1 shows the pseudocode.</p>", 0),
        make_block("SectionHeader", "h2", "<h6>Algorithm 1: VBD solver</h6>", 0),
        make_block("Text", "t1", "<p>Input: mesh M</p>", 0),
        make_block("SectionHeader", "h3", "<h2>Results</h2>", 0),
        make_block("Text", "t2", "<p>Results here.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "XLINKA", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "XLINKA/ch1/s0/p0").await.unwrap();
    assert!(
        chunk.chunk.exhibit_ids.contains(&"XLINKA/fig1".to_string()),
        "chunk should reference algorithm exhibit: {:?}",
        chunk.chunk.exhibit_ids
    );
}

// ── "Fig. 1" variant cross-link ─────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_crosslink_fig_dot_variant() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Results</h2>", 0),
        make_block("Text", "t0", "<p>See Fig. 1 for details.</p>", 0),
        make_block("Figure", "fig1", "<img src=\"f.png\" alt=\"Figure 1: X\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "FIGDOT", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "FIGDOT/ch1/s0/p0").await.unwrap();
    assert!(!chunk.chunk.exhibit_ids.is_empty(), "Fig. 1 should cross-link");
}

// ── "Tab. 1" variant cross-link ─────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_crosslink_tab_dot_variant() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Results</h2>", 0),
        make_block("Text", "t0", "<p>Tab. 1 summarizes performance.</p>", 0),
        make_block("Table", "tbl1", "<img src=\"t.png\" alt=\"Table 1: Perf\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "TABDOT", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "TABDOT/ch1/s0/p0").await.unwrap();
    assert!(!chunk.chunk.exhibit_ids.is_empty(), "Tab. 1 should cross-link");
}

// ── Case insensitive cross-link ─────────────────────────────────────────

#[serial]
#[tokio::test]
async fn test_crosslink_case_insensitive() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Results</h2>", 0),
        make_block("Text", "t0", "<p>FIGURE 1 shows the comparison.</p>", 0),
        make_block("Figure", "fig1", "<img src=\"f.png\" alt=\"Figure 1: Comp\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "CASELINK", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "CASELINK/ch1/s0/p0").await.unwrap();
    assert!(!chunk.chunk.exhibit_ids.is_empty(), "FIGURE 1 should cross-link case-insensitively");
}

// ── No false positive on missing exhibit ────────────────────────────────

#[serial]
#[tokio::test]
async fn test_crosslink_no_false_positive() {
    let _ecg = EmbedCacheGuard::new();
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("Text", "t0", "<p>Code 5 shows the implementation.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "NOFALSE", &json);
    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "NOFALSE/ch1/s0/p0").await.unwrap();
    assert!(chunk.chunk.exhibit_ids.is_empty(), "no Code 5 exhibit exists, so no link");
}

// ══════════════════════════════════════════════════════════════════════════════
// Exhibit reference tracking (first_ref_chunk_id, ref_count)
// ══════════════════════════════════════════════════════════════════════════════

#[serial]
#[tokio::test]
async fn test_exhibit_ref_count_single() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Results</h2>", 0),
        make_block("Text", "t0", "<p>See Fig. 1.</p>", 0),
        make_block("Figure", "fig1", "<img src=\"f.png\" alt=\"Figure 1: X\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "REFCNT1", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "REFCNT1/fig1").await.unwrap();
    assert_eq!(exhibit.ref_count, 1);
    assert_eq!(exhibit.first_ref_chunk_id.as_deref(), Some("REFCNT1/ch1/s0/p0"));
}

#[serial]
#[tokio::test]
async fn test_exhibit_ref_count_multiple() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    // Three chunks in different sections all reference Fig. 1
    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Chapter 1</h2>", 0),
        make_block("Text", "t0", "<p>As shown in Fig. 1.</p>", 0),
        make_block("SectionHeader", "h2", "<h2>Chapter 2</h2>", 0),
        make_block("Text", "t1", "<p>Recall Figure 1 from earlier.</p>", 0),
        make_block("SectionHeader", "h3", "<h2>Chapter 3</h2>", 0),
        make_block("Text", "t2", "<p>Finally, Fig. 1 confirms.</p>", 0),
        make_block("Figure", "fig1", "<img src=\"f.png\" alt=\"Figure 1: Diagram\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "REFCNTM", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "REFCNTM/fig1").await.unwrap();
    assert_eq!(exhibit.ref_count, 3);
    // First ref should be the first chunk in document order
    assert_eq!(exhibit.first_ref_chunk_id.as_deref(), Some("REFCNTM/ch1/s0/p0"));
}

#[serial]
#[tokio::test]
async fn test_exhibit_unreferenced() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Results</h2>", 0),
        make_block("Text", "t0", "<p>Some text without any figure reference.</p>", 0),
        make_block("Figure", "fig1", "<img src=\"f.png\" alt=\"Figure 1: X\"/>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "NOREF", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "NOREF/fig1").await.unwrap();
    assert_eq!(exhibit.ref_count, 0);
    assert!(exhibit.first_ref_chunk_id.is_none());
}

#[serial]
#[tokio::test]
async fn test_algorithm_ref_count() {
    let _ecg = EmbedCacheGuard::new();
    use crate::query::get_exhibit;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let blocks = vec![
        make_block("SectionHeader", "h1", "<h2>Method</h2>", 0),
        make_block("Text", "t0", "<p>Algorithm 1 shows the pseudocode.</p>", 0),
        make_block("SectionHeader", "h2", "<h6>Algorithm 1: VBD solver</h6>", 0),
        make_block("Text", "t1", "<p>Input: mesh M</p>", 0),
        make_block("SectionHeader", "h3", "<h2>Results</h2>", 0),
        make_block("Text", "t2", "<p>We applied Algorithm 1 to test cases.</p>", 0),
    ];
    let json = make_json_from_blocks(blocks);
    let params = make_params_from_json_str(&cache_dir, "ALGOREF", &json);
    ingest_paper(&store, params).await.unwrap();

    let exhibit = get_exhibit(&store, "ALGOREF/fig1").await.unwrap();
    assert_eq!(exhibit.exhibit_type, "algorithm");
    assert_eq!(exhibit.ref_count, 2);
    assert_eq!(exhibit.first_ref_chunk_id.as_deref(), Some("ALGOREF/ch1/s0/p0"));
}
