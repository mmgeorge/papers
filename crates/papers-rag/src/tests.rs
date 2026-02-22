//! Integration tests for the papers-rag crate.
//!
//! These tests exercise the full ingest → query pipeline using a fake embedder
//! (zero vectors, no model download) against a temporary LanceDB directory.

use std::fs;
use tempfile::TempDir;

use crate::ingest::{IngestParams, ingest_paper, is_ingested, list_cached_item_keys};
use crate::query::{
    get_chapter, get_chunk, get_paper_outline, get_section, list_papers, list_tags,
};
use crate::store::RagStore;
use crate::types::{ListPapersParams, ListTagsParams};

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
    }
}

/// Open a test store against a temporary directory.
async fn open_test_store(dir: &TempDir) -> RagStore {
    let path = dir.path().join("rag_db");
    RagStore::open_for_test(path.to_str().unwrap())
        .await
        .expect("open_for_test")
}

// ── Ingest tests ──────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ingest_basic_counts() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "TESTKEY1");

    let stats = ingest_paper(&store, params).await.unwrap();

    // 5 text-like blocks: t0, t1, t2, t3, lg0
    assert_eq!(stats.chunks_added, 5, "expected 5 chunks");
    // 2 figure-like blocks: fig1 (Figure) + tbl1 (Table); Caption is skipped
    assert_eq!(stats.figures_added, 2, "expected 2 figures");
}

#[tokio::test]
async fn test_ingest_chapter_section_assignment() {
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

#[tokio::test]
async fn test_ingest_skips_caption_and_pageheader() {
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

#[tokio::test]
async fn test_ingest_chunk_ids_have_correct_format() {
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

#[tokio::test]
async fn test_ingest_text_content_stripped() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "STRIPTEST");

    ingest_paper(&store, params).await.unwrap();

    let chunk = get_chunk(&store, "STRIPTEST/ch1/s0/p0").await.unwrap();
    // HTML tags must be stripped from "First intro paragraph."
    assert_eq!(chunk.chunk.text, "First intro paragraph.");
    assert!(!chunk.chunk.text.contains('<'));
}

#[tokio::test]
async fn test_ingest_reingest_deduplicates() {
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
    assert_eq!(papers[0].chunk_count, 5);
}

// ── is_ingested ───────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_is_ingested_false_before_ingest() {
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    assert!(!is_ingested(&store, "NOTINGESTED").await);
}

#[tokio::test]
async fn test_is_ingested_true_after_ingest() {
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

#[tokio::test]
async fn test_list_papers_returns_ingested_paper() {
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
    assert_eq!(p.chunk_count, 5);
    assert_eq!(p.figure_count, 2);
}

#[tokio::test]
async fn test_list_papers_year_filter() {
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

#[tokio::test]
async fn test_list_papers_author_filter() {
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

#[tokio::test]
async fn test_list_tags_returns_tag_counts() {
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

#[tokio::test]
async fn test_get_paper_outline_structure() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "OUTLINE1");

    ingest_paper(&store, params).await.unwrap();

    let outline = get_paper_outline(&store, "OUTLINE1").await.unwrap();

    assert_eq!(outline.paper_id, "OUTLINE1");
    assert_eq!(outline.title, "Test Paper");
    assert_eq!(outline.year, Some(2023));
    assert_eq!(outline.total_chunks, 5);
    assert_eq!(outline.total_figures, 2);
    assert_eq!(outline.chapters.len(), 2);

    // Chapter 1: Introduction with 2 sections
    let ch1 = &outline.chapters[0];
    assert_eq!(ch1.chapter_title, "Introduction");
    assert_eq!(ch1.sections.len(), 2);

    // Chapter 2: Method with 1 section
    let ch2 = &outline.chapters[1];
    assert_eq!(ch2.chapter_title, "Method");
    assert_eq!(ch2.sections.len(), 1);
    // Method section has t3 + lg0 = 2 chunks
    assert_eq!(ch2.sections[0].chunk_count, 2);
}

// ── get_chunk ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_get_chunk_not_found() {
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let result = get_chunk(&store, "NOEXIST/ch1/s0/p0").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_get_chunk_returns_prev_and_next() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "NEIGHBORS");

    ingest_paper(&store, params).await.unwrap();

    // Second chunk of ch1/s0 — prev=p0, next=none (only 2 chunks in s0)
    let result = get_chunk(&store, "NEIGHBORS/ch1/s0/p1").await.unwrap();
    assert!(result.prev.is_some(), "should have a previous chunk");
    assert_eq!(result.prev.as_ref().unwrap().chunk_id, "NEIGHBORS/ch1/s0/p0");
    // p1 is the last in s0, so no next within that section
    assert!(result.next.is_none(), "p1 should have no next in s0");
}

#[tokio::test]
async fn test_get_chunk_position_context() {
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

#[tokio::test]
async fn test_get_section_returns_chunks_in_order() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "SEC");

    ingest_paper(&store, params).await.unwrap();

    // ch2/s0 has 2 chunks: t3 and lg0
    let section = get_section(&store, "SEC", 2, 0).await.unwrap();
    assert_eq!(section.chapter_title, "Method");
    assert_eq!(section.total_chunks, 2);

    // Chunks must come out in reading order
    assert_eq!(section.chunks[0].chunk_idx, 0);
    assert_eq!(section.chunks[1].chunk_idx, 1);
    assert_eq!(section.chunks[0].text, "Method description text.");
}

#[tokio::test]
async fn test_get_section_empty_returns_zero_chunks() {
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

#[tokio::test]
async fn test_get_chapter_groups_by_section() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "CHAP");

    ingest_paper(&store, params).await.unwrap();

    // Chapter 1: Introduction has 2 sections
    let chapter = get_chapter(&store, "CHAP", 1).await.unwrap();
    assert_eq!(chapter.chapter_title, "Introduction");
    assert_eq!(chapter.sections.len(), 2);
    assert_eq!(chapter.total_chunks, 3); // t0, t1 in s0 + t2 in s1
}

#[tokio::test]
async fn test_get_chapter_section_titles() {
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

#[tokio::test]
async fn test_figure_ids_have_correct_format() {
    use crate::query::get_figure;

    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let params = make_test_cache(&cache_dir, "FIGID");

    ingest_paper(&store, params).await.unwrap();

    // Figures should be named FIGID/fig1 and FIGID/fig2
    let fig = get_figure(&store, "FIGID/fig1").await.unwrap();
    assert_eq!(fig.figure_id, "FIGID/fig1");
    assert_eq!(fig.figure_type, "figure");
    assert_eq!(fig.caption, "Figure 1: A diagram");

    let tbl = get_figure(&store, "FIGID/fig2").await.unwrap();
    assert_eq!(tbl.figure_id, "FIGID/fig2");
    assert_eq!(tbl.figure_type, "table");
    assert_eq!(tbl.caption, "Table 1: Results");
}

// ── paper metadata roundtrip ──────────────────────────────────────────────────

#[tokio::test]
async fn test_paper_metadata_preserved_in_chunks() {
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
    }
}

// 1. h1 is treated as paper title and skipped — no chapter is created.
#[tokio::test]
async fn test_h1_skipped_as_paper_title() {
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
#[tokio::test]
async fn test_h2_creates_chapter() {
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
#[tokio::test]
async fn test_h3_creates_section() {
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
#[tokio::test]
async fn test_h4_creates_section() {
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
#[tokio::test]
async fn test_h5_is_skipped() {
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
#[tokio::test]
async fn test_h6_is_skipped() {
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
#[tokio::test]
async fn test_multiple_h2_chapters_sequential() {
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
#[tokio::test]
async fn test_multiple_h3_sections_sequential() {
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
#[tokio::test]
async fn test_section_counter_resets_on_new_chapter() {
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
#[tokio::test]
async fn test_chunk_idx_resets_on_new_section() {
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
#[tokio::test]
async fn test_chunk_idx_resets_on_new_chapter() {
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
#[tokio::test]
async fn test_text_before_h2_lands_in_chapter_0() {
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
#[tokio::test]
async fn test_chapter_title_from_h2_html() {
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
#[tokio::test]
async fn test_section_title_from_h3_html() {
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
#[tokio::test]
async fn test_section_title_cleared_on_new_chapter() {
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

// 16. h6 between h3 sections does not advance the section counter.
#[tokio::test]
async fn test_h6_between_sections_does_not_advance_section() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;
    let json = make_json(&[
        ("SectionHeader", "sh1", "<h2>5 RESULTS</h2>"),
        ("SectionHeader", "sh2", "<h3>5.2 Unit Tests</h3>"),
        ("SectionHeader", "sh3", "<h6>Algorithm 1: VBD simulation for one time step.</h6>"),
        ("Text", "t0", "<p>Text after algorithm label.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "H6NOSEC", &json);
    ingest_paper(&store, params).await.unwrap();

    // h6 must not have bumped section_idx to 2
    let chunk = get_chunk(&store, "H6NOSEC/ch1/s1/p0").await.unwrap();
    assert_eq!(chunk.chunk.section_idx, 1, "h6 must not increment section_idx");
}

// 17. h1 appearing after an h2 does not start a new chapter.
#[tokio::test]
async fn test_h1_after_h2_does_not_start_new_chapter() {
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

    // Both texts should be in chapter 1
    let c0 = get_chunk(&store, "H1AFTER/ch1/s0/p0").await.unwrap();
    let c1 = get_chunk(&store, "H1AFTER/ch1/s0/p1").await.unwrap();
    assert_eq!(c0.chunk.chapter_idx, 1);
    assert_eq!(c1.chunk.chapter_idx, 1, "h1 must not start a new chapter");
}

// 18. h2 after h3 correctly starts a new chapter (not a section).
#[tokio::test]
async fn test_h2_after_h3_starts_new_chapter() {
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

// 19. Equation blocks are ingested as chunks.
#[tokio::test]
async fn test_equation_block_ingested_as_chunk() {
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

    assert_eq!(stats.chunks_added, 2, "Equation must be ingested as a chunk");
    let chunk = get_chunk(&store, "EQCHUNK/ch1/s0/p1").await.unwrap();
    assert_eq!(chunk.chunk.text, "x = y + z");
}

// 20. Realistic YFACFA8C-style structure: h1 title, h6 ACM ref, h2/h3 sections.
#[tokio::test]
async fn test_realistic_yfacfa8c_structure() {
    let cache_dir = TempDir::new().unwrap();
    let db_dir = TempDir::new().unwrap();
    let store = open_test_store(&db_dir).await;

    let json = make_json(&[
        // Paper title (h1) — skipped
        ("SectionHeader", "sh_title", "<h1>Vertex Block Descent</h1>"),
        // ACM reference (h6) — skipped
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
        // Algorithm label (h6) — skipped
        ("SectionHeader", "sh_alg", "<h6>Algorithm 1: VBD simulation.</h6>"),
        ("Text", "t_after_alg", "<p>Text after algorithm label.</p>"),
    ]);
    let params = make_cache_from_json(&cache_dir, "VBD", &json);
    ingest_paper(&store, params).await.unwrap();

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

    // Algorithm h6 must not have created a new section in ch4
    let gpu_chunk = get_chunk(&store, "VBD/ch4/s0/p0").await.unwrap();
    assert_eq!(gpu_chunk.chunk.chapter_idx, 4);
    assert_eq!(gpu_chunk.chunk.section_idx, 0);

    // Text after algorithm label is still in ch4/s0
    let after_alg = get_chunk(&store, "VBD/ch4/s0/p1").await.unwrap();
    assert_eq!(after_alg.chunk.chapter_idx, 4);
    assert_eq!(after_alg.chunk.section_idx, 0);
}
