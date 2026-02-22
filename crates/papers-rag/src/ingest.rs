use arrow_array::{
    FixedSizeListArray, Float32Array, ListArray, RecordBatch, RecordBatchIterator,
    StringArray, UInt16Array,
    builder::{ListBuilder, StringBuilder},
};
use arrow_schema::ArrowError;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::error::RagError;
use crate::schema::{EMBED_DIM, chunks_schema, figures_schema};
use crate::store::RagStore;
use crate::types::IngestStats;

pub struct IngestParams {
    pub item_key: String,
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    pub tags: Vec<String>,
    pub cache_dir: PathBuf,
    /// When `true`, bypass the embedding cache and re-embed from scratch.
    pub force: bool,
}

struct ChunkRecord {
    chunk_id: String,
    chapter_title: String,
    chapter_idx: u16,
    section_title: String,
    section_idx: u16,
    chunk_idx: u16,
    text: String,
    page: Option<u16>,
    figure_ids: Vec<String>,
}

struct FigureRecord {
    figure_id: String,
    figure_type: String,
    caption: String,
    image_path: Option<String>,
    page: Option<u16>,
    chapter_idx: u16,
    section_idx: u16,
}

/// Strip HTML tags and normalize whitespace.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    // Normalize whitespace
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Extract `src` attribute value from an `<img>` tag.
fn extract_img_src(html: &str) -> Option<String> {
    extract_attr(html, "src")
}

/// Extract `alt` attribute value from an `<img>` tag.
fn extract_img_alt(html: &str) -> Option<String> {
    extract_attr(html, "alt")
}

fn extract_attr(html: &str, attr: &str) -> Option<String> {
    let search = format!("{}=\"", attr);
    let start = html.find(search.as_str())? + search.len();
    let rest = &html[start..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Return the DataLab cache root directory.
fn cache_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PAPERS_DATALAB_CACHE_DIR") {
        return Some(PathBuf::from(p));
    }
    dirs::cache_dir().map(|d| d.join("papers").join("datalab"))
}

/// Build IngestParams from a cached item_key using meta.json.
pub fn ingest_params_from_cache(item_key: &str) -> Result<IngestParams, RagError> {
    let root = cache_root().ok_or_else(|| {
        RagError::NotFound("cannot determine cache directory".into())
    })?;
    let cache_dir = root.join(item_key);
    if !cache_dir.is_dir() {
        return Err(RagError::NotFound(format!(
            "cache directory not found: {}",
            cache_dir.display()
        )));
    }

    // Try to read meta.json via papers-core
    let meta = papers_core::text::read_extraction_meta(item_key);
    let (title, authors, year, venue, doi) = match meta {
        Some(m) => {
            let y = m.date.as_deref().and_then(parse_year);
            (
                m.title.unwrap_or_else(|| item_key.to_string()),
                m.authors.unwrap_or_default(),
                y,
                m.publication_title,
                m.doi,
            )
        }
        None => (item_key.to_string(), vec![], None, None, None),
    };

    let paper_id = doi
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| item_key.to_string());

    Ok(IngestParams {
        item_key: item_key.to_string(),
        paper_id,
        title,
        authors,
        year,
        venue,
        tags: vec![],
        cache_dir,
        force: false,
    })
}

fn parse_year(date: &str) -> Option<u16> {
    date.split('-').next()?.parse().ok()
}

/// Return the base directory for the embedding cache.
///
/// Checks `PAPERS_EMBED_CACHE_DIR` first, then falls back to
/// `{cache_dir}/papers` (platform cache directory).
pub fn embed_cache_base() -> PathBuf {
    if let Ok(p) = std::env::var("PAPERS_EMBED_CACHE_DIR") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("papers")
}

/// Return the default embedding model name from user config, falling back to the
/// built-in default.
fn default_embed_model() -> String {
    papers_core::config::PapersConfig::load()
        .map(|c| c.embedding_model)
        .unwrap_or_else(|_| "nomic-embed-text-v2-moe".to_string())
}

/// Convenience helper: convert an `EmbedCacheError` to `RagError::Cache`.
fn cache_err(e: crate::embed_cache::EmbedCacheError) -> RagError {
    RagError::Cache(e.to_string())
}

/// Return the current Unix timestamp as a decimal string.
fn unix_timestamp_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

/// Parse the DataLab Marker JSON for a paper into raw `ChunkRecord` and
/// `FigureRecord` lists.  This is shared between `ingest_paper` and
/// `cache_paper_embeddings`.
fn parse_paper_blocks(
    params: &IngestParams,
) -> Result<(Vec<ChunkRecord>, Vec<FigureRecord>), RagError> {
    let json_path = params.cache_dir.join(format!("{}.json", params.item_key));
    let json_bytes = std::fs::read(&json_path)?;
    let root: Value = serde_json::from_slice(&json_bytes)?;

    // ── Flatten blocks from all pages ──────────────────────────────────────
    let pages = root
        .get("children")
        .and_then(|c| c.as_array())
        .map(|a| a.as_slice())
        .unwrap_or(&[]);

    let mut flat_blocks: Vec<&Value> = Vec::new();
    for page in pages {
        if let Some(children) = page.get("children").and_then(|c| c.as_array()) {
            for block in children {
                flat_blocks.push(block);
            }
        }
    }

    eprintln!(
        "  [{}] parsed {} raw blocks across {} pages",
        params.item_key,
        flat_blocks.len(),
        pages.len()
    );

    // ── Build heading map: block_id → plain text ────────────────────────────
    let mut heading_map: HashMap<String, String> = HashMap::new();
    for block in &flat_blocks {
        if block
            .get("block_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            == "SectionHeader"
        {
            if let Some(id) = block.get("id").and_then(|v| v.as_str()) {
                let html = block.get("html").and_then(|v| v.as_str()).unwrap_or("");
                heading_map.insert(id.to_string(), strip_html(html));
            }
        }
    }

    // ── Walk blocks, assign chapter/section indices ─────────────────────────
    let mut chapter_idx: u16 = 0;
    let mut section_idx: u16 = 0;
    let mut current_chapter_title = String::new();
    let mut current_section_title = String::new();
    let mut chunk_idx: u16 = 0;
    let mut figure_seq: u16 = 0;

    let mut chunk_records: Vec<ChunkRecord> = Vec::new();
    let mut figure_records: Vec<FigureRecord> = Vec::new();

    // Track per-section chunk counter (reset when section changes)
    let mut last_section_key: (u16, u16) = (0, 0);

    for block in &flat_blocks {
        let block_type = block
            .get("block_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let block_id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
        let page_num: Option<u16> = block
            .get("page")
            .and_then(|v| v.as_u64())
            .map(|p| p as u16);

        match block_type {
            "Page" | "PageHeader" | "PageFooter" | "TableOfContents" | "Caption"
            | "Picture" => continue,

            "SectionHeader" => {
                // Determine level from the HTML heading tag (h1–h6).
                // section_hierarchy depth is unreliable; the tag is authoritative.
                let html = block.get("html").and_then(|v| v.as_str()).unwrap_or("");
                let h_level: u8 = if html.starts_with("<h1") || html.starts_with("<H1") {
                    1
                } else if html.starts_with("<h2") || html.starts_with("<H2") {
                    2
                } else if html.starts_with("<h3") || html.starts_with("<H3") {
                    3
                } else if html.starts_with("<h4") || html.starts_with("<H4") {
                    4
                } else if html.starts_with("<h5") || html.starts_with("<H5") {
                    5
                } else {
                    6 // h6 or unknown
                };

                match h_level {
                    1 | 5 | 6 => {
                        // Paper title, footnote-style headings, algorithm labels — skip
                    }
                    2 => {
                        // Major section (chapter-level)
                        chapter_idx += 1;
                        section_idx = 0;
                        chunk_idx = 0;
                        current_chapter_title =
                            heading_map.get(block_id).cloned().unwrap_or_default();
                        current_section_title = String::new();
                        last_section_key = (chapter_idx, section_idx);
                    }
                    3 | 4 => {
                        // Subsection
                        section_idx += 1;
                        chunk_idx = 0;
                        current_section_title =
                            heading_map.get(block_id).cloned().unwrap_or_default();
                        last_section_key = (chapter_idx, section_idx);
                    }
                    _ => {}
                }
                // Don't emit a chunk for headers
            }

            "Text" | "ListGroup" | "Equation" => {
                let html = block.get("html").and_then(|v| v.as_str()).unwrap_or("");
                let text = strip_html(html);
                if text.trim().is_empty() {
                    continue;
                }

                // Reset chunk_idx if section changed
                let section_key = (chapter_idx, section_idx);
                if section_key != last_section_key {
                    chunk_idx = 0;
                    last_section_key = section_key;
                }

                let chunk_id = format!(
                    "{}/ch{}/s{}/p{}",
                    params.paper_id, chapter_idx, section_idx, chunk_idx
                );
                chunk_records.push(ChunkRecord {
                    chunk_id,
                    chapter_title: current_chapter_title.clone(),
                    chapter_idx,
                    section_title: current_section_title.clone(),
                    section_idx,
                    chunk_idx,
                    text,
                    page: page_num,
                    figure_ids: vec![],
                });
                chunk_idx += 1;
            }

            "Figure" | "Table" => {
                let html = block.get("html").and_then(|v| v.as_str()).unwrap_or("");
                let caption = extract_img_alt(html).unwrap_or_default();
                let src = extract_img_src(html);
                let image_path = src.map(|s| {
                    params
                        .cache_dir
                        .join("images")
                        .join(&s)
                        .to_string_lossy()
                        .into_owned()
                });
                let figure_type = if block_type == "Table" { "table" } else { "figure" };
                figure_seq += 1;
                let figure_id = format!("{}/fig{}", params.paper_id, figure_seq);
                figure_records.push(FigureRecord {
                    figure_id,
                    figure_type: figure_type.to_string(),
                    caption,
                    image_path,
                    page: page_num,
                    chapter_idx,
                    section_idx,
                });
            }

            _ => {} // skip unknown block types
        }
    }

    eprintln!(
        "  [{}] extracted {} chunks, {} figures ({} section headers)",
        params.item_key,
        chunk_records.len(),
        figure_records.len(),
        heading_map.len()
    );

    Ok((chunk_records, figure_records))
}

/// Compute embeddings for a paper's chunks and write them to the embedding cache.
///
/// - Returns the number of chunks cached.
/// - If the cache already exists and `force` is `false`, returns the cached chunk
///   count without re-embedding.
/// - Saves figure embeddings are **not** cached — only text chunks.
pub async fn cache_paper_embeddings(
    store: &RagStore,
    params: &IngestParams,
    model: &str,
    force: bool,
) -> Result<usize, RagError> {
    let cache = crate::embed_cache::EmbedCache::new(embed_cache_base());

    // Cache hit: skip embedding
    if !force {
        if let Some(manifest) = cache.load_manifest(model, &params.item_key).map_err(cache_err)? {
            eprintln!(
                "  [{}] embed cache hit ({} chunks, model={})",
                params.item_key,
                manifest.chunks.len(),
                model
            );
            return Ok(manifest.chunks.len());
        }
    }

    let (chunk_records, _figure_records) = parse_paper_blocks(params)?;
    let n = chunk_records.len();

    let embeddings = if chunk_records.is_empty() {
        vec![]
    } else {
        eprintln!("  [{}] embedding {} chunks (model={})...", params.item_key, n, model);
        let texts: Vec<String> = chunk_records.iter().map(|c| c.text.clone()).collect();
        let result = store.embed_documents(texts).await?;
        eprintln!("  [{}] chunk embeddings done", params.item_key);
        result
    };

    let dim = embeddings.first().map(|v| v.len()).unwrap_or(crate::schema::EMBED_DIM as usize);
    let cached_chunks: Vec<crate::embed_cache::ChunkRecord> = chunk_records
        .iter()
        .map(|c| crate::embed_cache::ChunkRecord {
            chunk_id: c.chunk_id.clone(),
            text: c.text.clone(),
            section: c.section_title.clone(),
            heading: c.chapter_title.clone(),
            chapter_idx: c.chapter_idx as u32,
            section_idx: c.section_idx as u32,
            chunk_idx: c.chunk_idx as u32,
            page_start: c.page.map(|p| p as u32),
            page_end: c.page.map(|p| p as u32),
        })
        .collect();

    let manifest = crate::embed_cache::EmbedManifest {
        model: model.to_string(),
        dim,
        created_at: unix_timestamp_str(),
        chunks: cached_chunks,
    };

    cache
        .save(model, &params.item_key, &manifest, &embeddings, force)
        .map_err(cache_err)?;

    eprintln!(
        "  [{}] embed cache written ({} chunks, model={})",
        params.item_key, n, model
    );
    Ok(n)
}

/// Ingest a paper from the DataLab Marker JSON cache into LanceDB.
pub async fn ingest_paper(store: &RagStore, params: IngestParams) -> Result<IngestStats, RagError> {
    let (chunk_records, figure_records) = parse_paper_blocks(&params)?;

    let chunks_added = chunk_records.len();
    let figures_added = figure_records.len();

    // ── Delete existing records for this paper ──────────────────────────────
    let paper_id_esc = params.paper_id.replace('\'', "''");
    let delete_filter = format!("paper_id = '{paper_id_esc}'");
    if let Ok(chunks_table) = store.chunks_table().await {
        let _ = chunks_table.delete(&delete_filter).await;
    }
    if let Ok(figures_table) = store.figures_table().await {
        let _ = figures_table.delete(&delete_filter).await;
    }

    // ── Embed chunk texts (with cache) ─────────────────────────────────────
    let model = default_embed_model();
    let embed_cache = crate::embed_cache::EmbedCache::new(embed_cache_base());

    let embeddings = if chunk_records.is_empty() {
        vec![]
    } else {
        // Cache hit: load embeddings from disk, skip GPU inference
        let cached = if !params.force {
            embed_cache
                .load_manifest(&model, &params.item_key)
                .map_err(cache_err)?
                .map(|manifest| {
                    embed_cache
                        .load_embeddings(&model, &params.item_key, &manifest)
                        .map_err(cache_err)
                })
                .transpose()?
        } else {
            None
        };

        match cached {
            Some(embs) => {
                eprintln!(
                    "  [{}] embed cache hit ({} chunks)",
                    params.item_key,
                    embs.len()
                );
                embs
            }
            None => {
                eprintln!(
                    "  [{}] embedding {} chunks...",
                    params.item_key,
                    chunk_records.len()
                );
                let texts: Vec<String> =
                    chunk_records.iter().map(|c| c.text.clone()).collect();
                let result = store.embed_documents(texts).await?;
                eprintln!("  [{}] chunk embeddings done", params.item_key);

                // Save to cache for future re-ingests
                let dim = result
                    .first()
                    .map(|v| v.len())
                    .unwrap_or(crate::schema::EMBED_DIM as usize);
                let cached_chunks: Vec<crate::embed_cache::ChunkRecord> = chunk_records
                    .iter()
                    .map(|c| crate::embed_cache::ChunkRecord {
                        chunk_id: c.chunk_id.clone(),
                        text: c.text.clone(),
                        section: c.section_title.clone(),
                        heading: c.chapter_title.clone(),
                        chapter_idx: c.chapter_idx as u32,
                        section_idx: c.section_idx as u32,
                        chunk_idx: c.chunk_idx as u32,
                        page_start: c.page.map(|p| p as u32),
                        page_end: c.page.map(|p| p as u32),
                    })
                    .collect();
                let manifest = crate::embed_cache::EmbedManifest {
                    model: model.clone(),
                    dim,
                    created_at: unix_timestamp_str(),
                    chunks: cached_chunks,
                };
                if let Err(e) = embed_cache.save(
                    &model,
                    &params.item_key,
                    &manifest,
                    &result,
                    params.force,
                ) {
                    eprintln!("  [{}] warning: failed to write embed cache: {e}", params.item_key);
                }
                result
            }
        }
    };

    // ── Embed figure captions ───────────────────────────────────────────────
    let fig_texts: Vec<String> = figure_records.iter().map(|f| f.caption.clone()).collect();
    let fig_embeddings = if fig_texts.is_empty() {
        vec![]
    } else {
        eprintln!(
            "  [{}] embedding {} figure captions...",
            params.item_key,
            fig_texts.len()
        );
        let result = store.embed_documents(fig_texts).await?;
        eprintln!("  [{}] figure embeddings done", params.item_key);
        result
    };

    // ── Insert chunks ───────────────────────────────────────────────────────
    if chunks_added > 0 {
        eprintln!("  [{}] inserting {} chunks...", params.item_key, chunks_added);
        let batch = build_chunks_batch(&params, &chunk_records, &embeddings)?;
        let schema = chunks_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let table = store.chunks_table().await?;
        table
            .add(Box::new(reader))
            .execute()
            .await?;
    }

    // ── Insert figures ──────────────────────────────────────────────────────
    if figures_added > 0 {
        eprintln!(
            "  [{}] inserting {} figures...",
            params.item_key,
            figures_added
        );
        let batch = build_figures_batch(&params, &figure_records, &fig_embeddings)?;
        let schema = figures_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let table = store.figures_table().await?;
        table
            .add(Box::new(reader))
            .execute()
            .await?;
    }

    eprintln!("  [{}] done.", params.item_key);
    Ok(IngestStats {
        chunks_added,
        figures_added,
    })
}

fn build_string_list_array(lists: &[Vec<String>]) -> ListArray {
    let mut builder = ListBuilder::new(StringBuilder::new());
    for list in lists {
        for s in list {
            builder.values().append_value(s);
        }
        builder.append(true);
    }
    builder.finish()
}

fn build_vector_array(embeddings: &[Vec<f32>]) -> FixedSizeListArray {
    let flat: Vec<f32> = embeddings.iter().flat_map(|v| v.iter().copied()).collect();
    let flat_array = Arc::new(Float32Array::from(flat));
    let field = Arc::new(arrow_schema::Field::new(
        "item",
        arrow_schema::DataType::Float32,
        true,
    ));
    FixedSizeListArray::new(field, EMBED_DIM, flat_array, None)
}

fn build_chunks_batch(
    params: &IngestParams,
    records: &[ChunkRecord],
    embeddings: &[Vec<f32>],
) -> Result<RecordBatch, RagError> {
    let n = records.len();
    let schema = chunks_schema();

    let chunk_ids: Vec<&str> = records.iter().map(|r| r.chunk_id.as_str()).collect();
    let paper_ids: Vec<&str> = vec![params.paper_id.as_str(); n];
    let vectors = build_vector_array(embeddings);
    let chapter_titles: Vec<&str> = records.iter().map(|r| r.chapter_title.as_str()).collect();
    let chapter_idxs: Vec<u16> = records.iter().map(|r| r.chapter_idx).collect();
    let section_titles: Vec<&str> = records.iter().map(|r| r.section_title.as_str()).collect();
    let section_idxs: Vec<u16> = records.iter().map(|r| r.section_idx).collect();
    let chunk_idxs: Vec<u16> = records.iter().map(|r| r.chunk_idx).collect();
    let depths: Vec<&str> = vec!["paragraph"; n];
    let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
    let page_starts: Vec<Option<u16>> = records.iter().map(|r| r.page).collect();
    let page_ends: Vec<Option<u16>> = records.iter().map(|r| r.page).collect();
    let titles: Vec<&str> = vec![params.title.as_str(); n];
    let authors_list: Vec<Vec<String>> = vec![params.authors.clone(); n];
    let years: Vec<Option<u16>> = vec![params.year; n];
    let venues: Vec<Option<&str>> = vec![params.venue.as_deref(); n];
    let tags_list: Vec<Vec<String>> = vec![params.tags.clone(); n];
    let figure_ids_list: Vec<Vec<String>> = records.iter().map(|r| r.figure_ids.clone()).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(chunk_ids)),
            Arc::new(StringArray::from(paper_ids)),
            Arc::new(vectors),
            Arc::new(StringArray::from(chapter_titles)),
            Arc::new(UInt16Array::from(chapter_idxs)),
            Arc::new(StringArray::from(section_titles)),
            Arc::new(UInt16Array::from(section_idxs)),
            Arc::new(UInt16Array::from(chunk_idxs)),
            Arc::new(StringArray::from(depths)),
            Arc::new(StringArray::from(texts)),
            Arc::new(UInt16Array::from(page_starts)),
            Arc::new(UInt16Array::from(page_ends)),
            Arc::new(StringArray::from(titles)),
            Arc::new(build_string_list_array(&authors_list)),
            Arc::new(UInt16Array::from(years)),
            Arc::new(StringArray::from(venues)),
            Arc::new(build_string_list_array(&tags_list)),
            Arc::new(build_string_list_array(&figure_ids_list)),
        ],
    )
    .map_err(|e: ArrowError| RagError::Arrow(e.to_string()))?;

    Ok(batch)
}

fn build_figures_batch(
    params: &IngestParams,
    records: &[FigureRecord],
    embeddings: &[Vec<f32>],
) -> Result<RecordBatch, RagError> {
    let n = records.len();
    let schema = figures_schema();

    let figure_ids: Vec<&str> = records.iter().map(|r| r.figure_id.as_str()).collect();
    let paper_ids: Vec<&str> = vec![params.paper_id.as_str(); n];
    let vectors = build_vector_array(embeddings);
    let figure_types: Vec<&str> = records.iter().map(|r| r.figure_type.as_str()).collect();
    let captions: Vec<&str> = records.iter().map(|r| r.caption.as_str()).collect();
    let descriptions: Vec<&str> = records.iter().map(|r| r.caption.as_str()).collect();
    let image_paths: Vec<Option<&str>> = records
        .iter()
        .map(|r| r.image_path.as_deref())
        .collect();
    let pages: Vec<Option<u16>> = records.iter().map(|r| r.page).collect();
    let chapter_idxs: Vec<u16> = records.iter().map(|r| r.chapter_idx).collect();
    let section_idxs: Vec<u16> = records.iter().map(|r| r.section_idx).collect();
    let titles: Vec<&str> = vec![params.title.as_str(); n];
    let authors_list: Vec<Vec<String>> = vec![params.authors.clone(); n];
    let years: Vec<Option<u16>> = vec![params.year; n];
    let venues: Vec<Option<&str>> = vec![params.venue.as_deref(); n];
    let tags_list: Vec<Vec<String>> = vec![params.tags.clone(); n];

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(figure_ids)),
            Arc::new(StringArray::from(paper_ids)),
            Arc::new(vectors),
            Arc::new(StringArray::from(figure_types)),
            Arc::new(StringArray::from(captions)),
            Arc::new(StringArray::from(descriptions)),
            Arc::new(StringArray::from(image_paths)),
            Arc::new(UInt16Array::from(pages)),
            Arc::new(UInt16Array::from(chapter_idxs)),
            Arc::new(UInt16Array::from(section_idxs)),
            Arc::new(StringArray::from(titles)),
            Arc::new(build_string_list_array(&authors_list)),
            Arc::new(UInt16Array::from(years)),
            Arc::new(StringArray::from(venues)),
            Arc::new(build_string_list_array(&tags_list)),
        ],
    )
    .map_err(|e: ArrowError| RagError::Arrow(e.to_string()))?;

    Ok(batch)
}

/// Check if a paper is already indexed in the RAG database.
pub async fn is_ingested(store: &RagStore, paper_id: &str) -> bool {
    use futures::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};
    let table = match store.chunks_table().await {
        Ok(t) => t,
        Err(_) => return false,
    };
    let filter = format!("paper_id = '{}'", paper_id.replace('\'', "''"));
    match table.query().only_if(&filter).limit(1).execute().await {
        Ok(mut stream) => stream.try_next().await
            .ok()
            .flatten()
            .map(|b| b.num_rows() > 0)
            .unwrap_or(false),
        Err(_) => false,
    }
}

/// List all item keys in the DataLab cache that have a JSON extraction file.
pub fn list_cached_item_keys() -> Vec<String> {
    let base = match cache_root() {
        Some(b) => b,
        None => return vec![],
    };
    if !base.is_dir() {
        return vec![];
    }
    let mut keys = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let key = match entry.file_name().to_str() {
                Some(k) => k.to_string(),
                None => continue,
            };
            // Only include if JSON cache exists
            if entry.path().join(format!("{key}.json")).exists() {
                keys.push(key);
            }
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // ── strip_html ───────────────────────────────────────────────────────────

    #[test]
    fn strip_html_simple_tag() {
        assert_eq!(strip_html("<h2>Hello World</h2>"), "Hello World");
    }

    #[test]
    fn strip_html_multiple_tags() {
        // Tags are removed; no extra spaces inserted between adjacent elements
        assert_eq!(strip_html("<p>foo</p><p>bar</p>"), "foobar");
    }

    #[test]
    fn strip_html_no_tags_passes_through() {
        assert_eq!(strip_html("plain text"), "plain text");
    }

    #[test]
    fn strip_html_empty_string() {
        assert_eq!(strip_html(""), "");
    }

    #[test]
    fn strip_html_whitespace_normalized() {
        assert_eq!(strip_html("<p>  foo   bar  </p>"), "foo bar");
    }

    #[test]
    fn strip_html_nested_tags() {
        assert_eq!(strip_html("<ul><li>a</li><li>b</li></ul>"), "ab");
    }

    #[test]
    fn strip_html_self_closing() {
        assert_eq!(strip_html("before<br/>after"), "beforeafter");
    }

    // ── extract_attr ─────────────────────────────────────────────────────────

    #[test]
    fn extract_img_src_basic() {
        let html = r#"<img src="fig1.jpg" alt="Figure 1"/>"#;
        assert_eq!(extract_img_src(html), Some("fig1.jpg".to_string()));
    }

    #[test]
    fn extract_img_alt_basic() {
        let html = r#"<img src="fig1.jpg" alt="Figure 1: Caption here"/>"#;
        assert_eq!(
            extract_img_alt(html),
            Some("Figure 1: Caption here".to_string())
        );
    }

    #[test]
    fn extract_img_src_missing_returns_none() {
        assert_eq!(extract_img_src("<img alt=\"no src\"/>"), None);
    }

    #[test]
    fn extract_img_alt_missing_returns_none() {
        assert_eq!(extract_img_alt("<img src=\"x.jpg\"/>"), None);
    }

    #[test]
    fn extract_img_alt_with_special_chars() {
        let html = r#"<img alt="Table 2: Results (n=10, p<0.05)" src="tbl2.png"/>"#;
        let alt = extract_img_alt(html).unwrap();
        assert_eq!(alt, "Table 2: Results (n=10, p<0.05)");
    }

    // ── parse_year ────────────────────────────────────────────────────────────

    #[test]
    fn parse_year_iso_date() {
        assert_eq!(parse_year("2023-04-15"), Some(2023));
    }

    #[test]
    fn parse_year_year_only() {
        assert_eq!(parse_year("2021"), Some(2021));
    }

    #[test]
    fn parse_year_empty_returns_none() {
        assert_eq!(parse_year(""), None);
    }

    #[test]
    fn parse_year_non_numeric_returns_none() {
        assert_eq!(parse_year("not-a-date"), None);
    }

    // ── list_cached_item_keys ─────────────────────────────────────────────────

    #[test]
    fn list_cached_keys_finds_dirs_with_json() {
        let dir = TempDir::new().unwrap();

        // key1: dir with matching JSON → should be included
        let key1_dir = dir.path().join("ABCD1234");
        fs::create_dir(&key1_dir).unwrap();
        fs::write(key1_dir.join("ABCD1234.json"), "{}").unwrap();

        // key2: dir without JSON → should be excluded
        let key2_dir = dir.path().join("EFGH5678");
        fs::create_dir(&key2_dir).unwrap();

        // key3: file not a dir → should be excluded
        fs::write(dir.path().join("not_a_dir"), "").unwrap();

        // key4: dir with JSON of a different name → should be excluded
        let key4_dir = dir.path().join("IJKL9012");
        fs::create_dir(&key4_dir).unwrap();
        fs::write(key4_dir.join("other.json"), "{}").unwrap();

        let old = std::env::var("PAPERS_DATALAB_CACHE_DIR").ok();
        unsafe {
            std::env::set_var("PAPERS_DATALAB_CACHE_DIR", dir.path().to_str().unwrap());
        }

        let mut keys = list_cached_item_keys();
        keys.sort();

        unsafe {
            if let Some(old_val) = old {
                std::env::set_var("PAPERS_DATALAB_CACHE_DIR", old_val);
            } else {
                std::env::remove_var("PAPERS_DATALAB_CACHE_DIR");
            }
        }

        assert_eq!(keys, vec!["ABCD1234"]);
    }

    #[test]
    fn list_cached_keys_nonexistent_cache_dir_returns_empty() {
        let old = std::env::var("PAPERS_DATALAB_CACHE_DIR").ok();
        unsafe {
            std::env::set_var(
                "PAPERS_DATALAB_CACHE_DIR",
                "/nonexistent/path/that/does/not/exist",
            );
        }
        let keys = list_cached_item_keys();
        unsafe {
            if let Some(old_val) = old {
                std::env::set_var("PAPERS_DATALAB_CACHE_DIR", old_val);
            } else {
                std::env::remove_var("PAPERS_DATALAB_CACHE_DIR");
            }
        }
        assert!(keys.is_empty());
    }

    // ── embed cache integration ───────────────────────────────────────────────

    /// Build a minimal DataLab JSON with a few text blocks and write it to `dir/item_key/`.
    fn write_fake_datalab_json(dir: &std::path::Path, item_key: &str) {
        let item_dir = dir.join(item_key);
        fs::create_dir_all(&item_dir).unwrap();
        let json = serde_json::json!({
            "children": [{
                "block_type": "Page",
                "children": [
                    {
                        "block_type": "SectionHeader",
                        "id": "hdr1",
                        "html": "<h2>Introduction</h2>",
                        "page": 1
                    },
                    {
                        "block_type": "Text",
                        "id": "blk1",
                        "html": "<p>First chunk of text.</p>",
                        "page": 1
                    },
                    {
                        "block_type": "Text",
                        "id": "blk2",
                        "html": "<p>Second chunk of text.</p>",
                        "page": 2
                    }
                ]
            }]
        });
        fs::write(
            item_dir.join(format!("{item_key}.json")),
            serde_json::to_vec_pretty(&json).unwrap(),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn test_ingest_writes_embed_cache() {
        let datalab_dir = TempDir::new().unwrap();
        let rag_dir = TempDir::new().unwrap();
        let embed_dir = TempDir::new().unwrap();
        let key = "TESTKEY1";

        write_fake_datalab_json(datalab_dir.path(), key);

        unsafe {
            std::env::set_var("PAPERS_DATALAB_CACHE_DIR", datalab_dir.path());
            std::env::set_var("PAPERS_EMBED_CACHE_DIR", embed_dir.path());
        }

        let rag = crate::store::RagStore::open_for_test(
            rag_dir.path().to_str().unwrap(),
        )
        .await
        .unwrap();

        let params = IngestParams {
            item_key: key.to_string(),
            paper_id: key.to_string(),
            title: "Test Paper".to_string(),
            authors: vec![],
            year: None,
            venue: None,
            tags: vec![],
            cache_dir: datalab_dir.path().join(key),
            force: false,
        };

        ingest_paper(&rag, params).await.unwrap();

        unsafe {
            std::env::remove_var("PAPERS_DATALAB_CACHE_DIR");
            std::env::remove_var("PAPERS_EMBED_CACHE_DIR");
        }

        let embed_cache = crate::embed_cache::EmbedCache::new(embed_dir.path().to_path_buf());
        let model = "nomic-embed-text-v2-moe";
        assert!(embed_cache.exists(model, key), "manifest.json + embeddings.bin should exist");
        let manifest = embed_cache.load_manifest(model, key).unwrap().unwrap();
        assert_eq!(manifest.chunks.len(), 2, "should have 2 text chunks");
    }

    #[tokio::test]
    async fn test_ingest_cache_hit_skips_embedder() {
        let datalab_dir = TempDir::new().unwrap();
        let rag_dir = TempDir::new().unwrap();
        let embed_dir = TempDir::new().unwrap();
        let key = "TESTKEY2";

        write_fake_datalab_json(datalab_dir.path(), key);

        unsafe {
            std::env::set_var("PAPERS_DATALAB_CACHE_DIR", datalab_dir.path());
            std::env::set_var("PAPERS_EMBED_CACHE_DIR", embed_dir.path());
        }

        let rag = crate::store::RagStore::open_for_test(
            rag_dir.path().to_str().unwrap(),
        )
        .await
        .unwrap();

        let make_params = || IngestParams {
            item_key: key.to_string(),
            paper_id: key.to_string(),
            title: "Test Paper".to_string(),
            authors: vec![],
            year: None,
            venue: None,
            tags: vec![],
            cache_dir: datalab_dir.path().join(key),
            force: false,
        };

        // First ingest writes the cache
        ingest_paper(&rag, make_params()).await.unwrap();

        let embed_cache = crate::embed_cache::EmbedCache::new(embed_dir.path().to_path_buf());
        let model = "nomic-embed-text-v2-moe";
        assert!(embed_cache.exists(model, key));
        let mtime_after_first = embed_dir
            .path()
            .join("embeddings")
            .join(model)
            .join(key)
            .join("embeddings.bin")
            .metadata()
            .unwrap()
            .modified()
            .unwrap();

        // Second ingest: should hit cache (no re-embedding)
        // We can verify the embeddings.bin file is NOT rewritten
        // (mtime doesn't change on cache hit).
        // Note: force=false so cache hit applies.
        ingest_paper(&rag, make_params()).await.unwrap();

        let mtime_after_second = embed_dir
            .path()
            .join("embeddings")
            .join(model)
            .join(key)
            .join("embeddings.bin")
            .metadata()
            .unwrap()
            .modified()
            .unwrap();

        // Cache file was not rewritten (same mtime)
        // Note: we pass overwrite=false so save() would fail on second call —
        // meaning the file is never touched on cache hit.
        assert_eq!(mtime_after_first, mtime_after_second, "embeddings.bin should not be rewritten on cache hit");

        unsafe {
            std::env::remove_var("PAPERS_DATALAB_CACHE_DIR");
            std::env::remove_var("PAPERS_EMBED_CACHE_DIR");
        }
    }

    #[tokio::test]
    async fn test_ingest_cache_miss_on_force() {
        let datalab_dir = TempDir::new().unwrap();
        let rag_dir = TempDir::new().unwrap();
        let embed_dir = TempDir::new().unwrap();
        let key = "TESTKEY3";

        write_fake_datalab_json(datalab_dir.path(), key);

        unsafe {
            std::env::set_var("PAPERS_DATALAB_CACHE_DIR", datalab_dir.path());
            std::env::set_var("PAPERS_EMBED_CACHE_DIR", embed_dir.path());
        }

        let rag = crate::store::RagStore::open_for_test(
            rag_dir.path().to_str().unwrap(),
        )
        .await
        .unwrap();

        // First ingest (force=false) — writes cache
        ingest_paper(
            &rag,
            IngestParams {
                item_key: key.to_string(),
                paper_id: key.to_string(),
                title: "Test".to_string(),
                authors: vec![],
                year: None,
                venue: None,
                tags: vec![],
                cache_dir: datalab_dir.path().join(key),
                force: false,
            },
        )
        .await
        .unwrap();

        let embed_cache = crate::embed_cache::EmbedCache::new(embed_dir.path().to_path_buf());
        let model = "nomic-embed-text-v2-moe";
        assert!(embed_cache.exists(model, key));

        let mtime_first = embed_dir
            .path()
            .join("embeddings")
            .join(model)
            .join(key)
            .join("embeddings.bin")
            .metadata()
            .unwrap()
            .modified()
            .unwrap();

        // Small sleep so mtime changes if file is rewritten
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Second ingest with force=true — should re-embed and overwrite cache
        ingest_paper(
            &rag,
            IngestParams {
                item_key: key.to_string(),
                paper_id: key.to_string(),
                title: "Test".to_string(),
                authors: vec![],
                year: None,
                venue: None,
                tags: vec![],
                cache_dir: datalab_dir.path().join(key),
                force: true,
            },
        )
        .await
        .unwrap();

        let mtime_second = embed_dir
            .path()
            .join("embeddings")
            .join(model)
            .join(key)
            .join("embeddings.bin")
            .metadata()
            .unwrap()
            .modified()
            .unwrap();

        assert_ne!(mtime_first, mtime_second, "force=true should overwrite the cache file");

        unsafe {
            std::env::remove_var("PAPERS_DATALAB_CACHE_DIR");
            std::env::remove_var("PAPERS_EMBED_CACHE_DIR");
        }
    }
}
