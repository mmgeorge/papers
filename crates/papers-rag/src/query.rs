use arrow_array::{
    Array, Float32Array, LargeStringArray, ListArray, RecordBatch, StringArray, UInt16Array,
};
use arrow_schema::DataType;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::collections::HashMap;

use crate::error::RagError;
use crate::filter::{validate_scope, FilterBuilder};
use crate::store::RagStore;
use crate::types::{
    ChapterResult, ChapterSection, ChunkResult, ChunkSummary, ChunkWithPosition,
    FigureResult, FigureSearchResult, ListPapersParams, ListTagsParams, OutlineChapter,
    OutlineSection, PaperOutline, PaperSummary, PositionContext, ReferencedFigure,
    SearchChunkResult, SearchFiguresParams, SearchParams, SearchResult, SectionResult, TagSummary,
};

// ── Arrow extraction helpers ────────────────────────────────────────────────

fn arrow_err(name: &str, expected: &str, actual: &DataType) -> RagError {
    RagError::Arrow(format!(
        "column '{name}': expected {expected}, got {actual:?}"
    ))
}

fn missing_col(name: &str) -> RagError {
    RagError::Arrow(format!("missing column '{name}'"))
}

fn col_str(batch: &RecordBatch, name: &str, row: usize) -> Result<String, RagError> {
    let col = batch.column_by_name(name).ok_or_else(|| missing_col(name))?;
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        Ok(if arr.is_null(row) { String::new() } else { arr.value(row).to_string() })
    } else if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        Ok(if arr.is_null(row) { String::new() } else { arr.value(row).to_string() })
    } else {
        Err(arrow_err(name, "Utf8 or LargeUtf8", col.data_type()))
    }
}

fn col_str_opt(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<String>, RagError> {
    let col = batch.column_by_name(name).ok_or_else(|| missing_col(name))?;
    if *col.data_type() == DataType::Null {
        return Ok(None);
    }
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        Ok(if arr.is_null(row) { None } else { Some(arr.value(row).to_string()) })
    } else if let Some(arr) = col.as_any().downcast_ref::<LargeStringArray>() {
        Ok(if arr.is_null(row) { None } else { Some(arr.value(row).to_string()) })
    } else {
        Err(arrow_err(name, "Utf8 or LargeUtf8", col.data_type()))
    }
}

fn col_u16(batch: &RecordBatch, name: &str, row: usize) -> Result<u16, RagError> {
    let col = batch.column_by_name(name).ok_or_else(|| missing_col(name))?;
    let arr = col.as_any().downcast_ref::<UInt16Array>()
        .ok_or_else(|| arrow_err(name, "UInt16", col.data_type()))?;
    Ok(arr.value(row))
}

fn col_u16_opt(batch: &RecordBatch, name: &str, row: usize) -> Result<Option<u16>, RagError> {
    let col = batch.column_by_name(name).ok_or_else(|| missing_col(name))?;
    let arr = col.as_any().downcast_ref::<UInt16Array>()
        .ok_or_else(|| arrow_err(name, "UInt16", col.data_type()))?;
    Ok(if arr.is_null(row) { None } else { Some(arr.value(row)) })
}

fn col_str_list(batch: &RecordBatch, name: &str, row: usize) -> Result<Vec<String>, RagError> {
    let col = batch.column_by_name(name).ok_or_else(|| missing_col(name))?;
    if col.is_null(row) {
        return Ok(vec![]);
    }
    let arr = col.as_any().downcast_ref::<ListArray>()
        .ok_or_else(|| arrow_err(name, "List", col.data_type()))?;
    let list_val = arr.value(row);
    let str_arr = list_val.as_any().downcast_ref::<StringArray>()
        .ok_or_else(|| arrow_err(name, "List<Utf8>", col.data_type()))?;
    Ok((0..str_arr.len())
        .filter(|&i| !str_arr.is_null(i))
        .map(|i| str_arr.value(i).to_string())
        .collect())
}

fn col_f32(batch: &RecordBatch, name: &str, row: usize) -> Result<f32, RagError> {
    let col = batch.column_by_name(name).ok_or_else(|| missing_col(name))?;
    let arr = col.as_any().downcast_ref::<Float32Array>()
        .ok_or_else(|| arrow_err(name, "Float32", col.data_type()))?;
    Ok(arr.value(row))
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

// ── Chunk building ──────────────────────────────────────────────────────────

fn chunk_from_row(batch: &RecordBatch, row: usize) -> Result<ChunkData, RagError> {
    Ok(ChunkData {
        chunk_id: col_str(batch, "chunk_id", row)?,
        paper_id: col_str(batch, "paper_id", row)?,
        title: col_str(batch, "title", row)?,
        authors: col_str_list(batch, "authors", row)?,
        year: col_u16_opt(batch, "year", row)?,
        venue: col_str_opt(batch, "venue", row)?,
        text: col_str(batch, "text", row)?,
        chapter_title: col_str(batch, "chapter_title", row)?,
        chapter_idx: col_u16(batch, "chapter_idx", row)?,
        section_title: col_str(batch, "section_title", row)?,
        section_idx: col_u16(batch, "section_idx", row)?,
        chunk_idx: col_u16(batch, "chunk_idx", row)?,
        depth: col_str(batch, "depth", row)?,
        block_type: col_str(batch, "block_type", row)?,
        figure_ids: col_str_list(batch, "figure_ids", row)?,
    })
}

struct ChunkData {
    chunk_id: String,
    paper_id: String,
    title: String,
    authors: Vec<String>,
    year: Option<u16>,
    venue: Option<String>,
    text: String,
    chapter_title: String,
    chapter_idx: u16,
    section_title: String,
    section_idx: u16,
    chunk_idx: u16,
    depth: String,
    block_type: String,
    figure_ids: Vec<String>,
}

// ── Shared async helpers ────────────────────────────────────────────────────

const PREVIEW_MIN_CHARS: usize = 120;
const PREVIEW_MAX_CHARS: usize = 300;

/// Truncate `text` at a sentence boundary. Takes at least `min_chars`, then
/// scans forward for `.`/`?`/`!` followed by whitespace or end-of-string.
/// Caps at `max_chars` to prevent runaway.
fn truncate_at_sentence(text: &str, min_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= min_chars {
        return text.to_string();
    }

    let max_chars = PREVIEW_MAX_CHARS;
    let chars: Vec<char> = text.chars().collect();
    let limit = char_count.min(max_chars);

    // Scan from min_chars forward for sentence-ending punctuation
    for i in min_chars..limit {
        if matches!(chars[i], '.' | '?' | '!') {
            // Must be followed by whitespace or end-of-string
            let at_end = i + 1 >= char_count;
            let followed_by_space = !at_end && chars[i + 1].is_whitespace();
            if at_end || followed_by_space {
                return chars[..=i].iter().collect();
            }
        }
    }

    // No sentence boundary found — hard truncate at max_chars
    let mut result: String = chars[..limit].iter().collect();
    if limit < char_count {
        result.push_str("...");
    }
    result
}

/// Key for neighbor lookups: (paper_id, chapter_idx, section_idx, chunk_idx).
type NeighborKey = (String, u16, u16, u16);

struct NeighborRow {
    chunk_id: String,
    text: String,
    block_type: String,
}

/// Batch-fetch neighbor rows for multiple search results in a single query.
/// Returns a map from (paper_id, chapter_idx, section_idx, chunk_idx) to NeighborRow.
async fn batch_fetch_neighbor_rows(
    table: &lancedb::Table,
    keys: &[NeighborKey],
) -> Result<HashMap<NeighborKey, NeighborRow>, RagError> {
    if keys.is_empty() {
        return Ok(HashMap::new());
    }

    let clauses: Vec<String> = keys
        .iter()
        .map(|(pid, ch, sec, ci)| {
            format!(
                "(paper_id = '{}' AND chapter_idx = {} AND section_idx = {} AND chunk_idx = {})",
                pid.replace('\'', "''"),
                ch,
                sec,
                ci
            )
        })
        .collect();
    let filter = clauses.join(" OR ");

    let batches = table
        .query()
        .only_if(&filter)
        .select(Select::columns(&[
            "paper_id",
            "chapter_idx",
            "section_idx",
            "chunk_idx",
            "chunk_id",
            "text",
            "block_type",
        ]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(RagError::LanceDb)?;

    let mut map = HashMap::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let key = (
                col_str(batch, "paper_id", row)?,
                col_u16(batch, "chapter_idx", row)?,
                col_u16(batch, "section_idx", row)?,
                col_u16(batch, "chunk_idx", row)?,
            );
            map.insert(
                key,
                NeighborRow {
                    chunk_id: col_str(batch, "chunk_id", row)?,
                    text: col_str(batch, "text", row)?,
                    block_type: col_str(batch, "block_type", row)?,
                },
            );
        }
    }
    Ok(map)
}

/// Resolve prev/next neighbors for a single result from a pre-fetched neighbor map.
/// If the immediate neighbor is an equation, looks through to the "far" neighbor
/// (which must also be in the map).
fn resolve_neighbors_from_map(
    map: &HashMap<NeighborKey, NeighborRow>,
    paper_id: &str,
    chapter_idx: u16,
    section_idx: u16,
    chunk_idx: u16,
) -> (Option<ChunkSummary>, Option<ChunkSummary>) {
    // Prev
    let prev = if chunk_idx > 0 {
        let prev_key = (paper_id.to_string(), chapter_idx, section_idx, chunk_idx - 1);
        match map.get(&prev_key) {
            Some(row) if row.block_type == "equation" => {
                let far_preview = if chunk_idx >= 2 {
                    let far_key = (paper_id.to_string(), chapter_idx, section_idx, chunk_idx - 2);
                    map.get(&far_key)
                        .map(|far| truncate_at_sentence(&far.text, PREVIEW_MIN_CHARS))
                        .unwrap_or_default()
                } else {
                    String::new()
                };
                let preview = if far_preview.is_empty() {
                    row.text.clone()
                } else {
                    format!("{} {}", far_preview, row.text)
                };
                Some(ChunkSummary {
                    chunk_id: row.chunk_id.clone(),
                    text_preview: preview,
                })
            }
            Some(row) => Some(ChunkSummary {
                chunk_id: row.chunk_id.clone(),
                text_preview: truncate_at_sentence(&row.text, PREVIEW_MIN_CHARS),
            }),
            None => None,
        }
    } else {
        None
    };

    // Next
    let next_idx = chunk_idx + 1;
    let next_key = (paper_id.to_string(), chapter_idx, section_idx, next_idx);
    let next = match map.get(&next_key) {
        Some(row) if row.block_type == "equation" => {
            let far_key = (paper_id.to_string(), chapter_idx, section_idx, next_idx + 1);
            let far_preview = map
                .get(&far_key)
                .map(|far| truncate_at_sentence(&far.text, PREVIEW_MIN_CHARS))
                .unwrap_or_default();
            let preview = if far_preview.is_empty() {
                row.text.clone()
            } else {
                format!("{} {}", row.text, far_preview)
            };
            Some(ChunkSummary {
                chunk_id: row.chunk_id.clone(),
                text_preview: preview,
            })
        }
        Some(row) => Some(ChunkSummary {
            chunk_id: row.chunk_id.clone(),
            text_preview: truncate_at_sentence(&row.text, PREVIEW_MIN_CHARS),
        }),
        None => None,
    };

    (prev, next)
}

/// Single-result fetch_neighbors for use outside of search (get_chunk, etc.).
async fn fetch_neighbors(
    table: &lancedb::Table,
    paper_id: &str,
    chapter_idx: u16,
    section_idx: u16,
    chunk_idx: u16,
) -> Result<(Option<ChunkSummary>, Option<ChunkSummary>), RagError> {
    // Collect all candidate keys (prev, prev-1, next, next+1)
    let mut keys: Vec<NeighborKey> = Vec::with_capacity(4);
    if chunk_idx > 0 {
        keys.push((paper_id.to_string(), chapter_idx, section_idx, chunk_idx - 1));
        if chunk_idx >= 2 {
            keys.push((paper_id.to_string(), chapter_idx, section_idx, chunk_idx - 2));
        }
    }
    keys.push((paper_id.to_string(), chapter_idx, section_idx, chunk_idx + 1));
    keys.push((paper_id.to_string(), chapter_idx, section_idx, chunk_idx + 2));

    let map = batch_fetch_neighbor_rows(table, &keys).await?;
    Ok(resolve_neighbors_from_map(
        &map,
        paper_id,
        chapter_idx,
        section_idx,
        chunk_idx,
    ))
}

async fn resolve_figures(
    figures_table: &lancedb::Table,
    figure_ids: &[String],
) -> Result<Vec<ReferencedFigure>, RagError> {
    if figure_ids.is_empty() {
        return Ok(vec![]);
    }
    let id_list = figure_ids
        .iter()
        .map(|id| format!("'{}'", id.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(", ");
    let filter = format!("figure_id IN ({id_list})");
    let batches = figures_table
        .query()
        .only_if(&filter)
        .select(Select::columns(&[
            "figure_id",
            "figure_type",
            "caption",
            "description",
        ]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    let mut results = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            results.push(ReferencedFigure {
                figure_id: col_str(batch, "figure_id", row)?,
                figure_type: col_str(batch, "figure_type", row)?,
                caption: col_str(batch, "caption", row)?,
                description: col_str_opt(batch, "description", row)?,
            });
        }
    }
    Ok(results)
}

async fn position_context(
    table: &lancedb::Table,
    paper_id: &str,
    chapter_idx: u16,
    section_idx: u16,
    chunk_idx: u16,
) -> Result<PositionContext, RagError> {
    let paper_id_esc = paper_id.replace('\'', "''");

    // Total chunks in this section
    let section_filter = format!(
        "paper_id = '{paper_id_esc}' AND chapter_idx = {chapter_idx} AND section_idx = {section_idx}"
    );
    let section_batches = table
        .query()
        .only_if(&section_filter)
        .select(Select::columns(&["chunk_idx"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;
    let total_in_section = total_rows(&section_batches) as u32;

    // Total sections in this chapter
    let chapter_batches = table
        .query()
        .only_if(&format!(
            "paper_id = '{paper_id_esc}' AND chapter_idx = {chapter_idx}"
        ))
        .select(Select::columns(&["section_idx"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;
    let mut section_set = std::collections::HashSet::new();
    for b in &chapter_batches {
        for r in 0..b.num_rows() {
            section_set.insert(col_u16(b, "section_idx", r)?);
        }
    }
    let total_sections = section_set.len() as u32;

    // Total chapters in this paper
    let paper_batches = table
        .query()
        .only_if(&format!("paper_id = '{paper_id_esc}'"))
        .select(Select::columns(&["chapter_idx"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;
    let mut chapter_set = std::collections::HashSet::new();
    for b in &paper_batches {
        for r in 0..b.num_rows() {
            chapter_set.insert(col_u16(b, "chapter_idx", r)?);
        }
    }
    let total_chapters = chapter_set.len() as u32;

    Ok(PositionContext {
        total_chunks_in_section: total_in_section,
        total_sections_in_chapter: total_sections,
        total_chapters_in_paper: total_chapters,
        is_first_in_section: chunk_idx == 0,
        is_last_in_section: chunk_idx + 1 >= total_in_section as u16,
    })
}

async fn build_chunk_with_position(
    store: &RagStore,
    data: ChunkData,
) -> Result<ChunkWithPosition, RagError> {
    let chunks_table = store.chunks_table().await?;
    let figures_table = store.figures_table().await?;
    let pos = position_context(
        &chunks_table,
        &data.paper_id,
        data.chapter_idx,
        data.section_idx,
        data.chunk_idx,
    )
    .await?;
    let referenced_figures = resolve_figures(&figures_table, &data.figure_ids).await?;
    Ok(ChunkWithPosition {
        chunk_id: data.chunk_id,
        paper_id: data.paper_id,
        title: data.title,
        authors: data.authors,
        year: data.year,
        venue: data.venue,
        text: data.text,
        chapter_title: data.chapter_title,
        chapter_idx: data.chapter_idx,
        section_title: data.section_title,
        section_idx: data.section_idx,
        chunk_idx: data.chunk_idx,
        depth: data.depth,
        block_type: data.block_type,
        figure_ids: data.figure_ids,
        referenced_figures,
        position: pos,
    })
}

// ── Public query functions ──────────────────────────────────────────────────

/// Resolve flexible user input to an exact `paper_id` in the RAG database.
///
/// Resolution order:
/// 1. **Exact match** — if `input` matches an existing `paper_id`, return it immediately.
/// 2. **Title search** — case-insensitive substring match on the `title` column;
///    returns the first match's `paper_id`.
///
/// Returns a `NotFound` error with a list of available papers when no match is found.
pub async fn resolve_paper_id(store: &RagStore, input: &str) -> Result<String, RagError> {
    let table = store.chunks_table().await?;

    // Fetch all distinct (paper_id, title) pairs
    let batches = table
        .query()
        .select(Select::columns(&["paper_id", "title"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(RagError::LanceDb)?;

    // Deduplicate into (paper_id, title) pairs
    let mut seen = std::collections::HashSet::new();
    let mut papers: Vec<(String, String)> = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let pid = col_str(batch, "paper_id", row)?;
            if seen.insert(pid.clone()) {
                let title = col_str(batch, "title", row)?;
                papers.push((pid, title));
            }
        }
    }

    // 1. Exact paper_id match
    for (pid, _) in &papers {
        if pid == input {
            return Ok(pid.clone());
        }
    }

    // 2. Case-insensitive title substring match
    let input_lower = input.to_lowercase();
    let matches: Vec<&(String, String)> = papers
        .iter()
        .filter(|(_, title)| title.to_lowercase().contains(&input_lower))
        .collect();

    match matches.len() {
        0 => {
            let available: Vec<String> = papers
                .iter()
                .map(|(pid, title)| format!("  {pid} — {title}"))
                .collect();
            Err(RagError::NotFound(format!(
                "no paper matching '{}'. Available papers:\n{}",
                input,
                available.join("\n")
            )))
        }
        _ => Ok(matches[0].0.clone()),
    }
}

/// Semantic search across indexed paper chunks.
pub async fn search(
    store: &RagStore,
    params: SearchParams,
) -> Result<Vec<SearchResult>, RagError> {
    let embedding = store.embed_query(&params.query).await?;
    search_with_embedding(store, params, &embedding).await
}

/// Search with a pre-computed embedding vector (used by benchmarks to bypass the embedder).
#[cfg(any(test, feature = "bench"))]
pub async fn search_with_embedding(
    store: &RagStore,
    params: SearchParams,
    embedding: &[f32],
) -> Result<Vec<SearchResult>, RagError> {
    search_with_embedding_inner(store, params, embedding).await
}

#[cfg(not(any(test, feature = "bench")))]
async fn search_with_embedding(
    store: &RagStore,
    params: SearchParams,
    embedding: &[f32],
) -> Result<Vec<SearchResult>, RagError> {
    search_with_embedding_inner(store, params, embedding).await
}

async fn search_with_embedding_inner(
    store: &RagStore,
    params: SearchParams,
    embedding: &[f32],
) -> Result<Vec<SearchResult>, RagError> {
    validate_scope(
        params.chapter_idx,
        params.section_idx,
        params.paper_ids
            .as_deref()
            .and_then(|ids| ids.first())
            .map(|s| s.as_str()),
    )?;
    let table = store.chunks_table().await?;

    let mut fb = FilterBuilder::new();
    if let Some(ids) = params.paper_ids.as_deref() {
        fb = fb.paper_ids(ids);
    }
    if let Some(ch) = params.chapter_idx {
        fb = fb.chapter_idx(ch);
    }
    if let Some(sec) = params.section_idx {
        fb = fb.section_idx(sec);
    }
    fb = fb.year_range(params.filter_year_min, params.filter_year_max);
    if let Some(venue) = &params.filter_venue {
        fb = fb.eq_str("venue", venue);
    }
    if let Some(depth) = &params.filter_depth {
        fb = fb.eq_str("depth", depth);
    }
    if let Some(tags) = params.filter_tags.as_deref() {
        fb = fb.tags_any(tags);
    }

    let mut query_builder = table.query().nearest_to(embedding)?;
    query_builder = query_builder.limit(params.limit as usize);
    if let Some(filter) = fb.build() {
        query_builder = query_builder.only_if(filter);
    }

    let batches = query_builder
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    // Collect all chunk data and scores first
    let mut chunk_data_list: Vec<(ChunkData, f32)> = Vec::new();
    for batch in &batches {
        let has_distance = batch.column_by_name("_distance").is_some();
        for row in 0..batch.num_rows() {
            let score = if has_distance {
                col_f32(batch, "_distance", row)?
            } else {
                0.0
            };
            chunk_data_list.push((chunk_from_row(batch, row)?, score));
        }
    }

    // Collect all neighbor keys across all results in one pass
    let mut neighbor_keys: Vec<NeighborKey> = Vec::new();
    for (data, _) in &chunk_data_list {
        let pid = &data.paper_id;
        let ch = data.chapter_idx;
        let sec = data.section_idx;
        let ci = data.chunk_idx;
        if ci > 0 {
            neighbor_keys.push((pid.clone(), ch, sec, ci - 1));
            if ci >= 2 {
                neighbor_keys.push((pid.clone(), ch, sec, ci - 2));
            }
        }
        neighbor_keys.push((pid.clone(), ch, sec, ci + 1));
        neighbor_keys.push((pid.clone(), ch, sec, ci + 2));
    }
    // Deduplicate keys
    neighbor_keys.sort();
    neighbor_keys.dedup();

    // Single batch query for all neighbors
    let chunks_table = store.chunks_table().await?;
    let neighbor_map = batch_fetch_neighbor_rows(&chunks_table, &neighbor_keys).await?;

    // Build results using the pre-fetched map
    let mut results = Vec::new();
    for (data, score) in chunk_data_list {
        let (prev, next) = resolve_neighbors_from_map(
            &neighbor_map,
            &data.paper_id,
            data.chapter_idx,
            data.section_idx,
            data.chunk_idx,
        );
        let chunk = SearchChunkResult {
            chunk_id: data.chunk_id,
            paper_id: data.paper_id,
            paper_title: data.title,
            block_type: data.block_type,
            text: data.text,
            chapter_title: data.chapter_title,
            section_title: data.section_title,
            chunk_idx: data.chunk_idx,
            figure_ids: data.figure_ids,
        };
        results.push(SearchResult {
            chunk,
            prev,
            next,
            score,
        });
    }
    Ok(results)
}

/// Search for figures/tables by description.
pub async fn search_figures(
    store: &RagStore,
    params: SearchFiguresParams,
) -> Result<Vec<FigureSearchResult>, RagError> {
    let embedding = store.embed_query(&params.query).await?;
    search_figures_with_embedding_inner(store, params, &embedding).await
}

/// Search figures with a pre-computed embedding vector (for benchmarks).
#[cfg(any(test, feature = "bench"))]
pub async fn search_figures_with_embedding(
    store: &RagStore,
    params: SearchFiguresParams,
    embedding: &[f32],
) -> Result<Vec<FigureSearchResult>, RagError> {
    search_figures_with_embedding_inner(store, params, embedding).await
}

async fn search_figures_with_embedding_inner(
    store: &RagStore,
    params: SearchFiguresParams,
    embedding: &[f32],
) -> Result<Vec<FigureSearchResult>, RagError> {
    let table = store.figures_table().await?;

    let mut fb = FilterBuilder::new();
    if let Some(ids) = params.paper_ids.as_deref() {
        fb = fb.paper_ids(ids);
    }
    if let Some(ft) = &params.filter_figure_type {
        fb = fb.eq_str("figure_type", ft);
    }

    let mut query_builder = table.query().nearest_to(embedding)?;
    query_builder = query_builder.limit(params.limit as usize);
    if let Some(filter) = fb.build() {
        query_builder = query_builder.only_if(filter);
    }

    let batches = query_builder
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    let mut results = Vec::new();
    for batch in &batches {
        let has_distance = batch.column_by_name("_distance").is_some();
        for row in 0..batch.num_rows() {
            let score = if has_distance {
                col_f32(batch, "_distance", row)?
            } else {
                0.0
            };
            results.push(FigureSearchResult {
                figure_id: col_str(batch, "figure_id", row)?,
                paper_id: col_str(batch, "paper_id", row)?,
                figure_type: col_str(batch, "figure_type", row)?,
                caption: col_str(batch, "caption", row)?,
                description: col_str_opt(batch, "description", row)?,
                image_path: col_str_opt(batch, "image_path", row)?,
                content: col_str_opt(batch, "content", row)?,
                page: col_u16_opt(batch, "page", row)?,
                score,
            });
        }
    }
    Ok(results)
}

/// Get a single chunk by ID with prev/next neighbors.
pub async fn get_chunk(store: &RagStore, chunk_id: &str) -> Result<ChunkResult, RagError> {
    let table = store.chunks_table().await?;
    let escaped = chunk_id.replace('\'', "''");
    let filter = format!("chunk_id = '{escaped}'");
    let batches = table
        .query()
        .only_if(&filter)
        .limit(1)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    if total_rows(&batches) == 0 {
        return Err(RagError::NotFound(format!("chunk not found: {chunk_id}")));
    }
    let batch = &batches[0];
    let data = chunk_from_row(batch, 0)?;
    let paper_id = data.paper_id.clone();
    let chapter_idx = data.chapter_idx;
    let section_idx = data.section_idx;
    let chunk_idx = data.chunk_idx;
    let chunk = build_chunk_with_position(store, data).await?;
    let (prev, next) = fetch_neighbors(&table, &paper_id, chapter_idx, section_idx, chunk_idx).await?;
    Ok(ChunkResult { chunk, prev, next })
}

/// Fetch all chunks in a section in reading order.
pub async fn get_section(
    store: &RagStore,
    paper_id: &str,
    chapter_idx: u16,
    section_idx: u16,
) -> Result<SectionResult, RagError> {
    let table = store.chunks_table().await?;
    let paper_id_esc = paper_id.replace('\'', "''");
    let filter = format!(
        "paper_id = '{paper_id_esc}' AND chapter_idx = {chapter_idx} AND section_idx = {section_idx}"
    );
    let batches = table
        .query()
        .only_if(&filter)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    // Sort by chunk_idx
    let mut rows: Vec<(u16, &RecordBatch, usize)> = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            rows.push((col_u16(batch, "chunk_idx", row)?, batch, row));
        }
    }
    rows.sort_by_key(|(idx, _, _)| *idx);

    let mut chapter_title = String::new();
    let mut section_title = String::new();
    let mut chunks = Vec::new();
    for (_, batch, row) in rows {
        let data = chunk_from_row(batch, row)?;
        if chapter_title.is_empty() {
            chapter_title = data.chapter_title.clone();
            section_title = data.section_title.clone();
        }
        let chunk = build_chunk_with_position(store, data).await?;
        chunks.push(chunk);
    }

    let total = chunks.len();
    Ok(SectionResult {
        paper_id: paper_id.to_string(),
        chapter_title,
        section_title,
        chunks,
        total_chunks: total,
    })
}

/// Fetch all content of a chapter, grouped by section.
pub async fn get_chapter(
    store: &RagStore,
    paper_id: &str,
    chapter_idx: u16,
) -> Result<ChapterResult, RagError> {
    let table = store.chunks_table().await?;
    let paper_id_esc = paper_id.replace('\'', "''");
    let filter = format!("paper_id = '{paper_id_esc}' AND chapter_idx = {chapter_idx}");
    let batches = table
        .query()
        .only_if(&filter)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    // Sort by (section_idx, chunk_idx)
    let mut rows: Vec<(u16, u16, &RecordBatch, usize)> = Vec::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            rows.push((
                col_u16(batch, "section_idx", row)?,
                col_u16(batch, "chunk_idx", row)?,
                batch,
                row,
            ));
        }
    }
    rows.sort_by_key(|(sec, ch, _, _)| (*sec, *ch));

    let mut chapter_title = String::new();
    let mut all_figure_ids: Vec<String> = Vec::new();
    let mut sections: Vec<ChapterSection> = Vec::new();
    let mut current_section_idx: Option<u16> = None;
    let mut current_section_title = String::new();
    let mut current_chunks: Vec<ChunkWithPosition> = Vec::new();

    for (section_idx, _, batch, row) in rows {
        let data = chunk_from_row(batch, row)?;
        if chapter_title.is_empty() {
            chapter_title = data.chapter_title.clone();
        }
        for fid in &data.figure_ids {
            if !all_figure_ids.contains(fid) {
                all_figure_ids.push(fid.clone());
            }
        }

        if current_section_idx != Some(section_idx) {
            if let Some(_) = current_section_idx {
                sections.push(ChapterSection {
                    section_idx: current_section_idx.unwrap(),
                    section_title: current_section_title.clone(),
                    chunks: std::mem::take(&mut current_chunks),
                });
            }
            current_section_idx = Some(section_idx);
            current_section_title = data.section_title.clone();
        }
        let chunk = build_chunk_with_position(store, data).await?;
        current_chunks.push(chunk);
    }
    if current_section_idx.is_some() && !current_chunks.is_empty() {
        sections.push(ChapterSection {
            section_idx: current_section_idx.unwrap(),
            section_title: current_section_title,
            chunks: current_chunks,
        });
    }

    let total_chunks = sections.iter().map(|s| s.chunks.len()).sum();
    Ok(ChapterResult {
        paper_id: paper_id.to_string(),
        chapter_title,
        chapter_idx,
        sections,
        total_chunks,
        figure_ids: all_figure_ids,
    })
}

/// Retrieve a figure by ID.
pub async fn get_figure(store: &RagStore, figure_id: &str) -> Result<FigureResult, RagError> {
    let table = store.figures_table().await?;
    let escaped = figure_id.replace('\'', "''");
    let filter = format!("figure_id = '{escaped}'");
    let batches = table
        .query()
        .only_if(&filter)
        .limit(1)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    if total_rows(&batches) == 0 {
        return Err(RagError::NotFound(format!(
            "figure not found: {figure_id}"
        )));
    }
    let batch = &batches[0];
    Ok(FigureResult {
        figure_id: col_str(batch, "figure_id", 0)?,
        paper_id: col_str(batch, "paper_id", 0)?,
        figure_type: col_str(batch, "figure_type", 0)?,
        caption: col_str(batch, "caption", 0)?,
        description: col_str_opt(batch, "description", 0)?,
        image_path: col_str_opt(batch, "image_path", 0)?,
        content: col_str_opt(batch, "content", 0)?,
        page: col_u16_opt(batch, "page", 0)?,
        referenced_by: vec![],
    })
}

/// Get the table of contents for a paper.
pub async fn get_paper_outline(
    store: &RagStore,
    paper_id: &str,
) -> Result<PaperOutline, RagError> {
    let table = store.chunks_table().await?;
    let figures_table = store.figures_table().await?;
    let paper_id_esc = paper_id.replace('\'', "''");

    let batches = table
        .query()
        .only_if(&format!("paper_id = '{paper_id_esc}'"))
        .select(Select::columns(&[
            "chapter_idx",
            "chapter_title",
            "section_idx",
            "section_title",
            "chunk_idx",
            "depth",
            "title",
            "authors",
            "year",
            "venue",
            "tags",
        ]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    if total_rows(&batches) == 0 {
        return Err(RagError::NotFound(format!(
            "paper not found: {paper_id}"
        )));
    }

    // Collect paper metadata from first row
    let first_batch = &batches[0];
    let paper_title = col_str(first_batch, "title", 0)?;
    let authors = col_str_list(first_batch, "authors", 0)?;
    let year = col_u16_opt(first_batch, "year", 0)?;
    let venue = col_str_opt(first_batch, "venue", 0)?;
    let tags = col_str_list(first_batch, "tags", 0)?;

    // Group by chapter → section
    let mut chapter_map: HashMap<u16, (String, HashMap<u16, (String, usize)>)> = HashMap::new();
    let mut total_chunks = 0usize;

    for batch in &batches {
        for row in 0..batch.num_rows() {
            let ch_idx = col_u16(batch, "chapter_idx", row)?;
            let ch_title = col_str(batch, "chapter_title", row)?;
            let sec_idx = col_u16(batch, "section_idx", row)?;
            let sec_title = col_str(batch, "section_title", row)?;

            let entry = chapter_map
                .entry(ch_idx)
                .or_insert_with(|| (ch_title, HashMap::new()));
            let sec = entry.1.entry(sec_idx).or_insert_with(|| (sec_title, 0));
            sec.1 += 1;
            total_chunks += 1;
        }
    }

    // Count figures
    let fig_batches = figures_table
        .query()
        .only_if(&format!("paper_id = '{paper_id_esc}'"))
        .select(Select::columns(&["figure_id", "chapter_idx"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;
    let total_figures = total_rows(&fig_batches);

    // Count figures per chapter
    let mut fig_per_chapter: HashMap<u16, usize> = HashMap::new();
    for batch in &fig_batches {
        for row in 0..batch.num_rows() {
            let ch_idx = col_u16(batch, "chapter_idx", row)?;
            *fig_per_chapter.entry(ch_idx).or_insert(0) += 1;
        }
    }

    // Build sorted chapter list
    let mut chapter_idxs: Vec<u16> = chapter_map.keys().copied().collect();
    chapter_idxs.sort();
    let chapters = chapter_idxs
        .into_iter()
        .map(|ch_idx| {
            let (ch_title, sec_map) = chapter_map.remove(&ch_idx).unwrap();
            let mut sec_idxs: Vec<u16> = sec_map.keys().copied().collect();
            sec_idxs.sort();
            let sections = sec_idxs
                .into_iter()
                .map(|sec_idx| {
                    let (sec_title, count) = sec_map[&sec_idx].clone();
                    OutlineSection {
                        section_idx: sec_idx,
                        section_title: sec_title,
                        chunk_count: count,
                    }
                })
                .collect();
            OutlineChapter {
                chapter_idx: ch_idx,
                chapter_title: ch_title,
                sections,
                figure_count: *fig_per_chapter.get(&ch_idx).unwrap_or(&0),
            }
        })
        .collect();

    Ok(PaperOutline {
        paper_id: paper_id.to_string(),
        title: paper_title,
        authors,
        year,
        venue,
        tags,
        chapters,
        total_chunks,
        total_figures,
    })
}

/// Browse indexed papers with optional filters.
pub async fn list_papers(
    store: &RagStore,
    params: ListPapersParams,
) -> Result<Vec<PaperSummary>, RagError> {
    let table = store.chunks_table().await?;
    let figures_table = store.figures_table().await?;

    let mut fb = FilterBuilder::new();
    if let Some(ids) = params.paper_ids.as_deref() {
        fb = fb.paper_ids(ids);
    }
    fb = fb.year_range(params.filter_year_min, params.filter_year_max);
    if let Some(venue) = &params.filter_venue {
        fb = fb.eq_str("venue", venue);
    }
    if let Some(tags) = params.filter_tags.as_deref() {
        fb = fb.tags_any(tags);
    }

    let mut query = table
        .query()
        .select(Select::columns(&[
            "paper_id", "title", "authors", "year", "venue", "tags",
        ]));
    if let Some(filter) = fb.build() {
        query = query.only_if(filter);
    }

    let batches = query
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    // Aggregate by paper_id
    let mut paper_map: HashMap<String, PaperSummary> = HashMap::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let pid = col_str(batch, "paper_id", row)?;
            if !paper_map.contains_key(&pid) {
                paper_map.insert(pid.clone(), PaperSummary {
                    paper_id: pid.clone(),
                    title: col_str(batch, "title", row)?,
                    authors: col_str_list(batch, "authors", row)?,
                    year: col_u16_opt(batch, "year", row)?,
                    venue: col_str_opt(batch, "venue", row)?,
                    tags: col_str_list(batch, "tags", row)?,
                    chunk_count: 0,
                    figure_count: 0,
                });
            }
            paper_map.get_mut(&pid).unwrap().chunk_count += 1;
        }
    }

    // Apply author filter (post-filter since we can't do array search easily)
    if let Some(authors_filter) = &params.filter_authors {
        paper_map.retain(|_, p| {
            authors_filter.iter().any(|af| {
                p.authors
                    .iter()
                    .any(|a| a.to_lowercase().contains(&af.to_lowercase()))
            })
        });
    }

    // Count figures per paper
    let fig_batches = figures_table
        .query()
        .select(Select::columns(&["paper_id"]))
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;
    for batch in &fig_batches {
        for row in 0..batch.num_rows() {
            let pid = col_str(batch, "paper_id", row)?;
            if let Some(entry) = paper_map.get_mut(&pid) {
                entry.figure_count += 1;
            }
        }
    }

    let mut papers: Vec<PaperSummary> = paper_map.into_values().collect();

    // Sort
    match params.sort_by.as_deref().unwrap_or("year") {
        "title" => papers.sort_by(|a, b| a.title.cmp(&b.title)),
        _ => papers.sort_by(|a, b| b.year.cmp(&a.year)),
    }

    papers.truncate(params.limit as usize);
    Ok(papers)
}

/// List all tags with paper counts.
pub async fn list_tags(
    store: &RagStore,
    params: ListTagsParams,
) -> Result<Vec<TagSummary>, RagError> {
    let table = store.chunks_table().await?;

    let mut query = table.query().select(Select::columns(&["paper_id", "tags"]));
    if let Some(ids) = params.paper_ids.as_deref() {
        let fb = FilterBuilder::new().paper_ids(ids);
        if let Some(filter) = fb.build() {
            query = query.only_if(filter);
        }
    }

    let batches = query
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;

    // Collect (paper_id, tag) pairs and count unique papers per tag
    let mut tag_papers: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
    for batch in &batches {
        for row in 0..batch.num_rows() {
            let pid = col_str(batch, "paper_id", row)?;
            let tags = col_str_list(batch, "tags", row)?;
            for tag in tags {
                tag_papers
                    .entry(tag)
                    .or_insert_with(std::collections::HashSet::new)
                    .insert(pid.clone());
            }
        }
    }

    let mut result: Vec<TagSummary> = tag_papers
        .into_iter()
        .map(|(tag, papers)| TagSummary {
            tag,
            paper_count: papers.len(),
        })
        .collect();
    result.sort_by(|a, b| b.paper_count.cmp(&a.paper_count).then(a.tag.cmp(&b.tag)));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── truncate_at_sentence ──────────────────────────────────────────────

    #[test]
    fn short_text_under_min_chars() {
        assert_eq!(truncate_at_sentence("Hello world.", 120), "Hello world.");
    }

    #[test]
    fn empty_string() {
        assert_eq!(truncate_at_sentence("", 120), "");
    }

    #[test]
    fn sentence_ends_shortly_after_min_chars() {
        // Build text where sentence ends at ~125 chars
        let mut text = "A".repeat(120);
        text.push_str("word. Next sentence continues here.");
        let result = truncate_at_sentence(&text, 120);
        assert!(result.ends_with('.'), "should end at period: {}", result);
        assert!(result.len() <= 130, "should cut near 125: len={}", result.len());
    }

    #[test]
    fn no_sentence_boundary_before_max() {
        // Long run-on text with no sentence-ending punctuation
        let text = "a ".repeat(200); // 400 chars, no period
        let result = truncate_at_sentence(&text, 120);
        assert!(result.ends_with("..."), "should have ellipsis: {}", result);
        // 300 chars + "..."
        assert!(result.chars().count() <= 303);
    }

    #[test]
    fn period_inside_abbreviation_not_cut() {
        // "Fig. 1" has a period at char 3, but it's before min_chars so irrelevant
        let text = "Fig. 1 shows the results of the experiment that we conducted over multiple iterations of the algorithm to verify the convergence. The next step is analysis.";
        let result = truncate_at_sentence(&text, 120);
        // Should cut at the period after "convergence" (char ~128), not at "Fig."
        assert!(result.contains("convergence."), "got: {}", result);
    }

    #[test]
    fn multiple_sentence_endings_cuts_at_first_after_min() {
        let s1 = "a".repeat(115);
        let text = format!("{}First. Second. Third.", s1);
        let result = truncate_at_sentence(&text, 120);
        // Should cut at "First." (first sentence end after min_chars)
        assert!(result.ends_with("First."), "got: {}", result);
    }

    #[test]
    fn question_mark_ending() {
        let prefix = "a".repeat(118);
        let text = format!("{}what is the result? The answer is here.", prefix);
        let result = truncate_at_sentence(&text, 120);
        assert!(result.ends_with('?'), "should cut at question mark: {}", result);
    }

    #[test]
    fn exclamation_mark_ending() {
        let prefix = "a".repeat(118);
        let text = format!("{}converges! This means progress.", prefix);
        let result = truncate_at_sentence(&text, 120);
        assert!(result.ends_with('!'), "should cut at exclamation: {}", result);
    }

    #[test]
    fn period_at_end_of_string() {
        let text = "Short sentence that ends with a period.";
        let result = truncate_at_sentence(&text, 120);
        assert_eq!(result, text);
    }

    #[test]
    fn period_not_followed_by_space_decimal() {
        // "3.14" has a period not followed by space — should NOT cut here
        let prefix = "a".repeat(118);
        let text = format!("{}uses 3.14 as pi value. Then we continue.", prefix);
        let result = truncate_at_sentence(&text, 120);
        // Should skip "3.14" (no space after '.') and cut at "value."
        assert!(result.contains("value."), "got: {}", result);
    }

    #[test]
    fn period_not_followed_by_space_url() {
        let prefix = "a".repeat(118);
        let text = format!("{}see arxiv.org for details. The paper shows.", prefix);
        let result = truncate_at_sentence(&text, 120);
        // Should skip "arxiv.org" and cut at "details."
        assert!(result.contains("details."), "got: {}", result);
    }

    #[test]
    fn text_with_latex() {
        let prefix = "a".repeat(110);
        let text = format!("{}where \\mathbf{{x}} = 0. The result follows.", prefix);
        let result = truncate_at_sentence(&text, 120);
        assert!(result.ends_with('.'), "should cut at period: {}", result);
    }

    #[test]
    fn unicode_multibyte_chars() {
        // Text with accented characters — should handle char boundaries
        let text = "Lörem ïpsum dölor sit amet, cönsectetur adipïscing elït. Sed dö eïusmöd tempor incïdidunt ut labore et dölore magna aliqüa. Ut enim ad minim veniam.";
        let result = truncate_at_sentence(&text, 120);
        assert!(result.ends_with('.'), "got: {}", result);
    }

    #[test]
    fn exactly_min_chars_no_period_after() {
        // Text exactly at min_chars with no period after
        let text = "a".repeat(120);
        let result = truncate_at_sentence(&text, 120);
        // Under max, no period, returns as-is (no truncation needed since == min)
        assert_eq!(result, text);
    }
}
