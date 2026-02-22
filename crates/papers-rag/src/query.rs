use arrow_array::{
    Array, Float32Array, ListArray, RecordBatch, StringArray, UInt16Array,
};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::collections::HashMap;

use crate::error::RagError;
use crate::filter::{validate_scope, FilterBuilder};
use crate::store::RagStore;
use crate::types::{
    ChapterResult, ChapterSection, ChunkResult, ChunkSummary, ChunkWithPosition, FigureResult,
    ListPapersParams, ListTagsParams, OutlineChapter, OutlineSection, PaperOutline, PaperSummary,
    PositionContext, ReferencedFigure, SearchFiguresParams, SearchParams, SearchResult,
    SectionResult, TagSummary,
};

// ── Arrow extraction helpers ────────────────────────────────────────────────

fn col_str(batch: &RecordBatch, name: &str, row: usize) -> String {
    let col = batch.column_by_name(name).expect("column exists");
    let arr = col.as_any().downcast_ref::<StringArray>().expect("StringArray");
    if arr.is_null(row) {
        String::new()
    } else {
        arr.value(row).to_string()
    }
}

fn col_str_opt(batch: &RecordBatch, name: &str, row: usize) -> Option<String> {
    let col = batch.column_by_name(name).expect("column exists");
    let arr = col.as_any().downcast_ref::<StringArray>().expect("StringArray");
    if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row).to_string())
    }
}

fn col_u16(batch: &RecordBatch, name: &str, row: usize) -> u16 {
    let col = batch.column_by_name(name).expect("column exists");
    let arr = col
        .as_any()
        .downcast_ref::<UInt16Array>()
        .expect("UInt16Array");
    arr.value(row)
}

fn col_u16_opt(batch: &RecordBatch, name: &str, row: usize) -> Option<u16> {
    let col = batch.column_by_name(name).expect("column exists");
    let arr = col
        .as_any()
        .downcast_ref::<UInt16Array>()
        .expect("UInt16Array");
    if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    }
}

fn col_str_list(batch: &RecordBatch, name: &str, row: usize) -> Vec<String> {
    let col = batch.column_by_name(name).expect("column exists");
    if col.is_null(row) {
        return vec![];
    }
    let arr = col.as_any().downcast_ref::<ListArray>().expect("ListArray");
    let list_val = arr.value(row);
    let str_arr = list_val
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("StringArray in list");
    (0..str_arr.len())
        .filter(|&i| !str_arr.is_null(i))
        .map(|i| str_arr.value(i).to_string())
        .collect()
}

fn col_f32(batch: &RecordBatch, name: &str, row: usize) -> f32 {
    let col = batch.column_by_name(name).expect("column exists");
    let arr = col
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("Float32Array");
    arr.value(row)
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(|b| b.num_rows()).sum()
}

// ── Chunk building ──────────────────────────────────────────────────────────

fn chunk_from_row(batch: &RecordBatch, row: usize) -> ChunkData {
    ChunkData {
        chunk_id: col_str(batch, "chunk_id", row),
        paper_id: col_str(batch, "paper_id", row),
        title: col_str(batch, "title", row),
        authors: col_str_list(batch, "authors", row),
        year: col_u16_opt(batch, "year", row),
        venue: col_str_opt(batch, "venue", row),
        text: col_str(batch, "text", row),
        chapter_title: col_str(batch, "chapter_title", row),
        chapter_idx: col_u16(batch, "chapter_idx", row),
        section_title: col_str(batch, "section_title", row),
        section_idx: col_u16(batch, "section_idx", row),
        chunk_idx: col_u16(batch, "chunk_idx", row),
        depth: col_str(batch, "depth", row),
        figure_ids: col_str_list(batch, "figure_ids", row),
    }
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
    figure_ids: Vec<String>,
}

// ── Shared async helpers ────────────────────────────────────────────────────

async fn fetch_neighbors(
    table: &lancedb::Table,
    paper_id: &str,
    chapter_idx: u16,
    section_idx: u16,
    chunk_idx: u16,
) -> Result<(Option<ChunkSummary>, Option<ChunkSummary>), RagError> {
    // Fetch prev
    let prev = if chunk_idx > 0 {
        let prev_idx = chunk_idx - 1;
        let filter = format!(
            "paper_id = '{}' AND chapter_idx = {} AND section_idx = {} AND chunk_idx = {}",
            paper_id.replace('\'', "''"),
            chapter_idx,
            section_idx,
            prev_idx
        );
        let batches = table
            .query()
            .only_if(&filter)
            .select(Select::columns(&["chunk_id", "text", "depth"]))
            .limit(1)
            .execute()
            .await?
            .try_collect::<Vec<_>>()
            .await
            .map_err(|e| RagError::LanceDb(e))?;
        if total_rows(&batches) > 0 {
            let b = &batches[0];
            Some(ChunkSummary {
                chunk_id: col_str(b, "chunk_id", 0),
                text_preview: col_str(b, "text", 0).chars().take(120).collect(),
                depth: col_str(b, "depth", 0),
            })
        } else {
            None
        }
    } else {
        None
    };

    // Fetch next
    let next_idx = chunk_idx + 1;
    let filter = format!(
        "paper_id = '{}' AND chapter_idx = {} AND section_idx = {} AND chunk_idx = {}",
        paper_id.replace('\'', "''"),
        chapter_idx,
        section_idx,
        next_idx
    );
    let batches = table
        .query()
        .only_if(&filter)
        .select(Select::columns(&["chunk_id", "text", "depth"]))
        .limit(1)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await
        .map_err(|e| RagError::LanceDb(e))?;
    let next = if total_rows(&batches) > 0 {
        let b = &batches[0];
        Some(ChunkSummary {
            chunk_id: col_str(b, "chunk_id", 0),
            text_preview: col_str(b, "text", 0).chars().take(120).collect(),
            depth: col_str(b, "depth", 0),
        })
    } else {
        None
    };

    Ok((prev, next))
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
                figure_id: col_str(batch, "figure_id", row),
                figure_type: col_str(batch, "figure_type", row),
                caption: col_str(batch, "caption", row),
                description: col_str(batch, "description", row),
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
            section_set.insert(col_u16(b, "section_idx", r));
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
            chapter_set.insert(col_u16(b, "chapter_idx", r));
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
        figure_ids: data.figure_ids,
        referenced_figures,
        position: pos,
    })
}

// ── Public query functions ──────────────────────────────────────────────────

/// Semantic search across indexed paper chunks.
pub async fn search(
    store: &RagStore,
    params: SearchParams,
) -> Result<Vec<SearchResult>, RagError> {
    validate_scope(
        params.chapter_idx,
        params.section_idx,
        params.paper_ids
            .as_deref()
            .and_then(|ids| ids.first())
            .map(|s| s.as_str()),
    )?;

    let embedding = store.embed_query(&params.query).await?;
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

    let mut query_builder = table.query().nearest_to(embedding.as_slice())?;
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
        // Check if _distance column is present
        let has_distance = batch.column_by_name("_distance").is_some();
        for row in 0..batch.num_rows() {
            let score = if has_distance {
                col_f32(batch, "_distance", row)
            } else {
                0.0
            };
            let data = chunk_from_row(batch, row);
            let paper_id = data.paper_id.clone();
            let chapter_idx = data.chapter_idx;
            let section_idx = data.section_idx;
            let chunk_idx = data.chunk_idx;
            let chunk = build_chunk_with_position(store, data).await?;
            let (prev, next) = fetch_neighbors(
                &store.chunks_table().await?,
                &paper_id,
                chapter_idx,
                section_idx,
                chunk_idx,
            )
            .await?;
            results.push(SearchResult {
                chunk,
                prev,
                next,
                score,
            });
        }
    }
    Ok(results)
}

/// Search for figures/tables by description.
pub async fn search_figures(
    store: &RagStore,
    params: SearchFiguresParams,
) -> Result<Vec<FigureResult>, RagError> {
    let embedding = store.embed_query(&params.query).await?;
    let table = store.figures_table().await?;

    let mut fb = FilterBuilder::new();
    if let Some(ids) = params.paper_ids.as_deref() {
        fb = fb.paper_ids(ids);
    }
    if let Some(ft) = &params.filter_figure_type {
        fb = fb.eq_str("figure_type", ft);
    }

    let mut query_builder = table.query().nearest_to(embedding.as_slice())?;
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
        for row in 0..batch.num_rows() {
            results.push(FigureResult {
                figure_id: col_str(batch, "figure_id", row),
                paper_id: col_str(batch, "paper_id", row),
                figure_type: col_str(batch, "figure_type", row),
                caption: col_str(batch, "caption", row),
                description: col_str(batch, "description", row),
                image_path: col_str_opt(batch, "image_path", row),
                page: col_u16_opt(batch, "page", row),
                referenced_by: vec![],
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
    let data = chunk_from_row(batch, 0);
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
            rows.push((col_u16(batch, "chunk_idx", row), batch, row));
        }
    }
    rows.sort_by_key(|(idx, _, _)| *idx);

    let mut chapter_title = String::new();
    let mut section_title = String::new();
    let mut chunks = Vec::new();
    for (_, batch, row) in rows {
        let data = chunk_from_row(batch, row);
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
                col_u16(batch, "section_idx", row),
                col_u16(batch, "chunk_idx", row),
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
        let data = chunk_from_row(batch, row);
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
        figure_id: col_str(batch, "figure_id", 0),
        paper_id: col_str(batch, "paper_id", 0),
        figure_type: col_str(batch, "figure_type", 0),
        caption: col_str(batch, "caption", 0),
        description: col_str(batch, "description", 0),
        image_path: col_str_opt(batch, "image_path", 0),
        page: col_u16_opt(batch, "page", 0),
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
    let paper_title = col_str(first_batch, "title", 0);
    let authors = col_str_list(first_batch, "authors", 0);
    let year = col_u16_opt(first_batch, "year", 0);
    let venue = col_str_opt(first_batch, "venue", 0);
    let tags = col_str_list(first_batch, "tags", 0);

    // Group by chapter → section
    let mut chapter_map: HashMap<u16, (String, HashMap<u16, (String, usize)>)> = HashMap::new();
    let mut total_chunks = 0usize;

    for batch in &batches {
        for row in 0..batch.num_rows() {
            let ch_idx = col_u16(batch, "chapter_idx", row);
            let ch_title = col_str(batch, "chapter_title", row);
            let sec_idx = col_u16(batch, "section_idx", row);
            let sec_title = col_str(batch, "section_title", row);

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
            let ch_idx = col_u16(batch, "chapter_idx", row);
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
            let pid = col_str(batch, "paper_id", row);
            let entry = paper_map.entry(pid.clone()).or_insert_with(|| PaperSummary {
                paper_id: pid,
                title: col_str(batch, "title", row),
                authors: col_str_list(batch, "authors", row),
                year: col_u16_opt(batch, "year", row),
                venue: col_str_opt(batch, "venue", row),
                tags: col_str_list(batch, "tags", row),
                chunk_count: 0,
                figure_count: 0,
            });
            entry.chunk_count += 1;
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
            let pid = col_str(batch, "paper_id", row);
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
            let pid = col_str(batch, "paper_id", row);
            let tags = col_str_list(batch, "tags", row);
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
