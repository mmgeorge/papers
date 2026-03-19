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

use crate::config::*;
use crate::error::DbError;
use crate::schema::{EMBED_DIM, chunks_schema, exhibits_schema};
use crate::store::DbStore;
use crate::types::IngestStats;
use lancedb::index::Index;

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

pub(crate) struct ChunkRecord {
    pub(crate) chunk_id: String,
    pub(crate) chapter_title: String,
    pub(crate) chapter_idx: u16,
    pub(crate) section_title: String,
    pub(crate) section_idx: u16,
    pub(crate) chunk_idx: u16,
    pub(crate) block_type: String,
    pub(crate) text: String,
    pub(crate) page_start: Option<u16>,
    pub(crate) page_end: Option<u16>,
    pub(crate) exhibit_ids: Vec<String>,
}

pub(crate) struct ExhibitRecord {
    pub(crate) exhibit_id: String,
    pub(crate) exhibit_type: String,
    pub(crate) caption: String,
    pub(crate) description: Option<String>,
    pub(crate) image_path: Option<String>,
    pub(crate) content: Option<String>,
    pub(crate) page: Option<u16>,
    pub(crate) chapter_idx: u16,
    pub(crate) section_idx: u16,
    pub(crate) first_ref_chunk_id: Option<String>,
    pub(crate) ref_count: u16,
}

// ── Token estimation ──────────────────────────────────────────────────────────

fn estimate_tokens(text: &str) -> usize {
    let words = text.split_whitespace().count();
    ((words as f64) * TOKEN_ESTIMATE_MULTIPLIER).ceil() as usize
}

// ── ChunkBuffer ───────────────────────────────────────────────────────────────

struct ChunkBuffer {
    paragraphs: Vec<String>,
    token_count: usize,
    page_start: Option<u16>,
    page_end: Option<u16>,
}

struct FlushedChunk {
    text: String,
    page_start: Option<u16>,
    page_end: Option<u16>,
}

impl ChunkBuffer {
    fn new() -> Self {
        Self {
            paragraphs: Vec::new(),
            token_count: 0,
            page_start: None,
            page_end: None,
        }
    }

    fn push(&mut self, text: String, page: Option<u16>) {
        self.token_count += estimate_tokens(&text);
        self.paragraphs.push(text);
        if let Some(p) = page {
            if self.page_start.is_none() {
                self.page_start = Some(p);
            }
            self.page_end = Some(p);
        }
    }

    fn would_overflow(&self, text: &str) -> bool {
        self.token_count + estimate_tokens(text) > TARGET_CHUNK_TOKENS
    }

    fn is_empty(&self) -> bool {
        self.paragraphs.is_empty()
    }

    fn flush(&mut self) -> Option<FlushedChunk> {
        if self.paragraphs.is_empty() {
            return None;
        }
        let text = self.paragraphs.join("\n\n");
        let chunk = FlushedChunk {
            text,
            page_start: self.page_start,
            page_end: self.page_end,
        };
        self.paragraphs.clear();
        self.token_count = 0;
        self.page_start = None;
        self.page_end = None;
        Some(chunk)
    }

    /// Extract the last N sentences from text to use as overlap in the next buffer.
    fn overlap_tail(text: &str) -> String {
        if OVERLAP_SENTENCES == 0 {
            return String::new();
        }
        // Simple sentence boundary: period/question/exclamation followed by space or end
        let mut boundaries = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        for i in 0..chars.len() {
            if (chars[i] == '.' || chars[i] == '?' || chars[i] == '!')
                && (i + 1 >= chars.len() || chars[i + 1].is_whitespace())
            {
                boundaries.push(i + 1);
            }
        }
        if boundaries.len() < 2 {
            // Not enough sentences to extract overlap
            return String::new();
        }
        // Take the last OVERLAP_SENTENCES sentence boundaries
        let start_boundary = if boundaries.len() >= OVERLAP_SENTENCES + 1 {
            boundaries[boundaries.len() - OVERLAP_SENTENCES - 1]
        } else {
            0
        };
        text[start_boundary..].trim().to_string()
    }
}

// ── HTML processing ───────────────────────────────────────────────────────────

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

/// Strip HTML tags but preserve the text content of `<math>` elements as LaTeX.
///
/// `<math ...>content</math>` is replaced by `$content$` before other tags are
/// stripped, so equations survive as renderable inline LaTeX.
fn strip_html_preserve_math(html: &str) -> String {
    let mut buf = String::with_capacity(html.len());
    let mut remaining = html;

    while let Some(start) = remaining.find("<math") {
        buf.push_str(&remaining[..start]);
        remaining = &remaining[start..];

        if let Some(tag_end) = remaining.find('>') {
            remaining = &remaining[tag_end + 1..];
            if let Some(close) = remaining.find("</math>") {
                buf.push('$');
                buf.push_str(&remaining[..close]);
                buf.push('$');
                remaining = &remaining[close + 7..];
            }
        } else {
            buf.push('<');
            remaining = &remaining[1..];
        }
    }
    buf.push_str(remaining);

    strip_html(&buf)
}

/// Parsed cell from an HTML table row.
struct ParsedCell {
    text: String,
    colspan: usize,
    rowspan: usize,
}

/// Parse a `colspan` or `rowspan` attribute from an opening tag string.
fn parse_span_attr(tag_html: &str, attr: &str) -> usize {
    let needle = format!("{}=\"", attr);
    let start = match tag_html.find(&needle) {
        Some(s) => s + needle.len(),
        None => return 1,
    };
    let rest = &tag_html[start..];
    let end = rest.find('"').unwrap_or(0);
    rest[..end].parse::<usize>().unwrap_or(1).max(1)
}

/// Extract cells from a `<tr>` block, including span attributes.
fn extract_row_cells(tr_html: &str) -> Vec<ParsedCell> {
    let mut cells = Vec::new();
    let mut search = 0;
    let lower = tr_html.to_ascii_lowercase();

    while search < lower.len() {
        let th_pos = lower[search..].find("<th");
        let td_pos = lower[search..].find("<td");

        let cell_start = match (th_pos, td_pos) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let cell_abs = search + cell_start;

        let tag_close = match lower[cell_abs..].find('>') {
            Some(p) => cell_abs + p + 1,
            None => break,
        };

        let opening_tag_lower = &lower[cell_abs..tag_close];
        let colspan = parse_span_attr(opening_tag_lower, "colspan");
        let rowspan = parse_span_attr(opening_tag_lower, "rowspan");

        let close_tag = lower[tag_close..]
            .find("</th>")
            .or_else(|| lower[tag_close..].find("</td>"));

        let cell_content_end = match close_tag {
            Some(p) => tag_close + p,
            None => {
                search = tag_close;
                continue;
            }
        };

        let cell_html = &tr_html[tag_close..cell_content_end];
        let text = strip_html_preserve_math(cell_html);

        cells.push(ParsedCell {
            text,
            colspan,
            rowspan,
        });

        search = cell_content_end + 5;
    }

    cells
}

/// Build a 2D grid from parsed rows, filling colspan/rowspan blocks.
fn build_table_grid(
    html: &str,
) -> (Vec<Vec<Option<String>>>, usize) {
    let lower = html.to_ascii_lowercase();

    let thead_range = lower.find("<thead").and_then(|start| {
        lower.find("</thead>").map(|end| (start, end))
    });

    let mut raw_rows: Vec<(Vec<ParsedCell>, bool)> = Vec::new();
    let mut search_from = 0;
    while let Some(tr_start) = lower[search_from..].find("<tr") {
        let tr_abs = search_from + tr_start;
        let tr_end_tag = lower[tr_abs..].find("</tr>");
        let tr_end = match tr_end_tag {
            Some(e) => tr_abs + e,
            None => {
                search_from = tr_abs + 1;
                continue;
            }
        };
        let tr_html = &html[tr_abs..tr_end];

        let in_thead = thead_range.map_or(false, |(ts, te)| {
            tr_abs >= ts && tr_abs < te
        });

        let cells = extract_row_cells(tr_html);
        if !cells.is_empty() {
            raw_rows.push((cells, in_thead));
        }

        search_from = tr_end + 5;
    }

    if raw_rows.is_empty() {
        return (Vec::new(), 0);
    }

    let num_rows = raw_rows.len();
    let mut grid: Vec<Vec<Option<String>>> = vec![Vec::new(); num_rows];
    let mut thead_row_count = 0usize;

    for (row_idx, (cells, in_thead)) in raw_rows.iter().enumerate() {
        if *in_thead {
            thead_row_count = row_idx + 1;
        }

        let mut col = 0;
        for cell in cells {
            while col < grid[row_idx].len() && grid[row_idx][col].is_some() {
                col += 1;
            }

            for dr in 0..cell.rowspan {
                let r = row_idx + dr;
                if r >= num_rows {
                    break;
                }
                for dc in 0..cell.colspan {
                    let c = col + dc;
                    while grid[r].len() <= c {
                        grid[r].push(None);
                    }
                    grid[r][c] = Some(cell.text.clone());
                }
            }

            col += cell.colspan;
        }
    }

    let max_cols = grid.iter().map(|r| r.len()).max().unwrap_or(0);
    for row in &mut grid {
        row.resize(max_cols, None);
    }

    (grid, thead_row_count)
}

/// Flatten multi-row header into a single row.
fn flatten_header(grid: &[Vec<Option<String>>], thead_rows: usize) -> Vec<String> {
    if thead_rows == 0 || grid.is_empty() {
        return Vec::new();
    }

    let ncols = grid[0].len();
    let mut header = Vec::with_capacity(ncols);

    for col in 0..ncols {
        let mut parts: Vec<String> = Vec::new();
        for row in grid.iter().take(thead_rows) {
            let val = row.get(col).and_then(|v| v.clone()).unwrap_or_default();
            if parts.last().map_or(true, |last| last != &val) {
                parts.push(val);
            }
        }
        let label = parts
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        header.push(label);
    }

    header
}

/// Convert DataLab `<table>` HTML to pipe-delimited markdown.
fn html_table_to_markdown(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    if !lower.contains("<table") {
        return None;
    }

    let html = html
        .replace("<br/>", " ")
        .replace("<br>", " ")
        .replace("<br />", " ")
        .replace("<BR/>", " ")
        .replace("<BR>", " ")
        .replace("<BR />", " ");

    let (grid, thead_row_count) = build_table_grid(&html);
    if grid.is_empty() {
        return None;
    }

    let ncols = grid[0].len();
    let mut lines: Vec<String> = Vec::new();

    if thead_row_count > 1 {
        let header = flatten_header(&grid, thead_row_count);
        let header_line = format!(
            "| {} |",
            header
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        );
        lines.push(header_line);
        let sep = format!(
            "| {} |",
            (0..ncols).map(|_| "---").collect::<Vec<_>>().join(" | ")
        );
        lines.push(sep);

        for row in &grid[thead_row_count..] {
            let cells: Vec<&str> = row
                .iter()
                .map(|c| c.as_deref().unwrap_or(""))
                .collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    } else if thead_row_count == 1 {
        let cells: Vec<&str> = grid[0]
            .iter()
            .map(|c| c.as_deref().unwrap_or(""))
            .collect();
        lines.push(format!("| {} |", cells.join(" | ")));
        let sep = format!(
            "| {} |",
            (0..ncols).map(|_| "---").collect::<Vec<_>>().join(" | ")
        );
        lines.push(sep);

        for row in &grid[1..] {
            let cells: Vec<&str> = row
                .iter()
                .map(|c| c.as_deref().unwrap_or(""))
                .collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    } else {
        for row in &grid {
            let cells: Vec<&str> = row
                .iter()
                .map(|c| c.as_deref().unwrap_or(""))
                .collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    }

    Some(lines.join("\n"))
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

// ── Cross-linking helpers ─────────────────────────────────────────────────────

fn normalize_exhibit_kind(raw: &str) -> String {
    match raw.to_lowercase().as_str() {
        "figure" | "fig" | "figures" | "figs" => "figure".to_string(),
        "table" | "tab" | "tables" | "tabs" => "table".to_string(),
        "algorithm" | "alg" | "algorithms" | "algs"
        | "procedure" | "procedures"
        | "pseudocode" | "pseudocodes"
        | "listing" | "listings"
        | "code" | "codes" => "algorithm".to_string(),
        _ => raw.to_lowercase(),
    }
}

// ── Cache/path helpers ────────────────────────────────────────────────────────

/// Return the extract cache root directory (new pipeline).
fn extract_cache_root() -> Option<PathBuf> {
    papers_core::extract_cache::extract_cache_root()
}

/// Return the DataLab cache root directory (legacy pipeline).
fn datalab_cache_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PAPERS_DATALAB_CACHE_DIR") {
        return Some(PathBuf::from(p));
    }
    dirs::cache_dir().map(|d| d.join("papers").join("datalab"))
}

/// Locate the cache directory for an item_key, preferring extract cache over datalab.
fn resolve_cache_dir(item_key: &str) -> Option<PathBuf> {
    // Prefer new extract cache
    if let Some(root) = extract_cache_root() {
        let dir = root.join(item_key);
        if dir.join("reflow.json").exists() {
            return Some(dir);
        }
    }
    // Fall back to legacy datalab cache
    if let Some(root) = datalab_cache_root() {
        let dir = root.join(item_key);
        if dir.is_dir() {
            return Some(dir);
        }
    }
    None
}

/// Build IngestParams from a cached item_key using meta.json.
///
/// Prefers the new extract cache (`<cache>/papers/extracts/{key}/`) over the
/// legacy DataLab cache (`<cache>/papers/datalab/{key}/`).
pub fn ingest_params_from_cache(item_key: &str) -> Result<IngestParams, DbError> {
    let cache_dir = resolve_cache_dir(item_key).ok_or_else(|| {
        DbError::NotFound(format!("no cache directory found for {item_key}"))
    })?;

    // Try reading meta.json from the resolved cache dir first, then fall back
    // to the legacy datalab helper which looks in the datalab cache.
    let meta = {
        let meta_path = cache_dir.join("meta.json");
        if meta_path.exists() {
            std::fs::read(&meta_path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<papers_core::text::ExtractionMeta>(&bytes).ok())
        } else {
            papers_core::text::read_extraction_meta(item_key)
        }
    };
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
pub fn embed_cache_base() -> PathBuf {
    if let Ok(p) = std::env::var("PAPERS_EMBED_CACHE_DIR") {
        return PathBuf::from(p);
    }
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("papers")
}

fn default_embed_model() -> String {
    papers_core::config::PapersConfig::load()
        .map(|c| c.embedding_model)
        .unwrap_or_else(|_| "embedding-gemma-300m".to_string())
}

fn cache_err(e: crate::embed_cache::EmbedCacheError) -> DbError {
    DbError::Cache(e.to_string())
}

fn unix_timestamp_str() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

// ── Core parsing ──────────────────────────────────────────────────────────────

/// Parse the DataLab Marker JSON for a paper into raw `ChunkRecord` and
/// `ExhibitRecord` lists.
fn parse_paper_blocks(
    params: &IngestParams,
) -> Result<(Vec<ChunkRecord>, Vec<ExhibitRecord>), DbError> {
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
    let algo_header_re = regex::Regex::new(ALGO_HEADER_PATTERN).unwrap();

    let mut chapter_idx: u16 = 0;
    let mut section_idx: u16 = 0;
    let mut current_chapter_title = String::new();
    let mut current_section_title = String::new();
    let mut chunk_idx: u16 = 0;
    let mut exhibit_seq: u16 = 0;

    let mut chunk_records: Vec<ChunkRecord> = Vec::new();
    let mut exhibit_records: Vec<ExhibitRecord> = Vec::new();

    let mut buffer = ChunkBuffer::new();
    let mut in_references = false;

    // Algorithm accumulation state
    let mut algorithm_title: Option<String> = None;
    let mut algorithm_body: Vec<String> = Vec::new();
    let mut algorithm_page: Option<u16> = None;

    // Set of block indices consumed as captions
    let mut consumed_captions: std::collections::HashSet<usize> = std::collections::HashSet::new();

    /// Helper: look for an adjacent `Caption` block at `idx` in `blocks`.
    fn try_consume_caption(
        blocks: &[&Value],
        idx: usize,
        consumed: &mut std::collections::HashSet<usize>,
    ) -> Option<String> {
        if idx >= blocks.len() || consumed.contains(&idx) {
            return None;
        }
        let bt = blocks[idx]
            .get("block_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if bt != "Caption" {
            return None;
        }
        let html = blocks[idx]
            .get("html")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let text = strip_html_preserve_math(html);
        if text.trim().is_empty() {
            return None;
        }
        consumed.insert(idx);
        Some(text)
    }

    // Helper: emit an algorithm exhibit from accumulated state
    let emit_algorithm = |title: String,
                          body: &[String],
                          page: Option<u16>,
                          exhibit_seq: &mut u16,
                          chapter_idx: u16,
                          section_idx: u16,
                          exhibit_records: &mut Vec<ExhibitRecord>,
                          params: &IngestParams| {
        let content = body.join("\n\n");
        if content.trim().is_empty() {
            return;
        }
        *exhibit_seq += 1;
        let exhibit_id = format!("{}/fig{}", params.paper_id, *exhibit_seq);
        exhibit_records.push(ExhibitRecord {
            exhibit_id,
            exhibit_type: "algorithm".to_string(),
            caption: title,
            description: None,
            image_path: None,
            content: Some(content),
            page,
            chapter_idx,
            section_idx,
            first_ref_chunk_id: None,
            ref_count: 0,
        });
    };

    // Helper: create a ChunkRecord from a FlushedChunk
    let emit_chunk = |flushed: FlushedChunk,
                          chunk_idx: &mut u16,
                          chapter_title: &str,
                          chapter_idx: u16,
                          section_title: &str,
                          section_idx: u16,
                          chunk_records: &mut Vec<ChunkRecord>,
                          params: &IngestParams| {
        let chunk_id = format!(
            "{}/ch{}/s{}/p{}",
            params.paper_id, chapter_idx, section_idx, *chunk_idx
        );
        chunk_records.push(ChunkRecord {
            chunk_id,
            chapter_title: chapter_title.to_string(),
            chapter_idx,
            section_title: section_title.to_string(),
            section_idx,
            chunk_idx: *chunk_idx,
            block_type: "text".to_string(),
            text: flushed.text,
            page_start: flushed.page_start,
            page_end: flushed.page_end,
            exhibit_ids: vec![],
        });
        *chunk_idx += 1;
    };

    // Helper: smart merge or emit at section boundary / end-of-doc
    let smart_merge_or_emit = |flushed: Option<FlushedChunk>,
                               chunk_records: &mut Vec<ChunkRecord>,
                               chunk_idx: &mut u16,
                               chapter_title: &str,
                               chapter_idx: u16,
                               section_title: &str,
                               section_idx: u16,
                               params: &IngestParams| {
        if let Some(f) = flushed {
            let f_tokens = estimate_tokens(&f.text);
            if f_tokens < MIN_CHUNK_TOKENS && !chunk_records.is_empty() {
                // Try to merge into previous chunk (only within same chapter+section)
                let prev = chunk_records.last().unwrap();
                if prev.chapter_idx == chapter_idx && prev.section_idx == section_idx {
                    let prev_tokens = estimate_tokens(&prev.text);
                    if prev_tokens + f_tokens <= MAX_CHUNK_TOKENS {
                        let prev = chunk_records.last_mut().unwrap();
                        prev.text.push_str("\n\n");
                        prev.text.push_str(&f.text);
                        if let Some(p) = f.page_end {
                            prev.page_end = Some(p);
                        }
                        return;
                    }
                }
            }
            // Emit standalone
            emit_chunk(
                f,
                chunk_idx,
                chapter_title,
                chapter_idx,
                section_title,
                section_idx,
                chunk_records,
                params,
            );
        }
    };

    let mut i = 0;
    while i < flat_blocks.len() {
        if consumed_captions.contains(&i) {
            i += 1;
            continue;
        }

        let block = flat_blocks[i];
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
            "Page" | "PageHeader" | "PageFooter" | "TableOfContents" => {
                i += 1;
                continue;
            }

            "Caption" => {
                i += 1;
                continue;
            }

            "SectionHeader" => {
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
                    6
                };

                // Any SectionHeader exits algorithm mode
                if let Some(title) = algorithm_title.take() {
                    emit_algorithm(
                        title,
                        &algorithm_body,
                        algorithm_page,
                        &mut exhibit_seq,
                        chapter_idx,
                        section_idx,
                        &mut exhibit_records,
                        params,
                    );
                    algorithm_body.clear();
                }

                match h_level {
                    1 => {
                        // Paper title — skip
                    }
                    2 => {
                        // Check for references section
                        let title_text = heading_map.get(block_id).cloned().unwrap_or_default();
                        let lower = title_text.to_lowercase();
                        if REFERENCES_TITLES.iter().any(|t| lower.contains(t)) {
                            // Flush buffer at references boundary
                            smart_merge_or_emit(
                                buffer.flush(),
                                &mut chunk_records,
                                &mut chunk_idx,
                                &current_chapter_title,
                                chapter_idx,
                                &current_section_title,
                                section_idx,
                                params,
                            );
                            in_references = true;
                            i += 1;
                            continue;
                        }

                        // Section boundary: flush buffer with smart merge
                        smart_merge_or_emit(
                            buffer.flush(),
                            &mut chunk_records,
                            &mut chunk_idx,
                            &current_chapter_title,
                            chapter_idx,
                            &current_section_title,
                            section_idx,
                            params,
                        );

                        chapter_idx += 1;
                        section_idx = 0;
                        chunk_idx = 0;
                        current_chapter_title =
                            heading_map.get(block_id).cloned().unwrap_or_default();
                        current_section_title = String::new();
                    }
                    3 | 4 => {
                        // Section boundary: flush buffer with smart merge
                        smart_merge_or_emit(
                            buffer.flush(),
                            &mut chunk_records,
                            &mut chunk_idx,
                            &current_chapter_title,
                            chapter_idx,
                            &current_section_title,
                            section_idx,
                            params,
                        );

                        section_idx += 1;
                        chunk_idx = 0;
                        current_section_title =
                            heading_map.get(block_id).cloned().unwrap_or_default();
                    }
                    5 | 6 => {
                        // Check for algorithm-like headers
                        let header_text = heading_map.get(block_id).cloned().unwrap_or_default();
                        if algo_header_re.is_match(&header_text) {
                            // Flush current text buffer first
                            smart_merge_or_emit(
                                buffer.flush(),
                                &mut chunk_records,
                                &mut chunk_idx,
                                &current_chapter_title,
                                chapter_idx,
                                &current_section_title,
                                section_idx,
                                params,
                            );
                            // Enter algorithm accumulation mode
                            algorithm_title = Some(header_text);
                            algorithm_body = Vec::new();
                            algorithm_page = page_num;
                        }
                        // else: skip (paper title, footnotes, etc.)
                    }
                    _ => {}
                }
            }

            "Text" | "ListGroup" | "Equation" => {
                if in_references {
                    i += 1;
                    continue;
                }

                let html = block.get("html").and_then(|v| v.as_str()).unwrap_or("");
                let text = strip_html_preserve_math(html);
                if text.trim().is_empty() {
                    i += 1;
                    continue;
                }

                // Algorithm accumulation mode
                if algorithm_title.is_some() {
                    algorithm_body.push(text);
                    i += 1;
                    continue;
                }

                // Buffer-based accumulation
                if !buffer.is_empty() && buffer.would_overflow(&text) {
                    // Token-limit flush with overlap
                    if let Some(flushed) = buffer.flush() {
                        let overlap = ChunkBuffer::overlap_tail(&flushed.text);
                        emit_chunk(
                            flushed,
                            &mut chunk_idx,
                            &current_chapter_title,
                            chapter_idx,
                            &current_section_title,
                            section_idx,
                            &mut chunk_records,
                            params,
                        );
                        // Carry overlap into new buffer
                        if !overlap.is_empty() {
                            buffer.push(overlap, page_num);
                        }
                    }
                }
                buffer.push(text, page_num);
            }

            "Figure" | "Table" | "Picture" => {
                // Exit algorithm mode if active
                if let Some(title) = algorithm_title.take() {
                    emit_algorithm(
                        title,
                        &algorithm_body,
                        algorithm_page,
                        &mut exhibit_seq,
                        chapter_idx,
                        section_idx,
                        &mut exhibit_records,
                        params,
                    );
                    algorithm_body.clear();
                }

                let html = block.get("html").and_then(|v| v.as_str()).unwrap_or("");
                let alt_text = extract_img_alt(html).unwrap_or_default();
                let src = extract_img_src(html);

                let adjacent_caption =
                    try_consume_caption(&flat_blocks, i + 1, &mut consumed_captions)
                        .or_else(|| {
                            if i > 0 {
                                try_consume_caption(
                                    &flat_blocks,
                                    i - 1,
                                    &mut consumed_captions,
                                )
                            } else {
                                None
                            }
                        });

                if block_type == "Picture" && adjacent_caption.is_none() {
                    i += 1;
                    continue;
                }

                let caption = adjacent_caption
                    .clone()
                    .unwrap_or_else(|| alt_text.clone());
                let description = if alt_text.is_empty() { None } else { Some(alt_text) };

                let content = if block_type == "Table" {
                    html_table_to_markdown(html)
                } else {
                    None
                };

                let image_path = src.map(|s| {
                    params
                        .cache_dir
                        .join("images")
                        .join(&s)
                        .to_string_lossy()
                        .into_owned()
                });
                let exhibit_type = if block_type == "Table" { "table" } else { "figure" };
                exhibit_seq += 1;
                let exhibit_id = format!("{}/fig{}", params.paper_id, exhibit_seq);
                exhibit_records.push(ExhibitRecord {
                    exhibit_id,
                    exhibit_type: exhibit_type.to_string(),
                    caption,
                    description,
                    image_path,
                    content,
                    page: page_num,
                    chapter_idx,
                    section_idx,
                    first_ref_chunk_id: None,
                    ref_count: 0,
                });
            }

            _ => {}
        }

        i += 1;
    }

    // Flush any remaining algorithm
    if let Some(title) = algorithm_title.take() {
        emit_algorithm(
            title,
            &algorithm_body,
            algorithm_page,
            &mut exhibit_seq,
            chapter_idx,
            section_idx,
            &mut exhibit_records,
            params,
        );
    }

    // End-of-doc: smart merge remaining buffer
    smart_merge_or_emit(
        buffer.flush(),
        &mut chunk_records,
        &mut chunk_idx,
        &current_chapter_title,
        chapter_idx,
        &current_section_title,
        section_idx,
        params,
    );

    // ── Post-process: populate exhibit_ids on chunks ────────────────────────
    if !exhibit_records.is_empty() {
        // Build caption → exhibit_id map
        let caption_re = regex::Regex::new(
            r"(?i)(Figure|Fig|Table|Tab|Algorithm|Alg|Procedure|Pseudocode|Listing|Code)\s*\.?\s+(\d+)"
        ).unwrap();

        let mut exhibit_number_to_id: HashMap<(String, u32), String> = HashMap::new();
        for er in &exhibit_records {
            if let Some(caps) = caption_re.captures(&er.caption) {
                let kind = normalize_exhibit_kind(&caps[1]);
                if let Ok(n) = caps[2].parse::<u32>() {
                    exhibit_number_to_id.insert((kind, n), er.exhibit_id.clone());
                }
            }
        }

        // Text reference regexes
        let ref_re = regex::Regex::new(r"(?i)(?:Figures?|Figs?)\s*\.?\s+(\d+)").unwrap();
        let tbl_ref_re = regex::Regex::new(r"(?i)(?:Tables?|Tabs?)\s*\.?\s+(\d+)").unwrap();
        let algo_ref_re = regex::Regex::new(
            r"(?i)(?:Algorithms?|Algs?|Procedures?|Pseudocodes?|Listings?|Codes?)\s*\.?\s+(\d+)"
        ).unwrap();

        // Track exhibit references: exhibit_id → (first_ref_chunk_id, ref_count)
        let mut exhibit_ref_map: HashMap<String, (Option<String>, u16)> = HashMap::new();

        for chunk in &mut chunk_records {
            let mut ids: Vec<String> = Vec::new();
            // Figure refs
            for caps in ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(eid) = exhibit_number_to_id.get(&("figure".to_string(), n)) {
                        if !ids.contains(eid) {
                            ids.push(eid.clone());
                        }
                    }
                }
            }
            // Table refs
            for caps in tbl_ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(eid) = exhibit_number_to_id.get(&("table".to_string(), n)) {
                        if !ids.contains(eid) {
                            ids.push(eid.clone());
                        }
                    }
                }
            }
            // Algorithm refs
            for caps in algo_ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(eid) = exhibit_number_to_id.get(&("algorithm".to_string(), n)) {
                        if !ids.contains(eid) {
                            ids.push(eid.clone());
                        }
                    }
                }
            }

            // Update exhibit reference tracking
            for eid in &ids {
                let entry = exhibit_ref_map
                    .entry(eid.clone())
                    .or_insert((None, 0));
                if entry.0.is_none() {
                    entry.0 = Some(chunk.chunk_id.clone());
                }
                entry.1 += 1;
            }

            chunk.exhibit_ids = ids;
        }

        // Write ref tracking back to exhibits
        for er in &mut exhibit_records {
            if let Some((first_ref, count)) = exhibit_ref_map.get(&er.exhibit_id) {
                er.first_ref_chunk_id = first_ref.clone();
                er.ref_count = *count;
            }
        }
    }

    eprintln!(
        "  [{}] extracted {} chunks, {} exhibits ({} section headers)",
        params.item_key,
        chunk_records.len(),
        exhibit_records.len(),
        heading_map.len()
    );

    Ok((chunk_records, exhibit_records))
}

/// Compute embeddings for a paper's chunks and write them to the embedding cache.
pub async fn cache_paper_embeddings(
    store: &DbStore,
    params: &IngestParams,
    model: &str,
    force: bool,
) -> Result<usize, DbError> {
    let cache = crate::embed_cache::EmbedCache::new(embed_cache_base());

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

    let (chunk_records, _exhibit_records) = {
        let reflow_path = params.cache_dir.join("reflow.json");
        if reflow_path.exists() {
            let json_bytes = std::fs::read(&reflow_path).map_err(|e| {
                DbError::Ingest(format!("failed to read {}: {e}", reflow_path.display()))
            })?;
            let doc: ReflowDocument = serde_json::from_slice(&json_bytes).map_err(|e| {
                DbError::Ingest(format!("failed to parse reflow.json: {e}"))
            })?;
            parse_reflow_document(params, &doc)?
        } else {
            parse_paper_blocks(params)?
        }
    };
    let n = chunk_records.len();

    let embeddings = if chunk_records.is_empty() {
        vec![]
    } else {
        eprintln!("  [{}] embedding {} chunks (model={})...", params.item_key, n, model);
        let t = std::time::Instant::now();
        // Prepend title + section context for embedding
        let texts: Vec<String> = chunk_records
            .iter()
            .map(|c| embedding_text(params, c))
            .collect();
        let result = store.embed_documents(texts).await?;
        eprintln!("  [{}] chunk embeddings done ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
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
            page_start: c.page_start.map(|p| p as u32),
            page_end: c.page_end.map(|p| p as u32),
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

/// Build embedding text with title + section context prepended.
fn embedding_text(params: &IngestParams, c: &ChunkRecord) -> String {
    let mut s = String::new();
    if !params.title.is_empty() {
        s.push_str(&params.title);
        s.push_str(" — ");
    }
    if !c.chapter_title.is_empty() {
        s.push_str(&c.chapter_title);
        if !c.section_title.is_empty() {
            s.push_str(" — ");
            s.push_str(&c.section_title);
        }
    }
    if !s.is_empty() {
        s.push_str("\n\n");
    }
    s.push_str(&c.text);
    s
}

// ── Reflow-based chunking ─────────────────────────────────────────────────────

use papers_extract::types::{ReflowDocument, ReflowNode};

/// Mutable state for the recursive reflow tree walk.
struct ChunkingState<'a> {
    params: &'a IngestParams,
    chapter_idx: u16,
    section_idx: u16,
    chunk_idx: u16,
    chapter_title: String,
    section_title: String,
    buffer: ChunkBuffer,
    chunk_records: Vec<ChunkRecord>,
    exhibit_records: Vec<ExhibitRecord>,
    exhibit_seq: u16,
    in_references: bool,
}

impl<'a> ChunkingState<'a> {
    fn new(params: &'a IngestParams) -> Self {
        Self {
            params,
            chapter_idx: 0,
            section_idx: 0,
            chunk_idx: 0,
            chapter_title: String::new(),
            section_title: String::new(),
            buffer: ChunkBuffer::new(),
            chunk_records: Vec::new(),
            exhibit_records: Vec::new(),
            exhibit_seq: 0,
            in_references: false,
        }
    }

    /// Emit a chunk from a flushed buffer.
    fn emit_chunk(&mut self, flushed: FlushedChunk) {
        let chunk_id = format!(
            "{}/ch{}/s{}/p{}",
            self.params.paper_id, self.chapter_idx, self.section_idx, self.chunk_idx,
        );
        self.chunk_records.push(ChunkRecord {
            chunk_id,
            chapter_title: self.chapter_title.clone(),
            chapter_idx: self.chapter_idx,
            section_title: self.section_title.clone(),
            section_idx: self.section_idx,
            chunk_idx: self.chunk_idx,
            block_type: "text".to_string(),
            text: flushed.text,
            page_start: flushed.page_start,
            page_end: flushed.page_end,
            exhibit_ids: vec![],
        });
        self.chunk_idx += 1;
    }

    /// Flush at a section/chapter boundary with smart-merge for undersized chunks.
    fn flush_section_boundary(&mut self) {
        if let Some(flushed) = self.buffer.flush() {
            self.smart_merge_or_emit(flushed);
        }
    }

    /// Flush mid-section due to token overflow, carrying overlap into the new buffer.
    fn flush_mid_section(&mut self) {
        if let Some(flushed) = self.buffer.flush() {
            let overlap = ChunkBuffer::overlap_tail(&flushed.text);
            self.emit_chunk(flushed);
            if !overlap.is_empty() {
                self.buffer.push(overlap, None);
            }
        }
    }

    /// Smart merge: if chunk < MIN_CHUNK_TOKENS and previous chunk is same section,
    /// merge if combined <= MAX_CHUNK_TOKENS. Otherwise emit standalone.
    fn smart_merge_or_emit(&mut self, flushed: FlushedChunk) {
        let f_tokens = estimate_tokens(&flushed.text);
        if f_tokens < MIN_CHUNK_TOKENS && !self.chunk_records.is_empty() {
            let prev = self.chunk_records.last().unwrap();
            if prev.chapter_idx == self.chapter_idx && prev.section_idx == self.section_idx {
                let prev_tokens = estimate_tokens(&prev.text);
                if prev_tokens + f_tokens <= MAX_CHUNK_TOKENS {
                    let prev = self.chunk_records.last_mut().unwrap();
                    prev.text.push_str("\n\n");
                    prev.text.push_str(&flushed.text);
                    if let Some(p) = flushed.page_end {
                        prev.page_end = Some(p);
                    }
                    return;
                }
            }
        }
        self.emit_chunk(flushed);
    }

    /// Post-process: link exhibit references in chunk text to ExhibitRecords.
    fn link_exhibits(&mut self) {
        if self.exhibit_records.is_empty() {
            return;
        }

        let caption_re = regex::Regex::new(
            r"(?i)(Figure|Fig|Table|Tab|Algorithm|Alg|Procedure|Pseudocode|Listing|Code)\s*\.?\s+(\d+)"
        ).unwrap();

        let mut exhibit_number_to_id: HashMap<(String, u32), String> = HashMap::new();
        for er in &self.exhibit_records {
            // Try caption first
            if let Some(caps) = caption_re.captures(&er.caption) {
                let kind = normalize_exhibit_kind(&caps[1]);
                if let Ok(n) = caps[2].parse::<u32>() {
                    exhibit_number_to_id.insert((kind, n), er.exhibit_id.clone());
                    continue;
                }
            }
            // For algorithms: try extracting number from content first line
            if er.exhibit_type == "algorithm" {
                if let Some(content) = &er.content {
                    let first_line = content.lines().next().unwrap_or("");
                    if let Some(caps) = caption_re.captures(first_line) {
                        let kind = normalize_exhibit_kind(&caps[1]);
                        if let Ok(n) = caps[2].parse::<u32>() {
                            exhibit_number_to_id.insert((kind, n), er.exhibit_id.clone());
                        }
                    }
                }
            }
        }

        let ref_re = regex::Regex::new(r"(?i)(?:Figures?|Figs?)\s*\.?\s+(\d+)").unwrap();
        let tbl_ref_re = regex::Regex::new(r"(?i)(?:Tables?|Tabs?)\s*\.?\s+(\d+)").unwrap();
        let algo_ref_re = regex::Regex::new(
            r"(?i)(?:Algorithms?|Algs?|Procedures?|Pseudocodes?|Listings?|Codes?)\s*\.?\s+(\d+)"
        ).unwrap();

        let mut exhibit_ref_map: HashMap<String, (Option<String>, u16)> = HashMap::new();

        for chunk in &mut self.chunk_records {
            let mut ids: Vec<String> = Vec::new();
            for caps in ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(eid) = exhibit_number_to_id.get(&("figure".to_string(), n)) {
                        if !ids.contains(eid) {
                            ids.push(eid.clone());
                        }
                    }
                }
            }
            for caps in tbl_ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(eid) = exhibit_number_to_id.get(&("table".to_string(), n)) {
                        if !ids.contains(eid) {
                            ids.push(eid.clone());
                        }
                    }
                }
            }
            for caps in algo_ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(eid) = exhibit_number_to_id.get(&("algorithm".to_string(), n)) {
                        if !ids.contains(eid) {
                            ids.push(eid.clone());
                        }
                    }
                }
            }

            for eid in &ids {
                let entry = exhibit_ref_map.entry(eid.clone()).or_insert((None, 0));
                if entry.0.is_none() {
                    entry.0 = Some(chunk.chunk_id.clone());
                }
                entry.1 += 1;
            }
            chunk.exhibit_ids = ids;
        }

        for er in &mut self.exhibit_records {
            if let Some((first_ref, count)) = exhibit_ref_map.get(&er.exhibit_id) {
                er.first_ref_chunk_id = first_ref.clone();
                er.ref_count = *count;
            }
        }
    }
}

/// Process a single ReflowNode, recursing into Heading children.
fn process_reflow_node(node: &ReflowNode, state: &mut ChunkingState) {
    if state.in_references {
        return;
    }

    match node {
        ReflowNode::Heading { depth, title, children } => {
            let lower = title.to_lowercase();
            if REFERENCES_TITLES.iter().any(|t| lower.contains(t)) {
                state.flush_section_boundary();
                state.in_references = true;
                return;
            }

            match *depth {
                0 => {
                    // Paper title level — skip (title comes from params)
                }
                1 => {
                    // Chapter boundary
                    state.flush_section_boundary();
                    state.chapter_idx += 1;
                    state.section_idx = 0;
                    state.chunk_idx = 0;
                    state.chapter_title = title.clone();
                    state.section_title = String::new();
                }
                2 => {
                    // Section boundary
                    state.flush_section_boundary();
                    state.section_idx += 1;
                    state.chunk_idx = 0;
                    state.section_title = title.clone();
                }
                _ => {
                    // Subsection (depth 3+): inject as bold paragraph separator
                    if !state.buffer.is_empty()
                        && state.buffer.token_count > TARGET_CHUNK_TOKENS / 2
                    {
                        state.flush_mid_section();
                    }
                    state.buffer.push(format!("**{title}**"), None);
                }
            }

            for child in children {
                process_reflow_node(child, state);
            }
        }

        ReflowNode::Text { content, footnotes } => {
            let text = content.trim();
            if text.is_empty() {
                return;
            }
            if !state.buffer.is_empty() && state.buffer.would_overflow(text) {
                state.flush_mid_section();
            }
            state.buffer.push(text.to_string(), None);

            // Append associated footnotes inline
            for footnote in footnotes {
                let fn_text = footnote.trim();
                if !fn_text.is_empty() {
                    state.buffer.push(fn_text.to_string(), None);
                }
            }
        }

        ReflowNode::Formula { content, path } => {
            let formula_text = if let Some(latex) = content {
                latex.clone()
            } else if path.is_some() {
                "[display formula]".to_string()
            } else {
                return;
            };

            if !state.buffer.is_empty()
                && state.buffer.token_count + estimate_tokens(&formula_text) > MAX_CHUNK_TOKENS
            {
                state.flush_mid_section();
            }
            state.buffer.push(formula_text, None);
        }

        ReflowNode::Figure { path, caption } => {
            state.exhibit_seq += 1;
            let exhibit_id = format!("{}/fig{}", state.params.paper_id, state.exhibit_seq);
            let image_path = if path.is_empty() {
                None
            } else {
                Some(state.params.cache_dir.join(path).to_string_lossy().into_owned())
            };
            state.exhibit_records.push(ExhibitRecord {
                exhibit_id,
                exhibit_type: "figure".to_string(),
                caption: caption.as_deref().unwrap_or("").to_string(),
                description: None,
                image_path,
                content: None,
                page: None,
                chapter_idx: state.chapter_idx,
                section_idx: state.section_idx,
                first_ref_chunk_id: None,
                ref_count: 0,
            });
        }

        ReflowNode::FigureGroup { path, caption } => {
            state.exhibit_seq += 1;
            let exhibit_id = format!("{}/fig{}", state.params.paper_id, state.exhibit_seq);
            let image_path = if path.is_empty() {
                None
            } else {
                Some(state.params.cache_dir.join(path).to_string_lossy().into_owned())
            };
            state.exhibit_records.push(ExhibitRecord {
                exhibit_id,
                exhibit_type: "figure".to_string(),
                caption: caption.as_deref().unwrap_or("").to_string(),
                description: None,
                image_path,
                content: None,
                page: None,
                chapter_idx: state.chapter_idx,
                section_idx: state.section_idx,
                first_ref_chunk_id: None,
                ref_count: 0,
            });
        }

        ReflowNode::Table { content, caption } => {
            state.exhibit_seq += 1;
            let exhibit_id = format!("{}/fig{}", state.params.paper_id, state.exhibit_seq);
            state.exhibit_records.push(ExhibitRecord {
                exhibit_id,
                exhibit_type: "table".to_string(),
                caption: caption.as_deref().unwrap_or("").to_string(),
                description: None,
                image_path: None,
                content: Some(content.clone()),
                page: None,
                chapter_idx: state.chapter_idx,
                section_idx: state.section_idx,
                first_ref_chunk_id: None,
                ref_count: 0,
            });
        }

        ReflowNode::Algorithm { content } => {
            if content.trim().is_empty() {
                return;
            }
            state.exhibit_seq += 1;
            let exhibit_id = format!("{}/fig{}", state.params.paper_id, state.exhibit_seq);
            state.exhibit_records.push(ExhibitRecord {
                exhibit_id,
                exhibit_type: "algorithm".to_string(),
                caption: String::new(),
                description: None,
                image_path: None,
                content: Some(content.clone()),
                page: None,
                chapter_idx: state.chapter_idx,
                section_idx: state.section_idx,
                first_ref_chunk_id: None,
                ref_count: 0,
            });
        }

        ReflowNode::List { list_type, items } => {
            let list_text: String = items
                .iter()
                .enumerate()
                .map(|(i, item)| {
                    if list_type == "numbered" {
                        format!("{}. {}", i + 1, item)
                    } else {
                        format!("- {}", item)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            if list_text.is_empty() {
                return;
            }
            if !state.buffer.is_empty() && state.buffer.would_overflow(&list_text) {
                state.flush_mid_section();
            }
            state.buffer.push(list_text, None);
        }

        ReflowNode::CodeBlock { content, language } => {
            if content.trim().is_empty() {
                return;
            }
            let lang = language.as_deref().unwrap_or("");
            let code_text = format!("```{lang}\n{}\n```", content.trim());
            if !state.buffer.is_empty() && state.buffer.would_overflow(&code_text) {
                state.flush_mid_section();
            }
            state.buffer.push(code_text, None);
        }

        ReflowNode::FormattedText { content } => {
            let text = content.trim();
            if text.is_empty() {
                return;
            }
            if !state.buffer.is_empty() && state.buffer.would_overflow(text) {
                state.flush_mid_section();
            }
            state.buffer.push(text.to_string(), None);
        }

        ReflowNode::Footnote { marker, content } => {
            let fn_text = format!("[{marker}] {content}");
            state.buffer.push(fn_text, None);
        }

        ReflowNode::References { .. } => {
            state.flush_section_boundary();
            state.in_references = true;
        }

        ReflowNode::FootnoteBlock { .. } => {
            // Skip — footnotes already distributed to Text nodes
        }

        ReflowNode::FrontMatter { .. } => {
            // Skip — copyright, dedications, etc.
        }

        ReflowNode::Toc { .. } => {
            // Skip — TOC listing is structural, not content
        }
    }
}

/// Parse a ReflowDocument into chunks and exhibits for database ingestion.
pub(crate) fn parse_reflow_document(
    params: &IngestParams,
    doc: &ReflowDocument,
) -> Result<(Vec<ChunkRecord>, Vec<ExhibitRecord>), DbError> {
    let mut state = ChunkingState::new(params);

    for node in &doc.children {
        process_reflow_node(node, &mut state);
    }

    // Flush remaining buffer
    if let Some(flushed) = state.buffer.flush() {
        state.smart_merge_or_emit(flushed);
    }

    // Post-process: link exhibit references
    state.link_exhibits();

    eprintln!(
        "  [{}] chunked {} chunks, {} exhibits",
        params.item_key,
        state.chunk_records.len(),
        state.exhibit_records.len(),
    );

    Ok((state.chunk_records, state.exhibit_records))
}

/// Ingest a paper into LanceDB.
///
/// Tries the reflow pipeline first (`reflow.json` in cache_dir). Falls back to
/// the legacy DataLab Marker pipeline (`{item_key}.json`) if no reflow is found.
pub async fn ingest_paper(store: &DbStore, params: IngestParams) -> Result<IngestStats, DbError> {
    let t_total = std::time::Instant::now();
    let (chunk_records, exhibit_records) = {
        let reflow_path = params.cache_dir.join("reflow.json");
        if reflow_path.exists() {
            let json_bytes = std::fs::read(&reflow_path).map_err(|e| {
                DbError::Ingest(format!("failed to read {}: {e}", reflow_path.display()))
            })?;
            let doc: ReflowDocument = serde_json::from_slice(&json_bytes).map_err(|e| {
                DbError::Ingest(format!("failed to parse reflow.json: {e}"))
            })?;
            parse_reflow_document(&params, &doc)?
        } else {
            parse_paper_blocks(&params)?
        }
    };

    let chunks_added = chunk_records.len();
    let exhibits_added = exhibit_records.len();

    // ── Delete existing records for this paper ──────────────────────────────
    let paper_id_esc = params.paper_id.replace('\'', "''");
    let delete_filter = format!("paper_id = '{paper_id_esc}'");
    if let Ok(chunks_table) = store.chunks_table().await {
        let _ = chunks_table.delete(&delete_filter).await;
    }
    if let Ok(exhibits_table) = store.exhibits_table().await {
        let _ = exhibits_table.delete(&delete_filter).await;
    }

    // ── Embed chunk texts (with cache) ─────────────────────────────────────
    let model = default_embed_model();
    let embed_cache = crate::embed_cache::EmbedCache::new(embed_cache_base());

    let embeddings = if chunk_records.is_empty() {
        vec![]
    } else {
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
            Some(embs) if embs.len() == chunk_records.len() => {
                eprintln!(
                    "  [{}] embed cache hit ({} chunks)",
                    params.item_key,
                    embs.len()
                );
                embs
            }
            Some(embs) => {
                eprintln!(
                    "  [{}] embed cache stale ({} cached vs {} chunks), re-embedding...",
                    params.item_key,
                    embs.len(),
                    chunk_records.len()
                );
                let t = std::time::Instant::now();
                let texts: Vec<String> = chunk_records
                    .iter()
                    .map(|c| embedding_text(&params, c))
                    .collect();
                let result = store.embed_documents(texts).await?;
                eprintln!("  [{}] chunk embeddings done ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
                // Update cache with new embeddings
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
                        page_start: c.page_start.map(|p| p as u32),
                        page_end: c.page_end.map(|p| p as u32),
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
                    true, // force overwrite stale cache
                ) {
                    eprintln!("  [{}] warning: failed to write embed cache: {e}", params.item_key);
                }
                result
            }
            None => {
                eprintln!(
                    "  [{}] embedding {} chunks...",
                    params.item_key,
                    chunk_records.len()
                );
                let t = std::time::Instant::now();
                let texts: Vec<String> = chunk_records
                    .iter()
                    .map(|c| embedding_text(&params, c))
                    .collect();
                let result = store.embed_documents(texts).await?;
                eprintln!("  [{}] chunk embeddings done ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());

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
                        page_start: c.page_start.map(|p| p as u32),
                        page_end: c.page_end.map(|p| p as u32),
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

    // ── Embed exhibit captions ─────────────────────────────────────────────
    let exhibit_texts: Vec<String> = exhibit_records
        .iter()
        .map(|f| {
            let mut parts = Vec::new();
            if !f.caption.is_empty() {
                parts.push(f.caption.clone());
            }
            if let Some(desc) = &f.description {
                parts.push(desc.clone());
            }
            if let Some(content) = &f.content {
                if f.exhibit_type == "algorithm" {
                    // Include algorithm content for embedding
                    parts.push(content.clone());
                }
            }
            parts.join("\n")
        })
        .collect();
    let exhibit_embeddings = if exhibit_texts.is_empty() {
        vec![]
    } else {
        eprintln!(
            "  [{}] embedding {} exhibit captions...",
            params.item_key,
            exhibit_texts.len()
        );
        let t = std::time::Instant::now();
        let result = store.embed_documents(exhibit_texts).await?;
        eprintln!("  [{}] exhibit embeddings done ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
        result
    };

    // ── Insert chunks ───────────────────────────────────────────────────────
    if chunks_added > 0 {
        eprintln!("  [{}] inserting {} chunks...", params.item_key, chunks_added);
        let t = std::time::Instant::now();
        let batch = build_chunks_batch(&params, &chunk_records, &embeddings)?;
        let schema = chunks_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let table = store.chunks_table().await?;
        table
            .add(Box::new(reader))
            .execute()
            .await?;
        if let Err(e) = table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await
        {
            let _ = e;
            eprintln!("  [{}] chunks index rebuild skipped", params.item_key);
        }
        eprintln!("  [{}] chunks inserted ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
    }

    // ── Insert exhibits ────────────────────────────────────────────────────
    if exhibits_added > 0 {
        let mut type_counts: std::collections::BTreeMap<&str, usize> =
            std::collections::BTreeMap::new();
        for r in &exhibit_records {
            *type_counts.entry(r.exhibit_type.as_str()).or_insert(0) += 1;
        }
        let type_summary = type_counts
            .iter()
            .map(|(t, n)| if *n == 1 { format!("1 {t}") } else { format!("{n} {t}s") })
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!(
            "  [{}] inserting {} exhibits ({})...",
            params.item_key,
            exhibits_added,
            type_summary
        );
        let t = std::time::Instant::now();
        let batch = build_exhibits_batch(&params, &exhibit_records, &exhibit_embeddings)?;
        let schema = exhibits_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let table = store.exhibits_table().await?;
        table
            .add(Box::new(reader))
            .execute()
            .await?;
        if let Err(e) = table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await
        {
            let _ = e;
            eprintln!("  [{}] exhibits index rebuild skipped", params.item_key);
        }
        eprintln!("  [{}] exhibits inserted ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
    }

    eprintln!("  [{}] done (total {:.1}s)", params.item_key, t_total.elapsed().as_secs_f64());
    Ok(IngestStats {
        chunks_added,
        exhibits_added,
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
) -> Result<RecordBatch, DbError> {
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
    let block_types: Vec<&str> = records.iter().map(|r| r.block_type.as_str()).collect();
    let texts: Vec<&str> = records.iter().map(|r| r.text.as_str()).collect();
    let page_starts: Vec<Option<u16>> = records.iter().map(|r| r.page_start).collect();
    let page_ends: Vec<Option<u16>> = records.iter().map(|r| r.page_end).collect();
    let titles: Vec<&str> = vec![params.title.as_str(); n];
    let authors_list: Vec<Vec<String>> = vec![params.authors.clone(); n];
    let years: Vec<Option<u16>> = vec![params.year; n];
    let venues: Vec<Option<&str>> = vec![params.venue.as_deref(); n];
    let tags_list: Vec<Vec<String>> = vec![params.tags.clone(); n];
    let exhibit_ids_list: Vec<Vec<String>> = records.iter().map(|r| r.exhibit_ids.clone()).collect();

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
            Arc::new(StringArray::from(block_types)),
            Arc::new(StringArray::from(texts)),
            Arc::new(UInt16Array::from(page_starts)),
            Arc::new(UInt16Array::from(page_ends)),
            Arc::new(StringArray::from(titles)),
            Arc::new(build_string_list_array(&authors_list)),
            Arc::new(UInt16Array::from(years)),
            Arc::new(StringArray::from(venues)),
            Arc::new(build_string_list_array(&tags_list)),
            Arc::new(build_string_list_array(&exhibit_ids_list)),
        ],
    )
    .map_err(|e: ArrowError| DbError::Arrow(e.to_string()))?;

    Ok(batch)
}

fn build_exhibits_batch(
    params: &IngestParams,
    records: &[ExhibitRecord],
    embeddings: &[Vec<f32>],
) -> Result<RecordBatch, DbError> {
    let n = records.len();
    let schema = exhibits_schema();

    let exhibit_ids: Vec<&str> = records.iter().map(|r| r.exhibit_id.as_str()).collect();
    let paper_ids: Vec<&str> = vec![params.paper_id.as_str(); n];
    let vectors = build_vector_array(embeddings);
    let exhibit_types: Vec<&str> = records.iter().map(|r| r.exhibit_type.as_str()).collect();
    let captions: Vec<&str> = records.iter().map(|r| r.caption.as_str()).collect();
    let descriptions: Vec<Option<&str>> = records.iter().map(|r| r.description.as_deref()).collect();
    let image_paths: Vec<Option<&str>> = records
        .iter()
        .map(|r| r.image_path.as_deref())
        .collect();
    let contents: Vec<Option<&str>> = records
        .iter()
        .map(|r| r.content.as_deref())
        .collect();
    let pages: Vec<Option<u16>> = records.iter().map(|r| r.page).collect();
    let chapter_idxs: Vec<u16> = records.iter().map(|r| r.chapter_idx).collect();
    let section_idxs: Vec<u16> = records.iter().map(|r| r.section_idx).collect();
    let titles: Vec<&str> = vec![params.title.as_str(); n];
    let authors_list: Vec<Vec<String>> = vec![params.authors.clone(); n];
    let years: Vec<Option<u16>> = vec![params.year; n];
    let venues: Vec<Option<&str>> = vec![params.venue.as_deref(); n];
    let tags_list: Vec<Vec<String>> = vec![params.tags.clone(); n];
    let first_ref_chunk_ids: Vec<Option<&str>> = records
        .iter()
        .map(|r| r.first_ref_chunk_id.as_deref())
        .collect();
    let ref_counts: Vec<u16> = records.iter().map(|r| r.ref_count).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(exhibit_ids)),
            Arc::new(StringArray::from(paper_ids)),
            Arc::new(vectors),
            Arc::new(StringArray::from(exhibit_types)),
            Arc::new(StringArray::from(captions)),
            Arc::new(StringArray::from(descriptions)),
            Arc::new(StringArray::from(image_paths)),
            Arc::new(StringArray::from(contents)),
            Arc::new(UInt16Array::from(pages)),
            Arc::new(UInt16Array::from(chapter_idxs)),
            Arc::new(UInt16Array::from(section_idxs)),
            Arc::new(StringArray::from(titles)),
            Arc::new(build_string_list_array(&authors_list)),
            Arc::new(UInt16Array::from(years)),
            Arc::new(StringArray::from(venues)),
            Arc::new(build_string_list_array(&tags_list)),
            Arc::new(StringArray::from(first_ref_chunk_ids)),
            Arc::new(UInt16Array::from(ref_counts)),
        ],
    )
    .map_err(|e: ArrowError| DbError::Arrow(e.to_string()))?;

    Ok(batch)
}

/// Check if a paper is already indexed in the RAG database.
pub async fn is_ingested(store: &DbStore, paper_id: &str) -> bool {
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

/// List all item keys that have a cached extraction (extract cache or DataLab cache).
pub fn list_cached_item_keys() -> Vec<String> {
    let mut keys = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // New extract cache
    for key in papers_core::extract_cache::list_cached_extract_keys() {
        if seen.insert(key.clone()) {
            keys.push(key);
        }
    }

    // Legacy DataLab cache
    if let Some(base) = datalab_cache_root() {
        if base.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&base) {
                for entry in entries.flatten() {
                    if !entry.path().is_dir() {
                        continue;
                    }
                    let key = match entry.file_name().to_str() {
                        Some(k) => k.to_string(),
                        None => continue,
                    };
                    if entry.path().join(format!("{key}.json")).exists() && seen.insert(key.clone()) {
                        keys.push(key);
                    }
                }
            }
        }
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use serial_test::serial;
    use tempfile::TempDir;

    // ── strip_html ───────────────────────────────────────────────────────────

    #[test]
    fn strip_html_simple_tag() {
        assert_eq!(strip_html("<h2>Hello World</h2>"), "Hello World");
    }

    #[test]
    fn strip_html_multiple_tags() {
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

    // ── estimate_tokens ─────────────────────────────────────────────────────

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_tokens_single_word() {
        assert_eq!(estimate_tokens("hello"), 2); // ceil(1 * 1.3)
    }

    #[test]
    fn estimate_tokens_sentence() {
        let tokens = estimate_tokens("The quick brown fox jumps over the lazy dog");
        assert!(tokens >= 10 && tokens <= 15);
    }

    // ── ChunkBuffer ─────────────────────────────────────────────────────────

    #[test]
    fn chunk_buffer_push_updates_tokens_and_pages() {
        let mut buf = ChunkBuffer::new();
        buf.push("hello world".to_string(), Some(3));
        assert!(buf.token_count > 0);
        assert_eq!(buf.page_start, Some(3));
        assert_eq!(buf.page_end, Some(3));

        buf.push("more text".to_string(), Some(5));
        assert_eq!(buf.page_start, Some(3));
        assert_eq!(buf.page_end, Some(5));
    }

    #[test]
    fn chunk_buffer_flush_joins_and_resets() {
        let mut buf = ChunkBuffer::new();
        buf.push("paragraph one".to_string(), Some(1));
        buf.push("paragraph two".to_string(), Some(2));
        let flushed = buf.flush().unwrap();
        assert_eq!(flushed.text, "paragraph one\n\nparagraph two");
        assert_eq!(flushed.page_start, Some(1));
        assert_eq!(flushed.page_end, Some(2));
        assert!(buf.is_empty());
    }

    #[test]
    fn chunk_buffer_flush_empty_returns_none() {
        let mut buf = ChunkBuffer::new();
        assert!(buf.flush().is_none());
    }

    #[test]
    fn chunk_buffer_overlap_tail_extracts_sentences() {
        let text = "First sentence. Second sentence. Third sentence.";
        let overlap = ChunkBuffer::overlap_tail(text);
        assert!(overlap.contains("Second sentence."));
        assert!(overlap.contains("Third sentence."));
    }

    #[test]
    fn chunk_buffer_overlap_tail_single_sentence() {
        let text = "Only one sentence.";
        let overlap = ChunkBuffer::overlap_tail(text);
        assert!(overlap.is_empty());
    }

    // ── list_cached_item_keys ─────────────────────────────────────────────────

    #[serial]
    #[test]
    fn list_cached_keys_finds_dirs_with_json() {
        let dir = TempDir::new().unwrap();

        let key1_dir = dir.path().join("ABCD1234");
        fs::create_dir(&key1_dir).unwrap();
        fs::write(key1_dir.join("ABCD1234.json"), "{}").unwrap();

        let key2_dir = dir.path().join("EFGH5678");
        fs::create_dir(&key2_dir).unwrap();

        fs::write(dir.path().join("not_a_dir"), "").unwrap();

        let key4_dir = dir.path().join("IJKL9012");
        fs::create_dir(&key4_dir).unwrap();
        fs::write(key4_dir.join("other.json"), "{}").unwrap();

        let old = std::env::var("PAPERS_DATALAB_CACHE_DIR").ok();
        let old_extract = std::env::var("PAPERS_EXTRACT_CACHE_DIR").ok();
        unsafe {
            std::env::set_var("PAPERS_DATALAB_CACHE_DIR", dir.path().to_str().unwrap());
            // Point extract cache at an empty temp dir so it doesn't pick up stale data
            std::env::set_var("PAPERS_EXTRACT_CACHE_DIR", dir.path().join("_empty_extract").to_str().unwrap());
        }

        let mut keys = list_cached_item_keys();
        keys.sort();

        unsafe {
            if let Some(old_val) = old {
                std::env::set_var("PAPERS_DATALAB_CACHE_DIR", old_val);
            } else {
                std::env::remove_var("PAPERS_DATALAB_CACHE_DIR");
            }
            if let Some(old_val) = old_extract {
                std::env::set_var("PAPERS_EXTRACT_CACHE_DIR", old_val);
            } else {
                std::env::remove_var("PAPERS_EXTRACT_CACHE_DIR");
            }
        }

        assert_eq!(keys, vec!["ABCD1234"]);
    }

    #[serial]
    #[test]
    fn list_cached_keys_nonexistent_cache_dir_returns_empty() {
        let old = std::env::var("PAPERS_DATALAB_CACHE_DIR").ok();
        let old_extract = std::env::var("PAPERS_EXTRACT_CACHE_DIR").ok();
        unsafe {
            std::env::set_var(
                "PAPERS_DATALAB_CACHE_DIR",
                "/nonexistent/path/that/does/not/exist",
            );
            std::env::set_var(
                "PAPERS_EXTRACT_CACHE_DIR",
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
            if let Some(old_val) = old_extract {
                std::env::set_var("PAPERS_EXTRACT_CACHE_DIR", old_val);
            } else {
                std::env::remove_var("PAPERS_EXTRACT_CACHE_DIR");
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

    // ── strip_html_preserve_math ──────────────────────────────────────────

    #[test]
    fn test_strip_html_preserve_math() {
        let input = r#"<p><math display="block">\mathbf{x} = 0 \quad (1)</math></p>"#;
        assert_eq!(
            strip_html_preserve_math(input),
            r"$\mathbf{x} = 0 \quad (1)$"
        );
    }

    #[test]
    fn test_strip_html_preserve_math_inline() {
        let input = r#"<p>Some text <math>a+b</math> more text</p>"#;
        assert_eq!(strip_html_preserve_math(input), "Some text $a+b$ more text");
    }

    #[test]
    fn test_strip_html_preserve_math_no_math() {
        let input = "<p>No math here</p>";
        assert_eq!(strip_html_preserve_math(input), "No math here");
    }

    // ── normalize_exhibit_kind ──────────────────────────────────────────────

    #[test]
    fn normalize_exhibit_kind_figure_variants() {
        assert_eq!(normalize_exhibit_kind("Figure"), "figure");
        assert_eq!(normalize_exhibit_kind("Fig"), "figure");
        assert_eq!(normalize_exhibit_kind("Figures"), "figure");
        assert_eq!(normalize_exhibit_kind("Figs"), "figure");
    }

    #[test]
    fn normalize_exhibit_kind_table_variants() {
        assert_eq!(normalize_exhibit_kind("Table"), "table");
        assert_eq!(normalize_exhibit_kind("Tab"), "table");
    }

    #[test]
    fn normalize_exhibit_kind_algorithm_variants() {
        assert_eq!(normalize_exhibit_kind("Algorithm"), "algorithm");
        assert_eq!(normalize_exhibit_kind("Alg"), "algorithm");
        assert_eq!(normalize_exhibit_kind("Procedure"), "algorithm");
        assert_eq!(normalize_exhibit_kind("Pseudocode"), "algorithm");
        assert_eq!(normalize_exhibit_kind("Listing"), "algorithm");
        assert_eq!(normalize_exhibit_kind("Code"), "algorithm");
    }

    // ── Reflow chunking tests ────────────────────────────────────────────────

    use papers_extract::types::{ReflowDocument, ReflowNode, TocEntryRendered};

    fn test_params() -> IngestParams {
        IngestParams {
            item_key: "TESTKEY".into(),
            paper_id: "TESTKEY".into(),
            title: "Test Paper".into(),
            authors: vec!["Author A".into()],
            year: Some(2024),
            venue: Some("Test Journal".into()),
            tags: vec![],
            cache_dir: std::path::PathBuf::from("."),
            force: false,
        }
    }

    #[test]
    fn reflow_basic_chapter_section() {
        let doc = ReflowDocument {
            title: Some("Test".into()),
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Introduction".into(),
                    children: vec![
                        ReflowNode::Text { content: "Intro paragraph one.".into(), footnotes: vec![] },
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.1 Background".into(),
                            children: vec![
                                ReflowNode::Text { content: "Background text.".into(), footnotes: vec![] },
                            ],
                        },
                    ],
                },
                ReflowNode::Heading {
                    depth: 1,
                    title: "2 Methods".into(),
                    children: vec![
                        ReflowNode::Text { content: "Methods paragraph.".into(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _exhibits) = parse_reflow_document(&params, &doc).unwrap();

        // Should have 3 chunks: intro, background, methods
        assert_eq!(chunks.len(), 3);

        assert_eq!(chunks[0].chapter_idx, 1);
        assert_eq!(chunks[0].section_idx, 0);
        assert_eq!(chunks[0].chapter_title, "1 Introduction");
        assert!(chunks[0].text.contains("Intro paragraph one"));

        assert_eq!(chunks[1].chapter_idx, 1);
        assert_eq!(chunks[1].section_idx, 1);
        assert_eq!(chunks[1].section_title, "1.1 Background");

        assert_eq!(chunks[2].chapter_idx, 2);
        assert_eq!(chunks[2].section_idx, 0);
        assert_eq!(chunks[2].chapter_title, "2 Methods");
    }

    #[test]
    fn reflow_formula_absorption() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Math".into(),
                    children: vec![
                        ReflowNode::Text {
                            content: "The energy is given by".into(),
                            footnotes: vec![],
                        },
                        ReflowNode::Formula {
                            content: Some("$$E = mc^2$$".into()),
                            path: None,
                        },
                        ReflowNode::Text {
                            content: "where m is mass.".into(),
                            footnotes: vec![],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // All three should be in one chunk (formula absorbed)
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("energy is given by"));
        assert!(chunks[0].text.contains("$$E = mc^2$$"));
        assert!(chunks[0].text.contains("where m is mass"));
    }

    #[test]
    fn reflow_formula_image_fallback() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Math".into(),
                    children: vec![
                        ReflowNode::Text { content: "See the formula below.".into(), footnotes: vec![] },
                        ReflowNode::Formula { content: None, path: Some("images/eq1.png".into()) },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("[display formula]"));
    }

    #[test]
    fn reflow_exhibit_creation() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Results".into(),
                    children: vec![
                        ReflowNode::Text { content: "Some results.".into(), footnotes: vec![] },
                        ReflowNode::Figure {
                            path: "images/fig1.png".into(),
                            caption: Some("Figure 1: Overview of the system.".into()),
                        },
                        ReflowNode::Table {
                            content: "| A | B |\n|---|---|\n| 1 | 2 |".into(),
                            caption: Some("Table 1: Results summary.".into()),
                        },
                        ReflowNode::Algorithm {
                            content: "Algorithm 1: Main Loop\n1. Init\n2. Iterate".into(),
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (_, exhibits) = parse_reflow_document(&params, &doc).unwrap();

        assert_eq!(exhibits.len(), 3);
        assert_eq!(exhibits[0].exhibit_type, "figure");
        assert_eq!(exhibits[0].caption, "Figure 1: Overview of the system.");
        assert_eq!(exhibits[1].exhibit_type, "table");
        assert!(exhibits[1].content.as_ref().unwrap().contains("| A | B |"));
        assert_eq!(exhibits[2].exhibit_type, "algorithm");
        assert!(exhibits[2].content.as_ref().unwrap().contains("Main Loop"));
    }

    #[test]
    fn reflow_exhibit_linking() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Results".into(),
                    children: vec![
                        ReflowNode::Figure {
                            path: "fig1.png".into(),
                            caption: Some("Figure 1: The system.".into()),
                        },
                        ReflowNode::Table {
                            content: "| X |".into(),
                            caption: Some("Table 1: Data.".into()),
                        },
                        ReflowNode::Text {
                            content: "As shown in Figure 1, the system works. See Table 1 for data.".into(),
                            footnotes: vec![],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, exhibits) = parse_reflow_document(&params, &doc).unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].exhibit_ids.len(), 2);

        // Exhibits should have first_ref set
        let fig = exhibits.iter().find(|e| e.exhibit_type == "figure").unwrap();
        assert!(fig.first_ref_chunk_id.is_some());
        assert_eq!(fig.ref_count, 1);
    }

    #[test]
    fn reflow_footnotes_inline() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Intro".into(),
                    children: vec![
                        ReflowNode::Text {
                            content: "Main text with a citation.".into(),
                            footnotes: vec!["See Smith (2020) for details.".into()],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Main text"));
        assert!(chunks[0].text.contains("See Smith (2020)"));
    }

    #[test]
    fn reflow_list_in_chunk() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Steps".into(),
                    children: vec![
                        ReflowNode::Text { content: "The steps are:".into(), footnotes: vec![] },
                        ReflowNode::List {
                            list_type: "numbered".into(),
                            items: vec!["First step".into(), "Second step".into()],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("1. First step"));
        assert!(chunks[0].text.contains("2. Second step"));
    }

    #[test]
    fn reflow_subsection_inline() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Chapter".into(),
                    children: vec![
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.1 Section".into(),
                            children: vec![
                                ReflowNode::Text { content: "Section text.".into(), footnotes: vec![] },
                                ReflowNode::Heading {
                                    depth: 3,
                                    title: "1.1.1 Subsection".into(),
                                    children: vec![
                                        ReflowNode::Text { content: "Subsection text.".into(), footnotes: vec![] },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // Subsection should NOT create a new section boundary
        // All content in section 1.1 (including subsection) should stay in section_idx=1
        for c in &chunks {
            assert_eq!(c.chapter_idx, 1);
            assert_eq!(c.section_idx, 1);
        }

        // The subsection heading should appear as bold text
        let all_text: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n\n");
        assert!(all_text.contains("**1.1.1 Subsection**"));
    }

    #[test]
    fn reflow_references_stop() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Intro".into(),
                    children: vec![
                        ReflowNode::Text { content: "Main content.".into(), footnotes: vec![] },
                    ],
                },
                ReflowNode::References { content: "[1] Smith 2020.".into() },
                ReflowNode::Heading {
                    depth: 1,
                    title: "Appendix".into(),
                    children: vec![
                        ReflowNode::Text { content: "This should be skipped.".into(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].text.contains("skipped"));
    }

    #[test]
    fn reflow_no_headings() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Text { content: "Standalone paragraph.".into(), footnotes: vec![] },
                ReflowNode::Text { content: "Another paragraph.".into(), footnotes: vec![] },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].chapter_idx, 0);
        assert_eq!(chunks[0].section_idx, 0);
    }

    #[test]
    fn reflow_empty_heading() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Empty Chapter".into(),
                    children: vec![],
                },
                ReflowNode::Heading {
                    depth: 1,
                    title: "2 Real Chapter".into(),
                    children: vec![
                        ReflowNode::Text { content: "Real content.".into(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // Should not crash, and the real content should be in chapter 2
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].chapter_idx, 2);
    }

    #[test]
    fn reflow_code_block() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Code".into(),
                    children: vec![
                        ReflowNode::Text { content: "Example code:".into(), footnotes: vec![] },
                        ReflowNode::CodeBlock {
                            content: "fn main() {}".into(),
                            language: Some("rust".into()),
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("```rust"));
        assert!(chunks[0].text.contains("fn main()"));
    }

    #[test]
    fn reflow_chunk_ids_correct_format() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Ch".into(),
                    children: vec![
                        ReflowNode::Text { content: "P1.".into(), footnotes: vec![] },
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.1 Sec".into(),
                            children: vec![
                                ReflowNode::Text { content: "P2.".into(), footnotes: vec![] },
                            ],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        assert_eq!(chunks[0].chunk_id, "TESTKEY/ch1/s0/p0");
        assert_eq!(chunks[1].chunk_id, "TESTKEY/ch1/s1/p0");
    }

    #[test]
    fn reflow_frontmatter_skipped() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::FrontMatter {
                    children: vec![
                        ReflowNode::Text { content: "Copyright 2024".into(), footnotes: vec![] },
                    ],
                },
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Intro".into(),
                    children: vec![
                        ReflowNode::Text { content: "Real content.".into(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].text.contains("Copyright"));
    }

    #[test]
    fn reflow_standalone_footnote() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Intro".into(),
                    children: vec![
                        ReflowNode::Text { content: "Main text.".into(), footnotes: vec![] },
                        ReflowNode::Footnote {
                            marker: "1".into(),
                            content: "A standalone footnote.".into(),
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("[1] A standalone footnote."));
    }

    // ── Token overflow & overlap tests ───────────────────────────────────────

    /// Generate a text block with approximately `n` tokens (words * 1.3).
    fn words(n: usize) -> String {
        (0..n).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn reflow_mid_section_flush_with_overlap() {
        // TARGET = 400 tokens. Two big paragraphs that each exceed TARGET
        // force a mid-section flush with overlap carry.
        let para1 = words(320); // ~416 tokens
        let para2 = words(320); // ~416 tokens

        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Chapter".into(),
                    children: vec![
                        ReflowNode::Text { content: para1.clone(), footnotes: vec![] },
                        ReflowNode::Text { content: para2.clone(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // First para fills buffer past TARGET. Second para triggers would_overflow →
        // flush_mid_section → emit first chunk → carry overlap → push second para.
        // At end-of-doc, second chunk is emitted. But smart_merge may merge if combined < MAX.
        // Since each is ~416 tokens and MAX=600, they can't merge (416+416=832 > 600).
        assert!(chunks.len() >= 2, "expected at least 2 chunks, got {}", chunks.len());
        assert_eq!(chunks[0].chapter_idx, 1);
        assert_eq!(chunks[1].chapter_idx, 1);
    }

    #[test]
    fn reflow_small_chunk_merge_same_section() {
        // Two short paragraphs in the same section — should be merged
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Chapter".into(),
                    children: vec![
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.1 Section A".into(),
                            children: vec![
                                ReflowNode::Text { content: words(310), footnotes: vec![] },
                                ReflowNode::Text { content: "A short tail.".into(), footnotes: vec![] },
                            ],
                        },
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.2 Section B".into(),
                            children: vec![
                                ReflowNode::Text { content: "Tiny section.".into(), footnotes: vec![] },
                            ],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // Section A has a big para + short tail → flush big, then merge tail into big
        // Section B is tiny → should merge with... well it's a different section, so standalone
        // Check that the tiny section B is present
        let last = chunks.last().unwrap();
        assert_eq!(last.section_idx, 2);
        assert!(last.text.contains("Tiny section"));
    }

    #[test]
    fn reflow_small_chunk_merge_within_section() {
        // A section boundary with a small trailing chunk that gets merged back
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Chapter".into(),
                    children: vec![
                        ReflowNode::Text { content: "First paragraph of chapter.".into(), footnotes: vec![] },
                        ReflowNode::Text { content: "Second short paragraph.".into(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // Both are small, should be merged into one chunk
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("First paragraph"));
        assert!(chunks[0].text.contains("Second short"));
    }

    #[test]
    fn reflow_formula_overflow_flushes_before() {
        // Buffer near MAX (600 tokens), formula pushes it over → flush before formula
        // The formula overflow check uses MAX_CHUNK_TOKENS, not TARGET
        let big_para = words(460); // ~598 tokens, just under MAX=600
        let formula = "$$\\sum_{i=1}^{n} x_i^2 + y_i^2 = R^2 \\quad \\forall i \\in \\{1,\\ldots,n\\}$$";
        // Formula is ~15 words ≈ 20 tokens. 598+20=618 > MAX=600 → should flush
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Math".into(),
                    children: vec![
                        ReflowNode::Text { content: big_para, footnotes: vec![] },
                        ReflowNode::Formula {
                            content: Some(formula.into()),
                            path: None,
                        },
                        ReflowNode::Text { content: "where R is the radius.".into(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // big_para fills buffer to ~598 tokens. Then formula check:
        // 598 + ~20 = 618 > MAX=600 → flush_mid_section → emit big_para → formula in new buffer.
        assert!(chunks.len() >= 2, "expected ≥2 chunks, got {}", chunks.len());

        // Formula should be in the second (or later) chunk, not the first
        let first_text = &chunks[0].text;
        assert!(!first_text.contains("\\sum"), "formula should not be in the first chunk");

        // Formula should be in some later chunk
        let later = chunks[1..].iter().any(|c| c.text.contains("\\sum"));
        assert!(later, "formula should appear in a later chunk");
    }

    #[test]
    fn reflow_multiple_chapters_section_reset() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 First".into(),
                    children: vec![
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.1 Sec A".into(),
                            children: vec![
                                ReflowNode::Text { content: "Content A.".into(), footnotes: vec![] },
                            ],
                        },
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.2 Sec B".into(),
                            children: vec![
                                ReflowNode::Text { content: "Content B.".into(), footnotes: vec![] },
                            ],
                        },
                    ],
                },
                ReflowNode::Heading {
                    depth: 1,
                    title: "2 Second".into(),
                    children: vec![
                        ReflowNode::Heading {
                            depth: 2,
                            title: "2.1 Sec C".into(),
                            children: vec![
                                ReflowNode::Text { content: "Content C.".into(), footnotes: vec![] },
                            ],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // Chapter 2's first section should have section_idx=1 (reset from chapter 1)
        let ch2_chunks: Vec<_> = chunks.iter().filter(|c| c.chapter_idx == 2).collect();
        assert!(!ch2_chunks.is_empty());
        assert_eq!(ch2_chunks[0].section_idx, 1);
        assert_eq!(ch2_chunks[0].chunk_idx, 0);
    }

    #[test]
    fn reflow_figure_group_creates_exhibit() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Figures".into(),
                    children: vec![
                        ReflowNode::FigureGroup {
                            path: "images/group1.png".into(),
                            caption: Some("Figure 2: Composite overview.".into()),
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (_, exhibits) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(exhibits.len(), 1);
        assert_eq!(exhibits[0].exhibit_type, "figure");
        assert!(exhibits[0].caption.contains("Figure 2"));
    }

    #[test]
    fn reflow_algorithm_linking_from_content() {
        // Algorithm without a caption — number extracted from content first line
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Methods".into(),
                    children: vec![
                        ReflowNode::Algorithm {
                            content: "Algorithm 1: Training Loop\n1. Forward pass\n2. Backward pass\n3. Update weights".into(),
                        },
                        ReflowNode::Text {
                            content: "As shown in Algorithm 1, the training procedure is straightforward.".into(),
                            footnotes: vec![],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, exhibits) = parse_reflow_document(&params, &doc).unwrap();

        assert_eq!(exhibits.len(), 1);
        assert_eq!(exhibits[0].exhibit_type, "algorithm");

        // The text chunk should reference the algorithm
        assert_eq!(chunks[0].exhibit_ids.len(), 1);
        assert_eq!(exhibits[0].first_ref_chunk_id, Some(chunks[0].chunk_id.clone()));
    }

    #[test]
    fn reflow_bulleted_list() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Points".into(),
                    children: vec![
                        ReflowNode::List {
                            list_type: "bulleted".into(),
                            items: vec!["Alpha".into(), "Beta".into(), "Gamma".into()],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("- Alpha"));
        assert!(chunks[0].text.contains("- Beta"));
        assert!(chunks[0].text.contains("- Gamma"));
    }

    #[test]
    fn reflow_formatted_text_included() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Misc".into(),
                    children: vec![
                        ReflowNode::FormattedText { content: "Key: Value\nFoo: Bar".into() },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Key: Value"));
    }

    #[test]
    fn reflow_footnote_block_skipped() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Intro".into(),
                    children: vec![
                        ReflowNode::Text { content: "Main text.".into(), footnotes: vec![] },
                    ],
                },
                ReflowNode::FootnoteBlock { content: "1. Footnote one\n2. Footnote two".into() },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        // FootnoteBlock should be skipped entirely
        for c in &chunks {
            assert!(!c.text.contains("Footnote one"));
        }
    }

    #[test]
    fn reflow_heading_references_anywhere() {
        // References heading as a child of another heading (not just top-level)
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Intro".into(),
                    children: vec![
                        ReflowNode::Text { content: "Before references.".into(), footnotes: vec![] },
                    ],
                },
                ReflowNode::Heading {
                    depth: 1,
                    title: "References".into(),
                    children: vec![
                        ReflowNode::Text { content: "Should not appear.".into(), footnotes: vec![] },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Before references"));
        for c in &chunks {
            assert!(!c.text.contains("Should not appear"));
        }
    }

    #[test]
    fn reflow_multiple_exhibits_cross_referenced() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Results".into(),
                    children: vec![
                        ReflowNode::Figure {
                            path: "fig1.png".into(),
                            caption: Some("Figure 1: First.".into()),
                        },
                        ReflowNode::Figure {
                            path: "fig2.png".into(),
                            caption: Some("Figure 2: Second.".into()),
                        },
                        ReflowNode::Table {
                            content: "| X |".into(),
                            caption: Some("Table 1: Data.".into()),
                        },
                        ReflowNode::Text {
                            content: "See Figure 1 and Figure 2 for visuals. Table 1 has the numbers.".into(),
                            footnotes: vec![],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, exhibits) = parse_reflow_document(&params, &doc).unwrap();

        assert_eq!(exhibits.len(), 3);
        assert_eq!(chunks[0].exhibit_ids.len(), 3, "should reference all 3 exhibits");

        // All exhibits should have ref_count=1
        for ex in &exhibits {
            assert_eq!(ex.ref_count, 1, "exhibit {} should have ref_count=1", ex.exhibit_id);
        }
    }

    #[test]
    fn reflow_deep_nesting_depth_4_and_5() {
        let doc = ReflowDocument {
            title: None,
            toc: vec![],
            children: vec![
                ReflowNode::Heading {
                    depth: 1,
                    title: "1 Chapter".into(),
                    children: vec![
                        ReflowNode::Heading {
                            depth: 2,
                            title: "1.1 Section".into(),
                            children: vec![
                                ReflowNode::Heading {
                                    depth: 3,
                                    title: "1.1.1 Subsection".into(),
                                    children: vec![
                                        ReflowNode::Heading {
                                            depth: 4,
                                            title: "1.1.1.1 Sub-subsection".into(),
                                            children: vec![
                                                ReflowNode::Heading {
                                                    depth: 5,
                                                    title: "Deep heading".into(),
                                                    children: vec![
                                                        ReflowNode::Text {
                                                            content: "Deeply nested content.".into(),
                                                            footnotes: vec![],
                                                        },
                                                    ],
                                                },
                                            ],
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // All content should be in chapter 1, section 1 (depths 3+ are inlined)
        for c in &chunks {
            assert_eq!(c.chapter_idx, 1);
            assert_eq!(c.section_idx, 1);
        }

        let all_text: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n\n");
        assert!(all_text.contains("**1.1.1 Subsection**"));
        assert!(all_text.contains("**1.1.1.1 Sub-subsection**"));
        assert!(all_text.contains("**Deep heading**"));
        assert!(all_text.contains("Deeply nested content."));
    }

    #[test]
    fn reflow_large_document_many_sections() {
        // Stress test: 10 chapters, 3 sections each, 2 paragraphs per section
        let mut children = Vec::new();
        for ch in 1..=10 {
            let mut chapter_children = Vec::new();
            for sec in 1..=3 {
                let mut sec_children = Vec::new();
                sec_children.push(ReflowNode::Text {
                    content: format!("Paragraph 1 of chapter {ch} section {sec}."),
                    footnotes: vec![],
                });
                sec_children.push(ReflowNode::Text {
                    content: format!("Paragraph 2 of chapter {ch} section {sec}."),
                    footnotes: vec![],
                });
                chapter_children.push(ReflowNode::Heading {
                    depth: 2,
                    title: format!("{ch}.{sec} Section"),
                    children: sec_children,
                });
            }
            children.push(ReflowNode::Heading {
                depth: 1,
                title: format!("{ch} Chapter {ch}"),
                children: chapter_children,
            });
        }

        let doc = ReflowDocument { title: None, toc: vec![], children };

        let params = test_params();
        let (chunks, _) = parse_reflow_document(&params, &doc).unwrap();

        // 10 chapters × 3 sections × 1 merged chunk (2 short paras merge) = 30 chunks
        assert_eq!(chunks.len(), 30, "expected 30 chunks for 10×3 sections");

        // Verify chapter/section indices
        assert_eq!(chunks.last().unwrap().chapter_idx, 10);
        assert_eq!(chunks.last().unwrap().section_idx, 3);

        // Verify chunk IDs are unique
        let ids: std::collections::HashSet<_> = chunks.iter().map(|c| &c.chunk_id).collect();
        assert_eq!(ids.len(), chunks.len(), "all chunk IDs should be unique");
    }
}
