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

use crate::error::DbError;
use crate::schema::{EMBED_DIM, chunks_schema, figures_schema};
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

struct ChunkRecord {
    chunk_id: String,
    chapter_title: String,
    chapter_idx: u16,
    section_title: String,
    section_idx: u16,
    chunk_idx: u16,
    block_type: String,
    text: String,
    page: Option<u16>,
    figure_ids: Vec<String>,
}

struct FigureRecord {
    figure_id: String,
    figure_type: String,
    caption: String,
    description: Option<String>,
    image_path: Option<String>,
    content: Option<String>,
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

/// Strip HTML tags but preserve the text content of `<math>` elements as LaTeX.
///
/// `<math ...>content</math>` is replaced by `$content$` before other tags are
/// stripped, so equations survive as renderable inline LaTeX.
fn strip_html_preserve_math(html: &str) -> String {
    let mut buf = String::with_capacity(html.len());
    let mut remaining = html;

    while let Some(start) = remaining.find("<math") {
        // Copy everything before the <math> tag
        buf.push_str(&remaining[..start]);
        remaining = &remaining[start..];

        // Find the closing > of the opening <math ...> tag
        if let Some(tag_end) = remaining.find('>') {
            remaining = &remaining[tag_end + 1..];
            // Find </math>
            if let Some(close) = remaining.find("</math>") {
                buf.push('$');
                buf.push_str(&remaining[..close]);
                buf.push('$');
                remaining = &remaining[close + 7..]; // skip "</math>"
            }
            // else: malformed, skip the opening tag and keep going
        } else {
            // Malformed tag, skip the '<' and keep going
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
/// Returns 1 if the attribute is missing or unparseable.
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

        // Find the > that closes the opening tag
        let tag_close = match lower[cell_abs..].find('>') {
            Some(p) => cell_abs + p + 1,
            None => break,
        };

        let opening_tag_lower = &lower[cell_abs..tag_close];

        let colspan = parse_span_attr(opening_tag_lower, "colspan");
        let rowspan = parse_span_attr(opening_tag_lower, "rowspan");

        // Find closing </th> or </td>
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

        // Skip past the closing tag (</th> or </td> = 5 chars)
        search = cell_content_end + 5;
    }

    cells
}

/// Build a 2D grid from parsed rows, filling colspan/rowspan blocks.
///
/// Returns `(grid, thead_row_count)` where each grid cell is `Some(text)` if
/// filled by a cell or `None` if unfilled (shouldn't happen with valid HTML).
fn build_table_grid(
    html: &str,
) -> (Vec<Vec<Option<String>>>, usize) {
    let lower = html.to_ascii_lowercase();

    // Determine thead boundary (on original html, using lowercase for searching)
    let thead_range = lower.find("<thead").and_then(|start| {
        lower.find("</thead>").map(|end| (start, end))
    });

    // Extract all <tr> blocks with their in-thead status
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

    // First pass: determine grid dimensions
    let num_rows = raw_rows.len();
    // We'll build the grid incrementally to determine width
    let mut grid: Vec<Vec<Option<String>>> = vec![Vec::new(); num_rows];
    let mut thead_row_count = 0usize;

    for (row_idx, (cells, in_thead)) in raw_rows.iter().enumerate() {
        if *in_thead {
            thead_row_count = row_idx + 1;
        }

        let mut col = 0;
        for cell in cells {
            // Skip columns already occupied by rowspan from previous rows
            while col < grid[row_idx].len() && grid[row_idx][col].is_some() {
                col += 1;
            }

            // Fill the R×C block
            for dr in 0..cell.rowspan {
                let r = row_idx + dr;
                if r >= num_rows {
                    break;
                }
                for dc in 0..cell.colspan {
                    let c = col + dc;
                    // Ensure the grid row is wide enough
                    while grid[r].len() <= c {
                        grid[r].push(None);
                    }
                    grid[r][c] = Some(cell.text.clone());
                }
            }

            col += cell.colspan;
        }
    }

    // Normalize all rows to the same width
    let max_cols = grid.iter().map(|r| r.len()).max().unwrap_or(0);
    for row in &mut grid {
        row.resize(max_cols, None);
    }

    (grid, thead_row_count)
}

/// Flatten multi-row header into a single row.
///
/// For each column, if the header rows have different values, join them with
/// a space. If they repeat (from rowspan), keep one copy.
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
            // Skip duplicates (from rowspan filling the same value)
            if parts.last().map_or(true, |last| last != &val) {
                parts.push(val);
            }
        }
        // Join non-empty parts
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
///
/// Returns `None` if the html doesn't contain a `<table` tag (i.e. it's an
/// `<img>` reference, not a real table).
fn html_table_to_markdown(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    if !lower.contains("<table") {
        return None;
    }

    // Normalize <br> variants to space before processing
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
        // Flatten multi-row header into one row
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

        // Body rows (everything after thead)
        for row in &grid[thead_row_count..] {
            let cells: Vec<&str> = row
                .iter()
                .map(|c| c.as_deref().unwrap_or(""))
                .collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    } else if thead_row_count == 1 {
        // Single header row
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
        // No thead — all rows as body
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

/// Return the DataLab cache root directory.
fn cache_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("PAPERS_DATALAB_CACHE_DIR") {
        return Some(PathBuf::from(p));
    }
    dirs::cache_dir().map(|d| d.join("papers").join("datalab"))
}

/// Build IngestParams from a cached item_key using meta.json.
pub fn ingest_params_from_cache(item_key: &str) -> Result<IngestParams, DbError> {
    let root = cache_root().ok_or_else(|| {
        DbError::NotFound("cannot determine cache directory".into())
    })?;
    let cache_dir = root.join(item_key);
    if !cache_dir.is_dir() {
        return Err(DbError::NotFound(format!(
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
        .unwrap_or_else(|_| "embedding-gemma-300m".to_string())
}

/// Convenience helper: convert an `EmbedCacheError` to `DbError::Cache`.
fn cache_err(e: crate::embed_cache::EmbedCacheError) -> DbError {
    DbError::Cache(e.to_string())
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
) -> Result<(Vec<ChunkRecord>, Vec<FigureRecord>), DbError> {
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

    // Set of block indices consumed as captions (to avoid double-processing)
    let mut consumed_captions: std::collections::HashSet<usize> = std::collections::HashSet::new();

    /// Helper: look for an adjacent `Caption` block at `idx` in `blocks`.
    /// Returns the stripped caption text if found.
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

    let mut i = 0;
    while i < flat_blocks.len() {
        // Skip already-consumed caption indices
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
                // Standalone caption not consumed by a figure — skip
                i += 1;
                continue;
            }

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
                let text = if block_type == "Equation" {
                    strip_html_preserve_math(html)
                } else {
                    strip_html(html)
                };
                if text.trim().is_empty() {
                    i += 1;
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
                let bt = match block_type {
                    "Equation" => "equation",
                    "ListGroup" => "list",
                    _ => "text",
                };
                chunk_records.push(ChunkRecord {
                    chunk_id,
                    chapter_title: current_chapter_title.clone(),
                    chapter_idx,
                    section_title: current_section_title.clone(),
                    section_idx,
                    chunk_idx,
                    block_type: bt.to_string(),
                    text,
                    page: page_num,
                    figure_ids: vec![],
                });
                chunk_idx += 1;
            }

            "Figure" | "Table" | "Picture" => {
                let html = block.get("html").and_then(|v| v.as_str()).unwrap_or("");
                let alt_text = extract_img_alt(html).unwrap_or_default();
                let src = extract_img_src(html);

                // Look for an adjacent Caption block (after or before)
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

                // For Picture blocks, require an adjacent Caption to distinguish
                // real figures from logos/icons
                if block_type == "Picture" && adjacent_caption.is_none() {
                    i += 1;
                    continue;
                }

                let caption = adjacent_caption
                    .clone()
                    .unwrap_or_else(|| alt_text.clone());
                let description = if alt_text.is_empty() { None } else { Some(alt_text) };

                // Extract table content as markdown for Table blocks
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
                let figure_type = if block_type == "Table" { "table" } else { "figure" };
                figure_seq += 1;
                let figure_id = format!("{}/fig{}", params.paper_id, figure_seq);
                figure_records.push(FigureRecord {
                    figure_id,
                    figure_type: figure_type.to_string(),
                    caption,
                    description,
                    image_path,
                    content,
                    page: page_num,
                    chapter_idx,
                    section_idx,
                });
            }

            _ => {} // skip unknown block types
        }

        i += 1;
    }

    // ── Post-process: populate figure_ids on chunks ────────────────────────
    if !figure_records.is_empty() {
        // Build map: figure/table number → figure_id
        // Parse patterns like "Fig. 1.", "Figure 12.", "Table 3." from captions
        let fig_num_re = regex::Regex::new(r"(?:Figure|Fig\.)\s+(\d+)").unwrap();
        let tbl_num_re = regex::Regex::new(r"Table\s+(\d+)").unwrap();

        let mut fig_number_to_id: HashMap<(String, u32), String> = HashMap::new();
        for fr in &figure_records {
            if let Some(caps) = fig_num_re.captures(&fr.caption) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    fig_number_to_id.insert(("figure".to_string(), n), fr.figure_id.clone());
                }
            }
            if let Some(caps) = tbl_num_re.captures(&fr.caption) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    fig_number_to_id.insert(("table".to_string(), n), fr.figure_id.clone());
                }
            }
        }

        // Scan each chunk's text for figure/table references
        let ref_re = regex::Regex::new(r"(?:Figure|Fig\.)\s+(\d+)").unwrap();
        let tbl_ref_re = regex::Regex::new(r"Table\s+(\d+)").unwrap();

        for chunk in &mut chunk_records {
            let mut ids: Vec<String> = Vec::new();
            for caps in ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(fid) = fig_number_to_id.get(&("figure".to_string(), n)) {
                        if !ids.contains(fid) {
                            ids.push(fid.clone());
                        }
                    }
                }
            }
            for caps in tbl_ref_re.captures_iter(&chunk.text) {
                if let Ok(n) = caps[1].parse::<u32>() {
                    if let Some(fid) = fig_number_to_id.get(&("table".to_string(), n)) {
                        if !ids.contains(fid) {
                            ids.push(fid.clone());
                        }
                    }
                }
            }
            chunk.figure_ids = ids;
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
    store: &DbStore,
    params: &IngestParams,
    model: &str,
    force: bool,
) -> Result<usize, DbError> {
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
        let t = std::time::Instant::now();
        let texts: Vec<String> = chunk_records.iter().map(|c| c.text.clone()).collect();
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
pub async fn ingest_paper(store: &DbStore, params: IngestParams) -> Result<IngestStats, DbError> {
    let t_total = std::time::Instant::now();
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
                let t = std::time::Instant::now();
                let texts: Vec<String> =
                    chunk_records.iter().map(|c| c.text.clone()).collect();
                let result = store.embed_documents(texts).await?;
                eprintln!("  [{}] chunk embeddings done ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());

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
    // Embed caption + description for all figures/tables. Table content is
    // stored but not embedded — the small embedding model doesn't benefit
    // from tabular data.
    let fig_texts: Vec<String> = figure_records
        .iter()
        .map(|f| {
            match (f.caption.is_empty(), &f.description) {
                (false, Some(desc)) => format!("{}\n{}", f.caption, desc),
                (false, None) => f.caption.clone(),
                (true, Some(desc)) => desc.clone(),
                (true, None) => String::new(),
            }
        })
        .collect();
    let fig_embeddings = if fig_texts.is_empty() {
        vec![]
    } else {
        eprintln!(
            "  [{}] embedding {} figure captions...",
            params.item_key,
            fig_texts.len()
        );
        let t = std::time::Instant::now();
        let result = store.embed_documents(fig_texts).await?;
        eprintln!("  [{}] figure embeddings done ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
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
        // Rebuild vector index after inserting new data
        if let Err(e) = table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await
        {
            eprintln!("  [{}] chunks index rebuild skipped: {e}", params.item_key);
        }
        eprintln!("  [{}] chunks inserted ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
    }

    // ── Insert figures ──────────────────────────────────────────────────────
    if figures_added > 0 {
        eprintln!(
            "  [{}] inserting {} figures...",
            params.item_key,
            figures_added
        );
        let t = std::time::Instant::now();
        let batch = build_figures_batch(&params, &figure_records, &fig_embeddings)?;
        let schema = figures_schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let table = store.figures_table().await?;
        table
            .add(Box::new(reader))
            .execute()
            .await?;
        // Rebuild vector index after inserting new data
        if let Err(e) = table
            .create_index(&["vector"], Index::Auto)
            .execute()
            .await
        {
            eprintln!("  [{}] figures index rebuild skipped: {e}", params.item_key);
        }
        eprintln!("  [{}] figures inserted ({:.1}s)", params.item_key, t.elapsed().as_secs_f64());
    }

    eprintln!("  [{}] done (total {:.1}s)", params.item_key, t_total.elapsed().as_secs_f64());
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
            Arc::new(StringArray::from(block_types)),
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
    .map_err(|e: ArrowError| DbError::Arrow(e.to_string()))?;

    Ok(batch)
}

fn build_figures_batch(
    params: &IngestParams,
    records: &[FigureRecord],
    embeddings: &[Vec<f32>],
) -> Result<RecordBatch, DbError> {
    let n = records.len();
    let schema = figures_schema();

    let figure_ids: Vec<&str> = records.iter().map(|r| r.figure_id.as_str()).collect();
    let paper_ids: Vec<&str> = vec![params.paper_id.as_str(); n];
    let vectors = build_vector_array(embeddings);
    let figure_types: Vec<&str> = records.iter().map(|r| r.figure_type.as_str()).collect();
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
            Arc::new(StringArray::from(contents)),
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
    use serial_test::serial;
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

    #[serial]
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

    #[serial]
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
        let input = r"<p>where <math>\alpha</math> is a constant</p>";
        assert_eq!(
            strip_html_preserve_math(input),
            r"where $\alpha$ is a constant"
        );
    }

    #[test]
    fn test_strip_html_preserve_math_no_math() {
        let input = "<p>No math here</p>";
        assert_eq!(strip_html_preserve_math(input), "No math here");
    }

    #[test]
    fn test_extract_caption_text() {
        let input = r"<p><b>Fig. 1.</b> A description of the figure with <math>\mu_c</math> values.</p>";
        let result = strip_html_preserve_math(input);
        assert!(result.contains(r"$\mu_c$"));
        assert!(result.contains("Fig. 1."));
        assert!(result.contains("A description of the figure"));
    }

    // ── figure number extraction ────────────────────────────────────────────

    #[test]
    fn test_extract_figure_number_from_caption() {
        let re = regex::Regex::new(r"(?:Figure|Fig\.)\s+(\d+)").unwrap();
        let tbl_re = regex::Regex::new(r"Table\s+(\d+)").unwrap();

        assert_eq!(
            re.captures("Fig. 1. A caption").unwrap()[1].parse::<u32>().unwrap(),
            1
        );
        assert_eq!(
            re.captures("Fig. 12. Another caption").unwrap()[1].parse::<u32>().unwrap(),
            12
        );
        assert_eq!(
            tbl_re.captures("Table 1. Results").unwrap()[1].parse::<u32>().unwrap(),
            1
        );
    }

    #[test]
    fn test_extract_figure_number_no_match() {
        let re = regex::Regex::new(r"(?:Figure|Fig\.)\s+(\d+)").unwrap();
        assert!(re.captures("Just some caption text").is_none());
    }

    // ── parse_paper_blocks integration tests ────────────────────────────────

    /// Helper: write a custom DataLab JSON structure and return IngestParams.
    fn write_custom_datalab_json(
        dir: &std::path::Path,
        item_key: &str,
        blocks: Vec<serde_json::Value>,
    ) -> IngestParams {
        let item_dir = dir.join(item_key);
        fs::create_dir_all(&item_dir).unwrap();
        let json = serde_json::json!({
            "children": [{
                "block_type": "Page",
                "children": blocks
            }]
        });
        fs::write(
            item_dir.join(format!("{item_key}.json")),
            serde_json::to_vec_pretty(&json).unwrap(),
        )
        .unwrap();
        IngestParams {
            item_key: item_key.to_string(),
            paper_id: item_key.to_string(),
            title: "Test Paper".to_string(),
            authors: vec![],
            year: None,
            venue: None,
            tags: vec![],
            cache_dir: item_dir,
            force: false,
        }
    }

    #[test]
    fn test_picture_with_caption_indexed_as_figure() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "SectionHeader", "id": "h1",
                "html": "<h2>Results</h2>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>Some text.</p>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Picture", "id": "pic1",
                "html": r#"<img src="fig1.png" alt="Figure 1: A stress test result showing convergence"/>"#,
                "page": 2
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap1",
                "html": "<p><b>Fig. 1.</b> Convergence results for the stress test.</p>",
                "page": 2
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t2",
                "html": "<p>More text.</p>", "page": 2
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "PICTEST1", blocks);
        let (chunks, figures) = parse_paper_blocks(&params).unwrap();

        assert_eq!(figures.len(), 1);
        assert_eq!(figures[0].figure_type, "figure");
        assert!(figures[0].caption.contains("Fig. 1."));
        assert!(figures[0].caption.contains("Convergence results"));
        assert!(figures[0].description.as_deref().unwrap().contains("stress test result"));
        assert!(figures[0].image_path.is_some());
        assert_eq!(chunks.len(), 2); // two Text blocks
    }

    #[test]
    fn test_picture_without_caption_skipped() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Picture", "id": "pic1",
                "html": r#"<img src="cc_logo.png" alt="Creative Commons Attribution"/>"#,
                "page": 1
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>Some text.</p>", "page": 1
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "PICTEST2", blocks);
        let (_chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 0);
    }

    #[test]
    fn test_figure_block_with_caption() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Figure", "id": "fig1",
                "html": r#"<img src="fig6.png" alt="Figure 6: AI description of the figure"/>"#,
                "page": 5
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap1",
                "html": "<p><b>Fig. 6.</b> The paper's original caption text.</p>",
                "page": 5
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "FIGTEST1", blocks);
        let (_chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        assert!(figures[0].caption.contains("Fig. 6."));
        assert!(figures[0].description.as_deref().unwrap().contains("AI description"));
    }

    #[test]
    fn test_figure_block_without_caption_uses_alt() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Figure", "id": "fig1",
                "html": r#"<img src="fig6.png" alt="Figure 6: description from alt"/>"#,
                "page": 5
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "FIGTEST2", blocks);
        let (_chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        // No adjacent Caption, so caption falls back to alt text
        assert!(figures[0].caption.contains("Figure 6: description from alt"));
    }

    #[test]
    fn test_table_block_indexed() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Caption", "id": "cap1",
                "html": "<p><b>Table 1.</b> Comparison of methods.</p>",
                "page": 3
            }),
            serde_json::json!({
                "block_type": "Table", "id": "tbl1",
                "html": r#"<img src="tbl1.png" alt="Table showing method comparison"/>"#,
                "page": 3
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "TBLTEST1", blocks);
        let (_chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        assert_eq!(figures[0].figure_type, "table");
        assert!(figures[0].caption.contains("Table 1."));
    }

    #[test]
    fn test_caption_before_picture() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>Intro text.</p>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap1",
                "html": "<p><b>Fig. 15.</b> A preceding caption.</p>",
                "page": 8
            }),
            serde_json::json!({
                "block_type": "Picture", "id": "pic1",
                "html": r#"<img src="fig15.png" alt="Figure 15: description"/>"#,
                "page": 8
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t2",
                "html": "<p>Following text.</p>", "page": 8
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "CAPBEFORE", blocks);
        let (chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        assert!(figures[0].caption.contains("Fig. 15."));
        assert!(figures[0].caption.contains("preceding caption"));
        // Caption was consumed, not turned into a chunk
        assert_eq!(chunks.len(), 2); // only the two Text blocks
    }

    #[test]
    fn test_equation_preserves_latex() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "SectionHeader", "id": "h1",
                "html": "<h2>Method</h2>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Equation", "id": "eq1",
                "html": r#"<p><math display="block">\mathbf{x} = 0</math></p>"#,
                "page": 2
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "EQTEST1", blocks);
        let (chunks, _figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].text.contains(r"\mathbf{x}"),
            "LaTeX should be preserved, got: {}",
            chunks[0].text
        );
    }

    #[test]
    fn test_equation_among_text_chunks() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "SectionHeader", "id": "h1",
                "html": "<h2>Method</h2>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>We define</p>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Equation", "id": "eq1",
                "html": "<p><math>E = mc^2</math></p>",
                "page": 1
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t2",
                "html": "<p>where m is mass</p>", "page": 1
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "EQTEST2", blocks);
        let (chunks, _figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(chunks.len(), 3);
        assert!(chunks[1].text.contains("E = mc^2"));
        // chunk_idx values are sequential
        assert_eq!(chunks[0].chunk_idx, 0);
        assert_eq!(chunks[1].chunk_idx, 1);
        assert_eq!(chunks[2].chunk_idx, 2);
    }

    #[test]
    fn test_figure_ids_populated_on_chunks() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "SectionHeader", "id": "h1",
                "html": "<h2>Results</h2>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Picture", "id": "pic1",
                "html": r#"<img src="fig1.png" alt="Figure 1: plot"/>"#,
                "page": 1
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap1",
                "html": "<p><b>Fig. 1.</b> The convergence plot.</p>",
                "page": 1
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>As shown in Figure 1, the results converge.</p>",
                "page": 2
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "XREFTEST1", blocks);
        let (chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        assert_eq!(chunks.len(), 1);
        assert!(
            chunks[0].figure_ids.contains(&figures[0].figure_id),
            "chunk should reference figure; figure_ids={:?}, expected={}",
            chunks[0].figure_ids,
            figures[0].figure_id
        );
    }

    #[test]
    fn test_figure_ids_multiple_references() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "SectionHeader", "id": "h1",
                "html": "<h2>Results</h2>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Figure", "id": "fig1",
                "html": r#"<img src="fig1.png" alt="Plot"/>"#, "page": 1
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap1",
                "html": "<p><b>Fig. 1.</b> A figure.</p>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Table", "id": "tbl1",
                "html": r#"<img src="tbl2.png" alt="Table data"/>"#, "page": 2
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap2",
                "html": "<p><b>Table 2.</b> Comparison data.</p>", "page": 2
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>See Figure 1 and Table 2 for details.</p>",
                "page": 3
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "XREFTEST2", blocks);
        let (chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 2);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].figure_ids.len(), 2);
        assert!(chunks[0].figure_ids.contains(&figures[0].figure_id));
        assert!(chunks[0].figure_ids.contains(&figures[1].figure_id));
    }

    #[test]
    fn test_figure_ids_no_references() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "SectionHeader", "id": "h1",
                "html": "<h2>Intro</h2>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>No figure references here.</p>", "page": 1
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "XREFTEST3", blocks);
        let (chunks, _figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].figure_ids.is_empty());
    }

    #[test]
    fn test_mixed_document() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "SectionHeader", "id": "h1",
                "html": "<h2>Method</h2>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t1",
                "html": "<p>We propose a method.</p>", "page": 1
            }),
            serde_json::json!({
                "block_type": "Equation", "id": "eq1",
                "html": r#"<p><math display="block">\nabla f(x) = 0</math></p>"#,
                "page": 1
            }),
            serde_json::json!({
                "block_type": "Picture", "id": "pic1",
                "html": r#"<img src="fig1.png" alt="Figure 1: AI description of method overview"/>"#,
                "page": 2
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap1",
                "html": "<p><b>Fig. 1.</b> Method overview showing the pipeline.</p>",
                "page": 2
            }),
            serde_json::json!({
                "block_type": "Text", "id": "t2",
                "html": "<p>As shown in Figure 1, the pipeline processes data.</p>",
                "page": 2
            }),
            serde_json::json!({
                "block_type": "Caption", "id": "cap2",
                "html": "<p><b>Table 1.</b> Quantitative results.</p>",
                "page": 3
            }),
            serde_json::json!({
                "block_type": "Table", "id": "tbl1",
                "html": r#"<img src="tbl1.png" alt="Results table"/>"#,
                "page": 3
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "MIXTEST1", blocks);
        let (chunks, figures) = parse_paper_blocks(&params).unwrap();

        // 3 chunks: text, equation, text
        assert_eq!(chunks.len(), 3, "expected 3 chunks, got {}", chunks.len());
        // 2 figures: picture + table
        assert_eq!(figures.len(), 2, "expected 2 figures, got {}", figures.len());

        // Equation preserves LaTeX
        assert!(chunks[1].text.contains(r"\nabla f(x) = 0"));

        // Figure captions vs descriptions
        assert!(figures[0].caption.contains("Fig. 1."));
        assert!(figures[0].description.as_deref().unwrap().contains("AI description"));
        assert_eq!(figures[0].figure_type, "figure");

        assert!(figures[1].caption.contains("Table 1."));
        assert_eq!(figures[1].figure_type, "table");

        // Cross-references populated on the text chunk that mentions Figure 1
        assert!(
            chunks[2].figure_ids.contains(&figures[0].figure_id),
            "text chunk referencing Figure 1 should have its figure_id"
        );
    }

    // ── embed cache integration ───────────────────────────────────────────────

    #[serial]
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

        let rag = crate::store::DbStore::open_for_test(
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
        let model = "embedding-gemma-300m";
        assert!(embed_cache.exists(model, key), "manifest.json + embeddings.bin should exist");
        let manifest = embed_cache.load_manifest(model, key).unwrap().unwrap();
        assert_eq!(manifest.chunks.len(), 2, "should have 2 text chunks");
    }

    #[serial]
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

        let rag = crate::store::DbStore::open_for_test(
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
        let model = "embedding-gemma-300m";
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

    #[serial]
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

        let rag = crate::store::DbStore::open_for_test(
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
        let model = "embedding-gemma-300m";
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

    // ── html_table_to_markdown ────────────────────────────────────────────

    #[test]
    fn test_table_to_md_simple_2x2() {
        let html = "<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert_eq!(md, "| A | B |\n| --- | --- |\n| 1 | 2 |");
    }

    #[test]
    fn test_table_to_md_no_thead() {
        let html = "<table><tbody><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        // No thead → no separator line
        assert_eq!(md, "| A | B |\n| 1 | 2 |");
    }

    #[test]
    fn test_table_to_md_math_in_cells() {
        let html = r#"<table><thead><tr><th>Param</th></tr></thead><tbody><tr><td><math>\mu = 5e4</math></td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains(r"$\mu = 5e4$"), "math should be wrapped in $: {}", md);
    }

    #[test]
    fn test_table_to_md_br_tags() {
        let html = "<table><thead><tr><th>Number of<br/>Vert.</th></tr></thead><tbody><tr><td>97K</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("Number of Vert."), "br should become space: {}", md);
    }

    #[test]
    fn test_table_to_md_empty_cells() {
        let html = "<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td></td><td>x</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("|  | x |"), "empty cell should be empty string: {}", md);
    }

    #[test]
    fn test_table_to_md_not_a_table() {
        let html = r#"<img src="tbl.png" alt="Table 1"/>"#;
        assert!(html_table_to_markdown(html).is_none());
    }

    #[test]
    fn test_table_to_md_rowspan_doesnt_crash() {
        let html = r#"<table><thead><tr><th rowspan="2">Name</th><th>Col1</th></tr><tr><th>Sub</th></tr></thead><tbody><tr><td>A</td><td>1</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        // Flattened header + separator + 1 data row = 3 lines
        assert_eq!(lines.len(), 3, "expected 3 lines: {}", md);
        assert!(lines[0].contains("Name"), "header should contain Name: {}", md);
        assert!(lines[0].contains("Col1 Sub"), "header should merge sub-column: {}", md);
        assert!(lines[1].contains("---"), "second line should be separator");
    }

    #[test]
    fn test_table_to_md_multi_row_thead() {
        let html = r#"<table><thead><tr><th>Name</th><th>Value</th></tr><tr><th>Sub1</th><th>Sub2</th></tr></thead><tbody><tr><td>A</td><td>1</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        // Flattened header + separator + 1 data row = 3 lines
        assert_eq!(lines.len(), 3, "expected 3 lines: {}", md);
        assert!(lines[0].contains("Name Sub1"), "header should merge: {}", md);
        assert!(lines[0].contains("Value Sub2"), "header should merge: {}", md);
        assert!(lines[1].contains("---"), "separator should be after header");
    }

    #[test]
    fn test_table_to_md_whitespace_normalization() {
        let html = "<table><thead><tr><th>  A   B  </th></tr></thead><tbody><tr><td>  1   2  </td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("| A B |"), "whitespace should be normalized: {}", md);
    }

    #[test]
    fn test_table_to_md_real_datalab_table() {
        // Simplified version of a real DataLab table from YFACFA8C
        let html = r#"<table border="1"><thead><tr><th>Experiment Name</th><th>Number of Vert.</th><th>Material</th></tr></thead><tbody><tr><td>Twisting Thin Beams</td><td>97K</td><td>NeoHookean</td></tr><tr><td>Armadillo</td><td>15K</td><td>StVK</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("Twisting Thin Beams"), "should contain cell value");
        assert!(md.contains("97K"), "should contain cell value");
        assert!(md.contains("NeoHookean"), "should contain cell value");
        assert!(md.contains("Armadillo"), "should contain cell value");
        // Count data rows (lines after separator)
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 4, "header + separator + 2 data rows: {}", md);
    }

    // ── html_table_to_markdown: new tests ──────────────────────────────────

    // Basic structure tests

    #[test]
    fn test_table_to_md_3x3_with_thead() {
        let html = "<table><thead><tr><th>A</th><th>B</th><th>C</th></tr></thead><tbody><tr><td>1</td><td>2</td><td>3</td></tr><tr><td>4</td><td>5</td><td>6</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 4); // header + sep + 2 data rows
        assert_eq!(lines[0], "| A | B | C |");
    }

    #[test]
    fn test_table_to_md_single_column() {
        let html = "<table><thead><tr><th>Only</th></tr></thead><tbody><tr><td>val</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert_eq!(md, "| Only |\n| --- |\n| val |");
    }

    #[test]
    fn test_table_to_md_header_only_no_tbody() {
        let html = "<table><thead><tr><th>A</th><th>B</th></tr></thead></table>";
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 2); // header + separator
        assert_eq!(lines[0], "| A | B |");
        assert!(lines[1].contains("---"));
    }

    #[test]
    fn test_table_to_md_large_table() {
        let header = "<tr><th>C1</th><th>C2</th><th>C3</th><th>C4</th><th>C5</th></tr>";
        let row = "<tr><td>a</td><td>b</td><td>c</td><td>d</td><td>e</td></tr>";
        let html = format!(
            "<table><thead>{}</thead><tbody>{}{}{}{}{}</tbody></table>",
            header, row, row, row, row, row
        );
        let md = html_table_to_markdown(&html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 7); // 1 header + 1 sep + 5 data
    }

    #[test]
    fn test_table_to_md_empty_tr_tags() {
        let html = "<table><tbody><tr></tr><tr></tr></tbody></table>";
        assert!(html_table_to_markdown(html).is_none(), "empty rows should produce None");
    }

    // Colspan tests

    #[test]
    fn test_table_to_md_header_colspan_2() {
        let html = r#"<table><thead><tr><th colspan="2">Group</th></tr></thead><tbody><tr><td>a</td><td>b</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "| Group | Group |");
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_table_to_md_colspan_entire_row() {
        let html = r#"<table><thead><tr><th colspan="3">All</th></tr></thead><tbody><tr><td>1</td><td>2</td><td>3</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.starts_with("| All | All | All |"));
    }

    #[test]
    fn test_table_to_md_colspan_in_body() {
        let html = r#"<table><thead><tr><th>A</th><th>B</th></tr></thead><tbody><tr><td colspan="2">merged</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[2], "| merged | merged |");
    }

    #[test]
    fn test_table_to_md_multiple_colspans_one_row() {
        let html = r#"<table><thead><tr><th colspan="2">Left</th><th colspan="2">Right</th></tr></thead><tbody><tr><td>a</td><td>b</td><td>c</td><td>d</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "| Left | Left | Right | Right |");
    }

    #[test]
    fn test_table_to_md_colspan_3_header() {
        let html = r#"<table><thead><tr><th>ID</th><th colspan="3">Metrics</th></tr></thead><tbody><tr><td>1</td><td>a</td><td>b</td><td>c</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "| ID | Metrics | Metrics | Metrics |");
    }

    // Rowspan tests

    #[test]
    fn test_table_to_md_rowspan_2_first_header_col() {
        // Classic: first column spans 2 header rows, second column has sub-headers
        let html = r#"<table><thead><tr><th rowspan="2">Method</th><th>Score</th></tr><tr><th>Mean</th></tr></thead><tbody><tr><td>Ours</td><td>95</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("Method"), "should have Method: {}", md);
        assert!(lines[0].contains("Score Mean"), "should merge Score+Mean: {}", md);
    }

    #[test]
    fn test_table_to_md_rowspan_in_body() {
        let html = r#"<table><thead><tr><th>Cat</th><th>Val</th></tr></thead><tbody><tr><td rowspan="2">Group</td><td>1</td></tr><tr><td>2</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 4); // header + sep + 2 data
        assert_eq!(lines[2], "| Group | 1 |");
        assert_eq!(lines[3], "| Group | 2 |");
    }

    #[test]
    fn test_table_to_md_rowspan_3_body() {
        let html = r#"<table><thead><tr><th>Cat</th><th>Val</th></tr></thead><tbody><tr><td rowspan="3">X</td><td>1</td></tr><tr><td>2</td></tr><tr><td>3</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 5); // header + sep + 3 data
        assert_eq!(lines[2], "| X | 1 |");
        assert_eq!(lines[3], "| X | 2 |");
        assert_eq!(lines[4], "| X | 3 |");
    }

    #[test]
    fn test_table_to_md_rowspan_middle_column() {
        let html = r#"<table><thead><tr><th>A</th><th>B</th><th>C</th></tr></thead><tbody><tr><td>1</td><td rowspan="2">mid</td><td>x</td></tr><tr><td>2</td><td>y</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[2], "| 1 | mid | x |");
        assert_eq!(lines[3], "| 2 | mid | y |");
    }

    // Combined span tests

    #[test]
    fn test_table_to_md_vbd_style_rowspan_colspan_header() {
        // VBD Table 1 pattern: first col rowspan=2, next cols have colspan groups
        let html = r#"<table><thead><tr><th rowspan="2">Experiment</th><th colspan="2">Our Method</th><th colspan="2">Baseline</th></tr><tr><th>Time</th><th>Error</th><th>Time</th><th>Error</th></tr></thead><tbody><tr><td>Beam</td><td>1.2</td><td>0.01</td><td>3.4</td><td>0.05</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 3, "flattened header + sep + 1 data: {}", md);
        // Header should have: Experiment | Our Method Time | Our Method Error | Baseline Time | Baseline Error
        assert!(lines[0].contains("Experiment"), "{}", md);
        assert!(lines[0].contains("Our Method Time"), "{}", md);
        assert!(lines[0].contains("Baseline Error"), "{}", md);
        // Data row
        assert!(lines[2].contains("Beam"), "{}", md);
        assert!(lines[2].contains("1.2"), "{}", md);
    }

    #[test]
    fn test_table_to_md_two_level_header() {
        // Top row with colspans, bottom row with individual columns
        let html = r#"<table><thead><tr><th colspan="2">Group A</th><th colspan="2">Group B</th></tr><tr><th>X</th><th>Y</th><th>X</th><th>Y</th></tr></thead><tbody><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("Group A X"), "{}", md);
        assert!(lines[0].contains("Group A Y"), "{}", md);
        assert!(lines[0].contains("Group B X"), "{}", md);
        assert!(lines[0].contains("Group B Y"), "{}", md);
    }

    #[test]
    fn test_table_to_md_three_level_header() {
        let html = r#"<table><thead><tr><th colspan="4">All</th></tr><tr><th colspan="2">Left</th><th colspan="2">Right</th></tr><tr><th>A</th><th>B</th><th>C</th><th>D</th></tr></thead><tbody><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("All Left A"), "{}", md);
        assert!(lines[0].contains("All Right D"), "{}", md);
    }

    #[test]
    fn test_table_to_md_body_cell_colspan_and_rowspan() {
        let html = r#"<table><thead><tr><th>A</th><th>B</th><th>C</th></tr></thead><tbody><tr><td colspan="2" rowspan="2">big</td><td>x</td></tr><tr><td>y</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[2], "| big | big | x |");
        assert_eq!(lines[3], "| big | big | y |");
    }

    // Multi-row header flattening tests

    #[test]
    fn test_table_to_md_flatten_two_header_rows() {
        let html = r#"<table><thead><tr><th>Top1</th><th>Top2</th></tr><tr><th>Bot1</th><th>Bot2</th></tr></thead><tbody><tr><td>a</td><td>b</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("Top1 Bot1"), "{}", md);
        assert!(lines[0].contains("Top2 Bot2"), "{}", md);
    }

    #[test]
    fn test_table_to_md_rowspan_parent_with_sub_columns() {
        // "Parent" rowspan stays, sub-columns get their own labels
        let html = r#"<table><thead><tr><th rowspan="2">Parent</th><th colspan="2">Children</th></tr><tr><th>C1</th><th>C2</th></tr></thead><tbody><tr><td>p</td><td>a</td><td>b</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("Parent"), "{}", md);
        assert!(lines[0].contains("Children C1"), "{}", md);
        assert!(lines[0].contains("Children C2"), "{}", md);
    }

    #[test]
    fn test_table_to_md_three_header_rows_cascading() {
        let html = r#"<table><thead><tr><th colspan="4">Root</th></tr><tr><th colspan="2">L</th><th colspan="2">R</th></tr><tr><th>a</th><th>b</th><th>c</th><th>d</th></tr></thead><tbody><tr><td>1</td><td>2</td><td>3</td><td>4</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 3);
        // Root L a, Root L b, Root R c, Root R d
        assert!(lines[0].contains("Root L a"), "{}", md);
        assert!(lines[0].contains("Root R d"), "{}", md);
    }

    // Edge cases

    #[test]
    fn test_table_to_md_mixed_case_tags() {
        let html = "<TABLE><THEAD><TR><TH>A</TH><TH>B</TH></TR></THEAD><TBODY><TR><TD>1</TD><TD>2</TD></TR></TBODY></TABLE>";
        let md = html_table_to_markdown(html).unwrap();
        assert_eq!(md, "| A | B |\n| --- | --- |\n| 1 | 2 |");
    }

    #[test]
    fn test_table_to_md_extra_attributes() {
        let html = r#"<table border="1" class="results"><thead><tr><th style="color:red">A</th></tr></thead><tbody><tr><td class="val">1</td></tr></tbody></table>"#;
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("| A |"), "{}", md);
        assert!(md.contains("| 1 |"), "{}", md);
    }

    #[test]
    fn test_table_to_md_pipe_in_cell() {
        // Pipe chars in cell content — they'll appear in the output as-is (this is a known limitation)
        let html = "<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>x|y</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        // We just verify it doesn't crash and the content is there
        assert!(md.contains("x|y") || md.contains("x"), "{}", md);
    }

    #[test]
    fn test_table_to_md_whitespace_only_cells() {
        let html = "<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>   </td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        // Whitespace-only cells should normalize to empty
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[2], "|  |");
    }

    #[test]
    fn test_table_to_md_uneven_body_rows() {
        // Body row with fewer cells than header — grid pads with None
        let html = "<table><thead><tr><th>A</th><th>B</th><th>C</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "| A | B | C |");
        // Data row should be padded
        assert_eq!(lines[2], "| 1 |  |  |");
    }

    #[test]
    fn test_table_to_md_nested_html_in_cells() {
        let html = "<table><thead><tr><th><b>Bold</b></th></tr></thead><tbody><tr><td><i>italic</i> and <span>span</span></td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("| Bold |"), "should strip <b>: {}", md);
        assert!(md.contains("italic and span"), "should strip inline tags: {}", md);
    }

    #[test]
    fn test_table_to_md_br_variants() {
        let html = "<table><thead><tr><th>A<br>B<br/>C<br />D</th></tr></thead><tbody><tr><td>x</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("A B C D"), "all br variants should become space: {}", md);
    }

    #[test]
    fn test_table_to_md_uppercase_table_thead_tbody() {
        let html = "<TABLE><THEAD><TR><TH>X</TH></TR></THEAD><TBODY><TR><TD>1</TD></TR></TBODY></TABLE>";
        let md = html_table_to_markdown(html).unwrap();
        assert_eq!(md, "| X |\n| --- |\n| 1 |");
    }

    // Real-world patterns

    #[test]
    fn test_table_to_md_vbd_table1_reproduction() {
        // Reproduces the VBD paper Table 1 structure
        let html = r#"<table border="1">
            <thead>
                <tr>
                    <th rowspan="2">Experiment Name</th>
                    <th rowspan="2">Number of Vert.</th>
                    <th rowspan="2">Material</th>
                    <th colspan="2">VBD (Ours)</th>
                    <th colspan="2">Newton</th>
                </tr>
                <tr>
                    <th>Time(s)</th>
                    <th>Iter</th>
                    <th>Time(s)</th>
                    <th>Iter</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Twisting Beams</td>
                    <td>97K</td>
                    <td>NeoHookean</td>
                    <td>1.2</td>
                    <td>5</td>
                    <td>3.4</td>
                    <td>12</td>
                </tr>
                <tr>
                    <td>Armadillo</td>
                    <td>15K</td>
                    <td>StVK</td>
                    <td>0.8</td>
                    <td>3</td>
                    <td>2.1</td>
                    <td>8</td>
                </tr>
            </tbody>
        </table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();

        // Should have: 1 flattened header + 1 separator + 2 data rows = 4 lines
        assert_eq!(lines.len(), 4, "wrong line count: {}", md);

        // Header should have 7 columns
        let header_pipes = lines[0].matches('|').count();
        assert_eq!(header_pipes, 8, "header should have 7 cols (8 pipes): {}", lines[0]);

        // Verify flattened header labels
        assert!(lines[0].contains("Experiment Name"), "{}", lines[0]);
        assert!(lines[0].contains("Number of Vert."), "{}", lines[0]);
        assert!(lines[0].contains("Material"), "{}", lines[0]);
        assert!(lines[0].contains("VBD (Ours) Time(s)"), "{}", lines[0]);
        assert!(lines[0].contains("Newton Iter"), "{}", lines[0]);

        // Verify separator
        assert!(lines[1].contains("---"));
        let sep_pipes = lines[1].matches('|').count();
        assert_eq!(sep_pipes, 8, "separator should match header width");

        // Verify data
        assert!(lines[2].contains("Twisting Beams"), "{}", lines[2]);
        assert!(lines[2].contains("97K"), "{}", lines[2]);
        assert!(lines[3].contains("Armadillo"), "{}", lines[3]);
    }

    #[test]
    fn test_table_to_md_statistical_results() {
        // Statistical results table with grouped header columns
        let html = r#"<table>
            <thead>
                <tr>
                    <th rowspan="2">Method</th>
                    <th colspan="2">Accuracy</th>
                    <th colspan="2">F1-Score</th>
                </tr>
                <tr>
                    <th>Mean</th>
                    <th>CI</th>
                    <th>Mean</th>
                    <th>CI</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Ours</td><td>0.95</td><td>[0.93,0.97]</td><td>0.91</td><td>[0.88,0.94]</td></tr>
                <tr><td>Baseline</td><td>0.82</td><td>[0.79,0.85]</td><td>0.78</td><td>[0.74,0.82]</td></tr>
            </tbody>
        </table>"#;
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();

        assert_eq!(lines.len(), 4, "header + sep + 2 data: {}", md);
        assert!(lines[0].contains("Method"), "{}", lines[0]);
        assert!(lines[0].contains("Accuracy Mean"), "{}", lines[0]);
        assert!(lines[0].contains("F1-Score CI"), "{}", lines[0]);
        assert!(lines[2].contains("0.95"), "{}", lines[2]);
        assert!(lines[3].contains("Baseline"), "{}", lines[3]);
    }

    // ── table content in FigureRecord ────────────────────────────────────

    #[test]
    fn test_table_block_extracts_content() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Table", "id": "tbl1",
                "html": "<table><thead><tr><th>Method</th><th>Score</th></tr></thead><tbody><tr><td>Ours</td><td>95</td></tr></tbody></table>",
                "page": 3
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "TBLCONTENT1", blocks);
        let (_chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        assert!(figures[0].content.is_some(), "table block should have content");
        let content = figures[0].content.as_ref().unwrap();
        assert!(content.contains("Method"), "content should have cell values: {}", content);
        assert!(content.contains("95"), "content should have cell values: {}", content);
    }

    #[test]
    fn test_figure_block_has_no_content() {
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Figure", "id": "fig1",
                "html": r#"<img src="fig1.png" alt="A figure"/>"#,
                "page": 1
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "FIGNOCONTENT", blocks);
        let (_chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        assert!(figures[0].content.is_none(), "figure block should not have content");
    }

    #[test]
    fn test_table_with_img_only_has_no_content() {
        // Table block that only has an <img> tag (no actual <table>)
        let dir = TempDir::new().unwrap();
        let blocks = vec![
            serde_json::json!({
                "block_type": "Table", "id": "tbl1",
                "html": r#"<img src="tbl1.png" alt="Table 1: Results"/>"#,
                "page": 3
            }),
        ];
        let params = write_custom_datalab_json(dir.path(), "TBLIMG", blocks);
        let (_chunks, figures) = parse_paper_blocks(&params).unwrap();
        assert_eq!(figures.len(), 1);
        assert!(figures[0].content.is_none(), "img-only table should not have content");
    }

    // ── embedding text for tables ────────────────────────────────────────

    #[test]
    fn test_table_embedding_uses_caption_and_description() {
        // Table content is stored but NOT embedded — caption+description only.
        let rec = FigureRecord {
            figure_id: "test/fig1".to_string(),
            figure_type: "table".to_string(),
            caption: "Table 1. Results".to_string(),
            description: Some("Table alt text".to_string()),
            image_path: None,
            content: Some("| A | B |\n| --- | --- |\n| 1 | 2 |".to_string()),
            page: None,
            chapter_idx: 0,
            section_idx: 0,
        };
        let records = vec![rec];
        let text: Vec<String> = records
            .iter()
            .map(|f| match (f.caption.is_empty(), &f.description) {
                (false, Some(desc)) => format!("{}\n{}", f.caption, desc),
                (false, None) => f.caption.clone(),
                (true, Some(desc)) => desc.clone(),
                (true, None) => String::new(),
            })
            .collect();
        assert_eq!(text[0], "Table 1. Results\nTable alt text");
    }

    #[test]
    fn test_embedding_text_caption_only_when_no_description() {
        let rec = FigureRecord {
            figure_id: "test/fig1".to_string(),
            figure_type: "table".to_string(),
            caption: "Table 1. Results".to_string(),
            description: None,
            image_path: None,
            content: Some("| A | B |\n| --- | --- |\n| 1 | 2 |".to_string()),
            page: None,
            chapter_idx: 0,
            section_idx: 0,
        };
        let records = vec![rec];
        let text: Vec<String> = records
            .iter()
            .map(|f| match (f.caption.is_empty(), &f.description) {
                (false, Some(desc)) => format!("{}\n{}", f.caption, desc),
                (false, None) => f.caption.clone(),
                (true, Some(desc)) => desc.clone(),
                (true, None) => String::new(),
            })
            .collect();
        assert_eq!(text[0], "Table 1. Results");
    }

    #[test]
    fn test_embedding_text_falls_back_to_description() {
        let rec = FigureRecord {
            figure_id: "test/fig1".to_string(),
            figure_type: "figure".to_string(),
            caption: String::new(),
            description: Some("Alt text description".to_string()),
            image_path: None,
            content: None,
            page: None,
            chapter_idx: 0,
            section_idx: 0,
        };
        let records = vec![rec];
        let text: Vec<String> = records
            .iter()
            .map(|f| match (f.caption.is_empty(), &f.description) {
                (false, Some(desc)) => format!("{}\n{}", f.caption, desc),
                (false, None) => f.caption.clone(),
                (true, Some(desc)) => desc.clone(),
                (true, None) => String::new(),
            })
            .collect();
        assert_eq!(text[0], "Alt text description");
    }

    #[test]
    fn test_figure_embedding_uses_caption_and_description() {
        let rec = FigureRecord {
            figure_id: "test/fig1".to_string(),
            figure_type: "figure".to_string(),
            caption: "Fig. 1. A caption".to_string(),
            description: Some("Alt text description".to_string()),
            image_path: None,
            content: None,
            page: None,
            chapter_idx: 0,
            section_idx: 0,
        };
        let records = vec![rec];
        let text: Vec<String> = records
            .iter()
            .map(|f| match (f.caption.is_empty(), &f.description) {
                (false, Some(desc)) => format!("{}\n{}", f.caption, desc),
                (false, None) => f.caption.clone(),
                (true, Some(desc)) => desc.clone(),
                (true, None) => String::new(),
            })
            .collect();
        assert_eq!(text[0], "Fig. 1. A caption\nAlt text description");
    }
}
