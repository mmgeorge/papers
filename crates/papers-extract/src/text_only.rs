//! Text-only extraction: build text block regions from the PDF text layer
//! using geometric heuristics, without any ML model inference.

use crate::headings::DetectedHeading;
use crate::pdf::PdfChar;
use crate::reading_order;
use crate::text;
use crate::types::{Region, RegionKind};

/// Margin threshold in PDF points — chars within this distance of the page
/// top/bottom edge are filtered as running headers/footers.
const MARGIN_PT: f32 = 25.0;

/// Minimum alphanumeric character ratio — blocks below this are likely
/// formula fragments from overlapping PDF text layer streams.
const MIN_ALPHA_RATIO: f32 = 0.3;

/// Minimum text length for a block to be kept (after trimming).
const MIN_TEXT_LEN: usize = 2;

/// A character with image-space coordinates and font metadata.
struct ImgChar {
    /// Bounding box in image space (Y-down): [x1, y1, x2, y2].
    bbox: [f32; 4],
    font_size: f32,
    is_bold: bool,
}

/// A line of characters grouped by Y-proximity.
struct Line {
    y_center: f32,
    x_min: f32,
    x_max: f32,
    /// Dominant (median) font size of chars in this line.
    font_size: f32,
    /// Whether the majority of chars are bold.
    is_bold: bool,
    /// Bounding box encompassing all chars in this line.
    bbox: [f32; 4],
}

/// A text block: a group of consecutive lines forming a paragraph or heading.
struct Block {
    bbox: [f32; 4],
    lines: std::ops::Range<usize>,
    font_size: f32,
    is_bold: bool,
    /// Whether this is the topmost block on the page (candidate running header).
    is_topmost: bool,
}

/// Extract text blocks from a page's PDF text layer using geometric heuristics.
///
/// `headings` should be pre-filtered to only include `DetectedHeading`s for this page.
pub fn extract_page_text_blocks(
    chars: &[PdfChar],
    page_height_pt: f32,
    _page_width_pt: f32,
    page_idx: u32,
    headings: &[&DetectedHeading],
) -> Vec<Region> {
    // Step 1: Convert to image space + filter margins
    let img_chars = convert_and_filter(chars, page_height_pt);
    if img_chars.is_empty() {
        return Vec::new();
    }

    // Step 2: Group chars into lines
    let lines = group_into_lines(&img_chars);
    if lines.is_empty() {
        return Vec::new();
    }

    // Step 3: Group lines into text blocks
    let mut blocks = group_into_blocks(&lines);
    if blocks.is_empty() {
        return Vec::new();
    }

    // Mark the topmost block (candidate running header)
    if !blocks.is_empty() {
        let topmost_idx = blocks
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.bbox[1]
                    .partial_cmp(&b.bbox[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap();
        blocks[topmost_idx].is_topmost = true;
    }

    // Step 4: Extract text + classify + build regions
    let mut regions = Vec::with_capacity(blocks.len());
    let page_median_font = median_font_size(&lines);

    for (block_idx, block) in blocks.iter().enumerate() {
        // Extract text using the existing text pipeline
        let extracted = text::extract_region_text(
            chars,
            block.bbox,
            page_height_pt,
            &[],
            &[],
            text::AssemblyMode::Reflow,
        );
        let trimmed = extracted.trim();

        // Skip empty or too-short blocks
        if trimmed.len() < MIN_TEXT_LEN {
            continue;
        }

        // Skip formula fragments: short blocks with low alphanumeric ratio
        if is_formula_fragment(trimmed) {
            continue;
        }

        // Classify: heading, caption, or text
        let kind = classify_block(
            block,
            trimmed,
            headings,
            page_height_pt,
            page_median_font,
        );

        let id = format!("p{}_{}", page_idx + 1, block_idx);
        regions.push(Region {
            id,
            kind,
            bbox: block.bbox,
            confidence: 1.0,
            order: 0,
            text: Some(extracted),
            html: None,
            latex: None,
            image_path: None,
            caption: None,
            chart_type: None,
            tag: None,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
        });
    }

    // Step 5: Reading order via XY-Cut
    if regions.len() > 1 {
        reading_order::xy_cut_order(&mut regions);
        regions.sort_by_key(|r| r.order);
    }

    regions
}

/// Detect formula fragments — short blocks with mostly non-alphanumeric chars
/// that come from overlapping PDF text layer streams for mathematical content.
fn is_formula_fragment(text: &str) -> bool {
    let total = text.chars().count();
    if total > 80 {
        return false;
    }

    // Count meaningful word characters (letters in words ≥3 chars)
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_letter_count: usize = words
        .iter()
        .filter(|w| w.len() >= 3 && w.chars().any(|c| c.is_alphabetic()))
        .map(|w| w.len())
        .sum();
    let word_ratio = word_letter_count as f32 / total.max(1) as f32;

    // Very short blocks with few real words are likely formula debris
    if total <= 15 {
        return word_ratio < 0.4;
    }

    // Short blocks need reasonable word content
    word_ratio < MIN_ALPHA_RATIO
}

// ── Step 1: Convert + filter ────────────────────────────────────────

fn convert_and_filter(chars: &[PdfChar], page_height_pt: f32) -> Vec<ImgChar> {
    chars
        .iter()
        .filter_map(|c| {
            // Skip control characters and zero-width chars
            if c.codepoint.is_control() || c.codepoint == '\u{FEFF}' {
                return None;
            }

            // Convert to image space (Y-down)
            let y1 = page_height_pt - c.bbox[3]; // top
            let y2 = page_height_pt - c.bbox[1]; // bottom
            let x1 = c.bbox[0];
            let x2 = c.bbox[2];

            // Skip degenerate bboxes
            if (x2 - x1).abs() < 0.1 || (y2 - y1).abs() < 0.1 {
                return None;
            }

            let cy = (y1 + y2) / 2.0;

            // Filter margins (headers/footers)
            if cy < MARGIN_PT || cy > page_height_pt - MARGIN_PT {
                return None;
            }

            Some(ImgChar {
                bbox: [x1, y1, x2, y2],
                font_size: c.font_size,
                is_bold: c.is_bold,
            })
        })
        .collect()
}

// ── Step 2: Group chars into lines ──────────────────────────────────

fn group_into_lines(chars: &[ImgChar]) -> Vec<Line> {
    if chars.is_empty() {
        return Vec::new();
    }

    // Sort by Y then X
    let mut sorted: Vec<usize> = (0..chars.len()).collect();
    sorted.sort_by(|&a, &b| {
        let ay = (chars[a].bbox[1] + chars[a].bbox[3]) / 2.0;
        let by = (chars[b].bbox[1] + chars[b].bbox[3]) / 2.0;
        ay.partial_cmp(&by)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                chars[a].bbox[0]
                    .partial_cmp(&chars[b].bbox[0])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    // Compute average char height for Y-proximity threshold
    let avg_height: f32 =
        chars.iter().map(|c| c.bbox[3] - c.bbox[1]).sum::<f32>() / chars.len() as f32;
    let y_threshold = avg_height * 0.5;

    let mut lines = Vec::new();
    let mut line_start = 0;
    let mut current_y = (chars[sorted[0]].bbox[1] + chars[sorted[0]].bbox[3]) / 2.0;

    for i in 1..sorted.len() {
        let cy = (chars[sorted[i]].bbox[1] + chars[sorted[i]].bbox[3]) / 2.0;
        if (cy - current_y).abs() > y_threshold {
            lines.push(build_line(chars, &sorted[line_start..i]));
            line_start = i;
            current_y = cy;
        }
    }
    lines.push(build_line(chars, &sorted[line_start..]));

    lines
}

fn build_line(chars: &[ImgChar], indices: &[usize]) -> Line {
    let mut x_min = f32::MAX;
    let mut x_max = f32::MIN;
    let mut y_min = f32::MAX;
    let mut y_max = f32::MIN;
    let mut y_sum = 0.0f32;
    let mut font_sizes: Vec<f32> = Vec::with_capacity(indices.len());
    let mut bold_count = 0usize;

    for &i in indices {
        let c = &chars[i];
        x_min = x_min.min(c.bbox[0]);
        x_max = x_max.max(c.bbox[2]);
        y_min = y_min.min(c.bbox[1]);
        y_max = y_max.max(c.bbox[3]);
        y_sum += (c.bbox[1] + c.bbox[3]) / 2.0;
        font_sizes.push(c.font_size);
        if c.is_bold {
            bold_count += 1;
        }
    }

    font_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_font = font_sizes[font_sizes.len() / 2];

    Line {
        y_center: y_sum / indices.len() as f32,
        x_min,
        x_max,
        font_size: median_font,
        is_bold: bold_count > indices.len() / 2,
        bbox: [x_min, y_min, x_max, y_max],
    }
}

// ── Step 3: Group lines into blocks ─────────────────────────────────

fn group_into_blocks(lines: &[Line]) -> Vec<Block> {
    if lines.is_empty() {
        return Vec::new();
    }

    // Compute median line spacing
    let spacings: Vec<f32> = lines
        .windows(2)
        .map(|pair| (pair[1].y_center - pair[0].y_center).abs())
        .collect();
    let median_spacing = if spacings.is_empty() {
        0.0
    } else {
        let mut sorted = spacings.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };
    let para_threshold = median_spacing * 1.5;

    // Estimate avg char width for X-overlap tolerance
    let avg_line_width: f32 =
        lines.iter().map(|l| l.x_max - l.x_min).sum::<f32>() / lines.len() as f32;
    let avg_char_width = avg_line_width / 40.0; // rough estimate

    let mut blocks = Vec::new();
    let mut block_start = 0;

    for i in 1..lines.len() {
        let prev = &lines[i - 1];
        let curr = &lines[i];

        let y_gap = (curr.y_center - prev.y_center).abs();
        let y_break = para_threshold > 0.0 && y_gap > para_threshold;

        // X-range non-overlap: columns
        let x_overlap = prev.x_max.min(curr.x_max) - prev.x_min.max(curr.x_min);
        let x_break = x_overlap < -avg_char_width;

        // Font size jump
        let font_ratio = if prev.font_size > 0.0 && curr.font_size > 0.0 {
            (prev.font_size / curr.font_size).max(curr.font_size / prev.font_size)
        } else {
            1.0
        };
        let font_break = font_ratio > 1.3;

        if y_break || x_break || font_break {
            blocks.push(build_block(lines, block_start..i));
            block_start = i;
        }
    }
    blocks.push(build_block(lines, block_start..lines.len()));

    blocks
}

fn build_block(lines: &[Line], range: std::ops::Range<usize>) -> Block {
    let mut bbox = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    let mut font_sizes: Vec<f32> = Vec::new();
    let mut bold_count = 0usize;
    let line_count = range.len();

    for line in &lines[range.clone()] {
        bbox[0] = bbox[0].min(line.bbox[0]);
        bbox[1] = bbox[1].min(line.bbox[1]);
        bbox[2] = bbox[2].max(line.bbox[2]);
        bbox[3] = bbox[3].max(line.bbox[3]);
        font_sizes.push(line.font_size);
        if line.is_bold {
            bold_count += 1;
        }
    }

    // Pad bbox by 1pt
    bbox[0] -= 1.0;
    bbox[1] -= 1.0;
    bbox[2] += 1.0;
    bbox[3] += 1.0;

    font_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_font = font_sizes[font_sizes.len() / 2];

    Block {
        bbox,
        lines: range,
        font_size: median_font,
        is_bold: bold_count > line_count / 2,
        is_topmost: false,
    }
}

// ── Step 4: Classification ──────────────────────────────────────────

fn classify_block(
    block: &Block,
    text: &str,
    headings: &[&DetectedHeading],
    page_height_pt: f32,
    page_median_font: f32,
) -> RegionKind {
    let line_count = block.lines.len();

    // Check against detected headings (font-based, from headings::extract_headings)
    let block_pdf_y = page_height_pt - (block.bbox[1] + block.bbox[3]) / 2.0;
    let avg_line_height = (block.bbox[3] - block.bbox[1]) / line_count.max(1) as f32;

    for heading in headings {
        // Match by Y-position (heading.y_center is in PDF Y-up space)
        if (heading.y_center - block_pdf_y).abs() > avg_line_height * 1.5 {
            continue;
        }
        // Verify text match — must be exact or near-exact, not just prefix
        if titles_match_strict(&heading.title, text) {
            // Running header check: topmost single-line block whose text
            // matches a heading but has extra content (page number) appended
            if block.is_topmost && line_count == 1 && !titles_match_exact(&heading.title, text) {
                // This is a running header, not a real heading
                return RegionKind::Text;
            }
            return RegionKind::ParagraphTitle;
        }
    }

    // Caption detection via text patterns
    let lower = text.trim_start().to_lowercase();
    if is_caption_text(&lower) {
        if lower.starts_with("table") || lower.starts_with("tab.") || lower.starts_with("tab ") {
            return RegionKind::TableTitle;
        }
        return RegionKind::FigureTitle;
    }

    // Supplementary heading heuristic: only font-size based, NOT bold-only.
    // Bold-only heuristic causes too many false positives (running headers,
    // bold labels in text, theorem names, etc.)
    if line_count <= 2 && block.font_size > page_median_font * 1.2 {
        let char_count = text.chars().count();
        if char_count < 120 && char_count > 2 {
            // Reject if topmost single-line block (likely running header)
            if !(block.is_topmost && line_count == 1) {
                return RegionKind::ParagraphTitle;
            }
        }
    }

    RegionKind::Text
}

/// Strict title match: heading text must match block text closely.
/// Returns true if they match (ignoring case/whitespace), with at most
/// a small suffix allowed (e.g., trailing equation number).
fn titles_match_strict(heading_title: &str, block_text: &str) -> bool {
    let h = normalize_title(heading_title);
    let b = normalize_title(block_text);

    if h.is_empty() || b.is_empty() {
        return false;
    }

    // Exact match
    if h == b {
        return true;
    }

    // Block text starts with heading title
    if b.starts_with(&h) {
        // Check what's after the heading text — allow only whitespace + numbers
        // (trailing page number or equation number in running headers)
        let suffix = b[h.len()..].trim();
        if suffix.is_empty() || suffix.chars().all(|c| c.is_ascii_digit() || c == '.') {
            return true;
        }
    }

    // Heading title starts with page-number prefix, then matches block
    // e.g., heading = "1 Introduction", block = "4 1 Introduction"
    // → running header with "4" (page number) prepended. Don't match.

    false
}

/// Exact title match (no suffix tolerance).
fn titles_match_exact(heading_title: &str, block_text: &str) -> bool {
    normalize_title(heading_title) == normalize_title(block_text)
}

fn normalize_title(s: &str) -> String {
    // Strip leading/trailing markdown bold markers
    let s = s.trim().trim_start_matches("**").trim_end_matches("**").trim();
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Check if text looks like a caption ("Figure N", "Table N", etc.)
fn is_caption_text(lower: &str) -> bool {
    let prefixes = [
        "figure ", "fig.", "fig ",
        "table ", "tab.", "tab ",
        "algorithm ", "alg.", "alg ",
    ];
    for prefix in &prefixes {
        if lower.starts_with(prefix) {
            // Must have a digit somewhere after the prefix
            if lower[prefix.len()..].trim_start().starts_with(|c: char| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    false
}

// ── Cross-page running header filter ────────────────────────────────

/// Filter running headers across pages.
///
/// Running headers are ParagraphTitle regions that appear at the top of many
/// pages with text that matches (or nearly matches) the same heading. These
/// are page-level headers printed by the publisher (e.g., "1.1 Mathematical
/// optimization 3" on every page of section 1.1), not real headings.
///
/// Detection: for each page, find the topmost ParagraphTitle. Strip trailing
/// digits (page numbers) and collect normalized text. If the same text appears
/// on >3 pages, demote all instances to Text.
pub fn filter_running_headers(pages: &mut [crate::types::Page]) {
    use std::collections::HashMap;

    if pages.len() < 5 {
        return;
    }

    // Collect: (normalized_header_text → list of (page_idx, region_idx))
    let mut header_occurrences: HashMap<String, Vec<(usize, usize)>> = HashMap::new();

    for (page_i, page) in pages.iter().enumerate() {
        // Find topmost ParagraphTitle on this page
        let topmost = page
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| r.kind == RegionKind::ParagraphTitle)
            .min_by(|(_, a), (_, b)| {
                a.bbox[1]
                    .partial_cmp(&b.bbox[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some((region_i, r)) = topmost {
            let text = r.text.as_deref().unwrap_or("");
            let key = normalize_running_header(text);
            if !key.is_empty() {
                header_occurrences
                    .entry(key)
                    .or_default()
                    .push((page_i, region_i));
            }
        }
    }

    // Demote headers that appear on >3 pages (clearly running headers)
    for (_key, occurrences) in &header_occurrences {
        if occurrences.len() > 3 {
            for &(page_i, region_i) in occurrences {
                pages[page_i].regions[region_i].kind = RegionKind::Text;
            }
        }
    }
}

/// Normalize a running header: strip bold markers, trailing page numbers,
/// and collapse whitespace.
fn normalize_running_header(text: &str) -> String {
    let s = text
        .trim()
        .trim_start_matches("**")
        .trim_end_matches("**")
        .trim();
    // Strip trailing digits (page number)
    let s = s.trim_end_matches(|c: char| c.is_ascii_digit() || c == ' ');
    // Strip leading digits (page number on left-page headers like "4 1 Introduction")
    let s = s.trim_start_matches(|c: char| c.is_ascii_digit() || c == ' ');
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn median_font_size(lines: &[Line]) -> f32 {
    if lines.is_empty() {
        return 10.0;
    }
    let mut sizes: Vec<f32> = lines.iter().map(|l| l.font_size).collect();
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sizes[sizes.len() / 2]
}
