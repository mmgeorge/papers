//! Table of Contents parser — extract heading hierarchy from TOC pages.
//!
//! Anchors on page numbers (the one universal signal in all TOC formats),
//! uses easy pattern matches (CHAPTER N, N.N) to learn font signatures,
//! then applies learned signatures to classify ambiguous cases.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::pdf::is_bold_font;

use crate::headings;
use crate::pdf::PdfChar;

// ── Output types ──

/// A single entry in the parsed Table of Contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    /// Heading depth: 0 = Part, 1 = Chapter, 2 = Section, 3 = Subsection, etc.
    pub depth: u32,
    /// Title text (cleaned: no leader dots, no page number).
    pub title: String,
    /// Page number string as printed (e.g., "42", "xvii").
    pub page_label: String,
    /// Resolved page value: arabic as positive, roman as negative.
    pub page_value: i32,
    /// Classification kind.
    pub kind: TocEntryKind,
}

/// Classification of a TOC entry's structural level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TocEntryKind {
    Part,
    Chapter,
    Section,
    Subsection,
    FrontMatter,
    BackMatter,
    Appendix,
    SubEntry,
}

/// Complete parsed Table of Contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocResult {
    /// The TOC entries in document order.
    pub entries: Vec<TocEntry>,
    /// Which PDF pages (0-indexed) were identified as TOC pages.
    pub toc_pages: Vec<u32>,
    /// Font signatures learned for each depth level.
    pub font_profile: TocFontProfile,
}

/// Learned font properties per heading depth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocFontProfile {
    pub levels: Vec<TocFontLevel>,
}

/// Font characteristics learned for a specific depth level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocFontLevel {
    pub depth: u32,
    pub font: String,
    pub size: f32,
    pub is_bold: bool,
    pub is_all_caps: bool,
    pub example: String,
}

// ── Internal types ──

/// A character with metadata, in image-space coordinates (Y-down).
#[derive(Debug, Clone)]
struct TocChar {
    codepoint: char,
    /// [x1, y1, x2, y2] in image-space (Y-down).
    bbox: [f32; 4],
    space_threshold: f32,
    font_name: String,
    font_size: f32,
    is_italic: bool,
}

/// A single line of text extracted from a TOC page.
#[derive(Debug, Clone)]
struct TocRawLine {
    chars: Vec<TocChar>,
    /// X-positions of pdfium literal space characters in this line (image-space).
    /// Used as supplementary word-boundary signals for tightly-kerned fonts.
    pdfium_space_xs: Vec<f32>,
    y_center: f32,
    page_idx: u32,
}

impl TocRawLine {
    fn text(&self) -> String {
        let mut result = String::new();
        for (i, ch) in self.chars.iter().enumerate() {
            if i > 0 {
                let prev = &self.chars[i - 1];
                let gap = ch.bbox[0] - prev.bbox[2];
                let h = (ch.bbox[3] - ch.bbox[1]).abs();
                let height_floor = if h > 0.5 { h * 0.25 } else { 2.0 };
                let threshold = if ch.space_threshold > 0.1 {
                    ch.space_threshold.max(height_floor)
                } else {
                    height_floor
                };
                // Insert space if gap exceeds threshold OR a pdfium space
                // exists between these two chars (for tight-kerned fonts where
                // the physical gap is below our threshold).
                // Guard: only trust a pdfium space if the physical gap is at
                // least 15% of the threshold — some PDFs have space chars at
                // wrong x-positions that land between adjacent glyphs.
                let has_pdfium_space = gap >= threshold * 0.15
                    && self.pdfium_space_xs.iter().any(|&sx| {
                        sx >= prev.bbox[2] - 0.5 && sx <= ch.bbox[0] + 0.5
                    });
                if gap > threshold || has_pdfium_space {
                    result.push(' ');
                }
            }
            result.push(ch.codepoint);
        }
        result
    }

    fn x_left(&self) -> f32 {
        self.chars.first().map(|c| c.bbox[0]).unwrap_or(0.0)
    }

    fn x_right(&self) -> f32 {
        self.chars.last().map(|c| c.bbox[2]).unwrap_or(0.0)
    }
}

/// A line after title / page-number separation.
#[derive(Debug, Clone)]
struct TocSplitLine {
    title: String,
    page_label: Option<String>,
    page_value: Option<i32>,
    font_sig: FontSig,
    x_left: f32,
    page_idx: u32,
    y_center: f32,
    has_leader_dots: bool,
    /// Font chars for the title portion (for signature learning).
    title_chars: Vec<TocChar>,
}

/// Font signature for a TOC line's title portion.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FontSig {
    family: String,
    size_bucket: i32,
    is_bold: bool,
    is_italic: bool,
    is_all_caps: bool,
}

impl FontSig {
    fn from_chars(chars: &[TocChar]) -> Self {
        if chars.is_empty() {
            return Self {
                family: String::new(),
                size_bucket: 0,
                is_bold: false,
                is_italic: false,
                is_all_caps: false,
            };
        }

        // Use the most common font among the title chars
        let mut font_counts: HashMap<String, usize> = HashMap::new();
        let mut size_sum = 0.0f32;
        let mut size_count = 0usize;
        let mut any_italic = false;

        for ch in chars {
            if ch.font_name.is_empty() || ch.font_size < 0.0 {
                continue;
            }
            *font_counts.entry(ch.font_name.clone()).or_default() += 1;
            size_sum += ch.font_size;
            size_count += 1;
            if ch.is_italic {
                any_italic = true;
            }
        }

        let dominant_font = font_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name)
            .unwrap_or_default();

        let avg_size = if size_count > 0 {
            size_sum / size_count as f32
        } else {
            0.0
        };

        let family = headings::normalize_font_family(&dominant_font);
        let is_bold = is_bold_font(&dominant_font);

        // Check ALL CAPS: all alphabetic chars are uppercase
        let alpha_chars: Vec<char> = chars
            .iter()
            .map(|c| c.codepoint)
            .filter(|c| c.is_alphabetic())
            .collect();
        let is_all_caps = !alpha_chars.is_empty() && alpha_chars.iter().all(|c| c.is_uppercase());

        Self {
            family,
            size_bucket: (avg_size * 2.0).round() as i32,
            is_bold,
            is_italic: any_italic,
            is_all_caps,
        }
    }
}

/// Classification assigned during pattern matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Classification {
    Part,
    Chapter,
    Section,
    Subsection,
    SubSubsection,
    FrontMatter,
    BackMatter,
    SubEntry,
    Unknown,
}

impl Classification {
    fn to_kind(self) -> TocEntryKind {
        match self {
            Self::Part => TocEntryKind::Part,
            Self::Chapter => TocEntryKind::Chapter,
            Self::Section => TocEntryKind::Section,
            Self::Subsection | Self::SubSubsection => TocEntryKind::Subsection,
            Self::FrontMatter => TocEntryKind::FrontMatter,
            Self::BackMatter => TocEntryKind::BackMatter,
            Self::SubEntry => TocEntryKind::SubEntry,
            Self::Unknown => TocEntryKind::Chapter,
        }
    }
}

/// A classified TOC entry (internal, before final output).
#[derive(Debug, Clone)]
struct ClassifiedEntry {
    title: String,
    page_label: String,
    page_value: i32,
    classification: Classification,
    depth: u32,
    font_sig: FontSig,
    x_left: f32,
    page_idx: u32,
}

// ── Public API ──

/// Parse the Table of Contents from a PDF's text layer.
///
/// `page_chars` is the per-page output from `extract_page_chars()`:
/// each element is `(chars, page_height)`.
///
/// Returns `None` if no TOC is detected.
pub fn parse_toc(page_chars: &[(Vec<PdfChar>, f32)]) -> Option<TocResult> {
    // Phase 1: Find TOC pages
    let toc_pages = find_toc_pages(page_chars);
    if toc_pages.is_empty() {
        return None;
    }

    // Phase 2: Extract lines from main TOC pages only (exclude pre-TOC
    // pages like "Contents at a Glance" that are appended at the end of
    // toc_pages for skipping purposes but shouldn't be used for entry
    // extraction).
    // Main TOC pages are sorted ascending; pre-TOC pages are appended and
    // would break the ascending order.
    let main_toc_pages: Vec<u32> = {
        let mut main = Vec::new();
        let mut prev = 0u32;
        for &p in &toc_pages {
            if p >= prev || main.is_empty() {
                main.push(p);
                prev = p;
            } else {
                break; // pre-TOC pages start here (out of order)
            }
        }
        main
    };
    let raw_lines = extract_toc_lines(page_chars, &main_toc_pages);
    if raw_lines.is_empty() {
        return None;
    }

    // Phase 3: Split each line into title + page number
    let split_lines = split_page_numbers(raw_lines);

    // Phase 4: Merge multi-line titles
    let merged = merge_multiline_titles(split_lines);

    // Phase 5: Classify easy cases + learn font signatures + resolve ambiguous
    let mut entries = classify_and_learn(merged);

    // Phase 7: Validate
    validate_entries(&mut entries);

    if entries.len() < 3 {
        return None;
    }

    // Build font profile from classified entries
    let font_profile = build_font_profile(&entries);

    let toc_entries = entries
        .into_iter()
        .map(|e| TocEntry {
            depth: e.depth,
            title: e.title,
            page_label: e.page_label,
            page_value: e.page_value,
            kind: e.classification.to_kind(),
        })
        .collect();

    Some(TocResult {
        entries: toc_entries,
        toc_pages,
        font_profile,
    })
}

// ── Phase 1: Find TOC pages ──

/// Check if a page looks like a TOC page based on its content.
/// A TOC page has many lines ending with page numbers and/or leader dots.
fn is_toc_like_page(page_data: &(Vec<PdfChar>, f32), page_idx: u32) -> bool {
    let (chars, page_height) = page_data;
    if chars.is_empty() || *page_height <= 0.0 {
        return false;
    }

    let lines = build_lines_from_chars(chars, *page_height, page_idx);
    let non_empty: Vec<&TocRawLine> = lines.iter().filter(|l| !l.text().trim().is_empty()).collect();
    if non_empty.len() < 4 {
        return false;
    }

    let with_trailing_number = non_empty.iter().filter(|l| line_ends_with_number(l)).count();
    let with_dots = non_empty.iter().filter(|l| headings::has_leader_dots(&l.text())).count();

    let pct_numbers = with_trailing_number as f32 / non_empty.len() as f32;
    let pct_dots = with_dots as f32 / non_empty.len() as f32;

    // A page is TOC-like if ≥30% of lines end with page numbers OR ≥20% have leader dots
    pct_numbers >= 0.3 || pct_dots >= 0.2
}

fn find_toc_pages(page_chars: &[(Vec<PdfChar>, f32)]) -> Vec<u32> {
    let total = page_chars.len();
    let scan_limit = total.min(40).min(total / 3 + 5);

    // Find the "Contents" start page.
    // Two-pass: first look for exact heading match ("contents", "table of contents"),
    // then fall back to any heading containing "contents".
    let mut start_page: Option<u32> = None;
    let mut fallback_page: Option<u32> = None;
    for page_idx in 0..scan_limit {
        let (chars, page_height) = &page_chars[page_idx];
        if chars.is_empty() {
            continue;
        }
        let lines = build_lines_from_chars(chars, *page_height, page_idx as u32);
        // Check the 3 physically topmost lines (by Y-position, not list index)
        let mut top_indices: Vec<(usize, f32)> = lines
            .iter()
            .enumerate()
            .map(|(i, l)| (i, l.y_center))
            .collect();
        top_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, _) in top_indices.iter().take(3) {
            let text = lines[idx].text();
            let trimmed = text.trim().to_ascii_lowercase();
            if trimmed.starts_with("list of") {
                continue;
            }
            if trimmed == "contents" || trimmed == "table of contents" {
                start_page = Some(page_idx as u32);
                break;
            }
            if fallback_page.is_none() && trimmed.contains("contents") {
                fallback_page = Some(page_idx as u32);
            }
        }
        if start_page.is_some() {
            break;
        }
    }
    if start_page.is_none() {
        start_page = fallback_page;
    }

    // If no "contents" heading found, fall back to content-based detection:
    // scan from the front for consecutive pages with TOC characteristics
    // (many lines ending with page numbers, leader dots).
    let start = match start_page {
        Some(s) => s,
        None => {
            // Scan the first ~30 pages for runs of TOC-like pages
            let mut first_toc_page: Option<u32> = None;
            for page_idx in 0..scan_limit {
                if is_toc_like_page(&page_chars[page_idx], page_idx as u32) {
                    if first_toc_page.is_none() {
                        first_toc_page = Some(page_idx as u32);
                    }
                } else if first_toc_page.is_some() {
                    // Found end of TOC run
                    break;
                }
            }
            match first_toc_page {
                Some(s) => s,
                None => return vec![],
            }
        }
    };

    // Extend forward: include pages with TOC-like characteristics.
    let mut toc_pages = vec![start];
    // Scan backward to find preceding TOC-like pages (e.g. "Contents at a
    // Glance" before "Table of Contents"). These are added to toc_pages
    // for SKIPPING during reflow (so their content isn't in the body text),
    // but they're placed AFTER the start page in the list so they don't
    // interfere with entry extraction (which only uses lines from toc_pages
    // in order). We'll append them at the end after forward extension.
    let mut pre_toc_pages: Vec<u32> = Vec::new();
    if start > 0 {
        for page_idx in (0..start).rev() {
            if is_toc_like_page(&page_chars[page_idx as usize], page_idx) {
                pre_toc_pages.push(page_idx);
            } else {
                break;
            }
        }
    }
    for page_idx in (start + 1)..scan_limit as u32 {
        let (chars, page_height) = &page_chars[page_idx as usize];
        if chars.is_empty() {
            // Allow one blank page gap (two-sided printing)
            if toc_pages.last() == Some(&(page_idx - 1)) {
                continue;
            }
            break;
        }

        let lines = build_lines_from_chars(chars, *page_height, page_idx);

        // Check if this is a "List of Figures/Tables" page (stop).
        // Use physically topmost lines by Y-position (important for two-column layouts).
        let mut top_idx2: Vec<(usize, f32)> = lines
            .iter()
            .enumerate()
            .map(|(i, l)| (i, l.y_center))
            .collect();
        top_idx2.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, _) in top_idx2.iter().take(3) {
            let text = lines[idx].text();
            let trimmed = text.trim().to_ascii_lowercase();
            if trimmed.starts_with("list of figures")
                || trimmed.starts_with("list of tables")
                || trimmed.starts_with("list of algorithms")
            {
                return toc_pages;
            }
        }

        // Check TOC signals
        let non_empty_lines: Vec<&TocRawLine> =
            lines.iter().filter(|l| !l.text().trim().is_empty()).collect();
        if non_empty_lines.is_empty() {
            break;
        }

        let lines_with_trailing_number = non_empty_lines
            .iter()
            .filter(|l| line_ends_with_number(l))
            .count();
        let lines_with_dots = non_empty_lines
            .iter()
            .filter(|l| headings::has_leader_dots(&l.text()))
            .count();

        let pct_numbers = lines_with_trailing_number as f32 / non_empty_lines.len() as f32;
        let pct_dots = lines_with_dots as f32 / non_empty_lines.len() as f32;

        if pct_numbers >= 0.3 || pct_dots >= 0.2 {
            toc_pages.push(page_idx);
        } else {
            break;
        }
    }

    // Append pre-TOC pages (for skipping only, not entry extraction).
    // These go at the end of the list so extract_toc_lines processes the
    // main TOC pages first. The pages are still skipped during reflow.
    toc_pages.extend(pre_toc_pages);

    toc_pages
}

/// Check if a line ends with a number (arabic or lowercase roman).
fn line_ends_with_number(line: &TocRawLine) -> bool {
    let text = line.text();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Check last token
    if let Some(last) = trimmed.split_whitespace().next_back() {
        if !last.is_empty() && last.chars().all(|c: char| c.is_ascii_digit()) {
            return true;
        }
        // Lowercase roman numeral
        if last.len() <= 8
            && last
                .chars()
                .all(|c| matches!(c, 'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'))
            && parse_lowercase_roman(last).is_some()
        {
            return true;
        }
    }
    false
}

// ── Phase 2: Line extraction ──

fn extract_toc_lines(page_chars: &[(Vec<PdfChar>, f32)], toc_pages: &[u32]) -> Vec<TocRawLine> {
    let mut all_lines = Vec::new();

    for &page_idx in toc_pages {
        let (chars, page_height) = &page_chars[page_idx as usize];
        let mut lines = build_lines_from_chars(chars, *page_height, page_idx);

        // Filter headers/footers: remove lines in top/bottom margin of page.
        // Use 3% margin — enough to catch running headers/page numbers, but
        // not so wide that it eats Part headings near the top of the page.
        let margin = page_height * 0.03;
        lines.retain(|line| {
            let y = line.y_center;
            // Image-space: y=0 is top. top margin = y < margin, bottom margin = y > (page_height - margin)
            y >= margin && y <= (page_height - margin)
        });

        // Filter lines that are just a bare page number (running footer)
        lines.retain(|line| {
            let text = line.text();
            let trimmed = text.trim();
            // Keep lines that are not just a number
            !(trimmed.chars().all(|c| c.is_ascii_digit() || c.is_ascii_lowercase())
                && trimmed.len() <= 6
                && (trimmed.parse::<u32>().is_ok() || parse_lowercase_roman(trimmed).is_some()))
        });

        // Filter running headers: the topmost line on each page that starts
        // with a TOC-header keyword (e.g. "Contents vi", "TABLE OF CONTENTS 12").
        // These are repeated on every continuation page.
        if !lines.is_empty() {
            // Find the topmost line (smallest y_center in image-space)
            let top_idx = lines
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.y_center
                        .partial_cmp(&b.y_center)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap();
            let top_text = lines[top_idx].text();
            let top_lower = top_text.trim().to_ascii_lowercase();
            if top_lower == "contents"
                || top_lower == "table of contents"
                || top_lower == "detailed contents"
                || top_lower.starts_with("contents ")
                || top_lower.starts_with("table of contents ")
            {
                lines.remove(top_idx);
            }
        }

        all_lines.extend(lines);
    }

    all_lines
}

fn build_lines_from_chars(
    chars: &[PdfChar],
    page_height: f32,
    page_idx: u32,
) -> Vec<TocRawLine> {
    if chars.is_empty() {
        return vec![];
    }

    // Collect pdfium space X-positions (image-space) before filtering them out.
    // These mark word boundaries in tightly-kerned fonts where gap detection fails.
    let pdfium_space_positions: Vec<(f32, f32)> = chars
        .iter()
        .filter(|c| c.codepoint == ' ')
        .map(|c| {
            let y_img = page_height - (c.bbox[1] + c.bbox[3]) / 2.0;
            (c.bbox[0], y_img)
        })
        .collect();

    // Convert to TocChar in image-space.
    // Skip literal spaces (we reconstruct spacing from gaps) and control chars.
    // Expand TeX ligatures (0x0B-0x0F) to their component characters.
    let mut toc_chars: Vec<TocChar> = Vec::new();
    for c in chars {
        if c.codepoint == ' ' {
            continue;
        }
        let y1 = page_height - c.bbox[3];
        let y2 = page_height - c.bbox[1];
        // Expand ligatures: fi, fl, ff, ffi, ffl
        if let Some(ligature) = crate::pdf::expand_ligature(c.codepoint) {
            let has_glyph = (c.bbox[2] - c.bbox[0]) > 0.1;
            if has_glyph || !c.codepoint.is_control() {
                let glyph_w = c.bbox[2] - c.bbox[0];
                let char_w = glyph_w / ligature.len() as f32;
                for (li, lc) in ligature.chars().enumerate() {
                    let x_off = char_w * li as f32;
                    toc_chars.push(TocChar {
                        codepoint: lc,
                        bbox: [c.bbox[0] + x_off, y1, c.bbox[0] + x_off + char_w, y2],
                        space_threshold: c.space_threshold,
                        font_name: c.font_name.clone(),
                        font_size: c.font_size,
                        is_italic: c.is_italic,
                    });
                }
                continue;
            }
        }
        if c.codepoint.is_control() {
            continue;
        }
        toc_chars.push(TocChar {
            codepoint: c.codepoint,
            bbox: [c.bbox[0], y1, c.bbox[2], y2],
            space_threshold: c.space_threshold,
            font_name: c.font_name.clone(),
            font_size: c.font_size,
            is_italic: c.is_italic,
        });
    }

    if toc_chars.is_empty() {
        return vec![];
    }

    // Compute average glyph height for Y-grouping threshold
    let avg_height = {
        let heights: Vec<f32> = toc_chars
            .iter()
            .map(|c| (c.bbox[3] - c.bbox[1]).abs())
            .filter(|h| *h > 0.5)
            .collect();
        if heights.is_empty() {
            10.0
        } else {
            heights.iter().sum::<f32>() / heights.len() as f32
        }
    };
    let y_threshold = avg_height * 0.5;

    // Sort by center-Y (top to bottom), then center-X (left to right)
    let mut sorted: Vec<TocChar> = toc_chars;
    sorted.sort_by(|a, b| {
        let ay = (a.bbox[1] + a.bbox[3]) / 2.0;
        let by = (b.bbox[1] + b.bbox[3]) / 2.0;
        ay.partial_cmp(&by)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.bbox[0]
                    .partial_cmp(&b.bbox[0])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    // Group into lines
    let mut lines: Vec<Vec<TocChar>> = vec![];
    let mut current_line: Vec<TocChar> = vec![sorted[0].clone()];
    let mut current_y = (sorted[0].bbox[1] + sorted[0].bbox[3]) / 2.0;

    for ch in sorted.into_iter().skip(1) {
        let ch_y = (ch.bbox[1] + ch.bbox[3]) / 2.0;
        if (ch_y - current_y).abs() <= y_threshold {
            current_line.push(ch);
        } else {
            // Sort current line by x
            current_line.sort_by(|a, b| {
                a.bbox[0]
                    .partial_cmp(&b.bbox[0])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            lines.push(current_line);
            current_line = vec![ch];
            current_y = ch_y;
        }
    }
    // Don't forget last line
    current_line.sort_by(|a, b| {
        a.bbox[0]
            .partial_cmp(&b.bbox[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    lines.push(current_line);

    // ── Two-column detection and splitting ──
    // If most lines have a large internal X-gap at a consistent X position
    // (middle third of page), this is a two-column TOC. Split each line at
    // the gutter, then reorder: left column top-to-bottom, right column
    // top-to-bottom.
    let lines = split_two_column_lines(lines, page_height);

    lines
        .into_iter()
        .map(|chars| {
            let y_center = if chars.is_empty() {
                0.0
            } else {
                chars.iter().map(|c| (c.bbox[1] + c.bbox[3]) / 2.0).sum::<f32>()
                    / chars.len() as f32
            };
            // Assign pdfium space positions that belong to this line (by Y-proximity
            // and X-range). The X-range filter is important for two-column layouts
            // so that spaces from the other column don't leak in.
            let x_min = chars.first().map(|c| c.bbox[0]).unwrap_or(0.0) - 1.0;
            let x_max = chars.last().map(|c| c.bbox[2]).unwrap_or(0.0) + 1.0;
            let space_xs: Vec<f32> = pdfium_space_positions
                .iter()
                .filter(|(x, y)| {
                    (*y - y_center).abs() <= y_threshold && *x >= x_min && *x <= x_max
                })
                .map(|(x, _)| *x)
                .collect();
            TocRawLine {
                chars,
                pdfium_space_xs: space_xs,
                y_center,
                page_idx,
            }
        })
        .collect()
}

/// Detect and handle two-column TOC layouts.
///
/// For each line (a Y-grouped row of chars), find the largest internal X-gap.
/// If ≥60% of lines with ≥10 chars have a gap >30pt at a consistent X position
/// (within the middle 60% of the page), we declare a two-column layout.
/// Split each line at the gutter, producing separate left/right half-lines,
/// then reorder: all left-column lines (top to bottom), then all right-column
/// lines (top to bottom).
fn split_two_column_lines(lines: Vec<Vec<TocChar>>, _page_height: f32) -> Vec<Vec<TocChar>> {
    if lines.len() < 4 {
        return lines;
    }

    // For each line, compute the maximum internal X-gap and its X-position.
    // A "gap" is the distance between consecutive chars sorted by X.
    struct GapInfo {
        max_gap: f32,
        gap_x: f32, // X-position of the gap midpoint
    }

    let min_gap_threshold = 30.0; // minimum gap to consider as potential gutter
    let page_left = lines.iter().flat_map(|l| l.iter()).map(|c| c.bbox[0]).fold(f32::MAX, f32::min);
    let page_right = lines.iter().flat_map(|l| l.iter()).map(|c| c.bbox[2]).fold(f32::MIN, f32::max);
    let page_span = page_right - page_left;

    if page_span < 100.0 {
        return lines; // page too narrow for two columns
    }

    // Middle 80% of the content area — gutter should be here.
    // Use 10%-90% rather than 20%-80% to handle asymmetric column layouts.
    let gutter_zone_left = page_left + page_span * 0.1;
    let gutter_zone_right = page_left + page_span * 0.9;

    let mut gap_infos: Vec<Option<GapInfo>> = Vec::with_capacity(lines.len());
    let mut qualifying_count = 0usize;
    let mut gutter_gap_count = 0usize;

    for line in &lines {
        if line.len() < 10 {
            gap_infos.push(None);
            continue;
        }
        qualifying_count += 1;

        // Find the largest gap between consecutive chars
        let mut max_gap = 0.0f32;
        let mut max_gap_x = 0.0f32;
        for i in 1..line.len() {
            let gap = line[i].bbox[0] - line[i - 1].bbox[2];
            if gap > max_gap {
                let gap_mid = (line[i - 1].bbox[2] + line[i].bbox[0]) / 2.0;
                // Only consider gaps in the gutter zone
                if gap_mid >= gutter_zone_left && gap_mid <= gutter_zone_right {
                    max_gap = gap;
                    max_gap_x = gap_mid;
                }
            }
        }

        if max_gap > min_gap_threshold {
            gutter_gap_count += 1;
            gap_infos.push(Some(GapInfo {
                max_gap,
                gap_x: max_gap_x,
            }));
        } else {
            gap_infos.push(None);
        }
    }

    // Need ≥25% of qualifying lines to have a gutter gap AND at least 8 lines
    // with a gap. In two-column layouts, many lines are single-column (wrapped
    // titles, subheadings in one column only), so only a fraction show gutter gaps.
    // However, we also require consistency check below (clustering at same X).
    if qualifying_count < 4 || gutter_gap_count < 8 || gutter_gap_count < qualifying_count / 4 {
        return lines;
    }

    // Compute median gutter X-position
    let mut gutter_xs: Vec<f32> = gap_infos
        .iter()
        .filter_map(|g| g.as_ref().map(|gi| gi.gap_x))
        .collect();
    gutter_xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_gutter_x = gutter_xs[gutter_xs.len() / 2];

    // Verify consistency: ≥50% of gutter gaps are within ±15% of page_span from the median
    let tolerance = page_span * 0.15;
    let consistent_count = gutter_xs
        .iter()
        .filter(|&&x| (x - median_gutter_x).abs() <= tolerance)
        .count();
    if consistent_count < gutter_xs.len() / 2 {
        return lines;
    }

    // Distinguish true two-column gutter from title-to-page-number gap.
    // In two columns, both sides of the gap have substantial content.
    // In single-column with page numbers, the right side is just 1-4 digit chars.
    // Check: for lines with a gutter gap, count how many have ≥8 chars on both sides.
    let mut both_sides_substantial = 0usize;
    for (line, gi) in lines.iter().zip(gap_infos.iter()) {
        if let Some(info) = gi {
            if (info.gap_x - median_gutter_x).abs() <= tolerance {
                let left_chars = line.iter().filter(|c| c.bbox[2] < info.gap_x).count();
                let right_chars = line.iter().filter(|c| c.bbox[0] > info.gap_x).count();
                if left_chars >= 8 && right_chars >= 8 {
                    both_sides_substantial += 1;
                }
            }
        }
    }
    // Need majority of consistent-gap lines to have substantial content on both sides
    if both_sides_substantial < consistent_count / 2 {
        return lines;
    }

    // Also compute the median gap size to set the split threshold.
    // Use a lower threshold than the median to catch lines with smaller gutters.
    let mut gap_sizes: Vec<f32> = gap_infos
        .iter()
        .filter_map(|g| g.as_ref().map(|gi| gi.max_gap))
        .collect();
    gap_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let split_threshold = (gap_sizes[0] * 0.7).max(min_gap_threshold);

    // Split each line at the gutter
    let mut left_lines: Vec<(f32, Vec<TocChar>)> = Vec::new(); // (y_center, chars)
    let mut right_lines: Vec<(f32, Vec<TocChar>)> = Vec::new();

    for line in lines {
        let y_center = if line.is_empty() {
            0.0
        } else {
            line.iter().map(|c| (c.bbox[1] + c.bbox[3]) / 2.0).sum::<f32>() / line.len() as f32
        };

        if line.len() < 4 {
            // Short lines: assign to whichever column they're closer to
            let line_center_x = if line.is_empty() {
                0.0
            } else {
                (line.first().unwrap().bbox[0] + line.last().unwrap().bbox[2]) / 2.0
            };
            if line_center_x < median_gutter_x {
                left_lines.push((y_center, line));
            } else {
                right_lines.push((y_center, line));
            }
            continue;
        }

        // Find the best split point: the largest gap near the median gutter X
        let mut best_split_idx = None;
        let mut best_gap = 0.0f32;
        for i in 1..line.len() {
            let gap = line[i].bbox[0] - line[i - 1].bbox[2];
            let gap_mid = (line[i - 1].bbox[2] + line[i].bbox[0]) / 2.0;
            if gap > split_threshold
                && (gap_mid - median_gutter_x).abs() <= tolerance
                && gap > best_gap
            {
                best_gap = gap;
                best_split_idx = Some(i);
            }
        }

        if let Some(split_idx) = best_split_idx {
            let (left_chars, right_chars) = line.split_at(split_idx);
            if !left_chars.is_empty() {
                left_lines.push((y_center, left_chars.to_vec()));
            }
            if !right_chars.is_empty() {
                right_lines.push((y_center, right_chars.to_vec()));
            }
        } else {
            // No gutter found on this line — assign by position
            let line_center_x = (line.first().unwrap().bbox[0] + line.last().unwrap().bbox[2]) / 2.0;
            if line_center_x < median_gutter_x {
                left_lines.push((y_center, line));
            } else {
                right_lines.push((y_center, line));
            }
        }
    }

    // Reorder: left column top-to-bottom, then right column top-to-bottom
    left_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    right_lines.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<Vec<TocChar>> = Vec::with_capacity(left_lines.len() + right_lines.len());
    for (_, chars) in left_lines {
        result.push(chars);
    }
    for (_, chars) in right_lines {
        result.push(chars);
    }

    result
}

// ── Phase 3: Page number extraction ──

fn split_page_numbers(lines: Vec<TocRawLine>) -> Vec<TocSplitLine> {
    lines.into_iter().map(split_single_line).collect()
}

fn split_single_line(line: TocRawLine) -> TocSplitLine {
    let full_text = line.text();
    let x_left = line.x_left();
    let page_idx = line.page_idx;
    let y_center = line.y_center;

    // Strategy A: Leader dots
    if let Some(result) = try_split_by_dots(&full_text) {
        let title_chars = get_title_chars(&line.chars, result.0.len());
        return TocSplitLine {
            title: result.0,
            page_label: Some(result.1.clone()),
            page_value: Some(result.2),
            font_sig: FontSig::from_chars(&title_chars),
            x_left,
            page_idx,
            y_center,
            has_leader_dots: true,
            title_chars,
        };
    }

    // Strategy B: Right-aligned number with X-gap
    if let Some(result) = try_split_by_xgap(&line) {
        let title_chars = get_title_chars(&line.chars, result.0.len());
        return TocSplitLine {
            title: result.0,
            page_label: Some(result.1.clone()),
            page_value: Some(result.2),
            font_sig: FontSig::from_chars(&title_chars),
            x_left,
            page_idx,
            y_center,
            has_leader_dots: false,
            title_chars,
        };
    }

    // Strategy C: Fallback — last token is a number
    if let Some(result) = try_split_by_last_token(&full_text) {
        let title_chars = get_title_chars(&line.chars, result.0.len());
        return TocSplitLine {
            title: result.0,
            page_label: Some(result.1.clone()),
            page_value: Some(result.2),
            font_sig: FontSig::from_chars(&title_chars),
            x_left,
            page_idx,
            y_center,
            has_leader_dots: false,
            title_chars,
        };
    }

    // No page number found
    let title_chars = line.chars.clone();
    TocSplitLine {
        title: full_text.trim().to_string(),
        page_label: None,
        page_value: None,
        font_sig: FontSig::from_chars(&title_chars),
        x_left,
        page_idx,
        y_center,
        has_leader_dots: false,
        title_chars,
    }
}

/// Get the first N chars worth of TocChars (approximating the title portion).
fn get_title_chars(chars: &[TocChar], title_text_len: usize) -> Vec<TocChar> {
    let mut count = 0usize;
    let mut result = Vec::new();
    for ch in chars {
        if count >= title_text_len {
            break;
        }
        result.push(ch.clone());
        count += ch.codepoint.len_utf8();
    }
    result
}

/// Strategy A: Split by leader dots.
fn try_split_by_dots(text: &str) -> Option<(String, String, i32)> {
    // Find the first run of 3+ dots or the ". . ." pattern
    let dot_start = find_leader_dot_start(text)?;
    let title = text[..dot_start].trim().to_string();
    if title.is_empty() {
        return None;
    }

    // Find the page ref after the dots
    let after_dots = text[dot_start..].trim_start_matches(|c: char| c == '.' || c == ' ');
    let page_ref = after_dots.trim();
    if page_ref.is_empty() {
        return None;
    }

    parse_page_ref(page_ref).map(|(label, value)| (title, label, value))
}

/// Find where leader dots start in the text.
fn find_leader_dot_start(text: &str) -> Option<usize> {
    // Look for "..." (3+ consecutive dots)
    if let Some(pos) = text.find("...") {
        return Some(pos);
    }
    // Look for ". . ." pattern (3+ dots with spaces)
    let chars: Vec<char> = text.chars().collect();
    let mut dot_count = 0;
    let mut first_dot_byte = None;
    let mut byte_pos = 0;
    for (i, &ch) in chars.iter().enumerate() {
        if ch == '.' {
            if dot_count == 0 {
                first_dot_byte = Some(byte_pos);
            }
            dot_count += 1;
        } else if ch == ' ' {
            // spaces between dots are ok
        } else {
            if dot_count >= 3 {
                return first_dot_byte;
            }
            dot_count = 0;
            first_dot_byte = None;
        }
        byte_pos += ch.len_utf8();
    }
    if dot_count >= 3 {
        return first_dot_byte;
    }
    None
}

/// Strategy B: Split by X-gap between title text and right-aligned number.
fn try_split_by_xgap(line: &TocRawLine) -> Option<(String, String, i32)> {
    let chars = &line.chars;
    if chars.len() < 3 {
        return None;
    }

    // Compute median char width
    let widths: Vec<f32> = chars
        .iter()
        .map(|c| c.bbox[2] - c.bbox[0])
        .filter(|w| *w > 0.1)
        .collect();
    if widths.is_empty() {
        return None;
    }
    let median_width = {
        let mut sorted = widths.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };
    let gap_threshold = median_width * 4.0;

    // Scan right-to-left: find the rightmost digit run
    let mut num_end = chars.len();
    let mut num_start = chars.len();

    // Skip trailing whitespace-like chars
    while num_end > 0
        && (chars[num_end - 1].codepoint == ' ' || chars[num_end - 1].codepoint == '\u{00A0}')
    {
        num_end -= 1;
    }

    if num_end == 0 {
        return None;
    }

    // Collect digits from right
    let mut i = num_end;
    while i > 0 && chars[i - 1].codepoint.is_ascii_digit() {
        i -= 1;
    }
    // Also try lowercase roman
    if i == num_end {
        while i > 0
            && matches!(
                chars[i - 1].codepoint,
                'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'
            )
        {
            i -= 1;
        }
    }
    num_start = i;

    if num_start >= num_end {
        return None;
    }

    // Check x-gap between the char before num_start and the char at num_start
    if num_start == 0 {
        return None; // number is the entire line
    }
    let gap = chars[num_start].bbox[0] - chars[num_start - 1].bbox[2];
    if gap < gap_threshold {
        return None;
    }

    // Build title and page ref
    let mut title_text = String::new();
    for (j, ch) in chars[..num_start].iter().enumerate() {
        if j > 0 {
            let g = ch.bbox[0] - chars[j - 1].bbox[2];
            let thresh = if ch.space_threshold > 0.1 {
                ch.space_threshold
            } else {
                let h = (ch.bbox[3] - ch.bbox[1]).abs();
                if h > 0.5 { h * 0.15 } else { 2.0 }
            };
            if g > thresh {
                title_text.push(' ');
            }
        }
        title_text.push(ch.codepoint);
    }

    let page_str: String = chars[num_start..num_end]
        .iter()
        .map(|c| c.codepoint)
        .collect();

    let title = title_text.trim().to_string();
    if title.is_empty() {
        return None;
    }

    parse_page_ref(&page_str).map(|(label, value)| (title, label, value))
}

/// Strategy C: Split by last whitespace-delimited token.
fn try_split_by_last_token(text: &str) -> Option<(String, String, i32)> {
    let trimmed = text.trim();
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    let last = parts.last()?;
    let (label, value) = parse_page_ref(last)?;

    // Reconstruct title without the last token
    let title = parts[..parts.len() - 1].join(" ");
    if title.is_empty() {
        return None;
    }

    Some((title, label, value))
}

/// Parse a page reference string into (label, value).
/// Arabic → positive, lowercase roman → negative.
fn parse_page_ref(s: &str) -> Option<(String, i32)> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Strip internal spaces (PDF extraction sometimes inserts spaces within numbers)
    let compact: String = s.chars().filter(|c| !c.is_whitespace()).collect();

    // Try arabic
    if let Ok(n) = compact.parse::<u32>() {
        if n > 0 && n < 10000 {
            return Some((compact, n as i32));
        }
    }

    // Try lowercase roman
    if compact.len() <= 10
        && compact
            .chars()
            .all(|c| matches!(c, 'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'))
    {
        if let Some(val) = parse_lowercase_roman(&compact) {
            return Some((compact, -(val as i32)));
        }
    }

    None
}

/// Parse a lowercase roman numeral string to its numeric value.
fn parse_lowercase_roman(s: &str) -> Option<u32> {
    let upper: String = s.to_ascii_uppercase();
    headings::roman_to_u32(&upper)
}

// ── Phase 4: Multi-line title merging ──

/// Join two title fragments, handling line-break hyphenation.
/// "About Clas-" + "sical Mechanics" → "About Classical Mechanics"
/// "Some Title" + "Continued" → "Some Title Continued"
fn join_title_parts(left: &str, right: &str) -> String {
    if left.ends_with('-') {
        // Check if this is a real hyphenated compound word (both sides capitalized)
        // or a line-break split (second part is lowercase).
        let right_starts_lower = right.starts_with(|c: char| c.is_lowercase());
        if right_starts_lower {
            // Line-break hyphenation: remove hyphen and join directly
            format!("{}{}", &left[..left.len() - 1], right)
        } else {
            // Compound word like "Self-Adjoint": keep the hyphen
            format!("{} {}", left, right)
        }
    } else {
        format!("{} {}", left, right)
    }
}

fn merge_multiline_titles(lines: Vec<TocSplitLine>) -> Vec<TocSplitLine> {
    if lines.is_empty() {
        return vec![];
    }

    let mut result: Vec<TocSplitLine> = Vec::new();
    let mut buffer: Vec<TocSplitLine> = Vec::new();

    for line in lines {
        if line.page_label.is_some() {
            // This line has a page number
            if !buffer.is_empty() {
                let starts_with_pattern = starts_with_heading_pattern(&line.title);
                if !starts_with_pattern {
                    // This line looks like a continuation (no heading pattern).
                    // Walk backward through the buffer to find consecutive trailing
                    // entries with the same font signature — these are part of the
                    // same wrapped title. Earlier entries (different font, e.g.
                    // italic attribution lines) get flushed as standalone.
                    let current_font = &line.font_sig;
                    let mut merge_start = buffer.len();
                    for j in (0..buffer.len()).rev() {
                        if buffer[j].font_sig == *current_font {
                            merge_start = j;
                        } else {
                            break;
                        }
                    }

                    if merge_start < buffer.len() {
                        // Flush entries before the merge range
                        for entry in buffer.drain(..merge_start) {
                            result.push(entry);
                        }
                        // Merge remaining buffer entries + current line
                        let mut merged = buffer.remove(0);
                        for remaining in buffer.drain(..) {
                            merged.title = join_title_parts(
                                merged.title.trim(),
                                remaining.title.trim(),
                            );
                        }
                        merged.title = join_title_parts(
                            merged.title.trim(),
                            line.title.trim(),
                        );
                        merged.page_label = line.page_label;
                        merged.page_value = line.page_value;
                        merged.has_leader_dots = line.has_leader_dots;
                        result.push(merged);
                    } else {
                        // No font match — flush buffer and push standalone
                        flush_buffer(&mut buffer, &mut result);
                        result.push(line);
                    }
                    buffer.clear();
                } else {
                    // Flush buffer entries
                    flush_buffer(&mut buffer, &mut result);
                    result.push(line);
                    buffer.clear();
                }
            } else {
                result.push(line);
            }
        } else {
            // No page number — check if it's a Part heading (emit immediately)
            if is_part_pattern(&line.title) {
                flush_buffer(&mut buffer, &mut result);
                result.push(line);
            } else {
                buffer.push(line);
            }
        }
    }

    // Flush remaining buffer
    flush_buffer(&mut buffer, &mut result);
    result
}

fn flush_buffer(buffer: &mut Vec<TocSplitLine>, result: &mut Vec<TocSplitLine>) {
    result.extend(buffer.drain(..));
}

fn starts_with_heading_pattern(title: &str) -> bool {
    let t = title.trim();
    // CHAPTER N or Chapter N
    if t.starts_with("CHAPTER ") || t.starts_with("Chapter ") {
        return true;
    }
    // Part N
    if is_part_pattern(t) {
        return true;
    }
    // N.N or N
    if let Some(first) = t.chars().next() {
        if first.is_ascii_digit() {
            return true;
        }
    }
    // Roman numeral start
    if t.starts_with(|c: char| matches!(c, 'I' | 'V' | 'X' | 'L')) {
        if let Some(space_pos) = t.find(' ') {
            let prefix = &t[..space_pos];
            if headings::is_roman_numeral(prefix) {
                return true;
            }
        }
    }
    false
}

fn is_part_pattern(title: &str) -> bool {
    let t = title.trim();
    let lower = t.to_ascii_lowercase();
    if lower.starts_with("part ") {
        let rest = t[5..].trim();
        // Part followed by a roman numeral or digit
        if let Some(first) = rest.chars().next() {
            return first.is_ascii_digit()
                || matches!(first, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M');
        }
    }
    false
}

// ── Phase 5: Classification + font learning + ambiguous resolution ──

fn classify_and_learn(lines: Vec<TocSplitLine>) -> Vec<ClassifiedEntry> {
    // First pass: classify easy cases
    let mut entries: Vec<(TocSplitLine, Classification, u32)> = Vec::new();
    let mut last_classified_depth = 1u32;

    for line in &lines {
        if line.page_label.is_none() && !is_part_pattern(&line.title) {
            // No page number and not a Part heading → will be dropped
            continue;
        }
        let (class, depth) = classify_by_pattern(&line.title, last_classified_depth);
        if class != Classification::Unknown {
            last_classified_depth = depth;
        }
        entries.push((line.clone(), class, depth));
    }

    // Post-classification: detect roman-numeral Part headings
    // If a roman-numeral entry (classified as Chapter) is followed by bare-number
    // chapters that restart from 1, the roman entries are actually Parts.
    promote_roman_to_parts(&mut entries);

    // Also strip "CHAPTER N" prefix from titles
    for (line, class, _) in &mut entries {
        if *class == Classification::Chapter {
            let t = line.title.trim();
            if let Some(rest) = t
                .strip_prefix("CHAPTER ")
                .or_else(|| t.strip_prefix("Chapter "))
            {
                line.title = rest.to_string();
            }
        }
    }

    // Second pass: learn font signatures from classified entries
    let mut sig_to_depth: HashMap<FontSig, Vec<u32>> = HashMap::new();
    for (line, class, depth) in &entries {
        if *class != Classification::Unknown {
            sig_to_depth.entry(line.font_sig.clone()).or_default().push(*depth);
        }
    }

    // Majority vote per signature
    let learned: HashMap<FontSig, u32> = sig_to_depth
        .into_iter()
        .map(|(sig, depths)| {
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for d in &depths {
                *counts.entry(*d).or_default() += 1;
            }
            let best = counts.into_iter().max_by_key(|(_, c)| *c).unwrap().0;
            (sig, best)
        })
        .collect();

    // Also learn x-indentation levels — use ALL entries (not just classified)
    // so the indent clusters accurately reflect the full visual hierarchy.
    let x_lefts: Vec<f32> = entries
        .iter()
        .map(|(l, _, _)| l.x_left)
        .collect();
    // Normalize x_left per page: subtract each page's minimum x_left so that
    // indent positions are comparable across pages with different margins.
    let mut page_min_x: HashMap<u32, f32> = HashMap::new();
    for (line, _, _) in &entries {
        let min = page_min_x.entry(line.page_idx).or_insert(f32::MAX);
        if line.x_left < *min {
            *min = line.x_left;
        }
    }
    let x_lefts_normalized: Vec<f32> = entries
        .iter()
        .map(|(l, _, _)| {
            let page_min = page_min_x.get(&l.page_idx).copied().unwrap_or(0.0);
            l.x_left - page_min
        })
        .collect();
    let indent_levels = compute_indent_levels(&x_lefts_normalized);

    // (debug removed)

    // Build a mapping from indent level → most common depth among classified
    // entries at that indent. This lets us anchor indent levels to known depths.
    let mut indent_depth_votes: HashMap<u32, Vec<u32>> = HashMap::new();
    for (line, class, depth) in &entries {
        if *class != Classification::Unknown {
            let page_min = page_min_x.get(&line.page_idx).copied().unwrap_or(0.0);
            let indent = quantize_indent(line.x_left - page_min, &indent_levels);
            indent_depth_votes.entry(indent).or_default().push(*depth);
        }
    }
    let indent_to_depth: HashMap<u32, u32> = indent_depth_votes
        .into_iter()
        .map(|(indent, depths)| {
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for d in &depths {
                *counts.entry(*d).or_default() += 1;
            }
            let best = counts.into_iter().max_by_key(|(_, c)| *c).unwrap().0;
            (indent, best)
        })
        .collect();

    // Third pass: resolve ambiguous entries.
    // Priority: x-position indentation first (more reliable across visual
    // hierarchies), font signature as fallback.
    let mut result: Vec<ClassifiedEntry> = entries
        .into_iter()
        .map(|(line, mut class, mut depth)| {
            if class == Classification::Unknown {
                if !indent_levels.is_empty() {
                    let page_min = page_min_x.get(&line.page_idx).copied().unwrap_or(0.0);
                    let indent = quantize_indent(line.x_left - page_min, &indent_levels);
                    // Use anchored indent→depth if available, else raw indent+1
                    if let Some(&anchored) = indent_to_depth.get(&indent) {
                        depth = anchored;
                    } else {
                        // Interpolate: find the nearest anchored indent level
                        // and offset from it.
                        let nearest = indent_to_depth
                            .iter()
                            .min_by_key(|(k, _)| (**k as i32 - indent as i32).unsigned_abs());
                        if let Some((near_indent, near_depth)) = nearest {
                            let (near_indent, near_depth) = (*near_indent, *near_depth);
                            let offset = indent as i32 - near_indent as i32;
                            depth = (near_depth as i32 + offset).max(1).min(4) as u32;
                        } else {
                            depth = (indent + 1).min(4);
                        }
                    }
                    class = depth_to_classification(depth);
                } else if let Some(&learned_depth) = learned.get(&line.font_sig) {
                    // Font signature fallback when no indent levels available
                    depth = learned_depth.min(4);
                    class = depth_to_classification(depth);
                } else {
                    // Default to chapter
                    depth = 1;
                    class = Classification::Chapter;
                }
            }

            ClassifiedEntry {
                title: line.title,
                page_label: line.page_label.unwrap_or_default(),
                page_value: line.page_value.unwrap_or(0),
                classification: class,
                depth,
                font_sig: line.font_sig,
                x_left: line.x_left,
                page_idx: line.page_idx,
            }
        })
        .collect();

    // Fourth pass: demote per-chapter Bibliography/References from FrontMatter
    // to SubEntry. If a FM entry with a repeating title like "Bibliography" has
    // a page number that fits between the preceding chapter and the next chapter,
    // it belongs to that chapter, not to the book's front/back matter.
    demote_per_chapter_fm(&mut result);

    result
}

/// Classify a TOC entry by its title text pattern.
fn classify_by_pattern(title: &str, last_depth: u32) -> (Classification, u32) {
    let t = title.trim();

    // Rule 1: "CHAPTER N" / "Chapter N"
    if let Some(rest) = t
        .strip_prefix("CHAPTER ")
        .or_else(|| t.strip_prefix("Chapter "))
    {
        if rest.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            return (Classification::Chapter, 1);
        }
    }

    // Rule 2: "Part N" / "PART N"
    if is_part_pattern(t) {
        return (Classification::Part, 0);
    }

    // Rule 3-4: Dotted decimal numbering (multi-level only: 1.1, 1.2.3, etc.)
    let (num_depth, is_multi) = headings::infer_depth_from_numbering(t);
    if num_depth > 0 && is_multi {
        let class = match num_depth {
            1 => Classification::Chapter,
            2 => Classification::Section,
            3 => Classification::Subsection,
            _ => Classification::SubSubsection,
        };
        return (class, num_depth);
    }

    // Rule 5: Appendix section "A.1 Foo", "A.1.2 Bar", etc.
    if t.len() > 2 {
        let first = t.as_bytes()[0];
        if first.is_ascii_uppercase() && t.as_bytes().get(1) == Some(&b'.') {
            if let Some(ch) = t.as_bytes().get(2) {
                if ch.is_ascii_digit() {
                    // Count dots to determine depth: A.1 → 2, A.1.2 → 3
                    let num_end = t.find(|c: char| c == ' ' || c == '\t').unwrap_or(t.len());
                    let prefix = &t[..num_end];
                    let dot_count = prefix.chars().filter(|&c| c == '.').count();
                    let depth = (dot_count as u32) + 1;
                    let class = match depth {
                        1 => Classification::Chapter,
                        2 => Classification::Section,
                        3 => Classification::Subsection,
                        _ => Classification::SubSubsection,
                    };
                    return (class, depth);
                }
            }
        }
    }

    // Rule 6: Bare number + word (chapter), including "N. Title" format
    if let Some(first) = t.chars().next() {
        if first.is_ascii_digit() {
            // Check it's not something weird — find the space after the number
            if let Some(space_pos) = t.find(' ') {
                let num_part = &t[..space_pos];
                // Accept pure digits ("1 Title") or digits+dot ("1. Title")
                let stripped = num_part.trim_end_matches('.');
                if !stripped.is_empty() && stripped.chars().all(|c| c.is_ascii_digit()) {
                    return (Classification::Chapter, 1);
                }
            }
        }
    }

    // Rule 7: Appendix
    if t.starts_with("Appendix ") || t.starts_with("APPENDIX ") {
        return (Classification::BackMatter, 1);
    }

    // Rule 8: Front/back matter keywords
    if headings::is_frontmatter_heading(t) {
        return (Classification::FrontMatter, 1);
    }

    // Rule 9: Sub-entry keywords
    let lower = t.to_ascii_lowercase();
    if matches!(
        lower.as_str(),
        "exercises"
            | "problems"
            | "references"
            | "notes"
            | "notes and references"
            | "bibliographic notes"
            | "further reading"
            | "summary"
    ) {
        let sub_depth = (last_depth + 1).min(4);
        return (Classification::SubEntry, sub_depth);
    }

    // Rule 10: Roman numeral + word (ambiguous — could be Part or Chapter)
    if let Some(space_pos) = t.find(' ') {
        let prefix = &t[..space_pos];
        if headings::is_roman_numeral(prefix) {
            // If we've seen Part patterns elsewhere, this is likely a chapter within a Part.
            // Otherwise, it could be a Part. Default to Chapter for now.
            return (Classification::Chapter, 1);
        }
    }

    (Classification::Unknown, 0)
}

fn depth_to_classification(depth: u32) -> Classification {
    match depth {
        0 => Classification::Part,
        1 => Classification::Chapter,
        2 => Classification::Section,
        3 => Classification::Subsection,
        _ => Classification::SubSubsection,
    }
}

/// Compute distinct indentation levels from a set of x-positions.
fn compute_indent_levels(x_lefts: &[f32]) -> Vec<f32> {
    if x_lefts.is_empty() {
        return vec![];
    }
    let mut sorted = x_lefts.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.dedup_by(|a, b| (*a - *b).abs() < 5.0);

    // Cluster: merge values within 5pt of each other
    let mut levels: Vec<f32> = vec![sorted[0]];
    for &x in &sorted[1..] {
        if (x - levels.last().unwrap()).abs() > 5.0 {
            levels.push(x);
        }
    }
    levels
}

/// Quantize an x-position to an indent level (0-based).
fn quantize_indent(x: f32, levels: &[f32]) -> u32 {
    levels
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (x - **a)
                .abs()
                .partial_cmp(&(x - **b).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Detect roman-numeral entries classified as Chapter that are actually Part headings.
///
/// Pattern: if entries like "I ...", "II ...", "III ..." are classified as Chapter (depth 1)
/// and between them we see bare-number chapters that restart from 1, then the roman
/// numeral entries are really Parts (depth 0).
fn promote_roman_to_parts(entries: &mut [(TocSplitLine, Classification, u32)]) {
    // Collect indices of roman-numeral chapter entries
    let roman_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, (line, class, _))| {
            if *class != Classification::Chapter {
                return false;
            }
            let t = line.title.trim();
            if let Some(space_pos) = t.find(' ') {
                let prefix = &t[..space_pos];
                headings::is_roman_numeral(prefix)
            } else {
                // Bare roman numeral line like "II" (rare but possible)
                headings::is_roman_numeral(t)
            }
        })
        .map(|(i, _)| i)
        .collect();

    if roman_indices.len() < 2 {
        return; // Need at least 2 roman numeral entries to detect the pattern
    }

    // Check if between consecutive roman entries, there are bare-number chapters
    // that restart from low numbers (suggesting roman entries group them as Parts).
    let mut has_restart = false;
    for window in roman_indices.windows(2) {
        let (start, end) = (window[0], window[1]);
        // Look at entries between these two roman entries
        let between = &entries[start + 1..end];
        // Find bare-number chapters (depth 1) in between
        let chapter_numbers: Vec<u32> = between
            .iter()
            .filter(|(_, class, _)| *class == Classification::Chapter)
            .filter_map(|(line, _, _)| {
                let t = line.title.trim();
                // Extract leading number
                let num_str: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
                num_str.parse::<u32>().ok()
            })
            .collect();
        // If we see a chapter starting at 1 (or low number ≤3), that's a restart
        if chapter_numbers.first().copied().unwrap_or(999) <= 3 {
            has_restart = true;
            break;
        }
    }

    if !has_restart {
        return;
    }

    // Promote all roman-numeral chapters to Part (depth 0)
    for &idx in &roman_indices {
        entries[idx].1 = Classification::Part;
        entries[idx].2 = 0;
    }
}

/// Demote per-chapter Bibliography/References from FrontMatter to SubEntry.
///
/// A "Bibliography" classified as FrontMatter is actually a per-chapter sub-entry
/// if its page number falls between the preceding chapter's content and the next
/// chapter (or Part). We detect this by checking:
///   prev chapter starts at page X, next chapter starts at page Z,
///   and this "Bibliography" is at page Y where X ≤ Y < Z.
fn demote_per_chapter_fm(entries: &mut [ClassifiedEntry]) {
    let repeating_fm_titles: &[&str] = &[
        "bibliography",
        "references",
        "notes",
        "notes and references",
        "bibliographic notes",
        "further reading",
    ];

    // Find indices of chapter-level (depth ≤ 1) entries to define chapter boundaries
    let chapter_starts: Vec<(usize, i32)> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.depth <= 1 && e.page_value > 0)
        .map(|(i, e)| (i, e.page_value))
        .collect();

    for i in 0..entries.len() {
        if entries[i].classification != Classification::FrontMatter {
            continue;
        }
        let title_lower = entries[i].title.trim().to_ascii_lowercase();
        if !repeating_fm_titles.iter().any(|&t| title_lower == t) {
            continue;
        }
        let pv = entries[i].page_value;
        if pv <= 0 {
            continue; // actual front matter page
        }

        // Find which chapter boundary this falls in:
        // the last chapter_start with page ≤ pv, and the next chapter_start after it
        let prev_chapter = chapter_starts
            .iter()
            .filter(|(idx, pg)| *idx < i && *pg <= pv)
            .last();
        let next_chapter = chapter_starts
            .iter()
            .find(|(idx, _)| *idx > i);

        let is_between_chapters = match (prev_chapter, next_chapter) {
            (Some((_pi, pp)), Some((_ni, np))) => pv >= *pp && pv < *np,
            (Some((_pi, pp)), None) => pv >= *pp, // last chapter in the book
            _ => false,
        };

        if is_between_chapters {
            entries[i].classification = Classification::SubEntry;
            entries[i].depth = 2; // section-level, under the chapter
        }
    }
}

// ── Phase 7: Validation ──

fn validate_entries(entries: &mut Vec<ClassifiedEntry>) {
    // Remove duplicates (same title + same page)
    let mut seen = std::collections::HashSet::new();
    entries.retain(|e| seen.insert((e.title.clone(), e.page_value)));

    // Drop entries with page_value 0 that aren't Part headings
    entries.retain(|e| e.page_value != 0 || e.classification == Classification::Part);

    // Drop entries that break page-number sequencing.
    //
    // Correct order: front matter (negative, decreasing: -1, -3, -17)
    // then body (positive, non-decreasing: 1, 3, 45, 46).
    // Part headings with page_value 0 are exempt.
    //
    // Violations:
    //  - Negative page after we've entered the body (positive pages)
    //  - Body page that goes backwards by more than a small tolerance
    let mut in_body = false;
    let mut last_front = 0i32; // tracks most-negative front matter value seen
    let mut last_body = 0i32;

    entries.retain(|e| {
        let pv = e.page_value;

        // Part headings without a page number are always kept
        if pv == 0 && e.classification == Classification::Part {
            return true;
        }

        if pv > 0 {
            // Entering or continuing the body
            if !in_body {
                in_body = true;
                last_body = pv;
                return true;
            }
            // Allow small backwards jumps (repeated page for sub-entries)
            if pv >= last_body - 2 {
                last_body = last_body.max(pv);
                return true;
            }
            // Large backwards jump — drop
            tracing::debug!(
                "TOC: dropping out-of-sequence entry: {:?} (p.{}) after body p.{}",
                e.title, pv, last_body
            );
            false
        } else {
            // Negative (front matter)
            if in_body {
                // Roman numeral page after body pages — clearly wrong
                tracing::debug!(
                    "TOC: dropping front-matter page in body: {:?} (p.{})",
                    e.title, pv
                );
                return false;
            }
            // Within front matter: values decrease (-1, -3, -17)
            if last_front != 0 && pv > last_front + 2 {
                tracing::debug!(
                    "TOC: dropping out-of-sequence front matter: {:?} (p.{}) after p.{}",
                    e.title, pv, last_front
                );
                return false;
            }
            last_front = last_front.min(pv);
            true
        }
    });
}

// ── Font profile building ──

fn build_font_profile(entries: &[ClassifiedEntry]) -> TocFontProfile {
    let mut depth_sigs: HashMap<u32, Vec<(&FontSig, &str)>> = HashMap::new();
    for entry in entries {
        depth_sigs
            .entry(entry.depth)
            .or_default()
            .push((&entry.font_sig, &entry.title));
    }

    let mut levels: Vec<TocFontLevel> = depth_sigs
        .into_iter()
        .map(|(depth, sigs)| {
            // Most common signature
            let mut sig_counts: HashMap<&FontSig, usize> = HashMap::new();
            for (sig, _) in &sigs {
                *sig_counts.entry(sig).or_default() += 1;
            }
            let (best_sig, _) = sig_counts.into_iter().max_by_key(|(_, c)| *c).unwrap();
            let example = sigs.first().map(|(_, t)| t.to_string()).unwrap_or_default();

            TocFontLevel {
                depth,
                font: best_sig.family.clone(),
                size: best_sig.size_bucket as f32 / 2.0,
                is_bold: best_sig.is_bold,
                is_all_caps: best_sig.is_all_caps,
                example,
            }
        })
        .collect();

    levels.sort_by_key(|l| l.depth);

    TocFontProfile { levels }
}

// ── Unit tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_page_ref_arabic() {
        assert_eq!(parse_page_ref("42"), Some(("42".into(), 42)));
        assert_eq!(parse_page_ref("1"), Some(("1".into(), 1)));
        assert_eq!(parse_page_ref("683"), Some(("683".into(), 683)));
        assert_eq!(parse_page_ref("0"), None);
        assert_eq!(parse_page_ref(""), None);
    }

    #[test]
    fn test_parse_page_ref_roman() {
        assert_eq!(parse_page_ref("xvii"), Some(("xvii".into(), -17)));
        assert_eq!(parse_page_ref("iii"), Some(("iii".into(), -3)));
        assert_eq!(parse_page_ref("xxvii"), Some(("xxvii".into(), -27)));
        assert_eq!(parse_page_ref("i"), Some(("i".into(), -1)));
    }

    #[test]
    fn test_find_leader_dot_start() {
        assert_eq!(find_leader_dot_start("Title...42"), Some(5));
        assert_eq!(find_leader_dot_start("Title . . . . 42"), Some(6));
        assert_eq!(find_leader_dot_start("No dots here"), None);
        assert_eq!(find_leader_dot_start("1.1 Title"), None);
    }

    #[test]
    fn test_try_split_by_dots() {
        let r = try_split_by_dots("1.1 Introduction . . . . . . . . . 3");
        assert!(r.is_some());
        let (title, label, value) = r.unwrap();
        assert_eq!(title, "1.1 Introduction");
        assert_eq!(label, "3");
        assert_eq!(value, 3);
    }

    #[test]
    fn test_try_split_by_dots_roman() {
        let r = try_split_by_dots("Foreword...xvii");
        assert!(r.is_some());
        let (title, label, value) = r.unwrap();
        assert_eq!(title, "Foreword");
        assert_eq!(label, "xvii");
        assert_eq!(value, -17);
    }

    #[test]
    fn test_try_split_by_last_token() {
        let r = try_split_by_last_token("8 Quasi-Newton Methods 208");
        assert!(r.is_some());
        let (title, label, value) = r.unwrap();
        assert_eq!(title, "8 Quasi-Newton Methods");
        assert_eq!(label, "208");
        assert_eq!(value, 208);
    }

    #[test]
    fn test_classify_chapter_prefix() {
        assert_eq!(
            classify_by_pattern("CHAPTER 1 Introduction", 1),
            (Classification::Chapter, 1)
        );
        assert_eq!(
            classify_by_pattern("Chapter 2 Background", 1),
            (Classification::Chapter, 1)
        );
    }

    #[test]
    fn test_classify_part() {
        assert_eq!(
            classify_by_pattern("Part I Fundamental Concepts", 1),
            (Classification::Part, 0)
        );
        assert_eq!(
            classify_by_pattern("Part 2 Advanced Topics", 1),
            (Classification::Part, 0)
        );
    }

    #[test]
    fn test_classify_section() {
        assert_eq!(
            classify_by_pattern("1.1 Background", 1),
            (Classification::Section, 2)
        );
        assert_eq!(
            classify_by_pattern("3.4.1 Algorithms", 2),
            (Classification::Subsection, 3)
        );
    }

    #[test]
    fn test_classify_bare_number() {
        assert_eq!(
            classify_by_pattern("1 Introduction", 1),
            (Classification::Chapter, 1)
        );
        assert_eq!(
            classify_by_pattern("12 Conclusion", 1),
            (Classification::Chapter, 1)
        );
    }

    #[test]
    fn test_classify_frontmatter() {
        assert_eq!(
            classify_by_pattern("Preface", 1),
            (Classification::FrontMatter, 1)
        );
        assert_eq!(
            classify_by_pattern("Index", 1),
            (Classification::FrontMatter, 1)
        );
    }

    #[test]
    fn test_classify_sub_entry() {
        assert_eq!(
            classify_by_pattern("Exercises", 2),
            (Classification::SubEntry, 3)
        );
        assert_eq!(
            classify_by_pattern("Notes and References", 2),
            (Classification::SubEntry, 3)
        );
    }

    #[test]
    fn test_is_part_pattern() {
        assert!(is_part_pattern("Part I Fundamental Concepts"));
        assert!(is_part_pattern("Part 2 Advanced Topics"));
        assert!(is_part_pattern("PART III Theory"));
        assert!(!is_part_pattern("Partial Differential Equations"));
        assert!(!is_part_pattern("1 Introduction"));
    }

    #[test]
    fn test_starts_with_heading_pattern() {
        assert!(starts_with_heading_pattern("CHAPTER 1 Introduction"));
        assert!(starts_with_heading_pattern("1.1 Background"));
        assert!(starts_with_heading_pattern("1 Introduction"));
        assert!(starts_with_heading_pattern("Part II Advanced"));
        assert!(starts_with_heading_pattern("III Methods"));
        assert!(!starts_with_heading_pattern("Rasterization"));
        assert!(!starts_with_heading_pattern("With special contribution"));
    }

    #[test]
    fn test_classify_appendix_section() {
        assert_eq!(
            classify_by_pattern("A.1 Proofs", 1),
            (Classification::Section, 2)
        );
    }

    #[test]
    fn test_font_sig_bold_detection() {
        assert!(is_bold_font("NimbusSanL-Bold"));
        assert!(is_bold_font("LucidaBright-Demi"));
        assert!(is_bold_font("CMSSBX10"));
        assert!(!is_bold_font("NimbusSanL-Regu"));
        assert!(!is_bold_font("CMR10"));
    }

    #[test]
    fn test_compute_indent_levels() {
        let xs = vec![90.0, 90.5, 110.0, 110.2, 130.0];
        let levels = compute_indent_levels(&xs);
        assert_eq!(levels.len(), 3);
    }

    #[test]
    fn test_parse_lowercase_roman() {
        assert_eq!(parse_lowercase_roman("xvii"), Some(17));
        assert_eq!(parse_lowercase_roman("iii"), Some(3));
        assert_eq!(parse_lowercase_roman("xxvii"), Some(27));
        assert_eq!(parse_lowercase_roman("iv"), Some(4));
    }

    // ── Test helpers ──

    fn make_split_line(
        title: &str,
        page: Option<i32>,
        font_family: &str,
        font_size: f32,
        bold: bool,
        italic: bool,
    ) -> TocSplitLine {
        TocSplitLine {
            title: title.to_string(),
            page_label: page.map(|p| format!("{}", p.unsigned_abs())),
            page_value: page,
            font_sig: FontSig {
                family: font_family.to_string(),
                size_bucket: (font_size * 2.0).round() as i32,
                is_bold: bold,
                is_italic: italic,
                is_all_caps: false,
            },
            x_left: 90.0,
            page_idx: 0,
            y_center: 100.0,
            has_leader_dots: page.is_some(),
            title_chars: vec![],
        }
    }

    fn make_classified(
        title: &str,
        page_value: i32,
        classification: Classification,
        depth: u32,
    ) -> ClassifiedEntry {
        ClassifiedEntry {
            title: title.to_string(),
            page_label: format!("{}", page_value.unsigned_abs()),
            page_value,
            classification,
            depth,
            font_sig: FontSig {
                family: "CMR".to_string(),
                size_bucket: 20,
                is_bold: false,
                is_italic: false,
                is_all_caps: false,
            },
            x_left: 90.0,
            page_idx: 0,
        }
    }

    // ── Tests for merge_multiline_titles ──

    #[test]
    fn test_merge_single_continuation() {
        // "2.1 Title Start" (no page) → "Continuation" (page 321) → merged
        let lines = vec![
            make_split_line("2.1 Title Start", None, "CMR", 10.0, false, false),
            make_split_line("Continuation", Some(321), "CMR", 10.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].title, "2.1 Title Start Continuation");
        assert_eq!(merged[0].page_value, Some(321));
    }

    #[test]
    fn test_merge_with_attribution_line_before() {
        // attribution (italic) + section title (regular) + continuation (regular)
        // Attribution should be flushed standalone; title + continuation merged.
        let lines = vec![
            make_split_line("Julien Pettré", None, "CMR", 10.0, false, true),
            make_split_line("2.1 Modeling Agent", None, "CMR", 10.0, false, false),
            make_split_line("Decision Process", Some(321), "CMR", 10.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].title, "Julien Pettré");
        assert_eq!(merged[0].page_value, None);
        assert_eq!(merged[1].title, "2.1 Modeling Agent Decision Process");
        assert_eq!(merged[1].page_value, Some(321));
    }

    #[test]
    fn test_merge_triple_line_wrap() {
        // Three lines for one title: first two have no page, third has page
        let lines = vec![
            make_split_line("2.1 Modeling Agent", None, "CMR", 10.0, false, false),
            make_split_line("Navigation Using a", None, "CMR", 10.0, false, false),
            make_split_line("Markov Process", Some(321), "CMR", 10.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 1);
        assert_eq!(
            merged[0].title,
            "2.1 Modeling Agent Navigation Using a Markov Process"
        );
        assert_eq!(merged[0].page_value, Some(321));
    }

    #[test]
    fn test_merge_heading_pattern_breaks_merge() {
        // Buffer has a line, but next line starts with a heading pattern → no merge
        let lines = vec![
            make_split_line("Some Standalone Text", None, "CMR", 10.0, false, false),
            make_split_line("1.1 New Section", Some(42), "CMR", 10.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].title, "Some Standalone Text");
        assert_eq!(merged[0].page_value, None);
        assert_eq!(merged[1].title, "1.1 New Section");
        assert_eq!(merged[1].page_value, Some(42));
    }

    #[test]
    fn test_merge_no_buffer_passthrough() {
        // Line with page number and no buffer → just passed through
        let lines = vec![
            make_split_line("1.1 Background", Some(3), "CMR", 10.0, false, false),
            make_split_line("1.2 Related Work", Some(7), "CMR", 10.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].title, "1.1 Background");
        assert_eq!(merged[1].title, "1.2 Related Work");
    }

    #[test]
    fn test_merge_font_mismatch_no_merge() {
        // Buffer line has a different font from the page-number line → no merge
        let lines = vec![
            make_split_line("Bold Title", None, "CMR-Bold", 12.0, true, false),
            make_split_line("Continuation", Some(100), "CMR", 10.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].title, "Bold Title");
        assert_eq!(merged[0].page_value, None);
        assert_eq!(merged[1].title, "Continuation");
        assert_eq!(merged[1].page_value, Some(100));
    }

    #[test]
    fn test_merge_part_heading_emitted_immediately() {
        // Part heading (no page) should be emitted, not buffered
        let lines = vec![
            make_split_line("Part I Fundamentals", None, "CMR", 12.0, true, false),
            make_split_line("1 Introduction", Some(1), "CMR", 10.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].title, "Part I Fundamentals");
        assert_eq!(merged[0].page_value, None);
        assert_eq!(merged[1].title, "1 Introduction");
    }

    #[test]
    fn test_merge_preserves_first_entry_metadata() {
        // When merging, the first buffer entry's x_left and font_sig are preserved
        let mut line1 = make_split_line("2.1 Title Start", None, "CMR", 10.0, false, false);
        line1.x_left = 110.0; // indented section
        let line2 = make_split_line("Continuation Text", Some(50), "CMR", 10.0, false, false);

        let merged = merge_multiline_titles(vec![line1, line2]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].x_left, 110.0); // preserved from first line
        assert_eq!(merged[0].title, "2.1 Title Start Continuation Text");
    }

    #[test]
    fn test_merge_attribution_bold_title_regular_continuation() {
        // attribution (bold) + title (regular) + continuation (regular)
        // Different fonts for attr vs title/continuation
        let lines = vec![
            make_split_line("Editor Name", None, "NimbusSan", 10.0, true, false),
            make_split_line("2.3 Coupling the MDP", None, "CMR", 9.0, false, false),
            make_split_line("Solver with Crowd", None, "CMR", 9.0, false, false),
            make_split_line("Rendering", Some(333), "CMR", 9.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].title, "Editor Name");
        assert_eq!(merged[0].page_value, None);
        assert_eq!(
            merged[1].title,
            "2.3 Coupling the MDP Solver with Crowd Rendering"
        );
        assert_eq!(merged[1].page_value, Some(333));
    }

    #[test]
    fn test_merge_no_font_data_merges_all() {
        // When all fonts are empty (no font data), everything matches → merge all
        let lines = vec![
            make_split_line("Some Annotation", None, "", 0.0, false, false),
            make_split_line("2.1 Title Start", None, "", 0.0, false, false),
            make_split_line("Continuation", Some(100), "", 0.0, false, false),
        ];
        let merged = merge_multiline_titles(lines);
        // All fonts match (empty), so all get merged into one entry
        assert_eq!(merged.len(), 1);
        assert_eq!(
            merged[0].title,
            "Some Annotation 2.1 Title Start Continuation"
        );
        assert_eq!(merged[0].page_value, Some(100));
    }

    // ── Tests for demote_per_chapter_fm ──

    #[test]
    fn test_demote_bibliography_between_chapters() {
        let mut entries = vec![
            make_classified("1 Introduction", 1, Classification::Chapter, 1),
            make_classified("1.1 Background", 3, Classification::Section, 2),
            make_classified("Bibliography", 19, Classification::FrontMatter, 1),
            make_classified("2 Methods", 21, Classification::Chapter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        assert_eq!(entries[2].classification, Classification::SubEntry);
        assert_eq!(entries[2].depth, 2);
    }

    #[test]
    fn test_demote_references_between_chapters() {
        let mut entries = vec![
            make_classified("1 Introduction", 1, Classification::Chapter, 1),
            make_classified("References", 18, Classification::FrontMatter, 1),
            make_classified("2 Methods", 21, Classification::Chapter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        assert_eq!(entries[1].classification, Classification::SubEntry);
        assert_eq!(entries[1].depth, 2);
    }

    #[test]
    fn test_demote_fm_in_front_matter_not_demoted() {
        // Bibliography with negative page value (actual front matter) stays as FM
        let mut entries = vec![
            make_classified("Preface", -5, Classification::FrontMatter, 1),
            make_classified("Bibliography", -3, Classification::FrontMatter, 1),
            make_classified("1 Introduction", 1, Classification::Chapter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        assert_eq!(entries[1].classification, Classification::FrontMatter);
        assert_eq!(entries[1].depth, 1);
    }

    #[test]
    fn test_demote_bibliography_after_last_chapter() {
        let mut entries = vec![
            make_classified("1 Introduction", 1, Classification::Chapter, 1),
            make_classified("2 Conclusion", 100, Classification::Chapter, 1),
            make_classified("Bibliography", 115, Classification::FrontMatter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        // After last chapter → still demoted (belongs to the last chapter)
        assert_eq!(entries[2].classification, Classification::SubEntry);
        assert_eq!(entries[2].depth, 2);
    }

    #[test]
    fn test_demote_multiple_per_chapter_bibliographies() {
        let mut entries = vec![
            make_classified("1 Intro", 1, Classification::Chapter, 1),
            make_classified("Bibliography", 19, Classification::FrontMatter, 1),
            make_classified("2 Methods", 21, Classification::Chapter, 1),
            make_classified("Bibliography", 39, Classification::FrontMatter, 1),
            make_classified("3 Results", 41, Classification::Chapter, 1),
            make_classified("Bibliography", 58, Classification::FrontMatter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        assert_eq!(entries[1].classification, Classification::SubEntry);
        assert_eq!(entries[3].classification, Classification::SubEntry);
        assert_eq!(entries[5].classification, Classification::SubEntry);
    }

    #[test]
    fn test_demote_non_matching_title_not_demoted() {
        // "Introduction" is FrontMatter but not in the repeating-FM list → stays FM
        let mut entries = vec![
            make_classified("1 Overview", 1, Classification::Chapter, 1),
            make_classified("Introduction", 10, Classification::FrontMatter, 1),
            make_classified("2 Methods", 21, Classification::Chapter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        assert_eq!(entries[1].classification, Classification::FrontMatter);
    }

    #[test]
    fn test_demote_notes_and_references() {
        let mut entries = vec![
            make_classified("1 Intro", 1, Classification::Chapter, 1),
            make_classified("Notes and References", 18, Classification::FrontMatter, 1),
            make_classified("2 Methods", 21, Classification::Chapter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        assert_eq!(entries[1].classification, Classification::SubEntry);
        assert_eq!(entries[1].depth, 2);
    }

    #[test]
    fn test_demote_further_reading() {
        let mut entries = vec![
            make_classified("1 Intro", 1, Classification::Chapter, 1),
            make_classified("Further Reading", 15, Classification::FrontMatter, 1),
            make_classified("2 Methods", 21, Classification::Chapter, 1),
        ];
        demote_per_chapter_fm(&mut entries);
        assert_eq!(entries[1].classification, Classification::SubEntry);
    }

    #[test]
    fn test_demote_respects_part_boundaries() {
        // Part heading (depth 0) also acts as a chapter boundary
        let mut entries = vec![
            make_classified("Part I Foundations", 1, Classification::Part, 0),
            make_classified("1 Intro", 3, Classification::Chapter, 1),
            make_classified("Bibliography", 18, Classification::FrontMatter, 1),
            make_classified("Part II Advanced", 21, Classification::Part, 0),
        ];
        demote_per_chapter_fm(&mut entries);
        // Part (depth 0) counts as chapter boundary, so bibliography is between ch1 and Part II
        assert_eq!(entries[2].classification, Classification::SubEntry);
    }

    // ── Tests for validate_entries ──

    #[test]
    fn test_validate_drops_roman_after_body() {
        let mut entries = vec![
            make_classified("Preface", -3, Classification::FrontMatter, 1),
            make_classified("1 Intro", 1, Classification::Chapter, 1),
            make_classified("Contents", -6, Classification::FrontMatter, 1), // wrong!
            make_classified("2 Methods", 25, Classification::Chapter, 1),
        ];
        validate_entries(&mut entries);
        assert_eq!(entries.len(), 3);
        assert!(entries.iter().all(|e| e.title != "Contents"));
    }

    #[test]
    fn test_validate_drops_backwards_body_page() {
        let mut entries = vec![
            make_classified("1 Intro", 1, Classification::Chapter, 1),
            make_classified("2 Methods", 50, Classification::Chapter, 1),
            make_classified("Spurious", 10, Classification::Section, 2), // backwards
            make_classified("3 Results", 80, Classification::Chapter, 1),
        ];
        validate_entries(&mut entries);
        assert_eq!(entries.len(), 3);
        assert!(entries.iter().all(|e| e.title != "Spurious"));
    }

    #[test]
    fn test_validate_keeps_part_without_page() {
        let mut entries = vec![
            make_classified("1 Intro", 1, Classification::Chapter, 1),
            make_classified("2 End", 50, Classification::Chapter, 1),
        ];
        // Insert a Part with page_value 0
        entries.insert(
            0,
            ClassifiedEntry {
                title: "Part I".to_string(),
                page_label: String::new(),
                page_value: 0,
                classification: Classification::Part,
                depth: 0,
                font_sig: FontSig {
                    family: "CMR".to_string(),
                    size_bucket: 20,
                    is_bold: true,
                    is_italic: false,
                    is_all_caps: false,
                },
                x_left: 90.0,
                page_idx: 0,
            },
        );
        validate_entries(&mut entries);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].title, "Part I");
    }

    // ── Two-column detection tests ──

    /// Build a line of TocChars from text, starting at given X position.
    /// Each char is 6pt wide, 10pt tall, with 1pt gap between chars.
    fn make_char_run(text: &str, x_start: f32, y: f32) -> Vec<TocChar> {
        let char_width = 6.0;
        let char_height = 10.0;
        let char_gap = 1.0;
        let mut chars = Vec::new();
        let mut x = x_start;
        for ch in text.chars() {
            if ch == ' ' {
                x += char_width + char_gap; // space = advance without a char
                continue;
            }
            chars.push(TocChar {
                codepoint: ch,
                bbox: [x, y, x + char_width, y + char_height],
                space_threshold: 3.0,
                font_name: "TestFont".to_string(),
                font_size: 10.0,
                is_italic: false,
            });
            x += char_width + char_gap;
        }
        chars
    }

    /// Build a two-column line by concatenating left and right runs with a gutter gap.
    fn make_two_col_line(left: &str, right: &str, y: f32, gutter_gap: f32) -> Vec<TocChar> {
        let left_chars = make_char_run(left, 50.0, y);
        let left_end = left_chars.last().map(|c| c.bbox[2]).unwrap_or(50.0);
        let right_start = left_end + gutter_gap;
        let mut right_chars = make_char_run(right, right_start, y);
        let mut combined = left_chars;
        combined.append(&mut right_chars);
        combined
    }

    /// Build a single-column line (no gutter).
    fn make_single_col_line(text: &str, y: f32) -> Vec<TocChar> {
        make_char_run(text, 50.0, y)
    }

    #[test]
    fn test_two_col_detection_clear_signal() {
        // Build 20 two-column lines with 80pt gutter gap
        let mut lines: Vec<Vec<TocChar>> = Vec::new();
        for i in 0..20 {
            let y = i as f32 * 15.0;
            let line = make_two_col_line(
                &format!("{}.{} Section Title Here {}", i / 3 + 1, i % 3 + 1, i + 10),
                &format!("{}.{} Another Section Title {}", i / 3 + 5, i % 3 + 1, i + 200),
                y,
                80.0,
            );
            lines.push(line);
        }

        let result = split_two_column_lines(lines, 800.0);
        // Should have 40 lines (20 left + 20 right)
        assert_eq!(result.len(), 40);

        // First 20 should be left column (sorted by Y)
        // Last 20 should be right column (sorted by Y)
        let first_left = &result[0];
        let first_right = &result[20];
        // Left column chars start at x≈50
        assert!(first_left[0].bbox[0] < 200.0);
        // Right column chars start further right
        assert!(first_right[0].bbox[0] > 200.0);
    }

    #[test]
    fn test_two_col_not_triggered_for_single_column() {
        // Build 20 single-column lines
        let mut lines: Vec<Vec<TocChar>> = Vec::new();
        for i in 0..20 {
            let y = i as f32 * 15.0;
            let line = make_single_col_line(
                &format!("{}.{} A Section Title With Some Text {}", i / 3 + 1, i % 3 + 1, i + 10),
                y,
            );
            lines.push(line);
        }

        let original_len = lines.len();
        let result = split_two_column_lines(lines, 800.0);
        // Should remain unchanged — no column split
        assert_eq!(result.len(), original_len);
    }

    #[test]
    fn test_two_col_not_triggered_for_page_numbers() {
        // Build lines where the only large gap is at the right edge (page numbers).
        // The right portion is just a short number (1-3 chars), not another entry.
        let mut lines: Vec<Vec<TocChar>> = Vec::new();
        for i in 0..20 {
            let y = i as f32 * 15.0;
            // Long title on left, short page number on right (with large gap)
            let mut chars = make_char_run(
                &format!("{}.{} A Detailed Section Title Here", i / 3 + 1, i % 3 + 1),
                50.0,
                y,
            );
            let left_end = chars.last().map(|c| c.bbox[2]).unwrap_or(50.0);
            // Page number far to the right (simulating right-aligned page number)
            let page_chars = make_char_run(&format!("{}", i + 100), left_end + 100.0, y);
            chars.extend(page_chars);
            lines.push(chars);
        }

        let original_len = lines.len();
        let result = split_two_column_lines(lines, 800.0);
        // Should NOT be split — the right side has too few chars
        assert_eq!(result.len(), original_len);
    }

    #[test]
    fn test_two_col_mixed_full_and_half_lines() {
        // Realistic scenario: some lines span both columns, some are single-column
        // (wrapped continuation lines). Should still detect two-column layout.
        let mut lines: Vec<Vec<TocChar>> = Vec::new();

        // 12 full two-column lines
        for i in 0..12 {
            let y = i as f32 * 15.0;
            lines.push(make_two_col_line(
                &format!("{}.{} Left Column Entry {}", i / 3 + 1, i % 3 + 1, i + 10),
                &format!("{}.{} Right Column Entry {}", i / 3 + 5, i % 3 + 1, i + 200),
                y,
                80.0,
            ));
        }

        // 8 single-column lines (continuation lines in left column only)
        for i in 12..20 {
            let y = i as f32 * 15.0;
            lines.push(make_single_col_line(
                &format!("Continuation of previous entry {}", i),
                y,
            ));
        }

        let result = split_two_column_lines(lines, 800.0);
        // Should be split: 12 left + 12 right + 8 single = ~32 lines
        // (single-column lines go to left or right based on position)
        assert!(result.len() > 20); // More lines than input means split happened
    }

    #[test]
    fn test_two_col_too_few_lines() {
        // Only 3 lines — below the minimum threshold
        let mut lines: Vec<Vec<TocChar>> = Vec::new();
        for i in 0..3 {
            let y = i as f32 * 15.0;
            lines.push(make_two_col_line("Left Text Here", "Right Text Here", y, 80.0));
        }

        let result = split_two_column_lines(lines, 800.0);
        assert_eq!(result.len(), 3); // Should not split
    }

    #[test]
    fn test_two_col_preserves_y_ordering() {
        // Verify that after splitting, left-column lines are Y-ordered
        // and right-column lines are Y-ordered, with left before right.
        let mut lines: Vec<Vec<TocChar>> = Vec::new();
        for i in 0..15 {
            let y = i as f32 * 15.0;
            lines.push(make_two_col_line(
                &format!("{}.{} Left Entry Number {}", i / 3 + 1, i % 3 + 1, i + 10),
                &format!("{}.{} Right Entry Number {}", i / 3 + 5, i % 3 + 1, i + 200),
                y,
                80.0,
            ));
        }

        let result = split_two_column_lines(lines, 800.0);
        assert_eq!(result.len(), 30);

        // All left column lines (first 15) should have increasing Y
        for i in 1..15 {
            let prev_y = result[i - 1].iter().map(|c| c.bbox[1]).sum::<f32>()
                / result[i - 1].len() as f32;
            let curr_y =
                result[i].iter().map(|c| c.bbox[1]).sum::<f32>() / result[i].len() as f32;
            assert!(curr_y >= prev_y, "Left column not Y-sorted at index {i}");
        }

        // All right column lines (last 15) should have increasing Y
        for i in 16..30 {
            let prev_y = result[i - 1].iter().map(|c| c.bbox[1]).sum::<f32>()
                / result[i - 1].len() as f32;
            let curr_y =
                result[i].iter().map(|c| c.bbox[1]).sum::<f32>() / result[i].len() as f32;
            assert!(curr_y >= prev_y, "Right column not Y-sorted at index {i}");
        }
    }

    #[test]
    fn test_two_col_not_triggered_by_few_gutter_lines() {
        // Mostly single-column lines with only 5 out of 20 having a large gap.
        // Should not trigger two-column detection (below 25% / 8-minimum threshold).
        let mut lines: Vec<Vec<TocChar>> = Vec::new();
        for i in 0..20 {
            let y = i as f32 * 15.0;
            if i < 5 {
                // These 5 lines have a large gap at consistent position
                lines.push(make_two_col_line(
                    "LeftColumn Entry",
                    "RightColumn Entry",
                    y,
                    80.0,
                ));
            } else {
                // Remaining 15 are single-column with no large gaps
                lines.push(make_single_col_line(
                    &format!("{}.{} A Detailed Section Title Here Page Num", i / 3 + 1, i % 3 + 1),
                    y,
                ));
            }
        }

        let original_len = lines.len();
        let result = split_two_column_lines(lines, 800.0);
        // Only 5 out of 20 have gutter gaps — below threshold
        assert_eq!(result.len(), original_len);
    }

    #[test]
    fn test_two_col_header_line_assigned_correctly() {
        // A "Contents" header line at the top, centered, followed by
        // two-column content. Header should be assigned to one column.
        let mut lines: Vec<Vec<TocChar>> = Vec::new();

        // Centered header at y=0 — only in right half (x≈300)
        lines.push(make_char_run("Contents", 300.0, 0.0));

        // Two-column content below
        for i in 0..15 {
            let y = (i + 1) as f32 * 15.0;
            lines.push(make_two_col_line(
                &format!("{}.{} Left Title Text {}", i / 3 + 1, i % 3 + 1, i + 10),
                &format!("{}.{} Right Title Text {}", i / 3 + 5, i % 3 + 1, i + 200),
                y,
                80.0,
            ));
        }

        let result = split_two_column_lines(lines, 800.0);
        // Should be split into ~31 lines (1 header + 15 left + 15 right)
        assert!(result.len() > 16);

        // The "Contents" header should appear somewhere in the result
        let has_contents = result.iter().any(|line| {
            let text: String = line.iter().map(|c| c.codepoint).collect();
            text.contains("Contents")
        });
        assert!(has_contents, "Contents header should be preserved after split");
    }

    #[test]
    fn test_find_toc_pages_with_topmost_y() {
        // Verify that find_toc_pages uses Y-position (not list index) to
        // find the "Contents" header. This is important for two-column layouts
        // where column reordering moves the header away from index 0.
        // (Integration test — just verify the logic doesn't panic with empty input)
        let pages: Vec<(Vec<PdfChar>, f32)> = vec![];
        let result = find_toc_pages(&pages);
        assert!(result.is_empty());
    }
}
