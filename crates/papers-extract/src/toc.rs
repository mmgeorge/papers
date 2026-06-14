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

pub fn uses_local_page_labels(entries: &[TocEntry]) -> bool {
    let local = entries
        .iter()
        .filter(|e| is_local_page_label(&e.page_label))
        .count();
    local >= 3 && local * 2 >= entries.len().max(1)
}

pub fn outline_is_usable(entries: &[TocEntry]) -> bool {
    if entries.is_empty() {
        return false;
    }
    if uses_local_page_labels(entries) {
        return false;
    }
    let fatal = entries
        .iter()
        .filter(|e| toc_title_is_fatally_suspicious(&e.title))
        .count();
    // Allow a small number of falsely-flagged fatal entries: edge cases like
    // version strings "3.30" (looks like section "3.30") can trigger the check.
    // Fail only if more than 3% of entries are fatal (minimum threshold: 10).
    if fatal > 0 && (fatal >= 10 || fatal * 33 > entries.len()) {
        return false;
    }
    let suspicious = entries
        .iter()
        .filter(|e| toc_title_is_suspicious(&e.title))
        .count();
    suspicious * 10 < entries.len().max(1)
}

pub fn renderable_toc_is_usable(entries: &[TocEntry]) -> bool {
    if entries.is_empty() {
        return false;
    }

    let fatal = entries
        .iter()
        .filter(|e| toc_title_is_fatally_suspicious(&e.title))
        .count();
    if fatal * 6 >= entries.len().max(1) {
        return false;
    }

    let suspicious = entries
        .iter()
        .filter(|e| toc_title_is_suspicious(&e.title))
        .count();
    suspicious * 3 < entries.len().max(1)
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
    /// Typographic origin X (from `FPDFText_GetCharOrigin`).
    origin_x: f32,
    /// True if pdfium generated a space character before this char.
    pdfium_space_before: bool,
    space_threshold: f32,
    font_name: String,
    font_size: f32,
    is_italic: bool,
}

impl crate::text_cleanup::HasBbox for TocChar {
    fn bbox(&self) -> [f32; 4] { self.bbox }
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
                let pdfium_space_present = self.pdfium_space_xs.iter().any(|&sx| {
                    sx >= prev.bbox[2] - 0.5 && sx <= ch.bbox[0] + 0.5
                });
                let has_pdfium_space = gap >= threshold * 0.15 && pdfium_space_present;
                // Font-family change with a positive gap is also a word
                // boundary — PDFs often kern across font switches so the
                // gap is below the normal space threshold, but the glyphs
                // still belong to different words (e.g., bold "C++" then
                // regular "Classes").
                // Guard against false positives: don't split before/after
                // punctuation, brackets, or math symbols — those often
                // switch fonts for styling without a word break.
                let prev_ok = prev.codepoint.is_alphanumeric()
                    || prev.codepoint == '+' || prev.codepoint == ')'
                    || prev.codepoint == ']' || prev.codepoint == ','
                    || prev.codepoint == ';' || prev.codepoint == ':';
                let next_ok = ch.codepoint.is_alphanumeric()
                    || ch.codepoint == '"' || ch.codepoint == '\u{201C}';
                let font_boundary_ok = prev_ok && next_ok;
                let font_change_space = gap > 0.3
                    && prev.font_name != ch.font_name
                    && font_boundary_ok;

                // pdfium_space_before: pdfium's own space detection found a word
                // boundary here (via advance-width analysis in cpdf_textpage.cpp).
                // Trust it when the gap is at least slightly positive — this catches
                // cases like "ofGravity" where the gap is below our threshold but
                // pdfium correctly detected the space from advance widths.
                // Guard: require gap > 0 to avoid false spaces from overlapping glyphs.
                // pdfium_space_before: pdfium's own advance-width analysis
                // (cpdf_textpage.cpp) detected a word boundary here. This is
                // the most reliable signal — trust it when the gap is positive.
                let pdfium_indexed_space = ch.pdfium_space_before && gap > 0.0;

                if gap > threshold || has_pdfium_space || font_change_space || pdfium_indexed_space {
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

    /// Like `==` but tolerant of a ±1 size-bucket difference (≤0.5pt average
    /// glyph size). Heading lines that share a style sometimes differ by half a
    /// point — e.g. "Acknowledgments" (bucket 28) and the adjacent "About the
    /// Author" (bucket 27) — and an exact `size_bucket` match would wrongly treat
    /// them as distinct fonts. Family, weight, slant and caps must still match.
    fn matches_tolerant(&self, other: &FontSig) -> bool {
        self.family == other.family
            && self.is_bold == other.is_bold
            && self.is_italic == other.is_italic
            && self.is_all_caps == other.is_all_caps
            && (self.size_bucket - other.size_bucket).abs() <= 1
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
    /// True if the leading chapter number was inferred by
    /// `infer_chapter_numbers_from_children` (not present in the raw PDF TOC).
    chapter_num_inferred: bool,
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

    // Phase 5b: Infer page numbers for chapter entries that have none.
    // Some PDFs have chapter titles on separate lines without page numbers;
    // the page can be inferred from the first child section.
    infer_chapter_pages(&mut entries);

    // Phase 6: Infer chapter numbers from children + resolve bare-letter sections
    infer_chapter_numbers_from_children(&mut entries);
    resolve_bare_letter_sections(&mut entries);

    // Phase 7: Validate
    validate_entries(&mut entries);

    if entries.len() < 3 {
        return None;
    }

    // Phase 7b: Normalize SubEntry depths using indent analysis
    normalize_subentry_depths(&mut entries);

    // Phase 8: Synthesize section numbers for entries that don't have them
    synthesize_section_numbers(&mut entries);

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

fn normalize_toc_header_text(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphabetic())
        .flat_map(|c| c.to_lowercase())
        .collect()
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
            let normalized = normalize_toc_header_text(&text);
            if trimmed.starts_with("list of") || normalized.starts_with("listof") {
                continue;
            }
            // "Contents at a Glance" is a brief summary TOC that duplicates the
            // chapter/appendix headings of the detailed "Table of Contents".
            // Skip it as a start page so the detailed TOC is used instead; the
            // glance page then falls outside the ascending main-TOC run and is
            // not extracted, avoiding duplicated (and phantom wrapped) headings.
            if normalized.contains("ataglance") {
                continue;
            }
            if normalized == "contents"
                || normalized == "tableofcontents"
                || normalized == "detailedcontents"
                || normalized.starts_with("contents")
                || normalized.starts_with("tableofcontents")
            {
                start_page = Some(page_idx as u32);
                break;
            }
            if fallback_page.is_none() && normalized.contains("contents") {
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
    // Track the highest page reference number seen on confirmed TOC pages,
    // used to check page-number continuity for borderline pages.
    let mut max_page_ref: i32 = 0;

    // Helper: extract the maximum trailing page-reference number from lines.
    fn max_trailing_page_number(lines: &[TocRawLine]) -> i32 {
        let mut max_val: i32 = 0;
        for line in lines {
            let text = line.text();
            let trimmed = text.trim();
            if let Some(last) = trimmed.split_whitespace().next_back() {
                if let Ok(n) = last.parse::<i32>() {
                    if n > max_val {
                        max_val = n;
                    }
                }
            }
        }
        max_val
    }

    // Gather max page ref from the start page itself.
    {
        let (chars, page_height) = &page_chars[start as usize];
        if !chars.is_empty() {
            let lines = build_lines_from_chars(chars, *page_height, start);
            max_page_ref = max_trailing_page_number(&lines);
        }
    }

    // Helper: check if a page is a "List of ..." page that terminates the TOC.
    fn is_list_of_page(lines: &[TocRawLine]) -> bool {
        let mut top_idx: Vec<(usize, f32)> = lines
            .iter()
            .enumerate()
            .map(|(i, l)| (i, l.y_center))
            .collect();
        top_idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, _) in top_idx.iter().take(3) {
            let text = lines[idx].text();
            let trimmed = text.trim().to_ascii_lowercase();
            if trimmed.starts_with("list of figures")
                || trimmed.starts_with("list of tables")
                || trimmed.starts_with("list of algorithms")
            {
                return true;
            }
        }
        false
    }

    // Number of consecutive blank pages seen.
    let mut blank_gap: u32 = 0;

    for page_idx in (start + 1)..scan_limit as u32 {
        let (chars, page_height) = &page_chars[page_idx as usize];
        if chars.is_empty() {
            // Allow up to 2 blank page gaps (two-sided printing can leave blanks).
            blank_gap += 1;
            if blank_gap <= 2 {
                continue;
            }
            break;
        }

        let lines = build_lines_from_chars(chars, *page_height, page_idx);

        // Check if this is a "List of Figures/Tables" page (stop).
        if is_list_of_page(&lines) {
            return toc_pages;
        }

        // Check TOC signals
        let non_empty_lines: Vec<&TocRawLine> =
            lines.iter().filter(|l| !l.text().trim().is_empty()).collect();
        if non_empty_lines.is_empty() {
            blank_gap += 1;
            if blank_gap <= 2 {
                continue;
            }
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

        // Strong match: same thresholds as initial detection.
        // Require a minimum number of entries: a single-line "Chapter N"
        // half-title divider page would otherwise pass the page-number ratio
        // (1 line, 100% ending in a number). is_toc_like_page enforces the same
        // >= 4 floor for initial detection; mirror it here. Genuinely sparse
        // final TOC pages (e.g. just "Index 623") still qualify via the
        // continuity-guarded relaxed / ultra-relaxed paths below.
        let strong = non_empty_lines.len() >= 4 && (pct_numbers >= 0.3 || pct_dots >= 0.2);

        // Relaxed match for continuation: lower thresholds, but require
        // page-number continuity (trailing numbers on this page should
        // reference pages at or beyond what we've seen so far — the TOC
        // is monotonically increasing in page references). Also require
        // a minimum absolute count of numbered lines to avoid false
        // positives on body text pages that happen to have a few numbers.
        let page_ref_on_this = max_trailing_page_number(&lines);
        let has_continuity = page_ref_on_this > 0 && page_ref_on_this >= max_page_ref;
        let relaxed = (pct_numbers >= 0.15 || pct_dots >= 0.10)
            && has_continuity
            && lines_with_trailing_number >= 2;

        // Ultra-relaxed: for the last page(s) of a TOC, which may have very
        // few entries (e.g., just appendices and an index). Require page-number
        // continuity and at least 1 numbered line, with very low percentage.
        let ultra_relaxed = !strong
            && !relaxed
            && has_continuity
            && lines_with_trailing_number >= 1
            && (pct_numbers >= 0.08 || lines_with_dots >= 1)
            && blank_gap == 0;

        if strong || relaxed || ultra_relaxed {
            blank_gap = 0;
            toc_pages.push(page_idx);
            if page_ref_on_this > max_page_ref {
                max_page_ref = page_ref_on_this;
            }
        } else {
            // This page isn't TOC-like. Before giving up, look ahead one
            // page: if the NEXT page is strongly TOC-like, skip this gap
            // page (don't add it to toc_pages) and continue extending.
            let next = page_idx + 1;
            if next < scan_limit as u32 {
                let (next_chars, next_height) = &page_chars[next as usize];
                if !next_chars.is_empty() {
                    let next_lines = build_lines_from_chars(next_chars, *next_height, next);
                    if !is_list_of_page(&next_lines)
                        && is_toc_like_page(&page_chars[next as usize], next)
                    {
                        // The next page is TOC-like — skip this gap page
                        // and let the next iteration pick it up.
                        continue;
                    }
                }
            }
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

/// True for an appendix-letter heading fragment: a single uppercase letter
/// followed by a space and a capitalized word ("B Message Crackers, …",
/// "A The Build Environment"). Used to keep such a wrapped heading on an
/// offset-page top band, where its page number sits on the continuation line so
/// the usual number/leader-dot/heading-number signals are absent. A running
/// header ("Table of Contents") fails because its second word is lowercase.
fn is_appendix_letter_heading(text: &str) -> bool {
    let t = text.trim();
    let mut chars = t.chars();
    chars.next().is_some_and(|c| c.is_ascii_uppercase())
        && chars.next() == Some(' ')
        && chars.next().is_some_and(|c| c.is_ascii_uppercase())
}

fn extract_toc_lines(page_chars: &[(Vec<PdfChar>, f32)], toc_pages: &[u32]) -> Vec<TocRawLine> {
    let mut all_lines = Vec::new();

    for &page_idx in toc_pages {
        let (chars, page_height) = &page_chars[page_idx as usize];
        let mut lines = build_lines_from_chars(chars, *page_height, page_idx);

        // Header/footer removal. The page box and the text layer are sometimes
        // vertically offset (cropmarked / oversized pages), so a genuine
        // top-of-page entry can have a small or even negative image-space
        // y_center. An absolute top-margin clip would then delete the first
        // entries of every continuation page (compilers, windows_cpp, handbook,
        // realtime_rendering …). So drop a top-band line only when it is NOT a
        // real TOC entry — a running header — while always keeping lines that
        // carry a trailing page number, leader dots, or a heading-pattern start.
        // The bottom-margin footer clip is kept (footers are also caught by the
        // bare-page-number filter below).
        let margin = page_height * 0.03;
        // Detect whether this page's text layer is vertically offset above the
        // declared page box (cropmarked / oversized pages): genuine entries then
        // have a small or negative image-space y_center. Only on such pages do we
        // relax the absolute top clip; normal pages keep it so ordinary top
        // running headers are still removed (and we don't admit duplicates).
        let is_offset = lines.iter().any(|l| l.y_center < 0.0);
        lines.retain(|line| {
            let y = line.y_center;
            if y > page_height - margin {
                return false; // bottom-margin footer
            }
            if y < margin {
                if !is_offset {
                    return false; // normal page: original absolute top clip
                }
                // Offset page: keep genuine entries; drop running headers. A
                // "Contents" / "Table of Contents" header may end in a roman page
                // number ("TABLE OF CONTENTS xxiii") or be preceded by one ("xxiv
                // TABLE OF CONTENTS"), which would otherwise look like an entry.
                let text = line.text();
                let norm = normalize_toc_header_text(&text);
                let is_contents_header = norm.contains("tableofcontents")
                    || norm.contains("detailedcontents")
                    || norm == "contents";
                if is_contents_header {
                    return false;
                }
                return line_ends_with_number(line)
                    || headings::has_leader_dots(&text)
                    || starts_with_heading_pattern(text.trim())
                    || is_appendix_letter_heading(&text);
            }
            true
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
            let top_normalized = normalize_toc_header_text(&top_text);
            if top_lower == "contents"
                || top_lower == "table of contents"
                || top_lower == "detailed contents"
                || top_lower.starts_with("contents ")
                || top_lower.starts_with("table of contents ")
                || top_normalized == "contents"
                || top_normalized == "tableofcontents"
                || top_normalized == "detailedcontents"
                || top_normalized.starts_with("contents")
                || top_normalized.starts_with("tableofcontents")
                // Page-number-first running headers ("xxiv TABLE OF CONTENTS")
                // normalize to e.g. "xxivtableofcontents".
                || top_normalized.contains("tableofcontents")
                || top_normalized.contains("detailedcontents")
            {
                lines.remove(top_idx);
            }
        }

        all_lines.extend(lines);
    }

    // Drop standalone "Contents" / "Table of Contents" running headers that
    // REPEAT across pages. A repeated pure-"Contents" line (optionally followed
    // by a roman page number) sits at an arbitrary, often far-right x and would
    // otherwise create a spurious indent level and clear the depth-resolution
    // stack. Gated on ≥2 occurrences so a single real "Contents (p. v)"
    // front-matter entry (e.g. in `edo`) is preserved. The topmost-line filter
    // above already handles the once-per-page header in the common case.
    let is_pure_contents = |l: &TocRawLine| {
        let norm = normalize_toc_header_text(&l.text());
        ["contents", "tableofcontents", "detailedcontents"]
            .iter()
            .any(|h| norm.starts_with(h) && norm[h.len()..].chars().all(|c| "ivxlcdm".contains(c)))
    };
    if all_lines.iter().filter(|l| is_pure_contents(l)).count() >= 2 {
        all_lines.retain(|l| !is_pure_contents(l));
    }

    all_lines
}

/// Remove fixed-position DRM "stamp" watermark glyphs from a page's char stream.
///
/// Pearson / Addison-Wesley ebooks stamp each page with a token like
/// "ptg49240929" at a fixed position. pdfium reports the glyphs in its char
/// stream but omits the token from `text.all()`; left in, the glyphs Y-group into
/// whichever TOC row shares their row, corrupting that entry's page number and
/// merging it with a neighbour. We identify the stamp by its token shape — "ptg"
/// followed by 5 or more digits — which no real TOC token matches.
fn strip_stamp_watermark(chars: &[PdfChar]) -> Vec<PdfChar> {
    let n = chars.len();
    let mut drop = vec![false; n];
    let mut i = 0;
    while i < n {
        if matches!(chars[i].codepoint, 'p' | 'P') {
            // Assemble the maximal alphanumeric run starting here.
            let mut j = i;
            let mut token = String::new();
            while j < n && chars[j].codepoint.is_ascii_alphanumeric() {
                token.push(chars[j].codepoint);
                j += 1;
            }
            let lower = token.to_ascii_lowercase();
            if lower.len() >= 8
                && lower.starts_with("ptg")
                && lower[3..].bytes().all(|b| b.is_ascii_digit())
            {
                for slot in drop.iter_mut().take(j).skip(i) {
                    *slot = true;
                }
                i = j;
                continue;
            }
        }
        i += 1;
    }
    chars
        .iter()
        .zip(drop)
        .filter(|(_, dropped)| !*dropped)
        .map(|(c, _)| c.clone())
        .collect()
}

fn build_lines_from_chars(
    chars: &[PdfChar],
    page_height: f32,
    page_idx: u32,
) -> Vec<TocRawLine> {
    if chars.is_empty() {
        return vec![];
    }

    // Strip fixed-position DRM "stamp" watermark glyphs before line grouping, so
    // they never Y-group into a TOC row.
    let filtered = strip_stamp_watermark(chars);
    let chars: &[PdfChar] = &filtered;

    // Collect pdfium space X-positions (image-space) before filtering them out.
    // These mark word boundaries in tightly-kerned fonts where gap detection fails.
    let pdfium_space_positions: Vec<(f32, f32)> = chars
        .iter()
        .filter(|c| c.codepoint == ' ')
        .map(|c| {
            let y = (c.bbox[1] + c.bbox[3]) / 2.0;
            (c.bbox[0], y)
        })
        .collect();


    // Convert to TocChar in image-space.
    // Skip literal spaces (we reconstruct spacing from gaps + pdfium_space_before flag).
    // Expand TeX ligatures (0x0B-0x0F) to their component characters.
    let mut toc_chars: Vec<TocChar> = Vec::new();
    let mut last_was_space = false;
    for idx in 0..chars.len() {
        let c = &chars[idx];
        if c.codepoint == ' ' {
            last_was_space = true;
            continue;
        }
        // Propagate space information: if the previous PdfChar was a literal space
        // (which we skip), mark this char as having a space before it. This
        // supplements the pdfium_space_before flag for cases where the flag wasn't
        // set on the PdfChar but a literal space char existed.
        let has_space_before = c.pdfium_space_before || last_was_space;
        last_was_space = false;
        let y1 = c.bbox[1];
        let y2 = c.bbox[3];
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
                        origin_x: c.origin_x + x_off,
                        pdfium_space_before: li == 0 && has_space_before,
                        space_threshold: c.space_threshold,
                        font_name: c.font_name.clone(),
                        font_size: c.font_size,
                        is_italic: c.is_italic,
                    });
                }
                continue;
            }
        }
        // Recover a wrap hyphen mis-encoded as a control / format code point.
        // This corpus has fonts whose broken ToUnicode maps the line-break
        // hyphen glyph to U+0002 (or U+00AD); it still draws as a real hyphen
        // (non-zero width). Treat it as '-' ONLY when it sits at a line break —
        // the next glyph is on a different line — so a wrapped word rejoins
        // ("Ital-" + "iano" -> "Italiano"). A mid-line U+0002 is a stray marker
        // (e.g. "S , I , R" in a formula) and is dropped as the control char.
        let codepoint = if matches!(c.codepoint, '\u{2}' | '\u{ad}')
            && (c.bbox[2] - c.bbox[0]) > 0.5
        {
            let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
            let line_h = (c.bbox[3] - c.bbox[1]).abs().max(1.0);
            let next_on_new_line = chars[idx + 1..]
                .iter()
                .find(|n| n.codepoint != ' ')
                .map(|n| ((n.bbox[1] + n.bbox[3]) / 2.0 - cy).abs() > line_h * 0.5)
                .unwrap_or(false);
            if next_on_new_line {
                '-'
            } else {
                c.codepoint
            }
        } else {
            c.codepoint
        };
        if codepoint.is_control() {
            continue;
        }
        toc_chars.push(TocChar {
            codepoint,
            bbox: [c.bbox[0], y1, c.bbox[2], y2],
            origin_x: c.origin_x,
            pdfium_space_before: has_space_before,
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
    let y_threshold = avg_height * 0.55;

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
/// Detect a two-column gutter via the line-coverage projection profile.
///
/// For each X bin, count how many lines have a character *spanning* it
/// (Nagy's projection profile). A real column gutter is a near-empty vertical
/// band (few lines cross it — only full-width headers) flanked by two WIDE dense
/// columns (Breuel's column-separator evaluation). The "wide column on both
/// sides" requirement rejects single-column TOCs (whose only interior low-density
/// band is the narrow strip before the right-aligned page numbers); the
/// "balanced / nearest the centre" tie-break picks the true gutter over a
/// column's own internal title→page-number gap. Returns the gutter X if confident.
fn detect_two_column_gutter(lines: &[Vec<TocChar>], page_left: f32, page_right: f32) -> Option<f32> {
    let n = lines.len();
    let span = page_right - page_left;
    if n < 8 || span < 150.0 {
        return None;
    }
    let bw = 3.0f32;
    let nb = ((span / bw).ceil() as usize).max(10);
    let mut cov = vec![0u32; nb];
    for line in lines {
        let mut hit = vec![false; nb];
        for c in line {
            let lo = (((c.bbox[0] - page_left) / bw).floor().max(0.0) as usize).min(nb - 1);
            let hi = (((c.bbox[2] - page_left) / bw).floor().max(0.0) as usize).min(nb - 1);
            for h in hit.iter_mut().take(hi + 1).skip(lo) {
                *h = true;
            }
        }
        for (b, &h) in hit.iter().enumerate() {
            if h {
                cov[b] += 1;
            }
        }
    }
    // A gutter is a deep dip *relative to the column peaks*, not necessarily
    // near-empty: on dense pages wrapped left-titles overflow into the gutter
    // band, so an absolute near-zero threshold misses it. Use 35% of the peak
    // column coverage (with the old absolute 12%-of-rows value as a floor). The
    // alpha-right / both-sides gates below still reject single-column pages.
    let col_density = *cov.iter().max().unwrap_or(&1) as f32;
    let near_zero = (col_density * 0.35).max(n as f32 * 0.12).max(1.0);
    let dense = n as f32 * 0.4;
    let center = nb as f32 / 2.0;
    let (lo, hi) = (nb / 5, nb * 4 / 5);
    // (gutter_x, two_column_row_count, distance_to_centre). We pick the valley
    // that best PARTITIONS the page (Breuel): the true gutter is the one with
    // the most rows carrying real text on *both* sides (left entry + right
    // entry). A spurious within-column valley separates far fewer such rows.
    let mut best: Option<(f32, usize, f32)> = None;
    let mut b = lo;
    while b < hi {
        if (cov[b] as f32) > near_zero {
            b += 1;
            continue;
        }
        let run_start = b;
        while b < hi && (cov[b] as f32) <= near_zero {
            b += 1;
        }
        let mid = (run_start + b) as f32 / 2.0;
        let gutter_x = page_left + mid * bw;
        // Dense-column extents on each side of this near-empty band.
        let left: Vec<usize> = (0..run_start).filter(|&k| cov[k] as f32 >= dense).collect();
        let right: Vec<usize> = (b..nb).filter(|&k| cov[k] as f32 >= dense).collect();
        if let (Some(&l0), Some(&l1), Some(&r0), Some(&r1)) =
            (left.first(), left.last(), right.first(), right.last())
        {
            let left_w = (l1 - l0) as f32 * bw;
            let right_w = (r1 - r0) as f32 * bw;
            // Count rows with real words (≥3 letters) on each side. The right
            // count is also the gate: a single-column TOC's only interior valley
            // sits before the right-aligned page numbers / after the dot
            // leaders, where the right half is just digits and '.' — almost no
            // letters. The both-sides count drives selection.
            let alpha = |line: &Vec<TocChar>, want_right: bool| -> usize {
                line.iter()
                    .filter(|c| {
                        let on_right = (c.bbox[0] + c.bbox[2]) / 2.0 > gutter_x;
                        on_right == want_right && c.codepoint.is_alphabetic()
                    })
                    .count()
            };
            let alpha_right_rows = lines.iter().filter(|l| alpha(l, true) >= 3).count();
            // A genuine two-column row has real words on both sides AND a clear
            // empty band straddling the gutter (the left entry ends, then a gap,
            // then the right entry begins). A long *single-column* title that
            // merely flows across the gutter has alpha on both sides but no gap,
            // so it must NOT count — that false signal is what made the relaxed
            // valley threshold fire on single-column pages (e.g. opt).
            let both_sides_rows = lines
                .iter()
                .filter(|l| {
                    if alpha(l, true) < 3 || alpha(l, false) < 3 {
                        return false;
                    }
                    let left_end = l
                        .iter()
                        .filter(|c| (c.bbox[0] + c.bbox[2]) / 2.0 < gutter_x)
                        .map(|c| c.bbox[2])
                        .fold(f32::MIN, f32::max);
                    let right_start = l
                        .iter()
                        .filter(|c| (c.bbox[0] + c.bbox[2]) / 2.0 >= gutter_x)
                        .map(|c| c.bbox[0])
                        .fold(f32::MAX, f32::min);
                    right_start - left_end >= 6.0
                })
                .count();
            if left_w >= span * 0.2
                && right_w >= span * 0.2
                && alpha_right_rows >= 5
                && (alpha_right_rows as f32) >= n as f32 * 0.25
                // A genuine two-column TOC has many rows carrying real words on
                // BOTH sides (a left entry AND a right entry). A single-column
                // page with an incidental interior dip does not, so this guards
                // the now-relaxed valley threshold against false splits.
                && (both_sides_rows as f32) >= n as f32 * 0.3
            {
                let dist = (mid - center).abs();
                let better = match best {
                    None => true,
                    Some((_, bc, bd)) => both_sides_rows > bc || (both_sides_rows == bc && dist < bd),
                };
                if better {
                    best = Some((gutter_x, both_sides_rows, dist));
                }
            }
        }
    }
    best.map(|(gx, _, _)| gx)
}

/// Split each line into a left- and right-column half at the column gutter, then
/// emit every left half (top to bottom) followed by every right half.
///
/// Rather than slicing at the fixed `gutter_x`, each row is cut at its own
/// largest internal whitespace gap that lies *near* the gutter. This is robust
/// to rows whose right entry starts left of the nominal gutter — e.g. an
/// unindented right-column chapter heading ("6 Differential Analysis") begins
/// further left than the indented subsection rows that set the gutter. Slicing
/// at the gap keeps "6 Differential Analysis" whole on the right instead of
/// leaving its "6" on the left (where it would be misread as a page number).
///
/// A row whose text flows continuously across the gutter (no real gap near it,
/// e.g. a centred "CONTENTS" header) is kept whole.
fn split_lines_at_gutter(lines: Vec<Vec<TocChar>>, gutter_x: f32) -> Vec<Vec<TocChar>> {
    const WINDOW: f32 = 55.0; // how far from the gutter a cut gap may sit
    const MIN_GAP: f32 = 6.0; // a real column gap, not inter-word spacing
    let has_alpha = |chars: &[TocChar]| chars.iter().any(|c| c.codepoint.is_alphabetic());
    let mut left: Vec<(f32, Vec<TocChar>)> = Vec::new();
    let mut right: Vec<(f32, Vec<TocChar>)> = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let y = line.iter().map(|c| (c.bbox[1] + c.bbox[3]) / 2.0).sum::<f32>() / line.len() as f32;
        // Chars are pre-sorted by left edge. Find the widest gap near the gutter
        // whose *right side still contains letters* — i.e. a cut that separates
        // two real entries. This rejects the title→page-number gap of a
        // left-only row (whose right side would be digits only) so the row stays
        // whole, while still pulling an unindented right-column heading whole to
        // the right even when its number sits left of the nominal gutter.
        // Score = (left side ends in a digit, gap width). Preferring a cut whose
        // left char is a digit keeps a left entry's trailing page number with
        // that entry — cutting before it would strand the page, make the entry
        // look like an open heading, and let it absorb the next row.
        let mut best_score = (false, 0.0f32);
        let mut split_at: Option<usize> = None;
        for i in 1..line.len() {
            let gap = line[i].bbox[0] - line[i - 1].bbox[2];
            let mid = (line[i - 1].bbox[2] + line[i].bbox[0]) / 2.0;
            if gap >= MIN_GAP && (mid - gutter_x).abs() <= WINDOW && has_alpha(&line[i..]) {
                let score = (line[i - 1].codepoint.is_numeric(), gap);
                if score > best_score {
                    best_score = score;
                    split_at = Some(i);
                }
            }
        }
        match split_at {
            Some(i) => {
                let mut chars = line;
                let r = chars.split_off(i);
                if !chars.is_empty() {
                    left.push((y, chars));
                }
                if !r.is_empty() {
                    right.push((y, r));
                }
            }
            None => {
                // No qualifying column gap: a row confined to one column or a
                // full-width header. If every char sits right of the gutter it
                // belongs to the right column; otherwise keep it left.
                if line.iter().all(|c| (c.bbox[0] + c.bbox[2]) / 2.0 >= gutter_x) {
                    right.push((y, line));
                } else {
                    left.push((y, line));
                }
            }
        }
    }
    left.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    right.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut result = Vec::with_capacity(left.len() + right.len());
    result.extend(left.into_iter().map(|(_, c)| c));
    result.extend(right.into_iter().map(|(_, c)| c));
    result
}

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

    // Primary: projection-profile gutter detection (Nagy XY-cut / Breuel
    // whitespace-column style). Finds a near-empty vertical band flanked by two
    // wide dense columns — robust where the per-line largest-gap heuristic fails
    // (the title→page-number gaps dominate each line's biggest gap).
    if let Some(gutter_x) = detect_two_column_gutter(&lines, page_left, page_right) {
        return split_lines_at_gutter(lines, gutter_x);
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
    explode_embedded_page_refs(lines.into_iter().map(split_single_line).collect())
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

    // Collect digits from right, stopping at a wide internal x-gap. The page
    // number is a contiguous right-aligned run; a large gap between two digit
    // runs (e.g. "ActionScript 3   475", where "3" is a title token and "475"
    // the page) marks the page-column boundary. Without this stop the title's
    // trailing digit would be swallowed into the page number ("3475").
    let mut i = num_end;
    while i > 0 && chars[i - 1].codepoint.is_ascii_digit() {
        if i < num_end {
            let internal_gap = chars[i].bbox[0] - chars[i - 1].bbox[2];
            if internal_gap >= gap_threshold {
                break;
            }
        }
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
            let prev = &chars[j - 1];
            let g = ch.bbox[0] - prev.bbox[2];
            let thresh = if ch.space_threshold > 0.1 {
                ch.space_threshold
            } else {
                let h = (ch.bbox[3] - ch.bbox[1]).abs();
                if h > 0.5 { h * 0.15 } else { 2.0 }
            };
            let font_change = g > 0.3 && prev.font_name != ch.font_name
                && prev.codepoint.is_alphanumeric() && ch.codepoint.is_alphanumeric();
            if g > thresh || (ch.pdfium_space_before && g > 0.0) || font_change {
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
    let spans = token_spans(trimmed);
    if spans.len() < 2 {
        return None;
    }

    if let Some((start_idx, end_idx, label, value)) = trailing_page_ref_token_group(trimmed, &spans) {
        let title = trimmed[..spans[start_idx].0].trim().to_string();
        if !title.is_empty() {
            return Some((title, label, value));
        }
        if start_idx > 0 {
            let title = trimmed[..spans[start_idx - 1].0].trim().to_string();
            if !title.is_empty() {
                return Some((title, label, value));
            }
        }
        let _ = end_idx;
    }

    let last = &trimmed[spans[spans.len() - 1].0..spans[spans.len() - 1].1];
    let (label, value) = parse_page_ref(last)?;
    let title = trimmed[..spans[spans.len() - 1].0].trim().to_string();
    if title.is_empty() {
        return None;
    }

    Some((title, label, value))
}

fn trailing_page_ref_token_group(
    text: &str,
    spans: &[(usize, usize)],
) -> Option<(usize, usize, String, i32)> {
    let max_group = spans.len().min(4);
    for group_len in (2..=max_group).rev() {
        let start_idx = spans.len() - group_len;
        let candidate = text[spans[start_idx].0..spans[spans.len() - 1].1].trim();
        if let Some((label, value)) = parse_page_ref(candidate) {
            return Some((start_idx, spans.len(), label, value));
        }
    }

    None
}

fn embedded_page_ref_group(
    text: &str,
    spans: &[(usize, usize)],
    idx: usize,
) -> Option<(usize, String, i32)> {
    for end_idx in idx..(idx + 4).min(spans.len()) {
        let candidate = text[spans[idx].0..spans[end_idx].1].trim();
        if let Some((label, value)) =
            parse_page_ref(candidate).or_else(|| parse_ocrish_page_ref_token(candidate))
        {
            return Some((end_idx + 1, label, value));
        }
    }

    None
}

/// Parse a page reference string into (label, value).
/// Arabic → positive, lowercase roman → negative.
fn parse_page_ref(s: &str) -> Option<(String, i32)> {
    let s = s.trim_matches(|c: char| {
        c.is_whitespace()
            || matches!(c, '.' | ',' | ';' | ':' | '·' | '•' | '|' | ')' | '(' | '[' | ']')
    });
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

    if is_local_page_label(&compact) {
        return Some((compact, 0));
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

fn parse_local_page_label_parts(s: &str) -> Option<(u32, u32)> {
    let mut parts = s.split('-');
    let left = parts.next()?.parse::<u32>().ok()?;
    let right = parts.next()?.parse::<u32>().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((left, right))
}

fn normalize_ocrish_page_digits(s: &str) -> Option<String> {
    let mut out = String::new();
    for ch in s.chars() {
        let mapped = match ch {
            '0'..='9' => ch,
            'O' | 'o' => '0',
            'I' | 'l' | '|' => '1',
            'Z' | 'z' => '2',
            'S' | 's' => '5',
            _ => return None,
        };
        out.push(mapped);
    }
    if out.is_empty() || out.len() > 4 {
        return None;
    }
    Some(out)
}

fn parse_ocrish_page_ref_token(s: &str) -> Option<(String, i32)> {
    let normalized = normalize_ocrish_page_digits(s.trim())?;
    let value = normalized.parse::<u32>().ok()?;
    if value == 0 || value >= 10000 {
        return None;
    }
    Some((normalized, value as i32))
}

fn is_local_page_label(s: &str) -> bool {
    parse_local_page_label_parts(s).is_some()
}

fn explode_embedded_page_refs(lines: Vec<TocSplitLine>) -> Vec<TocSplitLine> {
    let mut out = Vec::new();
    for line in lines {
        explode_embedded_page_refs_rec(line, &mut out);
    }
    out
}

fn explode_embedded_page_refs_rec(line: TocSplitLine, out: &mut Vec<TocSplitLine>) {
    let Some((left_title, split_label, split_value, right_title)) =
        find_embedded_page_ref_split(&line.title)
    else {
        out.push(line);
        return;
    };

    let mut left = line.clone();
    left.title = left_title;
    left.page_label = Some(split_label);
    left.page_value = Some(split_value);

    let mut right = line;
    right.title = right_title;

    explode_embedded_page_refs_rec(left, out);
    explode_embedded_page_refs_rec(right, out);
}

fn find_embedded_page_ref_split(title: &str) -> Option<(String, String, i32, String)> {
    let tokens = token_spans(title);
    if tokens.len() < 3 {
        return None;
    }

    for idx in (1..tokens.len() - 1).rev() {
        let Some((right_start_idx, label, value)) = embedded_page_ref_group(title, &tokens, idx) else {
            continue;
        };
        let left = title[..tokens[idx].0].trim();
        if right_start_idx >= tokens.len() {
            continue;
        }
        let right = title[tokens[right_start_idx].0..].trim();
        if left.is_empty() || right.is_empty() {
            continue;
        }
        if looks_like_toc_entry_start(left) && looks_like_toc_entry_start(right) {
            return Some((left.to_string(), label, value, right.to_string()));
        }
    }

    None
}

fn token_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(s) = start.take() {
                spans.push((s, idx));
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

fn looks_like_toc_entry_start(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    if starts_with_heading_pattern(t) || is_part_pattern(t) {
        return true;
    }
    if t.starts_with("Appendix ") || t.starts_with("APPENDIX ") || t.starts_with("APPENDICES") {
        return true;
    }
    if headings::is_frontmatter_heading(t) {
        return true;
    }
    let lower = t.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "references"
            | "addenda"
            | "final exercises"
            | "index"
            | "index of names"
            | "index of definitions"
            | "index of symbols"
            | "exercises"
    )
}

fn toc_title_is_suspicious(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return true;
    }
    let tokens = token_spans(trimmed);
    let heading_like = heading_start_count(trimmed);
    if heading_like >= 2 {
        return true;
    }
    for idx in 0..tokens.len().saturating_sub(1) {
        let token = &trimmed[tokens[idx].0..tokens[idx].1];
        if parse_page_ref(token).is_some() {
            let candidate = trimmed[tokens[idx + 1].0..].trim();
            if looks_like_toc_entry_start(candidate) {
                return true;
            }
        }
    }
    false
}

fn toc_title_is_fatally_suspicious(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return true;
    }

    if heading_start_count(trimmed) >= 2 {
        return true;
    }

    let tokens = token_spans(trimmed);
    if tokens.len() >= 2 {
        let first = &trimmed[tokens[0].0..tokens[0].1];
        let rest = trimmed[tokens[1].0..].trim();
        if parse_page_ref(first).is_some() && looks_like_toc_entry_start(rest) {
            // Don't flag "N M" where M is a bare number — that's more likely
            // "chapter N at page M" than a merged entry. A real merged entry
            // would have the rest as a multi-token title (e.g. "42 Introduction").
            let rest_is_bare_number = rest.split_whitespace().count() == 1
                && rest.chars().all(|c| c.is_ascii_digit());
            if !rest_is_bare_number {
                return true;
            }
        }
        if first.chars().all(|c| c.is_ascii_digit())
            && (rest.starts_with("PART ") || rest.starts_with("Part "))
        {
            return true;
        }
    }

    false
}

/// Returns true if `text` starts with a NUMERIC heading identifier:
/// a digit sequence (section number like "1.2") or a roman numeral.
/// Does NOT match textual patterns like "Chapter N" or "Part I" —
/// those are heading keywords that may legitimately appear at the end
/// of a title as cross-references (e.g. "Summary of Chapter 1").
fn starts_with_numeric_heading(text: &str) -> bool {
    let t = text.trim();
    // Leading digit(s): section number like "1", "1.2", "1.2.3"
    if let Some(first) = t.chars().next() {
        if first.is_ascii_digit() {
            let after_digits = t
                .char_indices()
                .find(|(_, c)| !c.is_ascii_digit())
                .map(|(i, c)| (i, c));
            let is_section_num = match after_digits {
                None => true,
                Some((_, ' ')) => true,
                Some((dot_i, '.')) => {
                    let after_dot = &t[dot_i + 1..];
                    let next_non_digit = after_dot.char_indices().find(|(_, c)| !c.is_ascii_digit());
                    matches!(next_non_digit, None | Some((_, ' ' | '.')))
                }
                _ => false,
            };
            if is_section_num {
                return true;
            }
        }
    }
    // Roman numeral start (uppercase) — require what follows to look like a title,
    // not just punctuation ("X / . ." is a math expression, not a section heading).
    if t.starts_with(|c: char| matches!(c, 'I' | 'V' | 'X' | 'L')) {
        if let Some(space_pos) = t.find(' ') {
            let prefix = &t[..space_pos];
            if headings::is_roman_numeral(prefix) {
                // Check what comes after: must be alphabetic, digit, "." separator
                // (for "II. Title"), or end of string.
                let after_space = t[space_pos + 1..].trim_start();
                let after_first_char = after_space.chars().next();
                let ok = match after_first_char {
                    None => true,                        // just the numeral alone
                    Some('.') => {
                        // "II. Title" — dot separator, then alphabetic
                        let after_dot = after_space[1..].trim_start();
                        after_dot.chars().next().map_or(false, |c| c.is_alphabetic() || c.is_ascii_digit())
                    }
                    Some(c) => c.is_alphabetic() || c.is_ascii_digit(), // "II Title"
                };
                if ok {
                    return true;
                }
            }
        }
    }
    false
}

fn heading_start_count(text: &str) -> usize {
    let trimmed = text.trim();
    let tokens = token_spans(trimmed);
    tokens
        .iter()
        .enumerate()
        .filter(|(idx, (start, _))| {
            if *idx == 0 {
                return true;
            }
            let candidate = trimmed[*start..].trim();
            // A non-initial heading start is only counted when:
            //  1. the candidate has at least 2 tokens
            //  2. it starts with a NUMERIC heading identifier (digits/roman)
            //  3. the preceding token is NOT "PART"/"Chapter" etc.
            //     (avoids "PART I . Title" → "I . Title" being a false heading start)
            let candidate_token_count = candidate.split_whitespace().count();
            if candidate_token_count < 2 || !starts_with_numeric_heading(candidate) {
                return false;
            }
            // Skip if immediately preceded by a structural keyword that owns the numeral
            let prev_token = &trimmed[tokens[idx - 1].0..tokens[idx - 1].1];
            let prev_lower = prev_token.to_ascii_lowercase();
            !matches!(prev_lower.as_str(), "part" | "chapter" | "appendix" | "section")
        })
        .count()
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
        // A wrapped title broke at a hyphen. Drop the wrap hyphen AND any space
        // `TocRawLine::text()` inserted just before it (the right-margin advance
        // gap reads as a word boundary), otherwise "Ital-" wrapped onto "iano"
        // rejoins as "Ital iano" instead of "Italiano".
        let base = left.trim_end_matches('-').trim_end();
        // Lowercase continuation = mid-word line break: join directly
        // ("Ital" + "iano" = "Italiano"). Uppercase continuation = a real
        // compound hyphenated at the wrap: keep one hyphen, no spaces
        // ("Chung" + "Kuan" = "Chung-Kuan").
        if right.starts_with(|c: char| c.is_lowercase()) {
            format!("{}{}", base, right)
        } else {
            format!("{}-{}", base, right)
        }
    } else if left.ends_with('—') || left.ends_with('–') {
        // A title wrapped immediately after an em/en-dash ("First Law of
        // Thermodynamics—" + "The Energy Equation"): the dash binds the two
        // phrases directly, so join without an inserted space.
        format!("{}{}", left, right)
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
                // A line that begins a new heading must never be merged into the
                // buffered fragment. Besides numeric / Chapter / Part starts, an
                // "APPENDIX X" line is a standalone top-level entry (it must not be
                // glued onto a preceding section, as happens across a column gutter).
                let t = line.title.trim_start();
                let starts_with_pattern = starts_with_heading_pattern(&line.title)
                    || t.starts_with("APPENDIX ")
                    || t.starts_with("Appendix ");
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

                    // If font-based merge failed but the last buffer entry is an
                    // incomplete heading, force-merge it with the current line
                    // regardless of font. Two cases:
                    //   - a bare chapter/section number ("3", "14.") — always a
                    //     prefix belonging to the following title;
                    //   - an "open heading" — a fragment that begins with a heading
                    //     number / Chapter / Part / APPENDIX label but carries no
                    //     page number ("3 Elementary Fluid Dynamics—The", "26
                    //     Approximate Geometric Query Structures"). Its page sits on
                    //     the continuation line, and the large chapter-number glyph
                    //     (or a different author-line font on the continuation)
                    //     skews the FontSig away from an exact match.
                    if merge_start >= buffer.len() {
                        if let Some(last_buf) = buffer.last() {
                            let trimmed_buf = last_buf.title.trim();
                            let is_bare_number = !trimmed_buf.is_empty()
                                && trimmed_buf
                                    .trim_end_matches('.')
                                    .chars()
                                    .all(|c| c.is_ascii_digit());
                            let is_open_heading = last_buf.page_label.is_none()
                                && starts_with_heading_pattern(trimmed_buf);
                            // A lone uppercase letter with no page is an appendix
                            // letter wrapped above its title ("A" / "Error
                            // Handling 999" → "A Error Handling").
                            let is_bare_letter = last_buf.page_label.is_none()
                                && trimmed_buf.chars().count() == 1
                                && trimmed_buf
                                    .chars()
                                    .next()
                                    .is_some_and(|c| c.is_ascii_uppercase());
                            if is_bare_number || is_open_heading || is_bare_letter {
                                merge_start = buffer.len() - 1;
                            }
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
            } else if buffer.is_empty()
                && !starts_with_heading_pattern(&line.title)
                && result
                    .last()
                    .is_some_and(|last| last.page_label.is_none() && is_part_pattern(&last.title))
            {
                // A no-page, non-heading line immediately after a Part heading is
                // the Part's wrapped title tail ("Part III … Between" + "Programs").
                // Absorb it so it is not mistaken for the next chapter's prefix
                // word. The chapter number that follows ("10") is a heading
                // pattern and is excluded.
                if let Some(last) = result.last_mut() {
                    last.title = join_title_parts(last.title.trim(), line.title.trim());
                }
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
    // N.N or N — but only if the digit prefix is a valid section number.
    // A section number ends with a space, dot, or end-of-string.
    // Exclude:
    //   "5×2"      — digit followed by non-dot/non-space (×)
    //   "2.0/GLSL" — looks like a section "2." but the suffix has "/" not space
    if let Some(first) = t.chars().next() {
        if first.is_ascii_digit() {
            // Find the end of the leading digit run
            let after_digits = t
                .char_indices()
                .find(|(_, c)| !c.is_ascii_digit())
                .map(|(i, c)| (i, c));
            let is_heading_num = match after_digits {
                None => true,          // entire string is digits: "42"
                Some((_, ' ')) => true, // "1 Title"
                Some((dot_i, '.')) => {
                    // "1.2 Title" or "1.2.3" — but NOT "2.0/GLSL"
                    // Verify that after the dot we have digits then space/dot/end
                    let after_dot = &t[dot_i + 1..];
                    let next_non_digit = after_dot
                        .char_indices()
                        .find(|(_, c)| !c.is_ascii_digit());
                    matches!(
                        next_non_digit,
                        None | Some((_, ' ' | '.'))
                    )
                }
                _ => false, // "5×2" or similar — not a heading number
            };
            if is_heading_num {
                return true;
            }
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
        let mut line = line.clone();
        line.title = trim_to_first_heading_start(&line.title);
        line.title = space_after_section_number(&line.title);
        if line.page_label.is_none() && !is_part_pattern(&line.title) {
            // No page number and not a Part heading.
            // Keep entries that look like chapter headings ("N Title" or
            // "Chapter N Title") — they may get page numbers inferred from
            // their first child section later. Only keep if the title is
            // reasonably short (real chapter titles, not body paragraphs).
            let t = line.title.trim();
            // Check it looks like a real chapter title: starts with a digit,
            // is reasonably short, and doesn't contain watermark/garbage patterns.
            let has_watermark = t.contains("ptg") || t.contains("PTG");
            let looks_like_chapter = t.len() < 120 && !has_watermark && {
                let (class, _) = classify_by_pattern(t, last_classified_depth);
                matches!(class, Classification::Chapter)
            };
            if !looks_like_chapter {
                continue;
            }
        }
        let (class, depth) = classify_by_pattern(&line.title, last_classified_depth);
        if class != Classification::Unknown {
            last_classified_depth = depth;
        }
        entries.push((line, class, depth));
    }

    // Post-classification: propagate FrontMatter to adjacent Unknown entries
    // with the same font signature and indent. E.g., if "Preface" is FrontMatter
    // and "About the Authors" is Unknown but has the same font+indent, promote it.
    propagate_frontmatter(&mut entries);

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

    // Strip the literal "Appendix "/"APPENDIX " prefix, keeping the bare letter id
    // ("Appendix A Separate Compilation" → "A Separate Compilation"), mirroring the
    // CHAPTER strip. resolve_bare_letter_sections then classifies the letter. Only
    // strip when a single uppercase letter id follows, so "Appendices" and prose
    // "Appendix" sentences are untouched.
    for (line, _, _) in &mut entries {
        let t = line.title.trim();
        if let Some(rest) = t
            .strip_prefix("Appendix ")
            .or_else(|| t.strip_prefix("APPENDIX "))
        {
            let mut rc = rest.chars();
            let is_letter_id = matches!(rc.next(), Some(c) if c.is_ascii_uppercase())
                && matches!(rc.next(), Some(' ') | Some('.'));
            if is_letter_id {
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
    // Normalize x_left per page by a registered baseline so indent positions are
    // comparable across pages with different margins AND continuation pages that
    // lack a chapter heading (see `compute_page_offsets`). Indent levels are
    // count-aware so a stray far-right entry cannot create a spurious deep level.
    let page_min_x: HashMap<u32, f32> = compute_page_offsets(&entries);
    let x_lefts_normalized: Vec<f32> = entries
        .iter()
        .map(|(l, _, _)| {
            let page_min = page_min_x.get(&l.page_idx).copied().unwrap_or(0.0);
            l.x_left - page_min
        })
        .collect();
    let indent_levels = compute_indent_levels_counted(&x_lefts_normalized, 3);

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

    // Third pass: resolve Unknown entries by walking sequentially using
    // indent + font size.  For each Unknown entry, search a context stack
    // (built from classified entries) for the right parent:
    //   - more indented → child of that stack entry
    //   - same indent but smaller font → child (subsection)
    //   - same indent and same/larger font → sibling (pop and keep looking)
    //   - stack empty → top level (depth 1)
    resolve_unknowns_sequential(&mut entries, &page_min_x, &indent_levels);

    let mut result: Vec<ClassifiedEntry> = entries
        .into_iter()
        .map(|(line, class, depth)| {
            ClassifiedEntry {
                title: line.title,
                page_label: line.page_label.unwrap_or_default(),
                page_value: line.page_value.unwrap_or(0),
                classification: class,
                depth,
                font_sig: line.font_sig,
                x_left: line.x_left,
                page_idx: line.page_idx,
                chapter_num_inferred: false,
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

/// Insert a missing space between a multi-level section number and the title
/// text ("10.5.10Modular Variable Expansion" → "10.5.10 Modular Variable
/// Expansion"). Tight kerning in some PDFs drops the space between a wide section
/// number and the first title word. Only fires for a *dotted* number followed
/// immediately by a letter, so "H2O" / "3D" are left untouched.
fn space_after_section_number(title: &str) -> String {
    let bytes = title.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_digit() {
        return title.to_string();
    }
    let mut i = 0;
    let mut dots = 0;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        if bytes[i] == b'.' {
            dots += 1;
        }
        i += 1;
    }
    if dots >= 1
        && i < bytes.len()
        && bytes[i - 1].is_ascii_digit()
        && bytes[i].is_ascii_alphabetic()
    {
        format!("{} {}", &title[..i], &title[i..])
    } else {
        title.to_string()
    }
}

fn trim_to_first_heading_start(title: &str) -> String {
    let trimmed = strip_leading_page_ref_garbage(title.trim());
    if trimmed.is_empty() || looks_like_toc_entry_start(trimmed) {
        return trimmed.to_string();
    }

    // Strip at most ONE leading token to reach a heading start, and only when the
    // remainder is a *textual* heading — a front-matter heading or a keyword
    // ("Series Foreword" → "Foreword", "A.4 References" → "References"). We never
    // trim to a remainder that merely starts with a number: a leading real word
    // followed by a number is part of the title ("Windows 10 and future Windows
    // versions" must not become "10 and future Windows versions"), and a genuine
    // leading page-number garbage token is already removed by
    // strip_leading_page_ref_garbage above. Trimming deeper than one token would
    // amputate compound titles ("Notes and References", "Proof of Theorem 4.3").
    let tokens = token_spans(trimmed);
    if tokens.len() >= 2 {
        let candidate = trimmed[tokens[1].0..].trim();
        if looks_like_toc_entry_start(candidate)
            && !candidate.starts_with(|c: char| c.is_ascii_digit())
        {
            return candidate.to_string();
        }
    }

    trimmed.to_string()
}

fn strip_leading_page_ref_garbage(title: &str) -> &str {
    let trimmed = title.trim();
    let tokens = token_spans(trimmed);
    if tokens.len() < 2 {
        return trimmed;
    }

    for idx in 0..tokens.len().saturating_sub(1).min(3) {
        let prefix = trimmed[..tokens[idx].1].trim();
        let rest = trimmed[tokens[idx + 1].0..].trim();
        let prefix_looks_page = parse_page_ref(prefix).is_some() || parse_ocrish_page_ref_token(prefix).is_some();
        if prefix_looks_page && looks_like_toc_entry_start(rest) {
            return rest;
        }
    }

    trimmed
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

    // Rule 9: Sub-entry keywords (Summary, Exercises, Bibliography, etc.)
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

/// Like `compute_indent_levels` but drops levels backed by fewer than
/// `min_count` entries. A single far-right entry (a page-number-column artifact
/// or a two-column right-edge fragment) would otherwise create a spurious deep
/// indent level that corrupts depth inference. Never returns empty if the
/// unfiltered set was non-empty.
fn compute_indent_levels_counted(x_lefts: &[f32], min_count: usize) -> Vec<f32> {
    let levels = compute_indent_levels(x_lefts);
    if levels.len() <= 1 {
        return levels;
    }
    let mut counts = vec![0usize; levels.len()];
    for &x in x_lefts {
        counts[quantize_indent(x, &levels) as usize] += 1;
    }
    let mut filtered: Vec<f32> = Vec::new();
    for (i, &l) in levels.iter().enumerate() {
        if counts[i] >= min_count {
            filtered.push(l);
        }
    }
    if filtered.is_empty() {
        levels
    } else {
        filtered
    }
}

/// Per-page x baseline that registers continuation pages to a canonical indent
/// grid (offset-invariant depth). A naïve per-page *minimum* fails on a
/// continuation page that carries only a chapter's sections (no chapter
/// heading): its leftmost entry is at the *section* level, so subtracting the
/// page min collapses every entry up a depth. Books also shift recto/verso pages
/// by a binding margin (Windows Internals offsets odd pages +30pt). So:
///   - Pages with a Chapter/Part heading anchor on that heading's x (the base
///     indent, which already encodes the page's physical margin).
///   - Continuation pages are aligned by the offset that best maps their indent
///     levels onto the canonical grid (inter-level spacings are stable; only the
///     absolute offset differs). Pages that do not align cleanly fall back to
///     their own minimum.
fn compute_page_offsets(entries: &[(TocSplitLine, Classification, u32)]) -> HashMap<u32, f32> {
    let mut page_min: HashMap<u32, f32> = HashMap::new();
    let mut pages: Vec<u32> = Vec::new();
    for (l, _, _) in entries {
        if !page_min.contains_key(&l.page_idx) {
            pages.push(l.page_idx);
        }
        let m = page_min.entry(l.page_idx).or_insert(f32::MAX);
        *m = m.min(l.x_left);
    }
    // Anchor each page on its CHAPTER level only. Parts/appendix dividers sit
    // further left than chapters (one indent above), so anchoring a page on its
    // part would shift that page's baseline relative to chapter-anchored pages
    // and smear the indent levels. Pages without a chapter (part-only divider
    // pages, appendix continuations, mid-chapter continuations) are instead
    // REGISTERED below by aligning their section/subsection levels to the canon.
    let mut page_offset: HashMap<u32, f32> = HashMap::new();
    for (l, class, _) in entries {
        if *class == Classification::Chapter {
            let e = page_offset.entry(l.page_idx).or_insert(f32::MAX);
            *e = e.min(l.x_left);
        }
    }
    if page_offset.is_empty() {
        return page_min; // no chapter anchor — keep per-page minimum
    }
    let canon_xs: Vec<f32> = entries
        .iter()
        .filter(|(l, _, _)| page_offset.contains_key(&l.page_idx))
        .map(|(l, _, _)| l.x_left - page_offset[&l.page_idx])
        .collect();
    let canon = compute_indent_levels_counted(&canon_xs, 3);
    for &p in &pages {
        if page_offset.contains_key(&p) {
            continue;
        }
        let fallback = page_min.get(&p).copied().unwrap_or(0.0);
        let page_xs: Vec<f32> = entries
            .iter()
            .filter(|(l, _, _)| l.page_idx == p)
            .map(|(l, _, _)| l.x_left)
            .collect();
        let page_levels = compute_indent_levels_counted(&page_xs, 2);
        let mut best_delta = fallback;
        let mut best_err = f32::MAX;
        for &pl in &page_levels {
            for &cl in &canon {
                let delta = pl - cl;
                let err: f32 = page_levels
                    .iter()
                    .map(|&x| canon.iter().map(|&c| (x - delta - c).abs()).fold(f32::MAX, f32::min))
                    .sum();
                if err < best_err {
                    best_err = err;
                    best_delta = delta;
                }
            }
        }
        let avg_err = best_err / page_levels.len().max(1) as f32;
        page_offset.insert(p, if avg_err <= 6.0 { best_delta } else { fallback });
    }
    page_offset
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
/// Resolve Unknown entries by walking sequentially, using a context stack of
/// classified entries.  Uses indent (normalized per page) and font size to
/// determine parent/child relationships:
///   - more indented than a stack entry → child
///   - same indent but smaller font   → child (subsection of a chapter title)
///   - same indent, same/larger font  → sibling (pop and look higher)
///   - stack empty                     → top level (depth 1)
fn resolve_unknowns_sequential(
    entries: &mut [(TocSplitLine, Classification, u32)],
    page_min_x: &HashMap<u32, f32>,
    indent_levels: &[f32],
) {
    // Stack of (indent_level, font_size_bucket, depth) from classified entries
    let mut stack: Vec<(u32, i32, u32)> = Vec::new();

    for i in 0..entries.len() {
        let page_min = page_min_x.get(&entries[i].0.page_idx).copied().unwrap_or(0.0);
        let norm_x = entries[i].0.x_left - page_min;
        let indent = if indent_levels.is_empty() {
            0
        } else {
            quantize_indent(norm_x, indent_levels)
        };
        let font = entries[i].0.font_sig.size_bucket;

        if entries[i].1 != Classification::Unknown {
            let depth = entries[i].2;
            // Pop stack entries at same or deeper depth
            while stack.last().map_or(false, |s| s.2 >= depth) {
                stack.pop();
            }
            stack.push((indent, font, depth));
            continue;
        }

        // Unknown entry — find parent in stack (deepest to shallowest)
        let mut parent_depth = None;
        for j in (0..stack.len()).rev() {
            let (sindent, sfont, sdepth) = stack[j];
            if indent > sindent {
                // More indented → child of this entry
                parent_depth = Some(sdepth);
                break;
            } else if indent == sindent {
                if font < sfont {
                    // Smaller font at same indent → child
                    parent_depth = Some(sdepth);
                    break;
                }
                // Same or larger font → sibling; keep searching up for parent
            }
            // Less indented or sibling → keep searching up
        }

        let d = parent_depth.map(|pd| pd + 1).unwrap_or(1);
        entries[i].2 = d;
        entries[i].1 = depth_to_classification(d);

        // Update stack
        while stack.last().map_or(false, |s| s.2 >= d) {
            stack.pop();
        }
        stack.push((indent, font, d));
    }
}

/// Propagate FrontMatter classification to adjacent Unknown entries that share
/// the same font signature and indent (within 5pt). Scans outward from each
/// FrontMatter entry to catch nearby entries like "About the Authors" next to
/// "Preface".
fn propagate_frontmatter(entries: &mut [(TocSplitLine, Classification, u32)]) {
    // Only propagate from FrontMatter entries that appear before the first
    // structural (Chapter/Part) entry in the TOC. FrontMatter-classified entries
    // within the body (e.g. "REFERENCES" at the end of a chapter, "INDEX" mid-TOC)
    // must NOT propagate — their backward/forward scans would incorrectly promote
    // chapter subsections. The second pass (BackMatter scan) handles genuine
    // back-matter entries at the end of the TOC.
    let first_chapter_idx = entries
        .iter()
        .position(|(_, c, _)| matches!(c, Classification::Chapter | Classification::Part));

    let fm_indices: Vec<usize> = entries
        .iter()
        .enumerate()
        .filter(|(_, (_, class, _))| *class == Classification::FrontMatter)
        .map(|(i, _)| i)
        .collect();

    for fm_idx in fm_indices {
        // Skip FM entries that appear after the first chapter — they are within
        // the chapter body and should not propagate to adjacent unknowns.
        if first_chapter_idx.map_or(false, |fc| fm_idx >= fc) {
            continue;
        }

        let fm_font = entries[fm_idx].0.font_sig.clone();
        let fm_x = entries[fm_idx].0.x_left;
        let fm_depth = entries[fm_idx].2;

        // Scan forward from the FrontMatter entry
        for i in (fm_idx + 1)..entries.len() {
            let (ref line, ref class, _) = entries[i];
            if *class != Classification::Unknown {
                break; // stop at next classified entry
            }
            if line.font_sig.matches_tolerant(&fm_font) && (line.x_left - fm_x).abs() < 5.0 {
                entries[i].1 = Classification::FrontMatter;
                entries[i].2 = fm_depth;
            } else {
                break; // different font or indent — stop
            }
        }

        // Scan backward
        for i in (0..fm_idx).rev() {
            let (ref line, ref class, _) = entries[i];
            if *class != Classification::Unknown {
                break;
            }
            if line.font_sig.matches_tolerant(&fm_font) && (line.x_left - fm_x).abs() < 5.0 {
                entries[i].1 = Classification::FrontMatter;
                entries[i].2 = fm_depth;
            } else {
                break;
            }
        }
    }

    // Second pass: scan backward from the end of the TOC to catch back matter
    // entries (e.g., References, Index) that share font+indent with front matter.
    // These may not be adjacent to a FM entry, so the forward/backward scan above
    // won't reach them.
    if let Some((fm_line, _, fm_depth)) = entries
        .iter()
        .find(|(_, class, _)| *class == Classification::FrontMatter)
    {
        let fm_font = fm_line.font_sig.clone();
        let fm_x = fm_line.x_left;
        let fm_depth = *fm_depth;

        for i in (0..entries.len()).rev() {
            let (ref line, ref class, _) = entries[i];
            // Stop when we hit a classified entry that isn't FM/BackMatter
            // (i.e., we've reached the body content)
            match class {
                Classification::FrontMatter | Classification::BackMatter => continue,
                Classification::Unknown => {}
                _ => break,
            }
            if line.font_sig == fm_font && (line.x_left - fm_x).abs() < 5.0 {
                entries[i].1 = Classification::BackMatter;
                entries[i].2 = fm_depth;
            }
        }
    }
}

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

    // Find indices of chapter-level (depth ≤ 1) entries to define chapter boundaries.
    // Exclude FrontMatter/BackMatter — they're not chapter boundaries.
    let chapter_starts: Vec<(usize, i32)> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.depth <= 1
                && e.page_value > 0
                && !matches!(
                    e.classification,
                    Classification::FrontMatter | Classification::BackMatter
                )
        })
        .map(|(i, e)| (i, e.page_value))
        .collect();

    // Count how many entries carry each repeating-FM title. A title that appears
    // multiple times is a per-chapter pattern (each chapter ends with its own
    // References/Bibliography); a title that appears once is book-level back
    // matter. This lets us safely demote a trailing per-chapter entry after the
    // LAST chapter (which has no following chapter boundary) while leaving a lone
    // book-level bibliography at depth 1.
    let mut title_counts: HashMap<String, usize> = HashMap::new();
    for e in entries.iter() {
        let tl = e.title.trim().to_ascii_lowercase();
        if repeating_fm_titles.iter().any(|&t| t == tl) {
            *title_counts.entry(tl).or_default() += 1;
        }
    }

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
            // Trailing per-chapter title after the last chapter (no following
            // boundary): demote only when this title repeats (a per-chapter
            // pattern), so a lone book-level bibliography stays at depth 1.
            (Some((_pi, pp)), None) => {
                pv >= *pp && title_counts.get(&title_lower).copied().unwrap_or(0) >= 2
            }
            _ => false,
        };

        if is_between_chapters {
            entries[i].classification = Classification::SubEntry;
            entries[i].depth = 2; // section-level, under the chapter
        }
    }
}

/// Normalize SubEntry depth: if a SubEntry (Summary, Exercises, etc.) was
/// assigned depth > 2 because `last_depth` was a deep subsection, promote it
/// to depth 2. But if the preceding entry at a lower depth is a section-level
/// SubEntry (like "Summary" at depth 2), keep the current SubEntry nested.
///
/// The key signal: look at the nearest preceding entry that is NOT a SubEntry.
/// If that entry is at depth 3+ (a subsection), promote this SubEntry to 2.
/// If that entry is at depth 2 (a section), this SubEntry is correctly nested.
fn normalize_subentry_depths(entries: &mut [ClassifiedEntry]) {
    for i in 0..entries.len() {
        if entries[i].classification != Classification::SubEntry || entries[i].depth <= 2 {
            continue;
        }
        // Look backward for the nearest non-SubEntry entry (its depth + indent).
        let mut nearest_non_sub: Option<(u32, f32)> = None;
        for j in (0..i).rev() {
            if entries[j].classification != Classification::SubEntry {
                nearest_non_sub = Some((entries[j].depth, entries[j].x_left));
                break;
            }
        }
        // Is this SubEntry part of a run of ≥2 consecutive SubEntries? A run
        // (Summary → Exercises → References) is a flat chapter-tail group whose
        // members all belong at depth 2, even when the source indents them one
        // level past the section column. A *lone* SubEntry indented further right
        // than its section (e.g. edo's "Problems" nested under the numbered "1.7
        // Summary" section) is a genuine deeper child and must be left alone.
        let len = entries.len();
        let in_run = (i > 0 && entries[i - 1].classification == Classification::SubEntry)
            || (i + 1 < len && entries[i + 1].classification == Classification::SubEntry);
        // Promote a deep SubEntry to depth 2 when it is a chapter child: either it
        // sits at (or left of) the nearest non-SubEntry's indentation — a sibling of
        // the sections — or it is part of a chapter-tail run as described above.
        if let Some((nsd, nsx)) = nearest_non_sub {
            if nsd >= 2 && (entries[i].x_left <= nsx + 3.0 || in_run) {
                entries[i].depth = 2;
            }
        }
    }
}

/// Infer page numbers for chapter-level entries that have no page number.
/// Uses the page of the first child section (which should be at or near the
/// chapter start).
fn infer_chapter_pages(entries: &mut [ClassifiedEntry]) {
    for i in 0..entries.len() {
        if entries[i].page_value != 0 || entries[i].page_label != "" {
            continue; // Already has a page
        }
        if entries[i].depth > 1 {
            continue; // Only chapter-level
        }
        if !matches!(
            entries[i].classification,
            Classification::Chapter | Classification::Unknown
        ) {
            continue;
        }
        // Find the first subsequent entry with a positive page value
        // that is deeper (a child section).
        for j in (i + 1)..entries.len() {
            if entries[j].depth <= entries[i].depth {
                break; // Next chapter — no children found
            }
            if entries[j].page_value > 0 {
                entries[i].page_value = entries[j].page_value;
                entries[i].page_label = entries[j].page_label.clone();
                break;
            }
        }
    }
}

// ── Phase 6b: Infer chapter numbers from children ──

/// If a chapter-depth entry has no section number but its children have
/// numbered sections like "1.1 ...", "1.2 ...", infer the parent is chapter 1
/// and prepend the number to its title.
fn infer_chapter_numbers_from_children(entries: &mut [ClassifiedEntry]) {
    // For each entry, check if it lacks a number prefix but its immediate
    // children (next entries at depth+1) share a common leading number.
    let len = entries.len();
    for i in 0..len {
        let depth = entries[i].depth;
        let title = entries[i].title.trim();

        // Skip if already has a number prefix
        if title.chars().next().map_or(true, |c| c.is_ascii_digit()) {
            continue;
        }

        // Collect the leading number from each immediate child (depth + 1)
        let mut child_prefixes: Vec<u32> = Vec::new();
        for j in (i + 1)..len {
            if entries[j].depth <= depth {
                break; // past this entry's children
            }
            if entries[j].depth == depth + 1 {
                // Try to extract the leading number: "1.1 Foo" → 1, "2.3 Bar" → 2
                let ct = entries[j].title.trim();
                if let Some(dot_pos) = ct.find('.') {
                    let prefix = &ct[..dot_pos];
                    if let Ok(n) = prefix.parse::<u32>() {
                        child_prefixes.push(n);
                    }
                }
            }
        }

        if child_prefixes.is_empty() {
            continue;
        }

        // Check if all children share the same leading number
        let first = child_prefixes[0];
        if child_prefixes.iter().all(|&n| n == first) {
            entries[i].title = format!("{} {}", first, entries[i].title);
            entries[i].chapter_num_inferred = true;
        }
    }
}

/// Resolve bare single-letter section prefixes (like "A" in "A Tour of...").
/// A bare letter is a valid section number only if sibling entries at the same
/// depth also use sequential letter numbering (B, C, D...). Otherwise the
/// letter is just part of the title (e.g., "A" is an English article).
/// Recognise a bare single-letter appendix heading: "A Mathematical background"
/// or "A. GLSLProgram C++ Class". Returns the letter and the normalized title
/// ("A …", dot dropped). Excludes "A.1 …" (an appendix subsection handled by
/// classify_by_pattern Rule 5).
fn bare_letter_heading(title: &str) -> Option<(char, String)> {
    let t = title.trim();
    let mut chars = t.chars();
    let letter = chars.next()?;
    if !letter.is_ascii_uppercase() {
        return None;
    }
    let rest = match chars.next() {
        Some(' ') => t[letter.len_utf8()..].trim_start(),
        Some('.') => {
            // "A. Title" but not "A.1" (dot followed by a digit is a subsection).
            let after = t[letter.len_utf8() + 1..].trim_start();
            if after.starts_with(|c: char| c.is_ascii_digit()) {
                return None;
            }
            after
        }
        _ => return None,
    };
    let first = rest.chars().next()?;
    if !first.is_alphabetic() {
        return None;
    }
    Some((letter, format!("{} {}", letter, rest)))
}

/// Recover bare single-letter appendix chapters ("A Mathematical background",
/// "A. GLSLProgram C++ Class") that classify_by_pattern leaves Unknown.
///
/// Two signals identify a genuine appendix letter (rather than the article "A"):
///   1. a group of ≥2 such headings at the same indentation whose letters form a
///      consecutive sequence (A,B,C / B,C,D), or
///   2. a lone "X …" heading whose immediate children are numbered "X.1", "X.2".
/// Members are forced to depth-1 chapters with the "A." dot dropped. Runs before
/// validate_entries (so a Part-misclassified member is not dropped) and before
/// synthesize_section_numbers (so no fake "20.x" numbers are minted).
fn resolve_bare_letter_sections(entries: &mut [ClassifiedEntry]) {
    // Candidate bare-letter headings. We include Part-classified entries too: an
    // appendix letter can be mis-classified as a Part (convex's "C"), and the
    // consecutive-sequence requirement below keeps genuine roman Part dividers
    // (zen0's single-letter "I"/"V", which are non-consecutive) from promoting.
    let candidates: Vec<(usize, char, f32)> = entries
        .iter()
        .enumerate()
        .filter_map(|(i, e)| bare_letter_heading(&e.title).map(|(c, _)| (i, c, e.x_left)))
        .collect();
    if candidates.is_empty() {
        return;
    }

    let mut promote: Vec<usize> = Vec::new();

    // Case 1: ≥2 bare-letter headings at the same indentation forming a
    // consecutive letter sequence.
    let mut used = vec![false; candidates.len()];
    for a in 0..candidates.len() {
        if used[a] {
            continue;
        }
        let xa = candidates[a].2;
        let group: Vec<usize> = (a..candidates.len())
            .filter(|&b| !used[b] && (candidates[b].2 - xa).abs() <= 3.0)
            .collect();
        if group.len() < 2 {
            continue;
        }
        let mut letters: Vec<char> = group.iter().map(|&g| candidates[g].1).collect();
        letters.sort_unstable();
        letters.dedup();
        let is_sequence =
            letters.len() >= 2 && letters.windows(2).all(|w| w[1] as u8 == w[0] as u8 + 1);
        if is_sequence {
            for &g in &group {
                used[g] = true;
                promote.push(candidates[g].0);
            }
        }
    }

    // Case 2: a lone "X …" heading whose following entries are numbered "X.1",
    // "X.2". The child relationship is by title prefix, not depth — Rule 5 places
    // "A.1" at depth 2, which can equal the bare "A" heading's own (mis-assigned)
    // depth — so scan forward for an "X." entry, stopping at the next appendix
    // letter or the next numbered top-level chapter.
    for &(idx, letter, _) in &candidates {
        if promote.contains(&idx) {
            continue;
        }
        let mut has_letter_children = false;
        for e in &entries[idx + 1..] {
            let t = e.title.trim();
            if bare_letter_heading(t).is_some_and(|(l, _)| l != letter) {
                break; // a different appendix letter
            }
            let cb = t.as_bytes();
            if cb.len() >= 2 && cb[0] == letter as u8 && cb[1] == b'.' {
                has_letter_children = true;
                break;
            }
            if e.depth <= 1 && cb.first().is_some_and(|c| c.is_ascii_digit()) {
                break; // the next numbered chapter
            }
        }
        if has_letter_children {
            promote.push(idx);
        }
    }

    for idx in promote {
        if let Some((_, normalized)) = bare_letter_heading(&entries[idx].title) {
            entries[idx].title = normalized;
        }
        entries[idx].classification = Classification::Chapter;
        entries[idx].depth = 1;
    }
}

// ── Phase 7: Validation ──

/// Drop Part divider entries when the book has numbered chapters.
///
/// Two mechanisms:
/// 1. If chapters form a contiguous 1…N sequence, drop Part entries AND stray
///    Unknown depth-≤1 entries between chapters (e.g. "Theory", "II Applications").
/// 2. Always drop entries whose title starts with "PART " followed by a roman
///    numeral when there are ≥ 4 numbered chapters (e.g. "PART I Particle Physics").
fn drop_part_dividers_if_continuous_chapters(entries: &mut Vec<ClassifiedEntry>) {
    // Collect chapter numbers from Chapter entries at depth ≤ 1
    let chapter_nums: Vec<(usize, u32)> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.classification == Classification::Chapter && e.depth <= 1)
        .filter_map(|(i, e)| {
            let t = e.title.trim();
            let num_str: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
            num_str.parse::<u32>().ok().map(|n| (i, n))
        })
        .collect();

    let has_enough_chapters = chapter_nums.len() >= 4;
    if !has_enough_chapters {
        return;
    }

    // Check contiguous 1, 2, 3, …, N
    let is_contiguous = chapter_nums
        .iter()
        .enumerate()
        .all(|(idx, (_, n))| *n == (idx + 1) as u32);

    // Collect page ranges between consecutive chapters (only used if contiguous)
    let chapter_pages: Vec<i32> = chapter_nums.iter().map(|(i, _)| entries[*i].page_value).collect();

    entries.retain(|e| {
        // Always keep structural entries
        if matches!(
            e.classification,
            Classification::Chapter
                | Classification::Section
                | Classification::Subsection
                | Classification::SubSubsection
        ) {
            return true;
        }
        // Always keep FrontMatter and BackMatter
        if matches!(
            e.classification,
            Classification::FrontMatter | Classification::BackMatter
        ) {
            return true;
        }

        // Drop "PART N" entries (any classification) when there are numbered chapters
        let title_lower = e.title.trim().to_ascii_lowercase();
        if title_lower.starts_with("part ")
            && e.page_value > 0
            && e.title.trim().len() > 5
        {
            // Verify the word after "part " looks like a roman numeral or number
            let after_part = e.title.trim()[5..].trim();
            let first_word = after_part.split_whitespace().next().unwrap_or("");
            if headings::is_roman_numeral(first_word)
                || first_word.chars().all(|c| c.is_ascii_digit())
            {
                tracing::debug!("TOC: dropping PART divider: {:?}", e.title);
                return false;
            }
        }

        if is_contiguous {
            // Drop Part entries with body page values
            if e.classification == Classification::Part && e.page_value > 0 {
                tracing::debug!("TOC: dropping Part divider in continuous-chapter book: {:?}", e.title);
                return false;
            }

            // Drop Unknown entries at depth ≤ 1 between chapter pages
            if e.classification == Classification::Unknown && e.depth <= 1 && e.page_value > 0 {
                for w in chapter_pages.windows(2) {
                    if e.page_value >= w[0] && e.page_value <= w[1] {
                        tracing::debug!(
                            "TOC: dropping Unknown divider between chapters: {:?} (p.{})",
                            e.title, e.page_value
                        );
                        return false;
                    }
                }
            }
        }

        true
    });
}

fn validate_entries(entries: &mut Vec<ClassifiedEntry>) {
    // Remove duplicates (same title + same page)
    let mut seen = std::collections::HashSet::new();
    entries.retain(|e| seen.insert((e.title.clone(), e.page_label.clone(), e.page_value)));

    // Remove "Contents" / "Table of Contents" entries that are running headers
    // (appear multiple times). A single "Contents" entry is kept — it's likely a
    // legitimate TOC entry pointing to itself. Multiple copies indicate running
    // headers that survived the per-page filter.
    {
        let contents_count = entries.iter().filter(|e| {
            let norm = normalize_toc_header_text(&e.title);
            matches!(norm.as_str(), "contents" | "tableofcontents" | "detailedcontents")
        }).count();
        if contents_count > 1 {
            entries.retain(|e| {
                let norm = normalize_toc_header_text(&e.title);
                !matches!(norm.as_str(), "contents" | "tableofcontents" | "detailedcontents")
            });
        }
    }

    for entry in entries.iter_mut() {
        let trimmed_title = entry.title.trim();
        let is_structural = matches!(
            entry.classification,
            Classification::Chapter
                | Classification::Section
                | Classification::Subsection
                | Classification::SubSubsection
        );
        // Decide whether to strip the leading numeric token:
        //  - Structural entries with inherent numbers (from the raw PDF) → never strip
        //  - Inferred numbers (added by infer_chapter_numbers_from_children) → never
        //    strip; the number was added deliberately from the children's section
        //    prefixes and is legitimate even when it equals the page number
        //    (e.g. "1 A Tour of Computer Systems" on page 1)
        //  - Non-structural (Unknown/Part/FrontMatter/BackMatter) → existing wide tolerance
        let should_strip = if is_structural || entry.chapter_num_inferred {
            false
        } else {
            should_strip_leading_page_number(trimmed_title, entry.page_value)
        };
        if should_strip {
            if let Some(stripped) = strip_initial_numeric_token(trimmed_title) {
                entry.title = stripped.to_string();
            }
        }
        // Strip trailing period from leading chapter numbers: "1. Title" → "1 Title"
        // Keep section numbers like "1.1 Title" unchanged.
        {
            let t = entry.title.trim();
            let digit_end = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
            if digit_end > 0 && digit_end < t.len() {
                let after_digits = &t[digit_end..];
                if let Some(rest) = after_digits.strip_prefix(". ") {
                    if !rest.starts_with(|c: char| c.is_ascii_digit()) {
                        entry.title = format!("{} {}", &t[..digit_end], rest);
                    }
                }
            }
        }
        if let Some(normalized_part) = normalize_roman_part_title(entry.title.trim()) {
            entry.title = normalized_part;
            entry.classification = Classification::Part;
            entry.depth = 0;
        }
    }

    // When chapters are continuously numbered (1, 2, 3, …), Part dividers
    // and stray Unknown entries at depth ≤1 between them are structural noise.
    // Drop them so the output is a clean chapter list.
    drop_part_dividers_if_continuous_chapters(entries);

    // Flatten FrontMatter leaf entries from depth > 1 to depth 1.
    // FrontMatter items (Preface, Acknowledgments, About the Author, …) are always
    // top-level in a TOC; depth > 1 is an x-indent artifact.
    for i in 0..entries.len() {
        if entries[i].depth > 1
            && entries[i].classification == Classification::FrontMatter
        {
            let is_leaf = i + 1 >= entries.len() || entries[i + 1].depth <= entries[i].depth;
            if is_leaf {
                entries[i].depth = 1;
            }
        }
    }

    // Drop entries with no page number
    entries.retain(|e| e.page_value != 0 || is_local_page_label(&e.page_label));

    // Sort entries into document page order.
    //
    // Some books have a "brief TOC" (chapters only) followed by a "detailed TOC"
    // (chapters + sections).  After deduplication the chapters end up first with
    // increasing page values, then the sections follow with smaller page values
    // (a huge backwards jump).  The sequencing check below would drop all of the
    // sections.  Sorting by page order fixes this without altering depth info.
    //
    // Sort key:
    //   front matter (pv < 0): map -pv so smaller roman numerals sort first
    //   body        (pv > 0): pv directly (non-decreasing = correct)
    //   zero-page parts:      stable at top of body (very large key within part group)
    entries.sort_by_key(|e| {
        let pv = e.page_value;
        if pv < 0 {
            // page i (-1) sorts before page ii (-2), etc.
            -pv as i64
        } else if pv == 0 {
            // Part headings with no page — keep in their original relative position.
            // We use i64::MAX so they sort last (degenerate case; rare).
            i64::MAX
        } else {
            1_000_000i64 + pv as i64
        }
    });

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
    let mut last_local: Option<(u32, u32)> = None;

    entries.retain(|e| {
        let pv = e.page_value;
        let is_local = is_local_page_label(&e.page_label);
        let title_lower = e.title.trim().to_ascii_lowercase();
        let trimmed_title = e.title.trim();

        // Part headings without a page number are always kept
        if pv == 0 && e.classification == Classification::Part {
            return true;
        }
        if title_lower.contains("(continued)") || title_lower == "continued" {
            tracing::debug!("TOC: dropping continuation stub: {:?}", e.title);
            return false;
        }
        if title_is_enumerated_body_spill(trimmed_title) {
            tracing::debug!("TOC: dropping enumerated body spill: {:?}", e.title);
            return false;
        }
        // The author-line heuristic removes stray author-name text that spilled into
        // the TOC as *unclassified* entries. It must only fire on Unknown entries — a
        // positively-classified BackMatter/FrontMatter/Part/SubEntry title that happens
        // to read like "Firstname Lastname" (e.g. "Revision History", "Bibliographic
        // Notes", "Geometry Manipulation") is a real heading and must be kept.
        if e.classification == Classification::Unknown && title_looks_like_author_line(&e.title) {
            tracing::debug!("TOC: dropping author-like entry: {:?}", e.title);
            return false;
        }
        if title_looks_like_first_local_page_author_spill(e) {
            tracing::debug!("TOC: dropping first-local-page author spill: {:?}", e.title);
            return false;
        }
        if is_local {
            let Some(curr) = parse_local_page_label_parts(&e.page_label) else {
                return false;
            };
            if let Some(prev) = last_local {
                if curr < prev {
                    tracing::debug!(
                        "TOC: dropping out-of-sequence local-label entry: {:?} (p.{}) after p.{}-{}",
                        e.title,
                        e.page_label,
                        prev.0,
                        prev.1
                    );
                    return false;
                }
            }
            last_local = Some(curr);
            return true;
        }

        if pv > 0 {
            let allow_backmatter_reorder =
                matches!(e.classification, Classification::BackMatter) || title_is_backmatter_like(trimmed_title);
            if allow_backmatter_reorder {
                return true;
            }
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

fn strip_initial_numeric_token(title: &str) -> Option<&str> {
    let trimmed = title.trim();
    let tokens = token_spans(trimmed);
    if tokens.len() < 2 {
        return None;
    }
    let first = &trimmed[tokens[0].0..tokens[0].1];
    if parse_page_ref(first).is_none() && parse_ocrish_page_ref_token(first).is_none() {
        return None;
    }
    Some(trimmed[tokens[1].0..].trim())
}

fn should_strip_leading_page_number(title: &str, page_value: i32) -> bool {
    let tokens = token_spans(title);
    if tokens.len() < 2 {
        return false;
    }
    let first = &title[tokens[0].0..tokens[0].1];
    // An uppercase roman numeral (Part prefix like "I", "II", "III") must not be
    // misread as an OCR-garbled page number (normalize_ocrish_page_digits maps
    // I→1, II→11, III→111). IV/V/VI escape only because they contain 'V'; guard
    // all of them uniformly so roman Part prefixes are never stripped.
    if headings::is_roman_numeral(first) {
        return false;
    }
    let Some((_, first_value)) = parse_page_ref(first).or_else(|| parse_ocrish_page_ref_token(first)) else {
        return false;
    };
    let rest = title[tokens[1].0..].trim();
    let second = &title[tokens[1].0..tokens[1].1];
    let second_lower = second.to_ascii_lowercase();

    if matches!(second_lower.as_str(), "appendix" | "index" | "references" | "addenda" | "final") {
        return true;
    }

    if first_value == 0 {
        return true;
    }

    if page_value > 0 && (page_value - first_value).abs() <= 80 {
        return true;
    }

    if second
        .chars()
        .next()
        .map(|c| c.is_uppercase())
        .unwrap_or(false)
        && page_value > 200
        && first_value > 100
    {
        return true;
    }

    rest.starts_with("Global ")
}

fn title_is_enumerated_body_spill(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.len() < 80 {
        return false;
    }
    let count = [" 1.", " 2.", " 3.", " 4.", " (a)", " (b)", " (c)"]
        .iter()
        .filter(|needle| trimmed.contains(**needle))
        .count();
    // Two enumeration markers is unambiguous spill. A single marker in a long
    // title, or an extremely long title, is also body-paragraph spill rather than
    // a real TOC entry (e.g. handbook's "2 The complexity of an algorithm … Note
    // that n = log2(x+"). Real TOC titles — even multi-author ones — stay well
    // under ~140 characters.
    count >= 2 || (count >= 1 && trimmed.len() > 140) || trimmed.len() > 220
}

fn title_looks_like_author_line(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.is_empty() || looks_like_toc_entry_start(trimmed) {
        return false;
    }
    if headings::is_frontmatter_heading(trimmed) {
        return false;
    }

    let tokens: Vec<&str> = trimmed
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '(' | ')')))
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.len() < 2 || tokens.len() > 8 {
        return false;
    }

    let mut capitalized = 0usize;
    let mut connector = 0usize;
    for token in &tokens {
        let lower = token.to_ascii_lowercase();
        if matches!(lower.as_str(), "and" | "&" | "de" | "van" | "von" | "da" | "del") {
            connector += 1;
            continue;
        }
        let first = token.chars().next().unwrap_or_default();
        let mut rest = token.chars().skip(1);
        let initials = token.len() <= 2
            && token
                .chars()
                .all(|c| c.is_ascii_uppercase() || c == '.');
        let name_like = initials
            || (first.is_uppercase()
                && rest.all(|c| !c.is_alphabetic() || c.is_lowercase() || c == '-' || c == '\''));
        if name_like {
            capitalized += 1;
        } else {
            return false;
        }
    }

    capitalized + connector == tokens.len() && capitalized >= 2
}

fn title_looks_like_first_local_page_author_spill(entry: &ClassifiedEntry) -> bool {
    if !is_local_page_label(&entry.page_label) || entry.depth == 0 {
        return false;
    }
    let Some((_, local_page)) = parse_local_page_label_parts(&entry.page_label) else {
        return false;
    };
    if local_page != 1 {
        return false;
    }

    let trimmed = entry.title.trim();
    if trimmed.is_empty()
        || trimmed.chars().any(|c| c.is_ascii_digit())
        || looks_like_toc_entry_start(trimmed)
        || headings::is_frontmatter_heading(trimmed)
    {
        return false;
    }

    let tokens: Vec<&str> = trimmed
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '(' | ')')))
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() || tokens.len() > 5 {
        return false;
    }

    let all_nameish = tokens.iter().all(|token| {
        let lower = token.to_ascii_lowercase();
        if matches!(lower.as_str(), "and" | "&" | "de" | "van" | "von" | "da" | "del" | "der") {
            return true;
        }
        token
            .chars()
            .all(|c| c.is_alphabetic() || c == '-' || c == '\'' || c == '.')
    });
    let has_capital = tokens.iter().any(|token| {
        token
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
    });

    all_nameish && has_capital
}

fn title_is_backmatter_like(title: &str) -> bool {
    let trimmed = title.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.starts_with("Appendix ") || trimmed.starts_with("APPENDIX ") {
        return true;
    }
    let lower = trimmed.to_ascii_lowercase();
    lower == "references"
        || lower == "index"
        || lower.starts_with("index of ")
        || lower == "addenda"
        || lower == "appendices"
}

fn normalize_roman_part_title(title: &str) -> Option<String> {
    let trimmed = title.trim();
    let tokens = token_spans(trimmed);
    if tokens.len() < 3 || tokens.len() > 6 {
        return None;
    }
    if trimmed.starts_with("PART ") || trimmed.starts_with("Part ") {
        return None;
    }
    let first = &trimmed[tokens[0].0..tokens[0].1];
    let second = &trimmed[tokens[1].0..tokens[1].1];
    let rest = trimmed[tokens[2].0..].trim();
    if second != "." || rest.is_empty() {
        return None;
    }
    let roman = first.replace(' ', "");
    if roman.is_empty()
        || !roman
            .chars()
            .all(|c| matches!(c.to_ascii_uppercase(), 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
    {
        return None;
    }
    if !rest.chars().any(|c| c.is_alphabetic()) {
        return None;
    }
    Some(format!("PART {} . {}", roman, rest))
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

// ── Phase 8: Synthesize section numbers ──

/// Returns true if `title` is a structural end-of-chapter item that should NOT
/// receive a section number (e.g. "Summary", "Exercises", "Bibliography").
fn is_structural_toc_title(title: &str) -> bool {
    let lower = title.to_ascii_lowercase();
    let lower = lower.trim();
    matches!(
        lower,
        "summary"
            | "exercises"
            | "problems"
            | "references"
            | "notes"
            | "notes and references"
            | "bibliographic notes"
            | "further reading"
            | "bibliography"
            | "conclusions"
            | "conclusion"
            | "review problems"
            | "homework problems"
            | "solutions to practice problems"
            | "learning objectives"
            | "chapter summary and study guide"
    )
}

/// Check if a title already starts with a section number like "2.1" or "2.1.3".
fn title_has_section_number(title: &str) -> bool {
    let t = title.trim();
    if let Some(first) = t.chars().next() {
        if first.is_ascii_digit() {
            // Check for N.N pattern
            if let Some(dot_pos) = t.find('.') {
                let before_dot = &t[..dot_pos];
                if before_dot.chars().all(|c| c.is_ascii_digit()) {
                    let after_dot = &t[dot_pos + 1..];
                    if after_dot.starts_with(|c: char| c.is_ascii_digit()) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Extract the chapter number from a chapter-level title like "3 The Laws of Motion".
fn extract_chapter_number(title: &str) -> Option<u32> {
    let t = title.trim();
    if let Some(space_pos) = t.find(' ') {
        let prefix = &t[..space_pos];
        prefix.parse::<u32>().ok()
    } else {
        t.parse::<u32>().ok()
    }
}

/// Synthesize hierarchical section numbers for TOC entries that don't have them.
///
/// Works per-chapter: for each numbered chapter, checks if its children need
/// section numbers. Handles mixed cases where some chapters have numbered
/// sections and others don't, and where depth-2 is numbered but depth-3 isn't.
fn synthesize_section_numbers(entries: &mut Vec<ClassifiedEntry>) {
    // Check that we have numbered chapters to anchor on
    let has_numbered_chapters = entries.iter().any(|e| {
        e.depth == 1
            && matches!(
                e.classification,
                Classification::Chapter | Classification::Unknown
            )
            && extract_chapter_number(&e.title).is_some()
    });
    if !has_numbered_chapters {
        return;
    }

    // First pass: determine per-chapter whether depth-2 children need numbering.
    // A chapter needs depth-2 numbering if its non-structural depth-2 children
    // are mostly unnumbered.
    let mut chapter_ranges: Vec<(usize, usize, Option<u32>)> = Vec::new(); // (start, end, ch_num)
    for i in 0..entries.len() {
        if entries[i].depth == 1 && extract_chapter_number(&entries[i].title).is_some() {
            let ch_num = extract_chapter_number(&entries[i].title);
            if let Some(last) = chapter_ranges.last_mut() {
                last.1 = i;
            }
            chapter_ranges.push((i, entries.len(), ch_num));
        }
    }

    let mut needs_d2_numbering: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for &(start, end, ch_num_opt) in &chapter_ranges {
        let Some(ch_num) = ch_num_opt else { continue };
        let d2_entries: Vec<usize> = (start + 1..end)
            .filter(|&j| {
                entries[j].depth == 2
                    && !is_structural_toc_title(&entries[j].title)
                    && !matches!(
                        entries[j].classification,
                        Classification::FrontMatter
                            | Classification::BackMatter
                            | Classification::SubEntry
                    )
            })
            .collect();
        if d2_entries.is_empty() {
            continue;
        }
        let numbered = d2_entries
            .iter()
            .filter(|&&j| title_has_section_number(&entries[j].title))
            .count();
        // If less than half of depth-2 content entries have numbers, synthesize them
        if numbered * 2 < d2_entries.len() {
            needs_d2_numbering.insert(ch_num);
        }
    }

    // Second pass: walk entries and assign section numbers
    let mut current_chapter: Option<u32> = None;
    let mut section_counter: u32 = 0;
    let mut subsection_counter: u32 = 0;

    for i in 0..entries.len() {
        let entry = &entries[i];

        if entry.depth <= 1 {
            if entry.depth == 1 {
                current_chapter = extract_chapter_number(&entry.title);
                section_counter = 0;
                subsection_counter = 0;
            } else {
                current_chapter = None;
                section_counter = 0;
                subsection_counter = 0;
            }
            continue;
        }

        let Some(ch_num) = current_chapter else {
            continue;
        };

        // Skip structural items
        if is_structural_toc_title(&entries[i].title) {
            continue;
        }

        // Skip if already has a section number — but track the counter
        if title_has_section_number(&entries[i].title) {
            if entry.depth == 2 {
                if let Some(dot_pos) = entries[i].title.find('.') {
                    let after_dot = &entries[i].title[dot_pos + 1..];
                    let sec_end = after_dot
                        .find(|c: char| !c.is_ascii_digit())
                        .unwrap_or(after_dot.len());
                    if let Ok(n) = after_dot[..sec_end].parse::<u32>() {
                        section_counter = n;
                        subsection_counter = 0;
                    }
                }
            } else if entry.depth == 3 {
                let t = entries[i].title.trim();
                let parts: Vec<&str> = t.splitn(4, '.').collect();
                if parts.len() >= 3 {
                    let sec_part = parts[2]
                        .find(|c: char| !c.is_ascii_digit())
                        .map(|p| &parts[2][..p])
                        .unwrap_or(parts[2]);
                    if let Ok(n) = sec_part.parse::<u32>() {
                        subsection_counter = n;
                    }
                }
            }
            continue;
        }

        // Back matter / front matter / sub-entries — don't number
        if matches!(
            entries[i].classification,
            Classification::FrontMatter
                | Classification::BackMatter
                | Classification::SubEntry
        ) {
            continue;
        }

        // Assign number based on depth
        if entry.depth == 2 && needs_d2_numbering.contains(&ch_num) {
            section_counter += 1;
            subsection_counter = 0;
            let prefix = format!("{}.{} ", ch_num, section_counter);
            entries[i].title = format!("{}{}", prefix, entries[i].title);
        } else if entry.depth == 3 && section_counter > 0 {
            // Always number depth-3 entries if they don't have numbers
            // (even if depth-2 siblings are already numbered)
            subsection_counter += 1;
            let prefix =
                format!("{}.{}.{} ", ch_num, section_counter, subsection_counter);
            entries[i].title = format!("{}{}", prefix, entries[i].title);
        }
    }
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
            chapter_num_inferred: false,
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
                chapter_num_inferred: false,
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
                origin_x: x,
                pdfium_space_before: false,
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
