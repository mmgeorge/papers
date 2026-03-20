use std::path::Path;

use image::DynamicImage;
use pdfium_render::prelude::*;

use crate::error::ExtractError;
use crate::pdf::{self, PdfChar};
use crate::toc::{TocEntry, TocEntryKind};
use crate::types::{ExtractionResult, Page, Region, RegionKind, ReflowDocument, ReflowNode};
use crate::DebugMode;

/// Write the extraction result as pretty-printed JSON.
pub fn write_json(result: &ExtractionResult, path: &Path) -> Result<(), ExtractError> {
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Write the extraction result as Markdown (via reflow intermediate).
pub fn write_markdown(result: &ExtractionResult, path: &Path) -> Result<(), ExtractError> {
    let doc = reflow(result, &std::collections::HashSet::new());
    let md = render_markdown_from_reflow(&doc);
    std::fs::write(path, md)?;
    Ok(())
}

/// Write the reflow document as pretty-printed JSON.
pub fn write_reflow_json(doc: &ReflowDocument, path: &Path) -> Result<(), ExtractError> {
    let json = serde_json::to_string_pretty(doc)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Save a single region's image if it has an `image_path`.
///
/// When the region has a caption, the crop bbox excludes the caption area
/// so the saved image contains only the visual content (not the caption text).
fn save_region_image(
    region: &Region,
    page_img: &DynamicImage,
    page: &Page,
    images_dir: &Path,
) -> Result<(), ExtractError> {
    if let Some(ref rel_path) = region.image_path {
        let full_path = images_dir
            .parent()
            .unwrap_or(images_dir)
            .join(rel_path);

        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Exclude the caption from the crop bbox. The region bbox may have been
        // expanded by expand_visual_bounds to include the caption; we want just
        // the visual content in the saved image.
        let crop_bbox = if let Some(ref cap) = region.caption {
            exclude_caption_from_bbox(region.bbox, cap.bbox)
        } else {
            region.bbox
        };

        let cropped = crate::figure::crop_region(
            page_img,
            crop_bbox,
            page.width_pt,
            page.height_pt,
            page.dpi,
        );

        cropped.save(&full_path)?;
    }
    Ok(())
}

/// Shrink `bbox` to exclude the area occupied by `caption_bbox`.
///
/// Determines whether the caption is below, above, left, or right of the
/// visual center and trims the corresponding edge.
fn exclude_caption_from_bbox(mut bbox: [f32; 4], cap: [f32; 4]) -> [f32; 4] {
    let vis_cy = (bbox[1] + bbox[3]) / 2.0;
    let cap_cy = (cap[1] + cap[3]) / 2.0;
    let vis_cx = (bbox[0] + bbox[2]) / 2.0;
    let cap_cx = (cap[0] + cap[2]) / 2.0;

    let dy = (cap_cy - vis_cy).abs();
    let dx = (cap_cx - vis_cx).abs();

    if dy >= dx {
        // Caption is primarily above or below
        if cap_cy > vis_cy {
            // Caption below → shrink bottom to top of caption
            bbox[3] = bbox[3].min(cap[1]);
        } else {
            // Caption above → shrink top to bottom of caption
            bbox[1] = bbox[1].max(cap[3]);
        }
    } else {
        // Caption is primarily left or right
        if cap_cx > vis_cx {
            bbox[2] = bbox[2].min(cap[0]);
        } else {
            bbox[0] = bbox[0].max(cap[2]);
        }
    }
    bbox
}

/// Save cropped region images to the output directory.
pub fn write_images(
    pages: &[Page],
    page_images: &[DynamicImage],
    images_dir: &Path,
) -> Result<(), ExtractError> {
    std::fs::create_dir_all(images_dir)?;

    for (page, page_img) in pages.iter().zip(page_images.iter()) {
        for region in &page.regions {
            save_region_image(region, page_img, page, images_dir)?;
            // Also save images for FigureGroup member items
            if let Some(ref items) = region.items {
                for item in items {
                    save_region_image(item, page_img, page, images_dir)?;
                }
            }
        }
    }

    Ok(())
}

/// Save cropped formula region images to the output directory.
pub fn write_formula_images(
    pages: &[Page],
    page_images: &[DynamicImage],
    formulas_dir: &Path,
) -> Result<(), ExtractError> {
    std::fs::create_dir_all(formulas_dir)?;

    for (page, page_img) in pages.iter().zip(page_images.iter()) {
        for region in &page.regions {
            if region.kind != RegionKind::DisplayFormula
                && region.kind != RegionKind::InlineFormula
            {
                continue;
            }

            let filename = format!("{}.png", region.id);
            let full_path = formulas_dir.join(&filename);

            let cropped = crate::figure::crop_region(
                page_img,
                region.bbox,
                page.width_pt,
                page.height_pt,
                page.dpi,
            );

            cropped.save(&full_path)?;
        }
    }

    Ok(())
}

/// Infer the heading depth from a heading's text based on numbering patterns.
///
/// Returns the depth (1-based: 1 = top-level section, 2 = sub-section, etc.).
/// The document title uses depth 0 but that is handled separately.
fn infer_heading_depth(text: &str) -> u32 {
    let text = text.trim();

    // Labeled blocks (Example, Tip, Algorithm) are not structural headings.
    // The caller should check this before calling, but guard here too.
    let label_prefixes = ["Example ", "Tip ", "Algorithm "];
    for prefix in label_prefixes {
        if text.starts_with(prefix) {
            // These get depth 0 as a sentinel — caller treats them as content.
            return 0;
        }
    }

    // Try dotted-decimal patterns: "1.2.3 ...", "1.2 ...", "1 ..."
    // Also handles appendix patterns: "A.1.2", "A.1", "A ..."
    if let Some(first_ch) = text.chars().next() {
        if first_ch.is_ascii_digit() || (first_ch.is_ascii_uppercase() && text.len() > 1) {
            // Find the end of the numbering prefix
            let num_end = text
                .find(|c: char| c == ' ' || c == '\t')
                .unwrap_or(text.len());
            let prefix = &text[..num_end];

            // Count dots to determine depth
            // Single uppercase letter followed by space: appendix chapter (depth 1)
            // But NOT if it looks like a word (more than 1 char before space)
            if first_ch.is_ascii_uppercase()
                && !first_ch.is_ascii_digit()
                && prefix.len() == 1
            {
                return 1;
            }

            // Check if it's actually a number pattern (digits and dots, possibly
            // starting with a letter for appendix)
            let is_number_pattern = prefix.chars().all(|c| {
                c.is_ascii_digit() || c == '.' || c.is_ascii_lowercase() || c.is_ascii_uppercase()
            }) && prefix.chars().any(|c| c.is_ascii_digit() || (c.is_ascii_uppercase() && prefix.len() <= 2));

            if is_number_pattern {
                // Trailing dot doesn't count: "1." is same as "1"
                let trimmed = prefix.trim_end_matches('.');
                let effective_dots = trimmed.chars().filter(|&c| c == '.').count();
                return (effective_dots as u32) + 1;
            }
        }

        // Roman numeral patterns (I, II, III, IV, V, VI, VII, VIII, IX, X, etc.)
        if first_ch.is_ascii_uppercase() {
            let num_end = text
                .find(|c: char| c == ' ' || c == '\t')
                .unwrap_or(text.len());
            let prefix = &text[..num_end];
            if is_roman_numeral(prefix) {
                return 1;
            }
        }
    }

    // Named sections with no number: return 0 as sentinel.
    // The caller uses context (last numbered heading depth) to assign the
    // actual depth, defaulting to 1 when no prior numbered heading exists.
    0
}

/// Check if a string is a valid uppercase Roman numeral.
fn is_roman_numeral(s: &str) -> bool {
    if s.is_empty() || s.len() > 8 {
        return false;
    }
    // Must be all valid Roman numeral characters
    s.chars().all(|c| matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
}

/// If a heading starts with an appendix-style prefix (e.g. "A.1", "B.3.2"),
/// return the leading letter. Returns None for numeric prefixes like "1.2".
fn appendix_letter(text: &str) -> Option<char> {
    let first = text.trim().chars().next()?;
    if first.is_ascii_uppercase() && !first.is_ascii_digit() {
        // Must be followed by a dot and digit: "A.1", "B.3"
        let rest = &text.trim()[1..];
        if rest.starts_with('.') && rest.len() > 1 && rest[1..].starts_with(|c: char| c.is_ascii_digit()) {
            return Some(first);
        }
    }
    None
}

/// Check if a heading text is a labeled block (Example, Figure, Theorem, etc.)
/// that should be treated as content rather than a structural heading.
///
/// Case-insensitive matching with flexible numbering: "Figure 2-4",
/// "EXAMPLE 3.1", "Definition 1", "Proof", etc.
fn is_labeled_block(text: &str) -> bool {
    let text = text.trim();
    let labels = [
        "example", "tip", "algorithm", "figure", "fig.", "fig",
        "table", "listing", "exercise", "definition", "theorem",
        "lemma", "corollary", "proposition", "proof", "remark",
    ];
    let lower = text.to_lowercase();
    for label in &labels {
        if lower.starts_with(label) {
            let rest = &lower[label.len()..];
            if rest.is_empty() {
                // Bare label like "Proof" — only if short (not a sentence)
                return text.len() < 40;
            }
            let next = rest.chars().next().unwrap();
            // Must be followed by space, digit, dot, hyphen, or colon
            if next == ' ' || next == '.' || next == '-' || next == ':'
                || next.is_ascii_digit()
            {
                return true;
            }
        }
    }
    lower.starts_with("acm reference format")
}

/// Detect if a line contains leader dots (TOC-style dot patterns).
///
/// Matches patterns like:
/// - `"Introduction .................. 3"` (dense dots + page number)
/// - `"3.1 Introduction . . . . . . . 3-1"` (spaced dots + hyphenated page)
/// - `"Contents ..................."` (dense dots, no page number)
fn has_leader_dots(line: &str) -> bool {
    let t = line.trim();
    if t.len() < 10 {
        return false;
    }
    // Pattern 1: 3+ consecutive dots with page-number-like ending
    if t.contains("...") {
        let last_word = t.split_whitespace().last().unwrap_or("");
        if is_page_number_like(last_word) {
            return true;
        }
        // Dense dots (5+) even without a page number
        if t.contains(".....") {
            return true;
        }
    }
    // Pattern 2: Spaced dots ". . ." with page-number-like ending
    if t.contains(". . .") {
        let last_word = t.split_whitespace().last().unwrap_or("");
        if is_page_number_like(last_word) {
            return true;
        }
    }
    // Pattern 3: Many dots (10+) making up >20% of the line
    let dot_count = t.chars().filter(|c| *c == '.').count();
    if dot_count >= 10 && (dot_count as f32 / t.len() as f32) > 0.2 {
        return true;
    }
    false
}

/// Check if a string looks like a page number: "3", "142", "3-1", "A-3", "xii".
fn is_page_number_like(s: &str) -> bool {
    let s = s.trim_end_matches(|c: char| matches!(c, '.' | ')' | ']'));
    if s.is_empty() || s.len() > 5 {
        return false;
    }
    // Pure digits: "3", "15", "142"
    if s.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    // Hyphenated: "3-1", "A-3"
    if s.contains('-') {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() == 2
            && parts
                .iter()
                .all(|p| !p.is_empty() && p.len() <= 4 && p.chars().all(|c| c.is_ascii_alphanumeric()))
        {
            return true;
        }
    }
    // Roman numerals: "xii", "iv", "VII"
    let lower = s.to_lowercase();
    if lower
        .chars()
        .all(|c| matches!(c, 'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm'))
    {
        return true;
    }
    false
}

/// Check if a heading text is a back-matter section (references, index, etc.)
/// that should be promoted to top-level when it appears exactly once.
fn is_back_matter_heading(text: &str) -> bool {
    let norm = text.trim().to_ascii_uppercase();
    matches!(
        norm.as_str(),
        "REFERENCES" | "BIBLIOGRAPHY" | "WORKS CITED" | "INDEX" | "GLOSSARY"
    )
}

/// A rendered section with metadata for cross-region processing.
struct Section {
    markdown: String,
    kind: RegionKind,
    /// 0-indexed PDF page this region came from.
    page_idx: u32,
    /// Region bbox positions in points.
    y_top: f32,
    y_bottom: f32,
    x_left: f32,
    x_right: f32,
    is_text: bool,
    is_references: bool,
    /// Display formulas are part of the paragraph flow — text before/after
    /// them should NOT be rearranged across them.
    is_formula: bool,
    /// A short single-line text block that doesn't span the page width.
    is_short_line: bool,
    /// Image path for formula regions (for unparsed formula fallback).
    formula_path: Option<String>,
}

/// Detect watermark text by scanning for font-family changes in PDF chars.
///
/// For each page, finds text at the end of the page (bottom 20%) where the
/// font family differs from the page's dominant body font. If the same
/// "font-tail" text appears on 3+ pages, it's a watermark.
///
/// Returns the set of watermark strings to strip (lowercased).
pub fn detect_watermarks(page_chars: &[(Vec<PdfChar>, f32)]) -> std::collections::HashSet<String> {
    let mut tail_pages: std::collections::HashMap<String, std::collections::HashSet<usize>> =
        std::collections::HashMap::new();

    for (page_idx, (chars, page_height)) in page_chars.iter().enumerate() {
        if chars.is_empty() || *page_height <= 0.0 {
            continue;
        }

        // Find the dominant font family on this page (by char count)
        let mut font_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for c in chars {
            if c.codepoint.is_control() || c.codepoint == ' ' {
                continue;
            }
            let family = normalize_font_family(&c.font_name);
            if !family.is_empty() {
                *font_counts.entry(family).or_default() += 1;
            }
        }
        let dominant = font_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(f, _)| f.clone());
        let Some(dominant) = dominant else { continue };

        // Look at chars in the bottom 20% of the page.
        // CropBox-relative PDF coords: y=0 is bottom, y=page_height is top.
        let bottom_threshold = *page_height * 0.20;
        let bottom_chars: Vec<&PdfChar> = chars
            .iter()
            .filter(|c| {
                !c.codepoint.is_control()
                    && c.codepoint != ' '
                    && c.bbox[3] < bottom_threshold // top of char in bottom 20%
            })
            .collect();

        if bottom_chars.is_empty() {
            continue;
        }

        // Group bottom-page chars by non-dominant font family
        let mut per_family: std::collections::HashMap<String, Vec<&PdfChar>> =
            std::collections::HashMap::new();
        for c in &bottom_chars {
            let family = normalize_font_family(&c.font_name);
            if family != dominant && !family.is_empty() {
                per_family.entry(family).or_default().push(c);
            }
        }

        // For each non-dominant font family, assemble the text and check
        for (_family, mut fchars) in per_family {
            // Sort by Y then X
            fchars.sort_by(|a, b| {
                a.bbox[1]
                    .partial_cmp(&b.bbox[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(
                        a.bbox[0]
                            .partial_cmp(&b.bbox[0])
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
            });

            let mut text = String::new();
            let mut prev_right: Option<f32> = None;
            for c in &fchars {
                if let Some(pr) = prev_right {
                    let gap = c.bbox[0] - pr;
                    if gap > c.space_threshold.max(1.5) {
                        text.push(' ');
                    }
                }
                text.push(c.codepoint);
                prev_right = Some(c.bbox[2]);
            }

            let trimmed = text.trim();
            if trimmed.len() >= 10 && trimmed.split_whitespace().count() >= 3 {
                tail_pages
                    .entry(trimmed.to_lowercase())
                    .or_default()
                    .insert(page_idx);
            }
        }
    }

    // Keep texts that appear on 3+ pages
    tail_pages
        .into_iter()
        .filter(|(_, pages)| pages.len() >= 3)
        .map(|(text, _)| text)
        .collect()
}

/// Normalize a font family name: strip subset prefix and style suffixes.
pub fn normalize_font_family(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    // Strip subset prefix (e.g. "ABCDEF+FontName" → "FontName")
    let name = name.split('+').last().unwrap_or(name);
    // Strip style suffix (e.g. "BerkeleyOldITC-Book" → "BerkeleyOldITC")
    let name = name
        .split(|c: char| c == '-' || c == ',')
        .next()
        .unwrap_or(name);
    name.to_string()
}

/// Build sections from extraction result: render regions to markdown,
/// Returns true if `inner` bbox is fully contained within `outer` bbox (with 1pt tolerance).
fn bbox_contains(outer: [f32; 4], inner: [f32; 4]) -> bool {
    const EPS: f32 = 1.0;
    inner[0] >= outer[0] - EPS
        && inner[1] >= outer[1] - EPS
        && inner[2] <= outer[2] + EPS
        && inner[3] <= outer[3] + EPS
}

/// merge references, dehyphenate across region boundaries, and rejoin
/// paragraphs split across column/page breaks.
fn build_sections(result: &ExtractionResult, skip_pages: &[u32], watermarks: &std::collections::HashSet<String>) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();

    for page in &result.pages {
        let page_idx = page.page.saturating_sub(1);
        if skip_pages.contains(&page_idx) {
            continue;
        }

        // Collect Image/FigureGroup bboxes to suppress text inside figures.
        let figure_bboxes: Vec<[f32; 4]> = page
            .regions
            .iter()
            .filter(|r| matches!(r.kind, RegionKind::Image | RegionKind::FigureGroup))
            .map(|r| r.bbox)
            .collect();

        for region in &page.regions {
            // Skip regions whose content was spliced into a parent region.
            if region.consumed {
                continue;
            }

            // Skip Text regions fully contained within a figure/image bbox
            // (figure-internal labels extracted as separate Text regions).
            // Use a small vertical margin to catch labels just above/below the
            // figure, but require horizontal containment (no margin).
            if region.kind == RegionKind::Text
                && region.bbox != [0.0; 4] // skip zero bboxes (synthetic/test data)
            {
                let is_inside_figure = figure_bboxes.iter().any(|fig| {
                    if *fig == [0.0; 4] {
                        return false;
                    }
                    let expanded = [
                        fig[0],       // no horizontal margin
                        fig[1] - 25.0, // 25pt above
                        fig[2],       // no horizontal margin
                        fig[3] + 5.0, // 5pt below
                    ];
                    bbox_contains(expanded, region.bbox)
                });
                // Only suppress if the text is short (< 100 chars) — long text
                // near a figure is likely real body content, not a label.
                if is_inside_figure {
                    let text_len = region
                        .text
                        .as_ref()
                        .map_or(0, |t| t.trim().len());
                    if text_len < 100 {
                        continue;
                    }
                }
            }
            // Skip text-like regions with no text content (layout model false positives).
            if matches!(
                region.kind,
                RegionKind::Text | RegionKind::ParagraphTitle | RegionKind::Abstract
            ) && region.text.as_ref().map_or(true, |t| t.trim().is_empty())
            {
                continue;
            }
            let section = region_to_markdown(region);
            let has_formula_image = region.image_path.is_some()
                && matches!(
                    region.kind,
                    RegionKind::DisplayFormula | RegionKind::InlineFormula
                );
            if section.is_empty() && !has_formula_image {
                continue;
            }
            let is_references = region.kind == RegionKind::References;
            let is_text = is_references
                || matches!(
                    region.kind,
                    RegionKind::Text
                        | RegionKind::VerticalText
                        | RegionKind::Abstract
                        | RegionKind::SidebarText
                );

            // Merge all references sections into a single section, even
            // when non-text regions (page headers/footers) sit between them.
            if is_references {
                if let Some(prev) = sections.iter_mut().rfind(|s| s.is_references) {
                    let prev_trimmed = prev.markdown.trim_end();
                    let ends_mid = prev_trimmed
                        .ends_with(|c: char| !matches!(c, '.' | '!' | '?' | ':' | '"' | '\u{201D}'));
                    if ends_mid {
                        prev.markdown.push(' ');
                    } else {
                        prev.markdown.push_str("\n\n");
                    }
                    prev.markdown.push_str(section.trim());
                    continue;
                }
            }

            let is_formula = region.kind == RegionKind::DisplayFormula;

            let is_short_line = is_text && {
                let region_width = region.bbox[2] - region.bbox[0];
                let has_bbox = region_width > 0.0;
                let narrow = has_bbox && region_width < page.width_pt * 0.35;
                let short_text = section.trim().len() < 80;
                narrow && short_text
            };

            let formula_path = if has_formula_image {
                region.image_path.clone()
            } else {
                None
            };

            sections.push(Section {
                markdown: section,
                kind: region.kind,
                page_idx,
                y_top: region.bbox[1],
                y_bottom: region.bbox[3],
                x_left: region.bbox[0],
                x_right: region.bbox[2],
                is_text,
                is_references,
                is_formula,
                is_short_line,
                formula_path,
            });
        }
    }

    // --- Detect and reclassify footnotes ---
    // Footnotes appear either at the bottom of the page or in a sidebar
    // (margin notes). The layout model often classifies them as Text.
    // We detect them by:
    // 1. Pages with existing Footnote regions: reclassify nearby Text that
    //    starts with a footnote number.
    // 2. Text regions in the bottom 25% of the page that start with a
    //    footnote number and are short — likely footnotes even without a
    //    Footnote region on the page.
    // 3. Short text regions in the outer margins (sidebar notes) that start
    //    with a number.
    {
        // Get page heights for position-based detection
        let page_heights: std::collections::HashMap<u32, f32> = result
            .pages
            .iter()
            .map(|p| (p.page.saturating_sub(1), p.height_pt))
            .collect();

        // Collect pages that have at least one Footnote region
        let mut footnote_pages: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for sec in &sections {
            if sec.kind == RegionKind::Footnote {
                footnote_pages.insert(sec.page_idx);
            }
        }

        for sec in &mut sections {
            if sec.kind != RegionKind::Text {
                continue;
            }
            let trimmed = sec.markdown.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Check if text starts with a footnote number pattern
            let is_footnote_text = {
                let first_char = trimmed.chars().next().unwrap_or(' ');
                if first_char.is_ascii_digit() {
                    let num_end = trimmed
                        .find(|c: char| !c.is_ascii_digit())
                        .unwrap_or(trimmed.len());
                    let after = trimmed[num_end..].trim_start();
                    num_end > 0
                        && !after.is_empty()
                        && after.starts_with(|c: char| {
                            c.is_alphabetic() || c == '"' || c == '*' || c == '(' || c == '\u{201C}'
                        })
                } else {
                    false
                }
            };

            if !is_footnote_text {
                continue;
            }

            // Reclassify if: (a) page already has Footnote regions, or
            // (b) region is in bottom 25% of page, or (c) region is narrow
            // and in the outer margin (sidebar note)
            let page_h = page_heights.get(&sec.page_idx).copied().unwrap_or(800.0);
            let in_bottom = sec.y_top > page_h * 0.75;
            let is_sidebar = {
                // Sidebar/margin notes are narrow (< 35% of page width)
                // and positioned in the outer margin
                let page_w = result
                    .pages
                    .iter()
                    .find(|p| p.page.saturating_sub(1) == sec.page_idx)
                    .map(|p| p.width_pt)
                    .unwrap_or(600.0);
                let region_w = sec.x_right - sec.x_left;
                region_w > 0.0 && region_w < page_w * 0.35
            };
            let on_footnote_page = footnote_pages.contains(&sec.page_idx);

            if on_footnote_page || in_bottom || is_sidebar {
                sec.kind = RegionKind::Footnote;
                sec.is_text = false;
            }
        }
    }

    // --- Fix E: Filter running headers/footers/watermarks ---
    //
    // Two-pass approach:
    // 1. Exact-match: short text regions whose full content repeats on 3+ pages.
    // 2. Positional: short text regions in the top/bottom 15% of the page that
    //    repeat on 5+ pages — these are headers/footers/watermarks even when
    //    truncated or slightly different. Also strips such fragments when they
    //    got merged into a longer paragraph.
    {
        // Pass 1: Exact repeated text — checks both whole sections AND
        // individual paragraphs within sections (separated by \n\n).
        // This catches watermarks that were separated into their own paragraph
        // by font-break detection but still live inside a larger region.
        let mut text_page_counts: std::collections::HashMap<String, std::collections::HashSet<u32>> =
            std::collections::HashMap::new();
        for sec in &sections {
            if !sec.is_text {
                continue;
            }
            // Check the whole section
            let norm = sec.markdown.trim().to_lowercase();
            if norm.len() >= 3 && norm.len() <= 150 {
                text_page_counts
                    .entry(norm)
                    .or_default()
                    .insert(sec.page_idx);
            }
            // Also check individual paragraphs within the section
            for para in sec.markdown.split("\n\n") {
                let pnorm = para.trim().to_lowercase();
                if pnorm.len() >= 10 && pnorm.len() <= 150 {
                    text_page_counts
                        .entry(pnorm)
                        .or_default()
                        .insert(sec.page_idx);
                }
            }
        }
        let running_texts: std::collections::HashSet<String> = text_page_counts
            .into_iter()
            .filter(|(_, pages)| pages.len() >= 3)
            .map(|(text, _)| text)
            .collect();

        if !running_texts.is_empty() {
            // Remove whole sections that match
            sections.retain(|sec| {
                if !sec.is_text {
                    return true;
                }
                let norm = sec.markdown.trim().to_lowercase();
                !running_texts.contains(&norm)
            });
            // Strip matching paragraphs from within sections
            for sec in &mut sections {
                if !sec.is_text {
                    continue;
                }
                let paras: Vec<&str> = sec.markdown.split("\n\n").collect();
                if paras.len() <= 1 {
                    continue;
                }
                let filtered: Vec<&str> = paras
                    .into_iter()
                    .filter(|p| {
                        let pnorm = p.trim().to_lowercase();
                        !running_texts.contains(&pnorm)
                    })
                    .collect();
                sec.markdown = filtered.join("\n\n");
            }
        }

        // Pass 2: Positional footer/header detection.
        // Collect short text from top/bottom 15% of pages, find repeated content.
        let mut margin_text_pages: std::collections::HashMap<String, std::collections::HashSet<u32>> =
            std::collections::HashMap::new();
        for sec in &sections {
            if !sec.is_text {
                continue;
            }
            let norm = sec.markdown.trim().to_lowercase();
            if norm.len() < 3 || norm.len() > 80 {
                continue;
            }
            // Check if this region is in the top or bottom 15% of the page.
            // We use y_top relative to page height. y_top is in points from
            // page top (after coordinate inversion).
            let page_height = result
                .pages
                .iter()
                .find(|p| p.page.saturating_sub(1) == sec.page_idx)
                .map(|p| p.height_pt)
                .unwrap_or(800.0);
            let in_top = sec.y_top < page_height * 0.12;
            let in_bottom = sec.y_bottom > page_height * 0.85;
            if in_top || in_bottom {
                margin_text_pages
                    .entry(norm)
                    .or_default()
                    .insert(sec.page_idx);
            }
        }
        // Collect all stamps, then cluster: if multiple stamps are prefixes
        // of each other (e.g. "from t", "from th", "from the lib"), they're
        // truncated versions of the same watermark. Use the shortest ≥5-char
        // prefix as the stripping pattern.
        let mut all_stamps: Vec<String> = margin_text_pages
            .into_iter()
            .filter(|(_, pages)| pages.len() >= 3)
            .map(|(text, _)| text)
            .collect();
        all_stamps.sort_by_key(|s| s.len());

        let mut margin_stamps: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for stamp in &all_stamps {
            // Check if this stamp is a prefix of (or equal to) a longer stamp
            let is_prefix_of_longer = all_stamps.iter().any(|other| {
                other.len() > stamp.len() && other.starts_with(stamp.as_str())
            });
            // Keep standalone stamps and the shortest prefix in each cluster.
            // Require ≥10 chars to avoid false positives from common short phrases.
            if !is_prefix_of_longer || stamp.len() >= 10 {
                // Don't add if a shorter prefix of this stamp is already present
                let has_shorter = margin_stamps.iter().any(|existing| {
                    stamp.starts_with(existing.as_str())
                });
                if !has_shorter {
                    margin_stamps.insert(stamp.clone());
                }
            }
        }

        if !margin_stamps.is_empty() {
            // Remove standalone stamp sections.
            sections.retain(|sec| {
                if !sec.is_text {
                    return true;
                }
                let norm = sec.markdown.trim().to_lowercase();
                !margin_stamps.contains(&norm)
            });
            // Strip embedded stamps from longer paragraphs and tables.
            // Only modify text-like sections — don't truncate headings/titles
            // (their "## " prefix would get chopped, leaving bare "##").
            for sec in &mut sections {
                if !sec.is_text && sec.kind != RegionKind::References {
                    continue;
                }
                for stamp in &margin_stamps {
                    let lower = sec.markdown.to_lowercase();
                    if let Some(pos) = lower.find(stamp.as_str()) {
                        // Remove the stamp and any trailing truncated text.
                        // For tables, remove the entire line containing the stamp.
                        if sec.markdown.contains('|') {
                            // Table: remove lines containing the stamp
                            let lines: Vec<&str> = sec.markdown.lines().collect();
                            let cleaned: Vec<&str> = lines
                                .into_iter()
                                .filter(|line| !line.to_lowercase().contains(stamp.as_str()))
                                .collect();
                            sec.markdown = cleaned.join("\n");
                        } else {
                            sec.markdown = sec.markdown[..pos].trim_end().to_string();
                        }
                    }
                }
            }
        }
    }

    // --- Fix O: Filter TOC-like body content (mini-TOCs at chapter starts) ---
    // Covers both Text and FormattedText regions with leader dot patterns.
    for sec in &mut sections {
        if !sec.is_text && sec.kind != RegionKind::FormattedText {
            continue;
        }
        let text = sec.markdown.trim();
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() < 3 {
            continue;
        }
        let toc_lines = lines
            .iter()
            .filter(|line| has_leader_dots(line.trim()))
            .count();
        if toc_lines as f32 / lines.len() as f32 > 0.4 {
            sec.markdown.clear();
        }
    }

    // --- Watermark stripping (font-change detected) ---
    // Remove watermark text detected by font-family analysis of PDF chars.
    // Strips from both standalone sections and embedded within paragraphs.
    if !watermarks.is_empty() {
        // Build match prefixes: use the first 15+ chars of each watermark
        // for matching, since the assembled text may have slightly different
        // endings than what detect_watermarks extracted from raw chars.
        let wm_prefixes: Vec<String> = watermarks
            .iter()
            .map(|w| {
                // Use the first ~30 chars or up to the first digit-run at the end
                let words: Vec<&str> = w.split_whitespace().collect();
                // Keep meaningful words (skip trailing tracking numbers/codes)
                let meaningful: Vec<&str> = words
                    .iter()
                    .take_while(|w| !w.chars().any(|c| c.is_ascii_digit()))
                    .copied()
                    .collect();
                if meaningful.len() >= 3 {
                    meaningful.join(" ")
                } else {
                    w.clone()
                }
            })
            .filter(|p| p.len() >= 10)
            .collect();

        // Single pass: remove standalone watermark sections and strip
        // watermark text from all remaining sections.
        sections.retain(|sec| {
            let norm = sec.markdown.trim().to_lowercase();
            !wm_prefixes.iter().any(|p| norm.starts_with(p.as_str()))
        });
        for sec in &mut sections {
            // Filter table lines (one pass over lines, checking all prefixes)
            if sec.markdown.contains('|') {
                let lines: Vec<&str> = sec.markdown.lines().collect();
                let filtered: Vec<&str> = lines
                    .into_iter()
                    .filter(|line| {
                        let ln = line.to_lowercase();
                        !wm_prefixes.iter().any(|p| ln.contains(p.as_str()))
                    })
                    .collect();
                sec.markdown = filtered.join("\n");
            }
            // Strip watermark text from body: process line-by-line,
            // dropping lines that start with any watermark prefix and
            // truncating lines that contain one mid-line.
            let lower = sec.markdown.to_lowercase();
            if wm_prefixes.iter().any(|p| lower.contains(p.as_str())) {
                let mut result = String::with_capacity(sec.markdown.len());
                for line in sec.markdown.lines() {
                    let ll = line.to_lowercase();
                    if let Some(pos) = wm_prefixes.iter().find_map(|p| ll.find(p.as_str())) {
                        // Keep text before the watermark
                        let before = line[..pos].trim_end();
                        if !before.is_empty() {
                            if !result.is_empty() {
                                result.push('\n');
                            }
                            result.push_str(before);
                        }
                        // Skip everything from the watermark to end of line
                    } else {
                        if !result.is_empty() {
                            result.push('\n');
                        }
                        result.push_str(line);
                    }
                }
                sec.markdown = result;
            }
        }
    }

    // --- Dehyphenation across region boundaries ---
    let mut i = 0;
    while i < sections.len() {
        if sections[i].is_text && sections[i].markdown.ends_with('\u{0002}') {
            sections[i].markdown.pop(); // remove STX sentinel

            let split_pos = sections[i]
                .markdown
                .rfind(|c: char| c == ' ' || c == '\n')
                .map(|p| p + 1)
                .unwrap_or(0);
            let fragment = sections[i].markdown[split_pos..].to_string();
            sections[i].markdown.truncate(split_pos);

            if !fragment.is_empty() {
                let mut found = false;
                for j in (i + 1)..sections.len() {
                    if sections[j].is_text {
                        sections[j].markdown.insert_str(0, &fragment);
                        found = true;
                        break;
                    }
                }
                if !found {
                    sections[i].markdown.push_str(&fragment);
                }
            }
        }
        i += 1;
    }

    // --- Rejoin paragraphs split across column/page breaks ---
    {
        let mut i = 1;
        while i < sections.len() {
            if !sections[i].is_text || sections[i].markdown.trim().is_empty() {
                i += 1;
                continue;
            }
            if sections[i].kind == RegionKind::Abstract {
                i += 1;
                continue;
            }
            if sections[i].is_short_line {
                i += 1;
                continue;
            }
            let mut blocked = false;
            let mut prev_text = None;
            for k in (0..i).rev() {
                if sections[k].is_formula
                    || matches!(
                        sections[k].kind,
                        RegionKind::Title | RegionKind::ParagraphTitle
                    )
                {
                    blocked = true;
                    break;
                }
                if sections[k].is_text
                    && !sections[k].markdown.trim().is_empty()
                    && !sections[k].is_short_line
                {
                    prev_text = Some(k);
                    break;
                }
            }
            if blocked {
                i += 1;
                continue;
            }
            let Some(k) = prev_text else {
                i += 1;
                continue;
            };

            let prev_trimmed = sections[k].markdown.trim();
            let ends_mid = prev_trimmed
                .ends_with(|c: char| !matches!(c, '.' | '!' | '?' | ':' | '"' | '\u{201D}'));
            if !ends_mid {
                i += 1;
                continue;
            }

            // Fix K: Don't merge across large vertical gaps on the same page.
            // A big gap suggests the current region is a footnote at page
            // bottom, not a paragraph continuation. Use absolute threshold
            // (half the page) to avoid blocking column-break merges.
            if sections[k].page_idx == sections[i].page_idx {
                let gap = sections[i].y_top - sections[k].y_bottom;
                // Only block if gap > 200pt (about 1/3 of a typical page)
                // and the gap is > 5× the previous region's height.
                let region_height = (sections[k].y_bottom - sections[k].y_top).max(1.0);
                if gap > 200.0 && gap > region_height * 5.0 {
                    i += 1;
                    continue;
                }
            }

            let text = sections[k].markdown.trim_end().to_string();
            let next_md = sections[i].markdown.trim().to_string();
            if !next_md.is_empty() {
                sections[k].markdown = format!("{text} {next_md}");
                sections[i].markdown = String::new();
            }
            i += 1;
        }
    }

    sections
}

/// Convert a section to a ReflowNode for non-heading types.
/// Parse a footnote marker from the start of footnote text.
///
/// Handles these patterns:
/// - Numeric: "1 The definition..." → ("1", "The definition...")
/// - Superscript numeric: "$^{1}$ Text..." or "$.^{1}$ Text..." → ("1", "Text...")
/// - Symbol markers: "* Text..." → ("*", "Text...")
/// - Dagger/double-dagger: "† Text..." or "‡ Text..." → ("†", "Text...")
/// - No marker: "Just text..." → ("", "Just text...")
fn parse_footnote_marker(text: &str) -> (String, String) {
    let text = text.trim();

    // Pattern 1: Starts with LaTeX superscript like "$^{1}$", "$.^{1}$", "$^ {2} ...$"
    if text.starts_with('$') {
        if let Some(end) = text[1..].find('$').map(|p| p + 2) {
            let latex = &text[..end];
            // Pattern 1a: ^{...} or ^ {...} with braces
            if let Some(start) = latex.find("^{").or_else(|| latex.find("^ {")) {
                let brace_pos = latex[start..].find('{').unwrap() + start;
                if let Some(close) = latex[brace_pos..].find('}') {
                    let marker_raw = latex[brace_pos + 1..brace_pos + close].trim();
                    // Only accept numeric markers (allow spaces between digits from OCR)
                    if !marker_raw.is_empty()
                        && marker_raw
                            .chars()
                            .all(|c| c.is_ascii_digit() || c == ',' || c == ' ')
                        && marker_raw.chars().any(|c| c.is_ascii_digit())
                    {
                        // Remove internal spaces (e.g. "1 4" → "14")
                        let marker: String =
                            marker_raw.chars().filter(|c| !c.is_whitespace()).collect();
                        let body = text[end..].trim();
                        return (marker, body.to_string());
                    }
                }
            }
            // Pattern 1b: bare superscript digit(s) without braces, e.g. "$^ 8 \mathrm{An}$"
            if let Some(caret) = latex.find('^') {
                let after_caret = latex[caret + 1..].trim_start();
                let mut num_end = 0;
                for (i, c) in after_caret.char_indices() {
                    if c.is_ascii_digit() {
                        num_end = i + 1;
                    } else if c == ' ' && num_end > 0 {
                        // Allow one space between digits (OCR artifact)
                        if after_caret[i + 1..].chars().next().map_or(false, |nc| nc.is_ascii_digit()) {
                            continue;
                        }
                        break;
                    } else {
                        break;
                    }
                }
                if num_end > 0 {
                    let marker_raw = &after_caret[..num_end];
                    let marker: String =
                        marker_raw.chars().filter(|c| !c.is_whitespace()).collect();
                    let body = text[end..].trim();
                    return (marker, body.to_string());
                }
            }
        }
    }

    // Pattern 2: Starts with digits followed by space, period, or uppercase letter
    let mut chars = text.char_indices();
    let mut num_end = 0;
    for (i, c) in &mut chars {
        if c.is_ascii_digit() {
            num_end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if num_end > 0 && num_end < text.len() {
        let next_byte = text.as_bytes()[num_end];
        let next_char = text[num_end..].chars().next().unwrap_or(' ');
        // Accept if followed by a space, period, tab, or uppercase letter
        // (uppercase letter = marker stuck directly to word, e.g. "3Different")
        if next_byte == b' ' || next_byte == b'.' || next_byte == b'\t' || next_char.is_uppercase()
        {
            let marker = text[..num_end].to_string();
            let body = text[num_end..].trim_start().trim_start_matches('.').trim();
            return (marker, body.to_string());
        }
    }

    // Pattern 3: Symbol markers (*, †, ‡, §, ¶, #)
    let first = text.chars().next();
    if let Some(c) = first {
        if matches!(c, '*' | '†' | '‡' | '§' | '¶' | '#') {
            let body = text[c.len_utf8()..].trim();
            return (c.to_string(), body.to_string());
        }
    }

    // No recognized marker
    (String::new(), text.to_string())
}

fn section_to_content_node(sec: &Section) -> ReflowNode {
    let text = sec.markdown.trim();
    match sec.kind {
        RegionKind::DisplayFormula | RegionKind::InlineFormula => {
            if text.is_empty() {
                ReflowNode::Formula {
                    content: None,
                    path: sec.formula_path.clone(),
                }
            } else {
                ReflowNode::Formula {
                    content: Some(text.to_string()),
                    path: None,
                }
            }
        }
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => {
            let (path, caption) = parse_figure_markdown(text);
            if path.is_empty() {
                // No image path — emit caption as text if present
                ReflowNode::Text {
                    content: caption.unwrap_or_default(),
                    footnotes: Vec::new(),
                }
            } else {
                ReflowNode::Figure { path, caption }
            }
        }
        RegionKind::Table => {
            let (content, caption) = split_table_caption(text);
            ReflowNode::Table { content, caption }
        }
        RegionKind::Algorithm => ReflowNode::Algorithm {
            content: text.to_string(),
        },
        RegionKind::FigureGroup => {
            let (path, caption) = parse_figure_markdown(text);
            if path.is_empty() {
                ReflowNode::Text {
                    content: caption.unwrap_or_default(),
                    footnotes: Vec::new(),
                }
            } else {
                ReflowNode::FigureGroup { path, caption }
            }
        }
        RegionKind::References => ReflowNode::References {
            content: text.to_string(),
        },
        RegionKind::FormattedText => ReflowNode::FormattedText {
            content: text.to_string(),
        },
        RegionKind::Footnote => {
            let (marker, body) = parse_footnote_marker(text);
            ReflowNode::Footnote {
                marker,
                content: body,
            }
        }
        _ => ReflowNode::Text {
            content: text.to_string(),
            footnotes: Vec::new(),
        },
    }
}

/// Build a reflow document from the extraction result.
///
/// This performs all reflow logic (dehyphenation, paragraph merging, references
/// merging) and organizes the result into a heading-based tree.
pub fn reflow(result: &ExtractionResult, watermarks: &std::collections::HashSet<String>) -> ReflowDocument {
    let sections = build_sections(result, &[], watermarks);

    // --- Convert sections to ReflowNodes and build heading tree ---

    // Count occurrences of each back-matter heading name. When a name
    // appears exactly once (e.g. one "References", one "Index"), it's a
    // document-level section — promote it to depth 1 rather than nesting
    // it under whatever numbered section precedes it.
    let mut back_matter_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for s in &sections {
        if s.kind == RegionKind::ParagraphTitle && !s.markdown.is_empty() {
            let title = s.markdown.trim().strip_prefix("## ").unwrap_or(s.markdown.trim());
            if is_back_matter_heading(title) {
                let key = title.trim().to_ascii_uppercase();
                *back_matter_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let mut flat_nodes: Vec<ReflowNode> = Vec::new();
    // Track last numbered heading depth so unnumbered headings can nest
    // under the current section rather than breaking to top-level.
    let mut last_numbered_depth: u32 = 0;
    // Track which appendix letters we've seen so we can synthesize parent
    // headings (e.g. "Appendix A") when the PDF omits them.
    let mut seen_appendix_letters: std::collections::HashSet<char> =
        std::collections::HashSet::new();

    for sec in &sections {
        let text = sec.markdown.trim();
        if text.is_empty() && sec.formula_path.is_none() {
            continue;
        }

        let node = match sec.kind {
            RegionKind::Title => {
                // Document title — strip the "# " prefix added by region_to_markdown
                let title = text.strip_prefix("# ").unwrap_or(text).to_string();
                last_numbered_depth = 0;
                let (section, text_part) = parse_section_and_text(&title);
                ReflowNode::Heading {
                    depth: 0,
                    text: text_part,
                    section,
                    children: Vec::new(),
                }
            }
            RegionKind::ParagraphTitle => {
                // Section heading — strip the "## " prefix
                let title = text.strip_prefix("## ").unwrap_or(text).to_string();
                let title = collapse_spaced_letters(&title);
                let title = title.replace("**", "");
                if title.trim().is_empty() || title.trim() == "##" {
                    continue;
                }
                if is_labeled_block(&title) {
                    // Labeled blocks (Example, Tip, Algorithm headings) are content
                    ReflowNode::Text {
                        content: format!("## {title}"),
                        footnotes: Vec::new(),
                    }
                } else {
                    let mut depth = infer_heading_depth(&title);
                    if depth == 0 {
                        // Unnumbered heading: nest under current section.
                        let key = title.trim().to_ascii_uppercase();
                        if is_back_matter_heading(&title)
                            && back_matter_counts.get(&key).copied() == Some(1)
                        {
                            // Sole occurrence of this back-matter heading → top-level.
                            depth = 1;
                        } else if last_numbered_depth > 0 {
                            depth = last_numbered_depth + 1;
                        } else {
                            depth = 1;
                        }
                    } else {
                        // Synthesize appendix parent heading when we first
                        // encounter "A.1", "B.2", etc. without a prior "A"
                        // or "B" heading.
                        if depth >= 2 {
                            if let Some(letter) = appendix_letter(&title) {
                                if seen_appendix_letters.insert(letter) {
                                    flat_nodes.push(ReflowNode::Heading {
                                        depth: 1,
                                        text: format!("Appendix {letter}"),
                                        section: None,
                                        children: Vec::new(),
                                    });
                                }
                            }
                        }
                        last_numbered_depth = depth;
                    }
                    let (section, text_part) = parse_section_and_text(&title);
                    ReflowNode::Heading {
                        depth,
                        text: text_part,
                        section,
                        children: Vec::new(),
                    }
                }
            }
            _ => section_to_content_node(sec),
        };

        flat_nodes.push(node);
    }

    // Post-process and build the heading tree.
    postprocess_flat_nodes(&mut flat_nodes);
    build_heading_tree(flat_nodes)
}

// ── Outline-driven reflow ────────────────────────────────────────────

/// Collapse spaces around dots in leading section numbers.
/// "1. 2. 3 Foo Bar" → "1.2.3 Foo Bar"
fn collapse_section_number_spaces(s: &str) -> String {
    let mut result = String::new();
    let mut in_number = true;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if in_number {
            if c.is_ascii_digit() || c == '.' {
                result.push(c);
            } else if c == ' ' {
                // Check if next char continues the section number
                if chars
                    .peek()
                    .map_or(false, |&nc| nc.is_ascii_digit() || nc == '.')
                {
                    // Skip the space — it's inside a section number
                } else {
                    in_number = false;
                    result.push(c);
                }
            } else {
                in_number = false;
                result.push(c);
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Combine section number and text into a display title.
/// "1.4" + "Conclusions" -> "1.4 Conclusions"
/// None + "Conclusions" -> "Conclusions"
fn format_heading_title(section: Option<&str>, text: &str) -> String {
    match section {
        Some(s) if !s.is_empty() => format!("{s} {text}"),
        _ => text.to_string(),
    }
}

/// Normalize a title for fuzzy comparison: lowercase, collapse whitespace,
/// collapse section number spaces, and strip trailing punctuation.
fn normalize_title(s: &str) -> String {
    let s = s.trim().to_lowercase();
    // Collapse whitespace
    let s: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    // Collapse "1. 2. 3" → "1.2.3" in leading section numbers
    let s = collapse_section_number_spaces(&s);
    // Strip trailing punctuation
    s.trim_end_matches(|c: char| matches!(c, '.' | ':' | ',' | ';')).to_string()
}

/// Strip leading numbering prefix from a title for content-only comparison.
/// "1.2.3 Foo Bar" → "foo bar", "Chapter 3 Foo" → "foo", "Foo Bar" → "foo bar"
fn strip_numbering(s: &str) -> String {
    let s = s.trim();
    // Collapse "1. 2. 3" → "1.2.3" before parsing
    let s = collapse_section_number_spaces(s);
    // Skip "Chapter N ", "CHAPTER N ", "Part N " prefixes (case-insensitive)
    let lower = s.to_lowercase();
    let after_label = if let Some(rest) = lower
        .strip_prefix("chapter ")
        .or_else(|| lower.strip_prefix("part "))
    {
        rest.to_string()
    } else {
        s.to_string()
    };

    // Skip leading number+dots: "1.2.3 " or "A.1 " or "IV "
    let trimmed = after_label.trim();
    if let Some(space_pos) = trimmed.find(' ') {
        let prefix = &trimmed[..space_pos];
        let is_numbering = prefix.chars().all(|c| {
            c.is_ascii_digit() || c == '.' || c.is_ascii_uppercase()
        }) && prefix.chars().any(|c| c.is_ascii_digit() || c.is_ascii_uppercase());
        if is_numbering {
            return trimmed[space_pos + 1..].trim().to_lowercase();
        }
    }
    trimmed.to_lowercase()
}

/// Compute the offset mapping printed page numbers to 0-indexed PDF pages.
///
/// Returns `offset` such that `pdf_page = (page_value - 1) + offset` for body pages.
/// Compute body and front-matter page offsets.
///
/// Returns `(body_offset, front_matter_offset)`:
/// - `body_offset`: maps body pages → PDF pages via `pdf = (page_value - 1) + offset`
/// - `front_matter_offset`: maps roman numeral pages → PDF pages via `pdf = (-page_value - 1) + fm_offset`
pub(crate) fn compute_page_offsets(
    toc_entries: &[TocEntry],
    toc_pages: &[u32],
    result: &ExtractionResult,
) -> (i32, i32) {
    // Fix G: Use majority voting across multiple TOC-to-region matches,
    // skipping TOC pages themselves to avoid false matches.
    let body_entries: Vec<&TocEntry> = toc_entries
        .iter()
        .filter(|e| e.page_value > 0)
        .take(20)
        .collect();

    if body_entries.is_empty() {
        let fallback = toc_pages.last().map(|&p| p as i32 + 1).unwrap_or(0);
        return (fallback, 0);
    }

    let mut offset_votes: std::collections::HashMap<i32, usize> =
        std::collections::HashMap::new();

    for entry in &body_entries {
        for page in &result.pages {
            let page_idx = page.page.saturating_sub(1);
            // Skip TOC pages — matching a heading on a TOC page gives a
            // wrong offset since the heading there is a TOC entry, not the
            // actual section.
            if toc_pages.contains(&page_idx) {
                continue;
            }
            for region in &page.regions {
                if region.kind != RegionKind::ParagraphTitle {
                    continue;
                }
                let Some(ref text) = region.text else { continue };

                if titles_match(&entry.title, text) {
                    let offset = page_idx as i32 - (entry.page_value - 1) as i32;
                    *offset_votes.entry(offset).or_default() += 1;
                    break; // one match per page per entry is enough
                }
            }
        }
    }

    // Body offset: most-voted
    let body_offset = if let Some((&best, _)) = offset_votes.iter().max_by_key(|(_, count)| *count) {
        best
    } else {
        // Fallback: body starts right after last TOC page
        toc_pages.last().map(|&p| p as i32 + 1).unwrap_or(0)
    };

    // Front-matter offset: try matching front-matter TOC entries to regions.
    // Roman numeral pages (page_value < 0): pdf_page = (-page_value - 1) + fm_offset
    let fm_entries: Vec<&TocEntry> = toc_entries
        .iter()
        .filter(|e| e.page_value < 0)
        .take(10)
        .collect();

    let mut fm_offset_votes: std::collections::HashMap<i32, usize> =
        std::collections::HashMap::new();
    for entry in &fm_entries {
        for page in &result.pages {
            let page_idx = page.page.saturating_sub(1);
            if toc_pages.contains(&page_idx) {
                continue;
            }
            for region in &page.regions {
                if region.kind != RegionKind::ParagraphTitle {
                    continue;
                }
                let Some(ref text) = region.text else { continue };
                if titles_match(&entry.title, text) {
                    // fm_offset = page_idx - (-page_value - 1)
                    let fm_offset = page_idx as i32 - (-entry.page_value - 1);
                    *fm_offset_votes.entry(fm_offset).or_default() += 1;
                    break;
                }
            }
        }
    }

    let front_matter_offset = if let Some((&best, _)) = fm_offset_votes.iter().max_by_key(|(_, count)| *count) {
        best
    } else {
        // Fallback: front matter starts at PDF page 0
        0
    };

    (body_offset, front_matter_offset)
}

/// Map a printed page value to a 0-indexed PDF page.
/// Map a printed page value to a 0-indexed PDF page.
///
/// `offset` maps body pages: `pdf_page = (page_value - 1) + offset`.
/// `front_matter_offset` maps roman numeral pages: `pdf_page = (-page_value - 1) + fm_offset`.
pub(crate) fn toc_page_to_pdf_page(
    page_value: i32,
    offset: i32,
    front_matter_offset: i32,
    total_pages: u32,
) -> Option<u32> {
    let idx = if page_value > 0 {
        (page_value - 1) as i32 + offset
    } else {
        // Front matter: roman numeral pages (stored as negative values).
        // page_value = -1 → page i, -2 → page ii, etc.
        // Map: pdf_page = (-page_value - 1) + front_matter_offset
        (-page_value - 1) + front_matter_offset
    };
    if idx >= 0 && (idx as u32) < total_pages {
        Some(idx as u32)
    } else {
        None
    }
}

/// Match a TOC entry title to a ParagraphTitle region's text.
/// Returns true if they match by exact, prefix, or fuzzy criteria.
fn titles_match(toc_title: &str, region_text: &str) -> bool {
    let toc_norm = normalize_title(toc_title);
    let region_norm = normalize_title(region_text);

    // Reject obviously-too-long regions — real headings are short.
    // A ParagraphTitle region with 200+ chars is almost certainly body text
    // that the layout model misclassified.
    if region_norm.len() > 200 {
        return false;
    }

    // Exact match
    if toc_norm == region_norm {
        return true;
    }

    // Content-only match (ignoring numbering differences)
    let toc_stripped = strip_numbering(toc_title);
    let region_stripped = strip_numbering(region_text);
    if !toc_stripped.is_empty() && toc_stripped == region_stripped {
        return true;
    }

    // Prefix match: TOC title is prefix of region text.
    // But only if the region text isn't drastically longer (≤ 2× TOC length),
    // to avoid matching body text that starts with the heading text.
    if !toc_norm.is_empty() && region_norm.starts_with(&toc_norm) {
        if region_norm.len() <= toc_norm.len() * 2 + 20 {
            return true;
        }
    }
    // Region text is prefix of TOC title (truncated region text)
    if !region_norm.is_empty() && region_norm.len() >= 8 && toc_norm.starts_with(&region_norm) {
        return true;
    }

    // Fuzzy: >80% of TOC title words appear in region text.
    // Require at least 3 words in the TOC title to avoid spurious matches.
    if !toc_stripped.is_empty() {
        let toc_words: Vec<&str> = toc_stripped.split_whitespace().collect();
        if toc_words.len() >= 3 {
            let matched = toc_words
                .iter()
                .filter(|w| w.len() >= 3 && region_stripped.contains(**w))
                .count();
            if matched as f32 / toc_words.len() as f32 > 0.8 {
                return true;
            }
        }
    }

    false
}

/// Build an outline-driven reflow document using TOC entries as the
/// authoritative heading structure.
///
/// This is a purely TOC-driven, page-range based approach:
/// 1. Each TOC entry defines a heading with a known start page.
/// 2. Content sections are assigned to headings by page position.
/// 3. Everything before the first TOC entry's page is FrontMatter.
/// 4. ParagraphTitle regions matching a TOC heading are suppressed (no dupes).
pub fn reflow_with_outline(
    result: &ExtractionResult,
    toc_entries: &[TocEntry],
    toc_pages: &[u32],
    total_pages: u32,
    watermarks: &std::collections::HashSet<String>,
) -> ReflowDocument {
    let sections = build_sections(result, toc_pages, watermarks);

    // Step 1: Compute page offsets (body + front-matter)
    let (offset, fm_offset) = compute_page_offsets(toc_entries, toc_pages, result);

    // Step 2: Check if Parts exist — if so, shift all depths +1
    let has_parts = toc_entries.iter().any(|e| e.depth == 0);

    // Step 3: Build heading schedule from TOC entries.
    // Each entry becomes (pdf_page, depth, title, is_toc_page).
    // Entries pointing to TOC pages are kept (they become the "Contents"
    // heading) but flagged so we can embed the TOC listing under them.
    let toc_page_set: std::collections::HashSet<u32> = toc_pages.iter().copied().collect();
    let mut headings: Vec<(u32, u32, String, bool)> = Vec::new();
    for entry in toc_entries {
        let pdf_page = toc_page_to_pdf_page(entry.page_value, offset, fm_offset, total_pages);
        let Some(page) = pdf_page else { continue };
        // Detect "Contents" entries: either by title or by pointing to a TOC page.
        let title_lower = entry.title.trim().to_lowercase();
        let is_toc_entry = toc_page_set.contains(&page)
            || title_lower == "contents"
            || title_lower == "table of contents";
        let mut depth = entry.depth;
        if has_parts
            && !matches!(
                entry.kind,
                TocEntryKind::FrontMatter | TocEntryKind::BackMatter
            )
        {
            depth += 1;
        }
        headings.push((page, depth.max(1), entry.title.clone(), is_toc_entry));
    }

    // Step 4: Determine front matter boundary.
    // Everything before the first heading's page is FrontMatter.
    let first_heading_page = headings.first().map(|(p, _, _, _)| *p).unwrap_or(0);

    // Pre-compute normalized TOC heading titles for duplicate suppression.
    let heading_titles_norm: Vec<String> =
        headings.iter().map(|(_, _, t, _)| normalize_title(t)).collect();
    let max_toc_depth = toc_entries.iter().map(|e| e.depth).max().unwrap_or(1);

    // Collect front matter sections (before first heading page)
    let mut fm_children: Vec<ReflowNode> = Vec::new();
    for sec in &sections {
        if sec.page_idx >= first_heading_page {
            break;
        }
        let text = sec.markdown.trim();
        if text.is_empty() {
            continue;
        }
        fm_children.push(section_to_content_node(sec));
    }

    // Step 5: Build flat nodes.
    let mut flat_nodes: Vec<ReflowNode> = Vec::new();

    if !fm_children.is_empty() {
        postprocess_flat_nodes(&mut fm_children);
        flat_nodes.push(ReflowNode::FrontMatter {
            children: fm_children,
        });
    }

    // Pre-build the rendered TOC entries for embedding in the Toc node.
    let toc_rendered: Vec<crate::types::TocEntryRendered> = toc_entries
        .iter()
        .map(|e| {
            let page = if e.page_value > 0 {
                Some(e.page_value.to_string())
            } else if e.page_value < 0 {
                let abs_val = (-e.page_value) as u32;
                Some(to_lower_roman(abs_val))
            } else {
                None
            };
            let (section, text_part) = parse_section_and_text(&e.title);
            crate::types::TocEntryRendered {
                depth: e.depth,
                text: text_part,
                section,
                page,
            }
        })
        .collect();

    // Walk sections, emitting content nodes.  TOC headings are NOT emitted
    // here — they are placed in a second pass that locates the real heading
    // text on the page and replaces it.  This avoids inserting headings at
    // the wrong position (page start) when the heading actually appears
    // partway through the page.
    //
    // We tag each content node with the page it came from so the second
    // pass can restrict its search to the correct page.

    // (page_idx, node) pairs — page_idx is u32::MAX for non-page nodes.
    let mut tagged_nodes: Vec<(u32, ReflowNode)> = Vec::new();

    // Track which pages are TOC pages so we can suppress their content.
    let toc_page_set_local: std::collections::HashSet<u32> =
        toc_pages.iter().copied().collect();

    // Track heading schedule index — we still need it to detect when we
    // cross into TOC-heading territory (to suppress raw TOC content).
    let mut heading_idx: usize = 0;
    let mut in_toc_heading = false;

    for sec in &sections {
        if sec.page_idx < first_heading_page {
            continue;
        }
        let text = sec.markdown.trim();
        if text.is_empty() && sec.formula_path.is_none() {
            continue;
        }

        // Advance heading_idx to track TOC-heading suppression.
        while heading_idx < headings.len() && headings[heading_idx].0 <= sec.page_idx {
            let (_, _, _, is_toc_entry) = &headings[heading_idx];
            if *is_toc_entry {
                in_toc_heading = true;
            } else {
                in_toc_heading = false;
            }
            heading_idx += 1;
        }

        if in_toc_heading {
            continue;
        }

        // Convert section to a node.
        let node = match sec.kind {
            RegionKind::Title => {
                let title = text.strip_prefix("# ").unwrap_or(text).to_string();
                if sec.page_idx < first_heading_page.min(5) {
                    let (section, text_part) = parse_section_and_text(&title);
                    ReflowNode::Heading {
                        depth: 0,
                        text: text_part,
                        section,
                        children: Vec::new(),
                    }
                } else {
                    continue;
                }
            }
            RegionKind::ParagraphTitle => {
                let title = text.strip_prefix("## ").unwrap_or(text).to_string();
                let title = collapse_spaced_letters(&title);
                // Strip bold markers — the region type already conveys
                // "heading"; inline ** from text extraction would cause
                // double-wrapping when we bold the label below.
                let title = title.replace("**", "");
                if title.trim().is_empty() || title.trim() == "##" {
                    continue;
                }

                // Don't suppress ParagraphTitles that match TOC headings —
                // let them flow through as Text so place_toc_headings can
                // find and replace them at the correct position.

                if is_labeled_block(&title) {
                    ReflowNode::Text {
                        content: format!("**{title}**"),
                        footnotes: Vec::new(),
                    }
                } else {
                    let inferred = infer_heading_depth(&title);
                    let adjusted = if has_parts { inferred + 1 } else { inferred };
                    if inferred > 0 && adjusted > max_toc_depth {
                        let (section, text_part) = parse_section_and_text(&title);
                        ReflowNode::Heading {
                            depth: adjusted.max(1),
                            text: text_part,
                            section,
                            children: Vec::new(),
                        }
                    } else {
                        // Emit as text — the TOC replace pass will promote
                        // it to a heading if it matches a TOC entry.
                        ReflowNode::Text {
                            content: title,
                            footnotes: Vec::new(),
                        }
                    }
                }
            }
            _ => section_to_content_node(sec),
        };

        tagged_nodes.push((sec.page_idx, node));
    }

    // Place TOC headings by locating the real heading text on each page.
    place_toc_headings(&mut tagged_nodes, &headings, &heading_titles_norm, &toc_rendered);

    // Strip page tags — from here on we work with plain flat_nodes.
    flat_nodes = tagged_nodes.into_iter().map(|(_, node)| node).collect();

    postprocess_flat_nodes(&mut flat_nodes);
    let mut doc = build_heading_tree(flat_nodes);

    // Clear garbage titles (e.g. just "#", short fragments, or truncated text)
    if let Some(ref title) = doc.title {
        let t = title.trim();
        if t.is_empty() || t == "#" || t.len() < 5 {
            doc.title = None;
        }
    }

    // Populate doc.toc — always store it for structured data consumers.
    // The markdown renderer will only emit it at the top if no embedded
    // Toc node exists in the tree.
    doc.toc = toc_rendered;

    // Auto-number unnumbered headings (up to 2 levels below main level).
    auto_number_document(&mut doc);

    doc
}

// ── Auto-numbering for unnumbered TOC entries ────────────────────────

/// Check if a heading title starts with a section number prefix.
/// Matches: "1 ...", "1. ...", "1.2 ...", "A.1 ...", "A. ...", etc.
fn has_section_number(title: &str) -> bool {
    let t = title.trim();
    let first = match t.chars().next() {
        Some(c) => c,
        None => return false,
    };
    if !first.is_ascii_digit() && !first.is_ascii_uppercase() {
        return false;
    }
    let space = match t.find(' ') {
        Some(p) => p,
        None => return false,
    };
    let prefix = &t[..space];
    let stripped = prefix.trim_end_matches('.');
    if stripped.is_empty() {
        return false;
    }
    // Pure digits or digits+dots: "1", "1.2", "1.2.3"
    if stripped.chars().all(|c| c.is_ascii_digit() || c == '.') && stripped.chars().any(|c| c.is_ascii_digit()) {
        return true;
    }
    // Appendix-style: "A", "A.1", "B.2.3" — single letter + dot + digits
    if first.is_ascii_uppercase() && !first.is_ascii_digit() {
        if stripped.len() == 1 {
            return true; // bare letter like "A"
        }
        if stripped.len() > 2 && stripped.as_bytes()[1] == b'.' {
            return stripped[2..].chars().all(|c| c.is_ascii_digit() || c == '.');
        }
    }
    false
}

/// Extract the section number prefix from a title.
/// "1.2 Foo" → "1.2", "3. Bar" → "3", "A.1 Baz" → "A.1"
fn extract_section_number(title: &str) -> Option<String> {
    if !has_section_number(title) {
        return None;
    }
    let t = title.trim();
    let space = t.find(' ')?;
    let prefix = t[..space].trim_end_matches('.');
    Some(prefix.to_string())
}

/// Split a heading title into (section_number, text).
///
/// "1.4 Conclusions" → (Some("1.4"), "Conclusions")
/// "Conclusions"      → (None, "Conclusions")
/// "A.1 Appendix"     → (Some("A.1"), "Appendix")
/// "Chapter 3 Foo"    → (Some("3"), "Foo")  // strips "Chapter" label
fn parse_section_and_text(title: &str) -> (Option<String>, String) {
    let t = title.trim();
    // Handle "Chapter N ..." prefix
    if let Some(r) = t
        .strip_prefix("Chapter ")
        .or_else(|| t.strip_prefix("CHAPTER "))
        .or_else(|| t.strip_prefix("chapter "))
    {
        if let Some(sp) = r.find(' ') {
            let num = r[..sp].trim_end_matches('.');
            if num.chars().all(|c| c.is_ascii_digit()) && !num.is_empty() {
                return (Some(num.to_string()), r[sp + 1..].trim().to_string());
            }
        }
    }

    if let Some(section) = extract_section_number(t) {
        let text = t[section.len()..].trim_start_matches('.').trim().to_string();
        (Some(section), text)
    } else {
        (None, t.to_string())
    }
}

/// Auto-number unnumbered headings in the document.
///
/// For each depth level (1, 2, 3), if fewer than 30% of entries already
/// have section numbers, all entries at that depth get auto-numbered.
/// Numbers are hierarchical: chapter 1's sections become 1.1, 1.2, etc.
///
/// This also updates embedded `Toc` nodes in the heading tree.
fn auto_number_document(doc: &mut ReflowDocument) {
    if doc.toc.is_empty() {
        return;
    }

    // Determine which depth levels need auto-numbering.
    let mut numbered_at_depth = [0u32; 5];
    let mut total_at_depth = [0u32; 5];
    for entry in &doc.toc {
        let d = entry.depth as usize;
        if d < 5 {
            total_at_depth[d] += 1;
            if entry.section.is_some() {
                numbered_at_depth[d] += 1;
            }
        }
    }

    let mut auto_depths = [false; 5];
    for d in 1..=3 {
        if total_at_depth[d] > 0 {
            let ratio = numbered_at_depth[d] as f32 / total_at_depth[d] as f32;
            auto_depths[d] = ratio < 0.3;
        }
    }

    if !auto_depths.iter().any(|&b| b) {
        return; // nothing to number
    }

    // Number the TOC entries.
    let mut counters = [0u32; 5];
    let mut parent_num = [String::new(), String::new(), String::new(), String::new(), String::new()];

    for entry in &mut doc.toc {
        let d = entry.depth as usize;
        if d == 0 || d > 3 {
            continue;
        }

        // Reset deeper counters.
        for i in (d + 1)..5 {
            counters[i] = 0;
            parent_num[i].clear();
        }

        if let Some(ref num) = entry.section {
            // Already numbered — track it.
            parent_num[d] = num.clone();
            // Sync counter to the last component.
            if let Some(last) = num.rsplit('.').next() {
                if let Ok(n) = last.parse::<u32>() {
                    counters[d] = n;
                }
            }
        } else if auto_depths[d] {
            counters[d] += 1;
            let number = if d == 1 || parent_num[d - 1].is_empty() {
                format!("{}", counters[d])
            } else {
                format!("{}.{}", parent_num[d - 1], counters[d])
            };
            parent_num[d] = number.clone();
            entry.section = Some(number);
        }
    }

    // Number the heading nodes in the tree with the same logic.
    let mut counters = [0u32; 5];
    let mut parent_num = [String::new(), String::new(), String::new(), String::new(), String::new()];
    auto_number_tree(&mut doc.children, &auto_depths, &mut counters, &mut parent_num);
}

/// Recursively auto-number heading nodes in the tree.
fn auto_number_tree(
    children: &mut [ReflowNode],
    auto_depths: &[bool; 5],
    counters: &mut [u32; 5],
    parent_num: &mut [String; 5],
) {
    for node in children.iter_mut() {
        match node {
            ReflowNode::Heading { depth, text, section, children } => {
                let d = *depth as usize;
                if d >= 1 && d <= 3 {
                    // Reset deeper counters.
                    for i in (d + 1)..5 {
                        counters[i] = 0;
                        parent_num[i].clear();
                    }

                    if let Some(num) = section.as_ref() {
                        parent_num[d] = num.clone();
                        if let Some(last) = num.rsplit('.').next() {
                            if let Ok(n) = last.parse::<u32>() {
                                counters[d] = n;
                            }
                        }
                    } else if d < auto_depths.len() && auto_depths[d] {
                        counters[d] += 1;
                        let number = if d == 1 || parent_num[d - 1].is_empty() {
                            format!("{}", counters[d])
                        } else {
                            format!("{}.{}", parent_num[d - 1], counters[d])
                        };
                        parent_num[d] = number.clone();
                        *section = Some(number);
                    }
                }

                auto_number_tree(children, auto_depths, counters, parent_num);
            }
            ReflowNode::Toc { entries } => {
                // Re-number the embedded TOC entries to match.
                let mut c = [0u32; 5];
                let mut p = [String::new(), String::new(), String::new(), String::new(), String::new()];
                for entry in entries.iter_mut() {
                    let d = entry.depth as usize;
                    if d == 0 || d > 3 { continue; }
                    for i in (d + 1)..5 { c[i] = 0; p[i].clear(); }
                    if let Some(ref num) = entry.section {
                        p[d] = num.clone();
                        if let Some(last) = num.rsplit('.').next() {
                            if let Ok(n) = last.parse::<u32>() { c[d] = n; }
                        }
                    } else if d < auto_depths.len() && auto_depths[d] {
                        c[d] += 1;
                        let number = if d == 1 || p[d - 1].is_empty() {
                            format!("{}", c[d])
                        } else {
                            format!("{}.{}", p[d - 1], c[d])
                        };
                        p[d] = number.clone();
                        entry.section = Some(number);
                    }
                }
            }
            _ => {}
        }
    }
}

/// Auto-number unnumbered raw TOC entries (for `--section` CLI lookup).
///
/// Same logic as `auto_number_document` but operates on `TocEntry` directly.
pub fn auto_number_toc_entries(entries: &mut [TocEntry]) {
    if entries.is_empty() {
        return;
    }

    let mut numbered_at_depth = [0u32; 5];
    let mut total_at_depth = [0u32; 5];
    for entry in entries.iter() {
        let d = entry.depth as usize;
        if d < 5 {
            total_at_depth[d] += 1;
            if has_section_number(&entry.title) {
                numbered_at_depth[d] += 1;
            }
        }
    }

    let mut auto_depths = [false; 5];
    for d in 1..=3 {
        if total_at_depth[d] > 0 {
            let ratio = numbered_at_depth[d] as f32 / total_at_depth[d] as f32;
            auto_depths[d] = ratio < 0.3;
        }
    }

    if !auto_depths.iter().any(|&b| b) {
        return;
    }

    let mut counters = [0u32; 5];
    let mut parent_num = [String::new(), String::new(), String::new(), String::new(), String::new()];

    for entry in entries.iter_mut() {
        let d = entry.depth as usize;
        if d == 0 || d > 3 {
            continue;
        }
        for i in (d + 1)..5 {
            counters[i] = 0;
            parent_num[i].clear();
        }

        if let Some(num) = extract_section_number(&entry.title) {
            parent_num[d] = num.clone();
            if let Some(last) = num.rsplit('.').next() {
                if let Ok(n) = last.parse::<u32>() {
                    counters[d] = n;
                }
            }
        } else if auto_depths[d] {
            counters[d] += 1;
            let number = if d == 1 || parent_num[d - 1].is_empty() {
                format!("{}", counters[d])
            } else {
                format!("{}.{}", parent_num[d - 1], counters[d])
            };
            parent_num[d] = number.clone();
            entry.title = format!("{number} {}", entry.title);
        }
    }
}

/// Convert an integer to lowercase roman numerals.
fn to_lower_roman(mut n: u32) -> String {
    let table = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    ];
    let mut result = String::new();
    for &(value, numeral) in &table {
        while n >= value {
            result.push_str(numeral);
            n -= value;
        }
    }
    result
}

/// Parse figure markdown like `"![](path)\n\ncaption"` into (path, caption).
fn parse_figure_markdown(md: &str) -> (String, Option<String>) {
    // Split on first double newline
    let parts: Vec<&str> = md.splitn(2, "\n\n").collect();
    let img_line = parts[0];
    let caption = parts.get(1).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());

    // Extract path from ![](path)
    let path = if let Some(start) = img_line.find("](") {
        let after = &img_line[start + 2..];
        after.strip_suffix(')').unwrap_or(after).to_string()
    } else {
        img_line.to_string()
    };

    (path, caption)
}

/// Split table markdown from its trailing caption (separated by \n\n).
fn split_table_caption(md: &str) -> (String, Option<String>) {
    // Find the last \n\n — caption comes after it if it looks like a caption
    if let Some(pos) = md.rfind("\n\n") {
        let table_part = md[..pos].trim().to_string();
        let after = md[pos + 2..].trim();
        // Check if the trailing part is a caption (starts with bold label)
        if after.starts_with("**Fig.") || after.starts_with("**Figure")
            || after.starts_with("**Table") || after.starts_with("**Algorithm")
        {
            return (table_part, Some(after.to_string()));
        }
    }
    (md.to_string(), None)
}

// ── TOC heading placement ────────────────────────────────────────────

/// Place TOC headings by locating the real heading text on each page.
///
/// For each TOC heading, searches for a text node on its target page whose
/// content matches the heading title.  When found, the text node is **replaced**
/// with the proper `Heading` node (carrying the TOC's numbering and depth).
///
/// If no matching text is found on the page, the heading is **injected** before
/// the first content node from that page as a fallback.
fn place_toc_headings(
    tagged_nodes: &mut Vec<(u32, ReflowNode)>,
    headings: &[(u32, u32, String, bool)],
    heading_titles_norm: &[String],
    toc_rendered: &[crate::types::TocEntryRendered],
) {
    let heading_stripped: Vec<String> = headings
        .iter()
        .map(|(_, _, title, _)| strip_numbering(title))
        .collect();

    let mut placed = vec![false; headings.len()];

    // Pass 1: find and replace matching text nodes.
    for (hi, (page, depth, title, is_toc_entry)) in headings.iter().enumerate() {
        let title_norm = &heading_titles_norm[hi];
        let title_stripped = &heading_stripped[hi];

        let mut match_idx: Option<usize> = None;
        for (ni, (pg, node)) in tagged_nodes.iter().enumerate() {
            if *pg < *page {
                continue;
            }
            if *pg > *page {
                break;
            }
            let content = match node {
                ReflowNode::Text { content, .. } => content.trim(),
                _ => continue,
            };
            if content.is_empty() || content.len() > 120 {
                continue;
            }
            let content_norm = normalize_title(content);
            let content_stripped = strip_numbering(content);

            let exact = content_norm == *title_norm;
            let stripped = !title_stripped.is_empty()
                && title_stripped.len() >= 3
                && *title_stripped == content_stripped;
            let section_match = {
                let tn = title_norm.split_whitespace().next().unwrap_or("");
                let cn = content_norm.split_whitespace().next().unwrap_or("");
                !tn.is_empty()
                    && tn == cn
                    && (tn.contains('.') || tn.starts_with("chapter"))
            };

            if exact || stripped || section_match {
                match_idx = Some(ni);
                break;
            }
        }

        if let Some(ni) = match_idx {
            let (sec, txt) = parse_section_and_text(title);
            tagged_nodes[ni] = (
                *page,
                ReflowNode::Heading {
                    depth: *depth,
                    text: txt,
                    section: sec,
                    children: Vec::new(),
                },
            );
            if *is_toc_entry {
                tagged_nodes.insert(
                    ni + 1,
                    (*page, ReflowNode::Toc { entries: toc_rendered.to_vec() }),
                );
            }
            placed[hi] = true;
        }
    }

    // Pass 2: inject unmatched headings at page boundary (fallback).
    // Process in reverse so insertions don't shift indices of earlier headings.
    for hi in (0..headings.len()).rev() {
        if placed[hi] {
            continue;
        }
        let (page, depth, ref title, is_toc_entry) = headings[hi];
        let insert_pos = tagged_nodes
            .iter()
            .position(|(pg, _)| *pg >= page)
            .unwrap_or(tagged_nodes.len());

        let (sec, txt) = parse_section_and_text(title);
        tagged_nodes.insert(
            insert_pos,
            (
                page,
                ReflowNode::Heading {
                    depth,
                    text: txt,
                    section: sec,
                    children: Vec::new(),
                },
            ),
        );
        if is_toc_entry {
            tagged_nodes.insert(
                insert_pos + 1,
                (page, ReflowNode::Toc { entries: toc_rendered.to_vec() }),
            );
        }
    }
}

// ── Post-processing passes on flat nodes ────────────────────────────

/// Apply all post-processing passes to the flat node list before building
/// the heading tree.
fn postprocess_flat_nodes(flat_nodes: &mut Vec<ReflowNode>) {
    dedup_heading_echo(flat_nodes);
    dedup_heading_echo_multiline(flat_nodes);
    reorder_parent_before_child(flat_nodes);
    reorder_headings_by_numbering(flat_nodes);
    dedup_consecutive_text(flat_nodes);
    dedup_footnotes(flat_nodes);
    detect_lists(flat_nodes);
    detect_code_blocks(flat_nodes);
    dedup_code_text(flat_nodes);
    // Collapse doubled characters (e.g. "EEXXPPEERRIIMMEENNTT" → "EXPERIMENT")
    for node in flat_nodes.iter_mut() {
        match node {
            ReflowNode::Text { content, .. } => {
                let collapsed = collapse_doubled_chars(content);
                if collapsed.len() < content.len() {
                    *content = collapsed;
                }
            }
            ReflowNode::Heading { text, .. } => {
                let collapsed = collapse_doubled_chars(text);
                if collapsed.len() < text.len() {
                    *text = collapsed;
                }
            }
            _ => {}
        }
    }
}

/// Remove duplicate footnotes with the same marker.
fn dedup_footnotes(flat_nodes: &mut Vec<ReflowNode>) {
    let mut seen_markers: std::collections::HashSet<String> = std::collections::HashSet::new();
    flat_nodes.retain(|node| {
        if let ReflowNode::Footnote { marker, .. } = node {
            if marker.is_empty() || seen_markers.insert(marker.clone()) {
                true // first occurrence — keep
            } else {
                false // duplicate — remove
            }
        } else {
            true
        }
    });
}

/// Fix A: Remove heading title echoed as the first child text node.
///
/// The layout model sometimes extracts heading text as both a ParagraphTitle
/// region AND as the opening line of the following Text region. This removes
/// the duplicate prefix from the text node (or removes it entirely).
fn dedup_heading_echo(flat_nodes: &mut Vec<ReflowNode>) {
    let mut i = 0;
    while i + 1 < flat_nodes.len() {
        let (title_norm, heading_depth) = if let ReflowNode::Heading { text, section, depth, .. } = &flat_nodes[i] {
            let combined = format_heading_title(section.as_deref(), text);
            let n = normalize_title(&combined);
            if n.len() >= 5 { (Some(n), *depth) } else { (None, 0) }
        } else {
            (None, 0)
        };

        if let Some(title_norm) = title_norm {
            // Search within a window of up to 5 nodes for the echo.
            // Stop at the next heading of same or shallower depth.
            let search_end = (i + 6).min(flat_nodes.len());
            let mut echo_idx = None;

            for j in (i + 1)..search_end {
                match &flat_nodes[j] {
                    ReflowNode::Heading { depth, .. } if *depth <= heading_depth => break,
                    ReflowNode::Text { content, .. } => {
                        let content_norm = normalize_title(content);
                        let section_match = {
                            let title_num = title_norm.split_whitespace().next().unwrap_or("");
                            let content_num = content_norm.split_whitespace().next().unwrap_or("");
                            !title_num.is_empty()
                                && title_num == content_num
                                && (title_num.contains('.') || title_num.starts_with("chapter"))
                        };
                        if content_norm == title_norm
                            || content_norm.starts_with(&title_norm)
                            || section_match
                        {
                            echo_idx = Some(j);
                            break;
                        }
                    }
                    _ => {}
                }
            }

            if let Some(j) = echo_idx {
                let content = if let ReflowNode::Text { content, .. } = &flat_nodes[j] {
                    content.clone()
                } else {
                    i += 1;
                    continue;
                };
                let remainder = strip_title_echo(&content, &title_norm);
                if remainder.trim().is_empty() {
                    flat_nodes.remove(j);
                    continue;
                } else {
                    if let ReflowNode::Text { content, .. } = &mut flat_nodes[j] {
                        *content = remainder;
                    }
                }
            }
        }
        i += 1;
    }
}

/// Fix A2: Remove multi-line heading echo.
///
/// Handles the pattern where a heading like "## I Theory" is followed by
/// 2+ short text nodes that reconstruct the heading: "Part I" / "Theory".
fn dedup_heading_echo_multiline(flat_nodes: &mut Vec<ReflowNode>) {
    let mut i = 0;
    while i < flat_nodes.len() {
        let title_norm = if let ReflowNode::Heading { text, section, .. } = &flat_nodes[i] {
            normalize_title(&format_heading_title(section.as_deref(), text))
        } else {
            i += 1;
            continue;
        };

        if title_norm.len() < 3 {
            i += 1;
            continue;
        }

        let title_words: Vec<&str> = title_norm.split_whitespace().collect();

        // Collect consecutive short text nodes after the heading.
        // Check incrementally: after each node, test if we have a match.
        let mut j = i + 1;
        let mut collected_words: Vec<String> = Vec::new();
        let mut nodes_to_remove: Vec<usize> = Vec::new();
        let mut removed = false;
        while j < flat_nodes.len() && nodes_to_remove.len() < 5 {
            if let ReflowNode::Text { content, .. } = &flat_nodes[j] {
                let trimmed = content.trim();
                let word_count = trimmed.split_whitespace().count();
                if trimmed.len() <= 40 && word_count <= 4 && !trimmed.is_empty() {
                    for w in trimmed.split_whitespace() {
                        collected_words.push(w.to_lowercase());
                    }
                    nodes_to_remove.push(j);
                    j += 1;

                    // Check match after each addition
                    if collected_words.len() >= 2 && nodes_to_remove.len() >= 2 {
                        let match_count = collected_words
                            .iter()
                            .filter(|w| title_words.iter().any(|tw| *tw == w.as_str()))
                            .count();
                        if match_count >= collected_words.len().saturating_sub(1)
                            && match_count >= title_words.len().saturating_sub(1)
                        {
                            for &idx in nodes_to_remove.iter().rev() {
                                flat_nodes.remove(idx);
                            }
                            removed = true;
                            break;
                        }
                    }
                    continue;
                }
            }
            break;
        }
        if removed {
            continue;
        }

        i += 1;
    }
}

/// Strip a heading-echo prefix from content text.
/// Compares word-by-word against the normalized title.
fn strip_title_echo(content: &str, title_norm: &str) -> String {
    let content_trimmed = content.trim();
    let content_lower = content_trimmed.to_lowercase();
    let title_words: Vec<&str> = title_norm.split_whitespace().collect();

    // Find how many words from the start of content match the title
    let content_words: Vec<&str> = content_trimmed.split_whitespace().collect();
    let content_lower_words: Vec<&str> = content_lower.split_whitespace().collect();

    let mut matched = 0;
    for (i, tw) in title_words.iter().enumerate() {
        if i < content_lower_words.len() && content_lower_words[i] == *tw {
            matched += 1;
        } else {
            break;
        }
    }

    if matched >= title_words.len().min(content_words.len())
        && matched >= 2
        && matched <= content_words.len()
    {
        // Strip the matched prefix
        content_words[matched..].join(" ")
    } else {
        content_trimmed.to_string()
    }
}

/// Fix B+: Reorder headings so that numbered sections appear in numeric order.
///
/// When a heading like "1.4.2" appears after "1.7" due to layout detection
/// ordering, move it to the correct position (after the last "1.4.x" heading).
/// Also ensures parent headings appear before their first child.
fn reorder_headings_by_numbering(flat_nodes: &mut Vec<ReflowNode>) {
    // Extract (index, numbering_key) for all heading nodes
    let mut heading_indices: Vec<(usize, Vec<u32>)> = Vec::new();
    for (i, node) in flat_nodes.iter().enumerate() {
        if let ReflowNode::Heading { text, section, .. } = node {
            let combined = format_heading_title(section.as_deref(), text);
            if let Some(key) = parse_section_number(&combined) {
                heading_indices.push((i, key));
            }
        }
    }

    if heading_indices.len() < 2 {
        return;
    }

    // Find out-of-order headings and relocate them.
    // A heading with key [1,4,2] should come after [1,4,1] but before [1,5].
    // If it currently sits after [1,5], move it to right after [1,4,1].
    for _pass in 0..10 {
        let mut moved = false;

        // Re-scan heading positions (they shift after each move)
        heading_indices.clear();
        for (i, node) in flat_nodes.iter().enumerate() {
            if let ReflowNode::Heading { text, section, .. } = node {
                let combined = format_heading_title(section.as_deref(), text);
                if let Some(key) = parse_section_number(&combined) {
                    heading_indices.push((i, key));
                }
            }
        }

        for hi in 1..heading_indices.len() {
            let (cur_idx, ref cur_key) = heading_indices[hi];
            let (prev_idx, ref prev_key) = heading_indices[hi - 1];

            // If the previous heading (by flat position) has a LARGER key
            // than us, we're out of order. Find where we should actually be.
            if prev_key > cur_key {
                // Find the last heading with key < cur_key
                let mut insert_after_hi = None;
                for hj in (0..hi).rev() {
                    if heading_indices[hj].1 <= *cur_key {
                        insert_after_hi = Some(hj);
                        break;
                    }
                }

                if let Some(ihi) = insert_after_hi {
                    let target_idx = heading_indices[ihi].0;
                    if target_idx < cur_idx {
                        // Move the heading (and its trailing content) right
                        // after target_idx. Find how many content nodes follow
                        // the heading before the next heading.
                        let next_heading = heading_indices.get(hi + 1).map(|(idx, _)| *idx).unwrap_or(flat_nodes.len());
                        let block_end = next_heading.min(cur_idx + 1); // just the heading for now

                        let node = flat_nodes.remove(cur_idx);
                        flat_nodes.insert(target_idx + 1, node);
                        moved = true;
                        break;
                    }
                }
            }
        }

        if !moved {
            break;
        }
    }
}

/// Parse a section number prefix like "1.4.2 Foo" into a sortable key [1, 4, 2].
fn parse_section_number(title: &str) -> Option<Vec<u32>> {
    let title = title.trim();
    let num_end = title.find(|c: char| c == ' ' || c == '\t').unwrap_or(title.len());
    let prefix = &title[..num_end];

    // Must start with a digit and contain only digits and dots
    if prefix.is_empty() || !prefix.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    if !prefix.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }

    let parts: Vec<u32> = prefix
        .split('.')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();

    if parts.is_empty() {
        None
    } else {
        Some(parts)
    }
}

/// Fix B: Ensure parent headings appear before their children.
///
/// When a child heading (higher depth) appears before its logical parent
/// (lower depth) in the flat list, move the parent before the child.
fn reorder_parent_before_child(flat_nodes: &mut Vec<ReflowNode>) {
    let mut i = 0;
    while i < flat_nodes.len() {
        let child_depth = if let ReflowNode::Heading { depth, .. } = &flat_nodes[i] {
            *depth
        } else {
            i += 1;
            continue;
        };

        if child_depth <= 1 {
            i += 1;
            continue;
        }

        // Look ahead for a parent heading (lower depth)
        let look_limit = (i + 6).min(flat_nodes.len());
        let mut parent_idx = None;
        for k in (i + 1)..look_limit {
            if let ReflowNode::Heading { depth, .. } = &flat_nodes[k] {
                if *depth < child_depth {
                    parent_idx = Some(k);
                }
                break; // stop at first heading encountered
            }
        }

        if let Some(pidx) = parent_idx {
            let parent = flat_nodes.remove(pidx);
            flat_nodes.insert(i, parent);
            // Don't advance i — re-check from the same position
        }
        i += 1;
    }
}

/// Fix D: Remove consecutive duplicate text blocks.
///
/// Detects adjacent Text nodes (possibly separated by a single Formula)
/// with identical or highly overlapping content and removes the duplicate.
fn dedup_consecutive_text(flat_nodes: &mut Vec<ReflowNode>) {
    let mut i = 0;
    while i + 1 < flat_nodes.len() {
        // Get text content from node i, allowing skip of one formula
        let (a_content, next_idx) = match &flat_nodes[i] {
            ReflowNode::Text { content, .. } => (content.clone(), i + 1),
            _ => {
                i += 1;
                continue;
            }
        };

        // Check the next node (or node after a formula)
        let b_idx = if next_idx < flat_nodes.len() {
            if matches!(&flat_nodes[next_idx], ReflowNode::Text { .. }) {
                Some(next_idx)
            } else if next_idx + 1 < flat_nodes.len()
                && matches!(&flat_nodes[next_idx], ReflowNode::Formula { .. })
                && matches!(&flat_nodes[next_idx + 1], ReflowNode::Text { .. })
            {
                Some(next_idx + 1)
            } else {
                None
            }
        } else {
            None
        };

        let Some(b_idx) = b_idx else {
            i += 1;
            continue;
        };

        let b_content = if let ReflowNode::Text { content, .. } = &flat_nodes[b_idx] {
            content.clone()
        } else {
            i += 1;
            continue;
        };

        let a_norm: String = a_content.split_whitespace().collect::<Vec<_>>().join(" ");
        let b_norm: String = b_content.split_whitespace().collect::<Vec<_>>().join(" ");

        if a_norm.len() < 15 && b_norm.len() < 15 {
            i += 1;
            continue;
        }

        let should_dedup = if a_norm == b_norm {
            true
        } else if a_norm.len() >= 20 && b_norm.starts_with(&a_norm) {
            true
        } else if b_norm.len() >= 20 && a_norm.starts_with(&b_norm) {
            true
        } else {
            text_overlap_ratio(&a_norm, &b_norm) > 0.8
                && a_norm.len().min(b_norm.len()) >= 20
        };

        if should_dedup {
            // Keep the longer one
            if a_norm.len() >= b_norm.len() {
                flat_nodes.remove(b_idx);
            } else {
                flat_nodes.remove(i);
            }
            continue; // re-check from same position
        }

        i += 1;
    }
}

/// Compute word-overlap ratio between two normalized text strings.
fn text_overlap_ratio(a: &str, b: &str) -> f32 {
    let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();
    let intersection = a_words.intersection(&b_words).count();
    let min_len = a_words.len().min(b_words.len());
    if min_len == 0 {
        return 0.0;
    }
    intersection as f32 / min_len as f32
}

/// Fix M: Collapse text where every character is doubled.
///
/// Some PDFs render bold/shadow text by printing each character twice at the
/// same position. The extractor picks up both, producing "EEXXPPEERRIIMMEENNTT".
/// Applied per-paragraph so mixed normal/doubled content is handled.
fn collapse_doubled_chars(text: &str) -> String {
    // Process each paragraph independently
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut any_changed = false;
    let mut result_parts: Vec<String> = Vec::with_capacity(paragraphs.len());

    for para in &paragraphs {
        let collapsed = collapse_doubled_paragraph(para);
        if collapsed.len() < para.len() {
            any_changed = true;
        }
        result_parts.push(collapsed);
    }

    if any_changed {
        result_parts.join("\n\n")
    } else {
        text.to_string()
    }
}

fn collapse_doubled_paragraph(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 10 {
        return text.to_string();
    }

    // Check non-space characters for the doubling pattern.
    // Pattern: each letter appears twice, but spaces may not be doubled.
    // E.g., "EE XX PP" → each non-space char is followed by itself.
    let non_space: Vec<(usize, char)> = chars
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.is_whitespace())
        .map(|(i, &c)| (i, c))
        .collect();

    if non_space.len() < 8 {
        return text.to_string();
    }

    // Check if non-space chars come in consecutive pairs
    let sample = non_space.len().min(40);
    let mut doubled = 0;
    let mut checked = 0;
    let mut i = 0;
    while i + 1 < sample {
        if non_space[i].1 == non_space[i + 1].1
            && non_space[i + 1].0 == non_space[i].0 + 1
        {
            doubled += 1;
            i += 2;
        } else {
            i += 1;
        }
        checked += 1;
    }

    if checked > 0 && (doubled as f32 / checked as f32) > 0.7 {
        // Collapse: skip every second non-space char
        let mut skip_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut j = 0;
        while j + 1 < non_space.len() {
            if non_space[j].1 == non_space[j + 1].1
                && non_space[j + 1].0 == non_space[j].0 + 1
            {
                skip_indices.insert(non_space[j + 1].0);
                j += 2;
            } else {
                j += 1;
            }
        }
        chars
            .iter()
            .enumerate()
            .filter(|(idx, _)| !skip_indices.contains(idx))
            .map(|(_, &c)| c)
            .collect()
    } else {
        text.to_string()
    }
}

/// Build a heading tree from a flat list of reflow nodes.
///
/// Heading nodes nest subsequent content until a heading of equal or lesser depth
/// is encountered.
fn build_heading_tree(flat: Vec<ReflowNode>) -> ReflowDocument {
    let mut doc = ReflowDocument {
        title: None,
        toc: Vec::new(),
        children: Vec::new(),
    };

    // Stack: (depth, node) for open headings. Deepest heading is at the top.
    let mut stack: Vec<(u32, ReflowNode)> = Vec::new();

    /// Pop all stack entries with depth >= the given depth, nesting each
    /// popped entry into its parent (or into doc.children if no parent).
    fn flush_to_depth(
        stack: &mut Vec<(u32, ReflowNode)>,
        root: &mut Vec<ReflowNode>,
        min_depth: u32,
    ) {
        while let Some((d, _)) = stack.last() {
            if *d >= min_depth {
                let (_, entry) = stack.pop().unwrap();
                if let Some((_, parent)) = stack.last_mut() {
                    if let ReflowNode::Heading { children, .. } = parent {
                        children.push(entry);
                    }
                } else {
                    root.push(entry);
                }
            } else {
                break;
            }
        }
    }

    for node in flat {
        match &node {
            ReflowNode::Heading { depth, text, section, .. } => {
                let depth = *depth;

                // Document title (depth 0): set as doc title
                if depth == 0 {
                    flush_to_depth(&mut stack, &mut doc.children, 1);
                    doc.title = Some(format_heading_title(section.as_deref(), text));
                    continue;
                }

                // Pop headings with depth >= current
                flush_to_depth(&mut stack, &mut doc.children, depth);
                stack.push((depth, node));
            }
            _ => {
                // Content node — append to current heading or root
                if let Some((_, top)) = stack.last_mut() {
                    if let ReflowNode::Heading { children, .. } = top {
                        children.push(node);
                    }
                } else {
                    doc.children.push(node);
                }
            }
        }
    }

    // Flush remaining stack
    flush_to_depth(&mut stack, &mut doc.children, 1);

    doc
}

/// Render a TOC entry list as a markdown bullet list.
/// If `with_heading` is true, prepends a `## Contents` heading.
fn render_toc_listing(entries: &[crate::types::TocEntryRendered], with_heading: bool) -> String {
    // Normalize depths to be contiguous (e.g., 0,1,4 → 0,1,2)
    let mut unique_depths: Vec<u32> = entries.iter().map(|e| e.depth).collect();
    unique_depths.sort();
    unique_depths.dedup();
    let depth_map: std::collections::HashMap<u32, usize> = unique_depths
        .iter()
        .enumerate()
        .map(|(i, &d)| (d, i))
        .collect();

    let mut toc_lines = Vec::new();
    if with_heading {
        toc_lines.push("## Contents\n".to_string());
    }
    for entry in entries {
        let normalized_depth = depth_map.get(&entry.depth).copied().unwrap_or(0);
        let indent = "  ".repeat(normalized_depth);
        let page_str = entry.page.as_deref().unwrap_or("");
        // Combine section + text into display title, then strip leader dots
        let full_title = format_heading_title(entry.section.as_deref(), &entry.text);
        let title = if let Some(dot_pos) = full_title.find(". . .") {
            full_title[..dot_pos].trim()
        } else if let Some(dot_pos) = full_title.find("...") {
            full_title[..dot_pos].trim()
        } else {
            full_title.trim()
        };
        if title.is_empty() {
            continue;
        }
        if page_str.is_empty() {
            toc_lines.push(format!("{indent}- {title}"));
        } else {
            toc_lines.push(format!("{indent}- {title} (p. {page_str})"));
        }
    }
    toc_lines.join("\n")
}

/// Check if any node in the tree is a Toc node (recursively).
fn has_toc_node(nodes: &[ReflowNode]) -> bool {
    for node in nodes {
        match node {
            ReflowNode::Toc { .. } => return true,
            ReflowNode::Heading { children, .. } => {
                if has_toc_node(children) {
                    return true;
                }
            }
            ReflowNode::FrontMatter { children } => {
                if has_toc_node(children) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Render a reflow document as Markdown.
pub fn render_markdown_from_reflow(doc: &ReflowDocument) -> String {
    let mut parts: Vec<String> = Vec::new();

    // Document title
    if let Some(ref title) = doc.title {
        let t = title.trim();
        if !t.is_empty() && t != "#" {
            parts.push(format!("# {t}"));
        }
    }

    // Render TOC at the top only if there's no embedded Toc node in the tree
    // (i.e. for papers/articles without a "Contents" heading in the TOC).
    let has_embedded_toc = has_toc_node(&doc.children);
    if !doc.toc.is_empty() && !has_embedded_toc {
        parts.push(render_toc_listing(&doc.toc, true));
    }

    // Render children
    render_children(&doc.children, &mut parts);

    let md = parts.join("\n\n");
    // Fix split bold label numbers: **Figure 3.**22 → **Figure 3.22.**
    // Happens when bold boundary cuts mid-number (font change after period)
    let md = fix_split_bold_labels(&md);
    md.trim_end().to_string()
}

/// Fix bold labels where the number is split across the bold boundary.
/// E.g. `**Figure 3.**22` → `**Figure 3.22.**`, `**Table 4.**1.` → `**Table 4.1.**`
fn fix_split_bold_labels(text: &str) -> String {
    // Match: **Label N.** followed by digits (and optional sub-number)
    // Labels: Figure, Fig., Table, Listing, Example, Footnote
    let re = regex::Regex::new(
        r"\*\*((?:Figure|Fig\.|Table|TABLE|Listing|Example|Footnote)\s+\d+\.)\*\*(\d+(?:\.\d+)*\.?)"
    ).unwrap();
    re.replace_all(text, |caps: &regex::Captures| {
        let label = caps[1].trim_end_matches('.');
        let suffix = caps[2].trim_end_matches('.');
        format!("**{label}.{suffix}.**")
    }).into_owned()
}

/// Trim trailing prose lines from code content.
///
/// The layout model sometimes includes prose text in the Algorithm region
/// bbox. This strips non-code lines from the end of the content.
fn trim_trailing_prose(code: &str) -> String {
    let lines: Vec<&str> = code.lines().collect();
    let mut last_code_line = lines.len();

    // Scan from the end, looking for lines that don't look like code
    for i in (0..lines.len()).rev() {
        let line = lines[i].trim();
        if line.is_empty() {
            continue;
        }
        // Code indicators: semicolons, braces, parentheses, //, #, operators
        let has_code_chars = line.contains(';')
            || line.contains('{')
            || line.contains('}')
            || line.contains("//")
            || line.contains('#')
            || line.contains("->")
            || line.contains("::")
            || (line.contains('(') && line.contains(')'))
            || line.ends_with(')')
            || line.ends_with("});")
            || line.ends_with(',');
        if has_code_chars {
            last_code_line = i + 1;
            break;
        }
        // If the line is long prose (no code indicators, mostly words), trim it
        if line.len() > 40 && line.split_whitespace().count() > 6 {
            continue; // keep scanning backwards
        }
        // Short line without code chars — could be a closing brace or similar
        last_code_line = i + 1;
        break;
    }

    if last_code_line < lines.len() {
        lines[..last_code_line].join("\n")
    } else {
        code.to_string()
    }
}

/// Remove Text nodes that duplicate adjacent CodeBlock content.
///
/// After code detection, the same code may exist as both a CodeBlock
/// (from Algorithm region) and a Text node (from overlapping Text region).
fn dedup_code_text(flat_nodes: &mut Vec<ReflowNode>) {
    let mut i = 0;
    while i + 1 < flat_nodes.len() {
        let (code_idx, text_idx) = if matches!(&flat_nodes[i], ReflowNode::CodeBlock { .. })
            && matches!(&flat_nodes[i + 1], ReflowNode::Text { .. })
        {
            (i, i + 1)
        } else if matches!(&flat_nodes[i], ReflowNode::Text { .. })
            && matches!(&flat_nodes[i + 1], ReflowNode::CodeBlock { .. })
        {
            (i + 1, i)
        } else {
            i += 1;
            continue;
        };

        let code_content = if let ReflowNode::CodeBlock { content, .. } = &flat_nodes[code_idx] {
            content.trim().to_string()
        } else {
            i += 1;
            continue;
        };
        let text_content = if let ReflowNode::Text { content, .. } = &flat_nodes[text_idx] {
            content.trim().to_string()
        } else {
            i += 1;
            continue;
        };

        // Check if the text is a subset of the code or vice versa
        let code_norm: String = code_content.split_whitespace().collect::<Vec<_>>().join(" ");
        let text_norm: String = text_content.split_whitespace().collect::<Vec<_>>().join(" ");

        if code_norm == text_norm
            || (text_norm.len() >= 10 && code_norm.contains(&text_norm))
        {
            // Text is identical or a subset of code — remove the text node
            flat_nodes.remove(text_idx);
            if text_idx < code_idx {
                // Removed before code, don't advance
            }
            continue;
        }
        if code_norm.len() >= 10 && text_norm.contains(&code_norm) {
            // Code is embedded within a prose text node — strip the code
            // portion from the text, keeping the surrounding prose.
            if let ReflowNode::Text { content, .. } = &mut flat_nodes[text_idx] {
                let text_lower = content.to_lowercase();
                let code_lower = code_content.to_lowercase();
                // Find the code substring in the text (case-insensitive)
                // Use normalized whitespace for matching
                let text_words: Vec<&str> = content.split_whitespace().collect();
                let code_words: Vec<&str> = code_content.split_whitespace().collect();
                if code_words.len() >= 3 {
                    // Find where code_words start in text_words
                    let mut match_start = None;
                    'outer: for si in 0..text_words.len().saturating_sub(code_words.len()) {
                        for (ci, cw) in code_words.iter().enumerate() {
                            if text_words[si + ci].to_lowercase() != cw.to_lowercase() {
                                continue 'outer;
                            }
                        }
                        match_start = Some(si);
                        break;
                    }
                    if let Some(start) = match_start {
                        // Remove the matched words from the text
                        let before: Vec<&str> = text_words[..start].to_vec();
                        let after_idx = start + code_words.len();
                        let after: Vec<&str> = if after_idx < text_words.len() {
                            text_words[after_idx..].to_vec()
                        } else {
                            vec![]
                        };
                        let new_text = format!("{} {}", before.join(" "), after.join(" "));
                        *content = new_text.trim().to_string();
                    }
                }
            }
            i += 1;
            continue;
        }
        i += 1;
    }
}

/// Detect consecutive Text nodes that form bulleted or numbered lists
/// and replace them with a single List node.
fn detect_lists(flat_nodes: &mut Vec<ReflowNode>) {
    let mut i = 0;
    while i < flat_nodes.len() {
        if let ReflowNode::Text { content, .. } = &flat_nodes[i] {
            let trimmed = content.trim();

            // Check for numbered list starting with "1." or "1)"
            if trimmed.starts_with("1.") || trimmed.starts_with("1)") {
                let sep = if trimmed.starts_with("1.") { '.' } else { ')' };
                let mut j = i + 1;
                let mut expected = 2u32;
                while j < flat_nodes.len() {
                    if let ReflowNode::Text { content: c2, .. } = &flat_nodes[j] {
                        let t2 = c2.trim();
                        let prefix = format!("{expected}{sep}");
                        if t2.starts_with(&prefix) {
                            expected += 1;
                            j += 1;
                            continue;
                        }
                    }
                    break;
                }
                if j - i >= 2 {
                    // Extract items, stripping number prefix
                    let items: Vec<String> = (i..j)
                        .map(|k| {
                            if let ReflowNode::Text { content, .. } = &flat_nodes[k] {
                                let t = content.trim();
                                // Strip "N." or "N)" prefix
                                if let Some(dot) = t.find(sep) {
                                    t[dot + 1..].trim().to_string()
                                } else {
                                    t.to_string()
                                }
                            } else {
                                String::new()
                            }
                        })
                        .collect();
                    // Replace the range with a single List node
                    flat_nodes.splice(i..j, std::iter::once(ReflowNode::List {
                        list_type: "numbered".to_string(),
                        items,
                    }));
                    continue; // re-check from same position
                }
            }

            // Check for period-bullet list: ". *Title*" or ". Text" patterns
            // Used in some textbooks as a bullet/list marker
            if trimmed.starts_with(". ") && trimmed.len() > 4 {
                let after_dot = &trimmed[2..];
                if after_dot.starts_with('*') || after_dot.starts_with(|c: char| c.is_uppercase()) {
                    let mut j = i + 1;
                    let mut count = 1;
                    while j < flat_nodes.len() {
                        if let ReflowNode::Text { content: c2, .. } = &flat_nodes[j] {
                            let t2 = c2.trim();
                            if t2.starts_with(". ") && t2.len() > 4 {
                                count += 1;
                                j += 1;
                                continue;
                            }
                        }
                        break;
                    }
                    if count >= 3 {
                        let items: Vec<String> = (i..j)
                            .filter_map(|k| {
                                if let ReflowNode::Text { content, .. } = &flat_nodes[k] {
                                    Some(content.trim()[2..].to_string())
                                } else {
                                    None
                                }
                            })
                            .collect();
                        flat_nodes.splice(i..j, std::iter::once(ReflowNode::List {
                            list_type: "bulleted".to_string(),
                            items,
                        }));
                        continue;
                    }
                }
            }

            // Check for bulleted list (skip Formula nodes between items)
            let prefix_len = bullet_prefix_len(trimmed);
            if prefix_len > 0 {
                let bullet_str: String = trimmed.chars().take(prefix_len).collect();
                let mut j = i + 1;
                let mut text_count = 1; // count of matching text items
                while j < flat_nodes.len() {
                    match &flat_nodes[j] {
                        ReflowNode::Text { content: c2, .. } => {
                            let t2 = c2.trim();
                            if bullet_prefix_len(t2) == prefix_len {
                                let p2: String = t2.chars().take(prefix_len).collect();
                                if p2 == bullet_str {
                                    text_count += 1;
                                    j += 1;
                                    continue;
                                }
                            }
                            break; // non-matching text → end of list
                        }
                        // Skip formula images and footnotes between list items
                        ReflowNode::Formula { .. } | ReflowNode::Footnote { .. } => {
                            j += 1;
                            continue;
                        }
                        _ => break,
                    }
                }
                // For single-letter bullets: "A" and "I" are valid English
                // words, so require 3+ items. All other single letters (B, C,
                // y, etc.) can't start a word alone, so 2+ items suffices.
                let min_items = if prefix_len == 1 {
                    let ch = bullet_str.chars().next().unwrap();
                    if ch == 'A' || ch == 'I' || ch == 'a' { 3 } else { 2 }
                } else {
                    3
                };
                if text_count >= min_items {
                    // For single letters, verify it's a real bullet
                    let is_real = if prefix_len == 1 && bullet_str.chars().next().unwrap().is_ascii_alphabetic() {
                        let second_words: Vec<&str> = (i..j)
                            .filter_map(|k| {
                                if let ReflowNode::Text { content, .. } = &flat_nodes[k] {
                                    content.trim().split_whitespace().nth(1)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let unique: std::collections::HashSet<&str> = second_words.iter().copied().collect();
                        unique.len() > second_words.len() / 2
                    } else {
                        true
                    };
                    if is_real {
                        let items: Vec<String> = (i..j)
                            .filter_map(|k| {
                                if let ReflowNode::Text { content, .. } = &flat_nodes[k] {
                                    let t = content.trim();
                                    let after: String = t.chars().skip(1).collect();
                                    Some(after.trim_start().to_string())
                                } else {
                                    None // skip Formula nodes
                                }
                            })
                            .collect();
                        flat_nodes.splice(i..j, std::iter::once(ReflowNode::List {
                            list_type: "bulleted".to_string(),
                            items,
                        }));
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
}

/// Detect Text nodes that look like source code and replace with CodeBlock nodes.
/// Also promote Algorithm nodes to CodeBlock when the language can be identified.
/// Then merge consecutive CodeBlock/Algorithm nodes into a single block.
fn detect_code_blocks(flat_nodes: &mut Vec<ReflowNode>) {
    // Step 1: Promote Text → CodeBlock and Algorithm → CodeBlock.
    // For Algorithm nodes, also trim trailing prose that the layout model
    // accidentally included in the code region bbox.
    for node in flat_nodes.iter_mut() {
        match node {
            ReflowNode::Text { content, .. } => {
                if looks_like_code(content) {
                    let code_content = content.clone();
                    let lang = guess_language(&code_content);
                    *node = ReflowNode::CodeBlock {
                        content: code_content,
                        language: lang,
                    };
                }
            }
            ReflowNode::Algorithm { content } => {
                // Algorithm regions can be: real pseudocode, programming code, or
                // misclassified prose. Check structural signals to decide.
                let is_pseudocode = looks_like_pseudocode(content);
                if is_pseudocode || looks_like_code(content) || code_score(content) >= 2 {
                    // Check if it's actually prose with math — full sentences without
                    // algorithmic structure shouldn't be code blocks
                    let has_math = content.matches('$').count() >= 2
                        || content.contains("[[FORMULA")
                        || count_italic_spans(content) >= 2;
                    if has_math && !is_pseudocode && !looks_like_code(content) {
                        // Math-heavy prose misclassified as Algorithm
                        *node = ReflowNode::Text {
                            content: content.replace('\n', " "),
                            footnotes: Vec::new(),
                        };
                    } else {
                        let trimmed = trim_trailing_prose(content);
                        let lang = if is_pseudocode {
                            None
                        } else {
                            guess_language(&trimmed)
                        };
                        *node = ReflowNode::CodeBlock {
                            content: trimmed,
                            language: lang,
                        };
                    }
                } else {
                    // Not code — demote to Text so it renders as prose
                    *node = ReflowNode::Text {
                        content: content.replace('\n', " "),
                        footnotes: Vec::new(),
                    };
                }
            }
            _ => {}
        }
    }

    // Step 2: Merge consecutive CodeBlock nodes
    let mut i = 0;
    while i + 1 < flat_nodes.len() {
        let is_code_pair = matches!(&flat_nodes[i], ReflowNode::CodeBlock { .. })
            && matches!(&flat_nodes[i + 1], ReflowNode::CodeBlock { .. });
        if is_code_pair {
            if let ReflowNode::CodeBlock { content: c2, language: l2 } = flat_nodes.remove(i + 1) {
                if let ReflowNode::CodeBlock { content: c1, language: l1 } = &mut flat_nodes[i] {
                    c1.push('\n');
                    c1.push_str(&c2);
                    // Keep the more specific language
                    if l1.is_none() && l2.is_some() {
                        *l1 = l2;
                    }
                }
            }
            continue; // re-check same position for further merging
        }
        i += 1;
    }
}

/// Heuristic: does this text look like source code?
///
/// Uses a scoring system based on common code patterns. Each indicator
/// contributes points; the text is classified as code if the total
/// score reaches a threshold.
/// Detect pseudocode/algorithm structure: for/if/while/end keywords.
/// These should stay as code blocks even if they contain [[FORMULA]].
fn looks_like_pseudocode(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 3 {
        return false;
    }

    let algo_keywords = [
        "for ", "end for", "end if", "if ", "while ", "end while", "do", "then",
        "else", "return ", "repeat", "until ",
    ];

    let mut keyword_lines = 0;
    for line in &lines {
        let trimmed = line.trim().to_lowercase();
        // Strip leading line numbers (e.g. "7     for ...")
        let without_num = trimmed.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.');
        let without_num = without_num.trim_start();
        if algo_keywords
            .iter()
            .any(|kw| without_num.starts_with(kw) || without_num.ends_with(kw))
        {
            keyword_lines += 1;
        }
    }

    // Need at least 3 algorithmic keyword lines
    keyword_lines >= 3
}

/// Count italic spans (*word*) in text — matches `*non-space..non-space*` pairs.
/// Bare `*` used as multiplication (e.g. `a * b`) is NOT counted.
fn count_italic_spans(text: &str) -> usize {
    let mut count = 0;
    let mut in_italic = false;
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'*' {
            if !in_italic {
                // Opening: * followed by non-space
                if i + 1 < bytes.len() && bytes[i + 1] != b' ' && bytes[i + 1] != b'*' {
                    in_italic = true;
                }
            } else {
                // Closing: preceded by non-space
                if i > 0 && bytes[i - 1] != b' ' && bytes[i - 1] != b'*' {
                    count += 1;
                    in_italic = false;
                }
            }
        }
    }
    count
}

fn looks_like_code(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.len() < 15 {
        return false;
    }

    // === Strong negative signals — definitely not code ===

    // LaTeX math ($...$) or formula placeholders → academic prose
    if trimmed.matches('$').count() >= 2 {
        return false;
    }
    if trimmed.contains("[[FORMULA") {
        return false;
    }
    // Italic markers (*word*) → formatted prose, not code.
    // Count actual italic spans, not bare * (which appear as multiplication in code).
    if count_italic_spans(trimmed) >= 2 {
        return false;
    }

    // Prose detection: many words with average length typical of English
    let word_count = trimmed.split_whitespace().count();
    let avg_word_len = if word_count > 0 {
        trimmed.split_whitespace().map(|w| w.len()).sum::<usize>() / word_count
    } else { 0 };

    // Long text with typical English word lengths → prose
    if word_count > 20 && avg_word_len >= 3 && avg_word_len <= 8 {
        let code_symbol_count = trimmed.chars()
            .filter(|c| matches!(c, '{' | '}' | ';' | '#'))
            .count();
        // Need > 3% structural code symbols to override prose detection
        if (code_symbol_count as f32 / trimmed.len() as f32) < 0.03 {
            return false;
        }
    }

    // Require higher score threshold for longer texts (harder to be sure)
    let threshold = if word_count > 30 { 5 } else { 3 };
    code_score(trimmed) >= threshold
}

/// Score text for code-likeness. Higher = more likely code.
pub fn code_score(text: &str) -> u32 {
    let mut score: u32 = 0;

    // === Preprocessor directives (very strong: C/C++) ===
    if text.contains("#include") || text.contains("#define") || text.contains("#ifdef")
        || text.contains("#ifndef") || text.contains("#pragma")
    {
        score += 3;
    }

    // === Language keywords with syntax ===
    // C/C++/Java/C#
    if text.contains("void ") && (text.contains('(') || text.contains('{')) { score += 2; }
    if text.contains("public:") || text.contains("private:") || text.contains("protected:") { score += 2; }
    if text.contains("class ") && text.contains('{') { score += 2; }
    if text.contains("struct ") && text.contains('{') { score += 2; }
    if text.contains("enum ") && text.contains('{') { score += 2; }
    if text.contains("namespace ") && text.contains('{') { score += 2; }
    if text.contains("template") && text.contains('<') { score += 2; }
    if text.contains("typedef ") { score += 2; }
    if text.contains("static_cast") || text.contains("dynamic_cast") || text.contains("reinterpret_cast") { score += 2; }

    // Rust
    if text.contains("fn ") && text.contains("->") { score += 2; }
    if text.contains("let mut ") || text.contains("impl ") { score += 2; }
    if text.contains("pub fn ") || text.contains("pub struct ") { score += 2; }

    // Python
    if text.contains("def ") && text.contains("self") { score += 2; }
    if text.contains("import ") && text.contains("from ") { score += 2; }

    // JavaScript/TypeScript
    if text.contains("const ") && text.contains(" => ") { score += 2; }
    if text.contains("function ") && text.contains('{') { score += 2; }

    // GLSL shaders
    if text.contains("uniform ") || text.contains("varying ") { score += 2; }
    if text.contains("gl_Position") || text.contains("gl_FragColor") { score += 2; }
    if text.contains("vec2 ") || text.contains("vec3 ") || text.contains("vec4 ") { score += 1; }
    // HLSL shaders
    if text.contains("float2 ") || text.contains("float3 ") || text.contains("float4 ") { score += 1; }
    if text.contains("uint2 ") || text.contains("uint3 ") || text.contains("uint4 ") { score += 1; }
    if text.contains("half2 ") || text.contains("half3 ") || text.contains("half4 ") { score += 1; }
    if text.contains("SV_Position") || text.contains("SV_Target") || text.contains("SV_DispatchThreadID") { score += 2; }
    if text.contains("cbuffer ") || text.contains("StructuredBuffer") || text.contains("RWTexture") { score += 2; }
    if text.contains("[numthreads") || text.contains("[maxvertexcount") { score += 2; }

    // === Operators and syntax patterns ===
    let semicolons = text.matches(';').count();
    if semicolons >= 3 { score += 1; }
    if semicolons >= 6 { score += 1; }

    let braces = text.matches('{').count() + text.matches('}').count();
    if braces >= 2 { score += 1; }
    if braces >= 4 { score += 1; }

    if text.contains("->") && text.contains(';') { score += 1; }
    if text.contains("::") { score += 1; }
    if text.contains("!=") || text.contains("==") || text.contains(">=") || text.contains("<=") { score += 1; }

    // Control flow with parens — only score when followed by code-like content
    // "if (" in prose like "if (and only if)" is not code
    if (text.contains("if (") || text.contains("if(")) && text.contains('{') { score += 1; }
    if (text.contains("for (") || text.contains("for(")) && text.contains('{') { score += 1; }
    if (text.contains("while (") || text.contains("while(")) && text.contains('{') { score += 1; }
    if text.contains("switch (") || text.contains("switch(") { score += 1; }
    if text.contains("return;") || text.contains("return 0") || text.contains("return(") { score += 1; }

    // Type/keyword patterns — only when combined with assignment or declaration syntax
    if text.contains("NULL") || text.contains("nullptr") { score += 1; }
    if (text.contains("int ") || text.contains("char ") || text.contains("float ") || text.contains("double "))
        && (text.contains('=') || text.contains(';') || text.contains('('))
    { score += 1; }
    if text.contains("const ") && text.contains('*') { score += 1; }

    // Windows API types (stronger indicator if combined with semicolons)
    let has_win_api = text.contains("HANDLE ") || text.contains("DWORD ") || text.contains("BOOL ")
        || text.contains("LPVOID") || text.contains("HINSTANCE") || text.contains("HWND")
        || text.contains("TCHAR ") || text.contains("LPTSTR") || text.contains("LPCTSTR")
        || text.contains("HRESULT") || text.contains("LPCWSTR") || text.contains("WCHAR ");
    if has_win_api {
        score += 1;
        if semicolons >= 1 { score += 1; } // Win API + semicolons = definitely code
    }

    // Assignment patterns
    let assignments = text.matches(" = ").count();
    if assignments >= 2 { score += 1; }

    // Comment patterns
    if text.contains("//") || text.contains("/*") || text.contains("*/") { score += 1; }

    score
}

/// Guess the programming language from code content.
fn guess_language(code: &str) -> Option<String> {
    // C/C++ indicators
    let c_score = {
        let mut s = 0u32;
        if code.contains("#include") { s += 3; }
        if code.contains("#define") || code.contains("#ifdef") { s += 2; }
        if code.contains("HANDLE ") || code.contains("DWORD ") || code.contains("HWND") { s += 2; }
        if code.contains("printf") || code.contains("malloc") || code.contains("sizeof") { s += 1; }
        if code.contains("::") { s += 1; } // C++
        if code.contains("std::") { s += 2; } // C++
        if code.contains("void ") { s += 1; }
        s
    };

    // Rust indicators
    let rust_score = {
        let mut s = 0u32;
        if code.contains("fn ") && code.contains("->") { s += 3; }
        if code.contains("let mut ") { s += 2; }
        if code.contains("impl ") { s += 2; }
        if code.contains("pub fn ") || code.contains("pub struct ") { s += 2; }
        if code.contains("unwrap()") || code.contains(".expect(") { s += 1; }
        s
    };

    // Python indicators
    let python_score = {
        let mut s = 0u32;
        if code.contains("def ") && code.contains(":") { s += 2; }
        if code.contains("self.") { s += 2; }
        if code.contains("import ") { s += 1; }
        if code.contains("print(") { s += 1; }
        s
    };

    // JavaScript indicators
    let js_score = {
        let mut s = 0u32;
        if code.contains("const ") && code.contains("=>") { s += 2; }
        if code.contains("function ") { s += 1; }
        if code.contains("console.log") { s += 2; }
        if code.contains("require(") || code.contains("module.exports") { s += 2; }
        s
    };

    // GLSL indicators
    let glsl_score = {
        let mut s = 0u32;
        if code.contains("uniform ") || code.contains("varying ") { s += 2; }
        if code.contains("gl_Position") || code.contains("gl_FragColor") { s += 3; }
        if code.contains("#version") { s += 3; }
        s
    };

    // HLSL indicators
    let hlsl_score = {
        let mut s = 0u32;
        if code.contains("float2 ") || code.contains("float3 ") || code.contains("float4 ") { s += 1; }
        if code.contains("uint2 ") || code.contains("uint3 ") || code.contains("uint4 ") { s += 1; }
        if code.contains("SV_Position") || code.contains("SV_Target") { s += 3; }
        if code.contains("cbuffer ") || code.contains("StructuredBuffer") { s += 2; }
        if code.contains("[numthreads") { s += 3; }
        s
    };

    let max = c_score.max(rust_score).max(python_score).max(js_score).max(glsl_score).max(hlsl_score);
    if max < 2 {
        return None;
    }

    if hlsl_score == max && hlsl_score >= 2 { Some("hlsl".to_string()) }
    else if c_score == max { Some("c".to_string()) }
    else if rust_score == max { Some("rust".to_string()) }
    else if python_score == max { Some("python".to_string()) }
    else if js_score == max { Some("javascript".to_string()) }
    else if glsl_score == max { Some("glsl".to_string()) }
    else { None }
}

/// Check if a string starts with a bullet-like marker.
/// Returns the marker length (0 if not a bullet).
fn bullet_prefix_len(text: &str) -> usize {
    let trimmed = text.trim();
    if trimmed.len() < 3 {
        return 0;
    }
    let first = trimmed.chars().next().unwrap();
    // Known bullet symbols — single char
    if matches!(first, '■' | '▪' | '●' | '•' | '►' | '▸' | '◆' | '◇'
        | '★' | '☆' | '→' | '–' | '—' | '◦' | '⁃' | '‣')
    {
        if trimmed.chars().nth(1) == Some(' ') {
            return 1; // 1 character (not bytes)
        }
    }
    // Single letter as bullet — uppercase OR lowercase.
    // Pattern: "B Firstly" or "y multiplying" (single letter + space + word)
    // We validate later with 3+ consecutive items to avoid false positives.
    if first.is_ascii_alphabetic() {
        let chars: Vec<char> = trimmed.chars().take(3).collect();
        if chars.len() >= 3 && chars[1] == ' ' {
            // Check: is the character after the space uppercase or a common
            // item-start pattern? If the rest looks like a sentence starting
            // with a capitalized word, it's a bullet.
            // We validate later by checking 3+ consecutive — so just return
            // the prefix length and let the grouping check handle it.
            return 1;
        }
    }
    0
}

/// Recursively render reflow nodes into markdown parts.
fn render_children(children: &[ReflowNode], parts: &mut Vec<String>) {
    for (node_idx, node) in children.iter().enumerate() {
        let _ = node_idx; // used by some match arms
        match node {
            ReflowNode::Heading {
                depth,
                text,
                section,
                children,
            } => {
                // Skip Index section — page-number references are useless in markdown
                let combined = format_heading_title(section.as_deref(), text);
                let title_upper = combined.trim().to_ascii_uppercase();
                if title_upper.contains("INDEX") {
                    continue;
                }
                // Skip headings with no content — these are synthetic TOC
                // injections that couldn't find matching body content.
                if children.is_empty() {
                    continue;
                }
                let depth = (*depth as usize).min(5); // cap at h6
                let hashes = "#".repeat(depth + 1);
                let title = collapse_spaced_letters(&combined);
                let title = split_concatenated_allcaps(&title);
                let title = fix_missing_spaces_in_title(&title);
                // Strip bold markers — heading markdown already conveys emphasis
                let title = title.replace("**", "");
                parts.push(format!("{hashes} {title}"));
                render_children(children, parts);
            }
            ReflowNode::Text { content, footnotes, .. } => {
                if !content.trim().is_empty() {
                    parts.push(escape_leading_hashes(content.trim()));
                }
                // Render footnotes after the paragraph
                for footnote in footnotes {
                    parts.push(footnote.clone());
                }
            }
            ReflowNode::Footnote { marker, content } => {
                parts.push(format!("**Footnote {marker}.** {content}"));
            }
            ReflowNode::List { list_type, items } => {
                let mut list_lines = Vec::new();
                for (idx, item) in items.iter().enumerate() {
                    if list_type == "numbered" {
                        list_lines.push(format!("{}. {}", idx + 1, item));
                    } else {
                        list_lines.push(format!("- {}", item));
                    }
                }
                parts.push(list_lines.join("\n"));
            }
            ReflowNode::FormattedText { content } => {
                // Render as-is, preserving layout
                if !content.trim().is_empty() {
                    parts.push(content.clone());
                }
            }
            ReflowNode::CodeBlock { content, language } => {
                let lang = language.as_deref().unwrap_or("");
                parts.push(format!("```{lang}\n{content}\n```"));
            }
            ReflowNode::FrontMatter { children } => {
                // Only render Preface from front matter. Everything else
                // (copyright, dedication, acknowledgments) is skipped.
                let mut in_preface = false;
                for child in children {
                    if let ReflowNode::Text { content, .. } = child {
                        let t = content.trim().to_lowercase();
                        if t == "preface" || t == "## preface" || t.starts_with("preface") {
                            in_preface = true;
                            parts.push(format!("## Preface"));
                            continue;
                        }
                    }
                    if let ReflowNode::Heading { text, .. } = child {
                        let t = text.trim().to_lowercase();
                        if t == "preface" {
                            in_preface = true;
                        } else {
                            in_preface = false;
                        }
                    }
                    if in_preface {
                        let mut child_parts = Vec::new();
                        render_children(std::slice::from_ref(child), &mut child_parts);
                        parts.extend(child_parts);
                    }
                }
            }
            ReflowNode::Formula { content, path } => {
                if let Some(c) = content {
                    parts.push(c.clone());
                } else if let Some(p) = path {
                    parts.push(format!("![formula]({p})"));
                }
            }
            ReflowNode::Figure { path, caption } => {
                if path.is_empty() {
                    if let Some(cap) = caption {
                        parts.push(cap.clone());
                    }
                } else if let Some(cap) = caption {
                    parts.push(format!("![]({path})\n\n{cap}"));
                } else {
                    parts.push(format!("![]({path})"));
                }
            }
            ReflowNode::Table { content, caption } => {
                if let Some(cap) = caption {
                    parts.push(format!("{content}\n\n{cap}"));
                } else {
                    parts.push(content.clone());
                }
            }
            ReflowNode::Algorithm { content } => {
                // Wrap algorithm/code content in fenced code block
                parts.push(format!("```\n{content}\n```"));
            }
            ReflowNode::FigureGroup { path, caption } => {
                if path.is_empty() {
                    if let Some(cap) = caption {
                        parts.push(cap.clone());
                    }
                } else if let Some(cap) = caption {
                    parts.push(format!("![]({path})\n\n{cap}"));
                } else {
                    parts.push(format!("![]({path})"));
                }
            }
            ReflowNode::References { content } => {
                parts.push(content.clone());
            }
            ReflowNode::FootnoteBlock { content } => {
                parts.push(content.clone());
            }
            ReflowNode::Toc { entries } => {
                // Render embedded TOC listing (no heading — parent
                // Heading node already provides "Contents" or similar).
                if !entries.is_empty() {
                    parts.push(render_toc_listing(entries, false));
                }
            }
        }
    }
}

/// Collapse decorative per-letter spacing in heading titles.
///
/// Some PDFs use `C H A P T E R  3` style headings. Detects runs of ≥4
/// single-letter words separated by single spaces and collapses them.
fn collapse_spaced_letters(title: &str) -> String {
    // Split into words by double-space (which separates logical words in these titles)
    // or detect the pattern: ≥4 consecutive single-char tokens separated by single space.
    let chars: Vec<char> = title.chars().collect();
    if chars.len() < 7 {
        return title.to_string();
    }

    // Check if this title has the spaced-letter pattern:
    // Count single-letter tokens (letter followed by space followed by letter)
    let tokens: Vec<&str> = title.split_whitespace().collect();
    let single_count = tokens.iter().filter(|t| t.len() == 1 && t.chars().next().map_or(false, |c| c.is_alphabetic())).count();

    // If majority of tokens are single letters (≥4 and >50% of alpha tokens), collapse
    if single_count < 4 {
        return title.to_string();
    }
    let alpha_tokens = tokens.iter().filter(|t| t.chars().any(|c| c.is_alphabetic())).count();
    if alpha_tokens == 0 || (single_count as f32 / alpha_tokens as f32) < 0.5 {
        return title.to_string();
    }

    // Collapse: join consecutive single-letter tokens into words,
    // separated by double-space boundaries or non-single-letter tokens.
    let mut result = String::new();
    let mut i = 0;
    while i < tokens.len() {
        if !result.is_empty() {
            result.push(' ');
        }
        if tokens[i].len() == 1 && tokens[i].chars().next().map_or(false, |c| c.is_alphabetic()) {
            // Start of a spaced-letter word — collect consecutive single letters
            let start = i;
            while i < tokens.len() && tokens[i].len() == 1 && tokens[i].chars().next().map_or(false, |c| c.is_alphabetic()) {
                result.push_str(tokens[i]);
                i += 1;
            }
            // If we only got one letter, it was just a short word — that's fine
            let _ = start; // already pushed
        } else {
            result.push_str(tokens[i]);
            i += 1;
        }
    }
    result
}

/// Fix missing spaces in heading titles where words run together.
///
/// Detects `lowercaseUppercase` boundaries (e.g. "OrbitSatellites" → "Orbit Satellites")
/// and `letterOf`/`letterThe` patterns (e.g. "ofa" → "of a").
/// Split all-caps concatenated words like "CLASSICALLAMBDACALCULUS".
///
/// Uses a dictionary of common academic/technical words. Only applies to
/// all-uppercase single tokens ≥12 chars. Falls back to original if
/// the dictionary can't fully decompose the string.
fn split_concatenated_allcaps(title: &str) -> String {
    // Process each word in the title independently
    let words: Vec<&str> = title.split_whitespace().collect();
    let mut any_split = false;
    let mut result_words: Vec<String> = Vec::new();

    for word in &words {
        if word.len() >= 12 && word.chars().all(|c| c.is_ascii_uppercase()) {
            if let Some(split) = try_split_allcaps(word) {
                result_words.push(split);
                any_split = true;
                continue;
            }
        }
        result_words.push(word.to_string());
    }

    if any_split {
        result_words.join(" ")
    } else {
        title.to_string()
    }
}

fn try_split_allcaps(s: &str) -> Option<String> {
    // Common words in academic headings, sorted longest-first for greedy match
    static DICT: &[&str] = &[
        "INTRODUCTION", "DEFINITIONS", "CONSTRUCTION", "COMBINATORY",
        "INTERSECTION", "APPLICATIONS", "OPTIMIZATION",
        "FUNDAMENTAL", "INDEPENDENT", "ARBITRARILY", "PROGRAMMING",
        "STRATEGIES", "APPENDICES", "REFERENCES", "ALGORITHMS", "SEARCHING",
        "DIMENSIONS", "STRUCTURES",
        "CLASSICAL", "VARIABLES", "REDUCTION", "GEOMETRIC", "NUMERICAL",
        "SEARCHING", "GENERALIZED",
        "LABELLED", "SENSIBLE", "CALCULUS", "APPENDIX", "THEORIES",
        "PROBLEMS", "ORIENTED", "PARALLEL", "RESULTS", "SUMMARY",
        "OBJECTS", "MODELS", "LAMBDA", "GROUPS", "OTHER", "TYPED",
        "INDEX", "DATA", "TYPE", "FREE", "PURE",
    ];

    let mut result = Vec::new();
    let mut remaining = s;

    while !remaining.is_empty() {
        let mut found = false;
        for word in DICT {
            if remaining.starts_with(word) {
                result.push(*word);
                remaining = &remaining[word.len()..];
                found = true;
                break;
            }
        }
        if !found {
            // Single char left over after successful splits? Skip.
            if remaining.len() <= 2 && !result.is_empty() {
                result.push(remaining);
                return Some(result.join(" "));
            }
            return None;
        }
    }

    if result.len() >= 2 {
        Some(result.join(" "))
    } else {
        None
    }
}

fn fix_missing_spaces_in_title(title: &str) -> String {
    let chars: Vec<char> = title.chars().collect();
    if chars.len() < 3 {
        return title.to_string();
    }
    let mut result = String::with_capacity(title.len() + 8);
    for (i, &ch) in chars.iter().enumerate() {
        if i > 0 {
            let prev = chars[i - 1];
            // Insert space at lowercase→uppercase boundary
            // but not for patterns like "MHz", "kHz", "pH", "iPhone"
            if prev.is_lowercase() && ch.is_uppercase() {
                result.push(' ');
            }
        }
        result.push(ch);
    }
    result
}

/// Escape lines that start with `#` so they aren't parsed as markdown headings.
///
/// C/C++/GLSL preprocessor directives (`#include`, `#define`, `#version`, etc.)
/// and other code lines starting with `#` would otherwise be rendered as headings.
fn escape_leading_hashes(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for (i, line) in text.lines().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        // Only escape # when it looks like a code directive (#include, #define,
        // #ifdef, #version, etc.) or a standalone # symbol, NOT when it looks
        // like a markdown heading (## Title).
        if line.starts_with('#') {
            let after_hashes = line.trim_start_matches('#');
            let is_heading_like = after_hashes.starts_with(' ')
                && after_hashes.len() > 1
                && after_hashes.chars().nth(1).map_or(false, |c| c.is_uppercase());
            if !is_heading_like {
                out.push('\\');
            }
        }
        out.push_str(line);
    }
    out
}

/// Bold the label prefix in a figure/table caption.
///
/// Turns `"Fig. 1. Example..."` into `"**Fig. 1.** Example..."` and
/// `"Table 1. Performance..."` into `"**Table 1.** Performance..."`.
/// Leaves text unchanged if no recognized label prefix is found.
fn bold_caption_label(text: &str) -> String {
    use std::fmt::Write;

    // Match patterns: "Fig. N." / "Figure N." / "Table N." / "Algorithm N."
    // where N can be multi-part like "1" or "1a" or absent for sub-captions
    let prefixes = ["Fig.", "Figure", "Table", "Algorithm"];
    for prefix in prefixes {
        if !text.starts_with(prefix) {
            continue;
        }
        // Find the second period after the number: "Fig. 1." or "Table 1."
        let after_prefix = &text[prefix.len()..];
        if let Some(dot_pos) = after_prefix.find('.') {
            let end = prefix.len() + dot_pos + 1; // include the '.'
            let mut result = String::with_capacity(text.len() + 4);
            let _ = write!(result, "**{}**{}", &text[..end], &text[end..]);
            return result;
        }
    }
    text.to_string()
}

/// Convert a single region to its Markdown representation.
/// Strip `$$` or `$` delimiters the model may have included in its output.
fn strip_dollar_delimiters(s: &str) -> &str {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix("$$") {
        inner.strip_suffix("$$").unwrap_or(inner).trim()
    } else if let Some(inner) = s.strip_prefix('$') {
        inner.strip_suffix('$').unwrap_or(inner).trim()
    } else {
        s
    }
}

fn region_to_markdown(region: &Region) -> String {
    match region.kind {
        RegionKind::Title => {
            if let Some(ref text) = region.text {
                format!("# {text}")
            } else {
                String::new()
            }
        }
        RegionKind::ParagraphTitle => {
            if let Some(ref text) = region.text {
                let t = text.trim();
                if t.is_empty() {
                    String::new()
                } else {
                    format!("## {t}")
                }
            } else {
                String::new()
            }
        }
        RegionKind::Text
        | RegionKind::VerticalText
        | RegionKind::Abstract
        | RegionKind::SidebarText
        | RegionKind::References => region.text.clone().unwrap_or_default(),

        RegionKind::Table => {
            let table_md = region
                .text
                .as_deref()
                .filter(|s| !s.is_empty())
                .or(region.html.as_deref())
                .unwrap_or_default();
            let caption_text = region
                .caption
                .as_ref()
                .and_then(|c| c.text.as_deref());
            if let Some(cap) = caption_text {
                format!("{table_md}\n\n{}", bold_caption_label(cap))
            } else {
                table_md.to_string()
            }
        }
        RegionKind::DisplayFormula => {
            if let Some(ref latex) = region.latex {
                let s = strip_dollar_delimiters(latex);
                format!("$${s}$$")
            } else {
                String::new()
            }
        }
        RegionKind::InlineFormula => {
            // Orphan inline formula (not consumed by a text region)
            if let Some(ref latex) = region.latex {
                let s = strip_dollar_delimiters(latex);
                format!("${s}$")
            } else {
                String::new()
            }
        }
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => {
            let path = region.image_path.as_deref().unwrap_or("");
            if path.is_empty() {
                // No image path — emit just the caption if present
                region
                    .caption
                    .as_ref()
                    .and_then(|c| c.text.as_deref())
                    .map(|cap| bold_caption_label(cap))
                    .unwrap_or_default()
            } else {
                let caption_text = region
                    .caption
                    .as_ref()
                    .and_then(|c| c.text.as_deref());
                if let Some(cap) = caption_text {
                    format!("![]({path})\n\n{}", bold_caption_label(cap))
                } else {
                    format!("![]({path})")
                }
            }
        }
        RegionKind::Algorithm => {
            if let Some(ref text) = region.text {
                text.clone()
            } else {
                String::new()
            }
        }
        RegionKind::FormattedText => {
            if let Some(ref text) = region.text {
                text.clone()
            } else {
                String::new()
            }
        }
        RegionKind::Footnote => {
            // Keep footnote text — will be converted to a Footnote node
            // and placed after the paragraph that references it.
            if let Some(ref text) = region.text {
                text.clone()
            } else {
                String::new()
            }
        }
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => {
            // Consumed captions are handled by their parent (Image/Table/Chart).
            // Orphan captions that weren't consumed still get rendered with bold labels.
            if region.consumed {
                String::new()
            } else if let Some(ref text) = region.text {
                bold_caption_label(text)
            } else {
                String::new()
            }
        }
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber | RegionKind::TOC => {
            // Skip page structural elements
            String::new()
        }
        RegionKind::FormulaNumber => {
            // Formula numbers are inline metadata, skip
            String::new()
        }
        RegionKind::FigureGroup => {
            let mut parts = Vec::new();
            // Single composite image for the entire figure group
            if let Some(ref path) = region.image_path {
                parts.push(format!("![]({path})"));
            }
            // Group-level caption
            if let Some(ref cap) = region.caption {
                if let Some(ref text) = cap.text {
                    parts.push(bold_caption_label(text));
                }
            }
            parts.join("\n\n")
        }
    }
}

// ── Debug visualization ──────────────────────────────────────────────

/// Map a RegionKind to an (R, G, B) debug visualization color.
pub fn region_color_rgb(kind: RegionKind) -> [u8; 3] {
    match kind {
        RegionKind::Title | RegionKind::ParagraphTitle => [255, 50, 50],
        RegionKind::Text
        | RegionKind::VerticalText
        | RegionKind::Abstract
        | RegionKind::SidebarText => [50, 100, 255],
        RegionKind::References | RegionKind::Footnote | RegionKind::TOC => [0, 180, 180],
        RegionKind::Table => [0, 200, 0],
        RegionKind::DisplayFormula | RegionKind::FormulaNumber => [200, 0, 200],
        RegionKind::InlineFormula => [255, 100, 255],
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => [255, 140, 0],
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => [220, 200, 0],
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber => [150, 150, 150],
        RegionKind::Algorithm => [0, 160, 120],
        RegionKind::FormattedText => [120, 160, 0],
        RegionKind::FigureGroup => [0, 200, 200],
    }
}

/// Draw a colored bounding box for a detected region onto an RGBA image.
///
/// `bbox_px` is `[x1, y1, x2, y2]` in pixel coordinates matching the image.
pub fn draw_region_box(
    img: &mut image::RgbaImage,
    kind: RegionKind,
    bbox_px: [f32; 4],
    thickness: u32,
) {
    let [r, g, b] = region_color_rgb(kind);
    let color = image::Rgba([r, g, b, 200]);
    let (iw, ih) = (img.width(), img.height());

    let x1 = (bbox_px[0] as u32).min(iw.saturating_sub(1));
    let y1 = (bbox_px[1] as u32).min(ih.saturating_sub(1));
    let x2 = (bbox_px[2] as u32).min(iw.saturating_sub(1));
    let y2 = (bbox_px[3] as u32).min(ih.saturating_sub(1));

    // Draw horizontal lines (top and bottom edges)
    for t in 0..thickness {
        let yt = y1.saturating_add(t).min(y2);
        let yb = y2.saturating_sub(t).max(y1);
        for x in x1..=x2 {
            img.put_pixel(x, yt, color);
            img.put_pixel(x, yb, color);
        }
    }
    // Draw vertical lines (left and right edges)
    for t in 0..thickness {
        let xl = x1.saturating_add(t).min(x2);
        let xr = x2.saturating_sub(t).max(x1);
        for y in y1..=y2 {
            img.put_pixel(xl, y, color);
            img.put_pixel(xr, y, color);
        }
    }
}

/// Draw a colored bounding box on an RGBA image with a specific color.
fn draw_box_rgba(
    img: &mut image::RgbaImage,
    bbox_px: [f32; 4],
    color: image::Rgba<u8>,
    thickness: u32,
) {
    let (iw, ih) = (img.width(), img.height());
    let x1 = (bbox_px[0] as u32).min(iw.saturating_sub(1));
    let y1 = (bbox_px[1] as u32).min(ih.saturating_sub(1));
    let x2 = (bbox_px[2] as u32).min(iw.saturating_sub(1));
    let y2 = (bbox_px[3] as u32).min(ih.saturating_sub(1));

    for t in 0..thickness {
        let yt = y1.saturating_add(t).min(y2);
        let yb = y2.saturating_sub(t).max(y1);
        for x in x1..=x2 {
            img.put_pixel(x, yt, color);
            img.put_pixel(x, yb, color);
        }
    }
    for t in 0..thickness {
        let xl = x1.saturating_add(t).min(x2);
        let xr = x2.saturating_sub(t).max(x1);
        for y in y1..=y2 {
            img.put_pixel(xl, y, color);
            img.put_pixel(xr, y, color);
        }
    }
}

/// Write table crop images with cell bbox overlays to `layout/tables/`.
///
/// For each table with a `TablePrediction`, draws cell bounding boxes on the
/// cropped table image and saves it as a PNG.
pub fn write_table_debug(
    output_dir: &Path,
    table_entries: &[(usize, DynamicImage)],
    table_results: &std::collections::HashMap<usize, crate::pipeline::TableResult>,
    page_idx: u32,
) -> Result<(), ExtractError> {
    use crate::pipeline::TableResult;

    let tables_dir = output_dir.join("layout").join("tables");

    let mut any_written = false;
    for (idx, crop_image) in table_entries {
        let pred = match table_results.get(idx) {
            Some(TableResult::Prediction(p)) => p,
            _ => continue,
        };

        if pred.cell_bboxes.is_empty() {
            continue;
        }

        if !any_written {
            std::fs::create_dir_all(&tables_dir)?;
            any_written = true;
        }

        let mut img = crop_image.to_rgba8();
        let (w, h) = (img.width() as f32, img.height() as f32);

        // Alternate colors for adjacent cells
        let colors = [
            image::Rgba([0u8, 200, 0, 180]),   // green
            image::Rgba([50, 100, 255, 180]),   // blue
            image::Rgba([255, 140, 0, 180]),    // orange
            image::Rgba([200, 0, 200, 180]),    // magenta
        ];

        for (cell_idx, bbox_opt) in pred.cell_bboxes.iter().enumerate() {
            let Some(bbox) = bbox_opt else { continue };
            let px = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h];
            let color = colors[cell_idx % colors.len()];
            draw_box_rgba(&mut img, px, color, 2);
        }

        let path = tables_dir.join(format!("p{}_{}.png", page_idx + 1, idx));
        img.save(&path)
            .map_err(|e| ExtractError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    }

    Ok(())
}

/// Map a RegionKind to a debug visualization color (pdfium).
fn region_color(kind: RegionKind) -> PdfColor {
    match kind {
        RegionKind::Title | RegionKind::ParagraphTitle => PdfColor::new(255, 50, 50, 255),
        RegionKind::Text
        | RegionKind::VerticalText
        | RegionKind::Abstract
        | RegionKind::SidebarText => PdfColor::new(50, 100, 255, 255),
        RegionKind::References | RegionKind::Footnote | RegionKind::TOC => {
            PdfColor::new(0, 180, 180, 255)
        }
        RegionKind::Table => PdfColor::new(0, 200, 0, 255),
        RegionKind::DisplayFormula | RegionKind::FormulaNumber => PdfColor::new(200, 0, 200, 255),
        RegionKind::InlineFormula => PdfColor::new(255, 100, 255, 255),
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => PdfColor::new(255, 140, 0, 255),
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => PdfColor::new(220, 200, 0, 255),
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber => {
            PdfColor::new(150, 150, 150, 255)
        }
        RegionKind::Algorithm => PdfColor::new(0, 160, 120, 255),
        RegionKind::FormattedText => PdfColor::new(120, 160, 0, 255),
        RegionKind::FigureGroup => PdfColor::new(0, 200, 200, 255),
    }
}

/// Write debug visualization: annotate the original PDF with colored bounding boxes
/// and labels, then save as PNGs and/or a debug PDF under `layout/`.
pub fn write_debug(
    pdfium: &Pdfium,
    pdf_path: &Path,
    pages: &[Page],
    output_dir: &Path,
    mode: DebugMode,
) -> Result<(), ExtractError> {
    let mut doc = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| ExtractError::Pdf(format!("Failed to re-open PDF for debug: {e}")))?;

    let font = doc.fonts_mut().courier_bold();

    for page_info in pages {
        let page_idx = page_info.page - 1; // 0-indexed
        let mut page = doc.pages().get(page_idx as u16).map_err(|e| {
            ExtractError::Pdf(format!("Failed to get page {page_idx} for debug: {e}"))
        })?;

        let page_h = page_info.height_pt;

        // Get CropBox offset to convert CropBox-relative coords back to MediaBox
        // for drawing on the PDF page.
        let crop_rect = page.boundaries().crop().ok().map(|b| b.bounds);
        let (crop_x, crop_y_bottom) = if let Some(rect) = crop_rect {
            (rect.left().value, rect.bottom().value)
        } else {
            (0.0, 0.0)
        };

        for region in &page_info.regions {
            let color = region_color(region.kind);
            let [x1, y1, x2, y2] = region.bbox;

            // bbox is CropBox-relative, Y-down; convert to MediaBox, Y-up (PDF coords)
            let pdf_left = x1 + crop_x;
            let pdf_right = x2 + crop_x;
            let pdf_bottom = (page_h - y2) + crop_y_bottom;
            let pdf_top = (page_h - y1) + crop_y_bottom;

            // Draw bounding box rectangle (stroke only, no fill)
            page.objects_mut()
                .create_path_object_rect(
                    PdfRect::new_from_values(pdf_bottom, pdf_left, pdf_top, pdf_right),
                    Some(color),
                    Some(PdfPoints::new(1.0)),
                    None,
                )
                .map_err(|e| ExtractError::Pdf(format!("Failed to draw debug rect: {e}")))?;

            // Label text
            let label = format!(
                "{:?} #{} {}%",
                region.kind,
                region.order,
                (region.confidence * 100.0) as u32
            );

            let font_size = 7.0;
            // Place label just above the box top edge
            let media_h = page_h + crop_y_bottom * 2.0; // approximate
            let label_y = if pdf_top + font_size + 1.0 < media_h {
                pdf_top + 1.0
            } else {
                pdf_top - font_size - 1.0
            };

            let mut text_obj = PdfPageTextObject::new(
                &doc,
                &label,
                font,
                PdfPoints::new(font_size),
            )
            .map_err(|e| ExtractError::Pdf(format!("Failed to create debug text: {e}")))?;

            text_obj
                .translate(PdfPoints::new(pdf_left), PdfPoints::new(label_y))
                .map_err(|e| ExtractError::Pdf(format!("Failed to position debug text: {e}")))?;
            text_obj
                .set_fill_color(color)
                .map_err(|e| ExtractError::Pdf(format!("Failed to color debug text: {e}")))?;

            page.objects_mut()
                .add_text_object(text_obj)
                .map_err(|e| ExtractError::Pdf(format!("Failed to add debug text: {e}")))?;
        }
    }

    if mode == DebugMode::Images {
        // Render annotated pages to PNGs
        let layout_dir = output_dir.join("layout");
        std::fs::create_dir_all(&layout_dir)?;

        for page_info in pages {
            let page_idx = page_info.page - 1;
            let page = doc.pages().get(page_idx as u16).map_err(|e| {
                ExtractError::Pdf(format!("Failed to get page {page_idx} for render: {e}"))
            })?;
            let img = pdf::render_page(&page, page_info.dpi)?;
            let out_path = layout_dir.join(format!("p{}.png", page_info.page));
            img.save(&out_path)?;
        }
    } else {
        // Remove un-processed pages so the debug PDF only contains annotated pages.
        let keep: std::collections::HashSet<u16> =
            pages.iter().map(|p| (p.page - 1) as u16).collect();
        let total = doc.pages().len();

        for i in (0..total).rev() {
            if !keep.contains(&i) {
                doc.pages()
                    .get(i)
                    .map_err(|e| ExtractError::Pdf(format!("Failed to get page {i} for removal: {e}")))?
                    .delete()
                    .map_err(|e| ExtractError::Pdf(format!("Failed to delete page {i}: {e}")))?;
            }
        }

        let layout_dir = output_dir.join("layout");
        std::fs::create_dir_all(&layout_dir)?;
        let debug_pdf_path = layout_dir.join("layout.pdf");
        doc.save_to_file(&debug_pdf_path)
            .map_err(|e| ExtractError::Pdf(format!("Failed to save debug PDF: {e}")))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExtractionResult, Metadata, Page, Region, RegionKind};

    /// Convenience wrapper: reflow + render, replacing the old `render_markdown`.
    fn render_markdown(result: &ExtractionResult) -> String {
        let doc = reflow(result, &std::collections::HashSet::new());
        render_markdown_from_reflow(&doc)
    }

    fn make_region(kind: RegionKind) -> Region {
        Region {
            id: String::new(),
            kind,
            bbox: [0.0; 4],
            confidence: 1.0,
            order: 0,
            text: None,
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
        }
    }

    #[test]
    fn test_title_to_markdown() {
        let mut r = make_region(RegionKind::Title);
        r.text = Some("My Title".into());
        assert_eq!(region_to_markdown(&r), "# My Title");
    }

    #[test]
    fn test_paragraph_title_to_markdown() {
        let mut r = make_region(RegionKind::ParagraphTitle);
        r.text = Some("Section 1".into());
        assert_eq!(region_to_markdown(&r), "## Section 1");
    }

    #[test]
    fn test_text_to_markdown() {
        let mut r = make_region(RegionKind::Text);
        r.text = Some("Hello world.".into());
        assert_eq!(region_to_markdown(&r), "Hello world.");
    }

    #[test]
    fn test_table_to_markdown() {
        let mut r = make_region(RegionKind::Table);
        r.html = Some("<table><tr><td>A</td></tr></table>".into());
        assert_eq!(
            region_to_markdown(&r),
            "<table><tr><td>A</td></tr></table>"
        );
    }

    #[test]
    fn test_formula_to_markdown() {
        let mut r = make_region(RegionKind::DisplayFormula);
        r.latex = Some("E = mc^2".into());
        assert_eq!(region_to_markdown(&r), "$$E = mc^2$$");
    }

    #[test]
    fn test_formula_strips_double_dollar() {
        let mut r = make_region(RegionKind::DisplayFormula);
        r.latex = Some("$$\nx^2 + y^2\n$$".into());
        assert_eq!(region_to_markdown(&r), "$$x^2 + y^2$$");
    }

    #[test]
    fn test_formula_strips_single_dollar() {
        let mut r = make_region(RegionKind::InlineFormula);
        r.latex = Some("$x$".into());
        assert_eq!(region_to_markdown(&r), "$x$");
    }

    #[test]
    fn test_image_to_markdown() {
        let mut r = make_region(RegionKind::Image);
        r.image_path = Some("images/p1_7.png".into());
        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Figure 1".into());
        r.caption = Some(Box::new(cap));
        assert_eq!(
            region_to_markdown(&r),
            "![](images/p1_7.png)\n\nFigure 1"
        );
    }

    #[test]
    fn test_image_no_caption_to_markdown() {
        let mut r = make_region(RegionKind::Image);
        r.image_path = Some("images/p1_7.png".into());
        assert_eq!(region_to_markdown(&r), "![](images/p1_7.png)");
    }

    #[test]
    fn test_algorithm_to_markdown() {
        let mut r = make_region(RegionKind::Algorithm);
        r.text = Some("for i in range(n):".into());
        assert_eq!(region_to_markdown(&r), "for i in range(n):");
    }

    #[test]
    fn test_bold_caption_label() {
        assert_eq!(
            bold_caption_label("Fig. 1. Example simulation results"),
            "**Fig. 1.** Example simulation results"
        );
        assert_eq!(
            bold_caption_label("Table 1. Performance results"),
            "**Table 1.** Performance results"
        );
        assert_eq!(
            bold_caption_label("Figure 12. Some caption"),
            "**Figure 12.** Some caption"
        );
        assert_eq!(
            bold_caption_label("Algorithm 1. Pseudocode"),
            "**Algorithm 1.** Pseudocode"
        );
        // No recognized prefix — pass through unchanged
        assert_eq!(
            bold_caption_label("(a) Subfigure label"),
            "(a) Subfigure label"
        );
    }

    #[test]
    fn test_image_with_fig_caption() {
        let mut r = make_region(RegionKind::Image);
        r.image_path = Some("images/p1_5.png".into());
        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Fig. 1. Example simulation results".into());
        r.caption = Some(Box::new(cap));
        assert_eq!(
            region_to_markdown(&r),
            "![](images/p1_5.png)\n\n**Fig. 1.** Example simulation results"
        );
    }

    #[test]
    fn test_skip_page_header_footer() {
        let r1 = make_region(RegionKind::PageHeader);
        let r2 = make_region(RegionKind::PageFooter);
        assert!(region_to_markdown(&r1).is_empty());
        assert!(region_to_markdown(&r2).is_empty());
    }

    #[test]
    fn test_multi_page_assembly() {
        let result = ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 2,
                extraction_time_ms: 0,
            },
            pages: vec![
                Page {
                    page: 1,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::Title);
                        r.text = Some("Page 1 Title".into());
                        r.order = 0;
                        r
                    }],
                },
                Page {
                    page: 2,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::Text);
                        r.text = Some("Page 2 text.".into());
                        r.order = 0;
                        r
                    }],
                },
            ],
        };

        let md = render_markdown(&result);
        assert!(md.contains("# Page 1 Title"));
        assert!(md.contains("Page 2 text."));
    }

    #[test]
    fn test_figure_group_to_markdown() {
        let mut group = make_region(RegionKind::FigureGroup);
        group.image_path = Some("images/p1_0_grp.png".into());

        let item1 = make_region(RegionKind::Image);
        let item2 = make_region(RegionKind::Image);
        group.items = Some(vec![item1, item2]);

        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Fig. 5. Two panels".into());
        group.caption = Some(Box::new(cap));

        let md = region_to_markdown(&group);
        // Single composite image, not individual member images
        assert!(md.contains("![](images/p1_0_grp.png)"));
        assert!(md.contains("**Fig. 5.** Two panels"));
    }

    #[test]
    fn test_figure_group_no_caption() {
        let mut group = make_region(RegionKind::FigureGroup);
        group.image_path = Some("images/p1_0_grp.png".into());

        let item1 = make_region(RegionKind::Image);
        let item2 = make_region(RegionKind::Image);
        group.items = Some(vec![item1, item2]);

        let md = region_to_markdown(&group);
        // Single composite image
        assert!(md.contains("![](images/p1_0_grp.png)"));
        // No caption text
        assert!(!md.contains("Fig"));
    }

    #[test]
    fn test_figure_group_empty_items() {
        let mut group = make_region(RegionKind::FigureGroup);
        group.items = Some(vec![]);
        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Fig. 1. Empty group".into());
        group.caption = Some(Box::new(cap));

        let md = region_to_markdown(&group);
        assert!(md.contains("**Fig. 1.** Empty group"));
    }

    #[test]
    fn test_figure_group_consumed_skipped_in_markdown() {
        // FigureGroup renders a single composite image; consumed originals are skipped
        let result = ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 1,
                extraction_time_ms: 0,
            },
            pages: vec![Page {
                page: 1,
                width_pt: 612.0,
                height_pt: 792.0,
                dpi: 144,
                regions: vec![
                    {
                        // Consumed original member (no image_path in new design)
                        let mut r = make_region(RegionKind::Image);
                        r.consumed = true;
                        r.order = 0;
                        r
                    },
                    {
                        // The group with composite image
                        let mut group = make_region(RegionKind::FigureGroup);
                        group.order = 0;
                        group.image_path = Some("images/p1_0.png".into());
                        let item = make_region(RegionKind::Image);
                        group.items = Some(vec![item]);
                        group
                    },
                ],
            }],
        };

        let md = render_markdown(&result);
        // The composite image should appear exactly once
        let count = md.matches("![](images/p1_0.png)").count();
        assert_eq!(count, 1);
    }

    /// Helper to build an ExtractionResult from a list of regions on one page.
    fn make_result(regions: Vec<Region>) -> ExtractionResult {
        ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 1,
                extraction_time_ms: 0,
            },
            pages: vec![Page {
                page: 1,
                width_pt: 612.0,
                height_pt: 792.0,
                dpi: 144,
                regions,
            }],
        }
    }

    #[test]
    fn test_mid_sentence_merge_adjacent_text_sections() {
        // Two adjacent text sections split at a column break
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("method presented superior".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("convergence behavior over prior techniques.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("presented superior convergence behavior"),
            "adjacent text sections should merge mid-sentence, got: {md}"
        );
    }

    #[test]
    fn test_mid_sentence_merge_across_figure() {
        // Text → Figure → Text where the text is split mid-sentence by a figure
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("coloring the nodes of the".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p1_1.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("dual graph, which has more nodes.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(md.contains("![](images/p1_1.png)"), "figure should be present");
        // The trailing fragment moves to the next text section
        assert!(
            md.contains("coloring the nodes of the dual graph"),
            "text should rejoin across figure, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_at_sentence_boundary() {
        // Two text sections where the first ends with a period — no merge
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("This is a complete sentence.".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("This starts a new paragraph.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("sentence.\n\nThis starts"),
            "should have paragraph break, got: {md}"
        );
    }

    #[test]
    fn test_merge_when_prev_ends_mid_sentence_next_starts_uppercase() {
        // Mid-sentence ending and next starts uppercase — still merges
        // because the previous text clearly ended mid-sentence.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("some text ending with comma,".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("Another part of the same paragraph.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("comma, Another part"),
            "should merge when prev ends mid-sentence, got: {md}"
        );
    }

    #[test]
    fn test_references_merge_mid_sentence() {
        // References split across pages where entry is cut mid-sentence
        let result = ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 2,
                extraction_time_ms: 0,
            },
            pages: vec![
                Page {
                    page: 1,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("ACM Transactions on".into());
                        r
                    }],
                },
                Page {
                    page: 2,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("Graphics 31, 5 (Aug. 2012).".into());
                        r
                    }],
                },
            ],
        };
        let md = render_markdown(&result);
        assert!(
            md.contains("Transactions on Graphics"),
            "should join with space across pages, got: {md}"
        );
    }

    #[test]
    fn test_references_merge_at_entry_boundary() {
        // References across pages where first page ends at entry boundary
        let result = ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 2,
                extraction_time_ms: 0,
            },
            pages: vec![
                Page {
                    page: 1,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("First reference entry.".into());
                        r
                    }],
                },
                Page {
                    page: 2,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("Second reference entry.".into());
                        r
                    }],
                },
            ],
        };
        let md = render_markdown(&result);
        assert!(
            md.contains("entry.\n\nSecond"),
            "should have paragraph break between complete entries, got: {md}"
        );
    }

    #[test]
    fn test_mid_sentence_merge_across_algorithm() {
        // Text → Algorithm → Text: text split mid-sentence by an algorithm box.
        // The trailing sentence fragment moves to the next text section.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("As expected, VBD".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Algorithm);
                r.text = Some("Algorithm 1: VBD simulation".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("converges slower for stiffer materials.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // The text should be rejoined in the continuation section
        assert!(
            md.contains("As expected, VBD converges slower"),
            "sentence should rejoin across algorithm, got: {md}"
        );
        // Algorithm still appears as its own section
        assert!(
            md.contains("Algorithm 1"),
            "algorithm should be present, got: {md}"
        );
    }

    #[test]
    fn test_mid_sentence_partial_move_across_figure() {
        // Only the incomplete trailing sentence moves, not the whole section.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("First sentence ends here. Second part of the".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p1_1.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("paragraph continues.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // "First sentence ends here." stays in the first section
        assert!(
            md.contains("ends here."),
            "complete sentence should remain, got: {md}"
        );
        // "Second part of the" moves to join "paragraph continues."
        assert!(
            md.contains("Second part of the paragraph continues."),
            "incomplete sentence should rejoin, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_across_display_formula() {
        // Text → DisplayFormula → Text: formulas are part of the paragraph
        // flow, so text should NOT be rearranged across them.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("VBD avoids this by minimizing the variational energy [Simo et al.".into());
                r
            },
            {
                let mut r = make_region(RegionKind::DisplayFormula);
                r.latex = Some("x = argmin E(x)".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("1992] where the timestep size is used.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // Text should NOT be rearranged — formula is inline to the paragraph
        assert!(
            md.contains("[Simo et al.\n\n$$x = argmin E(x)$$\n\n1992]"),
            "text should stay in place around formula, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_authors_into_abstract() {
        // Title → author Text blocks → Image → Abstract: the author lines
        // end without terminal punctuation ("USA"), but should NOT merge
        // with the abstract that follows.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Title);
                r.text = Some("Augmented Vertex Block Descent".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CHRIS GILES, Roblox, USA".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CEM YUKSEL, University of Utah, USA".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p1_4.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Abstract);
                r.text = Some("Vertex Block Descent is a fast physics-based simulation method.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // Authors should remain separate from the abstract
        assert!(
            md.contains("University of Utah, USA\n\n"),
            "authors should not merge into abstract, got: {md}"
        );
        assert!(
            md.contains("Vertex Block Descent is a fast"),
            "abstract should be present, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_across_title_boundary() {
        // Text → Title → Text: a Title between text blocks should prevent
        // backward merge even when the first text ends mid-sentence.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("some text ending mid".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Title);
                r.text = Some("Section 2".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("sentence that continues here.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // Should NOT merge across the title — texts must remain separate
        assert!(
            md.contains("some text ending mid\n\nsentence that continues here."),
            "text blocks should not merge across title boundary, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_text_into_abstract_even_when_text_kind() {
        // When the abstract is detected as RegionKind::Abstract,
        // it should never absorb the previous text block.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("Author Name, Institution".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Abstract);
                r.text = Some("We present a novel approach to solving.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("Author Name, Institution\n\nWe present"),
            "abstract should not merge with preceding text, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_short_line_author_blocks() {
        // Three narrow author lines (< 50% of 612pt page) should NOT be merged
        // even though they end without terminal punctuation.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Title);
                r.text = Some("Augmented Vertex Block Descent".into());
                r.bbox = [49.0, 76.0, 289.0, 95.0]; // ~240pt wide
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CHRIS GILES, Roblox, USA".into());
                r.bbox = [49.0, 104.0, 178.0, 117.0]; // ~129pt wide (21% of 612)
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("ELIE DIAZ, University of Utah, USA".into());
                r.bbox = [49.0, 118.0, 211.0, 132.0]; // ~162pt wide (26% of 612)
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CEM YUKSEL, University of Utah, USA".into());
                r.bbox = [49.0, 132.0, 228.0, 146.0]; // ~179pt wide (29% of 612)
                r
            },
        ]);
        let md = render_markdown(&result);
        // Each author should be a separate paragraph, not merged together
        assert!(
            md.contains("CHRIS GILES, Roblox, USA\n\nELIE DIAZ"),
            "author lines should stay separate, got: {md}"
        );
        assert!(
            md.contains("ELIE DIAZ, University of Utah, USA\n\nCEM YUKSEL"),
            "author lines should stay separate, got: {md}"
        );
    }

    #[test]
    fn test_wide_text_blocks_still_merge() {
        // Two wide text blocks (> 50% of page) that end mid-sentence should
        // still merge — these are paragraph splits, not metadata lines.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("method presented superior".into());
                r.bbox = [49.0, 300.0, 560.0, 320.0]; // ~511pt wide (83% of 612)
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("convergence behavior over prior techniques.".into());
                r.bbox = [49.0, 400.0, 560.0, 420.0]; // ~511pt wide
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("presented superior convergence behavior"),
            "wide text blocks should still merge mid-sentence, got: {md}"
        );
    }

    #[test]
    fn test_merge_across_subfigure_labels() {
        // Paragraph split across multiple figures with short sub-figure labels.
        // The backward search should skip over the short labels to find the
        // real paragraph text and merge the two halves.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("Figure 6 shows a".into());
                r.bbox = [49.0, 300.0, 300.0, 400.0]; // column-width paragraph
                r
            },
            {
                // Sub-figure label: short + narrow → is_short_line
                let mut r = make_region(RegionKind::Text);
                r.text = Some("(a) Reference".into());
                r.bbox = [49.0, 410.0, 140.0, 425.0]; // narrow, 13 chars
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p8_5.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("(b) AVBD, 5 iter.".into());
                r.bbox = [49.0, 500.0, 160.0, 515.0]; // narrow, 17 chars
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p8_9.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("delicate card tower with very lightweight bodies.".into());
                r.bbox = [49.0, 600.0, 300.0, 700.0]; // column-width paragraph
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("shows a delicate card tower"),
            "paragraph should merge across sub-figure labels, got: {md}"
        );
    }

    // ── Page offset tests ──

    #[test]
    fn test_toc_page_to_pdf_page_body() {
        // Body page 1 with offset 20 → PDF page 20
        assert_eq!(toc_page_to_pdf_page(1, 20, 0, 100), Some(20));
        // Body page 5 with offset 20 → PDF page 24
        assert_eq!(toc_page_to_pdf_page(5, 20, 0, 100), Some(24));
    }

    #[test]
    fn test_toc_page_to_pdf_page_front_matter() {
        // Front matter page i (page_value = -1) with fm_offset 0 → PDF page 0
        assert_eq!(toc_page_to_pdf_page(-1, 20, 0, 100), Some(0));
        // Front matter page xxi (page_value = -21) with fm_offset 0 → PDF page 20
        assert_eq!(toc_page_to_pdf_page(-21, 20, 0, 100), Some(20));
        // Front matter with fm_offset 2 (2 unnumbered pages before page i)
        assert_eq!(toc_page_to_pdf_page(-1, 20, 2, 100), Some(2));
        assert_eq!(toc_page_to_pdf_page(-3, 20, 2, 100), Some(4));
    }

    #[test]
    fn test_toc_page_to_pdf_page_out_of_range() {
        // Page beyond document → None
        assert_eq!(toc_page_to_pdf_page(200, 0, 0, 100), None);
        // Negative result → None
        assert_eq!(toc_page_to_pdf_page(-50, 0, 0, 10), None);
    }

    // ── Footnote marker parsing tests ──

    #[test]
    fn test_parse_footnote_marker_numeric() {
        let (m, b) = parse_footnote_marker("1 Applications are only discussed");
        assert_eq!(m, "1");
        assert!(b.starts_with("Applications"));
    }

    #[test]
    fn test_parse_footnote_marker_multi_digit() {
        let (m, b) = parse_footnote_marker("23 This is footnote 23.");
        assert_eq!(m, "23");
        assert!(b.starts_with("This"));
    }

    #[test]
    fn test_parse_footnote_marker_symbol() {
        let (m, b) = parse_footnote_marker("† See also chapter 5");
        assert_eq!(m, "†");
        assert!(b.starts_with("See"));
    }

    #[test]
    fn test_parse_footnote_marker_no_marker() {
        let (m, b) = parse_footnote_marker("Just regular text");
        assert_eq!(m, "");
        assert_eq!(b, "Just regular text");
    }

    #[test]
    fn test_parse_footnote_marker_stuck_to_word() {
        // Digit immediately followed by uppercase letter (no space)
        let (m, b) = parse_footnote_marker("3Different volumes can be used");
        assert_eq!(m, "3");
        assert!(b.starts_with("Different"));
    }

    #[test]
    fn test_parse_footnote_marker_multi_digit_stuck() {
        let (m, b) = parse_footnote_marker("10Centroid sampling can cause");
        assert_eq!(m, "10");
        assert!(b.starts_with("Centroid"));
    }

    #[test]
    fn test_parse_footnote_marker_latex_superscript_spaced() {
        // LaTeX superscript with spaces: $^ {2} \mathrm{A}$
        let (m, b) = parse_footnote_marker("$^ {2}$ rest of text");
        assert_eq!(m, "2");
        assert!(b.starts_with("rest"));
    }

    #[test]
    fn test_parse_footnote_marker_lowercase_no_match() {
        // Digit followed by lowercase should NOT match (could be "2nd", "3rd" etc.)
        let (m, b) = parse_footnote_marker("2nd edition of the book");
        assert_eq!(m, "");
        assert_eq!(b, "2nd edition of the book");
    }

    #[test]
    fn test_parse_footnote_marker_latex_spaced_digits() {
        // LaTeX with space between digits: $^ {1 4}$
        let (m, b) = parse_footnote_marker("$^ {1 4}$ some text");
        assert_eq!(m, "14");
        assert!(b.starts_with("some"));
    }

    #[test]
    fn test_parse_footnote_marker_latex_bare_digit() {
        // Bare superscript digit without braces: $^ 8 \mathrm{An}$
        let (m, b) = parse_footnote_marker("$^ 8 \\mathrm{An}$ explanation");
        assert_eq!(m, "8");
        assert!(b.starts_with("explanation"));
    }

    // ── Split bold label fix tests ──

    #[test]
    fn test_fix_split_bold_figure() {
        let input = "**Figure 3.**22 illustrates the execution";
        let result = fix_split_bold_labels(input);
        assert!(result.starts_with("**Figure 3.22.**"));
    }

    #[test]
    fn test_fix_split_bold_table() {
        let input = "**Table 4.**1. Summary of transforms";
        let result = fix_split_bold_labels(input);
        assert!(result.starts_with("**Table 4.1.**"));
    }

    #[test]
    fn test_fix_split_bold_no_match() {
        // Don't modify normal bold text
        let input = "**Figure 3.22.** normal caption";
        let result = fix_split_bold_labels(input);
        assert_eq!(result, input);
    }

    // ── Collapsed doubled chars tests ──

    #[test]
    fn test_collapse_doubled_chars() {
        assert_eq!(
            collapse_doubled_chars("EEXXPPEERRIIMMEENNTT"),
            "EXPERIMENT"
        );
    }

    #[test]
    fn test_collapse_doubled_chars_normal_text() {
        // Normal text should not be collapsed
        let normal = "This is normal text with no doubling";
        assert_eq!(collapse_doubled_chars(normal), normal);
    }

    // ── Title echo dedup tests ──

    #[test]
    fn test_normalize_title() {
        assert_eq!(normalize_title("1.2 Some Title"), "1.2 some title");
        assert_eq!(normalize_title("  HELLO  WORLD  "), "hello world");
    }

    // ── Code detection tests ──

    #[test]
    fn test_looks_like_code_c() {
        assert!(looks_like_code("#include <stdio.h>\nint main() { return 0; }"));
    }

    #[test]
    fn test_looks_like_code_cpp() {
        assert!(looks_like_code("class Foo { public: void bar(); private: int x; };"));
    }

    #[test]
    fn test_looks_like_code_not_prose() {
        assert!(!looks_like_code("This is a normal paragraph about programming concepts."));
    }

    #[test]
    fn test_looks_like_code_windows_api() {
        assert!(looks_like_code("HANDLE hFile = CreateFile(TEXT(\"test.txt\"), GENERIC_READ, 0, NULL, OPEN_EXISTING, 0, NULL);"));
    }

    #[test]
    fn test_guess_language_c() {
        assert_eq!(guess_language("#include <stdio.h>\nint main() {}"), Some("c".to_string()));
    }

    #[test]
    fn test_guess_language_rust() {
        assert_eq!(guess_language("pub fn main() -> Result<(), Error> { let mut x = 5; }"), Some("rust".to_string()));
    }

    #[test]
    fn test_guess_language_glsl() {
        assert_eq!(guess_language("uniform vec3 lightPos;\nvoid main() { gl_Position = mvp * pos; }"), Some("glsl".to_string()));
    }

    #[test]
    fn test_guess_language_none() {
        assert_eq!(guess_language("Hello world, this is just text"), None);
    }

    // ── Code block merging tests ──

    #[test]
    fn test_merge_consecutive_code_blocks() {
        let mut nodes = vec![
            ReflowNode::CodeBlock { content: "int x = 1;".to_string(), language: Some("c".to_string()) },
            ReflowNode::CodeBlock { content: "int y = 2;".to_string(), language: None },
        ];
        detect_code_blocks(&mut nodes);
        assert_eq!(nodes.len(), 1);
        if let ReflowNode::CodeBlock { content, language } = &nodes[0] {
            assert!(content.contains("int x = 1;"));
            assert!(content.contains("int y = 2;"));
            assert_eq!(language, &Some("c".to_string()));
        } else {
            panic!("expected CodeBlock");
        }
    }

    // ── List detection tests ──

    #[test]
    fn test_detect_numbered_list() {
        let mut nodes = vec![
            ReflowNode::Text { content: "1. First item".to_string(), footnotes: Vec::new() },
            ReflowNode::Text { content: "2. Second item".to_string(), footnotes: Vec::new() },
            ReflowNode::Text { content: "3. Third item".to_string(), footnotes: Vec::new() },
        ];
        detect_lists(&mut nodes);
        assert_eq!(nodes.len(), 1);
        if let ReflowNode::List { list_type, items } = &nodes[0] {
            assert_eq!(list_type, "numbered");
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], "First item");
        } else {
            panic!("expected List");
        }
    }

    #[test]
    fn test_detect_bullet_list() {
        // Use dash-based bullets which are simpler to test
        let mut nodes = vec![
            ReflowNode::Text { content: "\u{2013} First item".to_string(), footnotes: Vec::new() },
            ReflowNode::Text { content: "\u{2013} Second item".to_string(), footnotes: Vec::new() },
            ReflowNode::Text { content: "\u{2013} Third item".to_string(), footnotes: Vec::new() },
        ];
        assert_eq!(bullet_prefix_len("\u{2013} test"), 1, "en-dash should be detected as bullet (1 char)");
        detect_lists(&mut nodes);
        assert_eq!(nodes.len(), 1, "got {} nodes", nodes.len());
        if let ReflowNode::List { list_type, items } = &nodes[0] {
            assert_eq!(list_type, "bulleted");
            assert_eq!(items.len(), 3);
        } else {
            panic!("expected List, got {:?}", nodes[0]);
        }
    }

    #[test]
    fn test_no_false_positive_list() {
        // Regular paragraphs starting with the same letter should NOT be detected
        let mut nodes = vec![
            ReflowNode::Text { content: "The first paragraph.".to_string(), footnotes: Vec::new() },
            ReflowNode::Text { content: "The second paragraph.".to_string(), footnotes: Vec::new() },
            ReflowNode::Text { content: "The third paragraph.".to_string(), footnotes: Vec::new() },
        ];
        detect_lists(&mut nodes);
        // Should remain as 3 separate Text nodes
        assert_eq!(nodes.len(), 3);
    }

    // ── Code-text dedup tests ──

    #[test]
    fn test_dedup_code_text_removes_duplicate_text() {
        let mut nodes = vec![
            ReflowNode::CodeBlock { content: "int x = 1;".to_string(), language: Some("c".to_string()) },
            ReflowNode::Text { content: "int x = 1;".to_string(), footnotes: Vec::new() },
        ];
        dedup_code_text(&mut nodes);
        assert_eq!(nodes.len(), 1, "duplicate text should be removed");
        assert!(matches!(&nodes[0], ReflowNode::CodeBlock { .. }));
    }

    #[test]
    fn test_dedup_code_text_strips_code_from_prose() {
        let mut nodes = vec![
            ReflowNode::CodeBlock { content: "int x = 1; int y = 2;".to_string(), language: None },
            ReflowNode::Text {
                content: "int x = 1; int y = 2; which is certainly shorter.".to_string(),
                footnotes: Vec::new(),
            },
        ];
        dedup_code_text(&mut nodes);
        assert_eq!(nodes.len(), 2, "both nodes should remain");
        if let ReflowNode::Text { content, .. } = &nodes[1] {
            assert!(content.contains("shorter"), "prose should remain, got: {content}");
        }
    }

    #[test]
    fn test_dedup_code_text_no_false_positive() {
        let mut nodes = vec![
            ReflowNode::CodeBlock { content: "int x = 1;".to_string(), language: None },
            ReflowNode::Text { content: "This is unrelated prose text.".to_string(), footnotes: Vec::new() },
        ];
        dedup_code_text(&mut nodes);
        assert_eq!(nodes.len(), 2, "unrelated text should not be removed");
    }

    // ── Italic collapse tests ──

    #[test]
    fn test_italic_collapse_double_star() {
        // Simulates "dif**ferent" from italic line wrap
        let chars = vec!['d', 'i', 'f', '*', '*', 'f', 'e', 'r'];
        let text: String = chars.iter().collect();
        // The collapse logic is in assemble_text, test the pattern
        assert!(text.contains("**"));
        // After collapse: "dif" + "fer" (both alpha around **)
        let mut collapsed = String::new();
        let cv: Vec<char> = text.chars().collect();
        let mut i = 0;
        while i < cv.len() {
            if cv[i] == '*' && i + 1 < cv.len() && cv[i + 1] == '*' {
                let before = i > 0 && cv[i - 1].is_alphanumeric();
                let after = i + 2 < cv.len() && cv[i + 2].is_alphanumeric();
                if before && after { i += 2; continue; }
            }
            collapsed.push(cv[i]);
            i += 1;
        }
        assert_eq!(collapsed, "differ");
    }

    /// Helper to build a heading node for tests.
    fn h(depth: u32, title: &str) -> ReflowNode {
        let (sec, txt) = parse_section_and_text(title);
        ReflowNode::Heading {
            depth,
            text: txt,
            section: sec,
            children: Vec::new(),
        }
    }

    /// Helper to build a text node for tests.
    fn t(content: &str) -> ReflowNode {
        ReflowNode::Text {
            content: content.to_string(),
            footnotes: Vec::new(),
        }
    }

    /// Format a node as a short label for test assertions.
    fn node_label(n: &ReflowNode) -> String {
        match n {
            ReflowNode::Heading { text, section, .. } => {
                let full = format_heading_title(section.as_deref(), text);
                format!("H:{full}")
            }
            ReflowNode::Text { content, .. } => {
                let s = content.trim();
                if s.len() > 40 {
                    format!("T:{}...", &s[..37])
                } else {
                    format!("T:{s}")
                }
            }
            ReflowNode::Figure { .. } => "Figure".to_string(),
            ReflowNode::Toc { .. } => "Toc".to_string(),
            _ => "Other".to_string(),
        }
    }

    fn titles(nodes: &[ReflowNode]) -> Vec<String> {
        nodes.iter().map(node_label).collect()
    }

    // ── parse_section_and_text tests ──────────────────────────────────

    #[test]
    fn test_parse_simple_numbered() {
        assert_eq!(parse_section_and_text("1.4 Conclusions"), (Some("1.4".into()), "Conclusions".into()));
        assert_eq!(parse_section_and_text("3. Methods"), (Some("3".into()), "Methods".into()));
        assert_eq!(parse_section_and_text("12.3.1 Detailed Results"), (Some("12.3.1".into()), "Detailed Results".into()));
    }

    #[test]
    fn test_parse_unnumbered() {
        assert_eq!(parse_section_and_text("Conclusions"), (None, "Conclusions".into()));
        assert_eq!(parse_section_and_text("The Fixed-Function Pipeline"), (None, "The Fixed-Function Pipeline".into()));
    }

    #[test]
    fn test_parse_appendix_style() {
        assert_eq!(parse_section_and_text("A.1 First Appendix"), (Some("A.1".into()), "First Appendix".into()));
        assert_eq!(parse_section_and_text("A.1.2 Detailed Appendix"), (Some("A.1.2".into()), "Detailed Appendix".into()));
        assert_eq!(parse_section_and_text("B.3 Another"), (Some("B.3".into()), "Another".into()));
        assert_eq!(parse_section_and_text("A Overview"), (Some("A".into()), "Overview".into()));
    }

    #[test]
    fn test_parse_chapter_prefix() {
        assert_eq!(parse_section_and_text("Chapter 3 Fundamental Concepts"), (Some("3".into()), "Fundamental Concepts".into()));
        assert_eq!(parse_section_and_text("CHAPTER 10 Advanced Topics"), (Some("10".into()), "Advanced Topics".into()));
    }

    #[test]
    fn test_parse_roman_not_section() {
        // Roman numerals like "IV" aren't valid section numbers (no dot pattern)
        // but single uppercase letters like "A" are (appendix style)
        let (sec, text) = parse_section_and_text("Foreword");
        assert_eq!(sec, None);
        assert_eq!(text, "Foreword");
    }

    #[test]
    fn test_format_heading_title_roundtrip() {
        // parse then format should reconstruct the original
        for input in &["1.4 Conclusions", "A.1.2 Detailed", "Conclusions", "Chapter 3 Foo"] {
            let (sec, text) = parse_section_and_text(input);
            let formatted = format_heading_title(sec.as_deref(), &text);
            // For "Chapter 3 Foo" → section="3", text="Foo" → formatted="3 Foo"
            // This is expected — the "Chapter" label is stripped
            if !input.starts_with("Chapter") {
                assert_eq!(&formatted, *input, "roundtrip failed for {input:?}");
            }
        }
    }

    // ── place_toc_headings tests ────────────────────────────────────

    /// Helper: run place_toc_headings and return (page, label) pairs.
    fn place(
        nodes: Vec<(u32, ReflowNode)>,
        headings: &[(u32, u32, &str, bool)],
    ) -> Vec<(u32, String)> {
        let mut tagged = nodes;
        let headings_owned: Vec<(u32, u32, String, bool)> = headings
            .iter()
            .map(|(p, d, t, toc)| (*p, *d, t.to_string(), *toc))
            .collect();
        let norms: Vec<String> = headings_owned
            .iter()
            .map(|(_, _, t, _)| normalize_title(t))
            .collect();
        place_toc_headings(&mut tagged, &headings_owned, &norms, &[]);
        tagged
            .iter()
            .map(|(pg, node)| (*pg, node_label(node)))
            .collect()
    }

    fn tp(page: u32, s: &str) -> (u32, ReflowNode) {
        (page, ReflowNode::Text { content: s.to_string(), footnotes: vec![] })
    }

    #[test]
    fn test_place_heading_replaces_echo_on_correct_page() {
        // "Conclusions" text on page 49 should be replaced by "1.4 Conclusions" heading
        let nodes = vec![
            tp(49, "glEnableClientState( GL_VERTEX_ARRAY ); glEnableClientState( GL_COLOR_ARRAY ); ...long code..."),
            tp(49, "which is certainly shorter and feels more elegant."),
            tp(49, "Conclusions"),
            tp(49, "The fixed-function graphics pipeline has shown itself to be very valuable..."),
        ];
        let result = place(nodes, &[(49, 2, "1.4 Conclusions", false)]);
        assert_eq!(result[2], (49, "H:1.4 Conclusions".into()));
        assert_eq!(result.len(), 4); // no extra nodes injected
    }

    #[test]
    fn test_place_heading_exact_match() {
        // Exact normalized match: "1.4 Conclusions" text matches "1.4 Conclusions" heading
        let nodes = vec![
            tp(10, "Some content before"),
            tp(10, "1.4 Conclusions"),
            tp(10, "Body text here"),
        ];
        let result = place(nodes, &[(10, 2, "1.4 Conclusions", false)]);
        assert_eq!(result[1], (10, "H:1.4 Conclusions".into()));
    }

    #[test]
    fn test_place_heading_stripped_match() {
        // Stripped match: TOC has "1.4 Conclusions", page text is just "Conclusions"
        let nodes = vec![
            tp(20, "Conclusions"),
            tp(20, "Body text"),
        ];
        let result = place(nodes, &[(20, 2, "1.4 Conclusions", false)]);
        assert_eq!(result[0], (20, "H:1.4 Conclusions".into()));
    }

    #[test]
    fn test_place_heading_section_number_match() {
        // Section number match: content has "1.4 Summary" but TOC has "1.4 Conclusions"
        // — both share "1.4" prefix
        let nodes = vec![
            tp(20, "1.4 Summary of Results"),
        ];
        let result = place(nodes, &[(20, 2, "1.4 Conclusions", false)]);
        assert_eq!(result[0], (20, "H:1.4 Conclusions".into()));
    }

    #[test]
    fn test_place_heading_no_match_injects_at_page_start() {
        // No matching text on the page — heading injected before first content
        let nodes = vec![
            tp(5, "Earlier page content"),
            tp(10, "First content on page 10"),
            tp(10, "More content"),
        ];
        let result = place(nodes, &[(10, 2, "1.4 Conclusions", false)]);
        assert_eq!(result[1], (10, "H:1.4 Conclusions".into()));
        assert_eq!(result[2], (10, "T:First content on page 10".into()));
    }

    #[test]
    fn test_place_heading_does_not_match_wrong_page() {
        // "Conclusions" exists but on page 48, heading targets page 49
        // Should NOT replace page 48 text; should inject at page 49 boundary
        let nodes = vec![
            tp(48, "Conclusions"),
            tp(49, "First content on page 49"),
        ];
        let result = place(nodes, &[(49, 2, "1.4 Conclusions", false)]);
        // Page 48 text unchanged
        assert_eq!(result[0], (48, "T:Conclusions".into()));
        // Heading injected before page 49 content
        assert_eq!(result[1], (49, "H:1.4 Conclusions".into()));
        assert_eq!(result[2], (49, "T:First content on page 49".into()));
    }

    #[test]
    fn test_place_multiple_headings_same_page() {
        // Two headings on same page — each replaces its own echo
        let nodes = vec![
            tp(10, "The Traditional View"),
            tp(10, "Some intro text about the traditional view that is long enough to skip."),
            tp(10, "The Vertex Operation"),
            tp(10, "Content about vertices"),
        ];
        let result = place(nodes, &[
            (10, 2, "1.1 The Traditional View", false),
            (10, 3, "1.1.1 The Vertex Operation", false),
        ]);
        assert_eq!(result[0], (10, "H:1.1 The Traditional View".into()));
        assert_eq!(result[2], (10, "H:1.1.1 The Vertex Operation".into()));
        assert_eq!(result.len(), 4); // no extras
    }

    #[test]
    fn test_place_heading_ignores_long_text() {
        // Text > 120 chars should never be matched as a heading echo
        let long_text = "Conclusions ".repeat(20); // well over 120 chars
        let nodes = vec![
            tp(10, &long_text),
            tp(10, "Conclusions"),
        ];
        let result = place(nodes, &[(10, 2, "1.4 Conclusions", false)]);
        // Should match the short one, not the long one
        assert_eq!(result[1], (10, "H:1.4 Conclusions".into()));
    }

    #[test]
    fn test_place_heading_preserves_content_before_echo() {
        // Content before the echo on the same page should NOT be displaced
        let nodes = vec![
            tp(49, "Code from previous section that runs over to this page"),
            tp(49, "More leftover content"),
            tp(49, "Conclusions"),
            tp(49, "The actual conclusions body text"),
        ];
        let result = place(nodes, &[(49, 2, "1.4 Conclusions", false)]);
        // Content before echo stays in place
        assert_eq!(result[0].1, "T:Code from previous section that runs ...");
        assert_eq!(result[1].1, "T:More leftover content");
        // Echo replaced
        assert_eq!(result[2], (49, "H:1.4 Conclusions".into()));
        // Content after echo stays
        assert_eq!(result[3].1, "T:The actual conclusions body text");
    }

    #[test]
    fn test_place_heading_fallback_empty_page() {
        // Heading targets a page with no content — inject at end
        let nodes = vec![
            tp(5, "Content on page 5"),
        ];
        let result = place(nodes, &[(10, 2, "1.4 Conclusions", false)]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[1], (10, "H:1.4 Conclusions".into()));
    }

    #[test]
    fn test_place_heading_chapter_match() {
        // "Chapter 3 Concepts" in TOC matches "Chapter 3 Concepts" on page
        // parse_section_and_text strips "Chapter" label → section="3", text="Concepts"
        let nodes = vec![
            tp(30, "Chapter 3 Concepts"),
            tp(30, "Body text"),
        ];
        let result = place(nodes, &[(30, 1, "Chapter 3 Concepts", false)]);
        assert_eq!(result[0], (30, "H:3 Concepts".into()));
    }

    // ── code detection tests ──────────────────────────────────────────

    #[test]
    fn test_count_italic_spans() {
        // Real italic: *word*
        assert_eq!(count_italic_spans("This is *italic* text"), 1);
        assert_eq!(count_italic_spans("*foo* and *bar*"), 2);
        // Multiplication in code: a * b
        assert_eq!(count_italic_spans("0.5f * 1023.0f"), 0);
        assert_eq!(count_italic_spans("a * b * c * d"), 0);
        // No italics
        assert_eq!(count_italic_spans("no stars here"), 0);
    }

    #[test]
    fn test_hlsl_code_not_demoted_to_text() {
        let hlsl = " 1     uint4 EncodeTBN(in float3 normal, in float3 tangent, in uint bitangentHandedness)\n \
 2     {\n \
 3       // octahedron normal vector encoding\n \
 4       uint2 encodedNormal = uint2((EncodeNormal(normal) * 0.5f + 0.5f) * 1023.0f);\n \
 5\n \
 6       // find largest component of tangent\n \
 7       float3 tangentAbs = abs(tangent);\n \
 8       float maxComp = max(max(tangentAbs.x, tangentAbs.y), tangentAbs.z);\n \
 9       float3 refVector;\n \
10       uint compIndex;\n \
11       if(maxComp == tangentAbs.x)\n \
12       {\n \
13         refVector = float3(1.0f, 0.0f, 0.0f);\n \
14         compIndex = 0;\n \
15       }";
        // Should score highly for code
        assert!(code_score(hlsl) >= 2, "HLSL code_score too low: {}", code_score(hlsl));
        // Multiplication * should NOT trigger italic detection
        assert_eq!(count_italic_spans(hlsl), 0);
        // Should be detected as code
        let mut nodes = vec![ReflowNode::Algorithm { content: hlsl.to_string() }];
        detect_code_blocks(&mut nodes);
        assert!(
            matches!(&nodes[0], ReflowNode::CodeBlock { .. }),
            "HLSL Algorithm should become CodeBlock, got: {:?}",
            std::mem::discriminant(&nodes[0])
        );
    }

    // ── is_labeled_block tests ───────────────────────────────────────

    #[test]
    fn test_is_labeled_block_basic() {
        assert!(is_labeled_block("Example 3"));
        assert!(is_labeled_block("Figure 1"));
        assert!(is_labeled_block("Table 2"));
        assert!(is_labeled_block("Listing 4"));
        assert!(is_labeled_block("Algorithm 1"));
        assert!(is_labeled_block("Fig. 5"));
        assert!(is_labeled_block("Fig 5"));
    }

    #[test]
    fn test_is_labeled_block_hyphenated() {
        assert!(is_labeled_block("Figure 2-4"));
        assert!(is_labeled_block("Example 3-3. A Sample Vertex Shader"));
        assert!(is_labeled_block("Table 3-1: Performance Results"));
    }

    #[test]
    fn test_is_labeled_block_case_insensitive() {
        assert!(is_labeled_block("EXAMPLE 1"));
        assert!(is_labeled_block("FIGURE 2"));
        assert!(is_labeled_block("TABLE 3"));
        assert!(is_labeled_block("figure 1"));
    }

    #[test]
    fn test_is_labeled_block_dotted() {
        assert!(is_labeled_block("Figure 2.4"));
        assert!(is_labeled_block("Table 1.2.3"));
        assert!(is_labeled_block("Example 3.1: Details"));
    }

    #[test]
    fn test_is_labeled_block_extended() {
        assert!(is_labeled_block("Definition 3.1"));
        assert!(is_labeled_block("Theorem 2"));
        assert!(is_labeled_block("Lemma 1"));
        assert!(is_labeled_block("Corollary 4.2"));
        assert!(is_labeled_block("Proposition 1"));
    }

    #[test]
    fn test_is_labeled_block_bare_short() {
        assert!(is_labeled_block("Proof"));
        assert!(is_labeled_block("Remark"));
        // Long sentence starting with a label word → not a block
        assert!(!is_labeled_block(
            "Examples of convergence in nonlinear optimization problems"
        ));
    }

    #[test]
    fn test_is_labeled_block_not_headings() {
        assert!(!is_labeled_block("Introduction"));
        assert!(!is_labeled_block("1.2 Methods"));
        assert!(!is_labeled_block("Background"));
        assert!(!is_labeled_block("Conclusions and Future Work"));
    }

    #[test]
    fn test_is_labeled_block_partial_word() {
        assert!(!is_labeled_block("Figurehead of the organization"));
        assert!(!is_labeled_block("Tipping point analysis"));
    }

    #[test]
    fn test_is_labeled_block_acm() {
        assert!(is_labeled_block("ACM Reference Format"));
    }

    // ── has_leader_dots / is_page_number_like tests ──────────────────

    #[test]
    fn test_has_leader_dots_dense() {
        assert!(has_leader_dots("Introduction .................. 3"));
        assert!(has_leader_dots("3.1 Introduction .......................... 3-1"));
        assert!(has_leader_dots("Background ........ 15"));
    }

    #[test]
    fn test_has_leader_dots_spaced() {
        assert!(has_leader_dots("2.1 Methods . . . . . . . . . . . 23"));
    }

    #[test]
    fn test_has_leader_dots_no_page() {
        assert!(has_leader_dots("Contents ..................."));
    }

    #[test]
    fn test_has_leader_dots_roman() {
        assert!(has_leader_dots("Preface ............... xii"));
    }

    #[test]
    fn test_has_leader_dots_hyphenated() {
        assert!(has_leader_dots("A.2 Setup .............. A-3"));
    }

    #[test]
    fn test_has_leader_dots_no_false_positive() {
        assert!(!has_leader_dots("Dr. Smith et al. found that..."));
        assert!(!has_leader_dots("i.e. the approach works well."));
        assert!(!has_leader_dots("Section 3.1.2 describes the method."));
    }

    #[test]
    fn test_has_leader_dots_ellipsis_prose() {
        assert!(!has_leader_dots("He said... and then left."));
        assert!(!has_leader_dots("The results were... inconclusive."));
    }

    #[test]
    fn test_is_page_number_like() {
        assert!(is_page_number_like("3"));
        assert!(is_page_number_like("142"));
        assert!(is_page_number_like("3-1"));
        assert!(is_page_number_like("A-3"));
        assert!(is_page_number_like("xii"));
        assert!(is_page_number_like("iv"));
        assert!(!is_page_number_like("hello"));
        assert!(!is_page_number_like("123456")); // too long
        assert!(!is_page_number_like(""));
    }

    // ── Empty image / heading echo tests ─────────────────────────────

    #[test]
    fn test_render_empty_figure_skipped() {
        let doc = ReflowDocument {
            title: None,
            toc: Vec::new(),
            children: vec![ReflowNode::Figure {
                path: String::new(),
                caption: Some("A caption".into()),
            }],
        };
        let md = render_markdown_from_reflow(&doc);
        assert!(!md.contains("![]()"), "empty image ref should not appear");
        assert!(md.contains("A caption"), "caption should still render");
    }

    #[test]
    fn test_render_figure_with_path() {
        let doc = ReflowDocument {
            title: None,
            toc: Vec::new(),
            children: vec![ReflowNode::Figure {
                path: "images/p1_0.png".into(),
                caption: None,
            }],
        };
        let md = render_markdown_from_reflow(&doc);
        assert!(md.contains("![](images/p1_0.png)"));
    }

    #[test]
    fn test_dedup_heading_echo_exact() {
        let mut nodes = vec![
            ReflowNode::Heading {
                depth: 1,
                text: "Introduction".into(),
                section: Some("3.1".into()),
                children: Vec::new(),
            },
            ReflowNode::Text {
                content: "3.1 Introduction".into(),
                footnotes: Vec::new(),
            },
            ReflowNode::Text {
                content: "Actual body text here.".into(),
                footnotes: Vec::new(),
            },
        ];
        dedup_heading_echo(&mut nodes);
        assert_eq!(nodes.len(), 2, "echo should be removed");
    }

    #[test]
    fn test_dedup_heading_echo_preserves_unrelated() {
        let mut nodes = vec![
            ReflowNode::Heading {
                depth: 1,
                text: "Methods".into(),
                section: Some("2.1".into()),
                children: Vec::new(),
            },
            ReflowNode::Text {
                content: "Completely different paragraph.".into(),
                footnotes: Vec::new(),
            },
        ];
        dedup_heading_echo(&mut nodes);
        assert_eq!(nodes.len(), 2, "unrelated text should not be removed");
    }

    #[test]
    fn test_dedup_heading_echo_across_formula() {
        let mut nodes = vec![
            ReflowNode::Heading {
                depth: 1,
                text: "Results".into(),
                section: Some("4.2".into()),
                children: Vec::new(),
            },
            ReflowNode::Formula {
                content: Some("$$x^2$$".into()),
                path: None,
            },
            ReflowNode::Text {
                content: "4.2 Results".into(),
                footnotes: Vec::new(),
            },
        ];
        dedup_heading_echo(&mut nodes);
        assert_eq!(nodes.len(), 2, "echo past formula should be removed");
    }
}
