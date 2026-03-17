use std::path::Path;

use image::DynamicImage;
use pdfium_render::prelude::*;

use crate::error::ExtractError;
use crate::pdf;
use crate::toc::TocEntry;
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
    let doc = reflow(result);
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

        let cropped = crate::figure::crop_region(
            page_img,
            region.bbox,
            page.width_pt,
            page.height_pt,
            page.dpi,
        );

        cropped.save(&full_path)?;
    }
    Ok(())
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

/// Check if a heading text is a labeled block (Example, Tip, Algorithm)
/// that should be treated as content rather than a structural heading.
fn is_labeled_block(text: &str) -> bool {
    let text = text.trim();
    text.starts_with("Example ") || text.starts_with("Tip ") || text.starts_with("Algorithm ")
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

/// Build sections from extraction result: render regions to markdown,
/// merge references, dehyphenate across region boundaries, and rejoin
/// paragraphs split across column/page breaks.
fn build_sections(result: &ExtractionResult, skip_pages: &[u32]) -> Vec<Section> {
    let mut sections: Vec<Section> = Vec::new();

    for page in &result.pages {
        let page_idx = page.page.saturating_sub(1);
        if skip_pages.contains(&page_idx) {
            continue;
        }
        for region in &page.regions {
            // Skip regions whose content was spliced into a parent region.
            if region.consumed {
                continue;
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
                is_text,
                is_references,
                is_formula,
                is_short_line,
                formula_path,
            });
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
            ReflowNode::Figure { path, caption }
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
            ReflowNode::FigureGroup { path, caption }
        }
        RegionKind::References => ReflowNode::References {
            content: text.to_string(),
        },
        _ => ReflowNode::Text {
            content: text.to_string(),
        },
    }
}

/// Build a reflow document from the extraction result.
///
/// This performs all reflow logic (dehyphenation, paragraph merging, references
/// merging) and organizes the result into a heading-based tree.
pub fn reflow(result: &ExtractionResult) -> ReflowDocument {
    let sections = build_sections(result, &[]);

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
                ReflowNode::Heading {
                    depth: 0,
                    title,
                    children: Vec::new(),
                }
            }
            RegionKind::ParagraphTitle => {
                // Section heading — strip the "## " prefix
                let title = text.strip_prefix("## ").unwrap_or(text).to_string();
                if is_labeled_block(&title) {
                    // Labeled blocks (Example, Tip, Algorithm headings) are content
                    ReflowNode::Text {
                        content: format!("## {title}"),
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
                                        title: format!("Appendix {letter}"),
                                        children: Vec::new(),
                                    });
                                }
                            }
                        }
                        last_numbered_depth = depth;
                    }
                    ReflowNode::Heading {
                        depth,
                        title,
                        children: Vec::new(),
                    }
                }
            }
            _ => section_to_content_node(sec),
        };

        flat_nodes.push(node);
    }

    // Build the heading tree from the flat list using a depth stack.
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
pub(crate) fn compute_page_offset(
    toc_entries: &[TocEntry],
    toc_pages: &[u32],
    result: &ExtractionResult,
) -> i32 {
    // Try to match body TOC entries to ParagraphTitle regions to compute offset.
    // Try the first several body entries — the first one may not be detected.
    let body_entries: Vec<&TocEntry> = toc_entries
        .iter()
        .filter(|e| e.page_value > 0)
        .take(20)
        .collect();

    if body_entries.is_empty() {
        return toc_pages.last().map(|&p| p as i32 + 1).unwrap_or(0);
    }

    for entry in &body_entries {
        for page in &result.pages {
            let page_idx = page.page.saturating_sub(1);
            for region in &page.regions {
                if region.kind != RegionKind::ParagraphTitle {
                    continue;
                }
                let Some(ref text) = region.text else { continue };

                if titles_match(&entry.title, text) {
                    return page_idx as i32 - (entry.page_value - 1) as i32;
                }
            }
        }
    }

    // Fallback: body starts right after last TOC page
    toc_pages.last().map(|&p| p as i32 + 1).unwrap_or(0)
}

/// Map a printed page value to a 0-indexed PDF page.
pub(crate) fn toc_page_to_pdf_page(page_value: i32, offset: i32, total_pages: u32) -> Option<u32> {
    let idx = if page_value > 0 {
        (page_value - 1) as i32 + offset
    } else {
        // Front matter: rough estimate, offset backwards from body start
        offset + page_value
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

    // Prefix match: TOC title is prefix of region text
    if !toc_norm.is_empty() && region_norm.starts_with(&toc_norm) {
        return true;
    }
    // Region text is prefix of TOC title (truncated region text)
    if !region_norm.is_empty() && region_norm.len() >= 8 && toc_norm.starts_with(&region_norm) {
        return true;
    }

    // Fuzzy: >80% of TOC title words appear in region text
    if !toc_stripped.is_empty() {
        let toc_words: Vec<&str> = toc_stripped.split_whitespace().collect();
        if toc_words.len() >= 2 {
            let matched = toc_words
                .iter()
                .filter(|w| region_stripped.contains(**w))
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
/// TOC entries determine heading depths. ParagraphTitle regions matched to
/// TOC entries use the TOC depth; unmatched ParagraphTitles are demoted to
/// text. Missing TOC headings are injected as synthetic heading nodes.
pub fn reflow_with_outline(
    result: &ExtractionResult,
    toc_entries: &[TocEntry],
    toc_pages: &[u32],
    total_pages: u32,
) -> ReflowDocument {
    let sections = build_sections(result, toc_pages);

    // Step 1: Compute page offset
    let offset = compute_page_offset(toc_entries, toc_pages, result);

    // Step 2: Check if Parts exist — if so, shift all depths +1
    let has_parts = toc_entries.iter().any(|e| e.depth == 0);

    // Step 3: Pre-match TOC entries to ParagraphTitle sections.
    // For each TOC entry, find the best matching section index.
    // matched_sections[section_idx] = Some(toc_entry_idx)
    let mut section_to_toc: Vec<Option<usize>> = vec![None; sections.len()];
    let mut toc_matched: Vec<bool> = vec![false; toc_entries.len()];

    for (toc_idx, entry) in toc_entries.iter().enumerate() {
        let expected_pdf_page =
            toc_page_to_pdf_page(entry.page_value, offset, total_pages);

        let mut best_section: Option<usize> = None;
        let mut best_score: u32 = 0;

        for (sec_idx, sec) in sections.iter().enumerate() {
            if sec.kind != RegionKind::ParagraphTitle {
                continue;
            }
            if section_to_toc[sec_idx].is_some() {
                continue; // already claimed
            }

            let title = sec
                .markdown
                .trim()
                .strip_prefix("## ")
                .unwrap_or(sec.markdown.trim());

            if !titles_match(&entry.title, title) {
                continue;
            }

            // Page proximity scoring
            let page_ok = match expected_pdf_page {
                Some(expected) => {
                    let diff = (sec.page_idx as i32 - expected as i32).unsigned_abs();
                    diff <= 3
                }
                None => true, // can't verify page, accept any match
            };

            if !page_ok {
                continue;
            }

            // Score: exact normalized match > prefix > fuzzy
            let toc_norm = normalize_title(&entry.title);
            let sec_norm = normalize_title(title);
            let score = if toc_norm == sec_norm {
                3
            } else if sec_norm.starts_with(&toc_norm) || toc_norm.starts_with(&sec_norm) {
                2
            } else {
                1
            };

            if score > best_score {
                best_score = score;
                best_section = Some(sec_idx);
            }
        }

        if let Some(sec_idx) = best_section {
            section_to_toc[sec_idx] = Some(toc_idx);
            toc_matched[toc_idx] = true;
        }
    }

    // Step 4: Build flat nodes with outline-aware heading assignment.
    let mut flat_nodes: Vec<ReflowNode> = Vec::new();
    // Track current position by page for injecting missing headings.
    // We'll inject missing headings between sections based on page order.
    let mut next_inject_check: usize = 0;

    for (sec_idx, sec) in sections.iter().enumerate() {
        let text = sec.markdown.trim();
        if text.is_empty() && sec.formula_path.is_none() {
            continue;
        }

        // Before processing this section, inject any missing TOC headings
        // that should appear before this section's page position.
        inject_missing_headings(
            &mut flat_nodes,
            toc_entries,
            &toc_matched,
            offset,
            total_pages,
            has_parts,
            sec.page_idx,
            &mut next_inject_check,
        );

        let node = match sec.kind {
            RegionKind::Title => {
                let title = text.strip_prefix("# ").unwrap_or(text).to_string();
                ReflowNode::Heading {
                    depth: 0,
                    title,
                    children: Vec::new(),
                }
            }
            RegionKind::ParagraphTitle => {
                let title = text.strip_prefix("## ").unwrap_or(text).to_string();

                if let Some(toc_idx) = section_to_toc[sec_idx] {
                    // Matched to TOC entry — use its depth and title
                    let mut depth = toc_entries[toc_idx].depth;
                    if has_parts {
                        depth += 1;
                    }
                    // Depth 0 entries (Parts) become depth 1 when shifted
                    let depth = depth.max(1);
                    let toc_title = toc_entries[toc_idx].title.clone();
                    ReflowNode::Heading {
                        depth,
                        title: toc_title,
                        children: Vec::new(),
                    }
                } else if is_labeled_block(&title) {
                    // Labeled blocks are content, not headings — bold the label
                    ReflowNode::Text {
                        content: format!("**{title}**"),
                    }
                } else if title.is_empty() {
                    // Empty heading — skip entirely
                    continue;
                } else {
                    // Not matched to TOC — demote to text
                    ReflowNode::Text {
                        content: title.clone(),
                    }
                }
            }
            _ => section_to_content_node(sec),
        };

        flat_nodes.push(node);
    }

    // Inject any remaining unmatched TOC headings that go after all sections.
    inject_missing_headings(
        &mut flat_nodes,
        toc_entries,
        &toc_matched,
        offset,
        total_pages,
        has_parts,
        u32::MAX,
        &mut next_inject_check,
    );

    build_heading_tree(flat_nodes)
}

/// Inject synthetic heading nodes for TOC entries that weren't matched to any
/// ParagraphTitle region and whose expected PDF page is <= `up_to_page`.
fn inject_missing_headings(
    flat_nodes: &mut Vec<ReflowNode>,
    toc_entries: &[TocEntry],
    toc_matched: &[bool],
    offset: i32,
    total_pages: u32,
    has_parts: bool,
    up_to_page: u32,
    next_check: &mut usize,
) {
    while *next_check < toc_entries.len() {
        let entry = &toc_entries[*next_check];
        let expected = toc_page_to_pdf_page(entry.page_value, offset, total_pages)
            .unwrap_or(u32::MAX);

        if expected > up_to_page {
            break;
        }

        if !toc_matched[*next_check] {
            let mut depth = entry.depth;
            if has_parts {
                depth += 1;
            }
            let depth = depth.max(1);
            flat_nodes.push(ReflowNode::Heading {
                depth,
                title: entry.title.clone(),
                children: Vec::new(),
            });
        }

        *next_check += 1;
    }
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

/// Build a heading tree from a flat list of reflow nodes.
///
/// Heading nodes nest subsequent content until a heading of equal or lesser depth
/// is encountered.
fn build_heading_tree(flat: Vec<ReflowNode>) -> ReflowDocument {
    let mut doc = ReflowDocument {
        title: None,
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
            ReflowNode::Heading { depth, title, .. } => {
                let depth = *depth;

                // Document title (depth 0): set as doc title
                if depth == 0 {
                    flush_to_depth(&mut stack, &mut doc.children, 1);
                    doc.title = Some(title.clone());
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

    // Render children
    render_children(&doc.children, &mut parts);

    let md = parts.join("\n\n");
    md.trim_end().to_string()
}

/// Recursively render reflow nodes into markdown parts.
fn render_children(children: &[ReflowNode], parts: &mut Vec<String>) {
    for node in children {
        match node {
            ReflowNode::Heading {
                depth,
                title,
                children,
            } => {
                let depth = (*depth as usize).min(5); // cap at h6
                let hashes = "#".repeat(depth + 1);
                let title = collapse_spaced_letters(title);
                let title = fix_missing_spaces_in_title(&title);
                parts.push(format!("{hashes} {title}"));
                render_children(children, parts);
            }
            ReflowNode::Text { content, .. } => {
                if !content.trim().is_empty() {
                    parts.push(escape_leading_hashes(content.trim()));
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
                if let Some(cap) = caption {
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
                parts.push(escape_leading_hashes(content));
            }
            ReflowNode::FigureGroup { path, caption } => {
                if let Some(cap) = caption {
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
        if line.starts_with('#') {
            out.push('\\');
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
        RegionKind::Algorithm => {
            if let Some(ref text) = region.text {
                text.clone()
            } else {
                String::new()
            }
        }
        RegionKind::Footnote => {
            // Footnotes are metadata (author addresses, permissions, etc.)
            // — omit from the main markdown body.
            String::new()
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

        for region in &page_info.regions {
            let color = region_color(region.kind);
            let [x1, y1, x2, y2] = region.bbox;

            // bbox is in top-left-origin points; convert to PDF bottom-left-origin
            let pdf_left = x1;
            let pdf_right = x2;
            let pdf_bottom = page_h - y2;
            let pdf_top = page_h - y1;

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
            let label_y = if pdf_top + font_size + 1.0 < page_h {
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
        let doc = reflow(result);
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
}
