//! Text-only extraction: build text block regions from the PDF text layer
//! using geometric heuristics, without any ML model inference.
//!
//! Includes column detection, watermark removal, and running header filtering.

use std::collections::{HashMap, HashSet};

use crate::headings::DetectedHeading;
use crate::pdf::PdfChar;
use crate::reading_order;
use crate::text;
use crate::text_cleanup;
use crate::types::{Region, RegionKind};

/// Margin threshold in PDF points — chars within this distance of the page
/// top/bottom edge are filtered as running headers/footers.
const MARGIN_PT: f32 = 25.0;

/// Minimum alphanumeric character ratio — blocks below this are likely
/// formula fragments from overlapping PDF text layer streams.
const MIN_ALPHA_RATIO: f32 = 0.3;

/// Minimum text length for a block to be kept (after trimming).
const MIN_TEXT_LEN: usize = 2;

// ── Data structures ─────────────────────────────────────────────────

/// A character with image-space coordinates and font metadata.
#[derive(Clone)]
struct ImgChar {
    /// Bounding box in image space (Y-down): [x1, y1, x2, y2].
    bbox: [f32; 4],
    font_size: f32,
    is_bold: bool,
    /// Unicode codepoint — used to build per-line text for formula detection.
    codepoint: char,
}

impl text_cleanup::HasBbox for ImgChar {
    fn bbox(&self) -> [f32; 4] { self.bbox }
}

/// A line of characters grouped by Y-proximity.
struct Line {
    y_center: f32,
    x_min: f32,
    x_max: f32,
    font_size: f32,
    is_bold: bool,
    bbox: [f32; 4],
    /// Rough text content assembled from codepoints — used for formula detection.
    text: String,
}

/// A text block: a group of consecutive lines forming a paragraph or heading.
struct Block {
    bbox: [f32; 4],
    line_count: usize,
    font_size: f32,
    is_bold: bool,
    is_topmost: bool,
}

/// A detected column gap.
struct ColumnGap {
    midpoint: f32,
    start: f32,
    end: f32,
}

// ── Watermark detection (cross-page) ────────────────────────────────

/// Detect watermark strings that appear on >50% of pages with the same font.
///
/// Returns a set of watermark text strings to strip from each page.
pub fn detect_watermark_strings(page_chars: &[(Vec<PdfChar>, f32)]) -> HashSet<String> {
    if page_chars.len() < 5 {
        return HashSet::new();
    }

    // For each page, group consecutive chars by font into short text segments
    let mut segment_pages: HashMap<(String, String), HashSet<usize>> = HashMap::new();

    for (page_idx, (chars, _)) in page_chars.iter().enumerate() {
        let mut segments = extract_font_segments(chars);
        for seg in &mut segments {
            if seg.len() >= 5 && seg.len() <= 40 {
                let key = (seg.clone(), String::new()); // font name omitted for simplicity
                segment_pages.entry(key).or_default().insert(page_idx);
            }
        }
    }

    let threshold = page_chars.len() / 2; // >50%
    let mut watermarks = HashSet::new();
    for ((text, _), pages) in &segment_pages {
        if pages.len() > threshold {
            watermarks.insert(text.clone());
        }
    }
    watermarks
}

/// Extract short text segments from chars by grouping consecutive chars
/// with the same font that are close together.
fn extract_font_segments(chars: &[PdfChar]) -> Vec<String> {
    if chars.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = String::new();
    let mut prev_x_end = f32::MIN;
    let mut prev_font = String::new();

    for c in chars {
        if c.codepoint.is_control() {
            continue;
        }
        let x_start = c.bbox[0];
        let char_width = (c.bbox[2] - c.bbox[0]).max(1.0);
        let gap = x_start - prev_x_end;
        let same_font = c.font_name == prev_font;
        let close_enough = gap < char_width * 3.0;

        if same_font && close_enough && !current.is_empty() {
            if gap > c.space_threshold {
                current.push(' ');
            }
            current.push(c.codepoint);
        } else {
            if current.len() >= 5 {
                segments.push(current.clone());
            }
            current.clear();
            current.push(c.codepoint);
        }
        prev_x_end = c.bbox[2];
        prev_font = c.font_name.clone();
    }
    if current.len() >= 5 {
        segments.push(current);
    }
    segments
}

// ── Column gap detection (char-level) ───────────────────────────────

/// Detect column gaps from char-level X-projection.
///
/// Returns gaps sorted by X position. Empty = single column.
fn detect_column_gaps(chars: &[ImgChar], page_width_pt: f32) -> Vec<ColumnGap> {
    if chars.len() < 20 {
        return Vec::new();
    }

    // Compute text content X-bounds
    let text_x_min = chars.iter().map(|c| c.bbox[0]).fold(f32::MAX, f32::min);
    let text_x_max = chars.iter().map(|c| c.bbox[2]).fold(f32::MIN, f32::max);
    let text_width = text_x_max - text_x_min;

    if text_width < page_width_pt * 0.4 || text_width < 100.0 {
        eprintln!("    column: text_width={text_width:.0} too narrow for page_width={page_width_pt:.0}");
        return Vec::new(); // Too narrow for multi-column
    }

    // Compute median char width
    let mut char_widths: Vec<f32> = chars.iter().map(|c| c.bbox[2] - c.bbox[0]).collect();
    char_widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_char_width = char_widths[char_widths.len() / 2];
    let min_gap_width = (median_char_width * 2.0).max(8.0);

    // Build 1D X-projection histogram (1pt bins)
    let bin_offset = text_x_min.floor() as usize;
    let bin_count = (text_x_max.ceil() as usize).saturating_sub(bin_offset) + 1;
    if bin_count < 10 {
        return Vec::new();
    }
    let mut histogram = vec![0u32; bin_count];
    for c in chars {
        let cx = ((c.bbox[0] + c.bbox[2]) / 2.0).floor() as usize;
        let idx = cx.saturating_sub(bin_offset);
        if idx < bin_count {
            histogram[idx] += 1;
        }
    }

    // Compute a "low count" threshold: bins with very few chars relative to
    // the median are treated as empty. This handles stray math symbols,
    // superscripts, or other elements that spill into the column gutter.
    let mut sorted_hist: Vec<u32> = histogram.iter().copied().filter(|&c| c > 0).collect();
    sorted_hist.sort();
    let median_count = if sorted_hist.is_empty() {
        1
    } else {
        sorted_hist[sorted_hist.len() / 2]
    };
    let low_threshold = (median_count / 10).max(1); // ≤10% of median = effectively empty

    // Find contiguous runs of low-count bins
    let mut gaps = Vec::new();
    let mut run_start = None;
    for (i, &count) in histogram.iter().enumerate() {
        if count <= low_threshold {
            if run_start.is_none() {
                run_start = Some(i);
            }
        } else if let Some(start) = run_start {
            let width = (i - start) as f32;
            if width >= min_gap_width {
                let gap_start_pt = start as f32 + bin_offset as f32;
                let gap_end_pt = i as f32 + bin_offset as f32;
                gaps.push((gap_start_pt, gap_end_pt));
            }
            run_start = None;
        }
    }

    // Vertical validation: each gap must span ≥50% of text height
    let text_y_min = chars.iter().map(|c| c.bbox[1]).fold(f32::MAX, f32::min);
    let text_y_max = chars.iter().map(|c| c.bbox[3]).fold(f32::MIN, f32::max);
    let text_height = text_y_max - text_y_min;
    if text_height < 50.0 {
        return Vec::new();
    }

    let slice_height = 20.0f32;
    let n_slices = (text_height / slice_height).ceil() as usize;

    gaps.into_iter()
        .filter_map(|(gs, ge)| {
            // Count Y-slices where any char's X-extent overlaps the gap
            let mut filled_slices = 0;
            for s in 0..n_slices {
                let sy = text_y_min + s as f32 * slice_height;
                let ey = sy + slice_height;
                let has_overlap = chars.iter().any(|c| {
                    let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
                    let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
                    cy >= sy && cy < ey && cx > gs && cx < ge
                });
                if has_overlap {
                    filled_slices += 1;
                }
            }
            let fill_ratio = filled_slices as f32 / n_slices.max(1) as f32;
            if fill_ratio <= 0.4 {
                Some(ColumnGap {
                    midpoint: (gs + ge) / 2.0,
                    start: gs,
                    end: ge,
                })
            } else {
                None // Too many Y-slices have chars in the gap → not a real column gap
            }
        })
        .collect()
}

// ── Column splitting (per Y-band) ───────────────────────────────────

/// Split chars into spanning group + per-column groups.
///
/// A Y-band is "spanning" if chars exist on both sides of the gap AND the
/// inter-char distance across the gap is smaller than the gap width itself
/// (text flows through the gutter, e.g., a full-width title).
fn split_chars_into_columns(
    chars: &[ImgChar],
    gaps: &[ColumnGap],
) -> (Vec<ImgChar>, Vec<Vec<ImgChar>>) {
    let n_columns = gaps.len() + 1;
    let mut spanning = Vec::new();
    let mut columns: Vec<Vec<ImgChar>> = (0..n_columns).map(|_| Vec::new()).collect();

    if gaps.is_empty() {
        columns[0] = chars.to_vec();
        return (spanning, columns);
    }

    // Compute average char height for Y-band grouping
    let avg_height: f32 =
        chars.iter().map(|c| c.bbox[3] - c.bbox[1]).sum::<f32>() / chars.len().max(1) as f32;
    let y_band_height = (avg_height * 1.5).max(8.0);

    // Find Y extent
    let y_min = chars.iter().map(|c| c.bbox[1]).fold(f32::MAX, f32::min);
    let y_max = chars.iter().map(|c| c.bbox[3]).fold(f32::MIN, f32::max);

    // Process each Y-band
    let mut y = y_min;
    while y < y_max {
        let band_top = y;
        let band_bottom = y + y_band_height;
        y = band_bottom;

        // Collect chars in this Y-band
        let band_chars: Vec<usize> = chars
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
                cy >= band_top && cy < band_bottom
            })
            .map(|(i, _)| i)
            .collect();

        if band_chars.is_empty() {
            continue;
        }

        // For each gap, check if this Y-band is spanning.
        // A Y-band is spanning if text flows continuously from one column
        // through the gap into the other. We verify:
        // 1. Chars exist on both sides of the gap
        // 2. No large horizontal break between consecutive chars (sorted by X)
        // This distinguishes two separate column lines (~20-30pt break at the
        // column boundary) from a spanning title (~3-6pt char spacing throughout).
        let mut is_spanning = false;
        let band_cxs: Vec<f32> = band_chars
            .iter()
            .map(|&i| (chars[i].bbox[0] + chars[i].bbox[2]) / 2.0)
            .collect();
        for gap in gaps {
            let has_left = band_cxs.iter().any(|&cx| cx < gap.start);
            let has_right = band_cxs.iter().any(|&cx| cx > gap.end);
            if has_left && has_right {
                let mut sorted_cxs = band_cxs.clone();
                sorted_cxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let max_char_gap = sorted_cxs
                    .windows(2)
                    .map(|w| w[1] - w[0])
                    .fold(0.0f32, f32::max);
                let gap_width = gap.end - gap.start;
                if max_char_gap < gap_width * 0.5 {
                    is_spanning = true;
                    break;
                }
            }
        }

        if is_spanning {
            for &i in &band_chars {
                spanning.push(chars[i].clone());
            }
        } else {
            // Assign each char to its column
            for &i in &band_chars {
                let cx = (chars[i].bbox[0] + chars[i].bbox[2]) / 2.0;
                let col = gaps
                    .iter()
                    .position(|g| cx < g.midpoint)
                    .unwrap_or(n_columns - 1);
                columns[col].push(chars[i].clone());
            }
        }
    }

    (spanning, columns)
}

// ── Formula zone detection ──────────────────────────────────────────

/// Detect Y-bands on the page that contain display formulas.
///
/// Does a preliminary line grouping on ALL chars (before heading partitioning)
/// and identifies lines that look like formulas by content + indentation.
/// Returns a list of (y_top, y_bottom) ranges in image space.
fn detect_formula_zones(
    chars: &[PdfChar],
    page_height_pt: f32,
    watermarks: &HashSet<String>,
) -> Vec<(f32, f32)> {
    // Convert ALL chars to ImgChars (no heading filter)
    let img_chars = convert_and_filter(chars, page_height_pt, watermarks);
    if img_chars.is_empty() {
        return Vec::new();
    }

    let lines = group_into_lines(&img_chars);
    if lines.len() < 3 {
        return Vec::new();
    }

    // Compute median line width for indentation detection
    let median_width = {
        let mut widths: Vec<f32> = lines.iter().map(|l| l.x_max - l.x_min).collect();
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        widths[widths.len() / 2]
    };

    // Identify formula lines: shorter than median (indented/centered) + math content
    let mut zones = Vec::new();
    for line in &lines {
        let t = line.text.trim();
        let line_width = line.x_max - line.x_min;
        let is_short = line_width < median_width * 0.85;
        let has_math = t.len() >= 5 && crate::text_cleanup::is_likely_formula_text(t);
        if is_short && has_math {
            // Expand the zone generously to cover the full line height
            // including the left-hand side of the equation (e.g., f_i =)
            // which may be at a slightly different Y due to bold baseline.
            let line_h = line.bbox[3] - line.bbox[1];
            let margin = line_h.max(8.0); // at least 8pt margin
            zones.push((line.bbox[1] - margin, line.bbox[3] + margin));
        }
    }

    zones
}

// ── Heading-first char partitioning ─────────────────────────────────

/// Partition page chars into heading regions and remaining body char indices.
///
/// Chars whose normalized font family matches a heading family are grouped
/// into heading regions (ParagraphTitle). Remaining chars are returned as
/// indices into the original `chars` slice for body processing.
fn partition_heading_chars(
    chars: &[PdfChar],
    page_height_pt: f32,
    page_idx: u32,
    heading_families: &HashSet<String>,
    known_headings: &[&DetectedHeading],
    formula_zones: &[(f32, f32)],
) -> (Vec<Region>, Vec<usize>) {
    if heading_families.is_empty() {
        // No font-based heading detection — all chars are body
        return (Vec::new(), (0..chars.len()).collect());
    }

    let mut heading_runs: Vec<(usize, usize)> = Vec::new(); // (start, end) index ranges
    let mut body_indices: Vec<usize> = Vec::new();
    let mut in_heading = false;
    let mut run_start = 0;

    for (i, c) in chars.iter().enumerate() {
        if c.codepoint.is_control() || c.codepoint == '\u{FEFF}' {
            if in_heading {
                // Keep control chars in heading run (they'll be filtered later)
            } else {
                body_indices.push(i);
            }
            continue;
        }

        let family = crate::headings::normalize_font_family(&c.font_name);
        // A char is heading-font if it's in a heading font family.
        // Formula zones suppress heading detection for bold math variables
        // (f_i, H_i) that use the heading font — UNLESS the char is near
        // a known heading position. The font IS the definitive signal for
        // headings; formula zones should not override confirmed headings.
        let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
        let in_formula_zone = formula_zones.iter().any(|&(yt, yb)| cy >= yt && cy <= yb);
        let near_known_heading = known_headings.iter().any(|h| (h.y_center - cy).abs() < 15.0);
        let is_heading_char = heading_families.contains(&family)
            && (!in_formula_zone || near_known_heading);

        if is_heading_char {
            if !in_heading {
                run_start = i;
                in_heading = true;
            }
        } else {
            if in_heading {
                heading_runs.push((run_start, i));
                in_heading = false;
            }
            body_indices.push(i);
        }
    }
    if in_heading {
        heading_runs.push((run_start, chars.len()));
    }

    // Convert heading runs into Region structs, tracking which run each region came from
    let mut heading_regions: Vec<(Region, usize, usize)> = Vec::new(); // (region, start, end)
    for (start, end) in &heading_runs {
        let run_chars = &chars[*start..*end];
        if run_chars.is_empty() {
            continue;
        }

        // Compute bbox in image space
        let mut bbox = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        let mut text = String::new();
        let mut prev_x_end = f32::MIN;

        for c in run_chars {
            if c.codepoint.is_control() || c.codepoint == '\u{FEFF}' {
                continue;
            }
            let x1 = c.bbox[0];
            let x2 = c.bbox[2];
            let y1 = c.bbox[1];
            let y2 = c.bbox[3];

            bbox[0] = bbox[0].min(x1);
            bbox[1] = bbox[1].min(y1);
            bbox[2] = bbox[2].max(x2);
            bbox[3] = bbox[3].max(y2);

            // Space detection
            if prev_x_end > f32::MIN {
                let gap = x1 - prev_x_end;
                if gap > c.space_threshold {
                    text.push(' ');
                }
            }
            text.push(c.codepoint);
            prev_x_end = x2;
        }

        let text = text.trim().to_string();
        if text.is_empty() || text.len() < 2 {
            // Too short — return chars to body pool
            for i in *start..*end {
                body_indices.push(i);
            }
            continue;
        }

        // Pad bbox
        bbox[0] -= 1.0;
        bbox[1] -= 1.0;
        bbox[2] += 1.0;
        bbox[3] += 1.0;

        let id = format!("p{}_{}", page_idx + 1, heading_regions.len());
        heading_regions.push((
            Region {
                id,
                kind: RegionKind::ParagraphTitle,
                bbox,
                confidence: 1.0,
                order: 0,
                text: Some(text),
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
            },
            *start,
            *end,
        ));
    }

    // Validate: only keep heading regions that match a known DetectedHeading.
    // Font-based partitioning is aggressive — it catches ALL text in the heading
    // font, including running headers, author names, bullet points, etc.
    // Cross-referencing with extract_headings() results filters to real headings only.
    let mut validated_regions = Vec::new();
    for (region, run_start, run_end) in heading_regions {
        let region_text = region.text.as_deref().unwrap_or("");
        let region_center_y = (region.bbox[1] + region.bbox[3]) / 2.0;

        let matches_known = known_headings.iter().any(|h| {
            // Y-position match (within ~20pt) — both in image space
            let y_close = (h.y_center - region_center_y).abs() < 20.0;
            // Text match (heading title is prefix of region text or vice versa)
            let h_norm = h.title.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
            let r_norm = region_text.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
            let text_match = !h_norm.is_empty()
                && !r_norm.is_empty()
                && (r_norm.starts_with(&h_norm) || h_norm.starts_with(&r_norm));
            y_close && text_match
        });

        if matches_known {
            validated_regions.push(region);
        } else {
            // Return chars to body pool — this wasn't a real heading
            for i in run_start..run_end {
                body_indices.push(i);
            }
        }
    }

    // Sort body indices (they may be out of order after returning short heading chars)
    body_indices.sort();

    (validated_regions, body_indices)
}

// ── Main extraction function ────────────────────────────────────────

/// Extract text blocks from a page's PDF text layer using geometric heuristics.
///
/// `headings` should be pre-filtered to only include `DetectedHeading`s for this page.
/// `watermarks` is the set of watermark strings to strip (from `detect_watermark_strings`).
/// `heading_families` are normalized font family names that correspond to headings
/// (different from body font). If non-empty, chars with these fonts are extracted
/// as heading regions BEFORE layout processing.
pub fn extract_page_text_blocks(
    chars: &[PdfChar],
    page_height_pt: f32,
    page_width_pt: f32,
    page_idx: u32,
    headings: &[&DetectedHeading],
    watermarks: &HashSet<String>,
    heading_families: &HashSet<String>,
) -> Vec<Region> {
    // Step 0a: Preliminary formula zone detection.
    // Scan ALL chars (before heading partitioning) to find Y-bands that
    // contain display formulas. These zones will be excluded from heading
    // font partitioning, preventing bold math variables (f_i, H_i, M_i)
    // from being extracted as heading chars.
    let formula_zones = detect_formula_zones(chars, page_height_pt, watermarks);

    // Step 0b: Extract heading chars by font, SKIPPING formula zones.
    let (heading_regions, body_char_indices) =
        partition_heading_chars(chars, page_height_pt, page_idx, heading_families, headings, &formula_zones);

    // Step 1: Convert BODY chars (heading chars removed) to image space
    let body_chars: Vec<PdfChar> = body_char_indices
        .iter()
        .map(|&i| chars[i].clone())
        .collect();
    let img_chars = convert_and_filter(&body_chars, page_height_pt, watermarks);

    // Step 2: Detect column gaps from body char X-projection
    let gaps = if img_chars.is_empty() {
        Vec::new()
    } else {
        detect_column_gaps(&img_chars, page_width_pt)
    };

    // Step 3: Build body blocks (column-aware or single-column)
    let body_blocks = if img_chars.is_empty() {
        Vec::new()
    } else if gaps.is_empty() {
        let lines = group_into_lines(&img_chars);
        group_into_blocks(&lines, &[])
    } else {
        let (spanning, columns) = split_chars_into_columns(&img_chars, &gaps);
        let mut blocks = Vec::new();
        for col_chars in &columns {
            if col_chars.is_empty() {
                continue;
            }
            let lines = group_into_lines(col_chars);
            blocks.extend(group_into_blocks(&lines, &[]));
        }
        if !spanning.is_empty() {
            let lines = group_into_lines(&spanning);
            blocks.extend(group_into_blocks(&lines, &[]));
        }
        blocks
    };

    // Start with heading regions (already classified as ParagraphTitle)
    let mut regions: Vec<Region> = heading_regions;

    if body_blocks.is_empty() && regions.is_empty() {
        return Vec::new();
    }

    // Mark topmost body block
    let mut blocks = body_blocks;
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

    // Step 4: Two-pass classification.
    // Pass 1: identify DisplayFormula blocks and collect their bboxes.
    // Pass 2: extract text for non-formula blocks, excluding chars in formula bboxes.
    let has_font_headings = !heading_families.is_empty();
    let page_median_font = if blocks.is_empty() {
        10.0
    } else {
        let mut sizes: Vec<f32> = blocks.iter().map(|b| b.font_size).collect();
        sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sizes[sizes.len() / 2]
    };

    // Pass 1: pre-classify blocks to find formulas
    let block_is_formula: Vec<bool> = blocks
        .iter()
        .map(|block| {
            // Quick text extraction for classification only
            let extracted = text::extract_region_text(
                &body_chars,
                block.bbox,
                page_height_pt,
                &[],
                &[],
                text::AssemblyMode::Reflow,
            );
            let trimmed = extracted.trim();
            if trimmed.len() < MIN_TEXT_LEN || is_formula_fragment(trimmed) {
                return false;
            }
            let kind = classify_block(block, trimmed, headings, page_height_pt, page_median_font, has_font_headings);
            kind == RegionKind::Text && text_cleanup::is_likely_formula_text(trimmed)
        })
        .collect();

    // Merge small adjacent fragments into formula blocks.
    // Formulas with fractions/subscripts can get split into multiple blocks
    // because the numerator/denominator chars are at different Y levels and
    // group_into_blocks() applies font_break between them. This creates
    // tiny fragments (e.g., "f_i = −") next to the main formula block.
    // We merge these fragments so the formula bbox is correct.
    let block_is_formula = block_is_formula;
    let mut block_consumed: Vec<bool> = vec![false; blocks.len()];
    {
        let formula_indices: Vec<usize> = block_is_formula
            .iter()
            .enumerate()
            .filter(|(_, is_f)| **is_f)
            .map(|(i, _)| i)
            .collect();

        for &fi in &formula_indices {
            let f_bbox = blocks[fi].bbox;
            let f_y_center = (f_bbox[1] + f_bbox[3]) / 2.0;
            let f_h = f_bbox[3] - f_bbox[1];

            for bi in 0..blocks.len() {
                if bi == fi || block_is_formula[bi] || block_consumed[bi] {
                    continue;
                }
                let b = &blocks[bi];
                let b_y_center = (b.bbox[1] + b.bbox[3]) / 2.0;
                let b_h = b.bbox[3] - b.bbox[1];

                // Must overlap in Y — the fragment's Y-center is within the
                // formula's vertical extent (with some tolerance for subscripts)
                let y_tolerance = f_h.max(b_h) * 0.5;
                if (f_y_center - b_y_center).abs() > f_h / 2.0 + y_tolerance {
                    continue;
                }

                // Must be horizontally adjacent (not overlapping, within
                // a small gap — typically just the space between "f_i =" and
                // the fraction bar)
                let h_gap = if b.bbox[2] < f_bbox[0] {
                    f_bbox[0] - b.bbox[2] // fragment is to the left
                } else if b.bbox[0] > f_bbox[2] {
                    b.bbox[0] - f_bbox[2] // fragment is to the right
                } else {
                    0.0 // overlapping
                };
                if h_gap > 30.0 {
                    continue; // too far away
                }

                // Must be small (formula fragments are typically just a few chars)
                let b_text = text::extract_region_text(
                    &body_chars,
                    b.bbox,
                    page_height_pt,
                    &[],
                    &[],
                    text::AssemblyMode::Reflow,
                );
                let b_trimmed = b_text.trim();
                if b_trimmed.len() > 40 {
                    continue; // too large to be a fragment
                }

                // Must look like math content (operators, variables, not prose)
                let has_math_chars = b_trimmed.chars().any(|c| {
                    matches!(c, '=' | '+' | '−' | '-' | '×' | '·' | '(' | ')')
                        || text_cleanup::is_math_char(c)
                        || text_cleanup::is_math_italic_unicode(c)
                });
                let prose_words = b_trimmed
                    .split_whitespace()
                    .filter(|w| w.len() >= 4 && w.chars().all(|c| c.is_alphabetic()))
                    .count();
                if !has_math_chars || prose_words >= 2 {
                    continue;
                }

                // Merge: expand formula bbox and mark fragment as consumed
                let b_bbox = b.bbox;
                blocks[fi].bbox[0] = blocks[fi].bbox[0].min(b_bbox[0]);
                blocks[fi].bbox[1] = blocks[fi].bbox[1].min(b_bbox[1]);
                blocks[fi].bbox[2] = blocks[fi].bbox[2].max(b_bbox[2]);
                blocks[fi].bbox[3] = blocks[fi].bbox[3].max(b_bbox[3]);
                block_consumed[bi] = true;
            }
        }
    }

    // Collect formula bboxes to exclude from text extraction
    let formula_bboxes: Vec<[f32; 4]> = blocks
        .iter()
        .enumerate()
        .zip(block_is_formula.iter())
        .filter(|((_, _), is_f)| **is_f)
        .map(|((_, b), _)| b.bbox)
        .collect();

    // Pass 2: extract text and classify, excluding formula regions from text blocks
    for (block_idx, block) in blocks.iter().enumerate() {
        // Skip blocks that were merged into a formula
        if block_consumed[block_idx] {
            continue;
        }
        let is_formula = block_is_formula[block_idx];

        // For non-formula blocks, filter out body chars that fall within
        // formula bboxes to prevent duplication.
        let extraction_chars: std::borrow::Cow<[PdfChar]> = if !is_formula && !formula_bboxes.is_empty() {
            let filtered: Vec<PdfChar> = body_chars
                .iter()
                .filter(|c| {
                    let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
                    let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
                    !formula_bboxes.iter().any(|fb| {
                        cx >= fb[0] && cx <= fb[2] && cy >= fb[1] && cy <= fb[3]
                    })
                })
                .cloned()
                .collect();
            std::borrow::Cow::Owned(filtered)
        } else {
            std::borrow::Cow::Borrowed(&body_chars[..])
        };

        let mut extracted = text::extract_region_text(
            &extraction_chars,
            block.bbox,
            page_height_pt,
            &[],
            &[],
            text::AssemblyMode::Reflow,
        );

        // Apply shared text cleanup (drop caps, ligatures, InDesign, dedup)
        extracted = text_cleanup::clean_block_text(&extracted);

        let trimmed = extracted.trim();
        // Skip empty/tiny blocks, but NOT if they're in a formula zone
        // (they might be the left-hand side of a display formula like "f_i =")
        let block_center_y = (block.bbox[1] + block.bbox[3]) / 2.0;
        let in_fzone = formula_zones.iter().any(|&(yt, yb)| block_center_y >= yt && block_center_y <= yb);
        if !in_fzone {
            if trimmed.len() < MIN_TEXT_LEN {
                continue;
            }
            if is_formula_fragment(trimmed) {
                continue;
            }
        } else if trimmed.is_empty() {
            continue;
        }

        // Classification
        let mut kind = if is_formula {
            RegionKind::DisplayFormula
        } else {
            classify_block(block, trimmed, headings, page_height_pt, page_median_font, has_font_headings)
        };
        let mut formula_tag = None;

        if is_formula {
            let (formula_text, tag) = text_cleanup::extract_formula_tag(&extracted);
            extracted = formula_text;
            formula_tag = tag;
        } else if kind == RegionKind::Text {
            // Re-check: might be formula after char filtering changed the text
            if text_cleanup::is_likely_formula_text(trimmed) {
                let (formula_text, tag) = text_cleanup::extract_formula_tag(&extracted);
                extracted = formula_text;
                formula_tag = tag;
                kind = RegionKind::DisplayFormula;
            } else {
                // Algorithm detection: line numbers (e.g., "1:", "2:") are
                // a definitive structural signal, even for 1-2 line blocks.
                // Pseudocode legitimately contains math variables — the
                // distinction is structure, not content.
                // Check for algorithm line numbers (e.g., "3:", "15:") anywhere
                // in the text — they may not be at the start of a line because
                // extracted text often lacks proper line breaks.
                let has_line_numbers = text_cleanup::has_algorithm_line_number(trimmed);

                let is_algorithm = if has_line_numbers {
                    true
                } else if block.line_count >= 3 {
                    let has_math_markers = trimmed.contains('$')
                        || trimmed.chars().any(|c| text_cleanup::is_math_italic_unicode(c));
                    !has_math_markers && crate::output::code_score(trimmed) >= 3
                } else {
                    false
                };
                if is_algorithm {
                    let preserved = text::extract_region_text(
                        &extraction_chars,
                        block.bbox,
                        page_height_pt,
                        &[],
                        &[],
                        text::AssemblyMode::PreserveLayout,
                    );
                    extracted = text_cleanup::clean_block_text(&preserved);
                    kind = RegionKind::Algorithm;
                }
            }
        }

        // Algorithm caption split: if an Algorithm block starts with
        // "Algorithm N <title>", split into a FigureTitle caption and
        // an Algorithm body. The caption is everything before the first
        // line number "N:".
        if kind == RegionKind::Algorithm {
            let lower_ext = extracted.to_lowercase();
            if lower_ext.starts_with("algorithm ") {
                if let Some(split_pos) = find_first_line_number_pos(&extracted) {
                    let caption_text = extracted[..split_pos].trim().to_string();
                    let body_text = extracted[split_pos..].trim().to_string();

                    // Emit caption
                    let caption_id = format!("p{}_{}_cap", page_idx + 1, block_idx);
                    regions.push(Region {
                        id: caption_id,
                        kind: RegionKind::FigureTitle,
                        bbox: block.bbox,
                        confidence: 1.0,
                        order: 0,
                        text: Some(caption_text),
                        html: None, latex: None, image_path: None,
                        caption: None, chart_type: None, tag: None,
                        items: None, formula_source: None, ocr_confidence: None,
                        consumed: false,
                    });

                    // Emit body (re-extract with preserved layout, strip caption)
                    let preserved = text::extract_region_text(
                        &extraction_chars,
                        block.bbox,
                        page_height_pt,
                        &[], &[],
                        text::AssemblyMode::PreserveLayout,
                    );
                    // Strip the caption line from the preserved text.
                    // The caption is the first line; body starts at the first
                    // line beginning with a digit (the line number).
                    let body_preserved = {
                        let mut found = false;
                        let mut pos = 0;
                        for (i, line) in preserved.split('\n').enumerate() {
                            if i > 0 && line.trim_start().starts_with(|c: char| c.is_ascii_digit()) {
                                found = true;
                                break;
                            }
                            pos += line.len() + 1; // +1 for \n
                        }
                        if found {
                            text_cleanup::clean_block_text(preserved[pos..].trim())
                        } else {
                            text_cleanup::clean_block_text(&preserved)
                        }
                    };
                    let body_id = format!("p{}_{}", page_idx + 1, block_idx);
                    regions.push(Region {
                        id: body_id,
                        kind: RegionKind::Algorithm,
                        bbox: block.bbox,
                        confidence: 1.0,
                        order: 0,
                        text: Some(body_preserved),
                        html: None, latex: None, image_path: None,
                        caption: None, chart_type: None, tag: None,
                        items: None, formula_source: None, ocr_confidence: None,
                        consumed: false,
                    });
                    continue; // skip normal region push
                }
            }
        }

        // Embedded caption splitting: Text blocks on figure-dense pages
        // can contain captions merged with sub-panel labels and prose from
        // adjacent columns. Split them out as separate FigureTitle regions.
        if kind == RegionKind::Text {
            let splits = split_embedded_captions(&extracted);
            if !splits.is_empty() {
                // Compute per-segment bboxes from character positions.
                // Each segment corresponds to a visual line or group of lines.
                // Find chars within the block and group them by which segment
                // their text contributes to.
                let seg_bboxes = compute_split_bboxes(
                    &extraction_chars,
                    block.bbox,
                    &splits,
                    page_height_pt,
                );

                for (seg_idx, (seg_text, seg_kind)) in splits.into_iter().enumerate() {
                    let seg_id = format!("p{}_{}_{}", page_idx + 1, block_idx, seg_idx);
                    let seg_bbox = seg_bboxes.get(seg_idx).copied().unwrap_or(block.bbox);
                    regions.push(Region {
                        id: seg_id,
                        kind: seg_kind,
                        bbox: seg_bbox,
                        confidence: 1.0,
                        order: 0,
                        text: Some(seg_text),
                        html: None, latex: None, image_path: None,
                        caption: None, chart_type: None, tag: None,
                        items: None, formula_source: None, ocr_confidence: None,
                        consumed: false,
                    });
                }
                continue; // skip normal region push
            }
        }

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
            tag: formula_tag,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
        });
    }

    // Step 5: Deduplicate overlapping DisplayFormula regions.
    // The two-pass classification can produce duplicate formula regions when
    // overlapping blocks both get classified as DisplayFormula.
    dedup_overlapping_formulas(&mut regions);

    // Step 6: Reading order via XY-Cut
    if regions.len() > 1 {
        reading_order::xy_cut_order(&mut regions);
        regions.sort_by_key(|r| r.order);
    }

    regions
}

// ── Convert + filter ────────────────────────────────────────────────

fn convert_and_filter(
    chars: &[PdfChar],
    page_height_pt: f32,
    watermarks: &HashSet<String>,
) -> Vec<ImgChar> {
    // Build a set of char indices to skip (watermark chars)
    let watermark_indices = find_watermark_char_indices(chars, watermarks);

    chars
        .iter()
        .enumerate()
        .filter_map(|(idx, c)| {
            if c.codepoint.is_control() || c.codepoint == '\u{FEFF}' {
                return None;
            }
            if watermark_indices.contains(&idx) {
                return None;
            }

            let y1 = c.bbox[1];
            let y2 = c.bbox[3];
            let x1 = c.bbox[0];
            let x2 = c.bbox[2];

            if (x2 - x1).abs() < 0.1 || (y2 - y1).abs() < 0.1 {
                return None;
            }

            let cy = (y1 + y2) / 2.0;
            if cy < MARGIN_PT || cy > page_height_pt - MARGIN_PT {
                return None;
            }

            Some(ImgChar {
                bbox: [x1, y1, x2, y2],
                font_size: c.font_size,
                is_bold: c.is_bold,
                codepoint: c.codepoint,
            })
        })
        .collect()
}

/// Find char indices that belong to watermark strings.
fn find_watermark_char_indices(chars: &[PdfChar], watermarks: &HashSet<String>) -> HashSet<usize> {
    if watermarks.is_empty() {
        return HashSet::new();
    }

    let mut skip = HashSet::new();
    let mut i = 0;
    while i < chars.len() {
        // Try to match any watermark starting at position i
        for wm in watermarks {
            let wm_chars: Vec<char> = wm.chars().collect();
            let mut j = i;
            let mut wi = 0;
            let mut matched_indices = Vec::new();

            while j < chars.len() && wi < wm_chars.len() {
                let c = &chars[j];
                if c.codepoint.is_control() || c.codepoint == '\u{FEFF}' {
                    j += 1;
                    continue;
                }
                if c.codepoint == wm_chars[wi] {
                    matched_indices.push(j);
                    wi += 1;
                    j += 1;
                } else if c.codepoint == ' ' && wi > 0 {
                    // Allow spaces between watermark chars
                    matched_indices.push(j);
                    j += 1;
                } else {
                    break;
                }
            }

            if wi == wm_chars.len() {
                // Full watermark matched — mark all matched indices for removal
                for &mi in &matched_indices {
                    skip.insert(mi);
                }
                i = j;
                break;
            }
        }
        i += 1;
    }
    skip
}

// ── Line grouping ───────────────────────────────────────────────────

fn group_into_lines(chars: &[ImgChar]) -> Vec<Line> {
    if chars.is_empty() {
        return Vec::new();
    }

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

    let avg_height: f32 =
        chars.iter().map(|c| c.bbox[3] - c.bbox[1]).sum::<f32>() / chars.len() as f32;
    let y_threshold = avg_height * 0.5;

    let mut lines = Vec::new();
    let mut line_start = 0;
    let mut line_y_min = chars[sorted[0]].bbox[1];
    let mut line_y_max = chars[sorted[0]].bbox[3];

    for i in 1..sorted.len() {
        let c = &chars[sorted[i]];
        let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
        let pad = y_threshold * 0.6;

        if cy >= line_y_min - pad && cy <= line_y_max + pad {
            line_y_min = line_y_min.min(c.bbox[1]);
            line_y_max = line_y_max.max(c.bbox[3]);
        } else {
            lines.push(build_line(chars, &sorted[line_start..i]));
            line_start = i;
            line_y_min = c.bbox[1];
            line_y_max = c.bbox[3];
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

    // Assemble rough text from codepoints sorted by X position.
    // Insert spaces at large gaps (> median char width).
    let mut x_sorted: Vec<usize> = indices.to_vec();
    x_sorted.sort_by(|&a, &b| {
        chars[a].bbox[0]
            .partial_cmp(&chars[b].bbox[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut text = String::with_capacity(x_sorted.len());
    let mut prev_x_end = f32::MIN;
    let avg_char_w = if x_sorted.len() > 1 {
        (x_max - x_min) / x_sorted.len() as f32
    } else {
        5.0
    };
    for &i in &x_sorted {
        let c = &chars[i];
        if prev_x_end > f32::MIN && (c.bbox[0] - prev_x_end) > avg_char_w * 0.5 {
            text.push(' ');
        }
        text.push(c.codepoint);
        prev_x_end = c.bbox[2];
    }

    Line {
        y_center: y_sum / indices.len() as f32,
        x_min,
        x_max,
        font_size: median_font,
        is_bold: bold_count > indices.len() / 2,
        bbox: [x_min, y_min, x_max, y_max],
        text,
    }
}

// ── Block grouping ──────────────────────────────────────────────────

/// Group lines into blocks, with heading Y-positions forcing block breaks.
///
/// `heading_ys` are Y-centers of detected headings in **image space** (Y-down).
/// When a line matches a heading Y, it becomes its own block (split from
/// surrounding text).
fn group_into_blocks(lines: &[Line], heading_ys: &[f32]) -> Vec<Block> {
    if lines.is_empty() {
        return Vec::new();
    }

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

    let avg_line_width: f32 =
        lines.iter().map(|l| l.x_max - l.x_min).sum::<f32>() / lines.len() as f32;
    let avg_char_width = avg_line_width / 40.0;
    let avg_line_height: f32 =
        lines.iter().map(|l| l.bbox[3] - l.bbox[1]).sum::<f32>() / lines.len() as f32;

    // Pre-mark which lines match a heading Y-position
    let line_is_heading: Vec<bool> = lines
        .iter()
        .map(|line| {
            heading_ys.iter().any(|&hy| (line.y_center - hy).abs() <= avg_line_height)
        })
        .collect();

    // Detect algorithm zones: Y-ranges containing numbered pseudocode lines.
    // Line numbers ("1:", "15:", "37:") are an unambiguous structural signal —
    // display formulas NEVER have them. Lines inside algorithm zones should NOT
    // be detected as formula, even if they contain math italic chars.
    let line_has_number: Vec<bool> = lines
        .iter()
        .map(|line| {
            let t = line.text.trim();
            // Check if any whitespace-separated token is "N:" (1-3 digits + colon)
            text_cleanup::has_algorithm_line_number(t)
        })
        .collect();

    // Build algorithm zones from clusters of numbered lines.
    // If ≥2 numbered lines exist within close Y-proximity, the Y-range
    // between them (plus padding) is an algorithm zone.
    let algo_zones: Vec<(f32, f32)> = {
        let numbered_ys: Vec<f32> = lines
            .iter()
            .zip(line_has_number.iter())
            .filter(|(_, has)| **has)
            .map(|(l, _)| l.y_center)
            .collect();
        if numbered_ys.len() >= 2 {
            // Single zone spanning all numbered lines, with padding
            let y_min = numbered_ys.iter().cloned().fold(f32::MAX, f32::min);
            let y_max = numbered_ys.iter().cloned().fold(f32::MIN, f32::max);
            let pad = avg_line_height * 2.0;
            vec![(y_min - pad, y_max + pad)]
        } else {
            Vec::new()
        }
    };

    // Pre-mark which lines look like display formulas.
    // Suppressed in algorithm zones — pseudocode with math variables is
    // algorithm content, not display formulas.
    let mut line_is_formula: Vec<bool> = lines
        .iter()
        .map(|line| {
            let t = line.text.trim();
            let total = t.chars().count();
            if total < 3 {
                return false;
            }
            // Suppress formula detection inside algorithm zones
            let in_algo = algo_zones.iter().any(|&(yt, yb)| line.y_center >= yt && line.y_center <= yb);
            if in_algo {
                return false;
            }
            // Path 1: content-based heuristics
            let p1 = total >= 5 && crate::text_cleanup::is_likely_formula_text(t);
            // Path 2: font-based — math italic Unicode chars are a direct
            // font signal. Require ≥3 to avoid false positives from prose
            // with a single inline math variable.
            let math_italic = t.chars().filter(|c| crate::text_cleanup::is_math_italic_unicode(*c)).count();
            let prose_words = t.split_whitespace()
                .filter(|w| w.len() >= 4 && w.chars().all(|c| c.is_ascii_alphabetic()))
                .count();
            let p2 = math_italic >= 3 && prose_words == 0;
            p1 || p2
        })
        .collect();


    // Expand formula detection to adjacent NARROW lines with math/garbled content.
    // Multi-line formulas (fractions, subscripts) get split into multiple
    // lines at different Y levels. Each sub-line may be too short or too
    // fragmented to pass is_likely_formula_text() on its own, but if it's
    // adjacent to a detected formula line and contains math-like chars,
    // it's part of the same formula.
    //
    // Key guard: only expand to NARROW lines. Formula fragments (numerator
    // "1", denominator "Δt²", subscript "j∈F_i") are always much narrower
    // than prose lines. This prevents false positives from prose with
    // inline math or algorithm pseudocode lines.
    let max_fragment_width = avg_line_width * 0.5;
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..lines.len() {
            if line_is_formula[i] || line_is_heading[i] {
                continue;
            }
            let line_width = lines[i].x_max - lines[i].x_min;
            if line_width > max_fragment_width {
                continue; // too wide to be a formula fragment
            }
            // Check if adjacent to a formula line
            let adj_formula = (i > 0 && line_is_formula[i - 1])
                || (i + 1 < lines.len() && line_is_formula[i + 1]);
            if !adj_formula {
                continue;
            }
            // Check if this line has math-like content (garbled Unicode
            // math italic, operators, digits, or short non-prose text)
            let t = lines[i].text.trim();
            if t.is_empty() {
                continue;
            }
            let math_italic = t.chars().filter(|c| crate::text_cleanup::is_math_italic_unicode(*c)).count();
            let math_syms = t.chars().filter(|c| crate::text_cleanup::is_math_char(*c)).count();
            let has_digits = t.chars().any(|c| c.is_ascii_digit());
            let prose_words = t.split_whitespace()
                .filter(|w| w.len() >= 4 && w.chars().all(|c| c.is_ascii_alphabetic()))
                .count();
            // A line with no ASCII letters at all (pure symbols, braces,
            // operators) is formula structure when adjacent to a formula.
            let no_ascii_alpha = !t.chars().any(|c| c.is_ascii_alphabetic());
            // Math content, digits, or pure symbols + no prose → fragment
            if (math_italic >= 1 || math_syms >= 1 || has_digits || no_ascii_alpha) && prose_words == 0 {
                line_is_formula[i] = true;
                changed = true;
            }
        }
    }

    let mut blocks = Vec::new();
    let mut block_start = 0;

    for i in 1..lines.len() {
        let prev = &lines[i - 1];
        let curr = &lines[i];

        // Suppress heuristic breaks inside algorithm zones. Algorithm
        // pseudocode has subscript fragments, varying indentation, and
        // font size changes that trigger y_break/x_break/font_break —
        // but these are all within one logical algorithm block. Line
        // numbers are the structural signal; formatting heuristics
        // should not override them.
        let both_in_algo = algo_zones.iter().any(|&(yt, yb)| {
            prev.y_center >= yt && prev.y_center <= yb
                && curr.y_center >= yt && curr.y_center <= yb
        });

        // Subscript/superscript fragments: tiny lines (≤3 chars) at a
        // different Y are attached to their parent line, not separate
        // content. They should never cause heuristic breaks.
        let prev_chars = prev.text.trim().chars().count();
        let curr_chars = curr.text.trim().chars().count();
        let either_is_fragment = prev_chars <= 3 || curr_chars <= 3;

        let y_gap = (curr.y_center - prev.y_center).abs();
        let y_break = !both_in_algo && !either_is_fragment
            && para_threshold > 0.0 && y_gap > para_threshold;

        let x_overlap = prev.x_max.min(curr.x_max) - prev.x_min.max(curr.x_min);
        let x_break = !both_in_algo && !either_is_fragment
            && x_overlap < -avg_char_width;

        let font_ratio = if prev.font_size > 0.0 && curr.font_size > 0.0 {
            (prev.font_size / curr.font_size).max(curr.font_size / prev.font_size)
        } else {
            1.0
        };
        let font_break = !both_in_algo && !either_is_fragment
            && font_ratio > 1.3;

        // Heading break: force split before a heading line or after a heading line
        let heading_break = line_is_heading[i] || line_is_heading[i - 1];

        // Formula break: split at transitions between formula and non-formula lines.
        // This separates display formulas from surrounding prose so they become
        // their own blocks (classified as DisplayFormula, not fused into Algorithm).
        let formula_break = line_is_formula[i] != line_is_formula[i - 1];

        if y_break || x_break || font_break || heading_break || formula_break {
            blocks.push(build_block(lines, block_start, i));
            block_start = i;
        }
    }
    blocks.push(build_block(lines, block_start, lines.len()));

    blocks
}

fn build_block(lines: &[Line], start: usize, end: usize) -> Block {
    let mut bbox = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    let mut font_sizes: Vec<f32> = Vec::new();
    let mut bold_count = 0usize;
    let line_count = end - start;

    for line in &lines[start..end] {
        bbox[0] = bbox[0].min(line.bbox[0]);
        bbox[1] = bbox[1].min(line.bbox[1]);
        bbox[2] = bbox[2].max(line.bbox[2]);
        bbox[3] = bbox[3].max(line.bbox[3]);
        font_sizes.push(line.font_size);
        if line.is_bold {
            bold_count += 1;
        }
    }

    bbox[0] -= 1.0;
    bbox[1] -= 1.0;
    bbox[2] += 1.0;
    bbox[3] += 1.0;

    font_sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_font = font_sizes[font_sizes.len() / 2];

    Block {
        bbox,
        line_count,
        font_size: median_font,
        is_bold: bold_count > line_count / 2,
        is_topmost: false,
    }
}

// ── Algorithm caption splitting ─────────────────────────────────────

/// Find the byte position of the first "N:" line-number token in text.
/// Returns the position of the digit, suitable for splitting caption from body.
fn find_first_line_number_pos(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if !b.is_ascii_digit() {
            continue;
        }
        // Must be at word boundary (start of text or preceded by whitespace)
        if i > 0 && !bytes[i - 1].is_ascii_whitespace() {
            continue;
        }
        // Find the colon after 1-3 digits
        let rest = &text[i..];
        if let Some(colon) = rest.find(':') {
            if colon >= 1 && colon <= 3 && rest[..colon].chars().all(|c| c.is_ascii_digit()) {
                return Some(i);
            }
        }
    }
    None
}

// ── Formula fragment detection ──────────────────────────────────────

fn is_formula_fragment(text: &str) -> bool {
    let total = text.chars().count();
    if total > 80 {
        return false;
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let word_letter_count: usize = words
        .iter()
        .filter(|w| w.len() >= 3 && w.chars().any(|c| c.is_alphabetic()))
        .map(|w| w.len())
        .sum();
    let word_ratio = word_letter_count as f32 / total.max(1) as f32;

    if total <= 15 {
        return word_ratio < 0.4;
    }

    word_ratio < MIN_ALPHA_RATIO
}

/// Compute per-segment bboxes after `split_embedded_captions`.
///
/// Groups chars within the block by Y-position into lines, then assigns
/// lines to segments based on cumulative character count matching each
/// segment's text length.
fn compute_split_bboxes(
    chars: &[crate::pdf::PdfChar],
    block_bbox: [f32; 4],
    segments: &[(String, crate::types::RegionKind)],
    _page_height_pt: f32,
) -> Vec<[f32; 4]> {
    // Both block_bbox and PdfChar.bbox are in image space (Y-down) after normalization.
    let block_chars: Vec<&crate::pdf::PdfChar> = chars
        .iter()
        .filter(|c| {
            let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
            let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
            cx >= block_bbox[0] - 1.0
                && cx <= block_bbox[2] + 1.0
                && cy >= block_bbox[1] - 1.0
                && cy <= block_bbox[3] + 1.0
        })
        .collect();

    if block_chars.is_empty() {
        return segments.iter().map(|_| block_bbox).collect();
    }

    // Group chars into lines by Y proximity.
    // Ascending image-space Y = top-to-bottom visual order.
    let mut lines: Vec<Vec<&crate::pdf::PdfChar>> = Vec::new();
    let mut sorted = block_chars.clone();
    sorted.sort_by(|a, b| {
        let ay = (a.bbox[1] + a.bbox[3]) / 2.0;
        let by = (b.bbox[1] + b.bbox[3]) / 2.0;
        ay.partial_cmp(&by).unwrap()
    });

    for ch in &sorted {
        let cy = (ch.bbox[1] + ch.bbox[3]) / 2.0;
        let added = lines.last_mut().and_then(|line| {
            let line_cy = (line[0].bbox[1] + line[0].bbox[3]) / 2.0;
            if (cy - line_cy).abs() < 5.0 {
                line.push(ch);
                Some(())
            } else {
                None
            }
        });
        if added.is_none() {
            lines.push(vec![ch]);
        }
    }

    let total_chars: usize = lines.iter().map(|l| l.len()).sum();

    // Assign lines to segments proportionally by text length
    let seg_char_targets: Vec<usize> = segments
        .iter()
        .map(|(text, _)| text.chars().filter(|c| !c.is_whitespace()).count())
        .collect();
    let total_seg_chars: usize = seg_char_targets.iter().sum();

    if total_seg_chars == 0 || total_chars == 0 {
        return segments.iter().map(|_| block_bbox).collect();
    }

    // Walk through lines top-to-bottom, assigning to segments in order
    let mut result = Vec::with_capacity(segments.len());
    let mut line_idx = 0;

    for (seg_i, target) in seg_char_targets.iter().enumerate() {
        let mut seg_x1 = f32::MAX;
        let mut seg_y1 = f32::MAX;
        let mut seg_x2 = f32::MIN;
        let mut seg_y2 = f32::MIN;

        let lines_for_seg = if seg_i == segments.len() - 1 {
            lines.len() - line_idx
        } else {
            let proportion = *target as f64 / total_seg_chars as f64;
            ((proportion * lines.len() as f64).round() as usize).max(1)
        };
        let chars_for_seg = lines[line_idx..].iter()
            .take(lines_for_seg)
            .map(|l| l.len())
            .sum::<usize>()
            .max(1);

        let mut seg_chars_consumed = 0;
        while line_idx < lines.len() && seg_chars_consumed < chars_for_seg {
            for ch in &lines[line_idx] {
                seg_x1 = seg_x1.min(ch.bbox[0]);
                seg_y1 = seg_y1.min(ch.bbox[1]);
                seg_x2 = seg_x2.max(ch.bbox[2]);
                seg_y2 = seg_y2.max(ch.bbox[3]);
            }
            seg_chars_consumed += lines[line_idx].len();
            line_idx += 1;
        }

        if seg_x1 < seg_x2 && seg_y1 < seg_y2 {
            result.push([seg_x1, seg_y1, seg_x2, seg_y2]);
        } else {
            result.push(block_bbox);
        }
    }

    result
}

// ── Classification ──────────────────────────────────────────────────

/// Split a Text block that contains embedded figure/table captions.
///
/// On figure-dense pages, text extraction can merge sub-panel labels,
/// captions, and prose from adjacent columns into one block like:
///   "(a) *VBD* (b) *AVBD* Fig. 6. *A card tower held by friction...*"
///
/// This function finds caption patterns mid-text and splits the block
/// into [Text, FigureTitle, Text, FigureTitle, ...] segments.
///
/// Returns empty Vec if no splitting is needed (block starts with a
/// caption or contains none).
fn split_embedded_captions(text: &str) -> Vec<(String, RegionKind)> {
    // Find all caption pattern positions in the text.
    // We look for patterns like "Fig. N.", "Figure N.", "Table N." that
    // appear MID-text (not at position 0 — those are already handled).
    let lower = text.to_lowercase();
    let caption_prefixes: &[&str] = &[
        "figure ", "fig. ", "fig ",
        "table ", "tab. ", "tab ", "tbl. ",
    ];

    let mut match_positions: Vec<(usize, &str)> = Vec::new();
    for &prefix in caption_prefixes {
        let mut search_start = 0;
        while let Some(pos) = lower[search_start..].find(prefix) {
            let abs_pos = search_start + pos;
            search_start = abs_pos + prefix.len();

            // Must not be at the very start (handled by classify_block)
            if abs_pos == 0 {
                continue;
            }
            let prev = lower.as_bytes()[abs_pos - 1];
            if prev != b' ' && prev != b'*' && prev != b')' {
                continue;
            }

            // Verify this looks like a real caption (has number + separator)
            if text_cleanup::match_label_prefix(&lower[abs_pos..]).is_none() {
                continue;
            }

            // Reject prose references: "shown in Figure 7.", "of Table 1.",
            // "(Figure 3)", "and Figure 5" — these are inline references,
            // not standalone captions.
            if is_prose_reference(&lower, abs_pos) {
                continue;
            }

            // For mid-text matches, require stronger evidence than the
            // block-start case. The segment text after the number must have
            // either a separator (. : )) after the number OR italic markers
            // (*...*). Without these, short tails like "Figure 6 shows a"
            // (end-of-block prose) pass the length check falsely.
            let after_prefix = &lower[abs_pos + prefix.len()..];
            let has_separator = {
                let mut it = after_prefix.chars().skip_while(|c| c.is_whitespace());
                // skip digits
                let mut it = it.skip_while(|c| c.is_ascii_digit());
                // skip optional letter suffix
                let mut saw_letter = false;
                let next = it.clone().next();
                if next.map_or(false, |c| c.is_ascii_lowercase()) {
                    saw_letter = true;
                    it.next();
                }
                matches!(it.next(), Some('.') | Some(':') | Some(')'))
            };
            let has_italic = text[abs_pos..].contains('*');
            if !has_separator && !has_italic {
                continue;
            }

            match_positions.push((abs_pos, prefix));
        }
    }

    if match_positions.is_empty() {
        return Vec::new();
    }

    // Sort by position
    match_positions.sort_by_key(|(pos, _)| *pos);
    // Deduplicate overlapping matches (e.g., "fig. " and "fig " both matching)
    match_positions.dedup_by(|(a, _), (b, _)| (*a as isize - *b as isize).unsigned_abs() < 5);

    // Build segments: split at each caption position
    let mut segments: Vec<(String, RegionKind)> = Vec::new();
    let mut cursor = 0;

    for (pos, prefix) in &match_positions {
        // Text before this caption
        if *pos > cursor {
            let before = text[cursor..*pos].trim();
            if before.len() >= MIN_TEXT_LEN {
                segments.push((before.to_string(), RegionKind::Text));
            }
        }
        cursor = *pos;
    }

    // Everything from the last cursor to end — could contain one or more captions.
    // Split remaining text at each caption boundary.
    let remaining_positions: Vec<usize> = match_positions.iter().map(|(p, _)| *p).collect();
    for (i, &pos) in remaining_positions.iter().enumerate() {
        let end = if i + 1 < remaining_positions.len() {
            remaining_positions[i + 1]
        } else {
            text.len()
        };
        let caption_text = text[pos..end].trim();
        if caption_text.len() >= MIN_TEXT_LEN {
            let kind = text_cleanup::match_label_prefix(caption_text)
                .map(text_cleanup::label_to_region_kind)
                .unwrap_or(RegionKind::Text);
            segments.push((caption_text.to_string(), kind));
        }
    }

    // Only return if we actually split (produced 2+ segments with at least one caption)
    let has_caption = segments.iter().any(|(_, k)| k.is_caption());
    if segments.len() >= 2 && has_caption {
        segments
    } else {
        Vec::new()
    }
}

/// Check if a figure/table reference at `pos` in the lowercased text is a
/// prose reference (e.g., "shown in Figure 7.", "of Table 1.") rather than
/// a standalone caption.
///
/// Looks at the word(s) immediately before the match. If they're reference
/// prepositions or conjunctions, it's prose.
fn is_prose_reference(lower: &str, pos: usize) -> bool {
    // Extract the last ~20 chars before the match position
    let before = &lower[..pos];
    let before = before.trim_end();

    // Common patterns preceding prose figure/table references
    const REFERENCE_SUFFIXES: &[&str] = &[
        " in",
        " see",
        " of",
        " and",
        " or",
        " from",
        " with",
        " using",
        " than",
        "(", // "(Figure 3)"
    ];

    for suffix in REFERENCE_SUFFIXES {
        if before.ends_with(suffix) {
            return true;
        }
    }
    false
}

fn classify_block(
    block: &Block,
    text: &str,
    headings: &[&DetectedHeading],
    page_height_pt: f32,
    _page_median_font: f32,
    has_font_headings: bool,
) -> RegionKind {
    let line_count = block.line_count;

    // Both block.bbox and heading.y_center are in image space (Y-down)
    let block_center_y = (block.bbox[1] + block.bbox[3]) / 2.0;
    let block_top_y = block.bbox[1];
    let avg_line_height = (block.bbox[3] - block.bbox[1]) / line_count.max(1) as f32;

    for heading in headings {
        let near_center = (heading.y_center - block_center_y).abs() <= avg_line_height * 1.5;
        let near_top = (heading.y_center - block_top_y).abs() <= avg_line_height * 2.0;
        if !near_center && !near_top {
            continue;
        }
        if titles_match_strict(&heading.title, text) {
            // When font-based headings are active, only classify as heading
            // if the block is a single line. Multi-line blocks that happen
            // to start with heading text are body blocks with fused heading
            // chars — the font IS the definitive signal, and body-font text
            // should never be absorbed into a heading region.
            if has_font_headings && line_count > 2 {
                continue;
            }
            if block.is_topmost && line_count == 1 && !titles_match_exact(&heading.title, text) {
                return RegionKind::Text;
            }
            return RegionKind::ParagraphTitle;
        }
    }

    // Caption/label detection using shared label constants
    if let Some(prefix) = text_cleanup::match_label_prefix(text) {
        return text_cleanup::label_to_region_kind(prefix);
    }

    // Note: font-size-only heading heuristic removed. With heading-first
    // font-based partitioning, real headings are already extracted before
    // block classification. The font-size heuristic caused false positives
    // (e.g., short body text fragments classified as headings).

    RegionKind::Text
}

fn titles_match_strict(heading_title: &str, block_text: &str) -> bool {
    let h = normalize_title(heading_title);
    let b = normalize_title(block_text);

    if h.is_empty() || b.is_empty() {
        return false;
    }
    if h == b {
        return true;
    }
    // Block text starts with the heading title — match regardless of suffix.
    // This catches both:
    // - Running headers with page numbers: "1.1 Introduction 42"
    // - Heading fused with body text: "1 INTRODUCTION Physics-based simulation..."
    if h.len() >= 5 && b.starts_with(&h) {
        return true;
    }
    false
}

fn titles_match_exact(heading_title: &str, block_text: &str) -> bool {
    normalize_title(heading_title) == normalize_title(block_text)
}

fn normalize_title(s: &str) -> String {
    let s = s.trim().trim_start_matches("**").trim_end_matches("**").trim();
    s.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// is_caption_text removed — now uses text_cleanup::match_label_prefix()

/// Remove duplicate DisplayFormula regions with overlapping bboxes.
/// When two formula regions overlap vertically by >50%, keep the one with
/// the more reasonable height (closer to single-line formula height ~15-25pt).
fn dedup_overlapping_formulas(regions: &mut Vec<Region>) {
    let mut to_remove: Vec<usize> = Vec::new();

    for i in 0..regions.len() {
        if regions[i].kind != RegionKind::DisplayFormula || regions[i].consumed {
            continue;
        }
        for j in (i + 1)..regions.len() {
            if regions[j].kind != RegionKind::DisplayFormula || regions[j].consumed {
                continue;
            }
            // Check vertical overlap
            let a = regions[i].bbox;
            let b = regions[j].bbox;
            let overlap_top = a[1].max(b[1]);
            let overlap_bot = a[3].min(b[3]);
            if overlap_bot <= overlap_top {
                continue; // no vertical overlap
            }
            let overlap_h = overlap_bot - overlap_top;
            let min_h = (a[3] - a[1]).min(b[3] - b[1]);
            if min_h <= 0.0 {
                continue;
            }
            if overlap_h / min_h > 0.5 {
                // Also require horizontal overlap — don't merge formulas
                // from different columns that happen to overlap in Y.
                let h_overlap_left = a[0].max(b[0]);
                let h_overlap_right = a[2].min(b[2]);
                if h_overlap_right <= h_overlap_left {
                    continue; // no horizontal overlap → different columns
                }

                // Overlapping — MERGE bboxes into region i, remove j.
                // This preserves the full extent of fragmented formulas
                // where multiple overlapping blocks each capture part.
                regions[i].bbox[0] = a[0].min(b[0]);
                regions[i].bbox[1] = a[1].min(b[1]);
                regions[i].bbox[2] = a[2].max(b[2]);
                regions[i].bbox[3] = a[3].max(b[3]);
                // Keep whichever tag is non-empty
                if regions[i].tag.is_none() && regions[j].tag.is_some() {
                    regions[i].tag = regions[j].tag.clone();
                }
                to_remove.push(j);
            }
        }
    }

    to_remove.sort();
    to_remove.dedup();
    for &idx in to_remove.iter().rev() {
        regions.remove(idx);
    }
}

// ── Running header filter (cross-page) ──────────────────────────────

/// Filter running headers across pages.
///
/// Uses topmost/bottommost block position + aggressive normalization.
/// Only consumes blocks that are the topmost or bottommost on their page.
pub fn filter_running_headers(pages: &mut [crate::types::Page]) {
    if pages.len() < 4 {
        return;
    }

    // Pass 1: Collect topmost and bottommost candidates
    let mut header_occurrences: HashMap<String, Vec<(usize, usize, bool)>> = HashMap::new();
    // (page_idx, region_idx, is_topmost)

    for (page_i, page) in pages.iter().enumerate() {
        if page.regions.is_empty() {
            continue;
        }

        // Find topmost region (smallest bbox[1])
        if let Some((ri, r)) = page
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.consumed && r.text.as_ref().map_or(false, |t| t.len() <= 150))
            .min_by(|(_, a), (_, b)| {
                a.bbox[1]
                    .partial_cmp(&b.bbox[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            let key = normalize_header_aggressive(r.text.as_deref().unwrap_or(""));
            if key.len() >= 4 {
                header_occurrences
                    .entry(key)
                    .or_default()
                    .push((page_i, ri, true));
            }
        }

        // Find bottommost region (largest bbox[3])
        if let Some((ri, r)) = page
            .regions
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.consumed && r.text.as_ref().map_or(false, |t| t.len() <= 150))
            .max_by(|(_, a), (_, b)| {
                a.bbox[3]
                    .partial_cmp(&b.bbox[3])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        {
            let key = normalize_header_aggressive(r.text.as_deref().unwrap_or(""));
            if key.len() >= 4 {
                header_occurrences
                    .entry(key)
                    .or_default()
                    .push((page_i, ri, false));
            }
        }
    }

    // Pass 2: Mark as consumed if ≥4 pages
    let header_patterns: HashSet<String> = header_occurrences
        .iter()
        .filter(|(_, occ)| occ.len() >= 4)
        .map(|(key, _)| key.clone())
        .collect();

    for (key, occurrences) in &header_occurrences {
        if occurrences.len() >= 4 {
            for &(page_i, region_i, _) in occurrences {
                pages[page_i].regions[region_i].consumed = true;
            }
        }
    }

    // Pass 3: Strip fused headers from non-consumed blocks
    // If a block's text starts with a known header pattern, strip the prefix.
    if !header_patterns.is_empty() {
        strip_fused_headers(pages, &header_patterns);
    }
}

/// Aggressive normalization for running header matching.
/// Strips bold/italic markers, ALL digits, punctuation, and collapses whitespace.
fn normalize_header_aggressive(text: &str) -> String {
    let s = text
        .replace("**", "")
        .replace('*', "");
    s.chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Strip fused running headers from the beginning of block text.
fn strip_fused_headers(pages: &mut [crate::types::Page], patterns: &HashSet<String>) {
    for page in pages.iter_mut() {
        for region in page.regions.iter_mut() {
            if region.consumed {
                continue;
            }
            let text = match region.text.as_ref() {
                Some(t) if t.len() > 50 => t,
                _ => continue,
            };

            let normalized = normalize_header_aggressive(text);
            for pattern in patterns {
                if pattern.len() < 4 {
                    continue;
                }
                if normalized.starts_with(pattern.as_str()) {
                    // Find where the pattern ends in the original text
                    // by counting how many alpha chars we need to skip
                    let pattern_alpha_count = pattern.chars().filter(|c| c.is_alphabetic()).count();
                    let mut alpha_seen = 0;
                    let mut cut_pos = 0;
                    for (i, ch) in text.char_indices() {
                        if ch.is_alphabetic() {
                            alpha_seen += 1;
                            if alpha_seen >= pattern_alpha_count {
                                cut_pos = i + ch.len_utf8();
                                break;
                            }
                        }
                    }
                    if cut_pos > 0 {
                        let remainder = text[cut_pos..].trim_start();
                        if remainder.len() >= 50 {
                            region.text = Some(remainder.to_string());
                            break;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper to make ImgChars ──

    fn make_char(x: f32, y: f32, w: f32, h: f32) -> ImgChar {
        ImgChar {
            bbox: [x, y, x + w, y + h],
            font_size: 10.0,
            is_bold: false,
            codepoint: 'x',
        }
    }

    // ── Column gap detection ──

    #[test]
    fn test_column_gap_two_columns() {
        // Simulate two-column layout: left [50,290] right [320,560]
        let mut chars = Vec::new();
        for y in (0..60).map(|i| 50.0 + i as f32 * 10.0) {
            // Left column chars
            for x in (0..24).map(|i| 50.0 + i as f32 * 10.0) {
                chars.push(make_char(x, y, 6.0, 10.0));
            }
            // Right column chars
            for x in (0..24).map(|i| 320.0 + i as f32 * 10.0) {
                chars.push(make_char(x, y, 6.0, 10.0));
            }
        }
        let gaps = detect_column_gaps(&chars, 612.0);
        assert!(!gaps.is_empty(), "Should detect column gap");
        let gap = &gaps[0];
        assert!(gap.midpoint > 280.0 && gap.midpoint < 330.0,
            "Gap midpoint should be ~305, got {}", gap.midpoint);
    }

    #[test]
    fn test_column_gap_single_column() {
        // Simulate single-column layout: [50, 400]
        let mut chars = Vec::new();
        for y in (0..60).map(|i| 50.0 + i as f32 * 10.0) {
            for x in (0..35).map(|i| 50.0 + i as f32 * 10.0) {
                chars.push(make_char(x, y, 6.0, 10.0));
            }
        }
        let gaps = detect_column_gaps(&chars, 500.0);
        assert!(gaps.is_empty(), "Should NOT detect gap for single column");
    }

    #[test]
    fn test_column_gap_with_spanning_title() {
        // Two columns with a full-width title at the top
        let mut chars = Vec::new();
        // Title line spanning full width (y=50)
        for x in (0..50).map(|i| 50.0 + i as f32 * 10.0) {
            chars.push(make_char(x, 50.0, 6.0, 12.0));
        }
        // Left column body (y=80..600)
        for y in (0..52).map(|i| 80.0 + i as f32 * 10.0) {
            for x in (0..24).map(|i| 50.0 + i as f32 * 10.0) {
                chars.push(make_char(x, y, 6.0, 10.0));
            }
        }
        // Right column body
        for y in (0..52).map(|i| 80.0 + i as f32 * 10.0) {
            for x in (0..24).map(|i| 320.0 + i as f32 * 10.0) {
                chars.push(make_char(x, y, 6.0, 10.0));
            }
        }
        let gaps = detect_column_gaps(&chars, 612.0);
        // Title fills a few Y-slices but body has clear gap
        assert!(!gaps.is_empty(), "Should detect gap even with spanning title");
    }

    // ── Column splitting with spanning ──

    #[test]
    fn test_split_spanning_vs_column() {
        let gap = ColumnGap { midpoint: 305.0, start: 290.0, end: 320.0 };

        // Spanning line at y=100: chars flowing through the gap region.
        // Left side ends at x=286 (center=289 < gap.start=290), right edge at 292.
        // Right side starts at x=295 (center=298, inside gap), left edge at 295.
        // inter_char_gap = 295 - 292 = 3, gap_width * 0.6 = 18. 3 < 18 → spanning
        let mut chars = Vec::new();
        for x in (0..24).map(|i| 50.0 + i as f32 * 10.0) {
            chars.push(make_char(x, 100.0, 6.0, 10.0));
        }
        // Last left char: bbox=[286,100,292,110], center_x=289 < gap.start=290
        chars.push(make_char(286.0, 100.0, 6.0, 10.0));
        // First right char: bbox=[295,100,301,110], center_x=298 which is inside gap
        // So it won't be counted in either left (center < gap.start) or right (center > gap.end)
        // We need a char with center > gap.end=320:
        chars.push(make_char(318.0, 100.0, 6.0, 10.0)); // center=321 > 320
        for x in (0..23).map(|i| 330.0 + i as f32 * 10.0) {
            chars.push(make_char(x, 100.0, 6.0, 10.0));
        }
        // left_max_x = 292 (right edge of char at x=286)
        // right_min_x = 318 (left edge of char at x=318)
        // inter_char_gap = 318 - 292 = 26, gap_width * 0.6 = 18. 26 > 18 → NOT spanning!
        // For a true spanning line, we need chars INSIDE the gap.
        // Let's add a char IN the gap:
        chars.push(make_char(300.0, 100.0, 6.0, 10.0)); // center=303, inside gap
        // Now: left has chars, right has chars, and there's a char bridging.
        // But the algorithm checks left_max_x (rightmost char BEFORE gap) vs right_min_x (leftmost char AFTER gap).
        // The char at 300 has center=303 which is between start(290) and end(320), so it's neither left nor right.
        // left_max_x = 292, right_min_x = 318. Gap = 26 > 18.
        // The spanning detection needs the chars to be CLOSE, not just present.
        // A real spanning title has chars continuously across the gap with normal word spacing.
        // Let's make a proper spanning line with normal char spacing through the gap:

        // Actually, let me just create a spanning line properly:
        let mut spanning_chars = Vec::new();
        // Title spanning full width: chars every 10pt from x=50 to x=550
        for x in (0..50).map(|i| 50.0 + i as f32 * 10.0) {
            spanning_chars.push(make_char(x, 100.0, 6.0, 10.0));
        }
        // Column line at y=200: separate columns with large gap
        for x in (0..24).map(|i| 50.0 + i as f32 * 10.0) {
            spanning_chars.push(make_char(x, 200.0, 6.0, 10.0));
        }
        for x in (0..24).map(|i| 330.0 + i as f32 * 10.0) {
            spanning_chars.push(make_char(x, 200.0, 6.0, 10.0));
        }

        let (spanning, columns) = split_chars_into_columns(&spanning_chars, &[gap]);
        assert_eq!(columns.len(), 2, "Should have 2 column groups");
        assert!(!columns[0].is_empty(), "Left column should have chars from y=200");
        assert!(!columns[1].is_empty(), "Right column should have chars from y=200");
        assert!(!spanning.is_empty(), "Title at y=100 should be spanning (chars every 10pt through gap)");
    }

    #[test]
    fn test_split_edge_char_not_spanning() {
        // Regression: right-column char at the gap boundary should NOT trigger
        // spanning. The gap [295, 319] overshoots into the right column, and
        // a right-column char `(` at bbox [317.7, 287, 320.3, 299] (cx=319.0)
        // would previously trigger false spanning for the entire Y-band.
        let gap = ColumnGap { midpoint: 307.0, start: 295.0, end: 319.0 };

        let mut chars = Vec::new();
        // Left column chars at y=290: x from 50 to 280 (every 10pt, width 6)
        for i in 0..24 {
            chars.push(make_char(50.0 + i as f32 * 10.0, 290.0, 6.0, 10.0));
        }
        // Right column chars at y=290: first char near gap boundary, then regular
        chars.push(ImgChar {
            bbox: [317.7, 287.0, 320.3, 299.4], // cx=319.0, exactly at gap.end
            font_size: 10.0,
            is_bold: false,
            codepoint: '(',
        });
        for i in 1..24 {
            chars.push(make_char(320.0 + i as f32 * 10.0, 290.0, 6.0, 10.0));
        }

        let (spanning, columns) = split_chars_into_columns(&chars, &[gap]);
        assert!(
            spanning.is_empty(),
            "Edge char at cx=319 should NOT trigger spanning; got {} spanning chars",
            spanning.len()
        );
        assert!(!columns[0].is_empty(), "Left column should have chars");
        assert!(!columns[1].is_empty(), "Right column should have chars");
    }

    #[test]
    fn test_split_true_spanning_still_detected() {
        // A full-width title with chars flowing continuously through the gap
        // should still be detected as spanning.
        let gap = ColumnGap { midpoint: 307.0, start: 295.0, end: 319.0 };

        let mut chars = Vec::new();
        // Spanning title at y=100: chars every 8pt from x=50 to x=530
        // Max char gap = 8pt - 6pt(width) = 2pt between chars. gap_width*0.5 = 12.
        // 2 < 12 → spanning.
        for i in 0..60 {
            chars.push(make_char(50.0 + i as f32 * 8.0, 100.0, 6.0, 10.0));
        }
        // Column lines at y=200: separate columns with large gap
        for i in 0..24 {
            chars.push(make_char(50.0 + i as f32 * 10.0, 200.0, 6.0, 10.0));
        }
        for i in 0..24 {
            chars.push(make_char(330.0 + i as f32 * 10.0, 200.0, 6.0, 10.0));
        }

        let (spanning, columns) = split_chars_into_columns(&chars, &[gap]);
        assert!(
            !spanning.is_empty(),
            "Full-width title should be detected as spanning"
        );
        assert!(!columns[0].is_empty(), "Left column at y=200 should have chars");
        assert!(!columns[1].is_empty(), "Right column at y=200 should have chars");
    }

    // ── Running header normalization ──

    #[test]
    fn test_normalize_header_strips_digits() {
        assert_eq!(
            normalize_header_aggressive("**128 4 Convex optimization problems**"),
            "convex optimization problems"
        );
    }

    #[test]
    fn test_normalize_header_strips_italic() {
        assert_eq!(
            normalize_header_aggressive("*7.5. BRDF Theory* 223"),
            "brdf theory"
        );
    }

    #[test]
    fn test_normalize_header_strips_section_numbers() {
        assert_eq!(
            normalize_header_aggressive("1.1 Mathematical optimization 3"),
            "mathematical optimization"
        );
    }

    #[test]
    fn test_normalize_header_strips_chapter_prefix() {
        assert_eq!(
            normalize_header_aggressive("6 CHAPTER 1. INTRODUCTION"),
            "chapter introduction"
        );
    }

    #[test]
    fn test_normalize_header_handles_spaced_text() {
        assert_eq!(
            normalize_header_aggressive("**12** C h a p t e r 2 . F u n d a m e n t a l s"),
            "c h a p t e r f u n d a m e n t a l s"
        );
    }

    #[test]
    fn test_normalize_header_consistency_across_pages() {
        // Even/odd page variants should normalize to same string
        let even = normalize_header_aggressive("**34 2 Convex sets**");
        let odd = normalize_header_aggressive("**2.1 Affine and convex sets 35**");
        // These are different sections, so they should be different
        assert_ne!(even, odd);

        // Same section on different pages should match
        let p1 = normalize_header_aggressive("**128 4 Convex optimization problems**");
        let p2 = normalize_header_aggressive("**130 4 Convex optimization problems**");
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_normalize_header_watermark() {
        assert_eq!(
            normalize_header_aggressive("6ROG WR PDWWPJ #JPDLO FRP"),
            "rog wr pdwwpj jpdlo frp"
        );
    }

    #[test]
    fn test_normalize_header_short_rejected() {
        // Normalized text < 4 chars should be empty enough to skip
        let n = normalize_header_aggressive("**3**");
        assert!(n.is_empty() || n.len() < 4);
    }

    // ── Formula fragment detection ──

    #[test]
    fn test_formula_fragment_short_math() {
        assert!(is_formula_fragment("k n"));
        assert!(is_formula_fragment("= +"));
        assert!(is_formula_fragment("≤ ≥ ∈"));
    }

    #[test]
    fn test_formula_fragment_long_prose_not_filtered() {
        assert!(!is_formula_fragment("This is a normal paragraph with more than enough words to be real text content"));
    }

    // Caption detection tests are in text_cleanup::tests (shared module)

    // ── Title matching ──

    #[test]
    fn test_titles_match_exact() {
        assert!(titles_match_strict("1.1 Introduction", "1.1 Introduction"));
        assert!(titles_match_strict("**1.1 Introduction**", "1.1 Introduction"));
    }

    #[test]
    fn test_titles_match_with_page_number() {
        // Running header: heading title + page number suffix
        assert!(titles_match_strict("1.1 Introduction", "1.1 Introduction 42"));
    }

    #[test]
    fn test_titles_dont_match_different_content() {
        assert!(!titles_match_strict("1.1 Introduction", "1.2 Background and related work"));
    }

    // ── Heading block splitting ──

    #[test]
    fn test_heading_break_splits_block() {
        // When a heading Y is provided, it should force a block break,
        // so the heading line becomes its own block separate from body text.
        let mut lines = Vec::new();
        // Line 0: heading at y=100
        lines.push(Line {
            y_center: 100.0, x_min: 50.0, x_max: 250.0,
            font_size: 12.0, is_bold: true, bbox: [50.0, 95.0, 250.0, 105.0],
            text: "1 Introduction".into(),
        });
        // Line 1: body text at y=115 (close, no Y-gap break)
        lines.push(Line {
            y_center: 115.0, x_min: 50.0, x_max: 250.0,
            font_size: 10.0, is_bold: false, bbox: [50.0, 110.0, 250.0, 120.0],
            text: "This is body text about the topic".into(),
        });
        // Line 2: body text at y=130
        lines.push(Line {
            y_center: 130.0, x_min: 50.0, x_max: 250.0,
            font_size: 10.0, is_bold: false, bbox: [50.0, 125.0, 250.0, 135.0],
            text: "More body text continues here".into(),
        });

        // Without heading_ys: all 3 lines would be 1 block
        let blocks_no_heading = group_into_blocks(&lines, &[]);
        assert_eq!(blocks_no_heading.len(), 1, "Without headings: should be 1 block");

        // With heading at y=100: heading line becomes its own block
        let blocks_with_heading = group_into_blocks(&lines, &[100.0]);
        assert!(blocks_with_heading.len() >= 2,
            "With heading at y=100: should be ≥2 blocks, got {}",
            blocks_with_heading.len());
        // First block should be just the heading line
        assert_eq!(blocks_with_heading[0].line_count, 1,
            "First block (heading) should be 1 line");
    }

    // ── Code detection (Algorithm classification) ──

    #[test]
    fn test_code_score_python_sage() {
        let code = "def compute(self, x):\n    result = 0\n    for i in range(len(x)):\n        result += x[i]\n    return result\nfrom numpy import array";
        let score = crate::output::code_score(code);
        assert!(score >= 3, "Python code should score >= 3, got {score}");
    }

    #[test]
    fn test_code_score_c_code() {
        let code = "int max(int* a, int n) {\n    int m = a[0];\n    for (int i = 1; i < n; i++) {\n        if (a[i] > m) m = a[i];\n    }\n    return m;\n}";
        assert!(crate::output::code_score(code) >= 3, "C code should score >= 3");
    }

    #[test]
    fn test_code_score_glsl() {
        let code = "uniform mat4 uModelViewMatrix;\nuniform mat4 uProjectionMatrix;\nvoid main() {\n    gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPos, 1.0);\n}";
        assert!(crate::output::code_score(code) >= 3, "GLSL should score >= 3");
    }

    #[test]
    fn test_code_score_prose_not_code() {
        let prose = "In this chapter we will introduce the key concepts of convex optimization and discuss their applications in signal processing and control systems.";
        assert!(crate::output::code_score(prose) < 3, "Prose should NOT score as code");
    }

    #[test]
    fn test_code_score_with_assignments_and_loops() {
        // Code with assignments, loops, and function calls should score high
        let code = "int max(int* a, int n) {\n    int m = a[0];\n    for (int i = 1; i < n; i++) {\n        if (a[i] > m) m = a[i];\n    }\n    return m;\n}\nvoid sort(int* arr, int n) {\n    for (int i = 0; i < n; i++) {};\n}";
        let score = crate::output::code_score(code);
        assert!(score >= 3, "C code with loops and braces should score >= 3, got {score}");
    }
}
