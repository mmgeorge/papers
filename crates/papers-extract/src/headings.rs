//! Font-based heading hierarchy extraction.
//!
//! Analyzes a PDF's text layer to derive a heading hierarchy using font family,
//! rendered glyph height, and character frequency as signals.

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::pdf::PdfChar;

// ── Types ───────────────────────────────────────────────────────────

/// Height bucket key: rendered glyph height rounded to nearest 0.5pt,
/// stored as `(height * 2).round()` to avoid float comparison issues.
type HeightBucket = i32;

/// A font group identified by normalized family name + em-size bucket.
///
/// We group by em_size (nominal font size) rather than rendered glyph height
/// because glyph heights vary within the same font run (caps vs lowercase,
/// ascenders vs descenders, drop caps). Em-size is consistent within a run.
/// Rendered height is computed per-group for ranking purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FontGroupKey {
    pub family: String,
    /// Em-size bucket: `(font_size * 2).round()` — 0.5pt bins.
    pub size_bucket: HeightBucket,
}

/// Statistics for a font group across the document.
#[derive(Debug, Clone, Serialize)]
pub struct FontGroupStats {
    /// Normalized font family name (empty for Type3 fonts).
    pub font: String,
    /// Em-size (nominal font size) in points.
    pub em_size: f32,
    /// Median rendered glyph height in points (from bounding boxes).
    pub height: f32,
    /// Total character count across all pages.
    pub char_count: usize,
    /// Number of text segments using this font group.
    pub segment_count: usize,
    /// Role assigned during analysis: "body", "heading_1", "heading_2", "math", "other".
    pub role: String,
    /// Sample text from this font group (first ~60 chars), or null for math fonts.
    pub sample: Option<String>,
    /// Up to 3 original (un-normalized) font names seen in this group.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub raw_font_names: Vec<String>,
}

/// A text segment: consecutive chars on the same line with the same font group.
#[derive(Debug, Clone)]
pub struct TextSegment {
    pub text: String,
    pub font_group: FontGroupKey,
    /// 0-indexed page number.
    pub page: u32,
    /// Center Y in PDF coordinates (Y-up, higher = higher on page).
    pub y_center: f32,
    /// Leftmost X coordinate.
    pub x_left: f32,
    /// Rightmost X coordinate.
    pub x_right: f32,
    /// Median rendered glyph height across chars in this segment.
    pub median_height: f32,
    /// Page height in PDF points (for margin calculations).
    pub page_height: f32,
    /// Original (un-normalized) font name from the first char.
    pub raw_font_name: String,
}

/// A detected heading with assigned depth.
#[derive(Debug, Clone, Serialize)]
pub struct DetectedHeading {
    /// Heading depth: 1 = chapter/top-level, 2 = section, 3 = subsection, etc.
    pub depth: u32,
    /// Heading text content.
    pub title: String,
    /// 1-indexed page number.
    pub page: u32,
    /// Total characters contained under this heading (until next same-or-shallower heading).
    pub contained_chars: usize,
    /// Y-center position for internal ordering (not serialized).
    #[serde(skip_serializing)]
    y_center: f32,
    /// Right edge X coordinate for adjacency checks (not serialized).
    #[serde(skip_serializing)]
    x_right: f32,
}

/// Font profile: body text identification and heading level assignments.
#[derive(Debug, Clone, Serialize)]
pub struct FontProfile {
    pub body: FontProfileEntry,
    pub heading_levels: Vec<HeadingLevelEntry>,
    /// Font groups that were candidates but skipped (single-page, too few chars, etc.).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub skipped: Vec<SkippedFontEntry>,
    /// True when font names are available (false for all-Type3 documents).
    pub has_font_names: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FontProfileEntry {
    pub font: String,
    pub height: f32,
    pub char_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct HeadingLevelEntry {
    pub depth: u32,
    pub font: String,
    pub height: f32,
    pub char_count: usize,
    /// Number of detected heading instances at this level.
    pub instances: usize,
    /// Number of distinct pages this font group appears on.
    pub pages: usize,
}

/// A font group that was considered as a heading candidate but skipped.
#[derive(Debug, Clone, Serialize)]
pub struct SkippedFontEntry {
    pub font: String,
    pub height: f32,
    pub char_count: usize,
    pub pages: usize,
    pub reason: String,
}

/// Complete heading extraction result.
#[derive(Debug, Clone, Serialize)]
pub struct HeadingExtractionResult {
    /// Complete font group frequency table (all groups, sorted by char_count descending).
    pub font_groups: Vec<FontGroupStats>,
    /// Derived font profile (body + heading levels).
    pub font_profile: FontProfile,
    /// Detected headings in document order.
    pub headings: Vec<DetectedHeading>,
}

// ── Math/Symbol Font Detection ──────────────────────────────────────

/// Broad check for math, symbol, and decorative fonts that should not be
/// considered heading candidates.
///
/// This is more aggressive than `pdf::is_math_font` (which only gates italic
/// detection). For heading extraction, we want to exclude all non-text fonts.
fn is_non_text_font(name: &str) -> bool {
    if name.is_empty() {
        return false; // Type3 fonts — can't classify by name
    }
    let n = name.to_ascii_uppercase();

    // TeX math/symbol fonts
    if n.starts_with("CMMI")   // Computer Modern Math Italic
        || n.starts_with("CMSY")  // Computer Modern Symbols
        || n.starts_with("CMEX")  // Computer Modern Extensions
        || n.starts_with("LMMI")  // Latin Modern Math Italic
        || n.starts_with("LMSY")  // Latin Modern Symbols
        || n.starts_with("NEWPXMI")  // New PX Math Italic
        || n.starts_with("PXMI")  // PX Math Italic (including pxmiaX)
        || n.starts_with("PXSY")  // PX Symbols
        || n.starts_with("TXEX")  // TX Extensions (txexs)
        || n.starts_with("TXSY")  // TX Symbols (txsym)
        || n.starts_with("PXEX")  // PX Extensions
        || n.starts_with("TXMI")  // TX Math Italic
        || n.starts_with("MSAM")  // AMS Math Symbols A
        || n.starts_with("MSBM")  // AMS Math Symbols B
        || n.starts_with("EUSM")  // Euler Script Medium
        || n.starts_with("EUFM")  // Euler Fraktur Medium
        || n.starts_with("RSFS")  // Ralph Smith Formal Script
        || n.starts_with("STMARY") // St Mary's Road symbols
        || n.starts_with("WASY")  // wasy symbols
        || n.starts_with("LCIRCLE") // LaTeX circle font
        || n.starts_with("LINE")  // LaTeX line font
    {
        return true;
    }

    // Font name contains math/symbol keywords
    if n.contains("MATH")
        || n.contains("SYMBOL")
        || n.contains("DINGBAT")
        || n.contains("ARROW")
    {
        return true;
    }

    // PX/TX system fonts (used in pxfonts/txfonts packages)
    if n.starts_with("PXSYS")  // pxsys — PX system symbols (braces, radicals)
        || n.starts_with("TXSYS")  // txsys
    {
        return true;
    }

    // Known decorative fonts that aren't text
    if n == "ZAPFDINGBATS" || n.starts_with("WINGDING") {
        return true;
    }

    false
}

// ── Font Family Normalization ───────────────────────────────────────

/// Strip PDF subset prefix and normalize a font name to its base family.
///
/// Rules:
/// 1. Strip 6-uppercase-letter subset prefix followed by '+' (e.g., "BMKCHC+CMR10" → "CMR10")
/// 2. Strip trailing size digits for TeX fonts (e.g., "CMR10" → "CMR", "CMSSBX10" → "CMSSBX")
/// 3. Strip weight/style suffixes for OpenType fonts (e.g., "OpenSans-Bold" → "OpenSans")
/// 4. Empty string (Type3 fonts) passes through as empty.
pub fn normalize_font_family(font_name: &str) -> String {
    if font_name.is_empty() {
        return String::new();
    }

    let mut name = font_name;

    // Step 1: Strip subset prefix "ABCDEF+" (exactly 6 uppercase ASCII + '+')
    if name.len() > 7 && name.as_bytes()[6] == b'+' {
        let prefix = &name[..6];
        if prefix.chars().all(|c| c.is_ascii_uppercase()) {
            name = &name[7..];
        }
    }

    // Step 2: Check for TeX-style font names (all alphanumeric, ending in digits)
    // Examples: CMR10 → CMR, CMSS17 → CMSS, CMSSBX10 → CMSSBX, LCIRCLE10 → LCIRCLE
    if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric()) {
        // Find where trailing digits start
        let last_alpha = name.rfind(|c: char| !c.is_ascii_digit());
        if let Some(pos) = last_alpha {
            let alpha_part = &name[..=pos];
            let digit_part = &name[pos + 1..];
            // Only strip if there ARE trailing digits and the alpha part is non-empty
            if !digit_part.is_empty() && !alpha_part.is_empty() {
                // TeX fonts are typically all-uppercase (CMR, CMSS, etc.)
                // But also handle mixed case like "Calibri" (no trailing digits → won't match)
                if alpha_part.chars().all(|c| c.is_ascii_uppercase()) {
                    return alpha_part.to_string();
                }
            }
        }
    }

    // Step 3: Strip TeX-style weight/style single-letter suffixes.
    // Many TeX fonts use trailing T (text), I (italic), B (bold), Z (small caps):
    //   LinLibertineT → LinLibertine, LinLibertineTBI → LinLibertine
    //   LinBiolinumT → LinBiolinum
    // Only strip if preceded by a lowercase letter (to avoid stripping from all-caps names).
    {
        let bytes = name.as_bytes();
        let mut end = bytes.len();
        while end > 0 && matches!(bytes[end - 1], b'T' | b'I' | b'B' | b'Z') {
            end -= 1;
        }
        if end > 0 && end < bytes.len() && bytes[end - 1].is_ascii_lowercase() {
            let stripped = &name[..end];
            if stripped.len() >= 3 {
                return stripped.to_string();
            }
        }
    }

    // Step 4: Strip OpenType weight/style suffixes
    // Try hyphenated suffixes first (most specific), then bare suffixes
    let style_suffixes = [
        // Hyphenated (most common in OpenType)
        "-BoldItalic",
        "-BoldOblique",
        "-SemiboldItalic",
        "-DemiItalic",
        "-ExtraBold",
        "-Bold",
        "-Semibold",
        "-Demi",
        "-Medium",
        "-Light",
        "-Black",
        "-Thin",
        "-UltraLight",
        "-Italic",
        "-Oblique",
        "-Regular",
        "-Regu",
        "-Roman",
        "-Roma",
        "-Book",
        "-BookItalic",
        // PostScript-style suffixes (no hyphen)
        "PSMT",
        "PSMTBold",
        "PS-BoldMT",
        "PS-BoldItalicMT",
        "PS-ItalicMT",
    ];

    let mut result = name.to_string();
    for suffix in &style_suffixes {
        if let Some(stripped) = result.strip_suffix(suffix) {
            if !stripped.is_empty() {
                result = stripped.to_string();
                break;
            }
        }
    }

    result
}

// ── Size Bucketing ──────────────────────────────────────────────────

/// Convert a size (em_size or height) to a bucket key (0.5pt bins).
fn size_to_bucket(size: f32) -> HeightBucket {
    (size * 2.0).round() as HeightBucket
}

/// Convert a bucket key back to a representative size.
fn bucket_to_size(bucket: HeightBucket) -> f32 {
    bucket as f32 / 2.0
}

// ── Phase 1: Build Text Segments ────────────────────────────────────

/// Build text segments from PdfChars for a single page.
///
/// A segment is a run of consecutive non-control characters on the same line
/// (similar Y position) with the same font group (normalized family + height bucket).
pub fn build_segments(chars: &[PdfChar], page_idx: u32, page_height: f32) -> Vec<TextSegment> {
    let mut segments: Vec<TextSegment> = Vec::new();

    // Previous character's right edge for word-boundary detection.
    let mut prev_right: Option<f32> = None;
    let mut prev_threshold: f32 = 0.0;

    for c in chars {
        if c.codepoint.is_control() {
            continue;
        }

        // Track spaces for word-boundary detection but don't add them as chars
        if c.codepoint == ' ' {
            // Explicit space resets gap tracking — next char starts fresh
            if let Some(last) = segments.last_mut() {
                last.text.push(' ');
            }
            prev_right = None;
            continue;
        }

        let glyph_height = c.bbox[3] - c.bbox[1]; // rendered height from bbox
        if glyph_height <= 0.0 {
            continue;
        }

        // Group by em-size (font_size) when available; fall back to rendered glyph
        // height for Type3 fonts that report em_size=0.
        let family = normalize_font_family(&c.font_name);
        let bucket = if c.font_size > 0.0 {
            size_to_bucket(c.font_size)
        } else {
            size_to_bucket(glyph_height)
        };
        let key = FontGroupKey {
            family: family.clone(),
            size_bucket: bucket,
        };

        let y_center = (c.bbox[1] + c.bbox[3]) / 2.0;

        // Try to extend the current segment
        if let Some(last) = segments.last_mut() {
            if last.font_group == key && last.page == page_idx {
                // Check if on the same line: Y centers within tolerance
                let y_tol = last.median_height.max(glyph_height) * 0.6;
                if (last.y_center - y_center).abs() < y_tol {
                    // Check for word boundary: insert space if gap exceeds threshold.
                    // For Type3 fonts (font_size=0), space_threshold is 0, so fall back
                    // to a fraction of the rendered glyph height as the threshold.
                    if let Some(pr) = prev_right {
                        let gap = c.bbox[0] - pr;
                        let threshold = if prev_threshold > 0.01 {
                            prev_threshold
                        } else {
                            // Fallback: ~30% of glyph height / 2 ≈ 15% of height
                            glyph_height * 0.15
                        };
                        if gap > threshold && gap > 0.0 {
                            last.text.push(' ');
                        }
                    }
                    last.text.push(c.codepoint);
                    if c.bbox[2] > last.x_right {
                        last.x_right = c.bbox[2];
                    }
                    prev_right = Some(c.bbox[2]);
                    prev_threshold = c.space_threshold;
                    continue;
                }
            }
        }

        // Start a new segment
        segments.push(TextSegment {
            text: c.codepoint.to_string(),
            font_group: key,
            page: page_idx,
            y_center,
            x_left: c.bbox[0],
            x_right: c.bbox[2],
            median_height: glyph_height,
            page_height,
            raw_font_name: c.font_name.clone(),
        });
        prev_right = Some(c.bbox[2]);
        prev_threshold = c.space_threshold;
    }

    segments
}

// ── Phase 2: Font Frequency Table ───────────────────────────────────

/// Build the complete font frequency table from all segments.
///
/// Returns all font groups sorted by char_count descending.
pub fn build_font_frequency_table(segments: &[TextSegment]) -> Vec<FontGroupStats> {
    struct GroupAccum {
        char_count: usize,
        segment_count: usize,
        raw_names: Vec<String>,
        sample: Option<String>,
        /// Collect rendered heights to compute median.
        rendered_heights: Vec<f32>,
    }

    let mut groups: HashMap<FontGroupKey, GroupAccum> = HashMap::new();

    for seg in segments {
        let entry = groups.entry(seg.font_group.clone()).or_insert(GroupAccum {
            char_count: 0,
            segment_count: 0,
            raw_names: Vec::new(),
            sample: None,
            rendered_heights: Vec::new(),
        });
        entry.char_count += seg.text.chars().count();
        entry.segment_count += 1;
        entry.rendered_heights.push(seg.median_height);
        if entry.raw_names.len() < 3 && !entry.raw_names.contains(&seg.raw_font_name) {
            entry.raw_names.push(seg.raw_font_name.clone());
        }
        if entry.sample.is_none() && !seg.text.trim().is_empty() {
            let sample = if seg.text.len() > 60 {
                format!("{}...", &seg.text[..57])
            } else {
                seg.text.clone()
            };
            entry.sample = Some(sample);
        }
    }

    let mut stats: Vec<FontGroupStats> = groups
        .into_iter()
        .map(|(key, mut acc)| {
            let is_math = acc.raw_names.iter().any(|n| is_non_text_font(n));
            // Compute median rendered height
            acc.rendered_heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_h = if acc.rendered_heights.is_empty() {
                bucket_to_size(key.size_bucket)
            } else {
                acc.rendered_heights[acc.rendered_heights.len() / 2]
            };
            FontGroupStats {
                font: key.family.clone(),
                em_size: bucket_to_size(key.size_bucket),
                height: median_h,
                char_count: acc.char_count,
                segment_count: acc.segment_count,
                role: if is_math {
                    "math".to_string()
                } else {
                    "other".to_string()
                },
                sample: if is_math { None } else { acc.sample },
                raw_font_names: acc.raw_names,
            }
        })
        .collect();

    stats.sort_by(|a, b| b.char_count.cmp(&a.char_count));
    stats
}

/// Identify the body text font group (highest char count, excluding math fonts).
pub fn identify_body_text(stats: &[FontGroupStats]) -> Option<FontGroupKey> {
    stats
        .iter()
        .find(|s| s.role != "math")
        .map(|s| FontGroupKey {
            family: s.font.clone(),
            size_bucket: size_to_bucket(s.em_size),
        })
}

// ── Phase 3: Filter Heading Candidates ──────────────────────────────

/// Check if a segment is in the page margin (top/bottom 5%).
fn is_in_margin(seg: &TextSegment) -> bool {
    // Only filter bottom margin (footers, page numbers).
    // Top-of-page running headers are handled by filter_repeated_headings instead,
    // since real headings often appear near the very top of the page and a
    // percentage/fixed margin clips them (e.g., y_center can even exceed page_height
    // due to coordinate rounding in some PDFs).
    let margin = 25.0;
    seg.y_center < margin
}

/// Check if text looks like a figure/table caption.
fn is_caption_text(text: &str) -> bool {
    let t = text.trim();
    t.starts_with("Figure ")
        || t.starts_with("Fig. ")
        || t.starts_with("Fig.")
        || t.starts_with("Table ")
        || t.starts_with("TABLE ")
        || t.starts_with("Algorithm ")
        || t.starts_with("Listing ")
}

/// Filter segments to heading candidates.
///
/// Two-pass approach:
/// 1. Collect segments that are taller than body (regardless of font family).
/// 2. If no taller candidates exist (AVBD case), fall back to different-family
///    segments ranked by frequency.
pub fn filter_heading_candidates<'a>(
    segments: &'a [TextSegment],
    body_key: &FontGroupKey,
    body_height: f32,
) -> Vec<&'a TextSegment> {
    let basic_filter = |seg: &&TextSegment| -> bool {
        // Must not be a math/symbol font
        if is_non_text_font(&seg.raw_font_name) {
            return false;
        }
        // Must not be in page margin
        if is_in_margin(seg) {
            return false;
        }
        // Must not be a caption
        if is_caption_text(&seg.text) {
            return false;
        }
        // Must be reasonably short (headings aren't paragraphs)
        if seg.text.chars().count() > 200 {
            return false;
        }
        // Must have meaningful text content (not just symbols/punctuation/short labels)
        let meaningful_chars = seg.text.chars().filter(|c| c.is_alphanumeric()).count();
        if meaningful_chars < 3 {
            return false;
        }
        // Must be predominantly alphabetic — filters out figure axis tick labels
        // ("0 2 4 6 8 10") which use heading-sized fonts but are all digits/spaces.
        let total_non_space = seg.text.chars().filter(|c| !c.is_whitespace()).count();
        if total_non_space > 0 {
            let alpha_chars = seg.text.chars().filter(|c| c.is_alphabetic()).count();
            if (alpha_chars as f32) / (total_non_space as f32) < 0.5 {
                return false;
            }
        }
        // Characters must not be too spread out — catches diagram labels like
        // "9 10 11 (getToken( ), installID ( ))" where individual tokens are
        // scattered across a figure. For continuous text, average width per char
        // is ~0.5× font height. Reject if average exceeds 4× font height.
        // Characters must not be too spread out — catches diagram labels like
        // "9 10 11 (getToken( ), installID ( ))" where individual tokens are
        // scattered across a figure. For continuous text, average width per char
        // is ~0.45× font height. Scattered diagram labels average ~0.75×.
        // Threshold 0.65× separates continuous headings from spread-out labels.
        let char_count = seg.text.chars().count();
        if char_count > 1 {
            let width = seg.x_right - seg.x_left;
            let avg_width_per_char = width / char_count as f32;
            if avg_width_per_char > seg.median_height * 0.65 {
                return false;
            }
        }
        true
    };

    // Include any segment that is:
    // - Different font family from body (heading font signal, works for AVBD too), OR
    // - Same font family but taller (body font used at larger size for titles)
    //
    // The MAX_HEADING_LEVELS cap in assign_heading_levels ensures noise fonts
    // (code, math-adjacent) get truncated — they rank low by height.
    segments
        .iter()
        .filter(|seg| {
            if !basic_filter(seg) {
                return false;
            }
            let diff_family = seg.font_group.family != body_key.family;
            let taller = seg.median_height > body_height * 1.1;
            diff_family || taller
        })
        .collect()
}

// ── Phase 4: Assign Heading Levels ──────────────────────────────────

/// Result of heading level assignment: level map + page counts + skipped groups.
pub struct HeadingLevelAssignment {
    /// Map from FontGroupKey to depth (1 = chapter/top-level, 2 = section, etc.).
    pub level_map: HashMap<FontGroupKey, u32>,
    /// Number of distinct pages each font group appears on (keyed by FontGroupKey).
    pub page_counts: HashMap<FontGroupKey, usize>,
    /// Font groups that were candidates but got filtered out.
    pub skipped: Vec<SkippedFontEntry>,
}

/// Assign heading depth levels based on font group characteristics.
pub fn assign_heading_levels(
    candidates: &[&TextSegment],
    body_height: f32,
    total_pages: usize,
) -> HeadingLevelAssignment {
    // Collect unique font groups with their stats
    struct GroupAccum {
        char_count: usize,
        segment_count: usize,
        median_height: f32,
        pages: Vec<u32>, // distinct pages this group appears on
    }
    let mut group_stats: HashMap<FontGroupKey, GroupAccum> = HashMap::new();
    for seg in candidates {
        let entry = group_stats
            .entry(seg.font_group.clone())
            .or_insert(GroupAccum {
                char_count: 0,
                segment_count: 0,
                median_height: seg.median_height,
                pages: Vec::new(),
            });
        entry.char_count += seg.text.chars().count();
        entry.segment_count += 1;
        if !entry.pages.contains(&seg.page) {
            entry.pages.push(seg.page);
        }
    }

    if group_stats.is_empty() {
        return HeadingLevelAssignment {
            level_map: HashMap::new(),
            page_counts: HashMap::new(),
            skipped: Vec::new(),
        };
    }

    // Track page counts for all candidate groups (before filtering).
    let page_counts: HashMap<FontGroupKey, usize> = group_stats
        .iter()
        .map(|(k, acc)| (k.clone(), acc.pages.len()))
        .collect();

    let mut skipped = Vec::new();

    // Minimum number of distinct pages a font group must appear on to be
    // considered a structural heading. For short documents (<50 pages),
    // even 2 pages is meaningful. For books (>=50 pages), require at least
    // 4 pages to filter out front-matter, copyright, and dedication pages.
    let min_pages = if total_pages < 50 { 2 } else { 4 };

    // Minimum number of candidate segment instances for a font group.
    // Filters out font groups used for rare one-off elements (figure labels,
    // code snippets) that happen to pass other filters.
    let min_instances = if total_pages < 50 { 5 } else { 10 };

    // Split groups into taller-than-body and shorter-than-body.
    let mut taller: Vec<(FontGroupKey, usize, f32)> = Vec::new();
    let mut shorter: Vec<(FontGroupKey, usize, f32)> = Vec::new();

    for (k, acc) in group_stats {
        if acc.pages.len() < min_pages {
            skipped.push(SkippedFontEntry {
                font: k.family.clone(),
                height: acc.median_height,
                char_count: acc.char_count,
                pages: acc.pages.len(),
                reason: format!("too few pages (<{min_pages})"),
            });
            continue;
        }
        if acc.segment_count < min_instances {
            skipped.push(SkippedFontEntry {
                font: k.family.clone(),
                height: acc.median_height,
                char_count: acc.char_count,
                pages: acc.pages.len(),
                reason: format!("too few instances ({}<{min_instances})", acc.segment_count),
            });
            continue;
        }
        if acc.median_height > body_height {
            taller.push((k, acc.char_count, acc.median_height));
        } else {
            shorter.push((k, acc.char_count, acc.median_height));
        }
    }

    // Taller groups: rank by height descending, tiebreak by fewer chars.
    taller.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    // Shorter groups (AVBD case): rank by fewer chars = higher level.
    shorter.sort_by(|a, b| a.1.cmp(&b.1));

    // Merge: taller groups first (they're the most visually prominent),
    // then shorter groups (different-family headings below body size).
    let mut sorted = taller;
    sorted.extend(shorter);

    // Filter out groups with very few total chars (e.g., single part numbers
    // like "I", "II" that eat up heading levels). Keep groups with >= 10 chars.
    for entry in sorted.iter().filter(|(_, count, _)| *count < 10) {
        skipped.push(SkippedFontEntry {
            font: entry.0.family.clone(),
            height: entry.2,
            char_count: entry.1,
            pages: page_counts.get(&entry.0).copied().unwrap_or(0),
            reason: "too few chars (<10)".to_string(),
        });
    }
    sorted.retain(|(_, count, _)| *count >= 10);

    // Cap at MAX_HEADING_LEVELS to avoid noise from minor font variants.
    const MAX_HEADING_LEVELS: usize = 10;
    if sorted.len() > MAX_HEADING_LEVELS {
        for entry in &sorted[MAX_HEADING_LEVELS..] {
            skipped.push(SkippedFontEntry {
                font: entry.0.family.clone(),
                height: entry.2,
                char_count: entry.1,
                pages: page_counts.get(&entry.0).copied().unwrap_or(0),
                reason: format!("exceeded max levels ({MAX_HEADING_LEVELS})"),
            });
        }
        sorted.truncate(MAX_HEADING_LEVELS);
    }

    // Assign sequential depths
    let mut level_map = HashMap::new();
    for (i, (key, _, _)) in sorted.iter().enumerate() {
        level_map.insert(key.clone(), (i as u32) + 1);
    }

    HeadingLevelAssignment {
        level_map,
        page_counts,
        skipped,
    }
}

// ── Phase 5: Merge and Cross-Validate ───────────────────────────────

/// Merge adjacent heading segments on the same line into single headings.
pub fn merge_heading_segments(
    candidates: &[&TextSegment],
    level_map: &HashMap<FontGroupKey, u32>,
) -> Vec<DetectedHeading> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort candidates by (page, y_center descending — higher Y = higher on page = earlier)
    let mut sorted: Vec<&TextSegment> = candidates.to_vec();
    sorted.sort_by(|a, b| {
        a.page
            .cmp(&b.page)
            .then(b.y_center.partial_cmp(&a.y_center).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.x_left.partial_cmp(&b.x_left).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut headings: Vec<DetectedHeading> = Vec::new();

    for seg in &sorted {
        let Some(depth) = level_map.get(&seg.font_group).copied() else {
            continue; // Font group not in level map (e.g., capped out) — skip
        };

        // Try to merge with previous heading if on the same line
        if let Some(last) = headings.last_mut() {
            let same_page = last.page == seg.page + 1; // last.page is 1-indexed
            let _y_tol = seg.median_height * 0.6;
            // We need to check Y proximity — but we've already converted to 1-indexed page
            // Use a simple heuristic: if previous heading was on same page and text is short,
            // check if they're on the same baseline
            if same_page {
                // For simplicity in first version: check if last heading title is short
                // and this segment continues on the same line
                // We stored y_center in the segment but not in DetectedHeading,
                // so we'll merge by checking if the gap between segments is small
                // For now, skip merging — handle in a future refinement
            }
        }

        headings.push(DetectedHeading {
            depth,
            title: seg.text.clone(),
            page: seg.page + 1, // 1-indexed
            contained_chars: 0,
            y_center: seg.y_center,
            x_right: seg.x_right,
        });
    }

    headings
}

/// Cross-validate heading depths using numbering patterns.
///
/// For headings with clear numbering prefixes (dotted decimal, roman numerals),
/// the numbering depth overrides the font-based depth.
pub fn cross_validate_depths(headings: &mut [DetectedHeading]) {
    for h in headings.iter_mut() {
        let numbering_depth = infer_depth_from_numbering(&h.title);
        if numbering_depth > 0 {
            h.depth = numbering_depth;
        }
    }
}

/// Infer heading depth from numbering patterns in the heading text.
///
/// Returns depth (1-based) or 0 if no numbering pattern is detected.
fn infer_depth_from_numbering(text: &str) -> u32 {
    let text = text.trim();

    if let Some(first_ch) = text.chars().next() {
        if first_ch.is_ascii_digit() || (first_ch.is_ascii_uppercase() && text.len() > 1) {
            let num_end = text
                .find(|c: char| c == ' ' || c == '\t')
                .unwrap_or(text.len());
            let prefix = &text[..num_end];

            // Single uppercase letter followed by a dot-separated number:
            // appendix section (e.g., "A.1 First Section" → depth 2).
            // We do NOT treat a bare single letter ("A Appendix") as depth 1
            // because it conflicts with article words ("A Short History...").
            // Bare letter chapters are handled by font-based depth instead.

            // Check if it's a number pattern (digits, dots, possibly starting with letter).
            // Must contain at least one digit to avoid matching article words ("A History").
            let is_number_pattern = prefix
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c.is_ascii_uppercase())
                && prefix.chars().any(|c| c.is_ascii_digit());

            if is_number_pattern {
                let trimmed = prefix.trim_end_matches('.');
                let dots = trimmed.chars().filter(|&c| c == '.').count();
                return (dots as u32) + 1;
            }
        }

        // Roman numerals
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

    0
}

fn is_roman_numeral(s: &str) -> bool {
    if s.is_empty() || s.len() > 8 {
        return false;
    }
    s.chars()
        .all(|c| matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M'))
}

/// Check if heading text starts with a structured numbering prefix.
///
/// Pattern: `<ident> <alpha-text>` where `<ident>` is one of:
///   - Digits joined by `.` or `-`: `1.1`, `1.2.3`, `A.1`, `1.0`, `B-2`, `1`
///   - A single uppercase letter: `A`, `B`
///   - A roman numeral: `I`, `II`, `III`, `IV`
///
/// The second word must start with an alphabetic character.
///
/// Used as an exception to the minimum contained_chars threshold — numbered
/// headings are kept even if they contain little text.
fn has_numbering_prefix(text: &str) -> bool {
    use std::sync::LazyLock;
    // Prefix: a single identifier token, then whitespace, then alphabetic title.
    //
    //  Identifier is one of:
    //    [A-Z0-9]+([.\-][A-Z0-9]+)*  — "1", "1.1", "A.1", "1.0", "B-2", "1.2.3"
    //    [IVXLCDM]+                   — roman numeral "III", "IV"
    //  Must contain at least one digit OR be a single letter OR be a roman numeral.
    //  Then: \s+[a-zA-Z]
    static RE: LazyLock<regex::Regex> = LazyLock::new(|| {
        regex::Regex::new(
            r"(?x)
            ^\s*
            (?:
                \d+(?:[.\-][A-Za-z0-9]+)*     # starts with digit: 1, 1.1, 1.2.3, 1.0
              | [A-Z]\d+(?:[.\-][A-Za-z0-9]+)* # letter+digit: A1, A.1, B-2
              | [A-Z][.\-][A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)* # letter.digit: A.1, B-2.3
              | [A-Z]                           # single letter: A, B
              | [IVXLCDM]{2,}                   # roman numeral 2+ chars: II, III, IV
            )
            \s+[a-zA-Z]
            ",
        )
        .unwrap()
    });
    RE.is_match(text.trim())
}

// ── Running Header/Footer Detection ─────────────────────────────────

// ── Numbering Sequence Continuity Filter ────────────────────────────

/// Parse a heading's numbering prefix into (parent, final_number).
///
/// "3.4.1 Transition Diagrams" → Some(("3.4", 1))
/// "3 Methods"                 → Some(("", 3))
/// "A.2 Proofs"                → Some(("A", 2))
/// "A.1.2 Lemma"               → Some(("A.1", 2))
/// "II Background"             → Some(("", 2))   (roman → ordinal)
/// "Convex functions"          → None
fn parse_numbering_prefix(text: &str) -> Option<(String, u32)> {
    let text = text.trim();
    let num_end = text.find(|c: char| c == ' ' || c == '\t')?;
    let prefix = &text[..num_end];

    // Try roman numeral first (before digit check, since romans have no digits)
    if is_roman_numeral(prefix) {
        let val = roman_to_u32(prefix)?;
        return Some((String::new(), val));
    }

    // Must contain at least one digit and only digits/dots/uppercase letters
    if !prefix.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }
    if !prefix
        .chars()
        .all(|c| c.is_ascii_digit() || c == '.' || c.is_ascii_uppercase())
    {
        return None;
    }
    let trimmed = prefix.trim_end_matches('.');
    if let Some(last_dot) = trimmed.rfind('.') {
        let parent = &trimmed[..last_dot];
        let tail = &trimmed[last_dot + 1..];
        let num: u32 = tail.parse().ok()?;
        Some((parent.to_string(), num))
    } else {
        // Bare number like "3" → parent is "", seq is 3
        let num: u32 = trimmed.parse().ok()?;
        Some((String::new(), num))
    }
}

/// Convert a roman numeral string to its numeric value.
fn roman_to_u32(s: &str) -> Option<u32> {
    let mut total: u32 = 0;
    let mut prev = 0u32;
    for c in s.chars().rev() {
        let val = match c {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        };
        if val < prev {
            total = total.checked_sub(val)?;
        } else {
            total += val;
        }
        prev = val;
    }
    if total == 0 { None } else { Some(total) }
}

/// Remove headings that interrupt a proven numbering sequence.
///
/// If we see headings numbered X.Y.1 ... X.Y.3 (same parent prefix, increasing
/// final number), any heading *between* them in document order that claims a
/// shallower depth must be a false positive — the numbering continuity proves
/// the hierarchy was not actually interrupted.
///
/// Example:
///   3.4.1 Transition Diagrams       [depth 3]
///   TOP HEADER                      [depth 1] ← remove: splits 3.4.1→3.4.3
///   3.4.3 Completion of the Example [depth 3]
fn filter_sequence_interrupters(headings: &mut Vec<DetectedHeading>) {
    if headings.len() < 3 {
        return;
    }

    // Collect (index, parent, seq_num, depth) for all numbered headings
    let numbered: Vec<(usize, String, u32, u32)> = headings
        .iter()
        .enumerate()
        .filter_map(|(i, h)| {
            let (parent, num) = parse_numbering_prefix(&h.title)?;
            Some((i, parent, num, h.depth))
        })
        .collect();

    // For each pair of numbered headings with the same parent where the
    // second has a higher sequence number, mark any interlopers that claim
    // a shallower depth.
    let mut to_remove: HashSet<usize> = HashSet::new();

    for (ai, (idx_a, parent_a, _num_a, depth_a)) in numbered.iter().enumerate() {
        // Find the next heading with the same parent and higher sequence number
        for (idx_b, parent_b, _num_b, depth_b) in &numbered[ai + 1..] {
            if parent_b != parent_a {
                continue;
            }
            // Same parent, later in sequence — check everything between idx_a and idx_b
            // for headings at a shallower depth (lower depth number = higher level)
            let seq_depth = (*depth_a).min(*depth_b);
            for mid_idx in (*idx_a + 1)..*idx_b {
                if to_remove.contains(&mid_idx) {
                    continue;
                }
                let mid = &headings[mid_idx];
                if mid.depth < seq_depth {
                    to_remove.insert(mid_idx);
                }
            }
            // Only need the nearest next match — further pairs are covered transitively
            break;
        }
    }

    if !to_remove.is_empty() {
        let mut idx = 0;
        headings.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
    }
}

/// Normalize heading text for repetition detection: replace digit runs with `#`.
///
/// "112.2213:4 • Chris Giles" → "#.#:# • Chris Giles"
/// "1 Introduction"           → "# Introduction"
/// "2 Background"             → "# Background"  (different from above)
fn normalize_for_repeat(text: &str) -> String {
    let mut result = String::new();
    let mut in_digits = false;
    for c in text.chars() {
        if c.is_ascii_digit() {
            if !in_digits {
                result.push('#');
                in_digits = true;
            }
        } else {
            in_digits = false;
            result.push(c);
        }
    }
    result
}

/// Filter out running headers/footers by detecting repeated heading patterns.
///
/// If the same text pattern (with digit runs normalized) appears on 3+ different
/// pages, it's a running header or footer and all instances are discarded.
fn filter_repeated_headings(headings: &mut Vec<DetectedHeading>) {
    // Count distinct pages per (normalized_pattern, depth) — only count
    // repetitions among headings at the same depth. Running headers (e.g.
    // "Introduction" at depth 10) should not cause a chapter title
    // ("Introduction" at depth 1) to be discarded.
    let mut pattern_pages: HashMap<(String, u32), HashSet<u32>> = HashMap::new();
    for h in headings.iter() {
        let norm = normalize_for_repeat(&h.title);
        pattern_pages
            .entry((norm, h.depth))
            .or_default()
            .insert(h.page);
    }

    headings.retain(|h| {
        let norm = normalize_for_repeat(&h.title);
        let page_count = pattern_pages
            .get(&(norm, h.depth))
            .map(|s| s.len())
            .unwrap_or(0);
        page_count < 3
    });
}

/// Filter out headings that have no text segments below them on the same page.
///
/// Catches footers, page numbers, and other bottom-of-page elements that
/// passed through earlier filters.
fn filter_no_content_below(headings: &mut Vec<DetectedHeading>, segments: &[TextSegment]) {
    // Collect all segment y_centers per page (0-indexed)
    let mut page_y: HashMap<u32, Vec<f32>> = HashMap::new();
    for seg in segments {
        page_y.entry(seg.page).or_default().push(seg.y_center);
    }

    headings.retain(|h| {
        let page_0 = h.page - 1;
        if let Some(ys) = page_y.get(&page_0) {
            // "Below" = lower y_center in PDF coords (Y goes up from bottom).
            // Use 2pt tolerance to avoid floating-point near-misses with
            // segments on the same baseline.
            ys.iter().any(|&y| y < h.y_center - 2.0)
        } else {
            false
        }
    });
}

/// Filter out headings that have body text immediately adjacent on the same line.
///
/// These are paragraph openers (e.g. small-caps first words) rather than
/// standalone section headings. A real heading occupies its own line.
///
/// Uses X-proximity to avoid false positives in multi-column layouts where
/// headings in one column share a Y-coordinate with body text in another.
fn filter_inline_headings(
    headings: &mut Vec<DetectedHeading>,
    segments: &[TextSegment],
    body_key: &FontGroupKey,
) {
    // Build per-page index of body text segments: (y_center, x_left)
    let mut page_body: HashMap<u32, Vec<(f32, f32)>> = HashMap::new();
    for seg in segments {
        if seg.font_group == *body_key {
            page_body
                .entry(seg.page)
                .or_default()
                .push((seg.y_center, seg.x_left));
        }
    }

    headings.retain(|h| {
        let page_0 = h.page - 1;
        if let Some(body_segs) = page_body.get(&page_0) {
            // Check if any body text segment sits on the same baseline AND
            // starts close to the heading's right edge (within ~30pt).
            // In multi-column layouts, body text in another column would be
            // hundreds of points away, not adjacent.
            let y_tol = 3.0; // ~quarter of a typical line height
            let x_gap_max = 30.0; // max gap between heading end and body start
            let has_adjacent_body = body_segs.iter().any(|&(by, bx)| {
                (by - h.y_center).abs() < y_tol
                    && bx > h.x_right
                    && (bx - h.x_right) < x_gap_max
            });
            !has_adjacent_body
        } else {
            true // no body segments on this page — keep
        }
    });
}

// ── Contained Chars Computation ─────────────────────────────────────

/// Compute how many characters of content are contained under each heading.
///
// ── Front-matter Collapse ───────────────────────────────────────────

/// Check if a heading title matches a front-matter / back-matter section name
/// whose sub-headings are noise (e.g. TOC entries, index letters).
fn is_frontmatter_heading(title: &str) -> bool {
    let t = title.trim().to_ascii_lowercase();
    // Strip a leading numbering prefix if present (e.g. "1 Contents")
    let t = if let Some(pos) = t.find(|c: char| c.is_ascii_alphabetic()) {
        t[pos..].trim()
    } else {
        return false;
    };
    matches!(
        t,
        "contents"
            | "table of contents"
            | "preface"
            | "foreword"
            | "index"
            | "subject index"
            | "author index"
            | "name index"
            | "keyword index"
            | "figures"
            | "list of figures"
            | "tables"
            | "list of tables"
            | "list of algorithms"
            | "bibliography"
            | "references"
            | "glossary"
            | "notation"
            | "list of symbols"
            | "acknowledgements"
            | "acknowledgments"
    )
}

/// Remove all sub-headings that fall under a front-matter heading.
///
/// Keeps the front-matter heading itself but drops every heading after it that
/// has a strictly deeper depth, stopping when we reach:
///   - a heading with depth ≤ the front-matter heading's depth, OR
///   - a heading with a numbering prefix (signals start of real content)
fn collapse_frontmatter_children(headings: &mut Vec<DetectedHeading>) {
    let mut to_remove: HashSet<usize> = HashSet::new();
    let mut i = 0;
    while i < headings.len() {
        if is_frontmatter_heading(&headings[i].title) {
            let fm_depth = headings[i].depth;
            // Remove everything after i that is deeper, but stop at numbered headings
            let mut j = i + 1;
            while j < headings.len()
                && headings[j].depth > fm_depth
                && parse_numbering_prefix(&headings[j].title).is_none()
                && !has_numbering_prefix(&headings[j].title)
            {
                to_remove.insert(j);
                j += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if !to_remove.is_empty() {
        let mut idx = 0;
        headings.retain(|_| {
            let keep = !to_remove.contains(&idx);
            idx += 1;
            keep
        });
    }
}

/// For heading[i] at depth d, contained_chars = total chars of all segments
/// between heading[i]'s position and the next heading[j] with depth <= d
/// (or end of document). Uses prefix sums for efficiency.
fn compute_contained_chars(headings: &mut [DetectedHeading], segments: &[TextSegment]) {
    let n = headings.len();
    if n == 0 {
        return;
    }

    // For each heading[i], find scope end: next heading j where depth[j] <= depth[i].
    let mut scope_end: Vec<usize> = vec![n; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if headings[j].depth <= headings[i].depth {
                scope_end[i] = j;
                break;
            }
        }
    }

    // Map each heading to its index in the segment stream.
    // Walk segments and headings in parallel — both are in document order
    // (page ascending, y_center descending within page).
    let mut heading_seg_idx: Vec<usize> = vec![segments.len(); n];
    let mut h_ptr = 0;

    for (s_idx, seg) in segments.iter().enumerate() {
        while h_ptr < n {
            let h = &headings[h_ptr];
            let h_page_0 = h.page - 1; // headings use 1-indexed pages
            // A segment is "at or past" a heading when it's on a later page,
            // or on the same page at the same or lower Y (same or lower on page).
            let at_or_past = seg.page > h_page_0
                || (seg.page == h_page_0 && seg.y_center <= h.y_center);
            if at_or_past {
                heading_seg_idx[h_ptr] = s_idx;
                h_ptr += 1;
            } else {
                break;
            }
        }
        if h_ptr >= n {
            break;
        }
    }

    // Build prefix sum of segment char counts for O(1) range queries.
    let mut prefix = vec![0usize; segments.len() + 1];
    for (i, seg) in segments.iter().enumerate() {
        prefix[i + 1] = prefix[i] + seg.text.chars().count();
    }

    // For each heading, contained_chars = chars from after this heading's segment
    // to before the scope-end heading's segment.
    for i in 0..n {
        let start = heading_seg_idx[i] + 1; // skip the heading segment itself
        let end = if scope_end[i] < n {
            heading_seg_idx[scope_end[i]]
        } else {
            segments.len()
        };
        if start <= end && start <= segments.len() {
            headings[i].contained_chars = prefix[end] - prefix[start];
        }
    }
}

// ── Top-Level Orchestrator ──────────────────────────────────────────

/// Extract headings from a PDF given pre-loaded pages of PdfChars.
///
/// Each element of `page_chars` is `(chars, page_height_pt)` for a page.
pub fn extract_headings(page_chars: &[(Vec<PdfChar>, f32)]) -> HeadingExtractionResult {
    // Phase 1: Build segments from all pages
    let mut all_segments = Vec::new();
    for (page_idx, (chars, page_height)) in page_chars.iter().enumerate() {
        all_segments.extend(build_segments(chars, page_idx as u32, *page_height));
    }

    // Phase 2: Build font frequency table and identify body text
    let mut font_groups = build_font_frequency_table(&all_segments);
    let body_key = identify_body_text(&font_groups);

    let (body_key, body_height) = match body_key {
        Some(key) => {
            // Use the median rendered glyph height (not em_size) for body height comparison.
            // Em-size and rendered height differ (e.g., em_size=10pt but rendered=13.8pt).
            // We compare segment rendered heights against body rendered height.
            let rendered_height = font_groups
                .iter()
                .find(|fg| fg.font == key.family && size_to_bucket(fg.em_size) == key.size_bucket)
                .map(|fg| fg.height)
                .unwrap_or(bucket_to_size(key.size_bucket));
            (key, rendered_height)
        }
        None => {
            // No non-math text found — return empty result
            return HeadingExtractionResult {
                font_groups,
                font_profile: FontProfile {
                    body: FontProfileEntry {
                        font: String::new(),
                        height: 0.0,
                        char_count: 0,
                    },
                    heading_levels: Vec::new(),
                    skipped: Vec::new(),
                    has_font_names: false,
                },
                headings: Vec::new(),
            };
        }
    };

    // Phase 3: Filter heading candidates
    let candidates = filter_heading_candidates(&all_segments, &body_key, body_height);

    // Phase 4: Assign heading levels
    let assignment = assign_heading_levels(&candidates, body_height, page_chars.len());
    let level_map = assignment.level_map;
    let page_counts = assignment.page_counts;
    let skipped = assignment.skipped;

    // Phase 5: Merge segments and build headings
    let mut headings = merge_heading_segments(&candidates, &level_map);

    // Phase 5b: Cross-validate with numbering patterns
    cross_validate_depths(&mut headings);

    // Phase 5b2: Remove headings that interrupt proven numbering sequences
    filter_sequence_interrupters(&mut headings);

    // Phase 5c: Filter running headers/footers and bottom-of-page noise
    filter_repeated_headings(&mut headings);
    filter_no_content_below(&mut headings, &all_segments);
    filter_inline_headings(&mut headings, &all_segments, &body_key);

    // Phase 5d: Compute contained chars and filter headings
    compute_contained_chars(&mut headings, &all_segments);
    // Require at least 2000 chars of content under a heading, unless the
    // heading has a well-known numbering prefix (1, 1.1, A, I, etc.).
    const MIN_CONTAINED_CHARS: usize = 2000;
    headings.retain(|h| {
        h.contained_chars >= MIN_CONTAINED_CHARS || has_numbering_prefix(&h.title)
    });

    // Phase 5e: Collapse front-matter sections (TOC, preface, index) — keep the
    // top-level heading but remove all sub-headings underneath it.
    collapse_frontmatter_children(&mut headings);

    // Update font group roles
    let body_bucket = body_key.size_bucket;
    for fg in &mut font_groups {
        let key = FontGroupKey {
            family: fg.font.clone(),
            size_bucket: size_to_bucket(fg.em_size),
        };
        if key == body_key {
            fg.role = "body".to_string();
        } else if let Some(&depth) = level_map.get(&key) {
            fg.role = format!("heading_{depth}");
        }
        // "math" and "other" roles are already set
    }

    // Check if font names are available
    let has_font_names = all_segments.iter().any(|s| !s.raw_font_name.is_empty());

    // Build font profile
    let body_stats = font_groups
        .iter()
        .find(|fg| {
            fg.font == body_key.family && size_to_bucket(fg.em_size) == body_bucket
        });

    let font_profile = FontProfile {
        body: FontProfileEntry {
            font: body_key.family.clone(),
            height: body_height,
            char_count: body_stats.map(|s| s.char_count).unwrap_or(0),
        },
        heading_levels: {
            // Count heading instances per depth
            let mut depth_counts: HashMap<u32, usize> = HashMap::new();
            for h in &headings {
                *depth_counts.entry(h.depth).or_insert(0) += 1;
            }

            let mut levels: Vec<_> = level_map
                .iter()
                .map(|(key, &depth)| {
                    let fg_match = font_groups
                        .iter()
                        .find(|fg| {
                            fg.font == key.family
                                && size_to_bucket(fg.em_size) == key.size_bucket
                        });
                    HeadingLevelEntry {
                        depth,
                        font: key.family.clone(),
                        height: fg_match.map(|fg| fg.height).unwrap_or(bucket_to_size(key.size_bucket)),
                        char_count: fg_match.map(|fg| fg.char_count).unwrap_or(0),
                        instances: depth_counts.get(&depth).copied().unwrap_or(0),
                        pages: page_counts.get(key).copied().unwrap_or(0),
                    }
                })
                .collect();
            levels.sort_by_key(|l| l.depth);
            levels
        },
        skipped,
        has_font_names,
    };

    HeadingExtractionResult {
        font_groups,
        font_profile,
        headings,
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_font_family_empty() {
        assert_eq!(normalize_font_family(""), "");
    }

    #[test]
    fn test_normalize_font_family_tex() {
        // Standard TeX fonts
        assert_eq!(normalize_font_family("CMR10"), "CMR");
        assert_eq!(normalize_font_family("CMR12"), "CMR");
        assert_eq!(normalize_font_family("CMSS17"), "CMSS");
        assert_eq!(normalize_font_family("CMSSBX10"), "CMSSBX");
        assert_eq!(normalize_font_family("CMMI10"), "CMMI");
        assert_eq!(normalize_font_family("CMSY10"), "CMSY");
        assert_eq!(normalize_font_family("CMEX10"), "CMEX");
        assert_eq!(normalize_font_family("CMBX10"), "CMBX");
        assert_eq!(normalize_font_family("CMR9"), "CMR");
        assert_eq!(normalize_font_family("LCIRCLE10"), "LCIRCLE");
    }

    #[test]
    fn test_normalize_font_family_subset_prefix() {
        assert_eq!(normalize_font_family("BMKCHC+CMR10"), "CMR");
        assert_eq!(normalize_font_family("BGKDGP+TimesNewRoman"), "TimesNewRoman");
        assert_eq!(
            normalize_font_family("BMKDMP+TimesNewRomanPSMT"),
            "TimesNewRoman"
        );
        assert_eq!(
            normalize_font_family("BMKCHC+Giovanni-Book"),
            "Giovanni"
        );
        assert_eq!(
            normalize_font_family("BMKDAB+Giovanni-Bold"),
            "Giovanni"
        );
    }

    #[test]
    fn test_normalize_font_family_opentype() {
        // EDO fonts
        assert_eq!(
            normalize_font_family("OpenSans-Regular"),
            "OpenSans"
        );
        assert_eq!(
            normalize_font_family("OpenSans-Bold"),
            "OpenSans"
        );
        assert_eq!(
            normalize_font_family("TeXGyrePagellaX-Regular"),
            "TeXGyrePagellaX"
        );
        assert_eq!(
            normalize_font_family("TeXGyrePagellaX-Italic"),
            "TeXGyrePagellaX"
        );

        // ACM fonts (AVBD/VBD) — TeX-style suffixes stripped
        assert_eq!(normalize_font_family("LinBiolinumT"), "LinBiolinum");
        assert_eq!(normalize_font_family("LinBiolinumTB"), "LinBiolinum");
        assert_eq!(normalize_font_family("LinLibertineT"), "LinLibertine");
        assert_eq!(normalize_font_family("LinLibertineTI"), "LinLibertine");
        assert_eq!(normalize_font_family("LinLibertineTB"), "LinLibertine");
        assert_eq!(normalize_font_family("LinLibertineTBI"), "LinLibertine");
        assert_eq!(normalize_font_family("LinLibertineTZ"), "LinLibertine");

        // opt.pdf fonts
        assert_eq!(
            normalize_font_family("Minion-Regular"),
            "Minion"
        );
        assert_eq!(
            normalize_font_family("ItcKabel-Bold"),
            "ItcKabel"
        );

        // ml.pdf fonts
        assert_eq!(
            normalize_font_family("LucidaBright"),
            "LucidaBright"
        );
        assert_eq!(
            normalize_font_family("LucidaBright-Demi"),
            "LucidaBright"
        );
        assert_eq!(
            normalize_font_family("LucidaBright-DemiItalic"),
            "LucidaBright"
        );

        // zen0 fonts
        assert_eq!(
            normalize_font_family("NimbusSanL-Regu"),
            "NimbusSanL"
        );
        assert_eq!(
            normalize_font_family("NimbusSanL-Bold"),
            "NimbusSanL"
        );
    }

    #[test]
    fn test_normalize_font_family_pmpp() {
        // programming_massively_parallel.pdf — Adv* fonts
        // These are mixed-case and don't follow TeX or OpenType conventions cleanly
        assert_eq!(normalize_font_family("AdvP6F00"), "AdvP6F00");
        assert_eq!(normalize_font_family("AdvTgb"), "AdvTgb");
        assert_eq!(normalize_font_family("AdvTgb2"), "AdvTgb2");
        assert_eq!(normalize_font_family("AdvTgl"), "AdvTgl");
    }

    #[test]
    fn test_normalize_font_family_standard_ps() {
        // PostScript standard fonts
        assert_eq!(normalize_font_family("Times-Roman"), "Times");
        assert_eq!(normalize_font_family("Times-Bold"), "Times");
        assert_eq!(normalize_font_family("Times-Italic"), "Times");
        assert_eq!(normalize_font_family("Helvetica"), "Helvetica");
        assert_eq!(normalize_font_family("Courier"), "Courier");
        assert_eq!(normalize_font_family("ArialMT"), "ArialMT");
    }

    #[test]
    fn test_height_bucket_roundtrip() {
        assert_eq!(bucket_to_size(size_to_bucket(10.0)), 10.0);
        assert_eq!(bucket_to_size(size_to_bucket(10.25)), 10.5); // rounds to nearest 0.5
        assert_eq!(bucket_to_size(size_to_bucket(10.3)), 10.5);
        assert_eq!(bucket_to_size(size_to_bucket(10.5)), 10.5);
        assert_eq!(bucket_to_size(size_to_bucket(10.75)), 11.0);
    }

    #[test]
    fn test_is_caption_text() {
        assert!(is_caption_text("Figure 1: Overview"));
        assert!(is_caption_text("Fig. 3. Results"));
        assert!(is_caption_text("Table 2: Summary"));
        assert!(is_caption_text("Algorithm 1 Main loop"));
        assert!(!is_caption_text("1.2 Figures of Merit"));
        assert!(!is_caption_text("Introduction"));
    }

    #[test]
    fn test_infer_depth_from_numbering() {
        assert_eq!(infer_depth_from_numbering("1 Introduction"), 1);
        assert_eq!(infer_depth_from_numbering("1.1 Background"), 2);
        assert_eq!(infer_depth_from_numbering("1.2.3 Details"), 3);
        assert_eq!(infer_depth_from_numbering("A Appendix"), 0); // bare letter → use font-based depth
        assert_eq!(infer_depth_from_numbering("A.1 First Section"), 2);
        assert_eq!(infer_depth_from_numbering("III Methods"), 1);
        assert_eq!(infer_depth_from_numbering("Introduction"), 0);
        assert_eq!(infer_depth_from_numbering("References"), 0);
    }

    #[test]
    fn test_is_roman_numeral() {
        assert!(is_roman_numeral("I"));
        assert!(is_roman_numeral("II"));
        assert!(is_roman_numeral("III"));
        assert!(is_roman_numeral("IV"));
        assert!(is_roman_numeral("IX"));
        assert!(is_roman_numeral("XIV"));
        assert!(!is_roman_numeral(""));
        assert!(!is_roman_numeral("ABCDEFGHI")); // too long
        assert!(!is_roman_numeral("ABC")); // not roman chars
    }

    #[test]
    fn test_has_numbering_prefix() {
        // ── Bare numbers ──
        assert!(has_numbering_prefix("1 Introduction"));
        assert!(has_numbering_prefix("12 Chapter Twelve"));
        assert!(has_numbering_prefix("3 Methods"));

        // ── Structured identifiers with join chars ──
        assert!(has_numbering_prefix("1.1 Background"));
        assert!(has_numbering_prefix("1.2.3 Subsection"));
        assert!(has_numbering_prefix("1.0 Overview"));
        assert!(has_numbering_prefix("3.4.2 Recognition of Reserved Words"));
        assert!(has_numbering_prefix("10.3 Advanced Topics"));
        assert!(has_numbering_prefix("A.1 Proofs"));
        assert!(has_numbering_prefix("B.2.3 Deep Subsection"));
        assert!(has_numbering_prefix("B-2 Appendix"));
        assert!(has_numbering_prefix("A1 Supplementary")); // letter+digit, no join

        // ── Single uppercase letters ──
        assert!(has_numbering_prefix("A Appendix"));
        assert!(has_numbering_prefix("B Bibliography"));
        assert!(has_numbering_prefix("Z Glossary"));

        // ── Roman numerals ──
        assert!(has_numbering_prefix("II Background"));
        assert!(has_numbering_prefix("III Methods"));
        assert!(has_numbering_prefix("IV Results"));
        assert!(has_numbering_prefix("XIV Supplementary"));

        // ── Leading whitespace ──
        assert!(has_numbering_prefix("  1.1 Background"));

        // ── Should NOT match ──
        // Space-separated numbers (not a single identifier token)
        assert!(!has_numbering_prefix("9 10 11 (getToken)"));
        assert!(!has_numbering_prefix("1 0 Introduction"));
        // No alphabetic title after prefix
        assert!(!has_numbering_prefix("9 10"));
        assert!(!has_numbering_prefix("1.1 23"));
        // No title text at all
        assert!(!has_numbering_prefix("123"));
        assert!(!has_numbering_prefix("1.1"));
        // Completely non-numeric
        assert!(!has_numbering_prefix("hello world"));
        // Empty
        assert!(!has_numbering_prefix(""));
        // Lowercase letter prefix (not a chapter identifier)
        assert!(!has_numbering_prefix("a introduction"));
        // Just symbols
        assert!(!has_numbering_prefix("* bullet point"));
        assert!(!has_numbering_prefix("- dash item"));
    }

    #[test]
    fn test_parse_numbering_prefix() {
        // ── Dot-separated numbers ──
        assert_eq!(parse_numbering_prefix("3.4.1 Transition Diagrams"), Some(("3.4".into(), 1)));
        assert_eq!(parse_numbering_prefix("1.1 Background"), Some(("1".into(), 1)));
        assert_eq!(parse_numbering_prefix("1.2 Methods"), Some(("1".into(), 2)));
        assert_eq!(parse_numbering_prefix("2.6.3 Minimum elements"), Some(("2.6".into(), 3)));
        assert_eq!(parse_numbering_prefix("10.3 Advanced"), Some(("10".into(), 3)));

        // ── Bare chapter numbers ──
        assert_eq!(parse_numbering_prefix("1 Introduction"), Some(("".into(), 1)));
        assert_eq!(parse_numbering_prefix("3 Methods"), Some(("".into(), 3)));
        assert_eq!(parse_numbering_prefix("12 Appendix"), Some(("".into(), 12)));

        // ── Appendix prefixes ──
        assert_eq!(parse_numbering_prefix("A.1 Proofs"), Some(("A".into(), 1)));
        assert_eq!(parse_numbering_prefix("A.1.2 Lemma"), Some(("A.1".into(), 2)));
        assert_eq!(parse_numbering_prefix("B.3 Tables"), Some(("B".into(), 3)));

        // ── Roman numerals ──
        assert_eq!(parse_numbering_prefix("I Introduction"), Some(("".into(), 1)));
        assert_eq!(parse_numbering_prefix("II Background"), Some(("".into(), 2)));
        assert_eq!(parse_numbering_prefix("III Methods"), Some(("".into(), 3)));
        assert_eq!(parse_numbering_prefix("IV Results"), Some(("".into(), 4)));
        assert_eq!(parse_numbering_prefix("IX Appendix"), Some(("".into(), 9)));
        assert_eq!(parse_numbering_prefix("XIV Supplementary"), Some(("".into(), 14)));

        // ── Non-matches ──
        assert_eq!(parse_numbering_prefix("Convex functions"), None);
        assert_eq!(parse_numbering_prefix("Introduction"), None);
        assert_eq!(parse_numbering_prefix(""), None);
        assert_eq!(parse_numbering_prefix("A Appendix"), None); // single letter, no digit → None (not roman)
    }

    #[test]
    fn test_roman_to_u32() {
        assert_eq!(roman_to_u32("I"), Some(1));
        assert_eq!(roman_to_u32("II"), Some(2));
        assert_eq!(roman_to_u32("III"), Some(3));
        assert_eq!(roman_to_u32("IV"), Some(4));
        assert_eq!(roman_to_u32("V"), Some(5));
        assert_eq!(roman_to_u32("IX"), Some(9));
        assert_eq!(roman_to_u32("X"), Some(10));
        assert_eq!(roman_to_u32("XIV"), Some(14));
        assert_eq!(roman_to_u32("XL"), Some(40));
        assert_eq!(roman_to_u32(""), None);
        assert_eq!(roman_to_u32("ABC"), None);
    }

    /// Helper to build a minimal DetectedHeading for filter tests.
    fn heading(depth: u32, title: &str, page: u32) -> DetectedHeading {
        DetectedHeading {
            depth,
            title: title.to_string(),
            page,
            contained_chars: 5000,
            y_center: 500.0,
            x_right: 300.0,
        }
    }

    #[test]
    fn test_filter_sequence_interrupters_basic() {
        // 3.4.1 → BOGUS [depth 1] → 3.4.3: the depth-1 heading interrupts
        let mut headings = vec![
            heading(3, "3.4.1 Transition Diagrams", 155),
            heading(1, "TOP HEADER", 157),
            heading(3, "3.4.3 Completion", 158),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].title, "3.4.1 Transition Diagrams");
        assert_eq!(headings[1].title, "3.4.3 Completion");
    }

    #[test]
    fn test_filter_sequence_interrupters_no_interruption() {
        // 2.6.3 → [depth 1] → 3.1: different parent, no interruption
        let mut headings = vec![
            heading(3, "2.6.3 Minimum elements", 68),
            heading(1, "Convex functions", 81),
            heading(2, "3.1 Basic properties", 81),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 3); // all kept
    }

    #[test]
    fn test_filter_sequence_interrupters_chapter_numbers() {
        // Chapters: 1 → BOGUS [depth 2] → 2 — interrupts sequence
        let mut headings = vec![
            heading(1, "1 Introduction", 10),
            heading(2, "Spurious Section", 15),
            heading(1, "2 Methods", 20),
        ];
        filter_sequence_interrupters(&mut headings);
        // depth 2 < depth 1 is false (2 > 1), so it's NOT shallower — keep it
        assert_eq!(headings.len(), 3);
    }

    #[test]
    fn test_filter_sequence_interrupters_deeper_not_removed() {
        // 1.1 → [depth 3] → 1.2: depth 3 is DEEPER than depth 2, not shallower — keep
        let mut headings = vec![
            heading(2, "1.1 Background", 10),
            heading(3, "1.1.1 Details", 12),
            heading(2, "1.2 Methods", 20),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 3); // all kept
    }

    #[test]
    fn test_filter_sequence_interrupters_multiple_interrupters() {
        // 1.1 → BOGUS1 [depth 1] → BOGUS2 [depth 1] → 1.2
        let mut headings = vec![
            heading(2, "1.1 Background", 10),
            heading(1, "BOGUS CHAPTER", 12),
            heading(1, "ANOTHER BOGUS", 14),
            heading(2, "1.2 Methods", 20),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].title, "1.1 Background");
        assert_eq!(headings[1].title, "1.2 Methods");
    }

    #[test]
    fn test_filter_sequence_interrupters_roman_sequence() {
        // II → BOGUS → III: roman numeral sequence
        let mut headings = vec![
            heading(1, "II Background", 10),
            heading(2, "Spurious subsection that is deeper", 15),
            heading(1, "III Methods", 20),
        ];
        filter_sequence_interrupters(&mut headings);
        // depth 2 is deeper than 1, not shallower — keep all
        assert_eq!(headings.len(), 3);

        // Now test with a shallower interloper (impossible in practice but tests logic)
        // Actually roman at depth 1, interloper can't be depth 0, so this case
        // doesn't apply for romans at depth 1. Test at depth 2 level:
        let mut headings = vec![
            heading(2, "2.1 First", 10),
            heading(1, "II BOGUS ROMAN", 15),
            heading(2, "2.2 Second", 20),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].title, "2.1 First");
        assert_eq!(headings[1].title, "2.2 Second");
    }

    #[test]
    fn test_filter_sequence_interrupters_appendix_sequence() {
        // A.1 → BOGUS → A.2: appendix subsection sequence
        let mut headings = vec![
            heading(2, "A.1 Proofs", 100),
            heading(1, "BOGUS CHAPTER", 105),
            heading(2, "A.2 Tables", 110),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].title, "A.1 Proofs");
        assert_eq!(headings[1].title, "A.2 Tables");
    }

    #[test]
    fn test_filter_sequence_interrupters_nested() {
        // Deep nesting: A.1.1 → BOGUS → A.1.2
        let mut headings = vec![
            heading(3, "A.1.1 Lemma one", 100),
            heading(1, "BOGUS", 102),
            heading(2, "ALSO BOGUS", 103),
            heading(3, "A.1.2 Lemma two", 105),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].title, "A.1.1 Lemma one");
        assert_eq!(headings[1].title, "A.1.2 Lemma two");
    }

    #[test]
    fn test_filter_sequence_interrupters_non_consecutive_seq() {
        // Gap in sequence: 3.4.1 → 3.4.3 (skipping 3.4.2) — still valid
        let mut headings = vec![
            heading(3, "3.4.1 First", 155),
            heading(1, "BOGUS", 157),
            heading(3, "3.4.3 Third", 158),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 2);
        assert_eq!(headings[0].title, "3.4.1 First");
        assert_eq!(headings[1].title, "3.4.3 Third");
    }

    #[test]
    fn test_filter_sequence_interrupters_no_numbered_headings() {
        // All unnumbered — nothing happens
        let mut headings = vec![
            heading(1, "Introduction", 1),
            heading(2, "Background", 5),
            heading(1, "Methods", 10),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 3);
    }

    #[test]
    fn test_filter_sequence_interrupters_same_depth_interrupter() {
        // 1.1 → [depth 2] → 1.2: same depth as sequence, not shallower — keep
        let mut headings = vec![
            heading(2, "1.1 First", 10),
            heading(2, "Unnumbered subsection", 15),
            heading(2, "1.2 Second", 20),
        ];
        filter_sequence_interrupters(&mut headings);
        assert_eq!(headings.len(), 3);
    }
}
