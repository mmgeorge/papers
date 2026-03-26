use std::collections::HashMap;
use std::path::Path;

use image::DynamicImage;
use pdfium_render::prelude::*;

use crate::error::ExtractError;

/// A character extracted from the PDF text layer with its bounding box.
#[derive(Debug, Clone)]
pub struct PdfChar {
    pub codepoint: char,
    /// Bounding box in PDF points (72pt/inch): [x1, y1, x2, y2]
    /// where (x1, y1) is bottom-left and (x2, y2) is top-right.
    pub bbox: [f32; 4],
    /// Pre-computed gap threshold for word boundary detection, in PDF points.
    ///
    /// When the horizontal gap between two consecutive characters exceeds this threshold,
    /// a space character should be inserted between them. This replicates pdfium's own
    /// space-insertion heuristic from `cpdf_textpage.cpp`:
    ///
    ///   threshold = font_size × font.GetCharWidthF(space_charcode) / 1000 / 2
    ///
    /// We compute this by extracting the embedded font data via `PdfFont::data()` and
    /// parsing it with `ttf-parser` to get the exact space character advance width.
    /// When font data is unavailable (non-embedded fonts), we fall back to an approximation
    /// using `space_width_ratio = 0.3` (typical for Latin fonts where space ≈ 250–333 / 1000).
    pub space_threshold: f32,
    /// Font name as reported by pdfium (e.g. "CMMI10", "LibertineMathMI").
    pub font_name: String,
    /// Rendered font size in PDF points.
    pub font_size: f32,
    /// Whether the font is italic (excluding symbolic/math fonts).
    pub is_italic: bool,
    /// Whether the font is bold (excluding symbolic/math fonts).
    pub is_bold: bool,
}

/// Get the CropBox offset for a page.
///
/// Returns (x_offset, y_offset) that should be subtracted from MediaBox
/// char coordinates to convert them to CropBox-relative coordinates
/// (matching the rendered image). Returns (0, 0) if no CropBox is set.
pub fn crop_offset(page: &PdfPage) -> (f32, f32) {
    match page.boundaries().crop() {
        Ok(b) => (b.bounds.left().value, b.bounds.bottom().value),
        Err(_) => (0.0, 0.0),
    }
}

/// Apply CropBox offset to a set of chars, converting from MediaBox
/// coordinates to CropBox-relative coordinates.
pub fn apply_crop_offset(chars: &mut [PdfChar], offset: (f32, f32)) {
    let (dx, dy) = offset;
    if dx.abs() < 0.01 && dy.abs() < 0.01 {
        return;
    }
    for c in chars.iter_mut() {
        c.bbox[0] -= dx;
        c.bbox[1] -= dy;
        c.bbox[2] -= dx;
        c.bbox[3] -= dy;
    }
}

/// Render a page to an RGB image at the given DPI.
pub fn render_page(page: &PdfPage, dpi: u32) -> Result<DynamicImage, ExtractError> {
    let width_pt = page.width().value;
    let height_pt = page.height().value;

    let scale = dpi as f32 / 72.0;
    let width_px = (width_pt * scale) as i32;
    let height_px = (height_pt * scale) as i32;

    let config = PdfRenderConfig::new()
        .set_target_width(width_px)
        .set_target_height(height_px);

    let bitmap = page
        .render_with_config(&config)
        .map_err(|e| ExtractError::Pdf(format!("Failed to render page: {e}")))?;

    Ok(bitmap.as_image())
}

/// Extract all characters from a page's text layer with bounding boxes.
pub fn extract_page_chars(page: &PdfPage, page_idx: u32) -> Result<Vec<PdfChar>, ExtractError> {
    let text = page.text().map_err(|e| {
        ExtractError::Pdf(format!(
            "Failed to get text layer for page {page_idx}: {e}"
        ))
    })?;

    let chars_collection = text.chars();
    let mut chars = Vec::new();

    // Cache space_width_ratio per font name to avoid re-parsing font data for every character.
    // Key: font name, Value: space_width / units_per_em ratio (typically 0.25–0.33 for Latin fonts).
    let mut font_space_ratios: HashMap<String, f32> = HashMap::new();

    let total = chars_collection.len();
    let mut i = 0usize;
    while i < total {
        let Ok(char_info) = chars_collection.get(i) else {
            i += 1;
            continue;
        };

        // Try direct Unicode mapping first (works for all BMP characters).
        let c = if let Some(c) = char_info.unicode_char() {
            c
        } else {
            // On Windows (16-bit wchar_t), pdfium returns supplementary plane
            // characters (U+10000+) as UTF-16 surrogate pairs across two
            // consecutive char indices. `char::from_u32()` returns None for
            // surrogate code units, so we detect and reassemble them here.
            let raw = char_info.unicode_value();
            if is_high_surrogate(raw) {
                if let Some(decoded) = decode_surrogate_pair(&chars_collection, i, total) {
                    i += 2; // consumed both surrogates
                    push_char(
                        &mut chars,
                        decoded,
                        &char_info,
                        &mut font_space_ratios,
                    );
                    continue;
                }
            }
            // Unmappable character (not a surrogate pair) — skip it.
            i += 1;
            continue;
        };

        push_char(&mut chars, c, &char_info, &mut font_space_ratios);
        i += 1;
    }

    Ok(chars)
}

/// Push a resolved character onto the chars vec, computing font metrics.
fn push_char(
    chars: &mut Vec<PdfChar>,
    c: char,
    char_info: &PdfPageTextChar,
    font_space_ratios: &mut HashMap<String, f32>,
) {
    let Ok(rect) = char_info.loose_bounds() else {
        return;
    };
    let font_size = char_info.scaled_font_size().value;
    let font_name = char_info.font_name();
    let space_ratio = *font_space_ratios
        .entry(font_name.clone())
        .or_insert_with(|| compute_space_width_ratio(char_info, &font_name));

    // Italic detection: trust font_is_italic() but skip math/symbol fonts
    // that report italic as a side effect (e.g. CMMI, CMSY, MathPI).
    // We can't use font_is_symbolic() alone because TeX-embedded text fonts
    // (like LinLibertineTI) also set the symbolic flag due to custom encodings.
    let is_italic = char_info.font_is_italic() && !is_math_font(&font_name);

    // Bold detection: use font name patterns (same heuristic as TOC depth
    // classification). Skip math/symbol fonts that may contain bold glyphs
    // as part of their design (not text bold).
    let is_bold = is_bold_font(&font_name) && !is_math_font(&font_name);

    chars.push(PdfChar {
        codepoint: c,
        bbox: [
            rect.left().value,
            rect.bottom().value,
            rect.right().value,
            rect.top().value,
        ],
        space_threshold: font_size * space_ratio / 2.0,
        font_name: font_name.clone(),
        font_size,
        is_italic,
        is_bold,
    });
}

/// Expand a character to its component letters when pdfium's Unicode
/// resolution failed (returned a control char or a Unicode ligature).
///
/// This handles two cases where pdfium gives us a character that needs
/// decomposition:
///
/// 1. **Unicode ligature codepoints** (U+FB00-U+FB06): Some PDFs map
///    ligature glyphs to these dedicated Unicode positions. We decompose
///    them to their constituent letters for cleaner text output.
///
/// 2. **Unresolved TeX font encodings**: Type3 fonts generated by TeX
///    often lack a ToUnicode CMap and use glyph names like `#0B` instead
///    of standard Adobe Glyph List names like `fi`. When pdfium can't
///    resolve the glyph name, it falls back to the raw character code,
///    producing control characters. The standard TeX OT1 encoding
///    (Computer Modern and derivatives) defines ligatures at fixed
///    positions:
///      - 0x0B = ff, 0x0C = fi, 0x0D = fl, 0x0E = ffi, 0x0F = ffl
///    This encoding is used by the vast majority of TeX-generated PDFs.
///    We only apply this mapping to control characters (U+0000-U+001F),
///    which are never valid text, so there's no risk of false positives.
pub fn expand_ligature(c: char) -> Option<&'static str> {
    match c {
        // Unicode ligature block (explicit ligature codepoints)
        '\u{FB00}' => Some("ff"),
        '\u{FB01}' => Some("fi"),
        '\u{FB02}' => Some("fl"),
        '\u{FB03}' => Some("ffi"),
        '\u{FB04}' => Some("ffl"),
        '\u{FB05}' | '\u{FB06}' => Some("st"),
        // TeX OT1 encoding: control char positions used for ligatures
        '\u{000B}' => Some("ff"),
        '\u{000C}' => Some("fi"),
        '\u{000D}' => Some("fl"),
        '\u{000E}' => Some("ffi"),
        '\u{000F}' => Some("ffl"),
        _ => None,
    }
}

/// Check if a font name matches known math/symbol font patterns.
/// These fonts report italic as a side effect but are not text italic fonts.
pub fn is_math_font(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    // TeX math fonts: CMMI (math italic), CMSY (math symbols), CMEX (extensions)
    // Also catch LMMI (Latin Modern Math Italic), MathPI, etc.
    n.starts_with("CMMI") || n.starts_with("CMSY") || n.starts_with("CMEX")
        || n.starts_with("LMMI") || n.starts_with("LMSY")
        || n.contains("MATHPI") || n.contains("MATH-")
        || n == "SYMBOL" || n.starts_with("SYMBOL-")
        || n.contains("DINGBAT")
        // Libertine/Stix math variants (not text italic)
        || n.contains("LIBERTINEMATH") || n.contains("MATHMI")
}

/// Check if a font name matches known bold font patterns.
///
/// Detects bold variants by common naming conventions: "Bold", "-Bd",
/// "Demi", "Heavy", "Black", and TeX bold-extended (bx10, bx12).
pub fn is_bold_font(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("bold")
        || lower.contains("-bd")
        || lower.contains("demi")
        || lower.contains("heavy")
        || lower.contains("black")
        || lower.ends_with("bx10")
        || lower.ends_with("bx12")
}

fn is_high_surrogate(raw: u32) -> bool {
    (0xD800..=0xDBFF).contains(&raw)
}

fn is_low_surrogate(raw: u32) -> bool {
    (0xDC00..=0xDFFF).contains(&raw)
}

/// Attempt to decode a UTF-16 surrogate pair at position `i` in the chars collection.
/// Returns the decoded char if `i` is a high surrogate followed by a low surrogate.
fn decode_surrogate_pair(
    chars_collection: &PdfPageTextChars,
    i: usize,
    total: usize,
) -> Option<char> {
    if i + 1 >= total {
        return None;
    }
    let Ok(next) = chars_collection.get(i + 1) else {
        return None;
    };
    let raw_hi = chars_collection.get(i).ok()?.unicode_value();
    let raw_lo = next.unicode_value();
    if !is_low_surrogate(raw_lo) {
        return None;
    }
    let codepoint = 0x10000 + ((raw_hi - 0xD800) << 10) + (raw_lo - 0xDC00);
    char::from_u32(codepoint)
}

/// Compute the space character's width as a ratio of the em square for the given font.
///
/// Uses pdfium-render's `PdfFont::data()` to extract embedded font bytes, then parses
/// with `ttf-parser` to look up the space character's (U+0020) horizontal advance width.
///
/// Returns `advance_width / units_per_em`, which is the same value that pdfium uses
/// internally as `font.GetCharWidthF(space_charcode) / 1000` (normalized to 1.0 = full em).
///
/// Falls back to `0.3` (approximate midpoint of 0.25–0.33 for typical Latin fonts) when:
/// - The font is not embedded (`font.data()` fails)
/// - The font data can't be parsed by ttf-parser
/// - The font has no space glyph
/// - The font has no horizontal advance for the space glyph
fn compute_space_width_ratio(char_info: &PdfPageTextChar, font_name: &str) -> f32 {
    const FALLBACK_RATIO: f32 = 0.3;

    let text_obj = match char_info.text_object() {
        Ok(obj) => obj,
        Err(_) => {
            tracing::warn!(
                "word-boundary: cannot access text object for font '{}', \
                 using fallback space ratio {FALLBACK_RATIO}",
                font_name
            );
            return FALLBACK_RATIO;
        }
    };

    let font = text_obj.font();

    let font_data = match font.data() {
        Ok(data) => data,
        Err(_) => {
            tracing::warn!(
                "word-boundary: font '{}' data unavailable, \
                 using fallback space ratio {FALLBACK_RATIO}",
                font_name
            );
            return FALLBACK_RATIO;
        }
    };

    let face = match ttf_parser::Face::parse(&font_data, 0) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!(
                "word-boundary: failed to parse font '{}': {e}, \
                 using fallback space ratio {FALLBACK_RATIO}",
                font_name
            );
            return FALLBACK_RATIO;
        }
    };

    let units_per_em = face.units_per_em() as f32;
    if units_per_em == 0.0 {
        tracing::warn!(
            "word-boundary: font '{}' has units_per_em=0, \
             using fallback space ratio {FALLBACK_RATIO}",
            font_name
        );
        return FALLBACK_RATIO;
    }

    let space_gid = match face.glyph_index(' ') {
        Some(gid) => gid,
        None => {
            tracing::warn!(
                "word-boundary: font '{}' has no space glyph, \
                 using fallback space ratio {FALLBACK_RATIO}",
                font_name
            );
            return FALLBACK_RATIO;
        }
    };

    match face.glyph_hor_advance(space_gid) {
        Some(advance) => {
            let ratio = advance as f32 / units_per_em;
            tracing::debug!(
                "word-boundary: font '{}' space width = {advance}/{units_per_em} = {ratio:.4}",
                font_name
            );
            ratio
        }
        None => {
            tracing::warn!(
                "word-boundary: font '{}' space glyph has no advance width, \
                 using fallback space ratio {FALLBACK_RATIO}",
                font_name
            );
            FALLBACK_RATIO
        }
    }
}

/// Try to recover a Form XObject's `/BBox` from the clip path on its children.
///
/// pdfium converts the Form's `/BBox` into a clip path applied to each child
/// during content parsing. The clip path is already transformed to page space
/// (PDF Y-up coordinates). We read it back to get the true visual boundary
/// that `FPDFPageObj_GetBounds()` ignores.
///
/// Returns `[left, bottom, right, top]` in PDF coordinates, or `None` if no
/// rectangular clip path is found.
fn form_clip_bbox(
    form: &PdfPageXObjectFormObject,
    form_obj: &PdfPageObject,
) -> Option<[f32; 4]> {
    // Get the form's placement matrix (maps local coords → page coords).
    let mat = form_obj.matrix().ok()?;
    let (a, b, c, d, e, f) = (mat.a(), mat.b(), mat.c(), mat.d(), mat.e(), mat.f());

    // Collect all rectangular clips from children and return the LARGEST one.
    // Children may have both the /BBox clip (outermost) and internal content
    // clips (tighter). CheckClip strips the /BBox clip from children that fit
    // inside it, leaving only internal clips. By taking the largest, we get
    // the /BBox when available, or the best approximation otherwise.
    let mut best: Option<[f32; 4]> = None;
    let mut best_area: f32 = 0.0;

    for i in 0..form.len() {
        let Ok(child) = form.get(i) else {
            continue;
        };
        let Some(clip) = child.get_clip_path() else {
            continue;
        };
        if clip.is_empty() {
            continue;
        }

        // Check all sub-paths — /BBox might not be at index 0 if there are
        // multiple clips (nested forms, content clips).
        let sub_path_count = clip.len().min(10);
        for sp_idx in 0..sub_path_count {
            let Ok(sub_path) = clip.get(sp_idx) else {
                continue;
            };
            if sub_path.len() < 4 {
                continue;
            }

            let (Ok(p0), Ok(p2)) = (sub_path.get(0), sub_path.get(2)) else {
                continue;
            };

            let lx1 = p0.x().value.min(p2.x().value);
            let ly1 = p0.y().value.min(p2.y().value);
            let lx2 = p0.x().value.max(p2.x().value);
            let ly2 = p0.y().value.max(p2.y().value);

            if (lx2 - lx1) < 10.0 || (ly2 - ly1) < 10.0 {
                continue;
            }

            // Transform to page space
            let corners = [
                (a * lx1 + c * ly1 + e, b * lx1 + d * ly1 + f),
                (a * lx2 + c * ly1 + e, b * lx2 + d * ly1 + f),
                (a * lx2 + c * ly2 + e, b * lx2 + d * ly2 + f),
                (a * lx1 + c * ly2 + e, b * lx1 + d * ly2 + f),
            ];
            let px1 = corners.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
            let py1 = corners.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
            let px2 = corners.iter().map(|c| c.0).fold(f32::NEG_INFINITY, f32::max);
            let py2 = corners.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);

            let area = (px2 - px1) * (py2 - py1);
            if area > best_area {
                best_area = area;
                best = Some([px1, py1, px2, py2]);
            }
        }
    }
    best
}

/// Detect figure-sized visual objects on a PDF page.
///
/// Returns bounding boxes (image-space, Y-down) for all Image and Form XObject
/// page objects that pass size filters. For Form XObjects, recovers the `/BBox`
/// clipping boundary from child clip paths (the true visual extent) and
/// transforms it to page space. Falls back to `obj.bounds()` when no clip path
/// is available (e.g. for direct Image objects).
pub fn detect_page_figures(
    page: &PdfPage,
    page_height_pt: f32,
    page_width_pt: f32,
    crop: (f32, f32),
) -> Vec<[f32; 4]> {
    let mut figures: Vec<[f32; 4]> = Vec::new();
    let (dx, dy) = crop;

    for obj in page.objects().iter() {
        let obj_type = obj.object_type();
        if obj_type != PdfPageObjectType::Image
            && obj_type != PdfPageObjectType::XObjectForm
        {
            continue;
        }

        // For Form XObjects, try to recover /BBox from child clip paths.
        // obj.bounds() returns the raw union of child bounds which can
        // extend beyond the /BBox clipping boundary, causing text bleed.
        if obj_type == PdfPageObjectType::XObjectForm {
            if let Some(form) = obj.as_x_object_form_object() {
                if let Some(clip_rect) = form_clip_bbox(form, &obj) {
                    let w_pt = clip_rect[2] - clip_rect[0];
                    let h_pt = clip_rect[3] - clip_rect[1];

                    if w_pt < 50.0 || h_pt < 50.0 {
                        continue;
                    }
                    if w_pt > page_width_pt * 0.9 && h_pt > page_height_pt * 0.9 {
                        continue;
                    }

                    // Convert PDF Y-up to image Y-down, apply crop offset
                    let img_x1 = clip_rect[0] - dx;
                    let img_y1 = page_height_pt - clip_rect[3] - dy;
                    let img_x2 = clip_rect[2] - dx;
                    let img_y2 = page_height_pt - clip_rect[1] - dy;
                    let bbox = [img_x1, img_y1, img_x2, img_y2];

                    if !is_dominated(&figures, &bbox) {
                        figures.push(bbox);
                    }
                    continue;
                }
                // Fall through to obj.bounds() if no clip path found
            }
        }

        let Ok(rect) = obj.bounds() else {
            continue;
        };

        let w_pt = rect.width().value;
        let h_pt = rect.height().value;

        if w_pt < 50.0 || h_pt < 50.0 {
            continue;
        }
        if w_pt > page_width_pt * 0.9 && h_pt > page_height_pt * 0.9 {
            continue;
        }

        // Convert PDF Y-up to image Y-down, apply crop offset
        let img_y1 = page_height_pt - rect.top().value - dy;
        let img_y2 = page_height_pt - rect.bottom().value - dy;
        let img_x1 = rect.left().value - dx;
        let img_x2 = rect.right().value - dx;
        let bbox = [img_x1, img_y1, img_x2, img_y2];

        if !is_dominated(&figures, &bbox) {
            figures.push(bbox);
        }
    }

    // Detect thin Path objects (lines) near figure bboxes: connector lines,
    // axis lines, tick marks. These bridge gaps between labels and graphics.
    // Includes both horizontal lines (h < 3pt) and vertical lines (w < 3pt).
    let mut connectors: Vec<[f32; 4]> = Vec::new();
    for obj in page.objects().iter() {
        if obj.object_type() != PdfPageObjectType::Path {
            continue;
        }
        let Ok(rect) = obj.bounds() else { continue };
        let w = rect.width().value;
        let h = rect.height().value;
        // Thin line: one dimension < 3pt, the other > 5pt
        let is_thin_line = (h < 3.0 && w > 5.0) || (w < 3.0 && h > 5.0);
        if !is_thin_line {
            continue;
        }
        let img_y1 = page_height_pt - rect.top().value - dy;
        let img_y2 = page_height_pt - rect.bottom().value - dy;
        let img_x1 = rect.left().value - dx;
        let img_x2 = rect.right().value - dx;
        let path_bbox = [img_x1, img_y1, img_x2, img_y2];

        // Only include if adjacent to an existing figure (gap ≤ 10pt)
        let near_figure = figures.iter().any(|fig| {
            let x_gap = (path_bbox[0] - fig[2]).max(fig[0] - path_bbox[2]).max(0.0);
            let y_gap = (path_bbox[1] - fig[3]).max(fig[1] - path_bbox[3]).max(0.0);
            x_gap <= 10.0 && y_gap <= 10.0
        });
        if near_figure {
            connectors.push(path_bbox);
        }
    }
    figures.extend(connectors);

    // Merge nearby bboxes into composite figures. Sub-panels of the same
    // figure overlap in one dimension and are adjacent in the other.
    // Repeat until no more merges occur (transitive closure).
    merge_nearby_bboxes(&mut figures);

    figures
}

/// Merge bboxes that overlap or are close together into composite groups.
///
/// Two bboxes merge if they overlap in one axis and the gap in the other
/// axis is small (≤10pt). This groups side-by-side or stacked sub-panels
/// into a single figure bbox.
fn merge_nearby_bboxes(bboxes: &mut Vec<[f32; 4]>) {
    const GAP_THRESHOLD: f32 = 10.0;

    loop {
        let mut merged = false;
        let mut i = 0;
        while i < bboxes.len() {
            let mut j = i + 1;
            while j < bboxes.len() {
                let [ax1, ay1, ax2, ay2] = bboxes[i];
                let [bx1, by1, bx2, by2] = bboxes[j];

                // Check if they overlap or are close in both dimensions
                let x_overlap = ax2.min(bx2) - ax1.max(bx1);
                let y_overlap = ay2.min(by2) - ay1.max(by1);

                // They share vertical extent and are horizontally close (side-by-side)
                let side_by_side = y_overlap > 0.0
                    && (bx1 - ax2).max(ax1 - bx2) <= GAP_THRESHOLD;

                // They share horizontal extent and are vertically close (stacked)
                let stacked = x_overlap > 0.0
                    && (by1 - ay2).max(ay1 - by2) <= GAP_THRESHOLD;

                if side_by_side || stacked {
                    // Merge: expand bbox[i] to include bbox[j]
                    bboxes[i] = [
                        ax1.min(bx1), ay1.min(by1),
                        ax2.max(bx2), ay2.max(by2),
                    ];
                    bboxes.remove(j);
                    merged = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        if !merged {
            break;
        }
    }
}

/// Check if a bbox is mostly covered (>50%) by an existing figure bbox.
fn is_dominated(figures: &[[f32; 4]], bbox: &[f32; 4]) -> bool {
    figures.iter().any(|existing| {
        let [ex1, ey1, ex2, ey2] = *existing;
        let x_overlap = bbox[2].min(ex2) - bbox[0].max(ex1);
        let y_overlap = bbox[3].min(ey2) - bbox[1].max(ey1);
        if x_overlap <= 0.0 || y_overlap <= 0.0 {
            return false;
        }
        let overlap_area = x_overlap * y_overlap;
        let this_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
        this_area > 0.0 && overlap_area / this_area > 0.5
    })
}

/// Detect table regions on a PDF page by finding clusters of horizontal rules.
///
/// Tables in academic papers typically have ≥3 horizontal rules (top rule,
/// header separator, bottom rule). We detect these from PDF path objects and
/// cluster them by vertical proximity + horizontal overlap.
///
/// Returns table bounding boxes in image space (Y-down, PDF points).
pub fn detect_page_tables(
    page: &PdfPage,
    page_height_pt: f32,
    page_width_pt: f32,
    crop: (f32, f32),
) -> Vec<[f32; 4]> {
    let (dx, dy) = crop;

    // Collect horizontal and vertical rules from path objects
    let mut h_rules: Vec<(f32, f32, f32)> = Vec::new(); // (y_center, x_min, x_max)
    let mut v_rules: Vec<(f32, f32, f32)> = Vec::new(); // (x_center, y_min, y_max)

    for obj in page.objects().iter() {
        if obj.object_type() != PdfPageObjectType::Path {
            continue;
        }

        let Ok(rect) = obj.bounds() else {
            continue;
        };

        let w = rect.width().value;
        let h = rect.height().value;

        // Convert to image space
        let y_top = page_height_pt - rect.top().value - dy;
        let y_bot = page_height_pt - rect.bottom().value - dy;
        let x_left = rect.left().value - dx;
        let x_right = rect.right().value - dx;

        // Horizontal rule: thin and wide
        if h < 3.0 && w > 50.0 {
            h_rules.push(((y_top + y_bot) / 2.0, x_left, x_right));
        }

        // Vertical rule: tall and thin (for chart grid detection)
        if w < 3.0 && h > 30.0 {
            v_rules.push(((x_left + x_right) / 2.0, y_top, y_bot));
        }
    }

    if h_rules.len() < 3 {
        return Vec::new();
    }

    // Cluster horizontal rules by X-extent alignment. Two rules belong to the
    // same table if their x_min and x_max match within a tolerance.
    let x_tol = 10.0f32;
    let mut clusters: Vec<Vec<(f32, f32, f32)>> = Vec::new();

    for &rule in &h_rules {
        let (_, x_min, x_max) = rule;
        let matched = clusters.iter_mut().find(|c| {
            let (_, cx_min, cx_max) = c[0];
            (x_min - cx_min).abs() < x_tol && (x_max - cx_max).abs() < x_tol
        });
        if let Some(cluster) = matched {
            cluster.push(rule);
        } else {
            clusters.push(vec![rule]);
        }
    }

    // Keep only clusters with ≥3 rules AND sufficient vertical extent.
    // A real table spans at least ~20pt (header + one row); graph axis
    // tick marks cluster in <5pt of vertical space.
    clusters.retain(|c| {
        if c.len() < 3 {
            return false;
        }
        let y_min = c.iter().map(|r| r.0).fold(f32::MAX, f32::min);
        let y_max = c.iter().map(|r| r.0).fold(f32::MIN, f32::max);
        (y_max - y_min) >= 20.0
    });

    // Convert clusters to bounding boxes, filtering out chart grids
    let padding = 5.0;
    clusters
        .iter()
        .filter_map(|cluster| {
            let y_min = cluster.iter().map(|r| r.0).fold(f32::MAX, f32::min);
            let y_max = cluster.iter().map(|r| r.0).fold(f32::MIN, f32::max);
            let x_min = cluster.iter().map(|r| r.1).fold(f32::MAX, f32::min);
            let x_max = cluster.iter().map(|r| r.2).fold(f32::MIN, f32::max);

            // Filter chart grids: if this area also contains ≥3 vertical rules,
            // it's likely a chart with grid lines, not a table.
            let v_count = v_rules.iter().filter(|&&(vx, vy_min, vy_max)| {
                vx >= x_min - 5.0
                    && vx <= x_max + 5.0
                    && vy_min <= y_max + 5.0
                    && vy_max >= y_min - 5.0
            }).count();
            if v_count >= 3 {
                return None;
            }

            Some([
                x_min.max(0.0),
                (y_min - padding).max(0.0),
                x_max.min(page_width_pt),
                (y_max + padding).min(page_height_pt),
            ])
        })
        .collect()
}

/// Load the pdfium library.
///
/// Search order: explicit path → `PDFIUM_PATH` env var (set by
/// `.cargo/config.toml` for `cargo run`) → next to executable (dist bundle)
/// → system library paths → error.
pub fn load_pdfium(pdfium_path: Option<&Path>) -> Result<Pdfium, ExtractError> {
    // 1. User-provided explicit path
    if let Some(path) = pdfium_path {
        let lib_name = Pdfium::pdfium_platform_library_name_at_path(
            path.to_str().unwrap_or_default(),
        );
        let bindings =
            Pdfium::bind_to_library(&lib_name).map_err(|_| ExtractError::PdfiumNotFound)?;
        return Ok(Pdfium::new(bindings));
    }

    // 2. PDFIUM_PATH env var (set by .cargo/config.toml for cargo run)
    if let Ok(dir) = std::env::var("PDFIUM_PATH") {
        let lib_name = Pdfium::pdfium_platform_library_name_at_path(&dir);
        if let Ok(bindings) = Pdfium::bind_to_library(&lib_name) {
            return Ok(Pdfium::new(bindings));
        }
    }

    // 3. Next to the executable (dist bundle)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let lib_name = Pdfium::pdfium_platform_library_name_at_path(
                exe_dir.to_str().unwrap_or_default(),
            );
            if let Ok(bindings) = Pdfium::bind_to_library(&lib_name) {
                return Ok(Pdfium::new(bindings));
            }
        }
    }

    // 4. System library paths
    if let Ok(bindings) = Pdfium::bind_to_system_library() {
        return Ok(Pdfium::new(bindings));
    }

    Err(ExtractError::PdfiumNotFound)
}
