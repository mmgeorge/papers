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
