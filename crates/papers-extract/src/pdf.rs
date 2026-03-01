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
    });
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

/// Load the pdfium library, trying system paths then a cached location.
pub fn load_pdfium(pdfium_path: Option<&Path>) -> Result<Pdfium, ExtractError> {
    // 1. User-provided explicit path
    if let Some(path) = pdfium_path {
        let bindings = Pdfium::bind_to_library(
            Pdfium::pdfium_platform_library_name_at_path(
                path.to_str().unwrap_or_default(),
            ),
        )
        .map_err(|_| ExtractError::PdfiumNotFound)?;
        return Ok(Pdfium::new(bindings));
    }

    // 2. Search system paths
    if let Ok(bindings) = Pdfium::bind_to_system_library() {
        return Ok(Pdfium::new(bindings));
    }

    // 3. Check our cache dir
    let cache = pdfium_cache_dir();
    let lib_name = Pdfium::pdfium_platform_library_name_at_path(
        cache.to_str().unwrap_or_default(),
    );
    if Path::new(&lib_name).exists() {
        let bindings = Pdfium::bind_to_library(&lib_name)
            .map_err(|_| ExtractError::PdfiumNotFound)?;
        return Ok(Pdfium::new(bindings));
    }

    // 4. Download as last resort
    eprintln!("Pdfium not found locally, downloading...");
    download_pdfium(&cache)?;
    let bindings = Pdfium::bind_to_library(&lib_name)
        .map_err(|_| ExtractError::PdfiumNotFound)?;
    Ok(Pdfium::new(bindings))
}

/// Cache directory for the pdfium binary.
fn pdfium_cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("papers")
        .join("pdfium")
}

/// Download pdfium binary from bblanchon/pdfium-binaries and extract to the cache dir.
fn download_pdfium(cache_dir: &Path) -> Result<(), ExtractError> {
    use flate2::read::GzDecoder;
    use std::io::Write;
    use tar::Archive;

    let url = pdfium_download_url();
    let lib_filename = pdfium_lib_filename();

    eprintln!("Downloading pdfium from {url}...");

    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| ExtractError::Download(format!("Failed to create HTTP client: {e}")))?;

    let response = client
        .get(&url)
        .send()
        .map_err(|e| ExtractError::Download(format!("Pdfium download failed: {e}")))?;

    if !response.status().is_success() {
        return Err(ExtractError::Download(format!(
            "Pdfium download failed: HTTP {}",
            response.status()
        )));
    }

    let bytes = response
        .bytes()
        .map_err(|e| ExtractError::Download(format!("Failed to read pdfium archive: {e}")))?;

    // Extract the library file from the .tgz archive
    std::fs::create_dir_all(cache_dir)?;
    let decoder = GzDecoder::new(bytes.as_ref());
    let mut archive = Archive::new(decoder);

    // Look for bin/pdfium.dll (Windows) or lib/libpdfium.so (Linux) or lib/libpdfium.dylib (macOS)
    let target_entry = pdfium_archive_path();

    for entry in archive
        .entries()
        .map_err(|e| ExtractError::Download(format!("Failed to read tar entries: {e}")))?
    {
        let mut entry =
            entry.map_err(|e| ExtractError::Download(format!("Failed to read tar entry: {e}")))?;
        let path = entry
            .path()
            .map_err(|e| ExtractError::Download(format!("Invalid path in archive: {e}")))?;

        if path.to_str().map_or(false, |p| p == target_entry) {
            let dest = cache_dir.join(lib_filename);
            let tmp = dest.with_extension("tmp");
            let mut file = std::fs::File::create(&tmp)?;
            std::io::copy(&mut entry, &mut file)
                .map_err(|e| ExtractError::Download(format!("Failed to extract pdfium: {e}")))?;
            file.flush()?;
            drop(file);
            std::fs::rename(&tmp, &dest)?;
            eprintln!("Pdfium extracted to {}", dest.display());
            return Ok(());
        }
    }

    Err(ExtractError::Download(format!(
        "Could not find {target_entry} in pdfium archive"
    )))
}

/// Platform-specific pdfium download URL.
fn pdfium_download_url() -> String {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-arm64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-mac-x64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-mac-arm64.tgz"
            .to_string()
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-linux-x64.tgz"
            .to_string()
    }
}

/// Path inside the .tgz archive where the pdfium library lives.
fn pdfium_archive_path() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "bin/pdfium.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "lib/libpdfium.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "lib/libpdfium.so"
    }
}

/// Filename of the pdfium library for the current platform.
fn pdfium_lib_filename() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "pdfium.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libpdfium.dylib"
    }
    #[cfg(target_os = "linux")]
    {
        "libpdfium.so"
    }
}
