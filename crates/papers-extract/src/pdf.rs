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

    for i in 0..chars_collection.len() {
        let Ok(char_info) = chars_collection.get(i) else {
            continue;
        };

        if let Some(c) = char_info.unicode_char() {
            if let Ok(rect) = char_info.loose_bounds() {
                chars.push(PdfChar {
                    codepoint: c,
                    bbox: [
                        rect.left().value,
                        rect.bottom().value,
                        rect.right().value,
                        rect.top().value,
                    ],
                });
            }
        }
    }

    Ok(chars)
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
