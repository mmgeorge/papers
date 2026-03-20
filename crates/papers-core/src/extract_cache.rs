//! Thin filesystem layer for the papers-extract cache.
//!
//! Cache location: `<cache-dir>/papers/extracts/{cache_id}/`
//! Override with env var `PAPERS_EXTRACT_CACHE_DIR`.
//!
//! Each cache entry contains:
//! - `meta.json`        — paper metadata ([`super::text::ExtractionMeta`])
//! - `extraction.json`  — raw ExtractionResult from Stage 1
//! - `reflow.json`      — ReflowDocument tree from Stage 2
//! - `output.md`        — pre-rendered markdown
//! - `images/`          — cropped figure/table images

use std::path::{Path, PathBuf};

use crate::text::ExtractionMeta;

// ── Path helpers ─────────────────────────────────────────────────────────────

/// Return the base directory for extract caches.
///
/// Uses `PAPERS_EXTRACT_CACHE_DIR` if set, otherwise `<cache_dir>/papers/extracts/`.
pub fn extract_cache_root() -> Option<PathBuf> {
    if let Ok(base) = std::env::var("PAPERS_EXTRACT_CACHE_DIR") {
        return Some(PathBuf::from(base));
    }
    dirs::cache_dir().map(|d| d.join("papers").join("extracts"))
}

/// Return the cache directory for a specific paper.
pub fn extract_cache_dir(cache_id: &str) -> Option<PathBuf> {
    extract_cache_root().map(|root| root.join(cache_id))
}

// ── Read helpers ─────────────────────────────────────────────────────────────

/// Check whether a cache entry exists (has `reflow.json`).
pub fn extract_cached(cache_id: &str) -> bool {
    extract_cache_dir(cache_id)
        .map(|d| d.join("reflow.json").exists())
        .unwrap_or(false)
}

/// Read the cached `reflow.json` as a raw JSON string.
pub fn read_cached_reflow_json(cache_id: &str) -> Option<String> {
    let dir = extract_cache_dir(cache_id)?;
    std::fs::read_to_string(dir.join("reflow.json")).ok()
}

/// Read the cached `extraction.json` as a raw JSON string.
pub fn read_cached_extraction_json(cache_id: &str) -> Option<String> {
    let dir = extract_cache_dir(cache_id)?;
    std::fs::read_to_string(dir.join("extraction.json")).ok()
}

/// Read the cached `meta.json`.
pub fn read_cached_meta(cache_id: &str) -> Option<ExtractionMeta> {
    let dir = extract_cache_dir(cache_id)?;
    let bytes = std::fs::read(dir.join("meta.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Read the pre-rendered `output.md`.
pub fn read_cached_markdown(cache_id: &str) -> Option<String> {
    let dir = extract_cache_dir(cache_id)?;
    std::fs::read_to_string(dir.join("output.md")).ok()
}

/// List all cached item keys (subdirectories containing `reflow.json`).
pub fn list_cached_extract_keys() -> Vec<String> {
    let base = match extract_cache_root() {
        Some(d) => d,
        None => return vec![],
    };
    if !base.is_dir() {
        return vec![];
    }
    let mut keys = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&base) {
        for entry in entries.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let key = entry.file_name().to_string_lossy().into_owned();
            if entry.path().join("reflow.json").exists() {
                keys.push(key);
            }
        }
    }
    keys
}

// ── Write helpers ────────────────────────────────────────────────────────────

/// Write all cache files for a paper extraction.
///
/// Creates the cache directory on demand. Returns the cache directory path.
pub fn write_extract_cache(
    cache_id: &str,
    meta_json: &str,
    extraction_json: &str,
    reflow_json: &str,
    markdown: &str,
) -> Result<PathBuf, std::io::Error> {
    let dir = extract_cache_dir(cache_id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine extract cache directory",
        )
    })?;
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("meta.json"), meta_json)?;
    std::fs::write(dir.join("extraction.json"), extraction_json)?;
    std::fs::write(dir.join("reflow.json"), reflow_json)?;
    std::fs::write(dir.join("output.md"), markdown)?;
    Ok(dir)
}

/// Write a single file into an existing cache directory.
///
/// Useful for writing images after the initial cache write.
pub fn write_cache_file(cache_id: &str, relative_path: &Path, data: &[u8]) -> Result<(), std::io::Error> {
    let dir = extract_cache_dir(cache_id).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "cannot determine extract cache directory",
        )
    })?;
    let full_path = dir.join(relative_path);
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(full_path, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn setup_temp_cache() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        unsafe { std::env::set_var("PAPERS_EXTRACT_CACHE_DIR", dir.path()) };
        dir
    }

    #[test]
    #[serial]
    fn test_write_and_read_cache() {
        let _guard = setup_temp_cache();

        let dir = write_extract_cache(
            "TEST01",
            r#"{"item_key":"TEST01"}"#,
            r#"{"metadata":{},"pages":[]}"#,
            r#"{"title":"Test","toc":[],"children":[]}"#,
            "# Test Paper\n\nContent here.",
        ).unwrap();

        assert!(dir.join("meta.json").exists());
        assert!(dir.join("extraction.json").exists());
        assert!(dir.join("reflow.json").exists());
        assert!(dir.join("output.md").exists());

        assert!(extract_cached("TEST01"));
        assert!(!extract_cached("NONEXIST"));

        let md = read_cached_markdown("TEST01").unwrap();
        assert!(md.contains("Test Paper"));

        let reflow = read_cached_reflow_json("TEST01").unwrap();
        assert!(reflow.contains(r#""title":"Test""#));

        let extraction = read_cached_extraction_json("TEST01").unwrap();
        assert!(extraction.contains("\"pages\""));
    }

    #[test]
    #[serial]
    fn test_list_cached_keys() {
        let _guard = setup_temp_cache();

        write_extract_cache("AAA", "{}", "{}", "{}", "").unwrap();
        write_extract_cache("BBB", "{}", "{}", "{}", "").unwrap();

        let keys = list_cached_extract_keys();
        assert!(keys.contains(&"AAA".to_string()));
        assert!(keys.contains(&"BBB".to_string()));
        assert_eq!(keys.len(), 2);
    }

    #[test]
    #[serial]
    fn test_write_cache_file() {
        let _guard = setup_temp_cache();

        write_extract_cache("IMG01", "{}", "{}", "{}", "").unwrap();
        write_cache_file("IMG01", Path::new("images/fig1.png"), b"fake-png-data").unwrap();

        let dir = extract_cache_dir("IMG01").unwrap();
        assert!(dir.join("images/fig1.png").exists());
        let data = std::fs::read(dir.join("images/fig1.png")).unwrap();
        assert_eq!(data, b"fake-png-data");
    }

    #[test]
    #[serial]
    fn test_read_cached_meta() {
        let _guard = setup_temp_cache();

        let meta_json = r#"{"item_key":"META01","title":"My Paper","authors":["Alice","Bob"]}"#;
        write_extract_cache("META01", meta_json, "{}", "{}", "").unwrap();

        let meta = read_cached_meta("META01").unwrap();
        assert_eq!(meta.item_key, "META01");
        assert_eq!(meta.title.as_deref(), Some("My Paper"));
        assert_eq!(meta.authors.as_ref().unwrap().len(), 2);
    }

    #[test]
    #[serial]
    fn test_nonexistent_cache() {
        let _guard = setup_temp_cache();

        assert!(!extract_cached("NOPE"));
        assert!(read_cached_markdown("NOPE").is_none());
        assert!(read_cached_reflow_json("NOPE").is_none());
        assert!(read_cached_extraction_json("NOPE").is_none());
        assert!(read_cached_meta("NOPE").is_none());
    }
}
