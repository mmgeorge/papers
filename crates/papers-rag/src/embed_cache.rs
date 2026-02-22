use serde::{Deserialize, Serialize};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem::size_of;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum EmbedCacheError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("cache already exists; use overwrite=true to replace")]
    AlreadyExists,
    #[error("dimension mismatch at chunk {chunk_idx}: expected {expected}, got {got}")]
    DimMismatch {
        chunk_idx: usize,
        expected: usize,
        got: usize,
    },
    #[error("chunk count mismatch: manifest has {manifest_chunks} chunks but {embedding_rows} embeddings provided")]
    ChunkCountMismatch {
        manifest_chunks: usize,
        embedding_rows: usize,
    },
}

/// Metadata for a single cached chunk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkRecord {
    pub chunk_id: String,
    pub text: String,
    pub section: String,
    pub heading: String,
    pub chapter_idx: u32,
    pub section_idx: u32,
    pub chunk_idx: u32,
    pub page_start: Option<u32>,
    pub page_end: Option<u32>,
}

/// The manifest stored alongside `embeddings.bin`.
#[derive(Debug, Serialize, Deserialize)]
pub struct EmbedManifest {
    pub model: String,
    pub dim: usize,
    /// RFC 3339 timestamp.
    pub created_at: String,
    pub chunks: Vec<ChunkRecord>,
}

/// Persistent per-paper embedding cache.
///
/// Layout: `<base_dir>/embeddings/<model>/<item_key>/manifest.json`
///                                                  `embeddings.bin`
///
/// `embeddings.bin` is a flat little-endian f32 array: `N * dim * 4` bytes total,
/// where row `i` maps to `manifest.chunks[i]`.
pub struct EmbedCache {
    base_dir: PathBuf,
}

impl EmbedCache {
    pub fn new(base_dir: PathBuf) -> Self {
        Self { base_dir }
    }

    /// `<base_dir>/embeddings/<model>/<item_key>/`
    pub fn cache_dir(&self, model: &str, item_key: &str) -> PathBuf {
        self.base_dir.join("embeddings").join(model).join(item_key)
    }

    /// `true` if both `manifest.json` and `embeddings.bin` exist.
    pub fn exists(&self, model: &str, item_key: &str) -> bool {
        let dir = self.cache_dir(model, item_key);
        dir.join("manifest.json").exists() && dir.join("embeddings.bin").exists()
    }

    /// Write `manifest.json` and `embeddings.bin` to disk.
    ///
    /// Validates that:
    /// - `embeddings.len() == manifest.chunks.len()`
    /// - every `embeddings[i].len() == manifest.dim`
    ///
    /// Returns `Err(AlreadyExists)` if files are present and `overwrite` is `false`.
    pub fn save(
        &self,
        model: &str,
        item_key: &str,
        manifest: &EmbedManifest,
        embeddings: &[Vec<f32>],
        overwrite: bool,
    ) -> Result<(), EmbedCacheError> {
        if manifest.chunks.len() != embeddings.len() {
            return Err(EmbedCacheError::ChunkCountMismatch {
                manifest_chunks: manifest.chunks.len(),
                embedding_rows: embeddings.len(),
            });
        }
        for (i, emb) in embeddings.iter().enumerate() {
            if emb.len() != manifest.dim {
                return Err(EmbedCacheError::DimMismatch {
                    chunk_idx: i,
                    expected: manifest.dim,
                    got: emb.len(),
                });
            }
        }

        if !overwrite && self.exists(model, item_key) {
            return Err(EmbedCacheError::AlreadyExists);
        }

        let dir = self.cache_dir(model, item_key);
        std::fs::create_dir_all(&dir)?;

        // Write manifest
        let manifest_json = serde_json::to_vec_pretty(manifest)?;
        std::fs::write(dir.join("manifest.json"), manifest_json)?;

        // Write binary embeddings: flat little-endian f32
        let mut bin_file = std::fs::File::create(dir.join("embeddings.bin"))?;
        for emb in embeddings {
            for &f in emb {
                bin_file.write_all(&f.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Load `manifest.json`. Returns `Ok(None)` if not cached.
    pub fn load_manifest(
        &self,
        model: &str,
        item_key: &str,
    ) -> Result<Option<EmbedManifest>, EmbedCacheError> {
        let path = self.cache_dir(model, item_key).join("manifest.json");
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(path)?;
        let manifest = serde_json::from_slice(&bytes)?;
        Ok(Some(manifest))
    }

    /// Load all embeddings by reading `embeddings.bin` sequentially.
    pub fn load_embeddings(
        &self,
        model: &str,
        item_key: &str,
        manifest: &EmbedManifest,
    ) -> Result<Vec<Vec<f32>>, EmbedCacheError> {
        let path = self.cache_dir(model, item_key).join("embeddings.bin");
        let mut file = std::fs::File::open(path)?;
        let n = manifest.chunks.len();
        let dim = manifest.dim;
        let mut result = Vec::with_capacity(n);
        let mut buf = vec![0u8; dim * size_of::<f32>()];
        for _ in 0..n {
            file.read_exact(&mut buf)?;
            let floats: Vec<f32> = buf
                .chunks_exact(size_of::<f32>())
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            result.push(floats);
        }
        Ok(result)
    }

    /// Read a single embedding by seeking to offset `chunk_index * dim * 4`.
    pub fn load_embedding_at(
        &self,
        model: &str,
        item_key: &str,
        chunk_index: usize,
        dim: usize,
    ) -> Result<Vec<f32>, EmbedCacheError> {
        let path = self.cache_dir(model, item_key).join("embeddings.bin");
        let mut file = std::fs::File::open(path)?;
        let offset = (chunk_index * dim * size_of::<f32>()) as u64;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; dim * size_of::<f32>()];
        file.read_exact(&mut buf)?;
        let floats: Vec<f32> = buf
            .chunks_exact(size_of::<f32>())
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Ok(floats)
    }

    /// Remove `<base_dir>/embeddings/<model>/<item_key>/`. No-op if not present.
    pub fn delete(&self, model: &str, item_key: &str) -> Result<(), EmbedCacheError> {
        let dir = self.cache_dir(model, item_key);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    /// List model subdirectory names that have a complete cache for `item_key`.
    pub fn list_models(&self, item_key: &str) -> Result<Vec<String>, EmbedCacheError> {
        let embeddings_dir = self.base_dir.join("embeddings");
        if !embeddings_dir.is_dir() {
            return Ok(vec![]);
        }
        let mut models = Vec::new();
        for entry in std::fs::read_dir(&embeddings_dir)?.flatten() {
            if !entry.path().is_dir() {
                continue;
            }
            let model_name = match entry.file_name().to_str() {
                Some(n) => n.to_string(),
                None => continue,
            };
            let item_dir = entry.path().join(item_key);
            if item_dir.join("manifest.json").exists() && item_dir.join("embeddings.bin").exists() {
                models.push(model_name);
            }
        }
        models.sort();
        Ok(models)
    }

    /// List all `(item_key, model)` pairs that have a complete cache.
    pub fn list_all(&self) -> Result<Vec<(String, String)>, EmbedCacheError> {
        let embeddings_dir = self.base_dir.join("embeddings");
        if !embeddings_dir.is_dir() {
            return Ok(vec![]);
        }
        let mut result = Vec::new();
        for model_entry in std::fs::read_dir(&embeddings_dir)?.flatten() {
            if !model_entry.path().is_dir() {
                continue;
            }
            let model_name = match model_entry.file_name().to_str() {
                Some(n) => n.to_string(),
                None => continue,
            };
            for item_entry in std::fs::read_dir(model_entry.path())?.flatten() {
                if !item_entry.path().is_dir() {
                    continue;
                }
                let item_key = match item_entry.file_name().to_str() {
                    Some(k) => k.to_string(),
                    None => continue,
                };
                if item_entry.path().join("manifest.json").exists()
                    && item_entry.path().join("embeddings.bin").exists()
                {
                    result.push((item_key, model_name.clone()));
                }
            }
        }
        result.sort();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const MODEL: &str = "nomic-embed-text-v2-moe";
    const DIM: usize = 768;

    fn make_manifest(dim: usize, n: usize) -> EmbedManifest {
        let chunks = (0..n)
            .map(|i| ChunkRecord {
                chunk_id: format!("ABCD/{i}"),
                text: format!("chunk text {i}"),
                section: "Background".to_string(),
                heading: "Introduction".to_string(),
                chapter_idx: 1,
                section_idx: 0,
                chunk_idx: i as u32,
                page_start: Some(i as u32 + 1),
                page_end: Some(i as u32 + 2),
            })
            .collect();
        EmbedManifest {
            model: MODEL.to_string(),
            dim,
            created_at: "2026-02-22T00:00:00Z".to_string(),
            chunks,
        }
    }

    fn make_embeddings(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32 * 0.001).collect())
            .collect()
    }

    // ── exists / round-trip ──────────────────────────────────────────────────

    #[test]
    fn test_exists_false_before_save() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        assert!(!cache.exists(MODEL, "ABCD1234"));
    }

    #[test]
    fn test_exists_true_after_save() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let manifest = make_manifest(DIM, 3);
        let embeddings = make_embeddings(3, DIM);
        cache.save(MODEL, "ABCD1234", &manifest, &embeddings, false).unwrap();
        assert!(cache.exists(MODEL, "ABCD1234"));
    }

    #[test]
    fn test_roundtrip_small() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let n = 3;
        let manifest = make_manifest(DIM, n);
        let embeddings = make_embeddings(n, DIM);
        cache.save(MODEL, "ABCD1234", &manifest, &embeddings, false).unwrap();

        let loaded_manifest = cache.load_manifest(MODEL, "ABCD1234").unwrap().unwrap();
        let loaded_embeddings = cache.load_embeddings(MODEL, "ABCD1234", &loaded_manifest).unwrap();

        assert_eq!(loaded_embeddings.len(), n);
        for (orig, loaded) in embeddings.iter().zip(loaded_embeddings.iter()) {
            assert_eq!(orig.len(), loaded.len());
            for (a, b) in orig.iter().zip(loaded.iter()) {
                assert_eq!(a.to_bits(), b.to_bits()); // exact f32 comparison
            }
        }
    }

    #[test]
    fn test_roundtrip_single() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let manifest = make_manifest(DIM, 1);
        let embeddings = make_embeddings(1, DIM);
        cache.save(MODEL, "KEY1", &manifest, &embeddings, false).unwrap();
        let m = cache.load_manifest(MODEL, "KEY1").unwrap().unwrap();
        let embs = cache.load_embeddings(MODEL, "KEY1", &m).unwrap();
        assert_eq!(embs.len(), 1);
        assert_eq!(embs[0].len(), DIM);
    }

    #[test]
    fn test_roundtrip_empty() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let manifest = make_manifest(DIM, 0);
        let embeddings: Vec<Vec<f32>> = vec![];
        cache.save(MODEL, "EMPTY", &manifest, &embeddings, false).unwrap();

        let bin_path = cache.cache_dir(MODEL, "EMPTY").join("embeddings.bin");
        assert_eq!(std::fs::metadata(&bin_path).unwrap().len(), 0);

        let m = cache.load_manifest(MODEL, "EMPTY").unwrap().unwrap();
        let embs = cache.load_embeddings(MODEL, "EMPTY", &m).unwrap();
        assert!(embs.is_empty());
    }

    #[test]
    fn test_binary_file_size() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let n = 5;
        let manifest = make_manifest(DIM, n);
        let embeddings = make_embeddings(n, DIM);
        cache.save(MODEL, "KEY_SIZE", &manifest, &embeddings, false).unwrap();

        let bin_path = cache.cache_dir(MODEL, "KEY_SIZE").join("embeddings.bin");
        let size = std::fs::metadata(&bin_path).unwrap().len();
        assert_eq!(size, (n * DIM * size_of::<f32>()) as u64);
    }

    #[test]
    fn test_binary_offset_seek() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let n = 4;
        let manifest = make_manifest(DIM, n);
        let embeddings = make_embeddings(n, DIM);
        cache.save(MODEL, "KEY_SEEK", &manifest, &embeddings, false).unwrap();

        let m = cache.load_manifest(MODEL, "KEY_SEEK").unwrap().unwrap();
        let all = cache.load_embeddings(MODEL, "KEY_SEEK", &m).unwrap();
        let single = cache.load_embedding_at(MODEL, "KEY_SEEK", 1, DIM).unwrap();

        for (a, b) in all[1].iter().zip(single.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn test_binary_endianness() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        // Known value: 1.0f32 in little-endian is [0x00, 0x00, 0x80, 0x3F]
        let val: f32 = 1.0;
        let manifest = EmbedManifest {
            model: MODEL.to_string(),
            dim: 1,
            created_at: "2026-02-22T00:00:00Z".to_string(),
            chunks: vec![ChunkRecord {
                chunk_id: "c0".to_string(),
                text: "t".to_string(),
                section: String::new(),
                heading: String::new(),
                chapter_idx: 0,
                section_idx: 0,
                chunk_idx: 0,
                page_start: None,
                page_end: None,
            }],
        };
        cache
            .save(MODEL, "KEY_ENDIAN", &manifest, &[vec![val]], false)
            .unwrap();

        let bin_path = cache.cache_dir(MODEL, "KEY_ENDIAN").join("embeddings.bin");
        let raw = std::fs::read(&bin_path).unwrap();
        assert_eq!(raw, &[0x00, 0x00, 0x80, 0x3F]);
    }

    #[test]
    fn test_manifest_fields_present() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let manifest = make_manifest(DIM, 1);
        let embeddings = make_embeddings(1, DIM);
        cache.save(MODEL, "KEY_FIELDS", &manifest, &embeddings, false).unwrap();

        let m = cache.load_manifest(MODEL, "KEY_FIELDS").unwrap().unwrap();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.dim, DIM);
        assert!(!m.created_at.is_empty());
        assert_eq!(m.chunks.len(), 1);
        let c = &m.chunks[0];
        assert!(!c.chunk_id.is_empty());
        assert!(!c.text.is_empty());
        assert_eq!(c.chapter_idx, 1);
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let result = cache.load_manifest(MODEL, "MISSING_KEY").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_model_isolation() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let key = "ISO_KEY";
        let m1 = make_manifest(DIM, 2);
        let e1 = make_embeddings(2, DIM);
        let m2 = EmbedManifest {
            model: "other-model".to_string(),
            dim: 64,
            created_at: "2026-02-22T00:00:00Z".to_string(),
            chunks: vec![ChunkRecord {
                chunk_id: "x".to_string(),
                text: "y".to_string(),
                section: String::new(),
                heading: String::new(),
                chapter_idx: 0,
                section_idx: 0,
                chunk_idx: 0,
                page_start: None,
                page_end: None,
            }],
        };
        let e2 = vec![vec![0.0f32; 64]];
        cache.save(MODEL, key, &m1, &e1, false).unwrap();
        cache.save("other-model", key, &m2, &e2, false).unwrap();

        // Model dirs are separate
        assert!(cache.cache_dir(MODEL, key).join("embeddings.bin").exists());
        assert!(cache.cache_dir("other-model", key).join("embeddings.bin").exists());

        // Each model's manifest is distinct
        let loaded1 = cache.load_manifest(MODEL, key).unwrap().unwrap();
        let loaded2 = cache.load_manifest("other-model", key).unwrap().unwrap();
        assert_eq!(loaded1.dim, DIM);
        assert_eq!(loaded2.dim, 64);
    }

    #[test]
    fn test_item_key_isolation() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let m = make_manifest(DIM, 1);
        let e = make_embeddings(1, DIM);
        cache.save(MODEL, "KEY_A", &m, &e, false).unwrap();
        cache.save(MODEL, "KEY_B", &m, &e, false).unwrap();
        assert!(cache.exists(MODEL, "KEY_A"));
        assert!(cache.exists(MODEL, "KEY_B"));
    }

    #[test]
    fn test_overwrite_false_errors() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let m = make_manifest(DIM, 1);
        let e = make_embeddings(1, DIM);
        cache.save(MODEL, "OW_KEY", &m, &e, false).unwrap();
        let err = cache.save(MODEL, "OW_KEY", &m, &e, false).unwrap_err();
        assert!(matches!(err, EmbedCacheError::AlreadyExists));
    }

    #[test]
    fn test_overwrite_true_succeeds() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let m = make_manifest(DIM, 1);
        let e = make_embeddings(1, DIM);
        cache.save(MODEL, "OW_KEY2", &m, &e, false).unwrap();
        // Second save with overwrite=true should succeed
        cache.save(MODEL, "OW_KEY2", &m, &e, true).unwrap();
        assert!(cache.exists(MODEL, "OW_KEY2"));
    }

    #[test]
    fn test_delete_removes_dir() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let m = make_manifest(DIM, 1);
        let e = make_embeddings(1, DIM);
        cache.save(MODEL, "DEL_KEY", &m, &e, false).unwrap();
        assert!(cache.exists(MODEL, "DEL_KEY"));
        cache.delete(MODEL, "DEL_KEY").unwrap();
        assert!(!cache.exists(MODEL, "DEL_KEY"));
        assert!(!cache.cache_dir(MODEL, "DEL_KEY").exists());
    }

    #[test]
    fn test_delete_noop_if_missing() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        // Should not error when nothing exists
        cache.delete(MODEL, "NEVER_EXISTED").unwrap();
    }

    #[test]
    fn test_list_models_single() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let m = make_manifest(DIM, 1);
        let e = make_embeddings(1, DIM);
        cache.save(MODEL, "LIST_KEY", &m, &e, false).unwrap();
        let models = cache.list_models("LIST_KEY").unwrap();
        assert_eq!(models, vec![MODEL.to_string()]);
    }

    #[test]
    fn test_list_models_empty() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let models = cache.list_models("NO_KEY").unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn test_list_all() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let m1 = make_manifest(DIM, 1);
        let e1 = make_embeddings(1, DIM);
        let m2 = make_manifest(DIM, 1);
        let e2 = make_embeddings(1, DIM);
        cache.save(MODEL, "PAPER_A", &m1, &e1, false).unwrap();
        cache.save(MODEL, "PAPER_B", &m2, &e2, false).unwrap();

        let mut all = cache.list_all().unwrap();
        all.sort();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&("PAPER_A".to_string(), MODEL.to_string())));
        assert!(all.contains(&("PAPER_B".to_string(), MODEL.to_string())));
    }

    #[test]
    fn test_dim_mismatch() {
        let dir = TempDir::new().unwrap();
        let cache = EmbedCache::new(dir.path().to_path_buf());
        let manifest = make_manifest(DIM, 2);
        // Give wrong dimension for second embedding
        let embeddings = vec![vec![0.0f32; DIM], vec![0.0f32; 64]];
        let err = cache.save(MODEL, "DIM_KEY", &manifest, &embeddings, false).unwrap_err();
        assert!(matches!(
            err,
            EmbedCacheError::DimMismatch { chunk_idx: 1, expected: DIM, got: 64 }
        ));
    }
}
