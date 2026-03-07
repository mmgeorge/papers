use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::DbError;
#[cfg(any(test, feature = "bench"))]
use crate::schema::EMBED_DIM;

/// Human-readable name of the embedding model.
pub const MODEL_NAME: &str = "embedding-gemma-300m";

/// Name of the execution provider selected at compile time.
pub fn ep_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "CUDA"
    }
    #[cfg(target_os = "macos")]
    {
        "CoreML"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "CPU"
    }
}

/// Initialize ORT with the dynamically loaded GPU runtime (Windows only).
///
/// Searches `ORT_DYLIB_PATH` env var (set by `.cargo/config.toml`) then
/// `onnxruntime.dll` next to the executable. No-op on macOS (statically linked).
#[cfg(target_os = "windows")]
fn init_ort_runtime() -> Result<(), DbError> {
    use std::sync::Once;
    static INIT: Once = Once::new();
    let mut init_err: Option<String> = None;

    INIT.call_once(|| {
        let path = std::env::var("ORT_DYLIB_PATH").ok().or_else(|| {
            std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(|d| d.join("onnxruntime.dll")))
                .filter(|p| p.exists())
                .map(|p| p.to_string_lossy().into_owned())
        });

        match path {
            Some(path) => match ort::init_from(&path) {
                Ok(b) => {
                    b.commit();
                }
                Err(e) => {
                    init_err = Some(format!("ORT init failed for '{path}': {e}"));
                }
            },
            None => {
                init_err = Some(
                    "onnxruntime.dll not found (set ORT_DYLIB_PATH or place next to exe)".into(),
                );
            }
        }
    });

    if let Some(err) = init_err {
        Err(DbError::Embed(err))
    } else {
        Ok(())
    }
}

pub struct Embedder {
    model: Option<TextEmbedding>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish()
    }
}

impl Embedder {
    /// Blocking constructor — call from spawn_blocking.
    /// Downloads model weights on first run from the HF Hub cache.
    pub fn new() -> Result<Self, DbError> {
        #[cfg(target_os = "windows")]
        init_ort_runtime()?;

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("papers")
            .join("fastembed");

        let mut opts = InitOptions::new(EmbeddingModel::EmbeddingGemma300M)
            .with_cache_dir(cache_dir);

        #[cfg(target_os = "windows")]
        {
            opts =
                opts.with_execution_providers(vec![ort::ep::cuda::CUDA::default().build()]);
        }
        #[cfg(target_os = "macos")]
        {
            opts = opts.with_execution_providers(vec![ort::ep::coreml::CoreML::default().build()]);
        }

        let model = TextEmbedding::try_new(opts).map_err(|e| DbError::Embed(e.to_string()))?;
        Ok(Self { model: Some(model) })
    }

    /// Test-only: create an embedder that returns zero vectors without loading any model.
    #[cfg(any(test, feature = "bench"))]
    pub fn fake() -> Self {
        Self { model: None }
    }

    /// Embed documents at ingest time.
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, DbError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let model = match &mut self.model {
            Some(m) => m,
            None => {
                #[cfg(any(test, feature = "bench"))]
                return Ok(texts
                    .iter()
                    .map(|_| vec![0.0f32; EMBED_DIM as usize])
                    .collect());
                #[cfg(not(any(test, feature = "bench")))]
                unreachable!("Embedder has no model; Embedder::fake() is test-only");
            }
        };
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        model
            .embed(refs, None)
            .map_err(|e| DbError::Embed(e.to_string()))
    }

    /// Embed a query at search time.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, DbError> {
        let model = match &mut self.model {
            Some(m) => m,
            None => {
                #[cfg(any(test, feature = "bench"))]
                return Ok(vec![0.0f32; EMBED_DIM as usize]);
                #[cfg(not(any(test, feature = "bench")))]
                unreachable!("Embedder has no model; Embedder::fake() is test-only");
            }
        };
        let result = model
            .embed(vec![query], None)
            .map_err(|e| DbError::Embed(e.to_string()))?;
        result
            .into_iter()
            .next()
            .ok_or_else(|| DbError::Embed("empty embedding result".into()))
    }
}
