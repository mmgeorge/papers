use std::path::PathBuf;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::RagError;
#[cfg(any(test, feature = "bench"))]
use crate::schema::EMBED_DIM;

/// Human-readable name of the embedding model.
pub const MODEL_NAME: &str = "embedding-gemma-300m";

/// Name of the execution provider selected at compile time.
pub fn ep_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "DirectML"
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
    pub fn new() -> Result<Self, RagError> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("papers")
            .join("fastembed");

        let mut opts = InitOptions::new(EmbeddingModel::EmbeddingGemma300M)
            .with_cache_dir(cache_dir);

        #[cfg(target_os = "windows")]
        {
            opts =
                opts.with_execution_providers(vec![ort::ep::directml::DirectML::default().build()]);
        }
        #[cfg(target_os = "macos")]
        {
            opts = opts.with_execution_providers(vec![ort::ep::coreml::CoreML::default().build()]);
        }

        let model = TextEmbedding::try_new(opts).map_err(|e| RagError::Embed(e.to_string()))?;
        Ok(Self { model: Some(model) })
    }

    /// Test-only: create an embedder that returns zero vectors without loading any model.
    #[cfg(any(test, feature = "bench"))]
    pub fn fake() -> Self {
        Self { model: None }
    }

    /// Embed documents at ingest time.
    pub fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, RagError> {
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
            .map_err(|e| RagError::Embed(e.to_string()))
    }

    /// Embed a query at search time.
    pub fn embed_query(&mut self, query: &str) -> Result<Vec<f32>, RagError> {
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
            .map_err(|e| RagError::Embed(e.to_string()))?;
        result
            .into_iter()
            .next()
            .ok_or_else(|| RagError::Embed("empty embedding result".into()))
    }
}
