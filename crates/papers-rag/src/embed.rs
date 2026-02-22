use candle_core::{DType, Device};
use fastembed::NomicV2MoeTextEmbedding;

use crate::error::RagError;
#[cfg(test)]
use crate::schema::EMBED_DIM;

pub struct Embedder {
    model: Option<NomicV2MoeTextEmbedding>,
}

impl Embedder {
    /// Blocking constructor — call from spawn_blocking.
    /// Downloads model weights on first run from the HF Hub cache.
    pub fn new() -> Result<Self, RagError> {
        let device = Device::new_cuda(0).unwrap();
        let model = NomicV2MoeTextEmbedding::from_hf(
            "nomic-ai/nomic-embed-text-v2-moe",
            &device,
            DType::F32,
            512,
        )
        .map_err(|e| RagError::Embed(e.to_string()))?;
        Ok(Self { model: Some(model) })
    }

    /// Test-only: create an embedder that returns zero vectors without loading any model.
    #[cfg(test)]
    pub(crate) fn fake() -> Self {
        Self { model: None }
    }

    /// Embed documents at ingest time. Prepends "search_document: " prefix.
    pub fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RagError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let model = match &self.model {
            Some(m) => m,
            None => {
                #[cfg(test)]
                return Ok(texts
                    .iter()
                    .map(|_| vec![0.0f32; EMBED_DIM as usize])
                    .collect());
                #[cfg(not(test))]
                unreachable!("Embedder has no model; Embedder::fake() is test-only");
            }
        };
        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("search_document: {t}"))
            .collect();
        let refs: Vec<&str> = prefixed.iter().map(|s| s.as_str()).collect();
        model
            .embed(&refs)
            .map_err(|e| RagError::Embed(e.to_string()))
    }

    /// Embed a query at search time. Prepends "search_query: " prefix.
    pub fn embed_query(&self, query: &str) -> Result<Vec<f32>, RagError> {
        let model = match &self.model {
            Some(m) => m,
            None => {
                #[cfg(test)]
                return Ok(vec![0.0f32; EMBED_DIM as usize]);
                #[cfg(not(test))]
                unreachable!("Embedder has no model; Embedder::fake() is test-only");
            }
        };
        let text = format!("search_query: {query}");
        let refs: &[&str] = &[text.as_str()];
        let result = model
            .embed(refs)
            .map_err(|e| RagError::Embed(e.to_string()))?;
        result
            .into_iter()
            .next()
            .ok_or_else(|| RagError::Embed("empty embedding result".into()))
    }
}
