use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::error::RagError;
#[cfg(test)]
use crate::schema::EMBED_DIM;

pub struct Embedder {
    model: Option<TextEmbedding>,
}

impl Embedder {
    /// Blocking constructor — call from spawn_blocking.
    /// Downloads model weights on first run from the HF Hub cache.
    pub fn new() -> Result<Self, RagError> {
        let model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::EmbeddingGemma300M))
            .map_err(|e| RagError::Embed(e.to_string()))?;
        Ok(Self { model: Some(model) })
    }

    /// Test-only: create an embedder that returns zero vectors without loading any model.
    #[cfg(test)]
    pub(crate) fn fake() -> Self {
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
                #[cfg(test)]
                return Ok(texts
                    .iter()
                    .map(|_| vec![0.0f32; EMBED_DIM as usize])
                    .collect());
                #[cfg(not(test))]
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
                #[cfg(test)]
                return Ok(vec![0.0f32; EMBED_DIM as usize]);
                #[cfg(not(test))]
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
