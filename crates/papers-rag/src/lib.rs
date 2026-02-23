pub mod embed_cache;
pub mod error;
pub mod ingest;
pub mod query;
pub mod schema;
pub mod store;
pub mod types;

mod embed;
mod filter;

pub use embed_cache::EmbedCache;
pub use error::RagError;
pub use ingest::{
    cache_paper_embeddings, embed_cache_base, IngestParams, ingest_paper, ingest_params_from_cache,
    is_ingested, list_cached_item_keys,
};
pub use query::resolve_paper_id;
#[cfg(any(test, feature = "bench"))]
pub use query::{search_figures_with_embedding, search_with_embedding};
pub use store::RagStore;
pub use types::*;

/// Returns an `EmbedCache` pointed at the default on-disk location.
pub fn default_embed_cache() -> EmbedCache {
    EmbedCache::new(embed_cache_base())
}

#[cfg(test)]
mod tests;
