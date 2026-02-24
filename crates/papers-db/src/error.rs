#[derive(thiserror::Error, Debug)]
pub enum DbError {
    #[error("LanceDB error: {0}")]
    LanceDb(#[from] lancedb::Error),
    #[error("Embedding error: {0}")]
    Embed(String),
    #[error("Arrow error: {0}")]
    Arrow(String),
    #[error("Scope error: {0}")]
    Scope(String),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Embed cache error: {0}")]
    Cache(String),
}
