#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("PDF error: {0}")]
    Pdf(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Layout detection failed: {0}")]
    Layout(String),

    #[error("Table recognition failed: {0}")]
    Table(String),

    #[error("Formula recognition failed: {0}")]
    Formula(String),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Model download failed: {0}")]
    Download(String),

    #[error("Model not found: {path}")]
    ModelNotFound { path: String },

    #[error("Pdfium library not found")]
    PdfiumNotFound,
}
