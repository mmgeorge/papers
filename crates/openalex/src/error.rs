/// Errors returned by [`OpenAlexClient`](crate::OpenAlexClient) methods.
///
/// # Variants
///
/// - [`Http`](OpenAlexError::Http) — network or connection failure (wraps
///   [`reqwest::Error`])
/// - [`Json`](OpenAlexError::Json) — response body could not be deserialized
///   (wraps [`serde_json::Error`])
/// - [`Api`](OpenAlexError::Api) — the OpenAlex API returned a non-success HTTP
///   status code (e.g. 404 for unknown entity, 403 for forbidden, 500 for
///   server error)
///
/// # Example
///
/// ```no_run
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// use openalex::{OpenAlexClient, GetParams, OpenAlexError};
///
/// let client = OpenAlexClient::new();
/// match client.get_work("nonexistent", &GetParams::default()).await {
///     Ok(work) => println!("Found: {}", work.id),
///     Err(OpenAlexError::Api { status, message }) => {
///         eprintln!("API error {status}: {message}");
///     }
///     Err(e) => eprintln!("Other error: {e}"),
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, thiserror::Error)]
pub enum OpenAlexError {
    /// Network or connection error from reqwest.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// Failed to deserialize the JSON response body.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// The API returned a non-success HTTP status code.
    ///
    /// `status` is the HTTP status code (e.g. 404, 403, 500) and `message`
    /// contains the response body text.
    #[error("API error (status {status}): {message}")]
    Api { status: u16, message: String },
}

/// A [`Result`](std::result::Result) alias with [`OpenAlexError`] as the error
/// type.
pub type Result<T> = std::result::Result<T, OpenAlexError>;
