//! Chunking pipeline configuration.
//!
//! All token counts use the `estimate_tokens` heuristic (words * 1.3).

/// Minimum chunk size in estimated tokens. Chunks smaller than this at section
/// boundaries are merged into the previous chunk (if combined <= MAX). Prevents
/// orphan fragments that embed poorly.
pub const MIN_CHUNK_TOKENS: usize = 200;

/// Target chunk size in estimated tokens. The buffer flushes when the next
/// block would push it past this threshold.
pub const TARGET_CHUNK_TOKENS: usize = 400;

/// Hard upper limit in estimated tokens. Smart merge at section boundaries is
/// blocked if the combined chunk would exceed this.
pub const MAX_CHUNK_TOKENS: usize = 600;

/// Number of trailing sentences carried into the next buffer on token-limit
/// flush (not on section boundary flush).
pub const OVERLAP_SENTENCES: usize = 2;

/// Multiplier for word-to-token estimation.
pub const TOKEN_ESTIMATE_MULTIPLIER: f64 = 1.3;

/// Regex pattern for detecting algorithm-like headers in h5/h6 SectionHeaders.
pub const ALGO_HEADER_PATTERN: &str =
    r"(?i)^(?:Algorithm|Procedure|Pseudocode|Listing|Code)\s+\d+";

/// Section header titles (h2 level) that trigger references-skip mode.
pub const REFERENCES_TITLES: &[&str] = &["references", "bibliography"];
