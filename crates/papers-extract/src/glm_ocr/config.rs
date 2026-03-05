//! Constants and configuration for GLM-OCR.

// ── Model constants ──────────────────────────────────────────────────
//
// Image preprocessing and tokenizer constants are hardcoded (not in model metadata).
// Decoder architecture params (NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ) are
// read from the ONNX model's input shapes at init time — see `extract_decoder_params()`.

pub(crate) const PATCH_SIZE: u32 = 14;
pub(crate) const SPATIAL_MERGE: u32 = 2;
pub(crate) const TEMPORAL_PATCH_SIZE: usize = 2;
pub(crate) const GRID_UNIT: u32 = PATCH_SIZE * SPATIAL_MERGE; // 28
pub(crate) const MIN_PIXELS: u32 = 12544; // from processor_config.json size.shortest_edge
pub(crate) const MAX_PIXELS: u32 = 9633792; // from processor_config.json size.longest_edge
pub(crate) const PATCH_ELEM: usize =
    TEMPORAL_PATCH_SIZE * (PATCH_SIZE as usize) * (PATCH_SIZE as usize) * 3; // 1176
pub(crate) const HIDDEN_SIZE: usize = 1536;
pub(crate) const EOS_IDS: [i64; 2] = [59246, 59253];
pub(crate) const IMAGE_TOKEN_ID: i64 = 59280;
// Image normalization (CLIP/SigLip-derived, matching GLM-OCR processor)
pub(crate) const NORM_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub(crate) const NORM_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

// ── Decoder architecture params (extracted from ONNX model at init) ──

/// Decoder architecture parameters extracted from the ONNX model's input shapes.
pub(crate) struct DecoderParams {
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub max_seq: usize,
    pub head_dim: usize,
}

// ── Backend selection ─────────────────────────────────────────────────

/// Inference backend for GLM-OCR decode loop.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    /// CUDA with IoBinding + CUDA graphs (requires NVIDIA GPU, SM75+).
    /// Errors on macOS or if no suitable GPU is found.
    Cuda,
    /// CoreML (Apple Silicon Neural Engine / GPU).
    /// Errors on non-macOS platforms.
    CoreMl,
    /// CPU fallback — works everywhere but slowest.
    Cpu,
    /// Auto-detect: try CUDA → CoreML → CPU.
    #[default]
    Auto,
}

impl Backend {
    /// Parse from CLI string (case-insensitive, lenient).
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cuda" => Backend::Cuda,
            "coreml" | "core_ml" | "apple" => Backend::CoreMl,
            "cpu" => Backend::Cpu,
            _ => Backend::Auto,
        }
    }
}

/// Configuration for GLM-OCR predictor.
pub struct GlmOcrConfig {
    /// Prompt text (e.g. "Formula Recognition:" or a full-page OCR instruction).
    pub prompt: String,
    /// Inference backend (default: Auto).
    pub backend: Backend,
}

impl Default for GlmOcrConfig {
    fn default() -> Self {
        Self {
            prompt: "Formula Recognition:".to_string(),
            backend: Backend::Auto,
        }
    }
}
