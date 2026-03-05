//! GLM-OCR formula predictor — autoregressive decoder with multiple backend support.
//!
//! ## CUDA backend (Windows, NVIDIA GPU)
//! Uses 4-part ONNX model split:
//!   - vision_encoder_mha.onnx  (BF16, CogViT with M-RoPE, MHA-fused)
//!   - embedding.onnx           (BF16, token embeddings for prefill)
//!   - llm.onnx                 (BF16, full LLM for prefill pass)
//!   - llm_decoder_gqa.onnx     (BF16, fixed-shape decode step with GQA / FlashAttention V2)
//!
//! Persistent IoBinding, pre-allocated GPU buffers, ORT built-in CUDA graph.
//!
//! ## CPU/CoreML backend (all platforms)
//! Uses 3-part ONNX model split (no separate decoder):
//!   - vision_encoder.onnx  (FP32, no MHA fusion)
//!   - embedding.onnx       (FP32, token embeddings)
//!   - llm.onnx             (FP32, dynamic KV cache for both prefill and decode)
//!
//! session.run()-based decode loop, growing KV cache per step.

mod config;
#[cfg(target_os = "windows")]
mod cuda;
mod other;
mod preprocess;
mod prefill;
mod session;

use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

use crate::error::ExtractError;

pub use config::{Backend, GlmOcrConfig};
use config::*;
use preprocess::*;
use prefill::*;
use session::build_session;

// ── Prompt tokens ─────────────────────────────────────────────────────

/// Tokenize the prompt template once at init, returning (prefix, suffix).
fn tokenize_prompt_parts(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
) -> Result<(Vec<i64>, Vec<i64>), ExtractError> {
    // Chat template: [gMASK]<sop><|user|>\n<|begin_of_image|>{IMAGE_TOKENS}<|end_of_image|>{PROMPT}<|assistant|>\n
    let prefix: Vec<i64> = vec![59248, 59250, 59253, 10, 59256]; // [gMASK]<sop><|user|>\n<|begin_of_image|>

    let prompt_tokens = tokenizer
        .encode(prompt, false)
        .map_err(|e| ExtractError::Model(format!("Tokenize prompt: {e}")))?;
    let prompt_ids: Vec<i64> = prompt_tokens.get_ids().iter().map(|&id| id as i64).collect();

    let mut suffix = vec![59257_i64]; // <|end_of_image|>
    suffix.extend_from_slice(&prompt_ids);
    suffix.push(59254); // <|assistant|>
    suffix.push(10); // \n

    Ok((prefix, suffix))
}

// ── Token decoding ────────────────────────────────────────────────────

/// Convert token IDs to a LaTeX string, stripping special tokens and delimiters.
fn decode_tokens(tokenizer: &tokenizers::Tokenizer, token_ids: &[i64]) -> String {
    let vocab_size = tokenizer.get_vocab_size(true) as i64;
    let valid: Vec<u32> = token_ids
        .iter()
        .copied()
        .take_while(|t| !EOS_IDS.contains(t))
        .filter(|&t| t >= 0 && t < vocab_size)
        .map(|t| t as u32)
        .collect();

    if valid.is_empty() {
        return String::new();
    }

    let text = tokenizer.decode(&valid, true).unwrap_or_default();

    // Strip surrounding $$ or $ delimiters that the model may produce
    let trimmed = text.trim();
    if trimmed.len() >= 4 && trimmed.starts_with("$$") && trimmed.ends_with("$$") {
        trimmed[2..trimmed.len() - 2].trim().to_string()
    } else if trimmed.len() >= 2 && trimmed.starts_with('$') && trimmed.ends_with('$') {
        trimmed[1..trimmed.len() - 1].trim().to_string()
    } else {
        trimmed.to_string()
    }
}

// ── Detect pixel dtype from vision encoder ────────────────────────────

/// Returns true if the vision encoder expects bf16 pixel values.
fn vision_expects_bf16(session: &Session) -> bool {
    if let Some(input) = session.inputs().first() {
        let dtype_str = format!("{:?}", input.dtype());
        dtype_str.contains("Bfloat16") || dtype_str.contains("bf16")
    } else {
        // Default to bf16 for backward compat
        true
    }
}

// ── Backend-specific enum ─────────────────────────────────────────────

enum ActiveBackend {
    #[cfg(target_os = "windows")]
    Cuda {
        decoder: Mutex<cuda::DecoderState>,
        num_layers: usize,
        num_kv_heads: usize,
        max_seq: usize,
        head_dim: usize,
        #[allow(dead_code)]
        cuda_ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    },
    Other {
        num_layers: usize,
        num_kv_heads: usize,
        max_seq: usize,
        head_dim: usize,
    },
}

// ── Resolved backend (internal) ──────────────────────────────────────

enum ResolvedBackend {
    #[cfg(target_os = "windows")]
    Cuda,
    /// CPU or CoreML — both use session.run() decode loop.
    Other,
}

// ── Main predictor struct ─────────────────────────────────────────────

/// GLM-OCR predictor with autoregressive decoding.
pub struct GlmOcrPredictor {
    vision_encoder: Mutex<Session>,
    embedding: Mutex<Session>,
    llm: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    prompt_prefix: Vec<i64>,
    prompt_suffix: Vec<i64>,
    use_bf16: bool,
    backend: ActiveBackend,
    #[cfg(target_os = "windows")]
    #[allow(dead_code)]
    cuda_mem: ort::memory::MemoryInfo,
}

impl GlmOcrPredictor {
    /// Create a new GLM-OCR predictor with default config (formula recognition prompt).
    pub fn new(
        vision_encoder_path: &Path,
        embedding_path: &Path,
        llm_path: &Path,
        decoder_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, ExtractError> {
        Self::with_config(
            vision_encoder_path,
            embedding_path,
            llm_path,
            decoder_path,
            tokenizer_path,
            GlmOcrConfig::default(),
        )
    }

    /// Create a new GLM-OCR predictor with custom config.
    pub fn with_config(
        vision_encoder_path: &Path,
        embedding_path: &Path,
        llm_path: &Path,
        decoder_path: &Path,
        tokenizer_path: &Path,
        config: GlmOcrConfig,
    ) -> Result<Self, ExtractError> {
        let resolved_backend = Self::resolve_backend(config.backend, decoder_path)?;

        // Vision encoder — Level2 (higher opt levels break MHA nodes on CUDA)
        let vision_ep = Self::make_ep(&resolved_backend);
        let vision_encoder = build_session(
            vision_encoder_path,
            GraphOptimizationLevel::Level2,
            vision_ep,
        )?;

        let use_bf16 = vision_expects_bf16(&vision_encoder);
        tracing::debug!("Vision encoder expects bf16: {use_bf16}");

        // Embedding + LLM
        let embed_ep = Self::make_ep(&resolved_backend);
        let embedding = build_session(
            embedding_path,
            GraphOptimizationLevel::Level3,
            embed_ep,
        )?;

        let llm_ep = Self::make_ep(&resolved_backend);
        let llm = build_session(
            llm_path,
            GraphOptimizationLevel::Level3,
            llm_ep,
        )?;

        // Build backend-specific state
        let active_backend = match &resolved_backend {
            #[cfg(target_os = "windows")]
            ResolvedBackend::Cuda => {
                let cuda_ctx = cudarc::driver::CudaContext::new(0)
                    .map_err(|e| ExtractError::Model(format!("cudarc CudaContext: {e}")))?;

                let cuda_mem = ort::memory::MemoryInfo::new(
                    ort::memory::AllocationDevice::CUDA,
                    0,
                    ort::memory::AllocatorType::Device,
                    ort::memory::MemoryType::Default,
                )
                .map_err(|e| ExtractError::Model(format!("CUDA MemoryInfo: {e}")))?;

                let decoder_ep = ort::execution_providers::CUDAExecutionProvider::default()
                    .with_cuda_graph(true)
                    .build();
                let decoder_session = build_session(
                    decoder_path,
                    GraphOptimizationLevel::Level3,
                    decoder_ep,
                )?;

                let decoder_params = extract_decoder_params(&decoder_session)?;
                let decoder_state =
                    cuda::build_decoder_state(decoder_session, &cuda_mem, &decoder_params)?;

                ActiveBackend::Cuda {
                    decoder: Mutex::new(decoder_state),
                    num_layers: decoder_params.num_layers,
                    num_kv_heads: decoder_params.num_kv_heads,
                    max_seq: decoder_params.max_seq,
                    head_dim: decoder_params.head_dim,
                    cuda_ctx,
                }
            }
            ResolvedBackend::Other => {
                // Read architecture from LLM session inputs
                let num_layers = llm
                    .inputs()
                    .iter()
                    .filter(|inp| inp.name().starts_with("past_key_"))
                    .count();
                let (num_kv_heads, head_dim) = if let Some(past_key_0) =
                    llm.inputs().iter().find(|inp| inp.name() == "past_key_0")
                {
                    if let Some(shape) = past_key_0.dtype().tensor_shape() {
                        let dims: Vec<i64> = shape.iter().copied().collect();
                        if dims.len() == 4 {
                            (dims[1] as usize, dims[3] as usize)
                        } else {
                            (8, 128) // fallback
                        }
                    } else {
                        (8, 128) // fallback
                    }
                } else {
                    (8, 128) // fallback
                };

                ActiveBackend::Other {
                    num_layers,
                    num_kv_heads,
                    max_seq: 512,
                    head_dim,
                }
            }
        };

        // CUDA MemoryInfo (only needed on Windows for CUDA backend)
        #[cfg(target_os = "windows")]
        let cuda_mem = match &active_backend {
            ActiveBackend::Cuda { .. } => {
                // Already created above, but we need a second one for the struct.
                // The one inside Cuda variant is used for decoder state.
                ort::memory::MemoryInfo::new(
                    ort::memory::AllocationDevice::CUDA,
                    0,
                    ort::memory::AllocatorType::Device,
                    ort::memory::MemoryType::Default,
                )
                .map_err(|e| ExtractError::Model(format!("CUDA MemoryInfo: {e}")))?
            }
            _ => ort::memory::MemoryInfo::new(
                ort::memory::AllocationDevice::CPU,
                0,
                ort::memory::AllocatorType::Device,
                ort::memory::MemoryType::Default,
            )
            .map_err(|e| ExtractError::Model(format!("CPU MemoryInfo: {e}")))?,
        };

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            ExtractError::Model(format!(
                "Tokenizer load from {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let (prompt_prefix, prompt_suffix) = tokenize_prompt_parts(&tokenizer, &config.prompt)?;

        let predictor = Self {
            vision_encoder: Mutex::new(vision_encoder),
            embedding: Mutex::new(embedding),
            llm: Mutex::new(llm),
            tokenizer,
            prompt_prefix,
            prompt_suffix,
            use_bf16,
            backend: active_backend,
            #[cfg(target_os = "windows")]
            cuda_mem,
        };

        predictor.warmup()?;
        Ok(predictor)
    }

    // ── Backend resolution ────────────────────────────────────────────

    fn resolve_backend(requested: Backend, decoder_path: &Path) -> Result<ResolvedBackend, ExtractError> {
        match requested {
            Backend::Cuda => {
                #[cfg(target_os = "macos")]
                return Err(ExtractError::Model(
                    "CUDA backend is not available on macOS".into(),
                ));
                #[cfg(not(target_os = "macos"))]
                {
                    if !decoder_path.exists() {
                        return Err(ExtractError::Model(format!(
                            "CUDA backend requires decoder model at {}",
                            decoder_path.display()
                        )));
                    }
                    Ok(ResolvedBackend::Cuda)
                }
            }
            Backend::CoreMl => {
                #[cfg(not(target_os = "macos"))]
                return Err(ExtractError::Model(
                    "CoreML backend is only available on macOS".into(),
                ));
                #[cfg(target_os = "macos")]
                Ok(ResolvedBackend::Other)
            }
            Backend::Cpu => Ok(ResolvedBackend::Other),
            Backend::Auto => {
                // Try CUDA (Windows with decoder model)
                #[cfg(target_os = "windows")]
                if decoder_path.exists() {
                    tracing::info!("Auto-detected CUDA backend");
                    return Ok(ResolvedBackend::Cuda);
                }
                // Try CoreML (macOS)
                #[cfg(target_os = "macos")]
                {
                    tracing::info!("Auto-detected CoreML backend");
                    return Ok(ResolvedBackend::Other);
                }
                // Fallback to CPU
                #[cfg(not(target_os = "macos"))]
                {
                    tracing::info!("Using CPU backend");
                    Ok(ResolvedBackend::Other)
                }
            }
        }
    }

    fn make_ep(backend: &ResolvedBackend) -> ort::execution_providers::ExecutionProviderDispatch {
        match backend {
            #[cfg(target_os = "windows")]
            ResolvedBackend::Cuda => {
                ort::execution_providers::CUDAExecutionProvider::default().build()
            }
            ResolvedBackend::Other => {
                #[cfg(target_os = "macos")]
                {
                    ort::execution_providers::CoreMLExecutionProvider::default().build()
                }
                #[cfg(not(target_os = "macos"))]
                {
                    ort::execution_providers::CPUExecutionProvider::default().build()
                }
            }
        }
    }

    // ── Warmup ─────────────────────────────────────────────────────────

    fn warmup(&self) -> Result<(), ExtractError> {
        tracing::info!("Warming up GLM-OCR predictor (3 iterations)...");
        let dummy = DynamicImage::new_rgb8(64, 64);
        for _ in 0..3 {
            let _ = self.predict_one(&dummy)?;
        }
        tracing::info!("GLM-OCR predictor warmup complete");
        Ok(())
    }

    // ── Public API ────────────────────────────────────────────────────

    /// Predict LaTeX for a batch of cropped formula images.
    pub fn predict(&self, images: &[DynamicImage]) -> Result<Vec<String>, ExtractError> {
        images.iter().map(|img| self.predict_one(img)).collect()
    }

    /// Predict LaTeX for a single formula image.
    fn predict_one(&self, image: &DynamicImage) -> Result<String, ExtractError> {
        // 1. Preprocess image → RGB normalized
        let (pixel_values, grid_thw) = preprocess_image(image);

        // 2. Vision encoder
        let vision_embeds =
            run_vision_encoder(&self.vision_encoder, &pixel_values, &grid_thw, self.use_bf16)?;

        // 3. Build input_ids for the full prompt
        let num_vision_tokens = {
            let t = grid_thw[[0, 0]] as usize;
            let h = grid_thw[[0, 1]] as usize;
            let w = grid_thw[[0, 2]] as usize;
            let merged_h = h / SPATIAL_MERGE as usize;
            let merged_w = w / SPATIAL_MERGE as usize;
            t * merged_h * merged_w
        };

        let mut input_ids: Vec<i64> = Vec::with_capacity(
            self.prompt_prefix.len() + num_vision_tokens + self.prompt_suffix.len(),
        );
        input_ids.extend_from_slice(&self.prompt_prefix);
        input_ids.extend(std::iter::repeat(IMAGE_TOKEN_ID).take(num_vision_tokens));
        input_ids.extend_from_slice(&self.prompt_suffix);

        let seq_len = input_ids.len();

        // 4. Embed tokens
        let token_embeds = run_embedding(&self.embedding, &input_ids)?;

        // 5. Merge vision embeddings into token embeddings
        let inputs_embeds = merge_vision_embeddings(
            token_embeds,
            &vision_embeds,
            &input_ids,
            seq_len,
            self.use_bf16,
        )?;

        // 6. Build 3D M-RoPE position IDs
        let position_ids = build_position_ids(&input_ids, &grid_thw);

        // 7. Prefill + decode (backend-specific)
        let token_ids = match &self.backend {
            #[cfg(target_os = "windows")]
            ActiveBackend::Cuda {
                decoder,
                num_layers,
                num_kv_heads,
                max_seq,
                head_dim,
                ..
            } => {
                let (first_token, kv_cache) = run_prefill_for_cuda(
                    &self.llm,
                    inputs_embeds,
                    &position_ids,
                    seq_len,
                    *num_layers,
                    *num_kv_heads,
                    *head_dim,
                )?;

                let mut state = decoder
                    .lock()
                    .map_err(|e| ExtractError::Formula(format!("decoder lock: {e}")))?;
                cuda::decode_loop(
                    &mut state,
                    first_token,
                    &kv_cache,
                    seq_len,
                    *num_kv_heads,
                    *max_seq,
                    *head_dim,
                )?
            }
            ActiveBackend::Other {
                num_layers,
                num_kv_heads,
                max_seq,
                head_dim,
            } => {
                let (first_token, kv_cache) = run_prefill_for_other(
                    &self.llm,
                    inputs_embeds,
                    &position_ids,
                    seq_len,
                    *num_layers,
                    *num_kv_heads,
                    *head_dim,
                )?;

                other::decode_loop(
                    &self.llm,
                    &self.embedding,
                    first_token,
                    kv_cache,
                    seq_len,
                    *num_layers,
                    *num_kv_heads,
                    *head_dim,
                    *max_seq,
                )?
            }
        };

        Ok(decode_tokens(&self.tokenizer, &token_ids))
    }
}

/// Extract decoder architecture params from the session's input metadata.
fn extract_decoder_params(session: &Session) -> Result<DecoderParams, ExtractError> {
    let num_layers = session
        .inputs()
        .iter()
        .filter(|inp| inp.name().starts_with("past_key_"))
        .count();
    if num_layers == 0 {
        return Err(ExtractError::Model(
            "decoder has no past_key_* inputs".into(),
        ));
    }

    let past_key_0 = session
        .inputs()
        .iter()
        .find(|inp| inp.name() == "past_key_0")
        .ok_or_else(|| ExtractError::Model("no past_key_0 input in decoder".into()))?;

    let shape = past_key_0
        .dtype()
        .tensor_shape()
        .ok_or_else(|| ExtractError::Model("past_key_0 is not a tensor".into()))?;

    let dims: Vec<i64> = shape.iter().copied().collect();
    if dims.len() != 4 {
        return Err(ExtractError::Model(format!(
            "past_key_0 has {} dims, expected 4",
            dims.len()
        )));
    }

    let params = DecoderParams {
        num_layers,
        num_kv_heads: dims[1] as usize,
        max_seq: dims[2] as usize,
        head_dim: dims[3] as usize,
    };

    tracing::debug!(
        "Decoder params from model: layers={}, kv_heads={}, max_seq={}, head_dim={}",
        params.num_layers,
        params.num_kv_heads,
        params.max_seq,
        params.head_dim,
    );

    Ok(params)
}
