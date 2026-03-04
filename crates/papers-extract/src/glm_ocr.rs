//! GLM-OCR formula predictor — autoregressive decoder with persistent IoBinding.
//!
//! Uses 4-part ONNX model split:
//!   - vision_encoder.onnx  (FP32, CogViT with M-RoPE)
//!   - embedding.onnx       (FP16, token embeddings for prefill)
//!   - llm.onnx             (FP32 I/O, full LLM for prefill pass)
//!   - llm_decoder_gqa.onnx  (FP16, fixed-shape decode step with GQA / FlashAttention V2)
//!
//! Architecture mirrors `formula.rs` FormulaPredictor: persistent IoBinding,
//! pre-allocated GPU buffers, pinned host memory, batched D2H via separate stream.
//!
//! Key differences from PP-FormulaNet:
//! - 4 sessions instead of 2 (vision encoder + embedding + LLM prefill + decoder)
//! - RGB image preprocessing (not grayscale) with variable-size input
//! - M-RoPE 3D position encoding
//! - Chat template prompt construction with image token splicing
//! - Prefill phase (one-shot LLM pass with full prompt before decode loop)
//! - GQA: 16 query heads, 8 KV heads (vs 16/16 for PP-FormulaNet)
//! - ORT built-in CUDA graph: enabled via `enable_cuda_graph=true` on the decoder EP.
//!   ORT captures the graph on the first `run_with_iobinding` call and replays on
//!   subsequent calls. Requires GatherND-free decoder export (4D causal mask
//!   pre-computed in the export wrapper). Async batched D2H via separate stream.

use std::path::Path;
use std::sync::Mutex;

use half::f16;
use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::{Array2, Array3};
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;

// ── Model constants ──────────────────────────────────────────────────
//
// Image preprocessing and tokenizer constants are hardcoded (not in model metadata).
// Decoder architecture params (NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ) are
// read from the ONNX model's input shapes at init time — see `extract_decoder_params()`.

const PATCH_SIZE: u32 = 14;
const SPATIAL_MERGE: u32 = 2;
const TEMPORAL_PATCH_SIZE: usize = 2;
const GRID_UNIT: u32 = PATCH_SIZE * SPATIAL_MERGE; // 28
const MIN_PIXELS: u32 = 12544; // from processor_config.json size.shortest_edge
const MAX_PIXELS: u32 = 9633792; // from processor_config.json size.longest_edge
const PATCH_ELEM: usize = TEMPORAL_PATCH_SIZE * (PATCH_SIZE as usize) * (PATCH_SIZE as usize) * 3; // 1176
const HIDDEN_SIZE: usize = 1536;
const EOS_IDS: [i64; 2] = [59246, 59253];
const IMAGE_TOKEN_ID: i64 = 59280;
// Image normalization (CLIP/SigLip-derived, matching GLM-OCR processor)
const NORM_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const NORM_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

// ── Decoder architecture params (extracted from ONNX model at init) ──

/// Decoder architecture parameters extracted from the ONNX model's input shapes.
struct DecoderParams {
    num_layers: usize,
    num_kv_heads: usize,
    max_seq: usize,
    head_dim: usize,
}

/// Extract decoder architecture params from the session's input metadata.
///
/// Reads `past_key_0` shape `[1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM]` and counts
/// `past_key_*` inputs to determine `NUM_LAYERS`.
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

// ── Prompt tokens ─────────────────────────────────────────────────────
//
// GLM-OCR chat template for "Formula Recognition:" prompt:
//   <|system|>\n你是一个乐于助人的助手。<|user|>\n<|begin_of_image|>
//   [IMAGE_TOKEN × N]
//   <|end_of_image|>\nFormula Recognition:<|assistant|>\n
//
// We pre-tokenize the fixed parts and splice image tokens at runtime.
// These token IDs come from the GLM-OCR tokenizer.json.

/// Tokenize the prompt template once at init, returning (prefix, suffix).
///
/// prefix: tokens before the image tokens
/// suffix: tokens after the image tokens (including the prompt text)
fn tokenize_prompt_parts(
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
) -> Result<(Vec<i64>, Vec<i64>), ExtractError> {
    // Chat template: [gMASK]<sop><|user|>\n<|begin_of_image|>{IMAGE_TOKENS}<|end_of_image|>{PROMPT}<|assistant|>\n
    let prefix: Vec<i64> = vec![59248, 59250, 59253, 10, 59256]; // [gMASK]<sop><|user|>\n<|begin_of_image|>

    // Tokenize the prompt text, then wrap with special tokens
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

// ── Configuration ─────────────────────────────────────────────────────

/// Configuration for GLM-OCR predictor.
pub struct GlmOcrConfig {
    /// Prompt text (e.g. "Formula Recognition:" or a full-page OCR instruction).
    pub prompt: String,
}

impl Default for GlmOcrConfig {
    fn default() -> Self {
        Self {
            prompt: "Formula Recognition:".to_string(),
        }
    }
}

// ── Main predictor struct ─────────────────────────────────────────────

/// GLM-OCR predictor with autoregressive decoding via persistent IoBinding.
pub struct GlmOcrPredictor {
    vision_encoder: Mutex<Session>,
    embedding: Mutex<Session>,
    llm: Mutex<Session>,
    decoder: Mutex<DecoderState>,
    tokenizer: tokenizers::Tokenizer,
    prompt_prefix: Vec<i64>,
    prompt_suffix: Vec<i64>,
    num_layers: usize,
    num_kv_heads: usize,
    max_seq: usize,
    head_dim: usize,
    #[allow(dead_code)]
    cuda_mem: MemoryInfo,
    #[cfg(target_os = "windows")]
    #[allow(dead_code)]
    cuda_ctx: std::sync::Arc<cudarc::driver::CudaContext>,
}

// ── Decoder state (Windows: full CUDA graph support) ──────────────────

#[cfg(target_os = "windows")]
struct DecoderState {
    session: Session,
    binding: ort::io_binding::IoBinding,
    run_options: ort::session::RunOptions<ort::session::NoSelectedOutputs>,
    // Input Values
    _input_ids: Value,
    _step: Value,
    _prefill_len: Value,
    _seqlens_k: Value,
    _total_seq_len: Value,
    _kv: Vec<Value>,
    // Raw GPU pointers
    input_ids_ptr: u64,
    step_ptr: u64,
    prefill_len_ptr: u64,
    seqlens_k_ptr: u64,
    total_seq_len_ptr: u64,
    kv_ptrs: Vec<u64>,
    kv_buf_bytes: usize,
    // Output pointer
    next_token_ptr: u64,
    _allocator: Allocator,
}

#[cfg(not(target_os = "windows"))]
struct DecoderState {
    session: Session,
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
        // CUDA context (shares primary context with ORT)
        #[cfg(target_os = "windows")]
        let cuda_ctx = cudarc::driver::CudaContext::new(0)
            .map_err(|e| ExtractError::Model(format!("cudarc CudaContext: {e}")))?;

        // Vision encoder — FP32, uses ORT_ENABLE_EXTENDED (FP16 conversion
        // blocks Cast-to-FLOAT nodes creating mixed-type graph)
        let vision_encoder = Session::builder()
            .map_err(|e| ExtractError::Model(format!("Vision encoder session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level2) // ENABLE_EXTENDED
            .map_err(|e| ExtractError::Model(format!("Vision encoder opt level: {e}")))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| ExtractError::Model(format!("Vision encoder EP: {e}")))?
            .commit_from_file(vision_encoder_path)
            .map_err(|e| {
                ExtractError::Model(format!(
                    "Vision encoder load from {}: {e}",
                    vision_encoder_path.display()
                ))
            })?;

        // Embedding — FP16, standard CUDA EP
        let embedding = Session::builder()
            .map_err(|e| ExtractError::Model(format!("Embedding session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ExtractError::Model(format!("Embedding opt level: {e}")))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| ExtractError::Model(format!("Embedding EP: {e}")))?
            .commit_from_file(embedding_path)
            .map_err(|e| {
                ExtractError::Model(format!(
                    "Embedding load from {}: {e}",
                    embedding_path.display()
                ))
            })?;

        // LLM for prefill — FP16, standard CUDA EP (dynamic shapes, no graph)
        let llm = Session::builder()
            .map_err(|e| ExtractError::Model(format!("LLM session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ExtractError::Model(format!("LLM opt level: {e}")))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| ExtractError::Model(format!("LLM EP: {e}")))?
            .commit_from_file(llm_path)
            .map_err(|e| {
                ExtractError::Model(format!(
                    "LLM load from {}: {e}",
                    llm_path.display()
                ))
            })?;

        // Decoder — FP16, ORT built-in CUDA graph (captures on first run, replays after)
        #[cfg(target_os = "windows")]
        let decoder_ep = ort::execution_providers::CUDAExecutionProvider::default()
            .with_cuda_graph(true)
            .build();
        #[cfg(not(target_os = "windows"))]
        let decoder_ep =
            ort::execution_providers::CUDAExecutionProvider::default().build();

        let decoder_session = Session::builder()
            .map_err(|e| ExtractError::Model(format!("Decoder session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ExtractError::Model(format!("Decoder opt level: {e}")))?
            .with_execution_providers([decoder_ep])
            .map_err(|e| ExtractError::Model(format!("Decoder EP: {e}")))?
            .commit_from_file(decoder_path)
            .map_err(|e| {
                ExtractError::Model(format!(
                    "Decoder load from {}: {e}",
                    decoder_path.display()
                ))
            })?;

        let cuda_mem = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| ExtractError::Model(format!("CUDA MemoryInfo: {e}")))?;

        // Extract decoder architecture from model metadata
        let decoder_params = extract_decoder_params(&decoder_session)?;
        let decoder_state = Self::build_decoder_state(decoder_session, &cuda_mem, &decoder_params)?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            ExtractError::Model(format!(
                "Tokenizer load from {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let (prompt_prefix, prompt_suffix) = tokenize_prompt_parts(&tokenizer, &config.prompt)?;
        tracing::debug!(
            "GLM-OCR prompt: {} prefix + N image + {} suffix tokens (max_seq={})",
            prompt_prefix.len(),
            prompt_suffix.len(),
            decoder_params.max_seq,
        );

        let predictor = Self {
            vision_encoder: Mutex::new(vision_encoder),
            embedding: Mutex::new(embedding),
            llm: Mutex::new(llm),
            decoder: Mutex::new(decoder_state),
            tokenizer,
            prompt_prefix,
            prompt_suffix,
            num_layers: decoder_params.num_layers,
            num_kv_heads: decoder_params.num_kv_heads,
            max_seq: decoder_params.max_seq,
            head_dim: decoder_params.head_dim,
            cuda_mem,
            #[cfg(target_os = "windows")]
            cuda_ctx,
        };

        predictor.warmup()?;
        Ok(predictor)
    }

    // ── Decoder state allocation ──────────────────────────────────────

    #[cfg(target_os = "windows")]
    fn build_decoder_state(
        decoder_session: Session,
        cuda_mem: &MemoryInfo,
        params: &DecoderParams,
    ) -> Result<DecoderState, ExtractError> {
        let allocator = Allocator::new(&decoder_session, cuda_mem.clone())
            .map_err(|e| ExtractError::Model(format!("CUDA allocator: {e}")))?;

        let n_kv_buffers = params.num_layers * 2;
        let kv_shape = [1usize, params.num_kv_heads, params.max_seq, params.head_dim];
        let kv_buf_bytes =
            params.num_kv_heads * params.max_seq * params.head_dim * std::mem::size_of::<f16>();

        // Allocate input tensors on GPU
        let mut input_ids_t = ort::value::Tensor::<i64>::new(&allocator, [1usize, 1])
            .map_err(|e| ExtractError::Model(format!("alloc input_ids: {e}")))?;
        let input_ids_ptr = input_ids_t.data_ptr_mut() as u64;

        let mut step_t = ort::value::Tensor::<i64>::new(&allocator, [1usize])
            .map_err(|e| ExtractError::Model(format!("alloc step: {e}")))?;
        let step_ptr = step_t.data_ptr_mut() as u64;

        let mut prefill_len_t = ort::value::Tensor::<i64>::new(&allocator, [1usize])
            .map_err(|e| ExtractError::Model(format!("alloc prefill_len: {e}")))?;
        let prefill_len_ptr = prefill_len_t.data_ptr_mut() as u64;

        // GQA sequence length inputs (int32)
        let mut seqlens_k_t = ort::value::Tensor::<i32>::new(&allocator, [1usize])
            .map_err(|e| ExtractError::Model(format!("alloc seqlens_k: {e}")))?;
        let seqlens_k_ptr = seqlens_k_t.data_ptr_mut() as u64;

        let mut total_seq_len_t = ort::value::Tensor::<i32>::new(&allocator, [1usize])
            .map_err(|e| ExtractError::Model(format!("alloc total_seq_len: {e}")))?;
        let total_seq_len_ptr = total_seq_len_t.data_ptr_mut() as u64;

        // KV cache buffers (in-place: same buffer for input and output)
        let mut kv: Vec<Value> = Vec::with_capacity(n_kv_buffers);
        let mut kv_ptrs: Vec<u64> = Vec::with_capacity(n_kv_buffers);
        for _ in 0..n_kv_buffers {
            let mut t = ort::value::Tensor::<f16>::new(&allocator, kv_shape)
                .map_err(|e| ExtractError::Model(format!("alloc kv: {e}")))?;
            kv_ptrs.push(t.data_ptr_mut() as u64);
            kv.push(t.into());
        }

        // next_token output
        let mut next_token_t = ort::value::Tensor::<i64>::new(&allocator, [1usize])
            .map_err(|e| ExtractError::Model(format!("alloc next_token: {e}")))?;
        let next_token_ptr = next_token_t.data_ptr_mut() as u64;

        // Create persistent IoBinding
        let mut binding = decoder_session
            .create_binding()
            .map_err(|e| ExtractError::Model(format!("decoder create_binding: {e}")))?;

        let input_ids_val: Value = input_ids_t.into();
        let step_val: Value = step_t.into();
        let prefill_len_val: Value = prefill_len_t.into();
        let seqlens_k_val: Value = seqlens_k_t.into();
        let total_seq_len_val: Value = total_seq_len_t.into();

        // Bind inputs
        binding
            .bind_input("input_ids", &input_ids_val)
            .map_err(|e| ExtractError::Model(format!("bind input_ids: {e}")))?;
        binding
            .bind_input("step", &step_val)
            .map_err(|e| ExtractError::Model(format!("bind step: {e}")))?;
        binding
            .bind_input("prefill_len", &prefill_len_val)
            .map_err(|e| ExtractError::Model(format!("bind prefill_len: {e}")))?;
        binding
            .bind_input("seqlens_k", &seqlens_k_val)
            .map_err(|e| ExtractError::Model(format!("bind seqlens_k: {e}")))?;
        binding
            .bind_input("total_sequence_length", &total_seq_len_val)
            .map_err(|e| ExtractError::Model(format!("bind total_seq_len: {e}")))?;

        for (i, kv_val) in kv.iter().enumerate() {
            let layer = i / 2;
            let kv_type = if i % 2 == 0 { "key" } else { "value" };
            binding
                .bind_input(&format!("past_{kv_type}_{layer}"), kv_val)
                .map_err(|e| ExtractError::Model(format!("bind kv_in {i}: {e}")))?;
        }

        // Bind KV outputs to SAME buffers as inputs (in-place update via C API)
        {
            use ort::AsPointer;
            let binding_ptr = binding.ptr() as *mut ort::sys::OrtIoBinding;
            let api = ort::api();
            for (i, kv_val) in kv.iter().enumerate() {
                let layer = i / 2;
                let kv_type = if i % 2 == 0 { "key" } else { "value" };
                let name = format!("present_{kv_type}_{layer}");
                let c_name = std::ffi::CString::new(name.as_str())
                    .map_err(|e| ExtractError::Model(format!("CString kv_out {i}: {e}")))?;
                let status = unsafe {
                    (api.BindOutput)(
                        binding_ptr,
                        c_name.as_ptr(),
                        kv_val.ptr() as *mut ort::sys::OrtValue,
                    )
                };
                if !status.0.is_null() {
                    unsafe { (api.ReleaseStatus)(status.0) };
                    return Err(ExtractError::Model(format!(
                        "raw BindOutput kv {i} failed"
                    )));
                }
            }
        }

        binding
            .bind_output("next_token", next_token_t)
            .map_err(|e| ExtractError::Model(format!("bind next_token: {e}")))?;

        let run_options = ort::session::RunOptions::new()
            .map_err(|e| ExtractError::Model(format!("RunOptions: {e}")))?;

        Ok(DecoderState {
            session: decoder_session,
            binding,
            run_options,
            _input_ids: input_ids_val,
            _step: step_val,
            _prefill_len: prefill_len_val,
            _seqlens_k: seqlens_k_val,
            _total_seq_len: total_seq_len_val,
            _kv: kv,
            input_ids_ptr,
            step_ptr,
            prefill_len_ptr,
            seqlens_k_ptr,
            total_seq_len_ptr,
            kv_ptrs,
            kv_buf_bytes,
            next_token_ptr,
            _allocator: allocator,
        })
    }

    #[cfg(not(target_os = "windows"))]
    fn build_decoder_state(
        decoder_session: Session,
        _cuda_mem: &MemoryInfo,
        _params: &DecoderParams,
    ) -> Result<DecoderState, ExtractError> {
        Ok(DecoderState {
            session: decoder_session,
        })
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
        let vision_embeds = self.run_vision_encoder(&pixel_values, &grid_thw)?;

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
        let token_embeds = self.run_embedding(&input_ids)?;

        // 5. Merge vision embeddings into token embeddings
        let inputs_embeds = merge_vision_embeddings(
            token_embeds,
            &vision_embeds,
            &input_ids,
            seq_len,
        )?;

        // 6. Build 3D M-RoPE position IDs
        let position_ids = build_position_ids(&input_ids, &grid_thw);

        // 7. Prefill — run LLM with full prompt, get first token + KV cache
        let (first_token, kv_cache) =
            self.run_prefill(inputs_embeds, &position_ids, seq_len)?;

        // 8. Copy KV cache from prefill into decoder's fixed buffers, then decode
        let token_ids = self.decode_loop(first_token, &kv_cache, seq_len)?;

        Ok(decode_tokens(&self.tokenizer, &token_ids))
    }

    // ── Vision encoder ────────────────────────────────────────────────

    fn run_vision_encoder(
        &self,
        pixel_values: &Array2<f32>,
        grid_thw: &Array2<i64>,
    ) -> Result<Value, ExtractError> {
        let mut session = self
            .vision_encoder
            .lock()
            .map_err(|e| ExtractError::Formula(format!("vision encoder lock: {e}")))?;

        // Compute vision position IDs (M-RoPE)
        let (pos_ids, max_grid_size) = compute_vision_pos_ids(grid_thw);

        let pv_tensor = Value::from_array(pixel_values.clone().into_dyn())
            .map_err(|e| ExtractError::Formula(format!("vision pixel_values tensor: {e}")))?;
        let pos_tensor = Value::from_array(pos_ids.into_dyn())
            .map_err(|e| ExtractError::Formula(format!("vision pos_ids tensor: {e}")))?;
        let grid_tensor =
            Value::from_array(ndarray::arr0(max_grid_size).into_dyn())
                .map_err(|e| ExtractError::Formula(format!("vision max_grid_size tensor: {e}")))?;

        // Use session.run() — returns CPU-side outputs (CUDA EP handles GPU internally).
        // We need CPU access for merge_vision_embeddings later.
        let output_name = session.outputs()[0].name().to_string();
        let pv_val: Value = pv_tensor.into();
        let pos_val: Value = pos_tensor.into();
        let grid_val: Value = grid_tensor.into();
        let mut outputs = session
            .run(ort::inputs![
                "pixel_values" => pv_val,
                "pos_ids" => pos_val,
                "max_grid_size" => grid_val
            ])
            .map_err(|e| ExtractError::Formula(format!("vision encoder run: {e}")))?;

        outputs
            .remove(&output_name)
            .ok_or_else(|| ExtractError::Formula("No vision encoder output".into()))
    }

    // ── Token embedding ───────────────────────────────────────────────

    fn run_embedding(&self, input_ids: &[i64]) -> Result<Value, ExtractError> {
        let mut session = self
            .embedding
            .lock()
            .map_err(|e| ExtractError::Formula(format!("embedding lock: {e}")))?;

        let seq_len = input_ids.len();
        let ids_array =
            Array2::from_shape_vec([1, seq_len], input_ids.to_vec())
                .map_err(|e| ExtractError::Formula(format!("embedding ids array: {e}")))?;
        let ids_tensor: Value = Value::from_array(ids_array.into_dyn())
            .map_err(|e| ExtractError::Formula(format!("embedding input tensor: {e}")))?
            .into();

        // Use session.run() — returns CPU-side outputs for merge_vision_embeddings.
        let output_name = session.outputs()[0].name().to_string();
        let mut outputs = session
            .run(ort::inputs!["input_ids" => ids_tensor])
            .map_err(|e| ExtractError::Formula(format!("embedding run: {e}")))?;

        outputs
            .remove(&output_name)
            .ok_or_else(|| ExtractError::Formula("No embedding output".into()))
    }

    // ── Prefill ───────────────────────────────────────────────────────

    /// Run the LLM prefill pass with the full prompt.
    /// Returns (first_generated_token, kv_cache_values).
    /// KV cache values are GPU-resident; logits are on CPU for argmax.
    /// Run prefill through llm.onnx with session.run() (CPU outputs).
    ///
    /// Returns (first_token, kv_cache) where kv_cache is a Vec of CPU-side
    /// f32 Values. The decode_loop converts f32→f16 when uploading to GPU.
    fn run_prefill(
        &self,
        inputs_embeds: Value,
        position_ids: &Array3<i64>,
        seq_len: usize,
    ) -> Result<(i64, Vec<Vec<f16>>), ExtractError> {
        let mut session = self
            .llm
            .lock()
            .map_err(|e| ExtractError::Formula(format!("llm lock: {e}")))?;

        let attention_mask = Array2::from_elem([1, seq_len], 1i64);
        let mask_tensor = Value::from_array(attention_mask.into_dyn())
            .map_err(|e| ExtractError::Formula(format!("prefill mask tensor: {e}")))?;
        let pos_tensor = Value::from_array(position_ids.clone().into_dyn())
            .map_err(|e| ExtractError::Formula(format!("prefill pos tensor: {e}")))?;

        // Empty KV cache for prefill (dynamic shape: past_seq_len = 0)
        // llm.onnx uses f32 I/O
        let n_kv_buffers = self.num_layers * 2;
        let empty_kv_shape = [1usize, self.num_kv_heads, 0, self.head_dim];
        let empty_kv: Vec<Value> = (0..n_kv_buffers)
            .map(|_| {
                let arr = ndarray::Array4::<f32>::from_elem(empty_kv_shape, 0.0);
                Value::from_array(arr.into_dyn()).expect("empty kv tensor").into()
            })
            .collect();

        // Build input list: inputs_embeds, attention_mask, position_ids, then KV pairs
        let mut inputs: Vec<(std::borrow::Cow<str>, Value)> = Vec::new();
        inputs.push(("inputs_embeds".into(), inputs_embeds));
        inputs.push(("attention_mask".into(), mask_tensor.into()));
        inputs.push(("position_ids".into(), pos_tensor.into()));

        for (i, kv_val) in empty_kv.into_iter().enumerate() {
            let layer = i / 2;
            let kv_type = if i % 2 == 0 { "key" } else { "value" };
            inputs.push((format!("past_{kv_type}_{layer}").into(), kv_val));
        }

        let outputs = session
            .run(inputs)
            .map_err(|e| ExtractError::Formula(format!("prefill run: {e}")))?;

        // Extract logits → argmax for first token
        let logits = outputs
            .get("logits")
            .ok_or_else(|| ExtractError::Formula("No prefill logits output".into()))?;
        let (logits_shape, logits_data) = logits
            .try_extract_tensor::<f32>()
            .map_err(|e| ExtractError::Formula(format!("prefill extract logits: {e}")))?;
        let logits_seq_len = logits_shape[1] as usize;
        let vocab = logits_shape[2] as usize;
        let offset = (logits_seq_len - 1) * vocab;
        let last_logits = &logits_data[offset..offset + vocab];
        let first_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap_or(EOS_IDS[0]);

        // Extract KV cache outputs (CPU f32) and convert to f16
        let mut kv_cache = Vec::with_capacity(n_kv_buffers);
        for i in 0..self.num_layers {
            for kv_type in &["key", "value"] {
                let name = format!("present_{kv_type}_{i}");
                let kv_val = outputs
                    .get(&name)
                    .ok_or_else(|| ExtractError::Formula(format!("No prefill {name}")))?;
                let (_, kv_data) = kv_val
                    .try_extract_tensor::<f32>()
                    .map_err(|e| ExtractError::Formula(format!("prefill extract {name}: {e}")))?;
                // Convert f32 → f16 for decoder compatibility
                let kv_f16: Vec<f16> = kv_data.iter().map(|&v| f16::from_f32(v)).collect();
                kv_cache.push(kv_f16);
            }
        }

        Ok((first_token, kv_cache))
    }

    // ── Decode loop (CUDA graph) ──────────────────────────────────────

    /// Copy prefill KV cache into decoder's fixed buffers and run decode loop.
    ///
    /// Uses ORT built-in CUDA graph: first `run_with_iobinding` captures the graph,
    /// subsequent calls replay it. Between steps we memcpy `step` and `input_ids`
    /// into the same fixed GPU buffers (addresses don't change, only content).
    #[cfg(target_os = "windows")]
    fn decode_loop(
        &self,
        first_token: i64,
        prefill_kv: &[Vec<f16>],
        prefill_len: usize,
    ) -> Result<Vec<i64>, ExtractError> {
        let mut state = self
            .decoder
            .lock()
            .map_err(|e| ExtractError::Formula(format!("decoder lock: {e}")))?;

        // Use null stream (default stream) for memcpy — synchronous with ORT's work
        let null_stream = std::ptr::null_mut();

        // Zero all KV cache buffers
        for &ptr in &state.kv_ptrs {
            unsafe {
                cudarc::driver::result::memset_d8_async(ptr, 0u8, state.kv_buf_bytes, null_stream)
            }
            .map_err(|e| ExtractError::Formula(format!("memset kv: {e}")))?;
        }

        // Copy prefill KV cache (CPU f16) into decoder's fixed GPU buffers
        // Copy head-by-head since strides differ (prefill_len vs max_seq).
        for (i, kv_data) in prefill_kv.iter().enumerate() {
            if prefill_len == 0 {
                continue;
            }

            let src_head_stride = prefill_len * self.head_dim;
            let dst_head_stride = self.max_seq * self.head_dim * std::mem::size_of::<f16>();
            let copy_elems = prefill_len * self.head_dim;

            for h in 0..self.num_kv_heads {
                let src_start = h * src_head_stride;
                let src_slice = &kv_data[src_start..src_start + copy_elems];
                let dst = state.kv_ptrs[i] + (h * dst_head_stride) as u64;
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(dst, src_slice, null_stream)
                }
                .map_err(|e| ExtractError::Formula(format!("H2D kv {i} head {h}: {e}")))?;
            }
        }

        // Set prefill_len
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                state.prefill_len_ptr,
                &[prefill_len as i64],
                null_stream,
            )
        }
        .map_err(|e| ExtractError::Formula(format!("H2D prefill_len: {e}")))?;

        let mut tokens = vec![first_token];

        // Seed input_ids with first token
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                state.input_ids_ptr,
                &[first_token],
                null_stream,
            )
        }
        .map_err(|e| ExtractError::Formula(format!("H2D first token: {e}")))?;

        if EOS_IDS.contains(&first_token) {
            return Ok(tokens);
        }

        // Decode loop: ORT CUDA graph handles compute, we update step/input_ids via memcpy
        let mut token_buf = [0i64; 1];

        for s in 0..self.max_seq {
            // H2D: update step counter
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    state.step_ptr,
                    &[s as i64],
                    null_stream,
                )
            }
            .map_err(|e| ExtractError::Formula(format!("H2D step: {e}")))?;

            // H2D: update GQA sequence length inputs
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    state.seqlens_k_ptr,
                    &[(prefill_len + s) as i32],
                    null_stream,
                )
            }
            .map_err(|e| ExtractError::Formula(format!("H2D seqlens_k: {e}")))?;
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    state.total_seq_len_ptr,
                    &[(prefill_len + s + 1) as i32],
                    null_stream,
                )
            }
            .map_err(|e| ExtractError::Formula(format!("H2D total_seq_len: {e}")))?;

            // Run decoder step (ORT captures graph on step 0, replays after)
            {
                let DecoderState {
                    ref mut session,
                    ref binding,
                    ref run_options,
                    ..
                } = *state;
                session
                    .run_binding_with_options(binding, run_options)
                    .map_err(|e| ExtractError::Formula(format!("decoder step {s}: {e}")))?;
            }

            // D2D: next_token → input_ids (for next step)
            unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    state.input_ids_ptr,
                    state.next_token_ptr,
                    std::mem::size_of::<i64>(),
                    null_stream,
                )
            }
            .map_err(|e| ExtractError::Formula(format!("D2D next→input: {e}")))?;

            // D2H: read next_token to check for EOS
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(
                    &mut token_buf,
                    state.next_token_ptr,
                    null_stream,
                )
            }
            .map_err(|e| ExtractError::Formula(format!("D2H next_token: {e}")))?;

            tokens.push(token_buf[0]);
            if EOS_IDS.contains(&token_buf[0]) {
                break;
            }
        }

        Ok(tokens)
    }

    #[cfg(not(target_os = "windows"))]
    fn decode_loop(
        &self,
        _first_token: i64,
        _prefill_kv: &[Vec<f16>],
        _prefill_len: usize,
    ) -> Result<Vec<i64>, ExtractError> {
        Err(ExtractError::Formula(
            "GLM-OCR CUDA decoder only supported on Windows".into(),
        ))
    }
}

// ── Image preprocessing ───────────────────────────────────────────────

/// Smart-resize dimensions to be multiples of GRID_UNIT (28), keeping the
/// total pixel area within [MIN_PIXELS, MAX_PIXELS].
///
/// Matches the exact logic from transformers Glm46VImageProcessor.smart_resize.
fn smart_resize(orig_h: u32, orig_w: u32) -> (u32, u32) {
    let factor = GRID_UNIT as f64;
    let num_frames = TEMPORAL_PATCH_SIZE as f64;

    // If either dimension < factor, scale up to ensure both >= factor
    let (mut height, mut width) = (orig_h as f64, orig_w as f64);
    if height < factor || width < factor {
        let scale = (factor / height).max(factor / width);
        height = (height * scale).floor();
        width = (width * scale).floor();
    }

    // Round to nearest multiple of factor
    let mut h_bar = (height / factor).round().max(1.0) as u32 * GRID_UNIT;
    let mut w_bar = (width / factor).round().max(1.0) as u32 * GRID_UNIT;
    let t_bar = num_frames; // = temporal_patch_size = 2

    // Clamp total area (t_bar * h_bar * w_bar) to [MIN_PIXELS, MAX_PIXELS]
    let total = t_bar * h_bar as f64 * w_bar as f64;
    if total > MAX_PIXELS as f64 {
        let beta = (num_frames * height * width / MAX_PIXELS as f64).sqrt();
        h_bar = (height / beta / factor).floor().max(1.0) as u32 * GRID_UNIT;
        w_bar = (width / beta / factor).floor().max(1.0) as u32 * GRID_UNIT;
    } else if total < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (num_frames * height * width)).sqrt();
        h_bar = (height * beta / factor).ceil() as u32 * GRID_UNIT;
        w_bar = (width * beta / factor).ceil() as u32 * GRID_UNIT;
    }

    (h_bar, w_bar)
}

/// Preprocess a formula image for the GLM-OCR vision encoder.
///
/// Pipeline (matches Glm46VImageProcessorFast):
///   1. Resize to (H, W) where both are multiples of 28
///   2. Normalize: (pixel/255 - mean) / std
///   3. Duplicate for temporal_patch_size=2 (same image × 2 frames)
///   4. Extract patches and flatten to [num_patches, 1176]
///
/// Returns:
///   - pixel_values: [num_patches, 1176] f32
///   - grid_thw: [1, 3] i64 = [temporal=1, h_patches, w_patches]
fn preprocess_image(image: &DynamicImage) -> (Array2<f32>, Array2<i64>) {
    let rgb = image.to_rgb8();
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    let (target_h, target_w) = smart_resize(orig_h, orig_w);

    // Resize to exact target dimensions
    // CatmullRom = Bicubic, matching HuggingFace's default resample=3 (PIL.BICUBIC)
    let resized = image.resize_exact(target_w, target_h, FilterType::CatmullRom);
    let resized_rgb = resized.to_rgb8();

    let grid_h = (target_h / PATCH_SIZE) as usize;
    let grid_w = (target_w / PATCH_SIZE) as usize;
    let num_patches = grid_h * grid_w;
    let ps = PATCH_SIZE as usize;
    let sm = SPATIAL_MERGE as usize;
    let merged_h = grid_h / sm;
    let merged_w = grid_w / sm;

    // Build pixel_values: [num_patches, 1176]
    // Patch ordering: [gh, gw, mh, mw] where gh/gw iterate over merged groups
    // and mh/mw (each 0..merge_size-1) iterate within each 2×2 merge group.
    // Memory layout per patch: C × T × H × W (channels outermost)
    // where C=3, T=temporal_patch_size=2, H=patch_size=14, W=patch_size=14
    // Since temporal_patch_size=2 and it's a single image, both frames are identical.
    let spatial = ps * ps; // 196
    let temporal_spatial = TEMPORAL_PATCH_SIZE * spatial; // 392
    let mut pixel_values = vec![0.0f32; num_patches * PATCH_ELEM];

    let mut patch_idx = 0;
    for bh in 0..merged_h {
        for bw in 0..merged_w {
            for mh in 0..sm {
                for mw in 0..sm {
                    let patch_offset = patch_idx * PATCH_ELEM;
                    let y_start = (bh * sm + mh) * ps;
                    let x_start = (bw * sm + mw) * ps;

                    for c in 0..3usize {
                        for t in 0..TEMPORAL_PATCH_SIZE {
                            for py in 0..ps {
                                for px in 0..ps {
                                    let pixel =
                                        resized_rgb.get_pixel((x_start + px) as u32, (y_start + py) as u32);
                                    let val = pixel[c] as f32 / 255.0;
                                    let normalized = (val - NORM_MEAN[c]) / NORM_STD[c];
                                    let idx = patch_offset
                                        + c * temporal_spatial
                                        + t * spatial
                                        + py * ps
                                        + px;
                                    pixel_values[idx] = normalized;
                                }
                            }
                        }
                    }
                    patch_idx += 1;
                }
            }
        }
    }

    let pv_array = Array2::from_shape_vec([num_patches, PATCH_ELEM], pixel_values)
        .expect("pixel_values shape mismatch");

    // grid_thw: [1, h_patches, w_patches]
    let grid_thw = Array2::from_shape_vec(
        [1, 3],
        vec![1i64, grid_h as i64, grid_w as i64],
    )
    .expect("grid_thw shape");

    (pv_array, grid_thw)
}

// ── Vision position IDs (M-RoPE) ─────────────────────────────────────

/// Compute position IDs for vision rotary embeddings.
/// Returns (pos_ids: [N, 2], max_grid_size: i64).
fn compute_vision_pos_ids(grid_thw: &Array2<i64>) -> (Array2<i64>, i64) {
    let mut pos_ids_list: Vec<Vec<[i64; 2]>> = Vec::new();
    let mut max_grid_size: i64 = 0;

    for row in 0..grid_thw.shape()[0] {
        let t = grid_thw[[row, 0]] as usize;
        let h = grid_thw[[row, 1]] as usize;
        let w = grid_thw[[row, 2]] as usize;
        let sm = SPATIAL_MERGE as usize;

        max_grid_size = max_grid_size.max(h as i64).max(w as i64);

        // Height position IDs
        let mut hpos = vec![0i64; h * w];
        for y in 0..h {
            for x in 0..w {
                hpos[y * w + x] = y as i64;
            }
        }

        // Reshape to [h/sm, sm, w/sm, sm], transpose to [h/sm, w/sm, sm, sm], flatten
        let mh = h / sm;
        let mw = w / sm;
        let mut h_merged = vec![0i64; h * w];
        for bh in 0..mh {
            for bw in 0..mw {
                for sh in 0..sm {
                    for sw in 0..sm {
                        let src_idx = (bh * sm + sh) * w + (bw * sm + sw);
                        let dst_idx = ((bh * mw + bw) * sm + sh) * sm + sw;
                        h_merged[dst_idx] = hpos[src_idx];
                    }
                }
            }
        }

        // Width position IDs
        let mut wpos = vec![0i64; h * w];
        for y in 0..h {
            for x in 0..w {
                wpos[y * w + x] = x as i64;
            }
        }

        let mut w_merged = vec![0i64; h * w];
        for bh in 0..mh {
            for bw in 0..mw {
                for sh in 0..sm {
                    for sw in 0..sm {
                        let src_idx = (bh * sm + sh) * w + (bw * sm + sw);
                        let dst_idx = ((bh * mw + bw) * sm + sh) * sm + sw;
                        w_merged[dst_idx] = wpos[src_idx];
                    }
                }
            }
        }

        // Stack [h, w] pairs and tile over time
        let frame_len = h * w;
        let mut frame_pos: Vec<[i64; 2]> = Vec::with_capacity(frame_len);
        for i in 0..frame_len {
            frame_pos.push([h_merged[i], w_merged[i]]);
        }

        // Tile over t frames
        let mut tiled = Vec::with_capacity(t * frame_len);
        for _ in 0..t {
            tiled.extend_from_slice(&frame_pos);
        }
        pos_ids_list.push(tiled);
    }

    // Concatenate all entries
    let total: usize = pos_ids_list.iter().map(|v| v.len()).sum();
    let mut flat = Vec::with_capacity(total * 2);
    for entry in &pos_ids_list {
        for &[h, w] in entry {
            flat.push(h);
            flat.push(w);
        }
    }

    let pos_ids = Array2::from_shape_vec([total, 2], flat).expect("pos_ids shape");
    (pos_ids, max_grid_size)
}

// ── M-RoPE position IDs for full prompt ───────────────────────────────

/// Build 3D M-RoPE position IDs for text+vision tokens.
/// Returns [3, 1, seq_len] position IDs.
fn build_position_ids(input_ids: &[i64], grid_thw: &Array2<i64>) -> Array3<i64> {
    let seq_len = input_ids.len();
    let mut position_ids = vec![0i64; 3 * seq_len];

    let mut pos: i64 = 0;
    let mut i = 0;
    let mut img_idx = 0;

    while i < seq_len {
        if input_ids[i] != IMAGE_TOKEN_ID {
            // Text token: all 3 dims get same position
            position_ids[i] = pos; // dim 0: temporal
            position_ids[seq_len + i] = pos; // dim 1: height
            position_ids[2 * seq_len + i] = pos; // dim 2: width
            pos += 1;
            i += 1;
        } else {
            // Vision tokens: spatial layout
            let t_grid = grid_thw[[img_idx, 0]] as usize;
            let h_grid = grid_thw[[img_idx, 1]] as usize;
            let w_grid = grid_thw[[img_idx, 2]] as usize;
            let merged_h = h_grid / SPATIAL_MERGE as usize;
            let merged_w = w_grid / SPATIAL_MERGE as usize;
            let num_vision_tokens = t_grid * merged_h * merged_w;

            let temporal_pos = pos;
            for vi in 0..num_vision_tokens {
                let row = (vi % (merged_h * merged_w)) / merged_w;
                let col = (vi % (merged_h * merged_w)) % merged_w;
                position_ids[i + vi] = temporal_pos; // dim 0: temporal
                position_ids[seq_len + i + vi] = pos + row as i64; // dim 1: height
                position_ids[2 * seq_len + i + vi] = pos + col as i64; // dim 2: width
            }

            pos += merged_h.max(merged_w) as i64;
            i += num_vision_tokens;
            img_idx += 1;
        }
    }

    Array3::from_shape_vec([3, 1, seq_len], position_ids).expect("position_ids shape")
}

// ── Merge vision embeddings ───────────────────────────────────────────

/// Replace image token positions in token_embeds with vision_embeds.
///
/// Both values are CPU-side. session.run() may return f32 (ORT upcasts FP16
/// on CPU), so we handle both f32 and f16 inputs. Output is f16 for the
/// FP16 LLM prefill.
fn merge_vision_embeddings(
    token_embeds: Value,
    vision_embeds: &Value,
    input_ids: &[i64],
    seq_len: usize,
) -> Result<Value, ExtractError> {
    let hidden = HIDDEN_SIZE;

    // All models use f32 I/O — extract as f32
    let (_shape, token_data) = token_embeds
        .try_extract_tensor::<f32>()
        .map_err(|e| ExtractError::Formula(format!("extract token_embeds: {e}")))?;

    let (_shape, vision_data) = vision_embeds
        .try_extract_tensor::<f32>()
        .map_err(|e| ExtractError::Formula(format!("extract vision_embeds: {e}")))?;

    // Build merged array in f32: replace IMAGE_TOKEN positions with vision embeddings
    let mut merged = vec![0.0f32; seq_len * hidden];

    let mut vis_idx = 0;
    for i in 0..seq_len {
        if input_ids[i] == IMAGE_TOKEN_ID {
            let src_offset = vis_idx * hidden;
            merged[i * hidden..(i + 1) * hidden]
                .copy_from_slice(&vision_data[src_offset..src_offset + hidden]);
            vis_idx += 1;
        } else {
            let src_offset = i * hidden;
            merged[i * hidden..(i + 1) * hidden]
                .copy_from_slice(&token_data[src_offset..src_offset + hidden]);
        }
    }

    let merged_array =
        ndarray::Array3::from_shape_vec([1, seq_len, hidden], merged)
            .map_err(|e| ExtractError::Formula(format!("merged embeds shape: {e}")))?;

    Value::from_array(merged_array.into_dyn())
        .map(|v| v.into())
        .map_err(|e| ExtractError::Formula(format!("merged embeds tensor: {e}")))
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
