//! Custom CUDA formula predictor — replaces oar-ocr FormulaRecognitionPredictor.
//!
//! Uses split encoder/decoder ONNX models with CUDA EP + FP16 for fast GPU
//! inference. The decoder uses a persistent IoBinding with pre-allocated GPU
//! buffers for zero-allocation autoregressive decoding. cudarc handles raw
//! memcpy updates between steps. Only next_token (8 bytes) crosses GPU→CPU.

use std::path::Path;
use std::sync::Mutex;

use half::f16;
use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::Array4;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::{Value, ValueType};

use crate::error::ExtractError;

// Model constants (must match the exported ONNX models)
const TARGET_SIZE: u32 = 768;
const BOS_ID: i64 = 0;
const EOS_ID: i64 = 2;
const N_LAYERS: usize = 8;
const N_HEADS: usize = 16;
const HEAD_DIM: usize = 32;
const MAX_SEQ: usize = 512;
const N_KV_BUFFERS: usize = N_LAYERS * 2; // key + value per layer

/// Custom formula predictor with persistent IoBinding on the decoder.
///
/// The encoder runs normally (one pass per formula). The decoder uses a
/// persistent IoBinding with all buffers pre-allocated on GPU at init time.
/// Between steps, cudarc handles small memcpy updates (input_ids, step,
/// next_token, and KV cache output→input copies). No allocations during
/// autoregressive decoding.
pub struct FormulaPredictor {
    encoder: Mutex<Session>,
    decoder: Mutex<DecoderState>,
    tokenizer: tokenizers::Tokenizer,
    cuda_mem: MemoryInfo,
    #[cfg(target_os = "windows")]
    cuda_stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

/// Pre-allocated decoder state with persistent IoBinding.
///
/// Field order matters: `binding` must be dropped before the input Values
/// it references (`_input_ids`, `_step`, `_enc_hidden`, `_kv`), because
/// `bind_input` stores raw pointers to their GPU buffers.
#[cfg(target_os = "windows")]
struct DecoderState {
    session: Session,
    binding: ort::io_binding::IoBinding,
    // Input Values (must outlive binding — bind_input stores raw pointers)
    _input_ids: Value,
    _step: Value,
    _enc_hidden: Value,
    _kv: Vec<Value>,
    // Raw GPU pointers for cudarc memcpy
    input_ids_ptr: u64,
    step_ptr: u64,
    enc_hidden_ptr: u64,
    enc_hidden_bytes: usize,
    // KV cache pointers (in-place: same buffer is both input and output, memset between formulas)
    kv_ptrs: Vec<u64>,
    kv_buf_bytes: usize,
    // Output pointer
    next_token_ptr: u64,
    // Allocator must outlive all Values (dropped last in struct drop order)
    _allocator: Allocator,
}

#[cfg(not(target_os = "windows"))]
struct DecoderState {
    session: Session,
}

/// Get the raw device pointer from any ORT Value (works for GPU-resident tensors).
fn value_data_ptr(value: &Value) -> u64 {
    use ort::AsPointer;
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    let api = ort::api();
    unsafe {
        let _ = (api.GetTensorMutableData)(
            value.ptr() as *mut ort::sys::OrtValue,
            &mut ptr as *mut *mut std::ffi::c_void as *mut *mut _,
        );
    }
    ptr as u64
}

impl FormulaPredictor {
    /// Create a new formula predictor from split encoder/decoder ONNX models.
    pub fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, ExtractError> {
        // --- CUDA context + stream (cudarc shares primary context with ORT) ---
        #[cfg(target_os = "windows")]
        let cuda_ctx = cudarc::driver::CudaContext::new(0)
            .map_err(|e| ExtractError::Model(format!("cudarc CudaContext: {e}")))?;
        #[cfg(target_os = "windows")]
        let cuda_stream = cuda_ctx.default_stream();

        // --- Encoder session (standard CUDA EP, no graphs) ---
        let encoder = Session::builder()
            .map_err(|e| ExtractError::Model(format!("Encoder session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ExtractError::Model(format!("Encoder opt level: {e}")))?
            .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default()
                .build()])
            .map_err(|e| ExtractError::Model(format!("Encoder EP: {e}")))?
            .commit_from_file(encoder_path)
            .map_err(|e| {
                ExtractError::Model(format!(
                    "Encoder load from {}: {e}",
                    encoder_path.display()
                ))
            })?;

        // --- Decoder session (CUDA EP + CUDA graphs + shared stream) ---
        #[cfg(target_os = "windows")]
        let decoder_ep = {
            let raw_stream = cuda_stream.cu_stream() as *mut ();
            unsafe {
                ort::execution_providers::CUDAExecutionProvider::default()
                    .with_cuda_graph(true)
                    .with_compute_stream(raw_stream)
                    .build()
            }
        };
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

        // --- Determine encoder output shape via dummy forward pass ---
        let enc_output_shape = {
            let mut enc_probe = Session::builder()
                .map_err(|e| ExtractError::Model(format!("Probe session builder: {e}")))?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(|e| ExtractError::Model(format!("Probe opt level: {e}")))?
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .map_err(|e| ExtractError::Model(format!("Probe EP: {e}")))?
                .commit_from_file(encoder_path)
                .map_err(|e| {
                    ExtractError::Model(format!("Probe load: {e}"))
                })?;

            let dummy = preprocess_image(&DynamicImage::new_rgb8(64, 64));
            let input_name = enc_probe.inputs()[0].name().to_string();
            let output_name = enc_probe.outputs()[0].name().to_string();
            let input_tensor = Value::from_array(dummy.into_dyn())
                .map_err(|e| ExtractError::Model(format!("dummy encoder input: {e}")))?;
            let mut binding = enc_probe
                .create_binding()
                .map_err(|e| ExtractError::Model(format!("dummy encoder binding: {e}")))?;
            binding
                .bind_input(&input_name, &input_tensor)
                .map_err(|e| ExtractError::Model(format!("dummy bind_input: {e}")))?;
            binding
                .bind_output_to_device(&output_name, &cuda_mem)
                .map_err(|e| ExtractError::Model(format!("dummy bind_output: {e}")))?;
            let outputs = enc_probe
                .run_binding(&binding)
                .map_err(|e| ExtractError::Model(format!("dummy encoder run: {e}")))?;
            let out = &outputs[0];
            match out.dtype() {
                ValueType::Tensor { shape, .. } => {
                    shape.iter().map(|&d| d as usize).collect::<Vec<_>>()
                }
                _ => {
                    return Err(ExtractError::Model(
                        "encoder output is not a tensor".into(),
                    ))
                }
            }
        };
        tracing::debug!("Encoder output shape: {enc_output_shape:?}");

        let decoder_state = Self::build_decoder_state(
            decoder_session,
            &cuda_mem,
            &enc_output_shape,
        )?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            ExtractError::Model(format!(
                "Tokenizer load from {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let predictor = Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder_state),
            tokenizer,
            cuda_mem,
            #[cfg(target_os = "windows")]
            cuda_stream,
        };

        predictor.warmup()?;
        Ok(predictor)
    }

    /// Build the decoder state with a persistent IoBinding and pre-allocated GPU buffers.
    #[cfg(target_os = "windows")]
    fn build_decoder_state(
        decoder_session: Session,
        cuda_mem: &MemoryInfo,
        enc_output_shape: &[usize],
    ) -> Result<DecoderState, ExtractError> {
        let allocator = Allocator::new(&decoder_session, cuda_mem.clone())
            .map_err(|e| ExtractError::Model(format!("CUDA allocator: {e}")))?;

        let enc_hidden_elems: usize = enc_output_shape.iter().product();
        let enc_hidden_bytes = enc_hidden_elems * std::mem::size_of::<f16>();
        let kv_shape = [1usize, N_HEADS, MAX_SEQ, HEAD_DIM];
        let kv_buf_bytes = N_HEADS * MAX_SEQ * HEAD_DIM * std::mem::size_of::<f16>();

        // --- Allocate input tensors on GPU ---
        let mut input_ids_t = ort::value::Tensor::<i64>::new(&allocator, [1usize, 1])
            .map_err(|e| ExtractError::Model(format!("alloc input_ids: {e}")))?;
        let input_ids_ptr = input_ids_t.data_ptr_mut() as u64;

        let mut step_t = ort::value::Tensor::<i64>::new(&allocator, [1usize])
            .map_err(|e| ExtractError::Model(format!("alloc step: {e}")))?;
        let step_ptr = step_t.data_ptr_mut() as u64;

        let mut enc_hidden_t =
            ort::value::Tensor::<f16>::new(&allocator, enc_output_shape)
                .map_err(|e| ExtractError::Model(format!("alloc enc_hidden: {e}")))?;
        let enc_hidden_ptr = enc_hidden_t.data_ptr_mut() as u64;

        // KV cache buffers (in-place: same buffer used as both input and output)
        let mut kv: Vec<Value> = Vec::with_capacity(N_KV_BUFFERS);
        let mut kv_ptrs: Vec<u64> = Vec::with_capacity(N_KV_BUFFERS);
        for _ in 0..N_KV_BUFFERS {
            let mut t = ort::value::Tensor::<f16>::new(&allocator, kv_shape)
                .map_err(|e| ExtractError::Model(format!("alloc kv: {e}")))?;
            kv_ptrs.push(t.data_ptr_mut() as u64);
            kv.push(t.into());
        }

        // logits output (on GPU, we don't read it — model uses it internally for argmax)
        let logits_t = ort::value::Tensor::<f16>::new(
            &allocator,
            [1usize, 1, 50000], // vocab_size from model metadata
        )
        .map_err(|e| ExtractError::Model(format!("alloc logits: {e}")))?;

        // next_token output
        let mut next_token_t = ort::value::Tensor::<i64>::new(&allocator, [1usize])
            .map_err(|e| ExtractError::Model(format!("alloc next_token: {e}")))?;
        let next_token_ptr = next_token_t.data_ptr_mut() as u64;

        // --- Create persistent IoBinding ---
        let mut binding = decoder_session
            .create_binding()
            .map_err(|e| ExtractError::Model(format!("decoder create_binding: {e}")))?;

        // Convert typed tensors to Values for binding
        let input_ids_val: Value = input_ids_t.into();
        let step_val: Value = step_t.into();
        let enc_hidden_val: Value = enc_hidden_t.into();

        // Bind inputs (borrows — Values must outlive binding)
        binding
            .bind_input("input_ids", &input_ids_val)
            .map_err(|e| ExtractError::Model(format!("bind input_ids: {e}")))?;
        binding
            .bind_input("step", &step_val)
            .map_err(|e| ExtractError::Model(format!("bind step: {e}")))?;
        binding
            .bind_input("encoder_hidden_states", &enc_hidden_val)
            .map_err(|e| ExtractError::Model(format!("bind enc_hidden: {e}")))?;

        for (i, kv_val) in kv.iter().enumerate() {
            binding
                .bind_input(&kv_input_name(i), kv_val)
                .map_err(|e| ExtractError::Model(format!("bind kv_in {i}: {e}")))?;
        }

        // Bind KV outputs to the SAME buffers as inputs (in-place update).
        // We bypass ort's bind_output (which moves the Value) and call the C API
        // directly, so the same GPU buffer serves as both past input and present output.
        {
            use ort::AsPointer;
            let binding_ptr = binding.ptr() as *mut ort::sys::OrtIoBinding;
            let api = ort::api();
            for (i, kv_val) in kv.iter().enumerate() {
                let name = kv_output_name(i);
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
            .bind_output("logits", logits_t)
            .map_err(|e| ExtractError::Model(format!("bind logits: {e}")))?;
        binding
            .bind_output("next_token", next_token_t)
            .map_err(|e| ExtractError::Model(format!("bind next_token: {e}")))?;

        Ok(DecoderState {
            session: decoder_session,
            binding,
            _input_ids: input_ids_val,
            _step: step_val,
            _enc_hidden: enc_hidden_val,
            _kv: kv,
            input_ids_ptr,
            step_ptr,
            enc_hidden_ptr,
            enc_hidden_bytes,
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
        _enc_output_shape: &[usize],
    ) -> Result<DecoderState, ExtractError> {
        Ok(DecoderState {
            session: decoder_session,
        })
    }

    /// Run warmup inferences to prime ORT/CUDA internals.
    fn warmup(&self) -> Result<(), ExtractError> {
        tracing::info!("Warming up formula predictor (3 iterations)...");
        let dummy = DynamicImage::new_rgb8(64, 64);
        for _ in 0..3 {
            let _ = self.predict_one(&dummy)?;
        }
        tracing::info!("Formula predictor warmup complete");
        Ok(())
    }

    /// Predict LaTeX for a batch of cropped formula images.
    pub fn predict(&self, images: &[DynamicImage]) -> Result<Vec<String>, ExtractError> {
        images.iter().map(|img| self.predict_one(img)).collect()
    }

    /// Predict LaTeX for a single formula image.
    fn predict_one(&self, image: &DynamicImage) -> Result<String, ExtractError> {
        let preprocessed = preprocess_image(image);

        // Run encoder — output stays on GPU via IoBinding
        let enc_hidden = self.run_encoder(preprocessed)?;

        // Run decoder loop
        let token_ids = self.decode_formula(enc_hidden)?;

        Ok(decode_tokens(&self.tokenizer, &token_ids))
    }

    /// Run encoder and return the hidden states as a GPU-resident Value.
    fn run_encoder(&self, image: Array4<f16>) -> Result<Value, ExtractError> {
        let mut session = self
            .encoder
            .lock()
            .map_err(|e| ExtractError::Formula(format!("encoder lock: {e}")))?;

        let input_name = session.inputs()[0].name().to_string();
        let output_name = session.outputs()[0].name().to_string();

        let input_tensor = Value::from_array(image.into_dyn())
            .map_err(|e| ExtractError::Formula(format!("encoder input tensor: {e}")))?;

        let mut binding = session
            .create_binding()
            .map_err(|e| ExtractError::Formula(format!("encoder create_binding: {e}")))?;
        binding
            .bind_input(&input_name, &input_tensor)
            .map_err(|e| ExtractError::Formula(format!("encoder bind_input: {e}")))?;
        binding
            .bind_output_to_device(&output_name, &self.cuda_mem)
            .map_err(|e| ExtractError::Formula(format!("encoder bind_output: {e}")))?;

        let mut outputs = session
            .run_binding(&binding)
            .map_err(|e| ExtractError::Formula(format!("encoder run: {e}")))?;

        outputs
            .remove(&output_name)
            .ok_or_else(|| ExtractError::Formula("No encoder output".into()))
    }

    /// Autoregressive decoder loop with persistent IoBinding (zero allocations).
    ///
    /// All GPU buffers are pre-allocated at init. KV cache is in-place: the same
    /// buffer is bound as both past input and present output (via raw ORT C API
    /// BindOutput to bypass Rust's ownership), so the model updates the cache
    /// directly with no D2D copies between steps.
    ///
    /// **Batched sync (K=4):** Profiling showed `cudaStreamSynchronize` accounts
    /// for ~52% of GPU time (~722µs per call). We need to sync to read
    /// `next_token` on the CPU for EOS checking, but we don't need to check
    /// every step. Instead we run K=4 decoder steps back-to-back, enqueuing
    /// async D2H copies into separate CPU buffer slots, then sync once and scan
    /// all K tokens for EOS. This cuts our sync calls by ~75%. Steps past EOS
    /// produce garbage tokens that we discard — at most K-1 wasted steps per
    /// formula, a negligible cost vs the sync savings.
    ///
    /// Token feeding stays entirely on GPU: after each step, a D2D copy moves
    /// `next_token → input_ids` (8 bytes), so the model's output feeds directly
    /// into the next step without crossing PCIe.
    #[cfg(target_os = "windows")]
    fn decode_formula(&self, enc_hidden_val: Value) -> Result<Vec<i64>, ExtractError> {
        const BATCH_K: usize = 4;

        let mut state = self
            .decoder
            .lock()
            .map_err(|e| ExtractError::Formula(format!("decoder lock: {e}")))?;

        let cu_stream = self.cuda_stream.cu_stream();

        // D2D copy: encoder output → fixed enc_hidden buffer
        let src_ptr = value_data_ptr(&enc_hidden_val);
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(
                state.enc_hidden_ptr,
                src_ptr,
                state.enc_hidden_bytes,
                cu_stream,
            )
        }
        .map_err(|e| ExtractError::Formula(format!("D2D enc_hidden: {e}")))?;

        // Zero KV cache buffers for new formula
        for &ptr in &state.kv_ptrs {
            unsafe {
                cudarc::driver::result::memset_d8_async(ptr, 0u8, state.kv_buf_bytes, cu_stream)
            }
            .map_err(|e| ExtractError::Formula(format!("memset kv: {e}")))?;
        }

        let mut tokens = vec![BOS_ID];

        // Seed input_ids with BOS via H2D (only PCIe crossing for input_ids)
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                state.input_ids_ptr,
                &[BOS_ID],
                cu_stream,
            )
        }
        .map_err(|e| ExtractError::Formula(format!("H2D input_ids init: {e}")))?;

        // Token buffer for batched D2H — each slot receives one async D2H copy.
        // Must stay pinned (not moved) until sync completes.
        let mut token_buf = [0i64; BATCH_K];
        let mut s = 0usize;

        while s < MAX_SEQ {
            // Run up to BATCH_K steps without syncing
            let batch_end = (s + BATCH_K).min(MAX_SEQ);
            let batch_len = batch_end - s;

            for k in 0..batch_len {
                // H2D: update step counter
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        state.step_ptr,
                        &[(s + k) as i64],
                        cu_stream,
                    )
                }
                .map_err(|e| ExtractError::Formula(format!("H2D step: {e}")))?;

                // Run decoder step (CUDA graph replay)
                let DecoderState {
                    ref mut session,
                    ref binding,
                    ..
                } = *state;
                session
                    .run_binding(binding)
                    .map_err(|e| {
                        ExtractError::Formula(format!("decoder step {}: {e}", s + k))
                    })?;

                // D2D: next_token → input_ids for next step (GPU-only)
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        state.input_ids_ptr,
                        state.next_token_ptr,
                        std::mem::size_of::<i64>(),
                        cu_stream,
                    )
                }
                .map_err(|e| ExtractError::Formula(format!("D2D next→input: {e}")))?;

                // Async D2H: next_token → token_buf[k] (no sync yet)
                unsafe {
                    cudarc::driver::result::memcpy_dtoh_async(
                        &mut token_buf[k..k + 1],
                        state.next_token_ptr,
                        cu_stream,
                    )
                }
                .map_err(|e| ExtractError::Formula(format!("D2H next_token: {e}")))?;
            }

            // One sync for the entire batch
            self.cuda_stream
                .synchronize()
                .map_err(|e| ExtractError::Formula(format!("stream sync: {e}")))?;

            // Scan batch for EOS
            for k in 0..batch_len {
                tokens.push(token_buf[k]);
                if token_buf[k] == EOS_ID {
                    return Ok(tokens);
                }
            }

            s = batch_end;
        }

        Ok(tokens)
    }

    /// Non-Windows fallback.
    #[cfg(not(target_os = "windows"))]
    fn decode_formula(&self, _enc_hidden_val: Value) -> Result<Vec<i64>, ExtractError> {
        Err(ExtractError::Formula(
            "CUDA decoder only supported on Windows".into(),
        ))
    }
}

/// Preprocess a formula image for the encoder.
///
/// Pipeline: RGB → grayscale threshold → content-crop → resize (fit 768, Lanczos)
/// → center-pad to 768×768 black → luminance → normalize → [1,1,768,768] f16
fn preprocess_image(image: &DynamicImage) -> Array4<f16> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());

    let pixels: Vec<f32> = rgb
        .pixels()
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .collect();
    let min_val = pixels.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0u32, 0u32);
    if max_val > min_val {
        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) as usize;
                let norm = (pixels[idx] - min_val) / (max_val - min_val) * 255.0;
                if norm < 127.0 {
                    x0 = x0.min(px);
                    y0 = y0.min(py);
                    x1 = x1.max(px + 1);
                    y1 = y1.max(py + 1);
                }
            }
        }
    }

    let cropped = if x1 > x0 && y1 > y0 {
        image.crop_imm(x0, y0, x1 - x0, y1 - y0)
    } else {
        image.clone()
    };

    let (cw, ch) = (cropped.width(), cropped.height());
    let scale = TARGET_SIZE as f32 / cw.max(ch) as f32;
    let new_w = (cw as f32 * scale) as u32;
    let new_h = (ch as f32 * scale) as u32;
    let resized = cropped.resize_exact(new_w, new_h, FilterType::Lanczos3);

    let mut padded = DynamicImage::new_rgb8(TARGET_SIZE, TARGET_SIZE);
    let offset_x = (TARGET_SIZE - new_w) / 2;
    let offset_y = (TARGET_SIZE - new_h) / 2;
    image::imageops::overlay(
        padded.as_mut_rgb8().unwrap(),
        &resized.to_rgb8(),
        offset_x as i64,
        offset_y as i64,
    );

    let padded_rgb = padded.to_rgb8();
    let ts = TARGET_SIZE as usize;
    let mut data = vec![f16::ZERO; ts * ts];
    for (i, p) in padded_rgb.pixels().enumerate() {
        let lum = 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
        let normalized = (lum / 255.0 - 0.5) / 0.5;
        data[i] = f16::from_f32(normalized);
    }

    Array4::from_shape_vec([1, 1, ts, ts], data).expect("shape mismatch in preprocess")
}

/// Convert token IDs to a LaTeX string, filtering special tokens.
fn decode_tokens(tokenizer: &tokenizers::Tokenizer, token_ids: &[i64]) -> String {
    let vocab_size = tokenizer.get_vocab_size(true) as i64;
    let valid: Vec<u32> = token_ids
        .iter()
        .copied()
        .take_while(|&t| t != EOS_ID)
        .filter(|&t| t != BOS_ID && t >= 0 && t < vocab_size)
        .map(|t| t as u32)
        .collect();
    if valid.is_empty() {
        return String::new();
    }
    tokenizer.decode(&valid, true).unwrap_or_default()
}

/// Generate the past_key_values input name for a KV buffer index.
fn kv_input_name(i: usize) -> String {
    let layer = i / 2;
    let kv = if i % 2 == 0 { "key" } else { "value" };
    format!("past_key_values.{layer}.{kv}")
}

/// Generate the present_key_values output name for a KV buffer index.
fn kv_output_name(i: usize) -> String {
    let layer = i / 2;
    let kv = if i % 2 == 0 { "key" } else { "value" };
    format!("present_key_values.{layer}.{kv}")
}
