//! Custom CUDA formula predictor — replaces oar-ocr FormulaRecognitionPredictor.
//!
//! Uses split encoder/decoder ONNX models with CUDA EP + FP16 for fast GPU
//! inference. The decoder runs an autoregressive loop using IoBinding to keep
//! KV cache on GPU between steps (only next_token crosses GPU→CPU per step).

use std::path::Path;
use std::sync::Mutex;

use half::f16;
use image::imageops::FilterType;
use image::DynamicImage;
use ndarray::Array4;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;

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

/// Custom formula predictor — replaces oar-ocr FormulaRecognitionPredictor.
pub struct FormulaPredictor {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    /// MemoryInfo for CUDA device 0 (reused across calls).
    cuda_mem: MemoryInfo,
    /// MemoryInfo for CPU (used for next_token output).
    cpu_mem: MemoryInfo,
}

impl FormulaPredictor {
    /// Create a new formula predictor from split encoder/decoder ONNX models.
    pub fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, ExtractError> {
        // Encoder session (standard CUDA EP)
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

        // Decoder session (CUDA EP)
        let decoder = Session::builder()
            .map_err(|e| ExtractError::Model(format!("Decoder session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| ExtractError::Model(format!("Decoder opt level: {e}")))?
            .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default()
                .build()])
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

        let cpu_mem = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| ExtractError::Model(format!("CPU MemoryInfo: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            ExtractError::Model(format!(
                "Tokenizer load from {}: {e}",
                tokenizer_path.display()
            ))
        })?;

        let predictor = Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            tokenizer,
            cuda_mem,
            cpu_mem,
        };

        // Warmup: run a few dummy inferences to prime ORT/CUDA
        predictor.warmup()?;

        Ok(predictor)
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

        // Decode tokens to LaTeX string
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

        // Use IoBinding so the encoder output stays on GPU
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

    /// Autoregressive decoder loop: generate tokens until EOS or MAX_SEQ.
    ///
    /// Uses IoBinding per step to keep KV cache on GPU. Only next_token
    /// (8 bytes) is copied back to CPU each step.
    fn decode_formula(&self, enc_hidden: Value) -> Result<Vec<i64>, ExtractError> {
        let mut session = self
            .decoder
            .lock()
            .map_err(|e| ExtractError::Formula(format!("decoder lock: {e}")))?;

        // Initial KV cache: zeros on CPU (ORT copies to GPU on first use)
        let kv_size = N_HEADS * MAX_SEQ * HEAD_DIM;
        let mut kv_values: Vec<Value> = (0..N_KV_BUFFERS)
            .map(|_| {
                let data = vec![f16::ZERO; kv_size];
                let arr =
                    Array4::<f16>::from_shape_vec([1, N_HEADS, MAX_SEQ, HEAD_DIM], data).unwrap();
                Value::from_array(arr.into_dyn())
                    .map(|v| v.into())
                    .map_err(|e| ExtractError::Formula(format!("init kv zeros: {e}")))
            })
            .collect::<Result<_, _>>()?;

        let mut tokens = vec![BOS_ID];

        for s in 0..MAX_SEQ {
            let input_ids_arr = ndarray::array![[*tokens.last().unwrap()]];
            let step_arr = ndarray::array![s as i64];

            let input_ids_val = Value::from_array(input_ids_arr.into_dyn())
                .map_err(|e| ExtractError::Formula(format!("input_ids tensor: {e}")))?;
            let step_val = Value::from_array(step_arr.into_dyn())
                .map_err(|e| ExtractError::Formula(format!("step tensor: {e}")))?;

            let mut binding = session
                .create_binding()
                .map_err(|e| ExtractError::Formula(format!("decoder create_binding: {e}")))?;

            // Bind scalar inputs
            binding
                .bind_input("input_ids", &input_ids_val)
                .map_err(|e| ExtractError::Formula(format!("bind input_ids: {e}")))?;
            binding
                .bind_input("step", &step_val)
                .map_err(|e| ExtractError::Formula(format!("bind step: {e}")))?;
            binding
                .bind_input("encoder_hidden_states", &enc_hidden)
                .map_err(|e| ExtractError::Formula(format!("bind encoder_hidden: {e}")))?;

            // Bind KV cache inputs (may be CPU for step 0, GPU for subsequent steps)
            for (i, kv) in kv_values.iter().enumerate() {
                let name = kv_input_name(i);
                binding
                    .bind_input(&name, kv)
                    .map_err(|e| ExtractError::Formula(format!("bind {name}: {e}")))?;
            }

            // Bind outputs: KV cache and logits stay on GPU, next_token to CPU
            for i in 0..N_KV_BUFFERS {
                let name = kv_output_name(i);
                binding
                    .bind_output_to_device(&name, &self.cuda_mem)
                    .map_err(|e| ExtractError::Formula(format!("bind_out {name}: {e}")))?;
            }
            binding
                .bind_output_to_device("logits", &self.cuda_mem)
                .map_err(|e| ExtractError::Formula(format!("bind_out logits: {e}")))?;
            binding
                .bind_output_to_device("next_token", &self.cpu_mem)
                .map_err(|e| ExtractError::Formula(format!("bind_out next_token: {e}")))?;

            // Run one decoder step
            let mut outputs = session
                .run_binding(&binding)
                .map_err(|e| ExtractError::Formula(format!("decoder step {s}: {e}")))?;

            // Read next_token (CPU-resident)
            let next_token_val = outputs
                .remove("next_token")
                .ok_or_else(|| ExtractError::Formula("missing next_token output".into()))?;
            let next_token = extract_i64_scalar(next_token_val)?;

            tokens.push(next_token);
            if next_token == EOS_ID {
                break;
            }

            // Update KV cache with GPU-resident output values
            kv_values = (0..N_KV_BUFFERS)
                .map(|i| {
                    let name = kv_output_name(i);
                    outputs
                        .remove(&name)
                        .ok_or_else(|| ExtractError::Formula(format!("missing output {name}")))
                })
                .collect::<Result<_, _>>()?;
        }

        Ok(tokens)
    }
}

/// Extract a single i64 from a Value (expected to be a 1-element tensor on CPU).
fn extract_i64_scalar(value: Value) -> Result<i64, ExtractError> {
    let view = value
        .try_extract_array::<i64>()
        .map_err(|e| ExtractError::Formula(format!("extract next_token: {e}")))?;
    Ok(*view
        .iter()
        .next()
        .ok_or_else(|| ExtractError::Formula("empty next_token tensor".into()))?)
}

/// Preprocess a formula image for the encoder.
///
/// Pipeline: RGB → grayscale threshold → content-crop → resize (fit 768, Lanczos)
/// → center-pad to 768×768 black → luminance → normalize → [1,1,768,768] f16
fn preprocess_image(image: &DynamicImage) -> Array4<f16> {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());

    // Grayscale and normalize to find content bounding box
    let pixels: Vec<f32> = rgb
        .pixels()
        .map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 3.0)
        .collect();
    let min_val = pixels.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Find content bounding box via thresholding
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

    // Crop to content (or use full image if no content found)
    let cropped = if x1 > x0 && y1 > y0 {
        image.crop_imm(x0, y0, x1 - x0, y1 - y0)
    } else {
        image.clone()
    };

    // Resize to fit TARGET_SIZE, preserving aspect ratio
    let (cw, ch) = (cropped.width(), cropped.height());
    let scale = TARGET_SIZE as f32 / cw.max(ch) as f32;
    let new_w = (cw as f32 * scale) as u32;
    let new_h = (ch as f32 * scale) as u32;
    let resized = cropped.resize_exact(new_w, new_h, FilterType::Lanczos3);

    // Center-pad to TARGET_SIZE × TARGET_SIZE on black background
    let mut padded = DynamicImage::new_rgb8(TARGET_SIZE, TARGET_SIZE);
    let offset_x = (TARGET_SIZE - new_w) / 2;
    let offset_y = (TARGET_SIZE - new_h) / 2;
    image::imageops::overlay(
        padded.as_mut_rgb8().unwrap(),
        &resized.to_rgb8(),
        offset_x as i64,
        offset_y as i64,
    );

    // Convert to luminance, normalize, and pack as f16
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
