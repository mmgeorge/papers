//! TableFormer V1 table structure recognition — OTSL token sequence → HTML skeleton.
//!
//! Split ONNX model:
//!   - `encoder.onnx` — ResNet-18 + TransformerEncoder (runs once per table)
//!   - `decoder.onnx` — autoregressive decoder with fixed-size KV cache (~200 steps)
//!   - `bbox_decoder.onnx` — attention-gated MLP for cell bounding boxes (runs once)
//!
//! ## CUDA backend (Windows, NVIDIA GPU)
//! Persistent IoBinding, pre-allocated GPU buffers, ORT built-in CUDA graph.
//!
//! ## CPU/CoreML backend (all platforms)
//! session.run()-based decode loop, ndarray KV swap per step.

#[cfg(target_os = "windows")]
mod cuda;
pub mod decode;
mod other;
pub mod otsl;
pub mod preprocess;

use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;

use decode::{extract_decoder_params, DecoderParams};

// ── Backend enum ─────────────────────────────────────────────────────

enum ActiveBackend {
    #[cfg(target_os = "windows")]
    Cuda {
        decoder: Mutex<cuda::DecoderState>,
        #[allow(dead_code)]
        cuda_ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    },
    Other {
        decoder: Mutex<Session>,
    },
}

/// TableFormer predictor — holds encoder, decoder, and bbox decoder sessions.
pub struct TableFormerPredictor {
    encoder: Mutex<Session>,
    bbox_decoder: Mutex<Session>,
    decoder_params: DecoderParams,
    backend: ActiveBackend,
}

impl TableFormerPredictor {
    /// Load the three ONNX models and extract decoder parameters.
    pub fn new(
        encoder_path: &Path,
        decoder_path: &Path,
        bbox_decoder_path: &Path,
    ) -> Result<Self, ExtractError> {
        // Encoder + bbox decoder: use platform-appropriate EP
        let encoder_ep = Self::make_ep(false);
        let encoder = build_session(encoder_path, GraphOptimizationLevel::Level3, encoder_ep)?;

        let bbox_ep = Self::make_ep(false);
        let bbox_decoder =
            build_session(bbox_decoder_path, GraphOptimizationLevel::Level3, bbox_ep)?;

        // Build backend-specific decoder state
        let (backend, decoder_params) = Self::build_backend(decoder_path)?;

        let backend_name = match &backend {
            #[cfg(target_os = "windows")]
            ActiveBackend::Cuda { .. } => "CUDA",
            ActiveBackend::Other { .. } => {
                if cfg!(target_os = "macos") {
                    "CoreML"
                } else {
                    "CPU"
                }
            }
        };

        tracing::info!(
            "TableFormer decoder: {} layers, {} heads, max_seq={}, head_dim={}, backend={}",
            decoder_params.num_layers,
            decoder_params.num_heads,
            decoder_params.max_seq,
            decoder_params.head_dim,
            backend_name,
        );
        eprintln!(
            "TableFormer: {} backend, {} layers, {} heads",
            backend_name, decoder_params.num_layers, decoder_params.num_heads,
        );

        Ok(Self {
            encoder: Mutex::new(encoder),
            bbox_decoder: Mutex::new(bbox_decoder),
            decoder_params,
            backend,
        })
    }

    fn build_backend(
        decoder_path: &Path,
    ) -> Result<(ActiveBackend, DecoderParams), ExtractError> {
        #[cfg(target_os = "windows")]
        {
            // Try CUDA backend
            let cuda_ctx = cudarc::driver::CudaContext::new(0);
            if let Ok(cuda_ctx) = cuda_ctx {
                tracing::info!("TableFormer: using CUDA backend");

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

                // encoder_memory shape: [1, 784, 512]
                // 784 = 28*28 spatial positions, 512 = d_model
                let enc_memory_shape = [1usize, 784, 512];

                let decoder_state = cuda::build_decoder_state(
                    decoder_session,
                    &cuda_mem,
                    &decoder_params,
                    &enc_memory_shape,
                )?;

                return Ok((
                    ActiveBackend::Cuda {
                        decoder: Mutex::new(decoder_state),
                        cuda_ctx,
                    },
                    decoder_params,
                ));
            }
            tracing::info!("TableFormer: CUDA unavailable, falling back to CPU");
        }

        // CPU/CoreML backend
        let decoder_ep = Self::make_ep(false);
        let decoder_session =
            build_session(decoder_path, GraphOptimizationLevel::Level3, decoder_ep)?;
        let decoder_params = extract_decoder_params(&decoder_session)?;

        Ok((
            ActiveBackend::Other {
                decoder: Mutex::new(decoder_session),
            },
            decoder_params,
        ))
    }

    fn make_ep(
        _cuda_graph: bool,
    ) -> ort::execution_providers::ExecutionProviderDispatch {
        #[cfg(target_os = "windows")]
        {
            if _cuda_graph {
                return ort::execution_providers::CUDAExecutionProvider::default()
                    .with_cuda_graph(true)
                    .build();
            }
            ort::execution_providers::CUDAExecutionProvider::default().build()
        }
        #[cfg(target_os = "macos")]
        {
            ort::execution_providers::CoreMLExecutionProvider::default().build()
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        {
            ort::execution_providers::CPUExecutionProvider::default().build()
        }
    }

    /// Predict table structure for one image, returning an HTML skeleton.
    pub fn predict_one(&self, image: &DynamicImage) -> Result<String, ExtractError> {
        // 1. Preprocess image → [1, 3, 448, 448]
        let pixel_values = preprocess::preprocess(image);

        // 2. Encode → encoder_memory [1, 784, 512] + enc_out_raw [1, 256, 28, 28]
        let (encoder_memory, enc_out_raw) = {
            let pv_tensor: Value = Value::from_array(pixel_values.into_dyn())
                .map_err(|e| ExtractError::Table(format!("pixel_values tensor: {e}")))?
                .into();

            let mut enc = self
                .encoder
                .lock()
                .map_err(|e| ExtractError::Table(format!("encoder lock: {e}")))?;
            let outputs = enc
                .run(ort::inputs!["pixel_values" => pv_tensor])
                .map_err(|e| ExtractError::Table(format!("encoder run: {e}")))?;

            let enc_mem = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| ExtractError::Table(format!("encoder_memory extract: {e}")))?
                .to_owned();
            let enc_raw = outputs[1]
                .try_extract_array::<f32>()
                .map_err(|e| ExtractError::Table(format!("enc_out_raw extract: {e}")))?
                .to_owned();
            (enc_mem, enc_raw)
        };

        // 3. Decode → token sequence + cell hidden states (backend-specific)
        let decode_result = match &self.backend {
            #[cfg(target_os = "windows")]
            ActiveBackend::Cuda { decoder, .. } => {
                let enc_mem_flat = encoder_memory
                    .as_slice()
                    .ok_or_else(|| {
                        ExtractError::Table("encoder_memory not contiguous".into())
                    })?;

                let mut state = decoder
                    .lock()
                    .map_err(|e| ExtractError::Table(format!("decoder lock: {e}")))?;
                cuda::decode_loop(&mut state, enc_mem_flat)?
            }
            ActiveBackend::Other { decoder } => {
                other::decode_loop(decoder, encoder_memory.view(), &self.decoder_params)?
            }
        };

        tracing::debug!(
            "TableFormer: {} tokens, {} cell hidden states",
            decode_result.tokens.len(),
            decode_result.cell_hidden_states.len(),
        );

        // 4. BBox prediction (for future text filling — run but don't use results yet)
        if !decode_result.cell_hidden_states.is_empty() {
            let n_cells = decode_result.cell_hidden_states.len();
            let d_model = decode_result.cell_hidden_states[0].len();

            // Stack hidden states → [N, d_model]
            let mut cell_hidden = Array2::<f32>::zeros([n_cells, d_model]);
            for (i, h) in decode_result.cell_hidden_states.iter().enumerate() {
                cell_hidden.row_mut(i).assign(&ndarray::ArrayView1::from(h));
            }

            let enc_raw_tensor: Value = Value::from_array(enc_out_raw.to_owned())
                .map_err(|e| ExtractError::Table(format!("enc_out_raw tensor: {e}")))?
                .into();
            let cell_hidden_tensor: Value = Value::from_array(cell_hidden.into_dyn())
                .map_err(|e| ExtractError::Table(format!("cell_hidden tensor: {e}")))?
                .into();

            let mut bbox_dec = self
                .bbox_decoder
                .lock()
                .map_err(|e| ExtractError::Table(format!("bbox_decoder lock: {e}")))?;

            let outputs = bbox_dec
                .run(ort::inputs![
                    "enc_out_raw" => enc_raw_tensor,
                    "cell_hidden_states" => cell_hidden_tensor,
                ])
                .map_err(|e| ExtractError::Table(format!("bbox_decoder run: {e}")))?;

            let bboxes = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| ExtractError::Table(format!("bboxes extract: {e}")))?;
            tracing::debug!(
                "TableFormer: {} bboxes predicted (shape {:?})",
                bboxes.shape()[0],
                bboxes.shape(),
            );
        }

        // 5. Convert token sequence → HTML skeleton
        let html = otsl::otsl_to_html(&decode_result.tokens);
        Ok(html)
    }

    /// Predict table structure for multiple images.
    pub fn predict(&self, images: &[DynamicImage]) -> Result<Vec<String>, ExtractError> {
        images.iter().map(|img| self.predict_one(img)).collect()
    }
}

/// Build an ORT session with the given execution provider.
fn build_session(
    path: &Path,
    opt_level: GraphOptimizationLevel,
    ep: ort::execution_providers::ExecutionProviderDispatch,
) -> Result<Session, ExtractError> {
    Session::builder()
        .map_err(|e| ExtractError::Model(format!("session builder: {e}")))?
        .with_optimization_level(opt_level)
        .map_err(|e| ExtractError::Model(format!("opt level: {e}")))?
        .with_execution_providers([ep])
        .map_err(|e| ExtractError::Model(format!("EP: {e}")))?
        .commit_from_file(path)
        .map_err(|e| ExtractError::Model(format!("load {}: {e}", path.display())))
}
