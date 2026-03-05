//! Vision encoder, embedding, merge, and LLM prefill pass.
//!
//! These are shared by all backends — the prefill
//! always uses `session.run()` (no IoBinding or CUDA graphs).

use std::sync::Mutex;

use half::bf16;
use ndarray::{Array2, Array3};
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;

use super::config::*;
use super::preprocess::compute_vision_pos_ids;

// ── Vision encoder ────────────────────────────────────────────────────

pub(crate) fn run_vision_encoder(
    session: &Mutex<Session>,
    pixel_values: &Array2<f32>,
    grid_thw: &Array2<i64>,
    use_bf16: bool,
) -> Result<Value, ExtractError> {
    let mut session = session
        .lock()
        .map_err(|e| ExtractError::Formula(format!("vision encoder lock: {e}")))?;

    // Compute vision position IDs (M-RoPE)
    let (pos_ids, max_grid_size) = compute_vision_pos_ids(grid_thw);

    let pv_tensor: Value = if use_bf16 {
        // Convert pixel_values f32→bf16 to match the BF16 vision encoder model
        let pv_bf16 = pixel_values.mapv(|v| bf16::from_f32(v));
        Value::from_array(pv_bf16.into_dyn())
            .map_err(|e| ExtractError::Formula(format!("vision pixel_values tensor: {e}")))?
            .into()
    } else {
        // FP32 model — use f32 directly
        Value::from_array(pixel_values.clone().into_dyn())
            .map_err(|e| ExtractError::Formula(format!("vision pixel_values tensor: {e}")))?
            .into()
    };

    let pos_tensor: Value = Value::from_array(pos_ids.into_dyn())
        .map_err(|e| ExtractError::Formula(format!("vision pos_ids tensor: {e}")))?
        .into();
    let grid_tensor: Value =
        Value::from_array(ndarray::arr0(max_grid_size).into_dyn())
            .map_err(|e| ExtractError::Formula(format!("vision max_grid_size tensor: {e}")))?
            .into();

    let output_name = session.outputs()[0].name().to_string();
    let mut outputs = session
        .run(ort::inputs![
            "pixel_values" => pv_tensor,
            "pos_ids" => pos_tensor,
            "max_grid_size" => grid_tensor
        ])
        .map_err(|e| ExtractError::Formula(format!("vision encoder run: {e}")))?;

    outputs
        .remove(&output_name)
        .ok_or_else(|| ExtractError::Formula("No vision encoder output".into()))
}

// ── Token embedding ───────────────────────────────────────────────────

pub(crate) fn run_embedding(
    session: &Mutex<Session>,
    input_ids: &[i64],
) -> Result<Value, ExtractError> {
    let mut session = session
        .lock()
        .map_err(|e| ExtractError::Formula(format!("embedding lock: {e}")))?;

    let seq_len = input_ids.len();
    let ids_array =
        Array2::from_shape_vec([1, seq_len], input_ids.to_vec())
            .map_err(|e| ExtractError::Formula(format!("embedding ids array: {e}")))?;
    let ids_tensor: Value = Value::from_array(ids_array.into_dyn())
        .map_err(|e| ExtractError::Formula(format!("embedding input tensor: {e}")))?
        .into();

    let output_name = session.outputs()[0].name().to_string();
    let mut outputs = session
        .run(ort::inputs!["input_ids" => ids_tensor])
        .map_err(|e| ExtractError::Formula(format!("embedding run: {e}")))?;

    outputs
        .remove(&output_name)
        .ok_or_else(|| ExtractError::Formula("No embedding output".into()))
}

// ── Merge vision embeddings ───────────────────────────────────────────

/// Replace image token positions in token_embeds with vision_embeds.
///
/// Both values are CPU-side. Merge is done in f32, then converted to the
/// appropriate dtype for the LLM prefill (bf16 for CUDA, f32 for CPU/CoreML).
pub(crate) fn merge_vision_embeddings(
    token_embeds: Value,
    vision_embeds: &Value,
    input_ids: &[i64],
    seq_len: usize,
    use_bf16: bool,
) -> Result<Value, ExtractError> {
    let hidden = HIDDEN_SIZE;

    let token_data = extract_f32(&token_embeds, "token_embeds")?;
    let vision_data = extract_f32(vision_embeds, "vision_embeds")?;

    // Build merged array in f32
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

    if use_bf16 {
        // Convert to bf16 for the BF16 LLM prefill model
        let merged_bf16: Vec<bf16> = merged.iter().map(|&v| bf16::from_f32(v)).collect();
        let merged_array =
            ndarray::Array3::from_shape_vec([1, seq_len, hidden], merged_bf16)
                .map_err(|e| ExtractError::Formula(format!("merged embeds shape: {e}")))?;
        Value::from_array(merged_array.into_dyn())
            .map(|v| v.into())
            .map_err(|e| ExtractError::Formula(format!("merged embeds tensor: {e}")))
    } else {
        // Keep as f32 for FP32 models
        let merged_array =
            ndarray::Array3::from_shape_vec([1, seq_len, hidden], merged)
                .map_err(|e| ExtractError::Formula(format!("merged embeds shape: {e}")))?;
        Value::from_array(merged_array.into_dyn())
            .map(|v| v.into())
            .map_err(|e| ExtractError::Formula(format!("merged embeds tensor: {e}")))
    }
}

// ── Prefill ───────────────────────────────────────────────────────────

/// Run the LLM prefill pass with the full prompt.
/// Returns (first_generated_token, kv_cache_values).
///
/// For CUDA backend: kv_cache is Vec<Vec<bf16>> for GPU upload.
/// For CPU/CoreML backend: kv_cache is Vec<Value> for session.run() reuse.
pub(crate) fn run_prefill_for_cuda(
    session: &Mutex<Session>,
    inputs_embeds: Value,
    position_ids: &Array3<i64>,
    seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(i64, Vec<Vec<bf16>>), ExtractError> {
    let mut session = session
        .lock()
        .map_err(|e| ExtractError::Formula(format!("llm lock: {e}")))?;

    let attention_mask = Array2::from_elem([1, seq_len], 1i64);
    let mask_tensor = Value::from_array(attention_mask.into_dyn())
        .map_err(|e| ExtractError::Formula(format!("prefill mask tensor: {e}")))?;
    let pos_tensor = Value::from_array(position_ids.clone().into_dyn())
        .map_err(|e| ExtractError::Formula(format!("prefill pos tensor: {e}")))?;

    // Empty KV cache for prefill (dynamic shape: past_seq_len = 0)
    let n_kv_buffers = num_layers * 2;
    let empty_kv_shape = [1usize, num_kv_heads, 0, head_dim];
    let empty_kv: Vec<Value> = (0..n_kv_buffers)
        .map(|_| {
            let arr = ndarray::Array4::<bf16>::from_elem(empty_kv_shape, bf16::ZERO);
            Value::from_array(arr.into_dyn()).expect("empty kv tensor").into()
        })
        .collect();

    // Build input list
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
    let logits_data = extract_f32(logits, "prefill logits")?;
    let logits_shape = logits.shape();
    let logits_seq_len = logits_shape[1] as usize;
    let vocab = logits_shape[2] as usize;
    let offset = (logits_seq_len - 1) * vocab;
    let last_logits = &logits_data[offset..offset + vocab];
    let first_token = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a): &(usize, &f32), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as i64)
        .unwrap_or(EOS_IDS[0]);

    // Extract KV cache outputs and convert to bf16 for decoder
    let mut kv_cache = Vec::with_capacity(n_kv_buffers);
    for i in 0..num_layers {
        for kv_type in &["key", "value"] {
            let name = format!("present_{kv_type}_{i}");
            let kv_val = outputs
                .get(&name)
                .ok_or_else(|| ExtractError::Formula(format!("No prefill {name}")))?;
            // Try bf16 directly first (BF16 model), then fall back to f32→bf16
            let kv_bf16: Vec<bf16> = if let Ok((_, data)) = kv_val.try_extract_tensor::<bf16>() {
                data.to_vec()
            } else {
                let (_, data) = kv_val
                    .try_extract_tensor::<f32>()
                    .map_err(|e| {
                        ExtractError::Formula(format!("prefill extract {name}: {e}"))
                    })?;
                data.iter().map(|&v| bf16::from_f32(v)).collect()
            };
            kv_cache.push(kv_bf16);
        }
    }

    Ok((first_token, kv_cache))
}

/// Run the LLM prefill pass for CPU/CoreML (returns f32 KV cache as Values).
pub(crate) fn run_prefill_for_other(
    session: &Mutex<Session>,
    inputs_embeds: Value,
    position_ids: &Array3<i64>,
    seq_len: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(i64, Vec<Value>), ExtractError> {
    let mut session = session
        .lock()
        .map_err(|e| ExtractError::Formula(format!("llm lock: {e}")))?;

    let attention_mask = Array2::from_elem([1, seq_len], 1i64);
    let mask_tensor = Value::from_array(attention_mask.into_dyn())
        .map_err(|e| ExtractError::Formula(format!("prefill mask tensor: {e}")))?;
    let pos_tensor = Value::from_array(position_ids.clone().into_dyn())
        .map_err(|e| ExtractError::Formula(format!("prefill pos tensor: {e}")))?;

    // Empty KV cache for prefill (dynamic shape: past_seq_len = 0)
    let n_kv_buffers = num_layers * 2;
    let empty_kv_shape = [1usize, num_kv_heads, 0, head_dim];
    let empty_kv: Vec<Value> = (0..n_kv_buffers)
        .map(|_| {
            let arr = ndarray::Array4::<f32>::from_elem(empty_kv_shape, 0.0f32);
            Value::from_array(arr.into_dyn()).expect("empty kv tensor").into()
        })
        .collect();

    // Build input list
    let mut inputs: Vec<(std::borrow::Cow<str>, Value)> = Vec::new();
    inputs.push(("inputs_embeds".into(), inputs_embeds));
    inputs.push(("attention_mask".into(), mask_tensor.into()));
    inputs.push(("position_ids".into(), pos_tensor.into()));

    for (i, kv_val) in empty_kv.into_iter().enumerate() {
        let layer = i / 2;
        let kv_type = if i % 2 == 0 { "key" } else { "value" };
        inputs.push((format!("past_{kv_type}_{layer}").into(), kv_val));
    }

    let mut outputs = session
        .run(inputs)
        .map_err(|e| ExtractError::Formula(format!("prefill run: {e}")))?;

    // Extract logits → argmax for first token
    let logits = outputs
        .get("logits")
        .ok_or_else(|| ExtractError::Formula("No prefill logits output".into()))?;
    let logits_data = extract_f32(logits, "prefill logits")?;
    let logits_shape = logits.shape();
    let logits_seq_len = logits_shape[1] as usize;
    let vocab = logits_shape[2] as usize;
    let offset = (logits_seq_len - 1) * vocab;
    let last_logits = &logits_data[offset..offset + vocab];
    let first_token = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a): &(usize, &f32), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as i64)
        .unwrap_or(EOS_IDS[0]);

    // Extract KV cache outputs as owned Values
    let mut kv_cache = Vec::with_capacity(n_kv_buffers);
    for i in 0..num_layers {
        for kv_type in &["key", "value"] {
            let name = format!("present_{kv_type}_{i}");
            let kv_val = outputs
                .remove(&name)
                .ok_or_else(|| ExtractError::Formula(format!("No prefill {name}")))?;
            kv_cache.push(kv_val);
        }
    }

    Ok((first_token, kv_cache))
}

// ── Tensor extraction helpers ─────────────────────────────────────────

/// Extract tensor data as f32 from a Value that may contain f32 or bf16 data.
/// ORT auto-upcasts FP16→f32 on CPU but does NOT auto-upcast BF16→f32.
pub(crate) fn extract_f32(value: &Value, name: &str) -> Result<Vec<f32>, ExtractError> {
    // Try f32 first (FP32 models, or FP16 models where ORT upcasts)
    if let Ok((_, data)) = value.try_extract_tensor::<f32>() {
        return Ok(data.to_vec());
    }
    // Try bf16 (BF16 models)
    if let Ok((_, data)) = value.try_extract_tensor::<bf16>() {
        return Ok(data.iter().map(|v| v.to_f32()).collect());
    }
    Err(ExtractError::Formula(format!(
        "extract {name}: unsupported tensor type (expected f32 or bf16)"
    )))
}
