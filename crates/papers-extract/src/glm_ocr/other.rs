//! CPU/CoreML decode loop using session.run() with growing KV cache.
//!
//! Uses `llm.onnx` (dynamic KV cache) for both prefill and decode.
//! No IoBinding, no CUDA graphs — works on CPU and CoreML.

use std::sync::Mutex;

use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;

use super::config::*;
use super::prefill::extract_f32;

/// Run the decode loop using session.run() with growing KV cache.
///
/// Slower than the CUDA path but works on CPU and CoreML.
pub(crate) fn decode_loop(
    llm: &Mutex<Session>,
    embedding: &Mutex<Session>,
    first_token: i64,
    prefill_kv: Vec<Value>,
    prefill_len: usize,
    num_layers: usize,
    _num_kv_heads: usize,
    _head_dim: usize,
    max_seq: usize,
) -> Result<(Vec<i64>, f32), ExtractError> {
    let mut tokens = vec![first_token];
    let mut log_probs: Vec<f32> = Vec::new();

    if EOS_IDS.contains(&first_token) {
        let confidence = crate::types::FormulaResult::sequence_confidence(&log_probs);
        return Ok((tokens, confidence));
    }

    let mut kv_cache = prefill_kv;
    let mut current_token = first_token;
    let mut current_pos = prefill_len as i64;

    for _s in 0..max_seq {
        // Embed the current token
        let next_ids = vec![current_token];
        let token_embeds = {
            let mut session = embedding
                .lock()
                .map_err(|e| ExtractError::Formula(format!("embedding lock: {e}")))?;
            let ids_array = Array2::from_shape_vec([1, 1], next_ids)
                .map_err(|e| ExtractError::Formula(format!("decode ids array: {e}")))?;
            let ids_tensor: Value = Value::from_array(ids_array.into_dyn())
                .map_err(|e| ExtractError::Formula(format!("decode ids tensor: {e}")))?
                .into();
            let output_name = session.outputs()[0].name().to_string();
            let mut outputs = session
                .run(ort::inputs!["input_ids" => ids_tensor])
                .map_err(|e| ExtractError::Formula(format!("decode embedding run: {e}")))?;
            outputs
                .remove(&output_name)
                .ok_or_else(|| ExtractError::Formula("No decode embedding output".into()))?
        };

        // Build position IDs (all 3 dims same for text tokens)
        let pos_val = current_pos;
        let position_ids = ndarray::Array3::from_shape_vec(
            [3, 1, 1],
            vec![pos_val, pos_val, pos_val],
        )
        .map_err(|e| ExtractError::Formula(format!("decode pos shape: {e}")))?;
        let pos_tensor: Value = Value::from_array(position_ids.into_dyn())
            .map_err(|e| ExtractError::Formula(format!("decode pos tensor: {e}")))?
            .into();

        // Attention mask: all 1s for total length
        let total_len = prefill_len + tokens.len();
        let attention_mask = Array2::from_elem([1, total_len], 1i64);
        let mask_tensor: Value = Value::from_array(attention_mask.into_dyn())
            .map_err(|e| ExtractError::Formula(format!("decode mask tensor: {e}")))?
            .into();

        // Build inputs
        let mut inputs: Vec<(std::borrow::Cow<str>, Value)> = Vec::new();
        inputs.push(("inputs_embeds".into(), token_embeds));
        inputs.push(("attention_mask".into(), mask_tensor));
        inputs.push(("position_ids".into(), pos_tensor));

        for (i, kv_val) in kv_cache.into_iter().enumerate() {
            let layer = i / 2;
            let kv_type = if i % 2 == 0 { "key" } else { "value" };
            inputs.push((format!("past_{kv_type}_{layer}").into(), kv_val));
        }

        // Run LLM — extract everything we need while session guard is alive
        let (next_token, token_log_prob, new_kv) = {
            let mut session = llm
                .lock()
                .map_err(|e| ExtractError::Formula(format!("llm lock: {e}")))?;
            let mut outputs = session
                .run(inputs)
                .map_err(|e| ExtractError::Formula(format!("decode run: {e}")))?;

            // Extract logits → argmax + log-prob
            let logits = outputs
                .get("logits")
                .ok_or_else(|| ExtractError::Formula("No decode logits".into()))?;
            let logits_data = extract_f32(logits, "decode logits")?;
            let vocab = logits.shape()[2] as usize;
            let last_logits = &logits_data[logits_data.len() - vocab..];
            let (next_token, token_log_prob) = {
                let (argmax_idx, &max_logit) = last_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap_or((EOS_IDS[0] as usize, &0.0));
                // Numerically stable log-softmax of the max logit
                let log_sum_exp = {
                    let max_val = max_logit;
                    let sum_exp: f32 = last_logits
                        .iter()
                        .map(|&x| (x - max_val).exp())
                        .sum();
                    max_val + sum_exp.ln()
                };
                (argmax_idx as i64, max_logit - log_sum_exp)
            };

            // Extract updated KV cache
            let mut new_kv = Vec::with_capacity(num_layers * 2);
            for i in 0..num_layers {
                for kv_type in &["key", "value"] {
                    let name = format!("present_{kv_type}_{i}");
                    let kv_val = outputs
                        .remove(&name)
                        .ok_or_else(|| ExtractError::Formula(format!("No decode {name}")))?;
                    new_kv.push(kv_val);
                }
            }
            (next_token, token_log_prob, new_kv)
        };

        tokens.push(next_token);
        log_probs.push(token_log_prob);
        if EOS_IDS.contains(&next_token) {
            break;
        }

        kv_cache = new_kv;
        current_token = next_token;
        current_pos += 1;

        // Repetition detection
        if tokens.len() >= 10 {
            let last_10 = &tokens[tokens.len() - 10..];
            if last_10.iter().all(|&t| t == last_10[9]) {
                break;
            }
        }
    }

    let confidence = crate::types::FormulaResult::sequence_confidence(&log_probs);
    Ok((tokens, confidence))
}
