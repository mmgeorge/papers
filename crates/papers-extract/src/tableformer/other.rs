//! CPU/CoreML decode loop using session.run() with ndarray KV swap.
//!
//! Uses the same `decoder.onnx` as the CUDA path. No IoBinding, no CUDA
//! graphs — works on CPU and CoreML.

use std::sync::Mutex;

use ndarray::{Array2, Array4, ArrayD, ArrayViewD};
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;

use super::decode::{argmax, DecodeResult, DecoderParams, TokenTracker};
use super::otsl;

/// Run the autoregressive decode loop using session.run().
pub(crate) fn decode_loop(
    decoder: &Mutex<Session>,
    encoder_memory: ArrayViewD<'_, f32>,
    params: &DecoderParams,
) -> Result<DecodeResult, ExtractError> {
    let kv_shape = [1, params.num_heads, params.max_seq, params.head_dim];

    // Pre-allocate KV cache buffers (all zeros)
    let mut past_kv: Vec<Array4<f32>> = (0..params.num_layers * 2)
        .map(|_| Array4::zeros(kv_shape))
        .collect();

    let mut input_ids = Array2::<i64>::zeros([1, 1]);
    input_ids[[0, 0]] = otsl::START;

    // Clone encoder_memory once — we'll reuse it every step
    let enc_mem_owned: ArrayD<f32> = encoder_memory.to_owned();

    let mut tracker = TokenTracker::new();

    for step in 0..params.max_seq {
        let step_arr = ndarray::arr1(&[step as i64]);

        // Build input feed
        let mut feed: Vec<(std::borrow::Cow<str>, Value)> =
            Vec::with_capacity(3 + params.num_layers * 2);

        let ids_val: Value = Value::from_array(input_ids.clone().into_dyn())
            .map_err(|e| ExtractError::Model(format!("input_ids tensor: {e}")))?
            .into();
        feed.push(("input_ids".into(), ids_val));

        let enc_val: Value = Value::from_array(enc_mem_owned.clone())
            .map_err(|e| ExtractError::Model(format!("encoder_memory tensor: {e}")))?
            .into();
        feed.push(("encoder_memory".into(), enc_val));

        let step_val: Value = Value::from_array(step_arr.into_dyn())
            .map_err(|e| ExtractError::Model(format!("step tensor: {e}")))?
            .into();
        feed.push(("step".into(), step_val));

        for (i, kv) in past_kv.iter().enumerate() {
            let layer = i / 2;
            let kv_type = if i % 2 == 0 { "key" } else { "value" };
            let name = format!("past_key_values.{layer}.{kv_type}");
            let kv_val: Value = Value::from_array(kv.clone().into_dyn())
                .map_err(|e| ExtractError::Model(format!("KV tensor: {e}")))?
                .into();
            feed.push((name.into(), kv_val));
        }

        // Run decoder step
        let mut dec = decoder
            .lock()
            .map_err(|e| ExtractError::Table(format!("decoder lock: {e}")))?;
        let outputs = dec
            .run(feed)
            .map_err(|e| ExtractError::Model(format!("Decoder run failed: {e}")))?;

        // outputs[0] = logits [1, 1, 13]
        // outputs[1] = hidden_state [1, 1, 512]
        // outputs[2..] = present KV tensors
        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| ExtractError::Model(format!("logits extract: {e}")))?;
        let (_, hidden_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| ExtractError::Model(format!("hidden extract: {e}")))?;

        // Argmax over vocab
        let vocab_start = logits_data.len() - otsl::VOCAB_SIZE;
        let raw_tag = argmax(&logits_data[vocab_start..]);
        let new_tag = tracker.correct(raw_tag);

        // Collect hidden state if needed
        let hidden = if tracker.needs_hidden(new_tag) {
            Some(hidden_data.to_vec())
        } else {
            None
        };
        tracker.push(new_tag, hidden);

        if new_tag == otsl::END {
            break;
        }

        // Swap present KV → past KV
        for i in 0..params.num_layers * 2 {
            let kv_arr = outputs[2 + i]
                .try_extract_array::<f32>()
                .map_err(|e| ExtractError::Model(format!("KV extract: {e}")))?;
            past_kv[i] = kv_arr.to_owned().into_dimensionality().unwrap();
        }

        input_ids[[0, 0]] = new_tag;
    }

    Ok(tracker.into_result())
}
