//! CUDA-specific decode loop with IoBinding and ORT built-in CUDA graphs.
//!
//! Uses the same `decoder.onnx` as the CPU path, with fixed-shape KV cache
//! and `torch.where` scatter for CUDA graph compatibility. All buffers are
//! pre-allocated on GPU; per-step PCIe traffic is ~2 KB (logits + hidden_state).

use ort::memory::{Allocator, MemoryInfo};
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;

use super::decode::{argmax, DecodeResult, DecoderParams, TokenTracker};
use super::otsl;

/// Decoder state with pre-allocated GPU buffers and persistent IoBinding.
pub(crate) struct DecoderState {
    pub session: Session,
    pub binding: ort::io_binding::IoBinding,
    pub run_options: ort::session::RunOptions<ort::session::NoSelectedOutputs>,
    // Input Values (must outlive binding)
    pub _input_ids: Value,
    pub _step: Value,
    pub _encoder_memory: Value,
    pub _kv: Vec<Value>,
    // Raw GPU pointers for memcpy
    pub input_ids_ptr: u64,
    pub step_ptr: u64,
    pub encoder_memory_ptr: u64,
    pub kv_ptrs: Vec<u64>,
    pub kv_buf_bytes: usize,
    // Output pointers
    pub logits_ptr: u64,
    pub hidden_ptr: u64,
    pub hidden_bytes: usize,
    pub _allocator: Allocator,
}

/// Allocate GPU buffers and bind all inputs/outputs for the decoder session.
pub(crate) fn build_decoder_state(
    decoder_session: Session,
    cuda_mem: &MemoryInfo,
    params: &DecoderParams,
    enc_memory_shape: &[usize], // [1, 784, 512]
) -> Result<DecoderState, ExtractError> {
    let allocator = Allocator::new(&decoder_session, cuda_mem.clone())
        .map_err(|e| ExtractError::Model(format!("CUDA allocator: {e}")))?;

    let n_kv_buffers = params.num_layers * 2;
    let kv_shape = [1usize, params.num_heads, params.max_seq, params.head_dim];
    let kv_buf_bytes =
        params.num_heads * params.max_seq * params.head_dim * std::mem::size_of::<f32>();

    // --- Allocate input tensors on GPU ---
    let mut input_ids_t = ort::value::Tensor::<i64>::new(&allocator, [1usize, 1])
        .map_err(|e| ExtractError::Model(format!("alloc input_ids: {e}")))?;
    let input_ids_ptr = input_ids_t.data_ptr_mut() as u64;

    let mut step_t = ort::value::Tensor::<i64>::new(&allocator, [1usize])
        .map_err(|e| ExtractError::Model(format!("alloc step: {e}")))?;
    let step_ptr = step_t.data_ptr_mut() as u64;

    let mut enc_memory_t =
        ort::value::Tensor::<f32>::new(&allocator, enc_memory_shape)
            .map_err(|e| ExtractError::Model(format!("alloc encoder_memory: {e}")))?;
    let encoder_memory_ptr = enc_memory_t.data_ptr_mut() as u64;

    // KV cache buffers (in-place: same buffer for input and output)
    let mut kv: Vec<Value> = Vec::with_capacity(n_kv_buffers);
    let mut kv_ptrs: Vec<u64> = Vec::with_capacity(n_kv_buffers);
    for _ in 0..n_kv_buffers {
        let mut t = ort::value::Tensor::<f32>::new(&allocator, kv_shape)
            .map_err(|e| ExtractError::Model(format!("alloc kv: {e}")))?;
        kv_ptrs.push(t.data_ptr_mut() as u64);
        kv.push(t.into());
    }

    // --- Allocate output tensors on GPU ---
    // logits [1, 1, 13] f32 — 52 bytes, D2H for CPU argmax
    let mut logits_t =
        ort::value::Tensor::<f32>::new(&allocator, [1usize, 1, otsl::VOCAB_SIZE])
            .map_err(|e| ExtractError::Model(format!("alloc logits: {e}")))?;
    let logits_ptr = logits_t.data_ptr_mut() as u64;

    // hidden_state [1, 1, d_model] f32 — D2H only for cell tokens
    let d_model = params.num_heads * params.head_dim; // 8*64 = 512
    let mut hidden_t =
        ort::value::Tensor::<f32>::new(&allocator, [1usize, 1, d_model])
            .map_err(|e| ExtractError::Model(format!("alloc hidden_state: {e}")))?;
    let hidden_ptr = hidden_t.data_ptr_mut() as u64;
    let hidden_bytes = d_model * std::mem::size_of::<f32>();

    // --- Create persistent IoBinding ---
    let mut binding = decoder_session
        .create_binding()
        .map_err(|e| ExtractError::Model(format!("decoder create_binding: {e}")))?;

    // Convert typed tensors to Values for binding
    let input_ids_val: Value = input_ids_t.into();
    let step_val: Value = step_t.into();
    let enc_memory_val: Value = enc_memory_t.into();

    // Bind inputs (borrows — Values must outlive binding)
    binding
        .bind_input("input_ids", &input_ids_val)
        .map_err(|e| ExtractError::Model(format!("bind input_ids: {e}")))?;
    binding
        .bind_input("step", &step_val)
        .map_err(|e| ExtractError::Model(format!("bind step: {e}")))?;
    binding
        .bind_input("encoder_memory", &enc_memory_val)
        .map_err(|e| ExtractError::Model(format!("bind encoder_memory: {e}")))?;

    for (i, kv_val) in kv.iter().enumerate() {
        let layer = i / 2;
        let kv_type = if i % 2 == 0 { "key" } else { "value" };
        binding
            .bind_input(
                &format!("past_key_values.{layer}.{kv_type}"),
                kv_val,
            )
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
            let name = format!("present_key_values.{layer}.{kv_type}");
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

    // Bind logits and hidden_state outputs
    binding
        .bind_output("logits", logits_t)
        .map_err(|e| ExtractError::Model(format!("bind logits: {e}")))?;
    binding
        .bind_output("hidden_state", hidden_t)
        .map_err(|e| ExtractError::Model(format!("bind hidden_state: {e}")))?;

    let run_options = ort::session::RunOptions::new()
        .map_err(|e| ExtractError::Model(format!("RunOptions: {e}")))?;

    Ok(DecoderState {
        session: decoder_session,
        binding,
        run_options,
        _input_ids: input_ids_val,
        _step: step_val,
        _encoder_memory: enc_memory_val,
        _kv: kv,
        input_ids_ptr,
        step_ptr,
        encoder_memory_ptr,
        kv_ptrs,
        kv_buf_bytes,
        logits_ptr,
        hidden_ptr,
        hidden_bytes,
        _allocator: allocator,
    })
}

/// Run the autoregressive decode loop using IoBinding + ORT built-in CUDA graph.
///
/// The encoder_memory data is copied to the pre-allocated GPU buffer once,
/// then the decode loop runs with minimal PCIe traffic per step:
/// - H2D: input_ids (8 bytes) + step (8 bytes) = 16 bytes
/// - D2H: logits (52 bytes) + hidden_state (2048 bytes, only for cell tokens)
///
/// ORT captures a CUDA graph on the first `run_binding_with_options` call
/// and replays it on subsequent steps.
pub(crate) fn decode_loop(
    state: &mut DecoderState,
    encoder_memory: &[f32], // flattened [1, 784, 512]
) -> Result<DecodeResult, ExtractError> {
    let null_stream = std::ptr::null_mut();

    // Zero all KV cache buffers
    for &ptr in &state.kv_ptrs {
        unsafe {
            cudarc::driver::result::memset_d8_async(ptr, 0u8, state.kv_buf_bytes, null_stream)
        }
        .map_err(|e| ExtractError::Table(format!("memset kv: {e}")))?;
    }

    // H2D: encoder_memory (one-time copy)
    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            state.encoder_memory_ptr,
            encoder_memory,
            null_stream,
        )
    }
    .map_err(|e| ExtractError::Table(format!("H2D encoder_memory: {e}")))?;

    // Seed with START token
    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            state.input_ids_ptr,
            &[otsl::START],
            null_stream,
        )
    }
    .map_err(|e| ExtractError::Table(format!("H2D start token: {e}")))?;

    let mut tracker = TokenTracker::new();
    let mut logits_buf = [0f32; otsl::VOCAB_SIZE]; // 13 floats = 52 bytes
    let d_model = state.hidden_bytes / std::mem::size_of::<f32>();

    for step in 0..512usize {
        // H2D: step counter
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                state.step_ptr,
                &[step as i64],
                null_stream,
            )
        }
        .map_err(|e| ExtractError::Table(format!("H2D step: {e}")))?;

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
                .map_err(|e| ExtractError::Table(format!("decoder step {step}: {e}")))?;
        }

        // D2H: logits [1, 1, 13] → 52 bytes
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                &mut logits_buf,
                state.logits_ptr,
                null_stream,
            )
        }
        .map_err(|e| ExtractError::Table(format!("D2H logits: {e}")))?;

        let raw_tag = argmax(&logits_buf);
        let new_tag = tracker.correct(raw_tag);

        // D2H: hidden_state only if needed for bbox collection
        let hidden = if tracker.needs_hidden(new_tag) {
            let mut hidden_buf = vec![0f32; d_model];
            unsafe {
                cudarc::driver::result::memcpy_dtoh_async(
                    &mut hidden_buf,
                    state.hidden_ptr,
                    null_stream,
                )
            }
            .map_err(|e| ExtractError::Table(format!("D2H hidden: {e}")))?;
            Some(hidden_buf)
        } else {
            None
        };
        tracker.push(new_tag, hidden);

        if new_tag == otsl::END {
            break;
        }

        // H2D: next token → input_ids (8 bytes)
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                state.input_ids_ptr,
                &[new_tag],
                null_stream,
            )
        }
        .map_err(|e| ExtractError::Table(format!("H2D next token: {e}")))?;

        // KV: in-place via torch.where in the ONNX graph — no action needed
    }

    Ok(tracker.into_result())
}
