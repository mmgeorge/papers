//! CUDA-specific decode loop with IoBinding and CUDA graphs.
//!
//! Uses `llm_decoder_gqa.onnx` — fixed-shape decoder with GQA fusion,
//! persistent IoBinding, pre-allocated GPU buffers, and ORT built-in
//! CUDA graph capture/replay.

use half::bf16;
use ort::memory::{Allocator, MemoryInfo};
use ort::session::Session;
use ort::value::Value;

use crate::error::ExtractError;

use super::config::*;

/// Decoder state with full IoBinding and CUDA graph support.
pub(crate) struct DecoderState {
    pub session: Session,
    pub binding: ort::io_binding::IoBinding,
    pub run_options: ort::session::RunOptions<ort::session::NoSelectedOutputs>,
    // Input Values (must outlive binding)
    pub _input_ids: Value,
    pub _step: Value,
    pub _prefill_len: Value,
    pub _seqlens_k: Value,
    pub _total_seq_len: Value,
    pub _kv: Vec<Value>,
    // Raw GPU pointers for memcpy
    pub input_ids_ptr: u64,
    pub step_ptr: u64,
    pub prefill_len_ptr: u64,
    pub seqlens_k_ptr: u64,
    pub total_seq_len_ptr: u64,
    pub kv_ptrs: Vec<u64>,
    pub kv_buf_bytes: usize,
    // Output pointers
    pub next_token_ptr: u64,
    pub token_log_prob_ptr: u64,
    pub _allocator: Allocator,
}

/// Allocate GPU buffers and bind all inputs/outputs for the decoder session.
pub(crate) fn build_decoder_state(
    decoder_session: Session,
    cuda_mem: &MemoryInfo,
    params: &DecoderParams,
) -> Result<DecoderState, ExtractError> {
    let allocator = Allocator::new(&decoder_session, cuda_mem.clone())
        .map_err(|e| ExtractError::Model(format!("CUDA allocator: {e}")))?;

    let n_kv_buffers = params.num_layers * 2;
    let kv_shape = [1usize, params.num_kv_heads, params.max_seq, params.head_dim];
    let kv_buf_bytes =
        params.num_kv_heads * params.max_seq * params.head_dim * std::mem::size_of::<bf16>();

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
        let mut t = ort::value::Tensor::<bf16>::new(&allocator, kv_shape)
            .map_err(|e| ExtractError::Model(format!("alloc kv: {e}")))?;
        kv_ptrs.push(t.data_ptr_mut() as u64);
        kv.push(t.into());
    }

    // next_token output
    let mut next_token_t = ort::value::Tensor::<i64>::new(&allocator, [1usize])
        .map_err(|e| ExtractError::Model(format!("alloc next_token: {e}")))?;
    let next_token_ptr = next_token_t.data_ptr_mut() as u64;

    // token_log_prob output (f32, [1] — log-probability of the argmax token)
    let mut token_log_prob_t = ort::value::Tensor::<f32>::new(&allocator, [1usize])
        .map_err(|e| ExtractError::Model(format!("alloc token_log_prob: {e}")))?;
    let token_log_prob_ptr = token_log_prob_t.data_ptr_mut() as u64;

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

    // Bind KV outputs to SAME buffers as inputs (in-place update).
    //
    // We use bind_output_to_device first to register each name in the Rust-side
    // output tracking (needed for debug_assert in SessionOutputs::new), then
    // overwrite the ORT-level binding via the raw C API to point at the actual
    // pre-allocated GPU buffer.
    {
        use ort::AsPointer;
        let binding_ptr = binding.ptr() as *mut ort::sys::OrtIoBinding;
        let api = ort::api();
        for (i, kv_val) in kv.iter().enumerate() {
            let layer = i / 2;
            let kv_type = if i % 2 == 0 { "key" } else { "value" };
            let name = format!("present_{kv_type}_{layer}");

            // Register output name in Rust-side tracking
            binding
                .bind_output_to_device(&name, cuda_mem)
                .map_err(|e| ExtractError::Model(format!("bind_output_to_device kv {i}: {e}")))?;

            // Overwrite with the actual pre-allocated buffer
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
    binding
        .bind_output("token_log_prob", token_log_prob_t)
        .map_err(|e| ExtractError::Model(format!("bind token_log_prob: {e}")))?;

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
        token_log_prob_ptr,
        _allocator: allocator,
    })
}

/// Copy prefill KV cache into decoder's fixed buffers and run decode loop.
///
/// Uses ORT built-in CUDA graph: first `run_with_iobinding` captures the graph,
/// subsequent calls replay it. Between steps we memcpy `step` and `input_ids`
/// into the same fixed GPU buffers (addresses don't change, only content).
pub(crate) fn decode_loop(
    state: &mut DecoderState,
    first_token: i64,
    prefill_kv: &[Vec<bf16>],
    prefill_len: usize,
    num_kv_heads: usize,
    max_seq: usize,
    head_dim: usize,
) -> Result<(Vec<i64>, f32), ExtractError> {
    // Use null stream (default stream) for memcpy — synchronous with ORT's work
    let null_stream = std::ptr::null_mut();

    // Zero all KV cache buffers
    for &ptr in &state.kv_ptrs {
        unsafe {
            cudarc::driver::result::memset_d8_async(ptr, 0u8, state.kv_buf_bytes, null_stream)
        }
        .map_err(|e| ExtractError::Formula(format!("memset kv: {e}")))?;
    }

    // Copy prefill KV cache (CPU bf16) into decoder's fixed GPU buffers
    // Copy head-by-head since strides differ (prefill_len vs max_seq).
    for (i, kv_data) in prefill_kv.iter().enumerate() {
        if prefill_len == 0 {
            continue;
        }

        let src_head_stride = prefill_len * head_dim;
        let dst_head_stride = max_seq * head_dim * std::mem::size_of::<bf16>();
        let copy_elems = prefill_len * head_dim;

        for h in 0..num_kv_heads {
            let src_start = h * src_head_stride;
            let src_slice = &kv_data[src_start..src_start + copy_elems];
            let dst = state.kv_ptrs[i] + (h * dst_head_stride) as u64;
            unsafe { cudarc::driver::result::memcpy_htod_async(dst, src_slice, null_stream) }
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
    let mut log_probs: Vec<f32> = Vec::new();

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
        let confidence = crate::types::FormulaResult::sequence_confidence(&log_probs);
        return Ok((tokens, confidence));
    }

    // Decode loop: ORT CUDA graph handles compute, we update step/input_ids via memcpy
    let mut token_buf = [0i64; 1];
    let mut lp_buf = [0f32; 1];

    for s in 0..max_seq {
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

        // Run decoder step (ORT CUDA graph replay — internally synchronous)
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

        // D2H: read next_token + token_log_prob to check for EOS
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                &mut token_buf,
                state.next_token_ptr,
                null_stream,
            )
        }
        .map_err(|e| ExtractError::Formula(format!("D2H next_token: {e}")))?;
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(
                &mut lp_buf,
                state.token_log_prob_ptr,
                null_stream,
            )
        }
        .map_err(|e| ExtractError::Formula(format!("D2H token_log_prob: {e}")))?;

        tokens.push(token_buf[0]);
        log_probs.push(lp_buf[0]);
        if EOS_IDS.contains(&token_buf[0]) {
            break;
        }
    }

    let confidence = crate::types::FormulaResult::sequence_confidence(&log_probs);
    Ok((tokens, confidence))
}
