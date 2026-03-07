# Porting GLM-OCR to Rust (ort + cudarc)

This documents the process of porting the GLM-OCR CUDA inference pipeline from
Python (onnxruntime + ctypes cudart) to Rust (ort 2.0.0-rc.11 + cudarc 0.16).

## Architecture: Backend-Dependent Model Split

GLM-OCR supports 3 backends: CUDA (BF16 + IoBinding + CUDA graphs), CoreML (FP32),
and CPU (FP32). The CUDA path uses persistent IoBinding + pre-allocated GPU buffers;
the CoreML/CPU path uses simpler `session.run()` with growing KV cache.

CUDA uses 4 ONNX sessions; CoreML/CPU use 3 (no separate decoder):

| Model | CUDA (BF16) | CoreML/CPU (FP32) | Purpose |
|-------|-------------|-------------------|---------|
| `vision_encoder_mha.onnx` | ~844 MB | — | MHA-fused CogViT (CUDA only) |
| `vision_encoder.onnx` | — | ~3.4 GB | Raw unfused CogViT (CoreML/CPU) |
| `embedding.onnx` | ~174 MB | ~348 MB | Token embeddings |
| `llm.onnx` | ~1.1 GB | ~2.2 GB | Full LLM (prefill + decode on CoreML/CPU) |
| `llm_decoder_gqa.onnx` | ~1.3 GB | — | GQA-fused decoder (CUDA only) |

The prefill phase processes the full prompt (system message + image tokens +
"Formula Recognition:") through the LLM once to populate the KV cache before
decoding begins.

On CUDA, the prefill uses `llm.onnx` (dynamic shapes) and decode uses
`llm_decoder_gqa.onnx` (fixed shapes, CUDA graphed). On CoreML/CPU, `llm.onnx`
handles both prefill and decode with growing KV cache per step.

## Export Wrapper Pattern

The export uses a thin PyTorch `nn.Module` wrapper around the HuggingFace model
that reshapes the interface for CUDA-graph-compatible ONNX export. The wrapper:

1. Takes fixed-shape inputs (`input_ids [1,1]`, `step [1]`, `prefill_len [1]`,
   KV cache `[1, 8, 512, 128]` × 32)
2. Calls into the HuggingFace model's forward method internally
3. Adds ArgMax to keep token selection on GPU

When `torch.onnx.export()` traces this wrapper, it flattens everything — our
wrapper code, the HuggingFace model internals, and the trained weights — into a
single static ONNX computation graph. No Python or HuggingFace code is needed at
runtime.

## Files

| File | Purpose |
|------|---------|
| `src/glm_ocr/` | `GlmOcrPredictor` — GLM-OCR with CUDA (IoBinding + CUDA graphs) and CPU/CoreML backends |
| `src/models.rs` | Model download, ORT runtime init, EP configuration |
| `src/bin/run_glm_ocr.rs` | Standalone GLM-OCR benchmark |
| `py/glm-ocr/cuda/export.py` | GLM-OCR 4-part ONNX export (vision, embedding, llm, decoder) |
| `py/glm-ocr/cuda/run.py` | GLM-OCR Python reference |

---

## Detailed Issues and Fixes

### 1. GatherND breaks CUDA graphs (decoder)

**Symptom:** `CUDA_ERROR_ILLEGAL_ADDRESS` on CUDA graph replay (step 1 with
in-place KV, step 2 with separate KV buffers). Both ORT built-in CUDA graphs
and manual cudarc capture crash identically. Confirmed in both Rust and Python.

**Diagnosis:** `compute-sanitizer --tool memcheck` on a minimal Python test
pinpointed the exact kernel:

```
Invalid __global__ read of size 1 bytes
  at void onnxruntime::cuda::_GatherNDKernel<bool>(...)
  Address 0xf859cd4bed00 is out of bounds
  Host Frame: cudaGraphLaunch
```

The GatherND op (node 13 in the ONNX graph) reads from a dynamically-computed
boolean tensor (`_to_copy_1`, the causal attention mask). During graph capture,
ORT places this tensor in an internal temporary buffer at some address X. During
graph replay, ORT's memory arena may reuse that temporary at a different address
Y. But the captured CUDA graph has address X baked in — so GatherND reads from
stale memory.

Operations like `Where`, `Less`, `Add` don't have this problem because they
operate element-wise on tensors at fixed IoBinding addresses or constant
initializers. GatherND does **indirect addressing** — the values in its index
tensor determine where in memory to read — which is fundamentally incompatible
with CUDA graphs when the data tensor is an ORT-internal temporary.

**Root cause chain:**

1. Our `DecoderStepWrapper` (in `export.py`) passed a **2D** `attention_mask [1, MAX_SEQ]`
   (just 1s and 0s) to `self.language_model(...)`
2. Inside `language_model.forward()`, HuggingFace's `create_causal_mask()`
   expanded this 2D mask into a 4D mask `[1, 1, 1, MAX_SEQ]` that attention
   layers consume
3. That expansion used index arithmetic that, when traced by PyTorch's ONNX
   exporter, became a GatherND node in the exported graph
4. The GatherND's data tensor was an ORT-internal temporary → unstable address
   across graph replays → crash

**Fix:** Pre-compute the 4D causal mask directly in the export wrapper, before
calling into HuggingFace's code:

```python
# Before (2D mask → HuggingFace expands internally → GatherND in ONNX):
positions = torch.arange(self.max_seq, device=input_ids.device).unsqueeze(0)
attention_mask = (positions < (cache_pos + 1).unsqueeze(-1)).long()  # [1, MAX_SEQ]

# After (4D mask → HuggingFace returns as-is → no GatherND):
min_dtype = torch.finfo(inputs_embeds.dtype).min
positions = torch.arange(self.max_seq, device=input_ids.device)
attend = positions < (cache_pos + 1)  # [MAX_SEQ] bool
attention_mask = torch.where(
    attend.view(1, 1, 1, -1),
    torch.tensor(0.0, dtype=inputs_embeds.dtype, device=input_ids.device),
    torch.tensor(min_dtype, dtype=inputs_embeds.dtype, device=input_ids.device),
)  # [1, 1, 1, MAX_SEQ]
```

This works because HuggingFace's `create_causal_mask()` has an early-exit check
(`masking_utils.py` line 788):

```python
if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
    return True, attention_mask, None, None, None  # return as-is
```

When the mask is already 4D, the entire internal expansion code (which generates
GatherND during ONNX tracing) is never executed. The mask content is identical —
`0.0` for attended positions, `-65504` (fp16 min) for masked positions.

**ONNX op counts before/after:**

| Op | Before | After |
|----|--------|-------|
| GatherND | 1 | **0** |
| ScatterND | 0 | 0 |
| IsNaN | 16 | **0** |
| Total nodes | 1591 | 1551 |

IsNaN ops (NaN guards after Softmax in attention) also disappeared — likely
because the 4D mask with proper `-inf` masking prevents NaN from appearing in
Softmax outputs.

**Result:** ORT built-in CUDA graph (`enable_cuda_graph=true`) works correctly.

### 2. ORT built-in CUDA graphs

GLM-OCR uses ORT's built-in CUDA graph support rather than external cudarc capture:

```rust
// Let ORT manage graph capture/replay
CUDAExecutionProvider::default()
    .with_cuda_graph(true)
    .build()

// ORT captures on first call, replays on subsequent calls
session.run_binding_with_options(&binding, &run_options)?;
```

ORT's built-in approach is simpler (no stream management, no capture/replay
code) but forces a `cudaStreamSynchronize` after every graph replay (hardcoded
in ORT's `CUDAGraphManager::Replay()`). For GLM-OCR, the per-step compute is
large enough (1536 hidden, 16 layers, GQA) that the sync overhead is
proportionally less significant.

The decode loop is correspondingly simple — just memcpy on the null stream
(synchronous with ORT's completed work) between steps:

```rust
for s in 0..MAX_SEQ {
    memcpy_htod_async(step_ptr, &[s as i64], null_stream)?;       // update step
    session.run_binding_with_options(&binding, &run_options)?;     // graph replay
    memcpy_dtod_async(input_ids_ptr, next_token_ptr, 8, null)?;   // feed token
    memcpy_dtoh_async(&mut token_buf, next_token_ptr, null)?;     // read for EOS
    if token_buf[0] == EOS { break; }
}
```

### 3. Image resize filter must match HuggingFace processor

**Symptom:** 3 out of 151 formulas produce `$$` (a single token meaning the
model failed to recognize the formula). All 3 are small images being upscaled
4x+ (33×19, 76×20, 339×43).

**Cause:** The Rust code used `FilterType::Lanczos3` for image resizing, but
HuggingFace's `Glm46VImageProcessorFast` uses `resample=3` which is
`PIL.Image.Resampling.BICUBIC`. For large images the difference is negligible,
but for small images being upscaled significantly, the interpolation method
produces meaningfully different pixel values:

```
p10_5 (33×19 → 112×84): max_diff=36/255, 11,931 differing pixels
p5_58 (339×43 → 336×56): max_diff=28/255, 14,001 differing pixels
```

These pixel differences propagate through the vision encoder and cause the model
to produce incorrect output for these specific images.

**Fix:** Change `FilterType::Lanczos3` to `FilterType::CatmullRom` (Rust
`image` crate's name for Bicubic interpolation):

```rust
let resized = image.resize_exact(target_w, target_h, FilterType::CatmullRom);
```

**Result:** All 3 previously-failing formulas now produce correct output. 9
remaining differences vs Ollama are all minor (`\mathbf` vs `\mathrm` for
single-variable formulas, `\mathrm{where}` vs `\text{where}`), caused by
inherent differences between Rust's `image` crate bicubic and PIL's bicubic
implementations. These same 9 differences exist between the Python ONNX pipeline
and Rust — they're not related to CUDA graphs or model changes.

### 4. Read model architecture from ONNX metadata

**Problem:** Hardcoded decoder architecture constants (`NUM_LAYERS`,
`NUM_KV_HEADS`, `HEAD_DIM`, `MAX_SEQ`) had to match the exported ONNX models.
When the model changed (e.g., re-exporting the decoder with `--max-seq 4096`),
these constants had to be manually updated in Rust code — error-prone and a
source of silent bugs (buffer overflows when MAX_SEQ is too small).

**Fix:** Read these values from the ONNX model's input metadata at init time.
The decoder's KV cache inputs encode the full architecture:

- `past_key_0` has shape `[1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM]`
- Counting `past_key_*` inputs gives `NUM_LAYERS`

```rust
struct DecoderParams {
    num_layers: usize,
    num_kv_heads: usize,
    max_seq: usize,
    head_dim: usize,
}

fn extract_decoder_params(session: &Session) -> Result<DecoderParams, ExtractError> {
    let num_layers = session.inputs().iter()
        .filter(|inp| inp.name().starts_with("past_key_"))
        .count();

    let past_key_0 = session.inputs().iter()
        .find(|inp| inp.name() == "past_key_0")
        .ok_or_else(|| ExtractError::Model("no past_key_0 input".into()))?;

    let shape = past_key_0.dtype().tensor_shape()
        .ok_or_else(|| ExtractError::Model("past_key_0 not a tensor".into()))?;
    let dims: Vec<i64> = shape.iter().copied().collect();

    Ok(DecoderParams {
        num_layers,
        num_kv_heads: dims[1] as usize,
        max_seq: dims[2] as usize,
        head_dim: dims[3] as usize,
    })
}
```

**What to extract from model vs keep hardcoded:**

| Extract from model | Keep hardcoded |
|---|---|
| `NUM_LAYERS` (KV input count) | `PATCH_SIZE`, `SPATIAL_MERGE` (image preprocessing) |
| `NUM_KV_HEADS` (KV shape dim 1) | `TARGET_SIZE` (encoder input size) |
| `MAX_SEQ` (KV shape dim 2) | `BOS_ID`, `EOS_ID`, `EOS_IDS` (tokenizer constants) |
| `HEAD_DIM` (KV shape dim 3) | `NORM_MEAN`, `NORM_STD` (image normalization) |
| | `BATCH_K` (decode sync strategy, not model-dependent) |

**Key insight:** `MAX_SEQ` is baked into the decoder ONNX model because CUDA
graph capture requires static tensor shapes. The KV cache dimension in the
model file is the source of truth — reading it at init time means re-exporting
the model with a different `--max-seq` value "just works" without any Rust
code changes.

### 5. IoBinding for LLM prefill — reverted (compute-bound, not transfer-bound)

**Context:** nsys profiling of the GLM-OCR CUDA path on a single Algorithm
region (~2000ms total) showed the prefill path as the apparent bottleneck:

- `cudaStreamSynchronize`: 42.4% (848ms) — ORT's internal syncs in `session.run()`
- `cudaMemcpy` (synchronous): 31.8% (636ms) — ORT pulling outputs to CPU
- H2D transfers: 98.6% of total transfer (7262 MB) — KV cache going CPU→GPU

**Root cause hypothesis:** `run_prefill_for_cuda()` uses `session.run()` which
pulls ALL outputs (including 32 KV cache tensors) to CPU. Then `decode_loop()`
copies them back to GPU via H2D. This GPU→CPU→GPU roundtrip was assumed to be
the dominant cost.

**What we tried (3 iterations):**

1. **IoBinding for LLM prefill only:** Rewrote `run_prefill_for_cuda` to use
   IoBinding with `bind_output_to_device("present_key/value_N", &cuda_mem)` to
   keep KV cache on GPU. Changed return type from `Vec<Vec<bf16>>` (CPU) to
   `Vec<Value>` (GPU-resident). Updated `decode_loop` to accept `&[Value]` and
   use D2D copies (`memcpy_dtod_async`) instead of H2D. Used `value_data_ptr()`
   (ORT C API `GetTensorMutableData`) to extract GPU addresses from the
   IoBinding output Values for head-by-head strided D2D copy.

2. **IoBinding for vision encoder + embedding:** Extended IoBinding to all three
   prefill-phase sessions. Vision encoder and embedding outputs stayed on GPU.
   Replaced CPU-side `merge_vision_embeddings` (extract → iterate → convert)
   with a single D2D copy overwriting image token positions in token_embeds
   (image tokens are contiguous, so one `memcpy_dtod_async`).

3. **Full GPU pipeline:** Combined all three into a single GPU-resident pipeline
   with manual CUDA stream synchronization between stages.

**nsys results (Algorithm region, single image):**

| Metric | Baseline (session.run) | IoBinding prefill only | Full GPU pipeline |
|--------|----------------------|----------------------|-------------------|
| Wall-clock median | 1937ms | 1938ms | 1919ms |
| H2D total | 7262 MB | 3593 MB | 3589 MB |
| D2D total | — | 848 MB | 852 MB |
| cudaMemcpy (sync) calls | 1394 | 697 | 697 |
| cuStreamSync calls | 2486 | 262 | 267 |
| cuGraphLaunch time | 733ms | 884ms | 636ms |

The IoBinding changes halved PCIe transfers and eliminated 89% of stream syncs,
but wall-clock time was unchanged. The pipeline is **compute-bound** on CUDA
graph decode steps (cuGraphLaunch), not transfer-bound. The KV cache round-trip
that looked expensive in the profile was overlapping with GPU compute — removing
it freed up PCIe bandwidth that wasn't the bottleneck.

**Additional issue:** The full GPU pipeline (iteration 3) crashed when running
multiple predictor groups sequentially (Text → DisplayFormula → Table). The
third group's decoder received a garbage `seqlens_k` value (0x3E1BA5A9 instead
of a valid sequence length), likely from leaked CUDA state across predictor
lifecycles due to `ManuallyDrop<Session>` (needed because `with_compute_stream`
causes session destructor crashes). This was not debugged further.

**Decision:** Reverted all changes. The complexity (IoBinding setup, raw C API
`value_data_ptr`, D2D strided copies, stream ordering, ManuallyDrop concerns)
was not justified by the negligible wall-clock improvement. The code stayed
with the simpler `session.run()` + H2D pattern.

**Lesson:** Profile metrics (transfer volume, sync counts) can be misleading
when operations overlap with compute. Always measure wall-clock time as the
primary metric. For this model, the ~636-884ms of CUDA graph decode compute
is the true floor — transfer optimizations yield diminishing returns once the
pipeline is compute-bound.

## Performance (vbd.pdf, 151 formulas)

| Version | Per-formula | Total | Notes |
|---------|-------------|-------|-------|
| Ollama (GLM-OCR via LLM server) | ~152ms | ~23s | Baseline |
| Rust, no CUDA graph | ~318ms | ~48s | Shared stream + async D2H |
| Rust, ORT built-in CUDA graph | **~148ms** | **~22s** | 2.2x vs no-graph |

The no-CUDA-graph version was slower than Ollama because each
`run_binding_with_options` call dispatches individual CUDA kernels with CPU
overhead per kernel. CUDA graph replay bundles all ~1551 ONNX ops into a single
`cudaGraphLaunch` call, eliminating that overhead.

## Accuracy (vbd.pdf, 151 formulas)

| Comparison | Differences | Nature |
|------------|-------------|--------|
| Rust vs Ollama | 9/151 | 7× `\mathbf`/`\mathrm`, 1× `\mathrm`/`\text`, 1× Rust more correct |
| Rust vs Python ONNX | 9/151 | Same 9 — all from bicubic implementation differences |
| Python ONNX vs Ollama | 0/151 | Identical (same PIL preprocessing) |

All 9 Rust-vs-reference differences are cosmetic (render identically in LaTeX).
One difference (p5_58: `\delta x_c` vs `\delta x_i`) is actually a case where
Rust is more accurate than Ollama — the source image clearly shows subscript `c`
in the numerator.
