# Porting a PaddleOCR Model to Optimized ONNX

Lessons learned from porting PP-FormulaNet Plus-L (Vary-VIT-B encoder + mBART decoder) from PaddleOCR's native inference to a fully optimized split ONNX pipeline. Final result: **5.3x speedup** over PaddleOCR GPU (340ms → 64ms per formula), **93x** over PaddleOCR CPU.

## The Problem with Monolithic paddle2onnx Export

paddle2onnx exports the entire autoregressive model — encoder, decoder, and the decode loop — as a single ONNX graph. The decode loop becomes an ONNX `Loop` op containing the full decoder body. This is architecturally broken for GPU inference:

- **~410K node dispatches per formula** with constant CPU-GPU synchronization on every loop iteration
- **CUDA was 2x slower than CPU** (~2084ms vs ~1184ms) — a huge red flag that something is fundamentally wrong
- No way to use CUDA graphs (dynamic shapes from concat-based KV cache)
- No way to apply FP16 quantization cleanly (loop body is opaque)
- Increasing batch size made things *worse* (16-42% slower, L model OOM at batch 24)
- ORT graph optimization levels had no meaningful effect

The fact that CUDA was underperforming DirectML (and even CPU!) was the key signal. When your GPU is slower than your CPU on a model that's clearly GPU-bound, the problem isn't the hardware — it's the graph structure forcing serialization.

### Why ONNX Loop ops are fundamentally broken on GPU

This isn't a paddle2onnx bug — it's an architectural limitation of the ONNX `Loop` operator itself. The ORT CUDA Loop kernel delegates entirely to CPU loop logic. Every iteration forces a `cudaStreamSynchronize` because the loop condition (a bool) must be read on CPU to decide whether to continue. The ORT source code (`loop.cc`) says it plainly: *"the logic to run the subgraph must be on CPU either way."*

CUDA Graphs are **explicitly unsupported** with Loop/If/Scan ops (per ORT docs). This is a hard limitation — CUDA graphs capture a fixed kernel launch sequence, but Loop has a data-dependent iteration count.

This is a recognized problem with no fix on the horizon:
- [ORT #23154](https://github.com/microsoft/onnxruntime/issues/23154) documents the same GPU-slower-than-CPU pattern (their benchmark: CPU 2.9ms vs GPU 197ms — 70x slower). The ORT team's workaround is to *fall back Loop subgraphs to CPU*, not to fix GPU execution.
- [ONNX #7689](https://github.com/onnx/onnx/issues/7689) proposes dedicated fused operators for recurrent patterns, arguing that Loop/Scan are "inherently sequential" and "opaque to the runtime."

**Any ONNX model with a `Loop` op on the critical path is a red flag.** It means CPU-orchestrated sequential execution with GPU sync on every iteration — the worst possible pattern for GPU inference. The fix is always to split the model and run the loop in your own code.

## Why Rebuild from Scratch

We first tried **ONNX graph surgery** (`split_model.py`) to extract the Loop body as a standalone decoder. This gave a 1.94x speedup with IOBinding but the surgically-extracted models had fundamental problems:

- Complex graphs from inlined `If` nodes
- Concat-based KV cache (shapes grow each step, incompatible with CUDA graphs)
- Split encoder produced NaN on DirectML
- Not amenable to further optimization

The key insight: **doing your own export from scratch isn't that hard, and gives you full optimization control.** The actual work of recreating the model in PyTorch was straightforward — the architecture is well-documented, and the weight mapping between PaddlePaddle and PyTorch is mechanical (transpose Linear weights, copy everything else as-is). The entire encoder is ~350 lines of PyTorch, the decoder ~250 lines.

What you get in return for this effort:

- Clean ONNX graphs that work on all execution providers (CUDA, DirectML, CPU)
- Static-shape KV cache enabling CUDA graphs
- Full control over I/O types (FP16 native vs keep_io)
- Ability to do graph surgery (fuse argmax, pre-pad attention)
- Models you actually understand and can debug

## The Re-export Pipeline

```
PaddlePaddle checkpoint (.pdparams)
  → paddle.load() → state_dict
  → Filter & rename params (strip prefixes)
  → Transpose Linear weights (Paddle [in,out] → PyTorch [out,in])
  → Save as encoder_weights.npz / decoder_weights.npz
  → Load into PyTorch modules
  → torch.onnx.export() with static shapes
  → encoder.onnx + decoder.onnx
```

### Key Design Decisions

**Fixed-size KV cache.** The decoder uses pre-allocated `[B, 16, 512, 32]` buffers instead of concat-based growing tensors. New KV entries are written at position `step` via `torch.where(positions == step, new_kv, buffer)`. This makes all shapes static, enabling CUDA graphs and DirectML compatibility. We avoided `ScatterElements` (buggy on DirectML) in favor of `where`-based writes.

**Cross-attention always recomputed.** Instead of caching cross-attention KV (which requires branching logic), the decoder recomputes it from `encoder_hidden_states` every step. This eliminates control flow from the graph, making it fully tracer-friendly and CUDA-graph-compatible. The cross-attention computation is cheap relative to the self-attention over the growing sequence.

**Split encoder/decoder.** Any autoregressive model that bakes the decode loop into the ONNX graph (via `Loop` op) is a red flag. It means every loop iteration — token selection, stopping conditions, KV cache updates — runs as ONNX CPU ops with GPU sync points on every iteration. This is why CUDA was *slower* than CPU on the monolithic export: the GPU never gets to run freely.

The fix is to split encoder and decoder into separate ONNX models and run the decode loop in Python. The encoder runs once per image; the decoder runs once per token. This gives you:
- CUDA graphs on the decoder (captures all GPU work for a decode step, replays at ~1ms/step)
- No wasted encoder re-computation (encoder output is computed once, fed to every decode step)
- Independent optimization per model (different execution providers, precision, graph surgery)

## Optimization Steps (What Worked)

Each optimization builds on the previous. All benchmarks on 24 formula images (872 total tokens, avg 36 tokens/formula).

### 1. Split model + CUDA IOBinding (525ms → 189ms, 2.8x)

The baseline split. Encoder and decoder are separate ONNX sessions. IOBinding keeps KV cache on GPU between decode steps — only the token ID (8 bytes) crosses to CPU each step.

### 2. CUDA Graphs (189ms → 89ms, 5.9x)

CUDA graphs capture all GPU kernel launches on the first `run_with_iobinding()` call, then replay them with near-zero CPU overhead. Requirements:

- All input/output tensors at **fixed GPU addresses** between capture and replay
- All shapes **static** (our fixed-size KV cache enables this)
- Small per-step inputs (`input_ids`, `step`) updated via `cudaMemcpy` to persistent GPU addresses

```python
cudart = ctypes.CDLL("cudart64_12.dll")
# Pre-allocate persistent GPU buffers
ov_input_ids = ort.OrtValue.ortvalue_from_numpy(
    np.array([[0]], dtype=np.int64), "cuda", 0)
# Update in-place each step (8 bytes, same address)
cudart.cudaMemcpy(ov_input_ids.data_ptr(), cpu_array.ctypes.data, 8, H2D)
```

First decode step takes ~24ms (graph capture), then ~1ms/step for replay.

### 3. FP16 Conversion (89ms → 72ms, 7.3x)

Used `onnxruntime.transformers.float16.convert_float_to_float16()` with `keep_io_types=False` to convert both encoder and decoder to native FP16. Model sizes halved (381 MB → 191 MB, 347 MB → 174 MB).

We used `keep_io_types=False` to make I/O natively FP16, eliminating 34 boundary Cast nodes in the decoder. The alternative (`keep_io_types=True`) keeps external I/O as FP32 with automatic Cast nodes — works fine but adds overhead. Either way, **encoder and decoder must agree**: both `True` or both `False`. Preprocessing and runtime buffers (KV cache, encoder hidden states) must also match the chosen type.

### 4. GPU-side ArgMax (72ms → 71ms, 7.4x)

The standard decode loop copies 100KB of logits (50,000 × FP16) from GPU to CPU every step for `numpy.argmax`. This forces a full pipeline sync.

ONNX graph surgery adds ArgMax + Reshape nodes directly to the decoder graph:

```python
argmax_node = helper.make_node("ArgMax", inputs=["logits"],
    outputs=["next_token_2d"], axis=2, keepdims=0)
reshape_node = helper.make_node("Reshape",
    inputs=["next_token_2d", "argmax_shape"], outputs=["next_token"])
```

Now only 8 bytes (one int64 token ID) cross from GPU to CPU via `cudaMemcpy D2H`. The improvement was larger than raw profiling predicted (~11% vs expected ~3%) because `.numpy()` forces a full pipeline sync that stalls subsequent kernel launches.

### 5. Encoder Pre-padding (71ms → 64ms, 8.2x)

nsys profiling revealed `PadKernel` as the #1 GPU bottleneck at 26.6% of encoder time (12ms). The ViT uses `window_size=14` but the 48×48 patch grid isn't divisible by 14, so every window-attention block pads to 56×56 then slices back — 16 PadKernel launches per formula.

The fix: pre-pad the feature map to 56×56 once before the transformer blocks, so `window_partition` sees `56 % 14 == 0` and skips its internal padding entirely. Global attention blocks (which don't use windowing) still need the original 48×48 size, so we slice before and pad after each global block.

The critical detail: after LayerNorm, zero-valued padding positions become non-zero (LayerNorm bias). A mask (`pad_mask`) zeros out the padding region after norm1 in each window block, ensuring the attention computation sees the same zero-padded input as the original. This was verified to produce **token-for-token identical output** on all 24 test formulas (max encoder output diff: ~5e-6, well within FP32 noise).

```python
# Pre-pad once
pos = F.pad(self.pos_embed, (0, 0, 0, PAD_AMOUNT, 0, PAD_AMOUNT))
x = F.pad(x, (0, 0, 0, PAD_AMOUNT, 0, PAD_AMOUNT))  # [B, 56, 56, 768]
x = x + pos

for i, blk in enumerate(self.blocks):
    if blk.window_size == 0:
        # Global attention: slice to 48×48, run, pad back
        x = x[:, :GRID_SIZE, :GRID_SIZE, :].contiguous()
        x = blk(x)
        if i < DEPTH - 1:
            x = F.pad(x, (0, 0, 0, PAD_AMOUNT, 0, PAD_AMOUNT))
    else:
        # Window attention: inline with mask after norm1
        shortcut = x
        x = blk.norm1(x)
        x = x * self.pad_mask  # zero padding to match original
        x, pad_hw = window_partition(x, WINDOW_SIZE)
        x = blk.attn(x)
        x = window_unpartition(x, WINDOW_SIZE, pad_hw, (H, W))
        x = shortcut + x
        x = x + blk.mlp(blk.norm2(x))
```

Result: PadKernel dropped from 48 to 24 instances (16 → 8 per formula), saving ~5.8ms.

## Final Performance

| Optimization | Encoder | Decode | Total | vs PaddleOCR GPU |
|---|---|---|---|---|
| PaddleOCR native (CPU) | — | — | 5,933ms | 0.06x |
| PaddleOCR native (GPU, CUDA) | — | — | 340ms | 1.0x |
| DirectML monolithic (paddle2onnx) | — | — | 525ms | 0.6x |
| DirectML split, FP32 (session.run) | 53ms | 388ms | 441ms | 0.8x |
| CUDA IOBinding (split, FP32) | 37ms | 141ms | 178ms | 1.9x |
| CUDA Graph FP32 | — | — | 89ms | 3.8x |
| FP16 native | — | — | 72ms | 4.7x |
| + GPU ArgMax | 29ms | 41ms | 71ms | 4.8x |
| + Pre-pad window blocks | 25ms | 39ms | 64ms | 5.3x |

PaddleOCR's native GPU inference (340ms) uses Paddle's own CUDA-based static graph engine — no ONNX conversion. The monolithic paddle2onnx export (525ms) was actually **slower** than PaddleOCR native, confirming that the ONNX Loop op baked into that export is actively harmful. Our optimized ONNX pipeline only pulls ahead once we have CUDA IOBinding keeping KV on GPU (178ms, 1.9x).

## Profiling with NVIDIA Nsight Systems

nsys was essential for identifying the PadKernel bottleneck. ORT's built-in profiling measures operator dispatch time (host-side), not actual GPU kernel time. nsys captures the real GPU picture.

### Setup

```bash
# Profile with CUDA kernels + NVTX annotations
nsys profile --trace=cuda,nvtx --output=profile_output \
    python profile_script.py
# GPU kernel time breakdown
nsys stats --report cuda_gpu_kern_sum profile_output.nsys-rep
# NVTX range timing (encoder vs decoder)
nsys stats --report nvtx_sum profile_output.nsys-rep
```

### Adding NVTX Annotations

```python
import nvtx

with nvtx.annotate("encoder", color="blue"):
    enc_out = enc_sess.run(None, {enc_sess.get_inputs()[0].name: img})

with nvtx.annotate("decoder_loop", color="red"):
    for s in range(max_steps):
        # ... decode step
```

### ORT Built-in Profiling

Useful for operator-level breakdown (which ONNX ops take time), though it measures dispatch time not GPU kernel time:

```python
opts = ort.SessionOptions()
opts.enable_profiling = True
session = ort.InferenceSession(model_path, sess_options=opts)
# ... run inference
profile_file = session.end_profiling()
# Parse the JSON for operator breakdown
```

### What to Look For

- **Kernel with disproportionate instance count**: PadKernel at 48 instances (vs 24 for GEMM) suggested per-block overhead
- **Kernel time vs transfer time**: GPU argmax saved more than expected because `.numpy()` forces a pipeline sync, not just a data copy
- **First-step outliers**: CUDA graph capture makes the first decode step ~24ms vs ~1ms for replay — always exclude warmup from measurements

## Things That Didn't Work

### Encoder Batching

Encoding multiple images in one pass (batch sizes 2-8) gave only 15-19% encoder speedup but massively increased VRAM. At batch 12+, VRAM exhaustion caused encoder time to spike from 27ms to 1780ms and poisoned GPU memory for all subsequent operations in the same process. Full pipeline improvement was only 7% (74ms → 69ms at batch 4). Not worth the VRAM cost or complexity.

### Pure Pre-padding (Without Mask)

Simply pre-padding the feature map to 56×56 without zeroing the padding after norm1 produced garbage output — 22 out of 24 formulas hit max tokens (513) with nonsensical LaTeX (Chinese characters, infinite repetition). The issue: LayerNorm converts zero padding into non-zero values (the LN bias), which corrupts attention in boundary windows. The global attention blocks (operating on 56×56 instead of 48×48) amplified this corruption. The fix was the mask-based hybrid approach (slice to 48×48 for global blocks + mask padding after norm1 for window blocks).

### Hybrid Pre-padding Without Mask

Slicing to 48×48 for global blocks but not masking after norm1 in window blocks produced plausible-looking but different output (different token counts, ~66 avg vs ~36 original). The LaTeX looked correct but wasn't token-for-token equivalent. The norm1 bias values in the padding region changed attention weights in boundary windows just enough to shift decoder behavior. Adding the mask after norm1 restored exact equivalence.

### DirectML Limitations

DirectML was our starting point (the monolithic paddle2onnx model ran on it), and it works for basic inference, but it has fundamental limitations that prevent the optimizations we applied on CUDA:

**IOBinding with pre-allocated device buffers crashes.** Allocating `OrtValue` on `"dml"` and binding them as KV cache inputs/outputs causes TDR (Timeout Detection and Recovery) — Windows kills the GPU process with `DXGI_ERROR_DEVICE_HUNG` (`887A0005`). This happens because DirectML runs on D3D12, which submits work as command lists subject to a ~2 second Windows timeout. With 16 KV cache tensors + encoder hidden states, the command list exceeds TDR. CUDA has no such timeout mechanism. This is a [recognized limitation](https://github.com/microsoft/onnxruntime/issues/26821) — ORT's D3D12 resource interop API is acknowledged as insufficient for advanced use cases.

**No CUDA graphs equivalent.** DML has `ep.dml.enable_graph_capture` but it's much less mature and doesn't support the IOBinding patterns needed for autoregressive decoding with KV cache.

**Splitting the model barely helps without IOBinding.** DirectML split with `session.run` (441ms) was only 1.2x faster than DirectML monolithic (525ms). Without IOBinding, every decode step copies all 16 KV tensors (~16MB FP32) through CPU, erasing most of the benefit of splitting. On CUDA, IOBinding keeps KV on GPU, making the split 2.9x faster (178ms).

**The surgically-extracted encoder produced NaN.** When we tried ONNX graph surgery to split the monolithic model, the extracted encoder worked on CUDA but produced NaN on DirectML. The re-exported encoder (from PyTorch) works fine on both.

In short: DirectML is viable for simple single-shot inference, but any optimization involving persistent GPU buffers (IOBinding, CUDA graphs, in-place KV cache) requires CUDA.

### TensorRT

TRT EP was tested but underperformed CUDA Graph FP16: 95-104ms (TRT FP16/FP32) vs 75ms (CUDA Graph FP16). TRT's optimization overhead and kernel selection didn't beat ORT's CUDA graphs for this particular model shape.

---

## GLM-OCR: Second Model Port

GLM-OCR (`zai-org/GLM-OCR`) is a higher-quality formula recognition model based
on a vision-language architecture (CogViT encoder + GLM decoder with GQA). It
correctly handles all 151 test formulas including the 6 that PP-FormulaNet gets
wrong.

### Export Architecture

GLM-OCR requires 4 ONNX models (vs PP-FormulaNet's 2):

| Model | Export script | Size | Purpose |
|-------|---------------|------|---------|
| `vision_encoder.onnx` | `export.py` | ~3.4 GB (FP32) | CogViT with M-RoPE |
| `embedding.onnx` | `export.py` | 348 MB (FP16) | Token embeddings |
| `llm.onnx` | `export.py` | 2.2 GB (FP16) | Full LLM for prefill |
| `llm_decoder.onnx` | `export_decoder.py` | 1.3 GB (FP16) | Decode step (CUDA graphed) |

The same export wrapper pattern applies: a thin `nn.Module` wrapper
(`DecoderStepWrapper`) around the HuggingFace model that reshapes the interface
for fixed-shape ONNX export. `torch.onnx.export()` traces through the wrapper
and all the HuggingFace model internals, flattening everything into a single
static ONNX graph with the trained weights embedded.

### ONNX Ops That Break CUDA Graphs

Two categories of problematic ops were found and eliminated from the decoder:

**ScatterND (from `index_copy_`):** HuggingFace's `StaticCache` uses
`index_copy_` to write new KV entries at the current position. This traces to
ONNX `ScatterND`, which is incompatible with CUDA graphs for the same reason as
GatherND (indirect addressing from temporaries).

*Fix:* `WhereStaticCache` replaces `index_copy_` with `torch.where()`:

```python
# ScatterND path (original):
self.keys.index_copy_(2, cache_position, key_states)

# Where path (CUDA-graph compatible):
mask = (positions == cache_position).view(1, 1, -1, 1)
self.keys = torch.where(mask, key_states.expand_as(self.keys), self.keys)
```

**GatherND (from causal mask expansion):** When a 2D `attention_mask [1, MAX_SEQ]`
is passed to HuggingFace's `GlmModel.forward()`, the internal `create_causal_mask()`
function expands it into a 4D mask using index arithmetic that traces to ONNX
`GatherND`. The GatherND reads from a dynamically-computed boolean tensor that ORT
places in an internal temporary buffer — the buffer address changes between CUDA
graph capture and replay, causing `ILLEGAL_ADDRESS`.

This was diagnosed using `compute-sanitizer --tool memcheck`:

```
Invalid __global__ read of size 1 bytes
  at void onnxruntime::cuda::_GatherNDKernel<bool>(...)
  Address 0xf859cd4bed00 is out of bounds
  Host Frame: cudaGraphLaunch
```

*Fix:* Pre-compute the 4D causal mask in our wrapper before calling into HuggingFace:

```python
# Before: 2D mask → HuggingFace expands internally → GatherND in ONNX
attention_mask = (positions < (cache_pos + 1).unsqueeze(-1)).long()  # [1, MAX_SEQ]

# After: 4D mask → HuggingFace returns as-is → no GatherND
min_dtype = torch.finfo(inputs_embeds.dtype).min
attend = positions < (cache_pos + 1)
attention_mask = torch.where(
    attend.view(1, 1, 1, -1),
    torch.tensor(0.0, dtype=inputs_embeds.dtype),
    torch.tensor(min_dtype, dtype=inputs_embeds.dtype),
)  # [1, 1, 1, MAX_SEQ]
```

This works because `create_causal_mask()` has an early-exit: if the mask is already
4D, it's returned as-is, bypassing all the internal expansion code. The mask
content is identical — `0.0` for attended positions, `-65504` (fp16 min) for masked.

**Key insight:** When using `torch.onnx.export()` on a wrapper that calls into
library code (HuggingFace transformers), the ONNX tracer captures everything —
your code AND the library internals. Problematic ops can come from deep inside the
library's forward pass. The fix is to change what you pass to the library to steer
the trace down a different code path, not to modify the library itself.

### ONNX Op Comparison

| Op | PP-FormulaNet decoder | GLM-OCR decoder (before) | GLM-OCR decoder (after) |
|----|----------------------|--------------------------|-------------------------|
| GatherND | 0 | 1 | **0** |
| ScatterND | 0 | 0 (fixed by WhereStaticCache) | 0 |
| IsNaN | 0 | 16 | **0** |
| Where | 24 | 49 | 33 |
| LayerNormalization | 26 (fused) | 0 | 0 |
| Pow+ReduceMean+Sqrt+Reciprocal | 0 | 65 each (unfused LayerNorm) | 65 each |
| Total nodes | 1278 | 1591 | 1551 |

### Image Preprocessing: Resize Filter Matters

The HuggingFace image processor uses `resample=3` (PIL Bicubic). Using a
different interpolation method (e.g., Lanczos) causes visible failures on small
images being upscaled 4x+. For p10_5 (33×19 → 112×84), Lanczos vs Bicubic
produces max pixel differences of 36/255 across ~12K pixels — enough to cause
the model to output `$$` instead of recognizing the formula.

The Rust `image` crate's `FilterType::CatmullRom` is the equivalent of PIL's
Bicubic. Even with matching filter types, minor implementation differences
between Rust's bicubic and PIL's bicubic cause 9/151 formulas to differ
slightly (`\mathbf` vs `\mathrm` for single-variable formulas). These are
cosmetic — both render identically in LaTeX.

### Performance (vbd.pdf, 151 formulas)

| Version | Per-formula | Total | Speedup |
|---------|-------------|-------|---------|
| Ollama (GLM-OCR via LLM server) | ~152ms | ~23s | 1.0x |
| Rust, no CUDA graph | ~318ms | ~48s | 0.5x |
| Rust, ORT built-in CUDA graph | **~148ms** | **~22s** | **1.5x** |

CUDA graphs provide a 2.2x speedup for GLM-OCR (48s → 22s). Without CUDA
graphs, each `run_binding` call dispatches ~1551 individual CUDA kernels with
CPU overhead per kernel. Graph replay bundles all ops into a single
`cudaGraphLaunch` call.

### Accuracy (vbd.pdf, 151 formulas)

| Comparison | Differences | Nature |
|------------|-------------|--------|
| Rust vs Ollama | 9/151 | Cosmetic: `\mathbf`/`\mathrm`, `\mathrm`/`\text` |
| Rust vs Python ONNX | 9/151 | Same 9 — bicubic implementation differences |
| Python ONNX vs Ollama | 0/151 | Identical (same PIL preprocessing) |
