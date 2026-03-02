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
