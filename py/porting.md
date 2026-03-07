# Porting GLM-OCR to Optimized ONNX

Lessons learned from porting GLM-OCR (`zai-org/GLM-OCR`) to an optimized ONNX pipeline.
GLM-OCR is a vision-language model (CogViT encoder + GLM decoder with GQA) for formula recognition.

## Export Architecture

GLM-OCR uses 4 ONNX models:

| Model | Export script | Size (BF16) | Purpose |
|-------|---------------|-------------|---------|
| `vision_encoder_mha.onnx` | `export.py` | ~844 MB | CogViT with M-RoPE + MHA fusion |
| `embedding.onnx` | `export.py` | ~174 MB | Token embeddings |
| `llm.onnx` | `export.py` | ~1.1 GB | Full LLM for prefill |
| `llm_decoder_gqa.onnx` | `export.py` | ~1.3 GB | Decode step with GQA (CUDA graphed) |

All models are exported in **BF16** directly from the native BF16 weights
(`zai-org/GLM-OCR`). The export traces in FP32 (BF16→FP32 upcast is lossless),
applies attention fusion surgery, then converts FP32→BF16 (lossless round-trip
to the original BF16 values). Ops without BF16 CUDA kernels (Conv, LayerNorm,
Einsum, Range) fall back to **FP32** (not FP16) to preserve full precision from
the native weights.

The export uses a thin `nn.Module` wrapper
(`DecoderStepWrapper`, `VisionEncoderWrapper`) around the HuggingFace model that
reshapes the interface for fixed-shape ONNX export. `torch.onnx.export()` traces
through the wrapper and all the HuggingFace model internals, flattening
everything into a single static ONNX graph with the trained weights embedded.

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

### ONNX Op Comparison (before/after surgery)

| Op | Before | After |
|----|--------|-------|
| GatherND | 1 | **0** |
| ScatterND | 0 (fixed by WhereStaticCache) | 0 |
| IsNaN | 16 | **0** |
| Where | 49 | 33 |
| Pow+ReduceMean+Sqrt+Reciprocal | 65 each (unfused LayerNorm) | 65 each |
| Total nodes | 1591 | 1551 |

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

### ONNX Attention Fusion: MHA and GQA

The raw ONNX models produced by `torch.onnx.export()` contain unfused attention — each self-attention layer traces to ~15 individual ops (MatMul, Reshape, Transpose, Softmax, etc.). ORT's `onnxruntime.transformers` library can fuse these into higher-level ops for better performance.

#### Three-Level Hierarchy

| Level | Op | Backends | What It Does |
|-------|-----|----------|-------------|
| 0 (raw) | ~15 ops/layer | All | Individual MatMul/Reshape/Transpose/Softmax. What `torch.onnx.export` produces. |
| 1 (MHA) | `com.microsoft.MultiHeadAttention` | CUDA, CPU, CoreML | Fused attention kernel. Single op replaces the entire Q/K/V projection + attention + output projection chain. |
| 2 (GQA) | `com.microsoft.GroupQueryAttention` | CUDA only | Fused attention with FlashAttention V2. O(1) memory scaling for attention — critical for long sequences. Supports GQA natively (different number of query vs KV heads). |

For GLM-OCR with 16 decoder layers, this reduces total node count from ~1551 (raw) to ~200-300 (fused).

#### How to Apply: optimize_model()

```python
from onnxruntime.transformers.optimizer import optimize_model

model = optimize_model(
    "llm.onnx",
    model_type="gpt2",      # triggers FusionRotaryAttention
    num_heads=16,            # query heads (GLM-OCR: 16 query, 8 KV)
    hidden_size=1536,
    opt_level=0,             # fusion only, no constant folding
)
model.save_model_to_file("llm_mha.onnx")
```

`model_type="gpt2"` tells the optimizer to look for decoder-only attention patterns. It runs `FusionRotaryAttention`, which pattern-matches the raw attention ops and replaces them with `MultiHeadAttention` nodes. `opt_level=0` means fusion only — no constant folding or graph restructuring that might break the model.

#### How to Apply: replace_mha_with_gqa()

```python
from onnxruntime.transformers.convert_generation import replace_mha_with_gqa

model = onnx.load("llm_mha.onnx")
replace_mha_with_gqa(model, "attention_mask", kv_num_heads=8)
onnx.save(model, "llm_gqa.onnx", ...)
```

This takes the MHA-fused model and upgrades `MultiHeadAttention` → `GroupQueryAttention`. The conversion:
- Adds `seqlens_k` input (cumulative KV sequence lengths, int32)
- Adds `total_sequence_length` input
- Builds a subgraph from `attention_mask` to compute these values
- Sets `kv_num_heads` attribute for GQA (8 KV heads vs 16 query heads for GLM-OCR)

#### M-RoPE Handling

GLM-OCR uses 3D M-RoPE (Multi-Resolution Rotary Position Embeddings) from Qwen2-VL. The standard MHA/GQA fused ops have a built-in `do_rotary` attribute for 1D rotary embeddings, but M-RoPE is 3D (temporal, height, width dimensions).

Two approaches:
1. **ORT absorbs RoPE** — If `FusionRotaryAttention` recognizes the RoPE pattern, it folds it into the fused node with `do_rotary=1`. This works for standard 1D RoPE but may not work for 3D M-RoPE.
2. **External RoPE** — Set `do_rotary=0` on the fused node and let the RoPE computation remain as explicit ops in the graph. The fused node handles everything except RoPE; RoPE is applied before the attention input.

Approach 2 is the safer bet for M-RoPE and is confirmed working by the `onnxruntime-genai` project's Qwen2.5-VL builder, which uses the same strategy.

#### When to Use Which Level

| Scenario | Level | Why |
|----------|-------|-----|
| CUDA inference (formulas) | GQA | FlashAttention V2, best throughput |
| CUDA inference (full-page PDF) | GQA | O(1) memory scaling critical for long sequences |
| CoreML / CPU inference | MHA | GQA is CUDA-only; MHA is the best available for other backends |
| Debugging / validation | Raw | Easier to inspect, matches PyTorch exactly |

#### Implementation

**Automatic fusion** (`optimize.py`) uses ORT's `optimize_model` and `replace_mha_with_gqa`. This works when the attention pattern matches ORT's expected layout (single fused QKV projection, standard RoPE). It failed for GLM-OCR because the traced HuggingFace Qwen2 attention has 3 separate Q/K/V MatMuls, inline M-RoPE, and WhereStaticCache — too different for FusionRotaryAttention to recognize.

**Manual graph surgery** (`optimize_gqa.py`) directly replaces raw attention ops with GQA nodes. Per layer, it removes ~20 nodes (WhereStaticCache KV update, repeat_kv expansion, scaled dot product attention, output reshape) and inserts 7 nodes (3 Transpose+Reshape pairs to convert BNSH→BSH, plus the GQA node). The RoPE computation is preserved externally with `do_rotary=0`. Reduces 1551 → 1336 nodes with 16 GQA nodes, numerically identical to the raw decoder.

```bash
# Full BF16 export (vision encoder MHA surgery is integrated):
python export.py --bf16

# Manual GQA surgery on decoder (CUDA + FlashAttention):
python optimize_gqa.py --model-dir ../model

# Manual MHA surgery on vision encoder (standalone, if needed):
python optimize_mha_vision.py --model-dir ../model

# Automatic MHA fusion (all backends — if pattern matches):
python optimize.py --model-dir ../model --target other
```

#### Vision encoder MHA surgery

The vision encoder (CogViT, 24 layers) also benefits from fused attention. ORT's automatic `optimize_model(model_type="vit")` fails (0 MHA nodes) because Q/K RMSNorm and 2D RoPE between projection and attention break pattern matching. SDPA re-export from PyTorch is not viable — SDPA decomposes back to the same matmul+softmax+matmul ops in ONNX anyway.

#### BFloat16 ONNX Type Constraints (ORT 1.24)

ORT has BF16 CUDA kernels for many ops (registered via `REGISTER_KERNEL_TYPED(BFloat16)` in the source), but the **ONNX op schema type constraints** — defined in the ONNX standard, not ORT — don't include `bfloat16` for all ops. ORT validates against the schema first, rejecting models before checking kernel availability.

| BF16 Support | Ops |
|---|---|
| **Supported** | Add, Mul, Div, Gemm, MatMul, Softmax, Sigmoid, Neg, Sqrt, Pow, Erf, Concat, Reshape, Transpose, Gather, Slice, GatherND, Unsqueeze |
| **Not supported** | Sin, Cos, Reciprocal, Squeeze, ReduceMean, Conv, LayerNormalization, Einsum, Range |

The unsupported ops include all components of unfused RMSNorm (Pow+ReduceMean+Sqrt+Reciprocal — though Pow and Sqrt are actually fine, ReduceMean and Reciprocal are not) and the RoPE trig ops (Sin, Cos). The vision encoder has 97 ReduceMean, 97 Reciprocal, 72 Squeeze, 2 Conv, and 1 each of Sin, Cos, Range, LayerNormalization — 272 ops total requiring FP32 fallback out of 1606 nodes.

The `_convert_fp32_to_bf16` function in `export.py` handles this by:
1. Converting most initializers FP32→BF16 (lossless round-trip from native BF16 weights)
2. Keeping initializers exclusively consumed by FP32 ops as FP32
3. Inserting Cast(BF16→FP32) before and Cast(FP32→BF16) after each FP32 op
4. Deduplicating shared Cast nodes when multiple FP32 ops share inputs

This uses FP32 (not FP16) for fallback ops to preserve full precision from the native BF16 weights. FP16 has a narrower exponent range (5 bits vs BF16's 8 bits), so FP16 would clip values outside its range.

The vision encoder is exported via `VisionEncoderWrapper`, which bypasses two tracing problems: (1) M-RoPE's data-dependent Python loops in `rot_pos_emb()` are replaced by taking pre-computed `pos_ids [N, 2]` and `max_grid_size` as inputs, and (2) `cu_seqlens`-based attention splitting is replaced by inline matmul attention (single-image only). The wrapper traces cleanly in FP32.

**Manual surgery** (`optimize_mha_vision.py`) replaces the attention core (MatMul Q@K^T → Mul scale → Softmax → MatMul attn@V → Squeeze → Transpose → Reshape) with a single `com.microsoft.MultiHeadAttention` node per layer. Q/K normalization and 2D RoPE ops stay outside the fused node. Adds Reshape nodes to convert Q/K/V from [seq, heads, dim] to [1, seq, hidden] format. Reduces ~1942 → ~1606 nodes with 24 MHA nodes.
