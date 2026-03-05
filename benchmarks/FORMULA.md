# Formula Model Benchmarks

**Test document:** VBD paper (16 pages, 575 regions, 151 formulas)
**Hardware:** Windows 11, NVIDIA GPU (CUDA + DirectML)
**Date:** 2026-02-28 (oar-ocr monolithic), 2026-03-01 (custom split model)

## Base Case (batch=8, ORT default Level1)

| Quality | Model | Size | Time (s) | vs Low |
|---------|-------|------|----------|--------|
| Low | PP-FormulaNet_plus-S | 221 MB | 23.3 | 1.0x |
| Med | PP-FormulaNet_plus-M | 565 MB | 115.8 | 5.0x |
| High | PP-FormulaNet_plus-L | 700 MB | 143.6 | 6.2x |

## Batch Size 32 (ORT default Level1)

| Quality | Model | Size | Time (s) | vs Low | vs Base |
|---------|-------|------|----------|--------|---------|
| Low | PP-FormulaNet_plus-S | 221 MB | 27.0 | 1.0x | 1.16x slower |
| Med | PP-FormulaNet_plus-M | 565 MB | 163.9 | 6.1x | 1.42x slower |
| High | PP-FormulaNet_plus-L | 700 MB | OOM | - | - |

Increasing batch size from 8 to 32 made things **worse** across the board. The S model was 16% slower, the M model 42% slower, and the L model ran out of GPU memory entirely (failed at batch_size=24). The autoregressive decoder in these monolithic ONNX models doesn't benefit from larger batches -- the overhead of larger tensor allocations dominates.

## ORT Full Optimization (batch=8, OrtGraphOptimizationLevel::All)

| Quality | Model | Size | Time (s) | vs Low | vs Base |
|---------|-------|------|----------|--------|---------|
| Low | PP-FormulaNet_plus-S | 221 MB | 22.2 | 1.0x | 1.05x faster |
| Med | PP-FormulaNet_plus-M | 565 MB | 115.4 | 5.2x | ~same |
| High | PP-FormulaNet_plus-L | 700 MB | 143.6 | 6.5x | ~same |

Full ORT graph optimization (Level3/All vs default Level1) had negligible impact. The S model showed a marginal 5% improvement; M and L models were unchanged. This suggests DirectML already applies its own graph optimizations, and the ORT-level optimizations (constant folding, node fusion, etc.) are either already covered or don't apply to the autoregressive decoder structure.

## Custom Split Encoder/Decoder (CUDA EP, IoBinding, FP16)

Replaced the monolithic oar-ocr FormulaRecognitionPredictor with a custom Rust implementation
using split encoder/decoder FP16 ONNX models + IoBinding. Encoder output stays on GPU, decoder
runs an autoregressive loop with KV cache on GPU. Only next_token (8 bytes) crosses GPU→CPU per step.

| Approach | Model | Size | Total (s) | Formula (s) | ms/formula | vs S mono |
|----------|-------|------|-----------|-------------|------------|-----------|
| oar-ocr mono S (baseline) | PP-FormulaNet_plus-S | 221 MB | 23.3 | ~20.5 | ~136 | 1.0x |
| Custom split (CUDA, IoBinding) | encoder_fp16 + decoder_fp16_argmax | 365 MB | 44.4 | 41.6 | 275 | ~2.0x slower |

Per-page breakdown (custom split model):

| Pages with N formulas | N formulas | Total (ms) | ms/formula |
|-----------------------|------------|------------|------------|
| Page 3 | 24 | 7706 | 321 |
| Page 4 | 23 | 9987 | 434 |
| Page 5 | 44 | 13614 | 309 |
| Page 6 | 14 | 3945 | 282 |
| Pages 7-8 | 5 + 4 | 1348 | 150 |
| Pages 10-11 | 2 + 29 | 4346 | 140 |
| Pages 14-16 | 3 + 3 | 637 | 106 |

**Key observations:**
- Warmup: first inference ~450ms (CUDA kernel compilation), subsequent ~60ms (cached)
- Average 275ms/formula, but varies widely: 106-434ms depending on LaTeX sequence length
- Pages with many formulas show higher per-formula times (suggests memory pressure or thermal)
- Short formulas (few decoder steps) approach ~80-100ms, long formulas (many steps) reach 400ms+
- Non-formula work (layout + text + tables + images): ~2.8s for 16 pages

**Why slower than Python reference (68ms/formula):**
The Python runner from `py/pp-formulanet/cuda/run.py` achieves ~68ms/formula using:
1. **CUDA graphs** — captures the decoder loop pattern and replays (eliminates per-step overhead)
2. **Pre-allocated GPU buffers** at fixed addresses (no allocation per step)
3. **Direct cudaMemcpy** for tiny 8-byte H2D/D2H transfers (no ORT binding overhead)

Our Rust implementation uses IoBinding per step but does NOT use CUDA graphs or direct cudaMemcpy.
Each decoder step creates a new IoBinding, binds all 18+ inputs/outputs, and extracts values through ORT.
This per-step overhead dominates when formulas require 50-200+ decoder steps.

**Next steps for performance:**
- Add CUDA graph support (requires `libloading` FFI to `cudart64` for `cudaMemcpy`)
- Pre-allocate fixed GPU buffers via `ort::Allocator` instead of recreating each step
- Profile to confirm per-step binding overhead is the bottleneck

## GLM-OCR: GQA Decoder + MHA Vision Encoder (2026-03-03)

Replaced raw attention ops with fused operators via manual ONNX graph surgery:
- **Decoder**: `optimize_gqa.py` — 16 GQA (GroupQueryAttention) nodes replace ~23 nodes/layer (WhereStaticCache, repeat_kv, scaled dot product attention). Enables FlashAttention V2. 1551 → 1336 nodes.
- **Vision encoder**: `optimize_mha_vision.py` — 24 MHA (MultiHeadAttention) nodes replace ~25 nodes/layer (matmul+softmax+matmul attention core). Q/K RMSNorm and 2D RoPE stay outside fused node. 3663 → 3183 nodes.

ORT's automatic `optimize_model` failed on both (0 fused nodes) due to non-standard attention patterns.

| Approach | Decoder | Vision Encoder | Median (ms/formula) | vs baseline |
|----------|---------|----------------|---------------------|-------------|
| Baseline (raw attention) | llm_decoder.onnx | vision_encoder.onnx | 136.1 | 1.0x |
| GQA + MHA surgery | llm_decoder_gqa.onnx | vision_encoder_mha.onnx | 136.9 | ~same |

**No speedup on formula images.** Formula images have few patches (~100-300) and short decoder sequences (~10-50 tokens), so:
- Vision encoder attention is not the bottleneck (O(n²) doesn't matter when n is small)
- FlashAttention's constant-factor advantage in the decoder is negligible for short sequences
- The bottleneck remains the decode loop: 16 KV cache copies + ORT graph replay per step, dominated by kernel launch overhead

**Quality:** 141/151 formulas (93.4%) match ollama reference exactly. The 10 differences are minor (`\mathrm` vs `\mathbf` for ambiguous glyphs, `\text` vs `\mathrm` for text commands, one tiny formula producing `$$` instead of `G.`).

**Where GQA/MHA will matter:** Full-page PDF OCR, where the vision encoder processes thousands of patches (O(n²) attention becomes the bottleneck) and the decoder generates hundreds of tokens per page.

## Conclusions (Monolithic Models)

- **Batch size 8 is optimal** for monolithic models on DirectML. Larger batches hurt performance and risk OOM.
- **ORT graph optimization level has no meaningful effect** when using DirectML as the execution provider.
- The performance gap between models is dominated by model size: M is ~5x slower than S, L is ~6x slower than S.
- The bottleneck is the autoregressive decoder baked into the ONNX graph -- no Rust-side optimization can change the number of sequential decode steps.
