# GLM-OCR Benchmark

## Setup

- **Paper**: VBD (Vertex Block Descent) — 16 pages, 577 total regions
- **Dump**: `data/dumps/vbd` (pre-extracted via `dump` binary)
- **GPU**: NVIDIA GPU with CUDA EP, ORT with CUDA graphs enabled on decoder
- **Model**: GLM-OCR 4-part ONNX split (vision_encoder + embedding + llm + llm_decoder_gqa)
- **Runs**: 2 per region (benchmark mode)
- **Warmup**: 1 prediction before timed runs

## Command

```bash
cargo run --release --bin run_glm_ocr -- data/dumps/vbd \
  -o .temp/results/vbd-glm \
  --region-type "Algorithm,Text,DisplayFormula,InlineFormula,Table" \
  --bench
```

## Results

302 regions across 5 types. Total wall time: 86.2s.

| Kind           | Count | Median | Min   | Max    | StdDev | Total |
|----------------|------:|-------:|------:|-------:|-------:|------:|
| Algorithm      |     1 | 2310ms | 2310ms| 2310ms |  0.0ms |  2.3s |
| DisplayFormula |    23 |  282ms |  117ms|  685ms |170.6ms |  7.9s |
| InlineFormula  |   131 |   86ms |   49ms|  236ms | 37.5ms | 13.1s |
| Table          |     1 | 7419ms | 7419ms| 7419ms |  0.0ms |  7.4s |
| Text           |   146 |  303ms |   52ms| 1861ms |273.7ms | 55.4s |
| **Total**      | **302** | | | | | **86.2s** |

## Observations

- **Text dominates wall time** — 146 regions account for 55.4s (64%) of total time despite a modest 303ms median, due to sheer count and high variance (52ms–1.9s).
- **InlineFormula** is the fastest per-region at 86ms median, but 131 regions still add up to 13.1s.
- **DisplayFormula** at 282ms median is ~3.3x slower than inline — larger, more complex expressions.
- **Table** is the slowest per-region (7.4s) — structured HTML output with many tokens.
- **Algorithm** is expensive per-region (2.3s) — pseudocode blocks produce long token sequences.
- Latency scales linearly with output token count (autoregressive decoder). Short outputs (inline formulas, short text) are fast; long structured outputs (tables, algorithms) are slow.
