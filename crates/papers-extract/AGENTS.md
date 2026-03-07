# papers-extract — Local PDF Extraction Pipeline

## Overview

Local, pure-Rust PDF processing pipeline using ONNX models and `pdfium-render`.
Converts PDFs into structured JSON + Markdown + extracted images. Uses PP-DocLayoutV3 for
25-class layout detection with built-in reading order via direct `ort` inference.
Formula recognition uses GLM-OCR (vision-language model with GQA-fused decoder).
Table recognition uses TableFormer or GLM-OCR.

## Architecture

```
PDF Input
  │
  [1] Load PDF ─── pdfium-render
  │     Bind pdfium, load document, get page dimensions
  │
  [2] Per-page processing:
  │     ├── Render page at 144 DPI (pdfium → image::DynamicImage)
  │     ├── Extract characters with bboxes (pdfium text layer)
  │     ├── Layout detection (PP-DocLayoutV3 via direct ort — LayoutDetector)
  │     │     → 25 region classes + reading order (model output column 6)
  │     ├── For DisplayFormula regions:
  │     │     crop image → FormulaPredictor.predict() → separate $$latex$$ regions
  │     ├── For InlineFormula regions:
  │     │     ├── [FAST] Char-based bypass: extract PDF chars in bbox → try_extract_inline_formula()
  │     │     │     If all chars map to known LaTeX tokens → use directly, skip ML OCR
  │     │     └── [FALLBACK] crop image → FormulaPredictor.predict() → only for formulas bypass can't handle
  │     │     Result merged into parent text as $latex$ (chars under formula bbox excluded)
  │     ├── For Table regions:
  │     │     crop image → TableStructureRecognitionPredictor.predict() → HTML tokens
  │     ├── Text extraction (match pdfium chars to detected regions, Y-axis converted,
  │     │     inline formula LaTeX spliced at spatial position)
  │     ├── Figure/chart crop (from rendered image)
  │     └── Caption association (proximity-based)
  │
  [3] Assembly ─── pure Rust
        Write JSON + Markdown + images to output directory
```

## Module Layout

| Module | Purpose |
|--------|---------|
| `lib.rs` | Public API: `extract()`, `Pipeline`, `ExtractOptions`, `Quality` |
| `types.rs` | `ExtractionResult`, `Page`, `Region`, `RegionKind` (24 variants), `Metadata` |
| `error.rs` | `ExtractError` enum |
| `layout.rs` | `LayoutDetector` — direct ONNX inference on PP-DocLayoutV3, `DetectedRegion` |
| `glm_ocr/` | `GlmOcrPredictor` — GLM-OCR with CUDA (IoBinding + CUDA graphs + GQA), CoreML, and CPU backends |
| `pipeline.rs` | `Pipeline` struct — owns pdfium + LayoutDetector + GlmOcrPredictor + TableFormerPredictor, orchestrates per-page processing |
| `models.rs` | Model download from GitHub releases, predictor + LayoutDetector builders, execution provider config |
| `pdf.rs` | `PdfChar`, `load_pdfium()`, `render_page()`, `extract_page_chars()` |
| `text.rs` | Match pdfium characters to layout regions, reconstruct text with word/paragraph detection. Splices inline formula LaTeX (`$...$`) at correct spatial positions, excluding pdfium chars under formula bboxes. Converts PdfChar Y-up coords to image Y-down space. Also provides `try_extract_inline_formula()` for char-based bypass of simple inline formulas. |
| `figure.rs` | Crop visual regions, associate captions by proximity |
| `reading_order.rs` | XY-Cut fallback algorithm for reading order |
| `output.rs` | Write JSON, Markdown, and cropped images to disk |

## Key Types

### RegionKind (24 classes)

```
Title, ParagraphTitle, Text, VerticalText, PageNumber,
Abstract, TOC, References, Footnote, PageHeader, PageFooter,
Algorithm, DisplayFormula, InlineFormula, FormulaNumber,
Image, Table,
FigureTableTitle, FigureTitle, TableTitle, ChartTitle,
Seal, Chart, SidebarText
```

Content routing by kind:
- **Text-bearing** → `text` field (pdfium char extraction + inline formula splicing)
- **Table** → `html` field (oar-ocr SLANet-Plus)
- **DisplayFormula** → `latex` field (GLM-OCR), rendered as `$$...$$`
- **InlineFormula** → char-based bypass if all chars are known LaTeX tokens; otherwise ML OCR. Merged into parent text region as `$...$`; orphans emitted as standalone `$...$` regions
- **Visual** (Image/Chart/Seal) → `image_path` field (cropped PNG)
- **Caption** → `text` field + associated with parent via `caption`

### Quality Modes

- **Fast** (default): SLANet-Plus (7 MB) — fast table recognition
- **Quality**: PP-LCNet classifier (6.5 MB) + SLANeXt-wired (351 MB) — better accuracy for complex tables

Formula recognition uses GLM-OCR (~3.3 GB total ONNX models). Simple inline formulas (single variables, Greek letters, basic sub/superscripts) are bypassed via char-based extraction from the PDF text layer, avoiding ML inference.

## Inline Formula Char-Based Bypass

**Purpose:** Skip expensive ML OCR (50-200ms GPU per formula) for simple inline formulas resolvable from the PDF text layer.

**How it works:** For each `InlineFormula` region detected by layout analysis, extract pdfium characters within its bbox, validate all map to known LaTeX tokens, fold sub/superscripts via `detect_scripts()`, assemble LaTeX. Returns `Option<String>` — `None` falls through to ML OCR unchanged.

**Supported patterns:** Single variables, Greek letters (α→`\alpha`, etc.), basic sub/superscripts (`x_{t}`, `x^{2}`), simple operators (+, -, =, <, >, ≤, ≥, ≠, ≈, ≡, ±, ×, ÷, ·, ∞), delimiters (`()[],./ |`), common math symbols (∂, ∇, ∈, ∉, ∩, ∪, ⊆, ⊇, ′).

**Rejection criteria (→ OCR fallback):**
- Any unknown character (PUA codepoints, combining chars, vector arrows, etc.)
- >20 visible characters
- 0 matched characters
- Vertical span > 3× tallest character (multi-line)

**Safety guarantee:** Conservative all-or-nothing — if any single character is unknown, the entire formula is rejected to OCR. The only risk is false negatives (sending simple formulas to OCR = wasted compute, not wrong output).

**Key functions in `text.rs`:**
- `try_extract_inline_formula()` — main entry point, called from `pipeline.rs`
- `char_to_latex()` — maps Unicode codepoints to LaTeX commands
- `replace_greek_in_latex()` — post-processes raw Unicode in detect_scripts output
- `is_known_formula_char()` / `get_latex_for_char()` — validation and conversion helpers

**Performance:** Negligible CPU cost (<1ms per formula) vs 50-200ms GPU per ML OCR call. ~70% of inline formulas bypassed based on analysis of academic papers.

## Dependencies

- `oar-ocr` 0.6 — TableStructureRecognitionPredictor for crop-based table recognition
- `pdfium-render` 0.8 — PDF loading, rendering, text extraction (requires pdfium binary)
- `ort` 2.0.0-rc.11 — Direct ONNX inference for layout detection + formula recognition; CUDA (Windows) / CoreML (macOS) / CPU execution providers
- `ndarray` 0.17 — Tensor construction for ort model inputs/outputs
- `half` 2 — FP16 type for formula model tensors
- `tokenizers` 0.21 — HuggingFace tokenizer for formula token decoding
- `image` 0.25 — Image processing
- `reqwest` (blocking) — Model download from GitHub releases
- `cudarc` 0.16 (Windows) — Low-level CUDA driver API for memcpy (H2D, D2D, D2H) and memset in formula decoder

## Model Management

Layout model auto-downloads from `github.com/GreatV/oar-ocr/releases` on first use.
GLM-OCR models must be pre-exported via `py/glm-ocr/cuda/export.py` and placed in the model cache directory.
Cache directory: `{dirs::cache_dir()}/papers/models/` (override: `PAPERS_MODEL_DIR` env var).

## Execution Providers

Three backends with auto-detection order CUDA → CoreML → CPU:

- **CUDA** (Windows, NVIDIA 3000+ GPU): BF16 precision, IoBinding + CUDA graphs + GQA-fused decoder. Uses `llm_decoder_gqa.onnx` for fast decode.
- **CoreML** (macOS, Apple Silicon): FP32 precision, `session.run()` decode with growing KV cache. Uses `llm.onnx` for both prefill and decode.
- **CPU** (everywhere): FP32 precision, same `session.run()` decode as CoreML. Universal fallback.

CLI: `--backend cuda|coreml|cpu|auto` (default: `auto`)

## Testing

Unit tests use synthetic data — no models or pdfium required.
Integration tests (manual): need pdfium binary + downloaded models + formula ONNX models.
