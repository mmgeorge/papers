# papers-extract — Local PDF Extraction Pipeline

## Overview

Local, pure-Rust PDF processing pipeline using ONNX models and `pdfium-render`.
Converts PDFs into structured JSON + Markdown + extracted images. Uses PP-DocLayoutV3 for
25-class layout detection with built-in reading order via direct `ort` inference.
Formula recognition uses custom split encoder/decoder ONNX models with CUDA EP.
Table recognition uses `oar-ocr` (SLANet).

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
  │     │     crop image → FormulaPredictor.predict() → merged into parent text as $latex$
  │     │     (chars under inline formula bbox excluded from text extraction)
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
| `formula.rs` | `FormulaPredictor` — custom CUDA formula predictor using split encoder/decoder FP16 ONNX models with IoBinding |
| `pipeline.rs` | `Pipeline` struct — owns pdfium + LayoutDetector + FormulaPredictor + TableStructureRecognitionPredictor, orchestrates per-page processing |
| `models.rs` | Model download from GitHub releases, predictor + LayoutDetector builders, execution provider config |
| `pdf.rs` | `PdfChar`, `load_pdfium()`, `render_page()`, `extract_page_chars()` |
| `text.rs` | Match pdfium characters to layout regions, reconstruct text with word/paragraph detection. Splices inline formula LaTeX (`$...$`) at correct spatial positions, excluding pdfium chars under formula bboxes. Converts PdfChar Y-up coords to image Y-down space. |
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
- **DisplayFormula** → `latex` field (custom FormulaPredictor), rendered as `$$...$$`
- **InlineFormula** → merged into parent text region as `$...$`; orphans emitted as standalone `$...$` regions
- **Visual** (Image/Chart/Seal) → `image_path` field (cropped PNG)
- **Caption** → `text` field + associated with parent via `caption`

### Quality Modes

- **Fast** (default): SLANet-Plus (7 MB) — fast table recognition
- **Quality**: PP-LCNet classifier (6.5 MB) + SLANeXt-wired (351 MB) — better accuracy for complex tables

Formula recognition always uses the custom split encoder/decoder models (~365 MB total).

## Dependencies

- `oar-ocr` 0.6 — TableStructureRecognitionPredictor for crop-based table recognition
- `pdfium-render` 0.8 — PDF loading, rendering, text extraction (requires pdfium binary)
- `ort` 2.0.0-rc.11 — Direct ONNX inference for layout detection + formula recognition; CUDA / DirectML (Windows) / CoreML (macOS) execution providers
- `ndarray` 0.17 — Tensor construction for ort model inputs/outputs
- `half` 2 — FP16 type for formula model tensors
- `tokenizers` 0.21 — HuggingFace tokenizer for formula token decoding
- `image` 0.25 — Image processing
- `reqwest` (blocking) — Model download from GitHub releases

## Model Management

Layout/table models auto-download from `github.com/GreatV/oar-ocr/releases` on first use.
Formula models (`encoder_fp16.onnx`, `decoder_fp16_argmax.onnx`) must be pre-exported via
`py/pp-formulanet/cuda/export.py` and placed in the model cache directory.
Cache directory: `{dirs::cache_dir()}/papers/models/` (override: `PAPERS_MODEL_DIR` env var).

## Execution Providers

- **Windows**: CUDA (NVIDIA GPU) with DirectML fallback (any DirectX 12 GPU), then CPU
- **macOS**: CoreML (Apple Neural Engine) with CPU fallback

## Testing

Unit tests use synthetic data — no models or pdfium required.
Integration tests (manual): need pdfium binary + downloaded models + formula ONNX models.
