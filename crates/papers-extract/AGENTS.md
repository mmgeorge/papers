# papers-extract — Local PDF Extraction Pipeline

## Overview

Local, pure-Rust PDF processing pipeline using ONNX models and `pdfium-render`.
Converts PDFs into structured JSON + Markdown + extracted images. Uses PP-DocLayoutV3 for
25-class layout detection with built-in reading order via direct `ort` inference.
Table/formula recognition uses `oar-ocr` (SLANet/FormulaNet).

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
  │     ├── Table/formula recognition (oar-ocr OARStructure — tables + formulas only)
  │     ├── Text extraction (match pdfium chars to detected regions, Y-axis converted)
  │     ├── Table HTML (IoU-matched from oar-ocr TableResult)
  │     ├── Formula LaTeX (IoU-matched from oar-ocr FormulaResult)
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
| `types.rs` | `ExtractionResult`, `Page`, `Region`, `RegionKind` (23 variants), `Metadata` |
| `error.rs` | `ExtractError` enum |
| `layout.rs` | `LayoutDetector` — direct ONNX inference on PP-DocLayoutV3, `DetectedRegion` |
| `pipeline.rs` | `Pipeline` struct — owns pdfium + LayoutDetector + OARStructure, orchestrates per-page processing |
| `models.rs` | Model download from GitHub releases, OARStructure + LayoutDetector builders, execution provider config |
| `pdf.rs` | `PdfChar`, `load_pdfium()`, `render_page()`, `extract_page_chars()` |
| `text.rs` | Match pdfium characters to layout regions, reconstruct text with word/paragraph detection. Converts PdfChar Y-up coords to image Y-down space. |
| `figure.rs` | Crop visual regions, associate captions by proximity |
| `reading_order.rs` | XY-Cut fallback algorithm for reading order |
| `output.rs` | Write JSON, Markdown, and cropped images to disk |

## Key Types

### RegionKind (23 classes)

```
Title, ParagraphTitle, Text, VerticalText, PageNumber,
Abstract, TOC, References, Footnote, PageHeader, PageFooter,
Algorithm, DisplayFormula, FormulaNumber,
Image, Table,
FigureTableTitle, FigureTitle, TableTitle, ChartTitle,
Seal, Chart, SidebarText
```

Content routing by kind:
- **Text-bearing** → `text` field (pdfium char extraction)
- **Table** → `html` field (oar-ocr SLANet-Plus)
- **DisplayFormula** → `latex` field (oar-ocr PP-FormulaNet)
- **Visual** (Image/Chart/Seal) → `image_path` field (cropped PNG)
- **Caption** → `text` field + associated with parent via `caption`

### Quality Modes

- **Fast** (default): SLANet-Plus (7 MB) + PP-FormulaNet_plus-S (248 MB) — ~379 MB total
- **Quality**: PP-LCNet classifier (6.5 MB) + SLANeXt-wired (351 MB) + PP-FormulaNet_plus-L (698 MB) — ~1.18 GB total

## Dependencies

- `oar-ocr` 0.6 — ONNX inference for table and formula recognition (layout detection bypassed)
- `pdfium-render` 0.8 — PDF loading, rendering, text extraction (requires pdfium binary)
- `ort` 2.0.0-rc.11 — Direct ONNX inference for layout detection + DirectML (Windows) / CoreML (macOS) execution providers
- `ndarray` 0.17 — Tensor construction for ort model inputs/outputs
- `image` 0.25 — Image processing
- `reqwest` (blocking) — Model download from GitHub releases

## Model Management

Models auto-download from `github.com/GreatV/oar-ocr/releases` on first use.
Cache directory: `{dirs::cache_dir()}/papers/models/` (override: `PAPERS_MODEL_DIR` env var).

## Execution Providers

- **Windows**: DirectML (any DirectX 12 GPU) with CPU fallback
- **macOS**: CoreML (Apple Neural Engine) with CPU fallback

## Testing

Unit tests use synthetic data — no models or pdfium required.
Integration tests (manual): need pdfium binary + downloaded models.
