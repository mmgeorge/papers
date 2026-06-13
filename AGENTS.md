# Project-Wide Agent Instructions

## Temporary files

All temporary or scratch files (test outputs, diagnostic dumps, intermediate data,
etc.) **must** go in the `.temp/` directory at the project root. Never write temp
files to `data/`, source directories, or the system temp directory.

```bash
# correct
.temp/vbd_p4_output.txt
.temp/debug_chars.json

# wrong
data/debug_output.txt
crates/papers-extract/scratch.txt
/tmp/papers-test.json
```

The `.temp/` directory is gitignored. Create it on demand if it doesn't exist.

## Extraction Pipeline Architecture

The PDF extraction pipeline has three stages. Fixes and logic should go
in the **correct stage** — not downstream.

### Stage 1: PDF → JSON (pipeline.rs, text.rs, toc.rs, pdf.rs)

The extraction stage produces a `.json` file with pages, regions, and
structured content. This is the **source of truth** — clients may consume
the JSON directly without ever generating markdown.

**All data quality fixes belong here:**
- Region deduplication (overlapping bboxes)
- Text extraction (space detection, ligature expansion, split-word fixes)
- Region classification (code detection, formula promotion, table demotion)
- Structural filtering (headers, footers, figure-internal text)
- TOC parsing and page offset computation

### Stage 2: JSON → Reflow JSON (output.rs)

The reflow stage reads the extraction JSON and builds a hierarchical
document tree (`ReflowDocument` with `ReflowNode` children). This is
where **structural organization** happens:
- Heading tree construction from TOC + page offsets
- Paragraph rejoining across page/column breaks
- List detection, footnote association
- Heading echo deduplication
- Front matter / back matter handling

### Stage 3: Reflow JSON → Markdown (output.rs `render_markdown_from_reflow`)

The markdown renderer should be **straightforward** — it just walks the
reflow tree and emits markdown. Minimal logic here. Only cosmetic
post-processing (e.g. fixing split bold labels in rendered text).

### Key principle

If a fix can be done in Stage 1, do it there — the JSON should be clean.
If it requires document-level context (cross-page heading structure), do
it in Stage 2. Stage 3 should have almost no logic.

## Pdfium Text Extraction — Reference Implementation

When fixing text extraction bugs (missing spaces, wrong word boundaries, etc.),
**always consult pdfium's C++ source code as the reference implementation**.
Pdfium already solves these problems correctly — our job is to replicate its
behavior, not invent heuristics.

### Key source file

`core/fpdftext/cpdf_textpage.cpp` in the [pdfium repo](https://pdfium.googlesource.com/pdfium/).

### Space detection: two paths

1. **Inter-object** (`ProcessInsertObject`): When a new text object starts,
   compare its position to where the previous character's advance would have
   placed the cursor. Threshold uses `NormalizeThreshold(max(lastCharWidth,
   thisCharWidth), 400, 700, 800) * fontSize / 1000`.

2. **Intra-object** (`CalculateSpaceThreshold`): Within a single text object,
   TJ kerning displacements are accumulated. If `spacing >= threshold`, a space
   is inserted. Threshold = `fontSize * font.GetCharWidthF(' ') / 1000 / 2`.

`NormalizeThreshold(t, a, b, c)` divides the input by 2/4/5/6 depending on
which tier it falls in (< a → /2, < b → /4, < c → /5, else /6).

### What pdfium exposes via its C API

| API | What it gives you | Rust wrapper (pdfium-render) |
|-----|-------------------|-----------------------------|
| `FPDFText_GetText` / `FPDFText_GetBoundedText` | Full text with spaces already inserted | `PdfPageText::all()`, `inside_rect()` |
| `FPDFText_IsGenerated` | Whether a char was synthesized (space, etc.) | `PdfPageTextChar::is_generated()` |
| `FPDFText_GetCharOrigin` | Typographic origin (advance-based position) | `PdfPageTextChar::origin_x()` |
| `FPDFText_GetLooseCharBox` | Visual bounding box | `PdfPageTextChar::loose_bounds()` |
| `FPDFText_CountChars` | Count includes generated chars | `PdfPageTextChars::len()` |

### Common mistakes to avoid

- **Don't invent gap heuristics** when pdfium already solved the problem.
  `text.all()` returns correct text — if your char-by-char reconstruction
  disagrees, your code is wrong, not pdfium.
- **Don't assume origin_x == bbox left**. For italic/kerned glyphs they differ.
  Use origin for position tracking, bbox for visual extent.
- **Don't filter out space chars and reconstruct from gaps**. Instead, track
  which chars had pdfium-generated spaces before them (`pdfium_space_before`
  on `PdfChar`). This preserves pdfium's advance-width analysis.
- **Bounding box gaps underestimate real spacing** because bbox is the visual
  extent (tight around ink), not the advance width. Characters like italic 'f'
  have bbox extending past the advance, making the gap to the next char appear
  smaller than the actual word spacing.

### Current implementation

`PdfChar` carries `pdfium_space_before: bool` — set to `true` when the
preceding character in pdfium's char stream was a generated space. This is
the most reliable word-boundary signal. In `TocRawLine::text()`, this flag
is checked alongside gap-based and position-based space detection.

## TOC Fixture Tests

Fixture files in `crates/papers-extract/tests/fixtures/toc/*.md` are the
**ground truth** of what the TOC parser should produce. When fixtures and
parser disagree, fix the parser — never silently update fixtures to match
broken output. The only valid reason to update a fixture is when the parser
now produces *more correct* output (e.g., fixing a previously-missing space).

## Benchmarks

Benchmark results go in `benchmarks/<model-name>/`. Each benchmark directory
should contain a `README.md` with findings and the raw `results.json` output.

### Workflow

1. **Dump layout** from a PDF (one-time per paper):
   ```bash
   cargo run --release --bin dump -- data/<paper>.pdf data/dumps/<paper>
   ```
   This creates `data/dumps/<paper>/layout.json` and cropped region images
   organized by type (e.g. `DisplayFormula/`, `Text/`, `Algorithm/`).

2. **Run a model** on dumped regions:
   ```bash
   # GLM-OCR (any region type, comma-separated; --backend cuda|coreml|cpu|auto)
   cargo run --release --bin run_glm_ocr -- data/dumps/<paper> \
     -o .temp/results/<paper>-glm \
     --region-type "Text,DisplayFormula,Algorithm,Table"
   ```

3. **Benchmark mode** (`--bench`) runs each image multiple times (default 2)
   and reports median/min/max/stddev:
   ```bash
   cargo run --release --bin run_glm_ocr -- data/dumps/<paper> \
     -o .temp/results/<paper>-glm \
     --region-type DisplayFormula --bench --runs 3
   ```

4. **Useful flags**:
   - `--limit N` — process at most N regions per kind
   - `--page P` — filter to a single page
   - `--dump` — print OCR output to stdout

5. **Write up results** in `benchmarks/<model-name>/README.md`.
