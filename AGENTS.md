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
