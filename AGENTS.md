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
