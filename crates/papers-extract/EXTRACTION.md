# Extraction Pipeline Architecture

This crate has two extraction paths: the **layout path** (GPU-accelerated ML models)
and the **text-only path** (geometric heuristics on the PDF text layer).

## Layout Path (`Pipeline::extract()`)

```
PDF
 │
 ├─ extract_page_chars() ──→ page_chars (all pages)
 │   └─ toc::parse_toc()
 │
 └─ Per page:
     ├─ render_page() ──────────→ page image (CPU)
     ├─ layout.detect_batch() ──→ DetectedRegion[] with kinds + bboxes (GPU)
     │   (PageHeader, PageFooter, PageNumber, Text, ParagraphTitle,
     │    DisplayFormula, InlineFormula, Table, Image, Algorithm, ...)
     ├─ char-based formula extraction (fast)
     ├─ formula OCR batch (GPU) ──→ LaTeX per formula
     ├─ table OCR (GPU) ──→ HTML per table
     ├─ build_regions() ──→ Region[] with text per bbox
     │   └─ text::extract_region_text()
     ├─ xy_cut_order() ──→ reading order
     ├─ figure::associate_captions()
     └─ strip_structural_regions() ← removes PageHeader/Footer/PageNumber

Then:
 ├─ write_json() ──→ .json
 ├─ output::reflow() / reflow_with_outline()
 │   ├─ Running header filter (exact text 3+ pages)
 │   ├─ Positional header/footer filter (top 12%, bottom 15%)
 │   ├─ dedup_heading_echo()
 │   ├─ collapse_doubled_chars()
 │   └─ dedup_consecutive_text()
 ├─ write_reflow_json() ──→ .reflow.json
 └─ render_markdown_from_reflow() ──→ .md
```

## Text-Only Path (`extract_text_only()`)

```
PDF
 │
 ├─ extract_page_chars() ──→ page_chars (all pages)
 │   ├─ toc::parse_toc()
 │   ├─ headings::extract_headings() ← font-based, no models
 │   └─ detect_watermark_strings() ← cross-page text+font patterns
 │
 └─ Per page:
     └─ text_only::extract_page_text_blocks()
         ├─ convert_and_filter() ← margin filter + watermark removal
         ├─ detect_column_gaps() ← char-level X-projection
         │   ├─ no gaps → single column path
         │   └─ gaps found → split_chars_into_columns()
         │       ├─ spanning chars (cross-gap Y-bands)
         │       └─ per-column chars
         ├─ group_into_lines() + group_into_blocks() (per column)
         ├─ text::extract_region_text() ← shared with layout path
         ├─ text_cleanup::clean_block_text() ← shared utilities
         │   ├─ detect_drop_cap() ← moved from pipeline.rs
         │   ├─ fix_doubled_ligatures()
         │   ├─ strip_indesign_metadata()
         │   └─ dedup_within_block()
         ├─ classify_block() ← headings, captions, font heuristics
         └─ xy_cut_order() ← shared with layout path

Then:
 ├─ filter_running_headers() ← topmost/bottommost, aggressive normalization
 │   └─ strip_fused_headers() ← second pass for inline headers
 ├─ write_json() ──→ .json
 ├─ output::reflow() / reflow_with_outline() ← same as layout path
 └─ render_markdown_from_reflow() ──→ .md
```

## Shared Code

| Module | Function | Used by |
|--------|----------|---------|
| `pdf.rs` | `extract_page_chars()` | Both paths |
| `text.rs` | `extract_region_text()` | Both paths |
| `text_cleanup.rs` | `detect_drop_cap()` | Both paths (moved from pipeline.rs) |
| `text_cleanup.rs` | `fix_doubled_ligatures()` | Text-only |
| `text_cleanup.rs` | `strip_indesign_metadata()` | Text-only |
| `text_cleanup.rs` | `clean_block_text()` | Text-only (calls all above) |
| `text_cleanup.rs` | `match_label_prefix()` | Both paths — shared caption/label detection |
| `text_cleanup.rs` | `label_to_region_kind()` | Both paths — maps prefix → RegionKind |
| `text_cleanup.rs` | `ALL_LABEL_PREFIXES` | Both paths — single source of truth for label patterns |
| `text_cleanup.rs` | `is_likely_formula_text()` | Text-only — text-level formula detection |
| `text_cleanup.rs` | `extract_formula_tag()` | Text-only — extracts "(3.2)" tags from formulas |
| `reading_order.rs` | `xy_cut_order()` | Both paths |
| `toc.rs` | `parse_toc()` | Both paths |
| `headings.rs` | `extract_headings()` | Text-only (layout uses ML instead) |
| `output.rs` | `reflow()` / `reflow_with_outline()` | Both paths |
| `output.rs` | `render_markdown_from_reflow()` | Both paths |
| `output.rs` | `collapse_doubled_chars()` | Both (runs in reflow) |
| `output.rs` | `dedup_heading_echo()` | Both (runs in reflow) |
| `output.rs` | `dedup_consecutive_text()` | Both (runs in reflow) |

## Text-Only Detection Logic

### Font-based heading detection
The **font is the definitive signal** for headings. `partition_heading_chars()`
extracts chars matching the heading font family (e.g., LinBiolinum vs body
font LinLibertine). Heading chars are validated against `extract_headings()`
results. Known headings take **priority over formula zones** — a heading at
a Y-position overlapping a formula zone is kept because font overrides
geometric heuristics.

Body blocks should NEVER be classified as headings when font-based detection
is active (`has_font_headings` flag). Multi-line body blocks that happen to
start with heading text are not headings — the font is the tell.

### Formula detection
Two detection paths at the line level:
1. **Content-based** (`is_likely_formula_text()`) — operators, math symbols,
   prose word rejection, `$...$` marker density
2. **Font-based** — ≥3 math italic Unicode chars (U+1D400-1D7FF) with no
   prose words. These chars ARE the font signal.

Formula detection is **suppressed in algorithm zones** (see below).

**Key issues encountered:**
- `$`/`{`/`}`/`_`/`^` markers from `extract_region_text()` inflate char
  counts and dilute math_ratio. Use `content_total` (excluding these
  formatting chars) for ratio calculations.
- "fi"/"fifi" ligature artifacts from PDF absolute value bars `|...|` look
  like prose words. Excluded via `is_ligature_artifact()`.
- Pseudocode keywords ("if", "for") should only reject formulas when at the
  START of a line (pseudocode), not mid-line (piecewise formula conditions
  like "value, if condition").
- No upper char limit on `is_likely_formula_text()` — piecewise formulas
  and multi-line formulas can be very long.

### Algorithm detection
Algorithm zones are detected from **line numbers** ("1:", "15:", "37:") — an
unambiguous structural signal that display formulas never have. When ≥2
numbered lines exist in a column, their Y-range defines an algorithm zone.

Inside algorithm zones:
- **Formula detection suppressed** — pseudocode with math variables is not
  a display formula
- **All heuristic breaks suppressed** (y_break, x_break, font_break) —
  algorithm pseudocode has subscript fragments, varying indentation, and
  font size changes that would fragment the block

Algorithm caption splitting: when a block starts with "Algorithm N *title*"
and contains numbered lines, the caption is split into a separate
`FigureTitle` region and the body becomes an `Algorithm` region.

In the reflow stage, `Algorithm` nodes are **never demoted to Text**.
They either stay as `Algorithm` (pseudocode) or get promoted to `CodeBlock`
(actual programming code). `Algorithm` renders as plain text (not fenced
code blocks) because algorithms can contain `$...$` LaTeX math.

### Subscript / superscript handling
**Fundamental issue**: PDF text layer chars in math expressions span
multiple Y positions (subscripts, superscripts, fraction numerators and
denominators). `group_into_lines()` uses Y-proximity to group chars into
lines.

**Current approach**: bbox-based line grouping. A char belongs to the current
line if its Y-center falls within the line's Y bounding box (expanded by
`avg_height * 0.3` padding). This handles normal subscripts (3-4pt offset)
but NOT fraction numerators/denominators (6-8pt offset) because they would
merge actual separate prose lines.

**Remaining limitation**: fraction parts (`1/Δt²`) still fragment into
separate lines in the text-only path. The ML layout path handles this via
OCR (GLM-OCR produces per-line LaTeX). Fixing this in the text-only path
requires X-proximity-aware grouping — chars at different Y but overlapping
X are part of the same expression. This is a known TODO.

**Fragment break suppression**: tiny lines (≤3 chars) never cause heuristic
breaks (y_break, x_break, font_break) because they're subscript/superscript
fragments attached to adjacent content, not separate blocks.

### Overlapping formula deduplication
`dedup_overlapping_formulas()` **merges** (not picks-one) overlapping formula
regions. Requires both vertical AND horizontal overlap to prevent merging
formulas across columns. This handles cases where loose_bounds inflate
formula bboxes into overlapping territory.

## Known Limitations

### Text-only path
- **Two-column papers**: Column detection works for standard gutters. Complex layouts (mixed single/two-column, three-column) may partially interleave.
- **Fused words**: "forus", "helpedus" — Stage 1 space detection issue in pdf.rs.
- **Missing body content**: Some PDFs (computer_systems, programming_massively_parallel) have body pages that extract to empty.
- **Font mapping failures**: lambda book — TeX fonts lack Unicode mappings.
- **Margin notes**: Interleaved with body text in some books (EDO, fluids).
- **Fraction fragmentation**: Math fractions (`1/Δt²`) produce separate lines for numerator, bar, and denominator in `group_into_lines()`. No Y-threshold alone can merge them without also merging real separate prose lines. Needs X-proximity-aware grouping.

### Layout path
- Requires GPU for layout detection, formula OCR, and table OCR.
- Slower (minutes vs seconds for text-only).
