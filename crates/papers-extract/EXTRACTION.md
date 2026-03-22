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

## Known Limitations

### Text-only path
- **Two-column papers**: Column detection works for standard gutters. Complex layouts (mixed single/two-column, three-column) may partially interleave.
- **Fused words**: "forus", "helpedus" — Stage 1 space detection issue in pdf.rs.
- **Missing body content**: Some PDFs (computer_systems, programming_massively_parallel) have body pages that extract to empty.
- **Font mapping failures**: lambda book — TeX fonts lack Unicode mappings.
- **Margin notes**: Interleaved with body text in some books (EDO, fluids).

### Layout path
- Requires GPU for layout detection, formula OCR, and table OCR.
- Slower (minutes vs seconds for text-only).
