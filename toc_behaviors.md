# TOC Parsing Behaviors

This document describes the rules applied when parsing a PDF Table of Contents into
structured `TocEntry` records. It covers what input is accepted, how entries are
classified, and what transformations happen before the final output.

---

## 1. TOC Page Detection

The parser scans the first 40 pages (or the first third of the document, whichever is
smaller) looking for TOC pages.

**Primary signal - "Contents" heading:**
A page is the TOC start if one of its three topmost lines normalizes to `contents`,
`tableofcontents`, or `detailedcontents`. Pages beginning with "List of..." are
excluded.

**Fallback - content signals:**
If no Contents heading is found, the parser looks for consecutive pages near the front
where >= 30% of lines end with a page number, or >= 20% have leader dots.

**Continuation:**
From the start page, the parser extends forward as long as each successive page meets
the content-signal threshold. A single blank page is tolerated.

**Stopping conditions:**
- A page whose topmost line starts with "List of Figures", "List of Tables", or
  "List of Algorithms" stops TOC collection immediately.
- Pages that fail the content-signal test stop the run.

**Pre-TOC pages:**
TOC-like pages immediately before the Contents heading page (e.g., "Contents at a
Glance") are collected for skipping during body reflow, but are not used for entry
extraction.

---

## 2. Line Extraction and Cleanup

Characters are grouped into horizontal lines by Y-position (within half an average
glyph height).

**Margin trimming:** Lines in the top or bottom 3% of the page are discarded.

**Running header removal:** The topmost line on each TOC page is removed if it matches
"Contents", "Table of Contents", or "Detailed Contents".

**Bare page number lines:** Lines whose entire text is a page number (<= 6 chars,
digits or lowercase roman) are dropped.

**Ligature expansion:** TeX ligatures (fi, fl, ff, ffi, ffl) are expanded into their
component characters.

---

## 3. Two-Column Layout Detection and Splitting

If >= 25% of qualifying lines (>= 10 chars) have a large internal X-gap (> 30pt) at a
consistent horizontal position, and both sides of the gap have substantial content
(>= 8 chars), the page is treated as a two-column TOC.

Each line is split at the detected gutter. All left-column lines are read
top-to-bottom first, then all right-column lines.

---

## 4. Page Number Extraction

Each line is split into a title and a page number using three strategies in order:

**A - Leader dots:**
Text before `...` or `. . .` (3+ dots) is the title; tokens after are the page ref.

**B - X-gap:**
The rightmost digit run (or lowercase roman run) separated by a gap >= 4x the median
character width is the page number.

**C - Last token:**
The last whitespace-delimited token is tried as a page reference.

**Page value encoding:**
- Arabic: positive integer.
- Lowercase roman numerals: negative integer (i = -1, ii = -2, ...).
- Local page labels in N-M format (e.g., `3-14`): value 0, label preserved.
- Internal whitespace in numbers is stripped (OCR artifact).

Lines without a page number are kept only if they match the Part heading pattern (see
Section 7). All others are buffered as potential continuation fragments.

---

## 5. Embedded Multi-Entry Lines

If a title contains a mid-text token group that parses as a valid page reference, and
both sides individually look like valid TOC entry starts, the line is split into two
entries. This repeats recursively.

---

## 6. Multi-Line Title Merging

A line with a page number whose title does not start with a heading pattern is treated
as a continuation of the preceding titleless buffered line(s), when their font
signatures match (family, size bucket, bold, italic, all-caps).

Line-break hyphenation: if the fragment ends with `-` and the continuation starts
lowercase, the hyphen is removed. If the continuation starts uppercase, the hyphen is
kept as a compound word.

---

## 7. Entry Classification

Each entry is classified in priority order:

| Priority | Pattern | Classification | Depth |
|----------|---------|----------------|-------|
| 1 | `CHAPTER N` or `Chapter N` (digit follows) | Chapter | 1 |
| 2 | `Part N` or `PART N` (roman numeral or digit follows) | Part | 0 |
| 3 | Multi-level dotted decimal: `1.1`, `2.3.4`, etc. | Section / Subsection | by dot count |
| 4 | Appendix sub-section: `A.1 Title`, `A.1.2 Title` | Section / Subsection | by dot count |
| 5 | Bare number + word: `1 Title`, `1. Title` | Chapter | 1 |
| 6 | `Appendix N` or `APPENDIX N` | BackMatter | 1 |
| 7 | Front/back matter keywords (Preface, Index, ...) | FrontMatter | 1 |
| 8 | Sub-entry keywords: Exercises, References, Notes, Summary, ... | SubEntry | last_depth + 1 |
| 9 | Roman numeral + word: `II Introduction` | Chapter (tentative) | 1 |
| 10 | Anything else | Unknown | resolved later |

After classification, `CHAPTER N` / `Chapter N` prefixes are removed from the stored
title. `"Chapter 3 Foo"` becomes `"3 Foo"`.

---

## 8. Font Signature Learning

For each pattern-classified (non-Unknown) entry, its font signature is recorded: font
family, size bucket, bold, italic, all-caps. A majority vote per signature yields a
depth-to-font mapping used to resolve Unknown entries.

---

## 9. Indentation Level Learning

X-left positions are normalized per page (subtract the page minimum X) and clustered
into indent levels (positions within 5pt merge). Each level maps to the dominant depth
of classified entries at that level.

---

## 10. Unknown Entry Resolution

Unknown entries are resolved sequentially using a context stack of classified entries:

- **More indented than stack top:** child (depth = stack_top_depth + 1)
- **Same indent, smaller font:** child (subsection)
- **Same indent, same or larger font:** sibling (pop stack, search up)
- **Stack empty:** top-level (depth 1)

---

## 11. Front Matter Propagation

Unknown entries immediately adjacent to a FrontMatter-classified entry are promoted to
FrontMatter when they share the same font signature and x-indent (within 5pt).

Propagation only fires for FrontMatter entries that appear before the first Chapter or
Part entry in the TOC. Mid-body FrontMatter entries (e.g., per-chapter References) do
not propagate.

A second pass scans backwards from the end of the TOC: trailing Unknown entries sharing
the font and indent of any FrontMatter entry are reclassified as BackMatter.

---

## 12. Roman Numeral Chapter Groupings

When two or more entries are classified as Chapter with roman-numeral titles (`I Foo`,
`II Bar`), two cases are distinguished based on whether the arabic-numbered entries
between them restart or continue:

### Case A — Redundant grouping (Drop)

**Condition:** The arabic-numbered chapters between consecutive roman entries are
numbered **continuously** across roman entries (e.g., ch 1–5 under Part I, ch 6–10
under Part II).

**Action:** Drop the roman entries entirely. The arabic chapter numbering already
provides full structure; the roman labels are redundant noise.

**Example:** `I Fundamentals` / `1 Intro` / `2 Basics` / `II Advanced` / `3 Deep Dive`
→ `I Fundamentals` and `II Advanced` are removed; `1 Intro`, `2 Basics`, `3 Deep Dive`
remain.

### Case B — Roman entries ARE the chapters (Keep and promote)

**Condition:** The arabic-numbered entries **restart from 1** under each roman entry
(e.g., ch 1–2 under Part I, ch 1–2 under Part II). The roman entry has its own page
number and the arabic-numbered entries are sub-articles or articles within it.

**Action:** Keep the roman entries as depth-1 chapters (title unchanged). Demote the
arabic-numbered entries to depth-2 sections. Re-number sections and subsections with
the roman chapter as prefix: `I.1`, `I.2`, `I.1.1`, `II.3.4`, etc.

**Example (GPU Zen):** `I Geometry Manipulation` / `1 Attributed Vertex Clouds` /
`1.1 Introduction` / `II Lighting` / `1 Stable Indirect Illumination` →
`I Geometry Manipulation` at depth 1; children become `I.1 Attributed Vertex Clouds`,
`I.1.1 Introduction`; `II Lighting` at depth 1; `II.1 Stable Indirect Illumination`.

---

## 13. Per-Chapter Bibliography Demotion

"Bibliography", "References", "Notes", etc. classified as FrontMatter are demoted to
SubEntry (depth 2) when their page number falls between the start pages of two
consecutive non-front-matter chapters. This corrects books where per-chapter reference
sections share the book front-matter font.

---

## 14. Chapter Number Inference from Children

If a chapter-level entry has no leading number but all of its immediate children
(depth + 1 before the next sibling) share the same leading number prefix (all are
`1.1 ...`, `1.2 ...`, ...), that number is prepended to the parent title.
`"Foo Bar"` becomes `"1 Foo Bar"`.

---

## 15. Validation and Filtering

Applied after classification in this order:

**Deduplication:** Entries with the same (title, page_label, page_value) triple are
deduplicated; first occurrence kept.

**Leading page-number stripping:** A leading numeric token is stripped if it is close
in value to the entry's own page number, or is followed by a back-matter keyword
(Appendix, Index, References, Addenda).

**Continuation stubs dropped:** Entries whose title contains "(continued)" or is
exactly "continued" are removed.

**Enumerated body spills dropped:** Entries with titles >= 80 chars containing multiple
enumeration markers (` 1.`, ` 2.`, ` (a)`, etc.) are dropped as body text that leaked
into the TOC.

**Author-line filtering:** Non-structural entries (not Chapter/Section/etc.) whose
title matches a name-list pattern (2-8 title-case tokens, no structural keywords) are
dropped. Removes author attribution lines in handbooks.

**No-page entries dropped:** All entries without a page number are removed, except
unnumbered Part headings (depth 0).

**Sort by page order:** Before the sequencing check, entries are sorted:
- Front matter (negative pv): by |pv| ascending (page i before page ii).
- Body (positive pv): by pv ascending.
- Unnumbered Part headings (pv = 0): sorted last.

This handles brief-TOC + detailed-TOC structures: after deduplication the brief chapter
entries have increasing page values and the section entries have smaller values.
Sorting interleaves them correctly before the sequencing check.

**Sequencing check:** Entries that violate monotonic page order are dropped:
- Body entry dropped if pv < last_body - 2 (tolerance for sub-entries sharing a page).
- Front-matter entry dropped if it appears after any body entry, or jumps backwards by
  more than 2 within front matter.
- BackMatter entries and back-matter-titled entries (Bibliography, Index, ...) are
  exempt.

---

## 16. Outline Usability (outline_is_usable)

Decides whether the outline is reliable enough for heading anchoring during body reflow.

Fails if:
- The outline is empty.
- >= 50% of entries use local page labels (N-M format), which do not map to physical
  page numbers.
- > 3% of entries are fatally suspicious AND >= 10 in absolute count.
- >= 10% of entries are suspicious.

A title is **fatally suspicious** if `heading_start_count >= 2`, or the first token is
a page reference and the remainder looks like a new TOC entry start (except when the
remainder is a single bare number, which is likely "chapter N at page M" not a merged
entry).

**heading_start_count** counts how many positions in a title look like a new heading
start. Position 0 always counts. Later positions count only if:

- The candidate substring has >= 2 tokens.
- It starts with a *numeric* identifier: a digit sequence (`1`, `1.2`, `1.2.3`) or an
  uppercase roman numeral (`IV`). Textual patterns like `Chapter N` or `Part I` do not
  count, to avoid false positives from titles like "Summary of Chapter 1".
- The immediately preceding token is not a structural keyword (Part, Chapter, Appendix,
  Section), to avoid flagging "PART I . Title" where `I` belongs to Part.

---

## 17. Renderable TOC Usability (renderable_toc_is_usable)

A more lenient check for whether to render a TOC section in the output markdown.
Fails if >= 17% of entries are fatally suspicious, or >= 33% are suspicious.
