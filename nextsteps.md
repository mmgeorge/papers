# Next Steps

Status snapshot (verified against code 2026-06-19):

- TOC parser + V2 normalized fixtures: **41/41 curated green**, **603 lib tests green**.
- Exception ledger in `crates/papers-extract/tests/toc_fixtures.rs`: **15 per-line**
  (`EXCEPTIONS`, line 293) + **1 whole-file** (`CORRUPT_SOURCE_FILES = [lambda]`, line 352).
- Container-less appendix nesting (task #18) landed in `normalize_toc_entries`.
- **The V2 normalization is fixture-only.** It runs solely in `toc::render_fixture_markdown`
  (`toc.rs:520`); every production entry point calls `parse_toc` and reflows separately.

> **Section 2 (track the `data/open` corpus) is intentionally omitted.** The PDFs
> in `data/` and `data/open/` are copyrighted textbooks and cannot be committed to a
> public repo. Only the derived fixture `.md` files are tracked. This caveat shapes
> Section 1 (below): parser *code* fixes are public, but the *PDFs needed to run the
> regression suite are local-only*, so a fresh public checkout exercises 0 fixture cases.

---

## 1. TOC parser residual quality

Three open tasks (#19–#21) cover everything still diverging from ground truth. The
parser fixes all land in `crates/papers-extract/src/toc.rs` (public, committable);
the verifying fixtures are committable, but the PDFs that drive them are local-only.

### 1a. Task #19 — Stray "page" word + dropped/duplicate entries

Concrete defects to chase (render the TOC page + `dump_chars` each — do **not** infer):

| Book | Defect | Likely stage |
|------|--------|--------------|
| `d2l`, `understanding_ml` | `"Preface page"` — a column-header word ("page") fused into the title | Stage 1 line assembly / header strip |
| `gelman_bda3` | one dropped entry (`3.4.6 … continued`) — continuation stub wrongly removed | §15 continuation-stub filter |
| `mackay_itila` | page off-by-2 **and** line-count 63/121 (≈half the TOC lost) | page-offset + a structural drop |
| `opt` | `"Introduction"` page `xxi` vs `1` (roman/arabic confusion at the body boundary) | §4 page-value encoding |

`opt` and the off-by-2 in `mackay` are page-mapping bugs that also affect production
heading anchoring (Section 3), so they are worth doing first.

### 1b. Task #20 — Render-verify glyph-corruption candidates

Each of the 15 `EXCEPTIONS` must be proven source-side (render + `dump_chars`), then
*recover what is recoverable* and keep an exception only for the truly impossible.
Current ledger, split by tractability:

**Truly impossible (keep the exception):**
- `compilers` — `ε`→`ffl` and `"`→`\` are glyphs named only by raw code (`#0F`), no
  semantic name; needs a per-font code→Unicode table.
- `opt` — `ℝ` is drawn as two ASCII glyphs `I`+`R`; `glyph_recovery`'s 1:1 alignment
  cannot map 2 source glyphs → 1 character.
- `sutton_barto_rl` — `ffi` (`E ciency`) is raw code `#0E`, same class as `compilers`.
- `understanding_ml` — `≫` is control code `U+001D` (CMSY slot, no `/ToUnicode`).
- `erickson_algorithms` — `Mātrāvṛtta`: spacing diacritics emitted out of logical
  order; a 1:1 codepoint map cannot reorder/recombine them.

**Recoverable-but-regression-prone (try to eliminate):**
- `calculus` (5 lines) — primes `S′`, stacked fractions `dy/dt`, math `=`, superscript
  `e^a`. All present in the text layer but scattered across Y-grouped rows; needs
  stacked-fraction + overlap-based line assembly.
- `understanding_ml` `ℓ`, `≥` and `sutton` `σ` — present in the font, but the page's
  pdfium char-count diverges from the content stream, so `glyph_recovery`'s safe
  page-level 1:1 guard skips the page. A fuzzy per-line correlation would recover them
  but risks regressing currently-passing books — needs careful guarding.

Every line eliminated here shrinks `EXCEPTIONS` and proves the parser, not just the file.

### 1c. Task #21 — Remaining structural failures

Books that only pass via the ≥80% floor or carry the largest diffs. Root-cause each by
rendering the TOC page:

| Book | Symptom | Suspected root cause |
|------|---------|----------------------|
| `windows_cpp` | front-matter sub-entries not nested (≈384 lines changed) | front-matter indent → depth mapping |
| `handbook` | author names glued onto titles (≈65) | §15 author-line strip too weak |
| `lambda` | 143/144 (whole-file exception) | letter-spaced part numerals + broken `/ToUnicode` (genuinely corrupt) |
| `fluids` (271), `gpml` (140), `discrete_math` (112), `think_stats` (158), `pro_git` (29), `jurafsky_slp3` (23) | assorted: part merge, back-matter mis-class, footer splice | one render per book to classify |

Note most of #21's list (`gpml`, `discrete_math`, `think_stats`, `pro_git`,
`jurafsky_slp3`) is **open-corpus** — fixable in `toc.rs` but only verifiable locally.
Committable wins here are `windows_cpp`, `handbook`, `fluids`, `lambda`.

---

## 3. Wire V2 normalization into the production pipeline  ⟵ the consequential one

This is the gap that decides whether all the TOC work actually reaches LanceDB. Today
there are **two separate, independently-evolved engines** that turn a parsed TOC into
structured output, and only the fixture one received the V2 rules.

### Path A — fixture normalization (has the V2 rules)

```
render_fixture_markdown (toc.rs:505)
  → normalize_toc_entries (toc.rs:535)
  → render_normalized_rows (toc.rs:751)
```
Consumers: the `toc_fixtures` test and the `gen_toc_fixture` bin. **Not production.**
Rules applied (the V2 spec):
- **Position renumbering** — discards the book's printed numbers, assigns `N`, `N.M`,
  `N.M.K` purely by tree depth (`render_normalized_rows`, counters per depth).
- **Parts kept + chapters nested** under them via the `in_part` offset (`Part I Theory`
  → `2 Theory`, its chapters → `2.1`, `2.2`).
- **Appendix container synthesis + nesting** (`synth_appendix`/`app_offset`, task #18):
  container-less `A`/`B`/`C` → `N.1`, `N.2`, sub-sections `N.1.1`.
- **Front-matter** nested under its head and **unnumbered**; **back-matter** unnumbered.
- **Footer/copyright drop** (`is_footer_title`), **chapter-tail unnumbering**
  (`is_tail_title`: Exercises/Summary/Notes…), **hints/solutions nesting**, **page-less
  navigation drop**.

### Path B — production reflow (does NOT have the V2 rules)

```
parse_toc  →  reflow_with_outline (output.rs:1905)
  → toc_page_to_pdf_page  (map entry.page_value → physical page via offsets)
  → has_parts ? depth += 1
  → build_heading_tree (output.rs:3423, stack-based nesting by depth)
  → auto_number_document (output.rs:2240 / 2366)
  → render_markdown_from_reflow (output.rs:3555)
```
Entry points: `lib.rs:228` (reflow-from-JSON), `lib.rs:659` (full extract),
`pipeline.rs:301`. This is the markdown that feeds LanceDB.

### The divergences (confirmed from code — this is the audit checklist)

| Concern | Path A (fixtures/V2) | Path B (production) |
|---------|----------------------|---------------------|
| **Numbering** | Discards printed numbers, pure position numbers | **Preserves** printed `section`; only auto-numbers a depth when <30% are already numbered (`auto_number_document`, ratio < 0.3, line 2388) |
| **Parts** | Kept, chapters renumbered part-relative (`2.1`) | `depth += 1` so parts nest, but chapter keeps printed number (`9`, not `2.1`) (line 1935) |
| **Appendices** | Synthesizes "Appendices" container, nests `N.1`… | **No container synthesis** — appendices flow through as parser-tagged |
| **Front/back matter** | Unnumbered, front nested under head | `FrontMatter` node + page clamp; numbering only skipped if no printed `section` |
| **Footer / page-less / tail / hints** | Explicit drop & unnumber rules | Not applied to the heading tree |

Net effect: for any multi-part or appendixed book, production markdown and the fixture
ground truth are **structurally and numerically different**.

### Why you can't just swap A in for B

Path B is **page-anchored**: it maps every entry to a physical PDF page
(`toc_page_to_pdf_page`) to place body content under the correct heading. Path A has
*thrown that away* — it is a pure outline with no entry→page→content identity. So the
reflow path genuinely needs the page offsets and original entry order that A discards.

### Design options

- **(a) Extract a shared structural-rules module** (recommended direction). Pull the V2
  decisions — depth assignment, position-renumber vs preserve, part/appendix/front-back
  handling, tail/footer rules — out of `normalize_toc_entries` into a function over
  `&[TocEntry]` that returns per-entry `(depth, number, numbered)` **without** rendering.
  Both `render_normalized_rows` and `reflow_with_outline`/`auto_number_document` consume
  it. Production keeps its page anchoring; only the numbering/structuring is unified. One
  source of truth, fixtures stay the contract.
- **(b) Port the rules into the reflow path** directly (duplicate logic in `output.rs`).
  Faster but creates a second copy of every rule — exactly the drift we have now.
- **(c) Declare them intentionally different** — fixtures = idealized outline for
  retrieval metadata; production markdown = page-anchored reflow. Cheapest, but then the
  41-fixture investment never affects the LanceDB output and we should say so explicitly.

Recommendation: **(a)**. It removes the duplicate numbering engine
(`auto_number_document`/`auto_number_tree` vs `render_normalized_rows`) rather than
adding vocabulary, and it is the only option where the fixture suite actually guards the
shipped output.

### Pre-work / audit before implementing

1. Diff production markdown vs the V2 fixture for 3–4 representative books (a parted one,
   an appendixed one, a flat one) to quantify the gap concretely.
2. Decide whether LanceDB ingestion wants the **position numbers** (clean, stable anchors)
   or the **printed numbers** (match the physical book). This is a product call and it
   drives option (a)'s `number` field.
3. Confirm the `lib.rs:228` reflow-from-JSON path and the non-outline fallback `reflow`
   (output.rs:1602) get the same treatment — they currently number differently from the
   `reflow_with_outline` path.

---

## 4. Documentation reconciliation

Several docs describe a mid-rewrite state that the code has since moved past. They now
actively mislead. Fix or delete each.

### 4a. `EXCEPTIONS.md` (repo root, untracked)

The **"Normalized-format exceptions (2026-06-14)"** section (lines 93–126) is wrong:
- Claims **~13 whole-file `CORRUPT_SOURCE_FILES`** (`classical_mechanics`,
  `computer_systems`, `mit_math_for_cs`, `windows_cpp`, `discrete_math`, `gpml`,
  `handbook`, `mackay_itila`, `fluids`, `think_stats`, `pro_git`, `gelman_bda3`,
  `jurafsky_slp3`). **Code has exactly one: `lambda`.** Those books were fixed and moved
  to exact-match; the doc never got updated.
- Its per-line list (`category_theory`, `engine_physics`, `eloquent_js`, `shaders`,
  `think_os`, `windows`, `d2l`, `zen0`, …) does not match the actual 15-tuple `EXCEPTIONS`
  (only `compilers`, `calculus`, `opt`, `sutton_barto_rl`, `understanding_ml`,
  `erickson_algorithms`).

Action: rewrite that section to the real ledger (15 per-line + `lambda`), keep the
"Per-line / Whole-file mechanism" prose (still accurate), and **track the file** — the
test code comments reference it (`toc_fixtures.rs` "see EXCEPTIONS.md").

### 4b. Memory `toc-normalized-format-v2.md` + the `MEMORY.md` index line

- Description field still says **"parser not yet updated"** — false, it's implemented and
  green (the body even contradicts its own description).
- Body lines 25–31 repeat the stale **"~14 whole-file `CORRUPT_SOURCE_FILES`"** — now 1.
- The `MEMORY.md` index entry says **"appendices lettered … Parser update is the next
  phase."** Both are false now: appendices are position-numbered + nested (task #18), and
  the parser/normalizer is done.

Action: update the memory body + the index hook to current reality (implemented; 41/41;
1 whole-file; appendices nested-and-numbered, not lettered).

### 4c. `toc_behaviors.md` (repo root, untracked)

Accurate **as far as it goes** (17 sections on `parse_toc` → `TocEntry`), but it stops at
`TocEntry` production and **never documents the entire V2 normalization layer**
(`normalize_toc_entries`/`render_normalized_rows`). Anyone reading it would not know the
fixture format exists.

Action: add sections (18+) describing the normalization stage — position renumbering,
part nesting, appendix synthesis, front/back unnumbering, tail/footer/page-less rules.
This becomes the spec that Section 3's shared module implements against.

### 4d. `toc_behaviors_new.md` (repo root, untracked)

A completed action-item list (§12 Case A "drop romans" → `convex`; Case B "keep+promote"
→ `zen0`). Both behaviors are implemented and their fixtures pass.

Action: **delete it** (or fold any non-obvious notes into `toc_behaviors.md` §12, which
already documents the implemented behavior).

---

## 5. The larger non-VLM pipeline

The TOC/heading work is one foundation stone of the original goal: a **fast non-VLM
parse → clean markdown/JSON → LanceDB**, with expensive **VLM formula/table OCR deferred
to read-time and cached**. Two big bodies of work remain beyond TOC.

### 5a. Text-only extraction V2 (approved, not started)

Plan: `C:\Users\mattm\.claude\plans\jazzy-questing-lovelace.md` (approved 2026-03-21).
The `--text-only` flag and `text_only.rs` (~600 lines) already exist; baselines are in
`.temp/text-only-baseline/` (24 books) and `.temp/pdfium-raw/`. Eight phases, in order:

1. `text_cleanup.rs` shared module — move `detect_drop_cap` out of `pipeline.rs`, add
   ligature dedup, InDesign-artifact strip, within-block dedup.
2. Watermark detection — cross-page char-level text+font pattern matching (blocks false
   column detection downstream).
3. Column-gap detection — char-level X-projection with vertical validation.
4. Char-level column splitting — per-Y-band spanning detection.
5. Running-header filter rewrite — topmost/bottommost-only, aggressive normalization,
   ≥4 pages, `consumed=true`.
6. Fused inline header stripping — second pass over known header patterns.
7. `EXTRACTION.md` documentation.
8. Run all 24 PDFs, verify against ~100 unit tests.

Goal: lift readability from 2–5/10 to 7–9/10 — the quality bar for vector-DB ingestion.
This is the **text** half of "clean output for LanceDB"; Section 3 is the **structure**
half. They compose: structure (headings/numbers) + clean reflowed text = the document
tree that gets embedded.

### 5b. End-to-end pipeline + VLM-on-read

The pieces that turn the above into the product:
- **LanceDB ingestion** — define the row schema (doc / heading-path / page / text /
  embedding) and write the ingestion of the reflow tree. The heading numbers from
  Section 3 become the stable retrieval anchors, so 3 should land first.
- **VLM formula/table parsing on read** — the GLM-OCR work (see memory:
  GLM-OCR ONNX surgery, `run_glm_ocr` bin, `benchmarks/`) is the expensive read-time
  path. Formulas/tables are detected at parse time (region classification) but only
  OCR'd when a query hits them, then cached.
- **Cache layer** — store VLM outputs keyed by region hash so the expensive pass runs
  once per region, ever.

Sequencing note: 5b's LanceDB schema depends on the Section 3 decision (position vs
printed numbers) — pin that down before committing the schema.

---

## Suggested ordering (dependencies, not priorities)

1. **Section 3 pre-work audit** (diff production vs fixtures; decide number scheme) —
   unblocks both the parser cleanup target and the LanceDB schema.
2. **Section 1** parser fixes that are also page-mapping bugs (`opt`, `mackay`) — they
   improve production anchoring regardless of the Section 3 outcome.
3. **Section 4** doc reconciliation — cheap, and 4c becomes the spec for Section 3's
   shared module.
4. **Section 3** implementation (shared structural-rules module).
5. **Section 5a** text-only V2, then **5b** ingestion + VLM-on-read.
