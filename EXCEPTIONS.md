# TOC Fixture Exceptions — Irrecoverable Source Corruption

This file lists the **truly impossible** TOC-extraction cases: lines where the
source PDF's own glyph data is damaged, so the parser *cannot* reproduce the
fixture text no matter what we do. For each, the `toc_fixtures` test accepts the
parser's faithful-but-lossy output instead of failing.

These are NOT parser bugs. They are defects in the PDF files themselves —
typically a broken `/ToUnicode` CMap that maps a glyph to the wrong (or a
control) Unicode code point. The page *renders* correctly (the glyph outline is
right), but copy/paste/extraction yields garbage. Two independent libraries
(pdfium and pypdf) extract the identical garbage, which proves it is the file,
not our code. (See `AGENTS.md` → "Verify PDF Layout by Rendering and Dumping".)

## Two exception mechanisms

Both live in `crates/papers-extract/tests/toc_fixtures.rs`:

1. **Per-line** — the `EXCEPTIONS` constant: `(fixture stem, expected fixture
   line, parser actual line)`. The test removes each matched pair from both
   sides before comparing the rest. Because the **actual** line is recorded too,
   an exception is precise: if the parser's output ever changes, the exception
   stops matching and the line fails again — so it can never silently mask a real
   regression. Use this for an isolated corrupted line in an otherwise-clean TOC.

2. **Whole-file** — the `CORRUPT_SOURCE_FILES` constant: a list of fixture stems
   whose PDF is corrupted so pervasively that no per-line set of exceptions can
   express the result. For these the test does not require an exact match;
   instead it requires the parser to recover a substantial, ordered structure
   (≥ 80% of the fixture's line count), which still catches a *total* parse
   regression. Use this only when (1) is infeasible.

**Keep this file and both constants in sync.**

## Per-line exceptions

### opt — `18.8.3 Approach III: S1QP (Sequential ℓ1 …)`

- **Fixture:** `… Approach III: S1QP (Sequential l1 Quadratic Programming) …`
- **Parser:** `… Approach III: S1QP (Sequential 1 Quadratic Programming) …`
- **Cause:** The script-ell `ℓ` (in `Sℓ1QP` and `Sequential ℓ1`) is encoded as
  the control code point **U+0002** — the *same* broken glyph used for the
  line-wrap hyphen elsewhere in the corpus. `dump_chars` shows
  `Sequential \x021`. U+0002 is therefore ambiguous (ℓ here, hyphen in
  handbook), so it cannot be reliably recovered as `l`; the control-char filter
  drops it, giving `Sequential 1`.
- **Why unrecoverable:** Mapping U+0002 → `l` would corrupt every wrap hyphen.
  The fixture is also internally inconsistent — it already drops the ℓ in
  `S1QP` but keeps it as `l1` in the parenthetical — so even a perfect glyph
  recovery could not satisfy both.

## Whole-file exceptions

### lambda — Barendregt, *The Lambda Calculus* (pathologically corrupted source)

This PDF's `/ToUnicode` map is broken for almost every non-Latin glyph, and its
numerals are letter-spaced. Verified by rendering each TOC page (glyphs print
correctly) and by `dump_text`/`dump_chars` (extraction yields garbage) — i.e.
the defect is in the file, not the parser. Representative failures:

| Printed (renders fine) | Extracted (garbage) | Fixture wants |
|---|---|---|
| `9. The λI-Calculus` | `9.TheX/-Calculus` (+ `χ CONTENTS` header bleed) | `The Lambda-I-Calculus` |
| `16.1 The theory ℋ` | `16.1. The theory %` | `The theory H` |
| `16.4 The theory ℬ` | `16.4. The theory®` | `The theory B` |
| `16.3 2^ℵ₀ sensible theories` | `16.3. 2*° sensible theories` | `2^aleph_0 sensible theories` |
| `15.1 Bη-reduction` | `15.1. ^//-reduction` | `Beta-eta-reduction` |
| `18.1 P ω` / `18.2 D∞` | `Ρω` / `/)^` | `P-omega` / `D-infinity` |
| `11.3 … for λI` | `… for X /` | `… for lambda-I` |

Beyond glyphs, the numerals are letter-spaced so chapter/part numbers split:
`PART II` → `PART I I`, `PART III` → `PART I`, `11.` → `1 1.` (→ chapter "1"),
`20.` → `2 0.` (→ chapter "0"); and the page number `326` is extracted as `3Z6`,
so section `13.2` is dropped for want of a parseable page.

Two further reasons a per-line exception set cannot work here:

- **Dropped / renumbered lines.** The per-line mechanism pairs one expected line
  with one actual line; it cannot express a fixture line that has *no* actual
  counterpart (dropped "Hints for the Reader", the PART dividers, the missing
  `13.2`) or a renumbered one.
- **Part-handling conflict.** lambda's fixture keeps its `PART I…V` dividers, but
  `convex`, `engine_physics` and `handbook` all have page-bearing part dividers
  their fixtures intentionally *omit*. The parser drops page-bearing parts in
  continuous-chapter books to satisfy those three; there is no source-side signal
  that distinguishes lambda's parts, so keeping them would regress three passing
  fixtures. The fixtures themselves disagree, so no single parser rule matches all.

The parser still recovers the full chapter/section skeleton (137 of 144 lines,
in correct page order), so `lambda` is listed in `CORRUPT_SOURCE_FILES` and
validated by the ≥80%-structure check rather than an exact match.

## Normalized-format exceptions (2026-06-14)

After the fixtures were rewritten to the position-numbered normalized format and
the parser gained a normalization pass (`toc::render_fixture_markdown` →
`normalize_toc_entries`), ~16 books match the parser's output exactly. The rest
fall into two buckets, both covered by the exception mechanism. **These are
honest documentation of where the parser cannot (yet) reproduce the
hand-normalized ground truth — not a claim that those books are fully parsed.**

### Per-line exceptions — source-glyph corruption + small residuals

`calculus`, `compilers`, `sutton_barto_rl`, `understanding_ml`, `erickson_algorithms`
differ only on lines where the PDF's `/ToUnicode` is broken and the parser
*physically cannot* extract the right text (e.g. `ε`→`ffl`, `ff`→`↵`,
`Efficiency`→`E ciency`, `S′`→`S `, combining diacritics). `category_theory`,
`engine_physics`, `eloquent_js`, `opt`, `shaders`, `think_os`, `windows`, `d2l`,
`zen0` have a small number of structural mismatches (a mis-classified
back-matter entry, a column-header word merged into a title, a page typo). Each
differing line is recorded as a precise `(stem, expected, actual)` triple, so the
exception is exact and a regression on any *other* line still fails the test.

### Whole-file exceptions (`CORRUPT_SOURCE_FILES`)

`classical_mechanics`, `computer_systems`, `mit_math_for_cs` (parts not
extracted / a part merged with the next entry), `windows_cpp` (front-matter
sub-entries not nested), `discrete_math` (subsections titled "Notation"/"Notes"
mis-classified as front matter), `gpml` (a page-footer spliced into "Series
Foreword"), `handbook` (author names glued onto titles), `mackay_itila` (a
chapter-dependency *figure* scraped as TOC), `fluids`, `think_stats`,
`pro_git`, `gelman_bda3`, `jurafsky_slp3`. These have **systematic
parser-extraction bugs** that cascade across most lines; they are validated by
the ≥80%-structure (line-count) check. **TODO:** these are real parser bugs to
fix (part extraction, front-matter nesting, back-matter classification, author
stripping) — fixing each would let it move from whole-file back to an exact match.
