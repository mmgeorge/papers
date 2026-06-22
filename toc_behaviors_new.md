# TOC Behaviors — Action Items

New behaviors to implement. Each item references the relevant section of
`toc_behaviors.md` that will be updated once the behavior is in place.

---

## 1. Roman-Numeral Chapter Groupings — Two Cases (updates §12)

Two distinct behaviors depending on whether the arabic-numbered entries between
roman entries are **continuously numbered** or **restart from 1** per roman entry.

---

### Case A — Redundant grouping: Drop (updates §12, Case A)

**Current behavior:** Roman-numeral entries are *promoted* to `Part` (depth 0).

**Desired behavior:** Drop those entries entirely.

**Detection condition:**
- ≥ 2 entries classified as Chapter whose titles start with a roman numeral.
- The arabic-numbered chapters between consecutive roman entries are numbered
  **continuously** across roman entries (do NOT restart from 1 at each roman entry).

**Action:** In `promote_roman_to_parts` (toc.rs), instead of reclassifying entries
to `Part`, *remove* them from the entry list. Use `retain` to drop the roman entries.

**Fixture:** `tests/fixtures/toc/convex.md` — "I Theory", "II Applications",
"III Algorithms" entries removed; arabic chapters 1–11 remain at depth 1.

---

### Case B — Roman entries ARE the chapters: Keep and re-number (updates §12, Case B)

**Current behavior:** Roman entries are promoted to `Part` (depth 0); arabic entries
remain at depth 1.

**Desired behavior:** Roman entries become depth-1 chapters. Arabic-numbered entries
are demoted to depth-2 sections. All section/subsection numbers are prefixed with the
roman chapter identifier: `I.1`, `I.2`, `I.1.1`, etc.

**Detection condition:**
- ≥ 2 entries classified as Chapter whose titles start with a roman numeral.
- The arabic-numbered entries **restart from 1** under each roman entry (each
  roman chapter contains its own ch 1, ch 2, … independently).
- Each roman entry has its own page number.

**Action:** In `promote_roman_to_parts` (toc.rs), detect the restart pattern and
instead of dropping or promoting the roman entries, keep them at depth 1 (title
unchanged). Demote their children to depth 2 and re-number all section/subsection
labels using the roman chapter as prefix: `I.1`, `I.2`, `I.1.1`, etc.

**Fixture:** `tests/fixtures/toc/zen0.md` — `I Geometry Manipulation` at depth 1;
`I.1 Attributed Vertex Clouds`, `I.1.1 Introduction`, `II Lighting`,
`II.1 Stable Indirect Illumination`, etc.
