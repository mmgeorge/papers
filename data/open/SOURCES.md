# Open-Licensed Textbook Corpus (`data/open/`)

Twenty freely-available textbooks/lecture-notes, downloaded as extra TOC-parser
test material. Each has a real printed table of contents. Unlike the PDFs in
`data/`, these are openly downloadable, so the corpus is reproducible — re-fetch
with `.temp/fetch_open.sh` (URLs below). The PDFs themselves are **not committed**
(they're large and gitignored); only the generated fixtures in
`crates/papers-extract/tests/fixtures/toc/open/*.md` are.

Licenses are the authors'/publishers' terms for the free PDF; verify before any
use beyond local testing. "free PDF" = author/publisher offers a no-cost download
for personal/educational use.

| stem | title (authors) | license | source |
|------|-----------------|---------|--------|
| `mml` | Mathematics for Machine Learning (Deisenroth, Faisal, Ong) | free PDF (CUP) | https://mml-book.github.io/book/mml-book.pdf |
| `sutton_barto_rl` | Reinforcement Learning: An Introduction, 2e (Sutton & Barto) | free PDF | http://incompleteideas.net/book/RLbook2020.pdf |
| `d2l` | Dive into Deep Learning (Zhang et al.) | CC BY-SA 4.0 / MIT | https://d2l.ai/d2l-en.pdf |
| `gpml` | Gaussian Processes for Machine Learning (Rasmussen & Williams) | free PDF (MIT Press) | https://gaussianprocess.org/gpml/chapters/RW.pdf |
| `understanding_ml` | Understanding Machine Learning (Shalev-Shwartz & Ben-David) | free PDF (CUP) | https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf |
| `mackay_itila` | Information Theory, Inference, and Learning Algorithms (MacKay) | free PDF (CUP) | https://www.inference.org.uk/itprnn/book.pdf |
| `jurafsky_slp3` | Speech and Language Processing, 3e draft (Jurafsky & Martin) | free draft | https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf |
| `foundations_data_science` | Foundations of Data Science (Blum, Hopcroft, Kannan) | free PDF (CUP) | https://www.cs.cornell.edu/jeh/book.pdf |
| `erickson_algorithms` | Algorithms (Jeff Erickson) | CC BY 4.0 | https://jeffe.cs.illinois.edu/teaching/algorithms/book/Algorithms-JeffE.pdf |
| `discrete_math` | Discrete Mathematics: An Open Introduction, 3e (Oscar Levin) | CC BY-SA 4.0 | http://discrete.openmathbooks.org/pdfs/dmoi3-tablet.pdf |
| `mit_math_for_cs` | Mathematics for Computer Science (Lehman, Leighton, Meyer) | CC BY-SA (MIT OCW) | https://courses.csail.mit.edu/6.042/spring18/mcs.pdf |
| `grinstead_probability` | Introduction to Probability (Grinstead & Snell) | GFDL | https://math.dartmouth.edu/~prob/prob/prob.pdf |
| `gelman_bda3` | Bayesian Data Analysis, 3e (Gelman et al.) | free PDF (non-commercial) | http://www.stat.columbia.edu/~gelman/book/BDA3.pdf |
| `vmls_linear_algebra` | Introduction to Applied Linear Algebra — VMLS (Boyd & Vandenberghe) | free PDF (CUP) | https://web.stanford.edu/~boyd/vmls/vmls.pdf |
| `eloquent_js` | Eloquent JavaScript, 3e (Haverbeke) | CC BY-NC 3.0 | https://eloquentjavascript.net/Eloquent_JavaScript.pdf |
| `pro_git` | Pro Git, 2e (Chacon & Straub) | CC BY-NC-SA 3.0 | https://github.com/progit/progit2/releases (progit.pdf) |
| `category_theory` | Category Theory for Programmers (Milewski; comp. Tabachnik) | free PDF | https://github.com/hmemcpy/milewski-ctfp-pdf/releases |
| `think_python` | Think Python, 2e (Downey) | CC BY-NC 3.0 | https://greenteapress.com/thinkpython2/thinkpython2.pdf |
| `think_stats` | Think Stats, 2e (Downey) | CC BY-NC 3.0 | https://greenteapress.com/thinkstats2/thinkstats2.pdf |
| `think_os` | Think OS (Downey) | CC BY-NC 3.0 | https://greenteapress.com/thinkos/thinkos.pdf |

## Fixture status

Fixtures here are **generated** by `gen_toc_fixture` (parser output) — regression
snapshots, not hand-verified ground truth like the `data/` set. Each was then
verified by rendering the book's printed TOC and diffing against the fixture
(2026-06-14). Verdicts: **9 clean, 9 minor, 2 broken.** A "clean" fixture is
trustworthy as ground truth; "minor"/"broken" encode known parser defects (the
test still passes because the fixture *is* the current output — treat the listed
issues as a TODO before trusting those entries).

| status | stem | key issue(s) |
|--------|------|--------------|
| clean | `d2l` | — faithful, 3-level nesting, all chapters/appendices |
| clean | `jurafsky_slp3` | — (2 page-bearing part dividers omitted) |
| clean | `foundations_data_science` | — |
| clean | `grinstead_probability` | — |
| clean | `vmls_linear_algebra` | — (3 page-bearing part dividers omitted) |
| clean | `pro_git` | — |
| clean | `think_python` | — |
| clean | `think_stats` | — |
| clean | `think_os` | — |
| minor | `mml` | spurious copyright-footer entry `©2024 … (p. 2020)` under Index; 2 page-bearing part dividers dropped |
| minor | `sutton_barto_rl` | ligature garble `ff`→`↵` (`Off`→`O↵`, `Efficiency`→`E ciency`), Greek σ/λ/γ dropped; 3 page-bearing part dividers dropped |
| minor | `understanding_ml` | `Preface page` (leader word merged); `Notes` mis-nested under Appendix C; ℓ→`` ` ``, ε→`ffl` math garble |
| minor | `erickson_algorithms` | Preface sub-entries flattened to top level; `Mātrāvr̥tta`→`Ma¯tra¯…` diacritic decomposition |
| minor | `discrete_math` | `0.3 Sets` subsections flattened to top level; `Venn Diagrams` got spurious `0 ` chapter prefix |
| minor | `mit_math_for_cs` | 5 page-bearing `Introduction` entries dropped → `0.1 References` orphaned |
| minor | `gelman_bda3` | one subsection dropped (`13.6`) |
| minor | `eloquent_js` | `Exercise Hints` sub-entries (18) flattened to top level |
| minor | `category_theory` | stray `2` glyph misplaced between `28.7`/`28.8` titles |
| **broken** | `gpml` | starred (∗) entries mis-nested to top level with stray `∗` + phantom chapter-number prefixes; ISBN/copyright watermark embedded in titles; copyright-footer entry |
| **broken** | `mackay_itila` | per-character doubling (`PPrroobbaabbiilliittyy`), garbage blocks from body pages, doubled page numbers (`2200`); the one correct TOC copy is buried mid-file |

### Recurring parser-bug classes surfaced (cross-cutting)

1. **Footer/copyright captured as a TOC entry** — `(p. 2020)`/`(p. 2006)` parsed
   from a `©…(year)` footer (mml, gpml). Needs a bottom-of-page/copyright filter.
2. **Page-bearing part/"Introduction" dividers dropped** — the part-divider drop
   fires even when the divider has its own page number (mml, sutton, jurafsky,
   vmls, mit_math_for_cs — where it orphans `0.1 References`). Same tension as the
   lambda/convex/engine_physics part conflict; here several books want them kept.
3. **Unnumbered sub-entry nesting flattened to top level** (erickson Preface,
   discrete_math `0.3 Sets`, eloquent_js Exercise Hints, gpml starred entries).
4. **Ligature / Greek / math-glyph garble** (sutton `ff`→`↵`, understanding_ml
   ε→`ffl`, ℓ→backtick) — some recoverable, some source `/ToUnicode` corruption.
5. **Per-character doubling** (mackay) — doubled text layer not deduplicated.
6. **Stray glyph misplacement** (category_theory `2`).
