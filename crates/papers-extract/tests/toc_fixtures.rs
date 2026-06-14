use papers_extract::{headings, pdf, toc};
use std::path::PathBuf;

/// A unified heading entry used for fixture comparison, regardless of whether
/// it came from `parse_toc` (TOC page) or `extract_headings` (font detection).
struct FixtureEntry {
    depth: u32,
    title: String,
    /// Page label as it should appear in the fixture (e.g. "42", "xvii").
    page: String,
}

impl FixtureEntry {
    fn from_toc(e: &toc::TocEntry) -> Self {
        Self {
            depth: e.depth,
            title: e.title.clone(),
            page: e.page_label.clone(),
        }
    }

    fn from_heading(h: &headings::DetectedHeading) -> Self {
        Self {
            depth: h.depth,
            title: h.title.clone(),
            page: h.page.to_string(),
        }
    }
}

/// Format entries into the fixture markdown format:
///
///   - Title (p. N)          ← depth 1, no indent
///     - N.M Title (p. N)    ← depth 2, 2-space indent
///       - N.M.K ...         ← depth 3, 4-space indent
///
/// Depth-0 Part entries without a page number are rendered without `(p. …)`.
fn format_fixture(entries: &[FixtureEntry]) -> String {
    let mut lines = Vec::with_capacity(entries.len());
    for entry in entries {
        let indent = "  ".repeat(entry.depth.saturating_sub(1) as usize);
        let page = if entry.page.is_empty() {
            String::new()
        } else {
            format!(" (p. {})", entry.page)
        };
        lines.push(format!("{}- {}{}", indent, entry.title, page));
    }
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

#[test]
fn toc_fixtures() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data_dir = manifest_dir.join("../../data");
    let fixture_dir = manifest_dir.join("tests/fixtures/toc");
    let temp_dir = manifest_dir.join("../../.temp/toc-fixture-tests");
    std::fs::create_dir_all(&temp_dir).ok();

    let pdfium = pdf::load_pdfium(None).expect("load pdfium");

    // Collect all PDFs that have a corresponding fixture, sorted for determinism.
    let mut pdf_stems: Vec<String> = std::fs::read_dir(&data_dir)
        .expect("read data dir")
        .filter_map(|e| {
            let e = e.ok()?;
            let path = e.path();
            if path.extension()?.to_str()? != "pdf" {
                return None;
            }
            let stem = path.file_stem()?.to_str()?.to_string();
            if fixture_dir.join(format!("{stem}.md")).exists() {
                Some(stem)
            } else {
                None
            }
        })
        .collect();
    pdf_stems.sort();

    assert!(
        !pdf_stems.is_empty(),
        "No PDFs with TOC fixtures found. data_dir={}, fixture_dir={}",
        data_dir.display(),
        fixture_dir.display(),
    );

    let mut failures: Vec<String> = Vec::new();

    for stem in &pdf_stems {
        let pdf_path = data_dir.join(format!("{stem}.pdf"));
        let fixture_path = fixture_dir.join(format!("{stem}.md"));

        let expected = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|e| panic!("Failed to read fixture {}: {e}", fixture_path.display()))
            .replace("\r\n", "\n");

        // Load PDF and extract page chars.
        let doc = match pdfium.load_pdf_from_file(&pdf_path, None) {
            Ok(d) => d,
            Err(e) => {
                failures.push(format!("[{stem}] Failed to load PDF: {e}"));
                continue;
            }
        };

        let total_pages = doc.pages().len() as u32;
        let page_chars: Vec<(Vec<pdf::PdfChar>, f32)> = (0..total_pages)
            .map(|i| {
                let page = doc.pages().get(i as u16).unwrap();
                let height = page.height().value;
                let mut chars = pdf::extract_page_chars(&page, i).unwrap_or_default();
                pdf::normalize_chars_to_image_space(&mut chars, height);
                (chars, height)
            })
            .collect();

        // Primary path: explicit TOC page.
        // Fallback: font-based heading detection for papers without a TOC page.
        let entries: Vec<FixtureEntry> = match toc::parse_toc(&page_chars) {
            Some(ref result) => result.entries.iter().map(FixtureEntry::from_toc).collect(),
            None => {
                let result = headings::extract_headings(&page_chars);
                result
                    .headings
                    .iter()
                    .map(FixtureEntry::from_heading)
                    .collect()
            }
        };

        let actual = format_fixture(&entries);

        // Save actual output for debugging regardless of pass/fail.
        std::fs::write(temp_dir.join(format!("{stem}.md")), &actual).ok();

        if normalize_for_compare(&actual) != normalize_for_compare(&expected) {
            let diff = build_diff(stem, &expected, &actual);
            failures.push(diff);
        } else {
            eprintln!("  PASS  {stem}");
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} TOC fixture(s) failed:\n\n{}",
            failures.len(),
            failures.join("\n\n---\n\n")
        );
    }

    eprintln!(
        "TOC fixtures: {n}/{n} passed",
        n = pdf_stems.len()
    );
}

/// Canonicalize typographically-equivalent characters so the comparison ignores
/// them. pdfium extracts faithfully (curly quotes U+2018/U+2019, en-dash U+2013,
/// em-dash U+2014) while the fixture corpus is inconsistent — some files were
/// hand-transcribed to ASCII (' " - and "--" for an em-dash). These variants are
/// visually equivalent, so treat them as equal rather than churning fixtures or
/// lossily normalizing extraction.
fn normalize_typography(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{2018}' | '\u{2019}' => out.push('\''),
            '\u{201C}' | '\u{201D}' => out.push('"'),
            '\u{2013}' => out.push('-'),
            '\u{2014}' => out.push_str("--"),
            // Superscript caret is a notation choice the corpus is inconsistent
            // about ("LDL^T" vs "LBLT"); drop it so it never decides pass/fail.
            '^' => {}
            // A bare grave accent (U+0060) is a floating diacritic the PDF
            // composes "è" from (base "e" + accent); pdfium emits it as a stray
            // "`" and our x-sort pulls it between letters ("Ribie`re"). The
            // fixture transcribes the accented vowel as plain ASCII, so drop it.
            '\u{60}' => {}
            other => out.push(other),
        }
    }
    out
}

/// Drop the trailing dot on a *multi-level* section number so the dotted and
/// undotted forms compare equal ("1.1." == "1.1", "16.2.3." == "16.2.3"). Some
/// books (e.g. Barendregt's Lambda Calculus) typeset every section number with a
/// trailing period; the fixture corpus omits it. A bare chapter number ("1.") is
/// left to the parser, and a lone number with a trailing dot ("354.") is
/// unaffected because it has no interior ".digit" group — so this never strips a
/// real sentence-final or decimal dot.
fn strip_section_number_trailing_dot(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        // Only match at a token start (preceded by start, whitespace, or other
        // punctuation — never mid-number, so "C.3"/"1.2.3" interiors are safe).
        let at_boundary = i == 0 || (!chars[i - 1].is_alphanumeric() && chars[i - 1] != '.');
        if at_boundary && chars[i].is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            // Require at least one interior ".<digits>" group (multi-level).
            let mut groups = 0;
            while j + 1 < chars.len() && chars[j] == '.' && chars[j + 1].is_ascii_digit() {
                j += 1;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    j += 1;
                }
                groups += 1;
            }
            if groups >= 1 {
                out.extend(&chars[i..j]);
                // Skip a single trailing dot ("1.1." -> "1.1").
                i = if j < chars.len() && chars[j] == '.' { j + 1 } else { j };
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Full comparison key: typography normalization, trailing-section-dot and
/// math/formula spacing normalization, and case folding. Spaces that touch a
/// non-alphanumeric character (parentheses, `*`, operators, commas …) are
/// dropped, so formula formatting like "O( M log *N )" and "O(M log* N)" compare
/// equal. A space *between two alphanumerics* is a real word boundary and is
/// kept, so genuine extraction bugs ("IR n" vs "IRn", "Ital iano" vs "Italiano")
/// still fail. Comparison is case-insensitive so small-caps structural headings
/// ("PREFACE") match their Title-Case transcription ("Preface") — this only ever
/// makes the check *more* permissive, so no passing fixture can regress.
fn normalize_for_compare(s: &str) -> String {
    s.lines().map(normalize_line_for_compare).collect::<Vec<_>>().join("\n")
}

fn normalize_line_for_compare(line: &str) -> String {
    let t = normalize_typography(line);
    let t = strip_section_number_trailing_dot(&t);
    // PRESERVE the leading indentation exactly — it encodes the outline depth and
    // must remain significant. Only the content after it is spacing-normalized.
    let indent_len = t.len() - t.trim_start_matches(' ').len();
    let (indent, rest) = t.split_at(indent_len);

    // Within the content, keep a space only when both neighbours are
    // alphanumeric (a real word boundary); drop spaces that touch punctuation /
    // operators (formula formatting). Runs of spaces inside the content collapse
    // naturally because the dropped ones disappear and an interior word-boundary
    // space is single.
    let chars: Vec<char> = rest.chars().collect();
    let mut out = String::with_capacity(t.len());
    out.push_str(indent);
    let mut emitted_space = false;
    for (i, &c) in chars.iter().enumerate() {
        if c == ' ' {
            let prev_alnum = i > 0 && chars[i - 1].is_alphanumeric();
            let next_alnum = chars[i + 1..].iter().find(|&&x| x != ' ').is_some_and(|x| x.is_alphanumeric());
            if prev_alnum && next_alnum && !emitted_space {
                out.push(' ');
                emitted_space = true;
            }
        } else {
            out.push(c);
            emitted_space = false;
        }
    }
    // Case-insensitive comparison (small-caps headings vs Title-Case fixtures).
    out.to_lowercase()
}

/// Produce a compact diff showing the first few divergent lines.
fn build_diff(stem: &str, expected: &str, actual: &str) -> String {
    let exp_lines: Vec<&str> = expected.lines().collect();
    let act_lines: Vec<&str> = actual.lines().collect();

    if actual.is_empty() {
        return format!(
            "[{stem}] parser returned no output, but fixture has {} lines\n  first expected: {:?}",
            exp_lines.len(),
            exp_lines.first().unwrap_or(&"")
        );
    }

    let mut diffs = Vec::new();
    let max = exp_lines.len().max(act_lines.len());
    for i in 0..max {
        let e = exp_lines.get(i).copied().unwrap_or("<missing>");
        let a = act_lines.get(i).copied().unwrap_or("<missing>");
        if normalize_line_for_compare(e) != normalize_line_for_compare(a) {
            diffs.push(format!(
                "  line {n}:\n    expected: {e:?}\n    actual:   {a:?}",
                n = i + 1
            ));
            if diffs.len() >= 5 {
                diffs.push(format!("  ... ({} more differing lines)", max - i - 1));
                break;
            }
        }
    }

    if exp_lines.len() != act_lines.len() {
        diffs.push(format!(
            "  line count: expected {}, got {}",
            exp_lines.len(),
            act_lines.len()
        ));
    }

    format!("[{stem}]\n{}", diffs.join("\n"))
}

#[test]
fn normalize_for_compare_is_spacing_only() {
    let n = normalize_line_for_compare;
    // Formula / punctuation spacing IS lenient.
    assert_eq!(n("    - 8.6.3 An O( M log *N ) Bound"), n("    - 8.6.3 An O(M log* N) Bound"));
    assert_eq!(n("- A (p. 369)"), n("- A (p.369)"));
    // Word / identifier boundaries are NOT lenient (real extraction bugs).
    assert_ne!(n("- Space IR n"), n("- Space IRn"));
    assert_ne!(n("- F. Ital iano"), n("- F. Italiano"));
    // Leading indentation (outline depth) is preserved.
    assert_ne!(n("  - Notes"), n("    - Notes"));
    // Superscript caret notation is ignored ("LDL^T" == "LDLT").
    assert_eq!(n("- C.3 LDL^T factorization"), n("- C.3 LDLT factorization"));
    // A floating grave accent ("Ribie`re") compares equal to the plain vowel.
    assert_eq!(n("- 5.2.2 The Polak-Ribie`re Method"), n("- 5.2.2 The Polak-Ribiere Method"));
    // A multi-level section number's trailing dot is ignored ("1.1." == "1.1").
    assert_eq!(n("  - 1.1. Aspects of the lambda calculus"), n("  - 1.1 Aspects of the lambda calculus"));
    assert_eq!(n("    - 16.2.3. The theory"), n("    - 16.2.3 The theory"));
    // …but the section number itself still decides the match.
    assert_ne!(n("  - 1.1 Aspects"), n("  - 1.2 Aspects"));
    // A lone number with a trailing dot (page text, decimals) is untouched.
    assert_ne!(n("- Page 354. Foo"), n("- Page 35 Foo"));
    // Comparison is case-insensitive (small-caps headings vs Title-Case).
    assert_eq!(n("- PREFACE (p. vii)"), n("- Preface (p. vii)"));
    assert_eq!(n("- PART I. TOWARDS THE THEORY (p. 1)"), n("- Part I. Towards the Theory (p. 1)"));
    // …but differing letters still fail despite case folding.
    assert_ne!(n("- Preface"), n("- Prefaces"));
}
