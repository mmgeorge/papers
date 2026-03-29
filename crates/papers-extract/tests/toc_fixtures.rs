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

        if actual != expected {
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
        if e != a {
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
