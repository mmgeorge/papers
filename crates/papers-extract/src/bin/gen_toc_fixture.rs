//! Generate a TOC fixture (`- Title (p. N)` outline markdown) for a PDF.
//!
//! Usage: gen_toc_fixture <pdf_path> <output_md_path>
//!
//! Runs the same pipeline as the `toc_fixtures` test (parse the TOC page, else
//! fall back to font-detected headings) via `toc::render_fixture_markdown`, and
//! writes the result. Used to seed fixtures for newly added PDFs; the output is
//! a *draft* that should be spot-checked against the rendered TOC before being
//! trusted as ground truth.

use papers_extract::{pdf, toc};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: gen_toc_fixture <pdf_path> <output_md_path>");
        std::process::exit(1);
    }
    let pdf_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    let pdfium = pdf::load_pdfium(None).expect("Failed to load pdfium");
    let doc = pdfium
        .load_pdf_from_file(&pdf_path, None)
        .expect("Failed to load PDF");

    let total_pages = doc.pages().len() as u32;
    let page_chars: Vec<(Vec<pdf::PdfChar>, f32)> = (0..total_pages)
        .map(|i| {
            let page = doc.pages().get(i as u16).expect("Failed to get page");
            let height = page.height().value;
            let mut chars = pdf::extract_page_chars(&page, i).unwrap_or_default();
            pdf::normalize_chars_to_image_space(&mut chars, height);
            (chars, height)
        })
        .collect();

    let md = toc::render_fixture_markdown(&page_chars);

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }
    std::fs::write(&output_path, &md).expect("Failed to write fixture");

    let entry_count = md.lines().filter(|l| !l.trim().is_empty()).count();
    eprintln!(
        "Wrote {entry_count} TOC entries to {} ({total_pages} pages)",
        output_path.display()
    );
}
