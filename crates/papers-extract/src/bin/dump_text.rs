//! Dump pdfium's raw text extraction for an entire PDF.
//!
//! Usage: dump_text <pdf_path> <output_path>
//!
//! Writes each page's text (via `page.text().all()`) to a single .txt file,
//! separated by page markers.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dump_text <pdf_path> <output_path>");
        std::process::exit(1);
    }

    let pdf_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    let pdfium = papers_extract::pdf::load_pdfium(None).expect("Failed to load pdfium");
    let doc = pdfium
        .load_pdf_from_file(&pdf_path, None)
        .expect("Failed to load PDF");

    let total_pages = doc.pages().len();
    let mut output = String::new();

    for i in 0..total_pages {
        let page = doc.pages().get(i).expect("Failed to get page");
        let text_obj = page.text().expect("Failed to get text layer");
        let full_text = text_obj.all();

        output.push_str(&format!("══════ Page {} ══════\n", i + 1));
        output.push_str(&full_text);
        output.push_str("\n\n");
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }
    std::fs::write(&output_path, &output).expect("Failed to write output");

    eprintln!(
        "Wrote {} pages to {}",
        total_pages,
        output_path.display()
    );
}
