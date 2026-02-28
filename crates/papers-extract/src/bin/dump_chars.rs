//! Quick diagnostic: dump pdfium's raw text extraction for a PDF page.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dump_chars <pdf_path> [page_num]");
        std::process::exit(1);
    }

    let pdf_path = PathBuf::from(&args[1]);
    let page_num: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let page_idx = page_num - 1;

    let pdfium = papers_extract::pdf::load_pdfium(None).expect("Failed to load pdfium");
    let doc = pdfium
        .load_pdf_from_file(&pdf_path, None)
        .expect("Failed to load PDF");

    let page = doc
        .pages()
        .get(page_idx as u16)
        .expect("Failed to get page");

    // 1. Dump pdfium's full-page text (the text layer as a single string)
    let text_obj = page.text().expect("Failed to get text layer");
    let full_text = text_obj.all();
    println!("=== FULL PAGE TEXT (pdfium text layer) ===");
    println!("{full_text}");
    println!("\n=== END FULL PAGE TEXT ===\n");

    // 2. Dump per-character data
    let chars = papers_extract::pdf::extract_page_chars(&page, page_idx)
        .expect("Failed to extract chars");

    println!("=== PER-CHARACTER DATA ({} chars) ===", chars.len());
    println!("char | left     | bottom   | right    | top      | width");
    println!("-----|----------|----------|----------|----------|------");

    // Show first 200 chars with bbox details
    for (i, c) in chars.iter().take(200).enumerate() {
        let w = c.bbox[2] - c.bbox[0];
        let display_char = if c.codepoint.is_control() {
            format!("U+{:04X}", c.codepoint as u32)
        } else {
            format!("'{}'", c.codepoint)
        };
        println!(
            "{:>4} {:>6} | {:>8.2} | {:>8.2} | {:>8.2} | {:>8.2} | {:>5.2}",
            i, display_char, c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3], w
        );
    }

    if chars.len() > 200 {
        println!("... ({} more chars)", chars.len() - 200);
    }

    // 2b. Scan ALL chars for STX (U+0002) — pdfium's hyphen marker
    println!("\n=== STX (U+0002) HYPHEN MARKER SCAN ===");
    let mut stx_count = 0;
    for (i, c) in chars.iter().enumerate() {
        if c.codepoint == '\u{0002}' {
            stx_count += 1;
            // Show context: 10 chars before and 10 after
            let start = i.saturating_sub(10);
            let end = (i + 11).min(chars.len());
            let context: String = chars[start..end]
                .iter()
                .enumerate()
                .map(|(j, ch)| {
                    let abs = start + j;
                    if abs == i {
                        '|'  // mark the STX position
                    } else if ch.codepoint.is_control() {
                        '?'
                    } else {
                        ch.codepoint
                    }
                })
                .collect();
            println!(
                "  STX at char {}: bbox=[{:.2}, {:.2}, {:.2}, {:.2}] context: \"{}\"",
                i, c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3], context
            );
        }
    }
    if stx_count == 0 {
        println!("  No STX markers found.");
    } else {
        println!("  Total: {} STX markers", stx_count);
    }

    // 2c. Also scan for control chars in general
    println!("\n=== ALL CONTROL CHARS ===");
    let mut ctrl_count = 0;
    for (i, c) in chars.iter().enumerate() {
        if c.codepoint.is_control() {
            ctrl_count += 1;
            let start = i.saturating_sub(5);
            let end = (i + 6).min(chars.len());
            let context: String = chars[start..end]
                .iter()
                .enumerate()
                .map(|(j, ch)| {
                    let abs = start + j;
                    if abs == i {
                        '|'
                    } else if ch.codepoint.is_control() {
                        '?'
                    } else {
                        ch.codepoint
                    }
                })
                .collect();
            println!(
                "  U+{:04X} at char {}: bbox=[{:.2}, {:.2}, {:.2}, {:.2}] context: \"{}\"",
                c.codepoint as u32, i, c.bbox[0], c.bbox[1], c.bbox[2], c.bbox[3], context
            );
        }
    }
    if ctrl_count == 0 {
        println!("  No control chars found.");
    } else {
        println!("  Total: {} control chars", ctrl_count);
    }

    // 3. Show gap analysis for first line
    println!("\n=== GAP ANALYSIS (first ~80 chars) ===");
    let first_line_chars: Vec<_> = chars.iter().take(80).collect();
    if first_line_chars.len() > 1 {
        let widths: Vec<f32> = first_line_chars.iter().map(|c| c.bbox[2] - c.bbox[0]).collect();
        let avg_width = widths.iter().sum::<f32>() / widths.len() as f32;
        let threshold = avg_width * 0.3;
        println!("avg char width: {avg_width:.2}, word gap threshold (0.3x): {threshold:.2}");
        println!();

        for i in 1..first_line_chars.len() {
            let gap = first_line_chars[i].bbox[0] - first_line_chars[i - 1].bbox[2];
            let c = first_line_chars[i];
            let display_char = if c.codepoint.is_control() {
                format!("U+{:04X}", c.codepoint as u32)
            } else {
                format!("'{}'", c.codepoint)
            };
            let marker = if gap > threshold { " <-- SPACE" } else { "" };
            println!(
                "  {} gap={:.2}{marker}",
                display_char, gap
            );
        }
    }
}
