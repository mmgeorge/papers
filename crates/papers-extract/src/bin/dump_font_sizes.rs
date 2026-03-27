//! Dump text segments grouped by font size for heading detection analysis.
//!
//! Usage: dump_font_sizes <pdf_path> <page_num> [page_num...]

use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dump_font_sizes <pdf_path> <page_num> [page_num...]");
        std::process::exit(1);
    }

    let pdf_path = PathBuf::from(&args[1]);
    let page_nums: Vec<u32> = args[2..]
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    let pdfium = papers_extract::pdf::load_pdfium(None).expect("Failed to load pdfium");
    let doc = pdfium
        .load_pdf_from_file(&pdf_path, None)
        .expect("Failed to load PDF");

    for &page_num in &page_nums {
        let page_idx = page_num - 1;
        let page = doc
            .pages()
            .get(page_idx as u16)
            .expect("Failed to get page");

        let mut chars =
            papers_extract::pdf::extract_page_chars(&page, page_idx).expect("extract_page_chars");
        papers_extract::pdf::normalize_chars_to_image_space(&mut chars, page.height().value);

        println!("==============================");
        println!("PAGE {} ({} chars)", page_num, chars.len());
        println!("==============================");

        // 1. Show every char with font size and font name
        println!("\n--- ALL CHARS WITH FONT INFO ---");
        println!(
            "{:>5} | {:>6} | {:>7} | {:>32} | {:>6} | text",
            "idx", "char", "size", "font", "italic"
        );
        println!("{}", "-".repeat(90));

        // Group consecutive chars with same font size + font name into segments
        let mut segments: Vec<(f32, String, bool, String, usize, usize)> = Vec::new(); // (size, font, italic, text, start, end)

        for (i, c) in chars.iter().enumerate() {
            let display = if c.codepoint.is_control() {
                format!("U+{:04X}", c.codepoint as u32)
            } else {
                format!("{}", c.codepoint)
            };

            // Quantize font size to 1 decimal place for grouping
            let size_q = (c.font_size * 10.0).round() / 10.0;

            if let Some(last) = segments.last_mut() {
                if (last.0 - size_q).abs() < 0.05 && last.1 == c.font_name && last.2 == c.is_italic
                {
                    last.3.push_str(&display);
                    last.4 = last.4; // start unchanged
                    last.5 = i; // end = current
                } else {
                    segments.push((size_q, c.font_name.clone(), c.is_italic, display, i, i));
                }
            } else {
                segments.push((size_q, c.font_name.clone(), c.is_italic, display, i, i));
            }
        }

        println!("\n--- TEXT SEGMENTS BY FONT ---");
        println!(
            "{:>7} | {:>6} | {:>32} | {:>5}-{:<5} | text",
            "size", "italic", "font", "start", "end"
        );
        println!("{}", "-".repeat(100));
        for seg in &segments {
            let text_preview = if seg.3.len() > 60 {
                format!("{}...", &seg.3[..57])
            } else {
                seg.3.clone()
            };
            println!(
                "{:>7.1} | {:>6} | {:>32} | {:>5}-{:<5} | {}",
                seg.0, seg.2, truncate(&seg.1, 32), seg.4, seg.5, text_preview
            );
        }

        // 2. Font size histogram
        let mut size_counts: BTreeMap<String, (usize, Vec<String>)> = BTreeMap::new();
        for seg in &segments {
            let key = format!("{:.1}", seg.0);
            let entry = size_counts.entry(key).or_insert((0, Vec::new()));
            entry.0 += seg.3.chars().count();
            if entry.1.len() < 5 {
                let preview = if seg.3.len() > 40 {
                    format!("{}...", &seg.3[..37])
                } else {
                    seg.3.clone()
                };
                entry.1.push(preview);
            }
        }

        println!("\n--- FONT SIZE HISTOGRAM ---");
        println!("{:>7} | {:>6} | examples", "size", "chars");
        println!("{}", "-".repeat(80));
        // Sort by size descending
        let mut sizes: Vec<_> = size_counts.into_iter().collect();
        sizes.sort_by(|a, b| {
            b.0.parse::<f32>()
                .unwrap_or(0.0)
                .partial_cmp(&a.0.parse::<f32>().unwrap_or(0.0))
                .unwrap()
        });
        for (size, (count, examples)) in &sizes {
            println!("{:>7} | {:>6} | {}", size, count, examples.join(" | "));
        }

        println!();
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{:>width$}", s, width = max_len)
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
