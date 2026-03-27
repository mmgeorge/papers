//! Diagnostic: find spurious space insertions by scanning all pages for
//! locations where our gap-based threshold inserts a space within a word.

use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: find_gaps <pdf_path> <search_fragment>");
        eprintln!("  Searches all pages for chars matching the fragment,");
        eprintln!("  and shows gap/threshold data at each occurrence.");
        std::process::exit(1);
    }

    let pdf_path = PathBuf::from(&args[1]);
    let search = &args[2];

    let pdfium = papers_extract::pdf::load_pdfium(None).expect("Failed to load pdfium");
    let doc = pdfium
        .load_pdf_from_file(&pdf_path, None)
        .expect("Failed to load PDF");

    let num_pages = doc.pages().len();
    println!("Scanning {} pages for \"{}\"...\n", num_pages, search);

    for page_idx in 0..num_pages {
        let page = doc.pages().get(page_idx).expect("Failed to get page");
        let mut chars = papers_extract::pdf::extract_page_chars(&page, page_idx as u32)
            .expect("Failed to extract chars");
        papers_extract::pdf::normalize_chars_to_image_space(&mut chars, page.height().value);

        // Build the string of codepoints (filtering control chars)
        let codepoints: Vec<char> = chars.iter().map(|c| c.codepoint).collect();
        let text: String = codepoints.iter().filter(|c| !c.is_control()).collect();

        // Search for the fragment in the filtered text
        let mut search_from = 0;
        while let Some(pos) = text[search_from..].find(search) {
            let abs_pos = search_from + pos;

            // Map byte offset in filtered text back to chars index.
            // Count chars (not bytes) up to abs_pos.
            let char_pos = text[..abs_pos].chars().count();
            let mut char_idx = 0;
            let mut filtered_count = 0;
            for (i, c) in chars.iter().enumerate() {
                if !c.codepoint.is_control() {
                    if filtered_count == char_pos {
                        char_idx = i;
                        break;
                    }
                    filtered_count += 1;
                }
            }

            println!("=== Page {} (0-indexed), char index ~{} ===", page_idx, char_idx);

            // Show chars around the match with gap/threshold data
            let context_before = 10;
            let context_after = search.len() + 10;
            let start = char_idx.saturating_sub(context_before);
            let end = (char_idx + context_after).min(chars.len());

            for k in start..end {
                let ch = &chars[k];
                let dc = if ch.codepoint.is_control() {
                    format!("U+{:04X}", ch.codepoint as u32)
                } else {
                    format!("'{}'", ch.codepoint)
                };
                let w = ch.bbox[2] - ch.bbox[0];
                let gap = if k > start {
                    ch.bbox[0] - chars[k - 1].bbox[2]
                } else {
                    0.0
                };
                let exceeds = if k > start && !ch.codepoint.is_control() && ch.codepoint != ' ' {
                    gap >= ch.space_threshold && ch.space_threshold > 0.0
                } else {
                    false
                };
                let marker = if exceeds { " <-- SPACE INSERTED" } else { "" };
                println!(
                    "  [{:>4}] {:>6} left={:.2} right={:.2} bot={:.2} top={:.2} w={:.2} gap={:.2} thr={:.2} sz={:.1} font={}{}",
                    k, dc, ch.bbox[0], ch.bbox[2], ch.bbox[1], ch.bbox[3], w, gap, ch.space_threshold, ch.font_size, ch.font_name, marker
                );
            }
            println!();

            // Advance past the match start by one full character (may be multi-byte)
            search_from = abs_pos + text[abs_pos..].chars().next().map_or(1, |c| c.len_utf8());
        }
    }
}
