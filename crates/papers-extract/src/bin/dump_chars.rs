//! Quick diagnostic: dump pdfium's raw text extraction for a PDF page.
//!
//! Shows both successfully mapped characters AND dropped characters
//! (where pdfium's `unicode_char()` returns None), along with font info.

use std::path::PathBuf;

use pdfium_render::prelude::*;

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

    // 2. Raw pdfium character iteration — shows ALL chars including dropped ones
    let chars_collection = text_obj.chars();
    let total_raw = chars_collection.len();
    let mut mapped_count = 0u32;
    let mut dropped_count = 0u32;

    println!("=== RAW PDFIUM CHARS ({total_raw} total) ===");
    println!();

    // -- 2a. Dropped characters (unicode_char() returns None) --
    println!("--- DROPPED CHARS (unicode_char() = None) ---");
    println!("  idx | raw_u32  | font_name                        | italic | symbolic | size   | left     | bottom   | right    | top");
    println!("------|----------|----------------------------------|--------|----------|--------|----------|----------|----------|--------");

    for i in 0..total_raw {
        let Ok(char_info) = chars_collection.get(i) else {
            continue;
        };

        if char_info.unicode_char().is_some() {
            mapped_count += 1;
            continue;
        }

        dropped_count += 1;

        let raw_u32 = char_info.unicode_value();
        let font_name = char_info.font_name();
        let is_italic = char_info.font_is_italic();
        let is_symbolic = char_info.font_is_symbolic();
        let font_size = char_info.scaled_font_size().value;

        let bounds_str = if let Ok(rect) = char_info.loose_bounds() {
            format!(
                "{:>8.2} | {:>8.2} | {:>8.2} | {:>8.2}",
                rect.left().value,
                rect.bottom().value,
                rect.right().value,
                rect.top().value
            )
        } else {
            "  (no bounds)                                ".to_string()
        };

        // Classify the raw value
        let surrogate_info = if (0xD800..=0xDBFF).contains(&raw_u32) {
            " HIGH-SURROGATE"
        } else if (0xDC00..=0xDFFF).contains(&raw_u32) {
            " LOW-SURROGATE"
        } else {
            ""
        };

        // Show context: find nearest mapped chars before and after
        let context_before = find_context_before(&chars_collection, i, 15);
        let context_after = find_context_after(&chars_collection, i, 15);

        println!(
            "{:>5} | U+{:04X}{:<15} | {:>32} | {:>6} | {:>8} | {:>6.1} | {}",
            i,
            raw_u32,
            surrogate_info,
            truncate(&font_name, 32),
            is_italic,
            is_symbolic,
            font_size,
            bounds_str
        );
        println!(
            "        context: \"{}[???]{}\"",
            context_before, context_after
        );
    }

    if dropped_count == 0 {
        println!("  No dropped chars found.");
    } else {
        // Reconstruct surrogate pairs to show actual codepoints
        println!();
        println!("--- SURROGATE PAIR RECONSTRUCTION ---");
        let mut j = 0usize;
        while j < total_raw {
            if let Ok(ci) = chars_collection.get(j) {
                let raw = ci.unicode_value();
                if (0xD800..=0xDBFF).contains(&raw) {
                    // High surrogate — look for low surrogate next
                    if j + 1 < total_raw {
                        if let Ok(ci2) = chars_collection.get(j + 1) {
                            let raw2 = ci2.unicode_value();
                            if (0xDC00..=0xDFFF).contains(&raw2) {
                                let codepoint =
                                    0x10000 + ((raw - 0xD800) << 10) + (raw2 - 0xDC00);
                                let decoded = char::from_u32(codepoint)
                                    .map(|c| format!("'{c}' (U+{codepoint:04X})"))
                                    .unwrap_or_else(|| format!("U+{codepoint:04X} (invalid)"));
                                let bounds = ci
                                    .loose_bounds()
                                    .map(|r| {
                                        format!(
                                            "[{:.2}, {:.2}, {:.2}, {:.2}]",
                                            r.left().value,
                                            r.bottom().value,
                                            r.right().value,
                                            r.top().value
                                        )
                                    })
                                    .unwrap_or_else(|_| "(no bounds)".to_string());
                                let context_before =
                                    find_context_before(&chars_collection, j, 10);
                                let context_after =
                                    find_context_after(&chars_collection, j + 1, 10);
                                println!(
                                    "  indices {},{}: U+{:04X} + U+{:04X} => {} font={} bbox={}",
                                    j,
                                    j + 1,
                                    raw,
                                    raw2,
                                    decoded,
                                    ci.font_name(),
                                    bounds
                                );
                                println!(
                                    "        context: \"{context_before}[{decoded}]{context_after}\""
                                );
                                j += 2;
                                continue;
                            }
                        }
                    }
                }
            }
            j += 1;
        }
    }

    println!();
    println!(
        "Summary: {mapped_count} mapped, {dropped_count} dropped, {total_raw} total raw"
    );
    println!();

    // 3. Dump per-character data (using extract_page_chars — filtered/mapped only)
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

    // 3b. Scan ALL chars for STX (U+0002) — pdfium's hyphen marker
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
            // Show detailed bbox + space_threshold for chars around the STX
            let detail_start = i.saturating_sub(3);
            let detail_end = (i + 4).min(chars.len());
            for k in detail_start..detail_end {
                let ch = &chars[k];
                let dc = if ch.codepoint.is_control() {
                    format!("U+{:04X}", ch.codepoint as u32)
                } else {
                    format!("'{}'", ch.codepoint)
                };
                let w = ch.bbox[2] - ch.bbox[0];
                let gap = if k > detail_start {
                    ch.bbox[0] - chars[k - 1].bbox[2]
                } else {
                    0.0
                };
                println!(
                    "    [{:>4}] {:>6} left={:.2} right={:.2} bottom={:.2} top={:.2} w={:.2} gap={:.2} space_thr={:.2}",
                    k, dc, ch.bbox[0], ch.bbox[2], ch.bbox[1], ch.bbox[3], w, gap, ch.space_threshold
                );
            }
        }
    }
    if stx_count == 0 {
        println!("  No STX markers found.");
    } else {
        println!("  Total: {} STX markers", stx_count);
    }

    // 3c. Also scan for control chars in general
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

    // 4. Show gap analysis for first line
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

/// Find context text BEFORE a given index by scanning backwards for mapped chars.
fn find_context_before(chars: &PdfPageTextChars, idx: usize, max: usize) -> String {
    let mut result = Vec::new();
    let mut i = idx;
    while i > 0 && result.len() < max {
        i -= 1;
        if let Ok(ci) = chars.get(i) {
            if let Some(c) = ci.unicode_char() {
                if c.is_control() {
                    continue;
                }
                result.push(c);
            }
        }
    }
    result.reverse();
    result.into_iter().collect()
}

/// Find context text AFTER a given index by scanning forwards for mapped chars.
fn find_context_after(chars: &PdfPageTextChars, idx: usize, max: usize) -> String {
    let mut result = Vec::new();
    let mut i = idx + 1;
    let len = chars.len();
    while i < len && result.len() < max {
        if let Ok(ci) = chars.get(i) {
            if let Some(c) = ci.unicode_char() {
                if c.is_control() {
                    i += 1;
                    continue;
                }
                result.push(c);
            }
        }
        i += 1;
    }
    result.into_iter().collect()
}

/// Truncate a string to at most `max_len` chars, padding with spaces if shorter.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{:>width$}", s, width = max_len)
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
