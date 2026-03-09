//! Measure actual rendered glyph heights (bbox top - bottom) for heading detection.
//!
//! Usage: dump_glyph_heights <pdf_path> <page_num> [page_num...]

use std::collections::BTreeMap;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: dump_glyph_heights <pdf_path> <page_num> [page_num...]");
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

        let chars =
            papers_extract::pdf::extract_page_chars(&page, page_idx).expect("extract_page_chars");

        println!("==============================");
        println!("PAGE {} ({} chars)", page_num, chars.len());
        println!("==============================");

        // Group consecutive chars with same font into segments, measuring actual glyph heights
        struct Segment {
            font_name: String,
            font_size: f32,
            is_italic: bool,
            text: String,
            // Actual rendered measurements from bbox
            heights: Vec<f32>,     // per-char height (top - bottom)
            baselines: Vec<f32>,   // per-char bottom (baseline proxy)
            tops: Vec<f32>,        // per-char top
        }

        let mut segments: Vec<Segment> = Vec::new();

        for c in chars.iter() {
            if c.codepoint.is_control() {
                continue;
            }

            let height = c.bbox[3] - c.bbox[1]; // top - bottom in PDF coords
            let bottom = c.bbox[1];
            let top = c.bbox[3];
            let size_q = (c.font_size * 10.0).round() / 10.0;

            let display = format!("{}", c.codepoint);

            if let Some(last) = segments.last_mut() {
                if (last.font_size - size_q).abs() < 0.05
                    && last.font_name == c.font_name
                    && last.is_italic == c.is_italic
                {
                    last.text.push_str(&display);
                    last.heights.push(height);
                    last.baselines.push(bottom);
                    last.tops.push(top);
                    continue;
                }
            }

            segments.push(Segment {
                font_name: c.font_name.clone(),
                font_size: size_q,
                is_italic: c.is_italic,
                text: display,
                heights: vec![height],
                baselines: vec![bottom],
                tops: vec![top],
            });
        }

        // Print segments with actual glyph measurements
        println!(
            "{:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>32} | text",
            "em_size", "med_h", "max_h", "med_bl", "med_top", "font"
        );
        println!("{}", "-".repeat(110));

        for seg in &segments {
            if seg.text.trim().is_empty() {
                continue;
            }

            let mut heights = seg.heights.clone();
            heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_h = heights[heights.len() / 2];
            let max_h = heights.last().copied().unwrap_or(0.0);

            let mut baselines = seg.baselines.clone();
            baselines.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_bl = baselines[baselines.len() / 2];

            let mut tops = seg.tops.clone();
            tops.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med_top = tops[tops.len() / 2];

            let text_preview = if seg.text.len() > 50 {
                format!("{}...", &seg.text[..47])
            } else {
                seg.text.clone()
            };

            println!(
                "{:>7.1} | {:>7.2} | {:>7.2} | {:>7.2} | {:>7.2} | {:>32} | {}",
                seg.font_size,
                med_h,
                max_h,
                med_bl,
                med_top,
                truncate(&seg.font_name, 32),
                text_preview
            );
        }

        // Now bucket ALL non-control chars by rendered height
        println!("\n--- GLYPH HEIGHT BUCKETS (0.5pt bins) ---");

        // Bucket by height rounded to nearest 0.5
        let mut buckets: BTreeMap<i32, (usize, Vec<(char, f32, String)>)> = BTreeMap::new();
        for c in chars.iter() {
            if c.codepoint.is_control() || c.codepoint == ' ' {
                continue;
            }
            let height = c.bbox[3] - c.bbox[1];
            let bucket_key = (height * 2.0).round() as i32; // 0.5pt bins
            let entry = buckets.entry(bucket_key).or_insert((0, Vec::new()));
            entry.0 += 1;
            if entry.1.len() < 3 {
                entry.1.push((c.codepoint, c.font_size, c.font_name.clone()));
            }
        }

        println!("{:>8} | {:>6} | examples", "height", "count");
        println!("{}", "-".repeat(80));
        let mut bucket_vec: Vec<_> = buckets.into_iter().collect();
        bucket_vec.sort_by(|a, b| b.0.cmp(&a.0)); // descending by height
        for (key, (count, examples)) in &bucket_vec {
            let height = *key as f32 / 2.0;
            let ex_str: Vec<String> = examples
                .iter()
                .map(|(ch, sz, font)| format!("'{}' sz={:.1} {}", ch, sz, font))
                .collect();
            println!("{:>8.1} | {:>6} | {}", height, count, ex_str.join(" | "));
        }

        // Per-char detail for the first ~30 chars of each unique font-size combo
        println!("\n--- CHAR-LEVEL DETAIL (first chars of each font group) ---");
        println!(
            "{:>5} | {:>4} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | font",
            "idx", "char", "em_size", "height", "bottom", "top", "width"
        );
        println!("{}", "-".repeat(90));

        // Show all chars for heading-like segments (non-body fonts or large sizes)
        let mut shown_groups: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (i, c) in chars.iter().enumerate() {
            if c.codepoint.is_control() {
                continue;
            }
            let group_key = format!("{:.1}_{}", c.font_size, c.font_name);
            let is_new_group = shown_groups.insert(group_key.clone());
            let is_heading_font = c.font_name.contains("OpenSans");
            let is_large = c.font_size > 10.5;

            if is_new_group || is_heading_font || is_large {
                let height = c.bbox[3] - c.bbox[1];
                let width = c.bbox[2] - c.bbox[0];
                println!(
                    "{:>5} | {:>4} | {:>7.1} | {:>7.2} | {:>7.2} | {:>7.2} | {:>7.2} | {}",
                    i,
                    c.codepoint,
                    c.font_size,
                    height,
                    c.bbox[1],
                    c.bbox[3],
                    width,
                    c.font_name
                );
            }
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
