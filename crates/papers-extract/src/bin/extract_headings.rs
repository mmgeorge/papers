//! Extract document headings from a PDF using font-signal analysis.
//!
//! Analyzes font families, rendered glyph heights, and character frequency
//! to identify heading hierarchy without relying on a layout detection model.
//!
//! Usage:
//!   extract_headings <pdf_path>
//!   extract_headings <pdf_path> --json
//!   extract_headings <pdf_path> --pages 1,2,5-10 --verbose

use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "extract_headings",
    about = "Extract document headings from PDF using font analysis"
)]
struct Cli {
    /// Path to the input PDF file.
    pdf: PathBuf,

    /// Page range filter (e.g., "1,2,5-10"). Default: all pages.
    #[arg(long)]
    pages: Option<String>,

    /// Output JSON format (default: human-readable).
    #[arg(long)]
    json: bool,

    /// Show detailed font group table in human-readable mode.
    #[arg(long, short)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    let pdfium = papers_extract::pdf::load_pdfium(None).expect("Failed to load pdfium");
    let doc = pdfium
        .load_pdf_from_file(&cli.pdf, None)
        .expect("Failed to load PDF");

    let total_pages = doc.pages().len();

    // Parse page range or default to all pages
    let page_indices: Vec<u32> = if let Some(ref range) = cli.pages {
        parse_page_range(range, total_pages as u32)
    } else {
        (0..total_pages as u32).collect()
    };

    // Extract chars from each page
    let mut page_chars: Vec<(Vec<papers_extract::pdf::PdfChar>, f32)> = Vec::new();

    for &page_idx in &page_indices {
        let page = doc
            .pages()
            .get(page_idx as u16)
            .unwrap_or_else(|_| panic!("Failed to get page {}", page_idx + 1));

        let height = page.height().value;
        let chars = papers_extract::pdf::extract_page_chars(&page, page_idx)
            .unwrap_or_else(|e| {
                eprintln!("Warning: failed to extract chars from page {}: {e}", page_idx + 1);
                Vec::new()
            });

        page_chars.push((chars, height));
    }

    let result = papers_extract::headings::extract_headings(&page_chars);

    if cli.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).expect("Failed to serialize JSON")
        );
    } else {
        print_human_readable(&result, cli.verbose);
    }
}

fn print_human_readable(
    result: &papers_extract::headings::HeadingExtractionResult,
    verbose: bool,
) {
    // Font groups table (verbose mode or always show top entries)
    if verbose {
        println!("Font Groups (all):");
        for fg in &result.font_groups {
            let sample = fg
                .sample
                .as_deref()
                .map(|s| {
                    let truncated = if s.len() > 50 {
                        format!("{}...", &s[..47])
                    } else {
                        s.to_string()
                    };
                    format!("\"{}\"", truncated)
                })
                .unwrap_or_else(|| "(math/symbol)".to_string());

            let raw = if fg.raw_font_names.is_empty() {
                String::new()
            } else {
                format!(" [{}]", fg.raw_font_names.join(", "))
            };

            println!(
                "  {:>7} chars | {:>5} segs | {:>20} em={:>5.1} h={:>5.1} | {:>10} | {}{}",
                fg.char_count,
                fg.segment_count,
                if fg.font.is_empty() {
                    "(no name)"
                } else {
                    &fg.font
                },
                fg.em_size,
                fg.height,
                fg.role,
                sample,
                raw,
            );
        }
        println!();
    }

    // Font profile
    println!("Font Profile:");
    let fp = &result.font_profile;
    println!(
        "  Body text:  {} @ {:.1}pt ({} chars)",
        if fp.body.font.is_empty() {
            "(no name)"
        } else {
            &fp.body.font
        },
        fp.body.height,
        fp.body.char_count,
    );
    for level in &fp.heading_levels {
        println!(
            "  Heading L{}: {} @ {:.1}pt ({} chars, {} instances, {} pages)",
            level.depth,
            if level.font.is_empty() {
                "(no name)"
            } else {
                &level.font
            },
            level.height,
            level.char_count,
            level.instances,
            level.pages,
        );
    }
    if !fp.skipped.is_empty() {
        println!("  Skipped:");
        for s in &fp.skipped {
            println!(
                "    {} @ {:.1}pt ({} chars, {} pages) — {}",
                if s.font.is_empty() { "(no name)" } else { &s.font },
                s.height,
                s.char_count,
                s.pages,
                s.reason,
            );
        }
    }
    println!(
        "  Font names available: {}",
        if fp.has_font_names { "yes" } else { "no" }
    );
    println!();

    // Headings
    if result.headings.is_empty() {
        println!("No headings detected.");
    } else {
        println!("Headings:");
        for h in &result.headings {
            let indent = "  ".repeat(h.depth.saturating_sub(1) as usize);
            println!(
                "  p.{:<4} [{}] {}{} ({} chars)",
                h.page, h.depth, indent, h.title, h.contained_chars,
            );
        }
    }
}

/// Parse a page range string like "1,2,5-10" into a sorted Vec of 0-indexed page indices.
fn parse_page_range(s: &str, max_pages: u32) -> Vec<u32> {
    let mut pages = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if let Some((start, end)) = part.split_once('-') {
            let start: u32 = start.trim().parse().expect("invalid page number");
            let end: u32 = end.trim().parse().expect("invalid page number");
            for p in start..=end {
                if p >= 1 && p <= max_pages {
                    pages.push(p - 1);
                }
            }
        } else {
            let p: u32 = part.parse().expect("invalid page number");
            if p >= 1 && p <= max_pages {
                pages.push(p - 1);
            }
        }
    }
    pages.sort();
    pages.dedup();
    pages
}
