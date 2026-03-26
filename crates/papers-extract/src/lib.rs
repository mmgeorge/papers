pub mod error;
pub mod figure;
pub mod formula;
pub mod glm_ocr;
pub mod headings;
pub mod html_table;
pub mod layout;
pub mod models;
pub mod output;
pub mod pdf;
pub mod pipeline;
pub mod reading_order;
pub mod tableformer;
pub mod text;
pub mod text_cleanup;
pub mod text_only;
pub mod toc;
pub mod types;

pub use error::ExtractError;
pub use pipeline::Pipeline;
pub use types::*;

use std::path::{Path, PathBuf};

/// Options controlling the extraction pipeline.
pub struct ExtractOptions {
    /// DPI for rendering pages (default 144).
    pub dpi: u32,
    /// Minimum confidence threshold for layout regions (default 0.3).
    pub confidence_threshold: f32,
    /// Whether to extract figures as images (default true).
    pub extract_images: bool,
    /// Formula recognition model (default GLM-OCR).
    pub formula: FormulaModel,
    /// Formula parse mode (default Hybrid).
    pub formula_parse_mode: FormulaParseMode,
    /// Table recognition model (default TableFormer).
    pub table: TableModel,
    /// Path to the pdfium binary (auto-detected if None).
    pub pdfium_path: Option<PathBuf>,
    /// Directory for ONNX model cache (auto-detected if None).
    pub model_cache_dir: Option<PathBuf>,
    /// Extract only this page (1-indexed). If None, extract all pages.
    pub page: Option<u32>,
    /// Extract only these pages (1-indexed). Overrides `page`.
    pub pages: Option<Vec<u32>>,
    /// Reflow-only mode: skip model inference, re-run reflow from existing JSON.
    pub reflow_only: bool,
    /// Extract pages belonging to this TOC chapter (e.g. "3" or "Introduction").
    pub chapter: Option<String>,
    /// Extract pages belonging to this TOC section (e.g. "1.3.2").
    pub section: Option<String>,
    /// Debug visualization mode (default None).
    pub debug: DebugMode,
    /// Dump cropped formula region images to `formulas/` in the output directory.
    pub dump_formulas: bool,
    /// Text-only mode: skip all ML models, extract from PDF text layer only.
    pub text_only: bool,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            dpi: 144,
            confidence_threshold: 0.3,
            extract_images: true,
            formula: FormulaModel::default(),
            formula_parse_mode: FormulaParseMode::default(),
            table: TableModel::default(),
            pdfium_path: None,
            model_cache_dir: None,
            page: None,
            pages: None,
            reflow_only: false,
            chapter: None,
            section: None,
            debug: DebugMode::Off,
            dump_formulas: false,
            text_only: false,
        }
    }
}

/// Formula recognition model selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FormulaModel {
    /// PP-FormulaNet encoder/decoder with CUDA graph acceleration.
    PpFormulanet,
    /// GLM-OCR vision-language model with formula prompt (default).
    #[default]
    GlmOcr,
}

/// Formula parse mode — controls char-based vs OCR strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FormulaParseMode {
    /// Try char-based extraction first, fall back to OCR (default).
    #[default]
    Hybrid,
    /// Char-based only — skip formulas that can't be handled manually.
    Manual,
    /// Run OCR on every detected formula, skip char-based extraction.
    Ocr,
}

/// Table recognition model selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TableModel {
    /// GLM-OCR vision-language model with table prompt.
    GlmOcr,
    /// TableFormer V1 — OTSL structure recognition (~203 MB, default).
    #[default]
    TableFormer,
}

/// Controls what debug output to produce.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DebugMode {
    /// No debug output.
    #[default]
    Off,
    /// Write annotated page PNGs to `layout/`.
    Images,
    /// Write annotated page PNGs to `layout/` and a vector-overlay debug PDF.
    Pdf,
}

impl DebugMode {
    /// Returns true if any debug output is enabled.
    pub fn is_enabled(self) -> bool {
        self != Self::Off
    }
}

/// Parse a page-range spec like "1-50", "33,36,42", "1-10,15,20-30"
/// into a sorted, deduplicated Vec of 1-indexed page numbers.
pub fn parse_page_spec(spec: &str) -> Result<Vec<u32>, String> {
    let mut pages = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            let s: u32 = start
                .trim()
                .parse()
                .map_err(|_| format!("invalid page: {start}"))?;
            let e: u32 = end
                .trim()
                .parse()
                .map_err(|_| format!("invalid page: {end}"))?;
            if s == 0 || e == 0 || s > e {
                return Err(format!("invalid range: {part}"));
            }
            pages.extend(s..=e);
        } else {
            let p: u32 = part
                .parse()
                .map_err(|_| format!("invalid page: {part}"))?;
            if p == 0 {
                return Err("page 0 is invalid (1-indexed)".into());
            }
            pages.push(p);
        }
    }
    pages.sort();
    pages.dedup();
    Ok(pages)
}

/// One-shot extraction — loads models, processes a single PDF, writes output.
///
/// For processing multiple PDFs, use [`Pipeline`] to load models once.
pub fn extract(
    pdf_path: &Path,
    output_dir: &Path,
    options: &ExtractOptions,
) -> Result<ExtractionResult, ExtractError> {
    let pipeline = Pipeline::new(options)?;
    pipeline.extract(pdf_path, output_dir)
}

/// Reflow-only mode: re-run reflow from existing extraction JSON without model inference.
///
/// Loads the PDF only for TOC text extraction (fast), reads the existing `.json`,
/// re-runs the reflow pipeline, and writes `.reflow.json` and `.md`.
pub fn reflow_only(
    pdf_path: &Path,
    output_dir: &Path,
    _options: &ExtractOptions,
) -> Result<(), ExtractError> {
    let start = std::time::Instant::now();

    let pdfium = pdf::load_pdfium(None)?;

    let stem = pdf_path
        .file_stem()
        .and_then(|f| f.to_str())
        .unwrap_or("document");

    // Load existing extraction result
    let json_path = output_dir.join(format!("{stem}.json"));
    let json_str = std::fs::read_to_string(&json_path).map_err(|e| {
        ExtractError::Io(std::io::Error::new(
            e.kind(),
            format!("Cannot read {}: {e}", json_path.display()),
        ))
    })?;
    let result: ExtractionResult = serde_json::from_str(&json_str)?;

    // Parse TOC from PDF text layer (fast, no model inference)
    let doc = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| ExtractError::Pdf(format!("Failed to load PDF: {e}")))?;
    let total_pages = doc.pages().len() as u32;
    let page_chars: Vec<(Vec<pdf::PdfChar>, f32)> = (0..total_pages)
        .map(|i| {
            let page = doc.pages().get(i as u16).unwrap();
            let chars = pdf::extract_page_chars(&page, i).unwrap_or_default();
            let height = page.height().value;
            (chars, height)
        })
        .collect();
    let toc_result = toc::parse_toc(&page_chars);

    if let Some(ref toc) = toc_result {
        eprintln!(
            "  TOC: {} entries from {} pages",
            toc.entries.len(),
            toc.toc_pages.len(),
        );
    }

    // Detect watermarks via font-change analysis
    let watermarks = output::detect_watermarks(&page_chars);

    // Run reflow
    let reflow_doc = if let Some(ref toc) = toc_result {
        output::reflow_with_outline(&result, &toc.entries, &toc.toc_pages, total_pages, &watermarks)
    } else {
        output::reflow(&result, &watermarks)
    };

    // Write outputs
    std::fs::create_dir_all(output_dir)?;
    let reflow_path = output_dir.join(format!("{stem}.reflow.json"));
    output::write_reflow_json(&reflow_doc, &reflow_path)?;

    let md = output::render_markdown_from_reflow(&reflow_doc);
    let md_path = output_dir.join(format!("{stem}.md"));
    std::fs::write(&md_path, md)?;

    let elapsed = start.elapsed();
    eprintln!("  Reflow done in {:.1}ms", elapsed.as_secs_f64() * 1000.0);

    Ok(())
}

/// Compute a tight crop bbox for a formula using pdfium's line-break chars.
///
/// Scans the PdfChar stream for `\n` chars (which pdfium inserts between
/// text objects at different Y positions). Uses the \n Y-positions as
/// line boundaries to find the whitespace gaps above and below the formula.
///
/// `formula_bbox` is in image space (Y-down). Returns the crop bbox in
/// image space, expanded horizontally by `h_pad`.
fn compute_formula_crop_bbox(
    chars: &[pdf::PdfChar],
    formula_bbox: [f32; 4],
    page_height_pt: f32,
    h_pad: f32,
) -> [f32; 4] {
    // Collect Y positions of \n chars in image space.
    // These mark the boundaries between text lines as detected by pdfium.
    let mut newline_ys: Vec<f32> = Vec::new();
    for c in chars {
        if c.codepoint == '\n' {
            // \n char's bbox is a zero-area point at the previous char's position.
            // Convert to image space Y.
            let img_y = page_height_pt - (c.bbox[1] + c.bbox[3]) / 2.0;
            newline_ys.push(img_y);
        }
    }
    newline_ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let formula_center_y = (formula_bbox[1] + formula_bbox[3]) / 2.0;

    // Find the \n just ABOVE the formula (the line break between prose and formula).
    // This is the largest newline Y that's above the formula center.
    let break_above = newline_ys
        .iter()
        .filter(|&&y| y < formula_center_y)
        .copied()
        .last();

    // Find the \n just BELOW the formula (the line break between formula and next line).
    // This is the smallest newline Y that's below the formula center.
    let break_below = newline_ys
        .iter()
        .filter(|&&y| y > formula_center_y)
        .copied()
        .next();

    // The \n positions mark text object transitions. The actual formula content
    // is BETWEEN these two \n positions. Use the midpoint between the \n and
    // the formula bbox edge as the crop boundary — this captures the formula's
    // ascenders/descenders without including adjacent prose.
    // Use formula bbox edges as MINIMUM extent (they include ascenders/
    // descenders via loose_bounds), and \n positions as MAXIMUM extent
    // (they mark the actual line boundaries). This ensures:
    // - Formula superscripts/subscripts are never clipped (bbox minimum)
    // - Prose text from adjacent lines is never included (\n maximum)
    // - Multi-line formulas capture all lines (bbox spans them)
    let y_top = match break_above {
        Some(ny) => {
            // Don't go above the \n (prose boundary), but do include
            // the full formula bbox top (superscripts/ascenders)
            formula_bbox[1].max(ny)
        }
        None => formula_bbox[1] - 5.0,
    };

    let y_bot = match break_below {
        Some(ny) => {
            // Don't go below the \n (prose boundary), but do include
            // the full formula bbox bottom (subscripts/descenders)
            formula_bbox[3].min(ny)
        }
        None => formula_bbox[3] + 5.0,
    };

    [
        (formula_bbox[0] - h_pad).max(0.0),
        y_top,
        (formula_bbox[2] + h_pad).min(page_height_pt),
        y_bot,
    ]
}

/// Trim a formula image using whitespace-band detection.
///
/// Given an expanded crop containing the formula plus surrounding content,
/// finds horizontal whitespace bands (rows of mostly-white pixels) and uses
/// them to separate the formula from adjacent prose text.
///
/// `orig_y_top` and `orig_y_bot` are the pixel Y coordinates of the original
/// text-layer bbox within the expanded image — used to identify which content
/// band contains the formula.
/// Expand figure bboxes to include adjacent sub-figure label text regions.
///
/// PDF authors often place sub-figure labels (e.g. "(a) Reference") as separate
/// text objects adjacent to the figure graphics. We absorb short, non-caption
/// text regions within a small gap of the figure into the figure bbox, so they
/// appear in the cropped figure image and are suppressed from the text output.
fn expand_with_labels(
    figure_bboxes: &mut Vec<[f32; 4]>,
    regions: &mut [Region],
) {
    loop {
        let mut expanded = false;
        for fig_bbox in figure_bboxes.iter_mut() {
            for region in regions.iter_mut() {
                if region.consumed {
                    continue;
                }
                // Only absorb plain text regions, not headings/captions/images
                if region.kind != RegionKind::Text {
                    continue;
                }

                let text = match region.text.as_deref() {
                    Some(t) => t.trim(),
                    None => continue,
                };

                // Must not start with a caption prefix
                if is_figure_caption_prefix(text) {
                    continue;
                }

                // Must be short (≤50 chars) OR contain sub-figure label patterns
                // like "(a)", "(b)" which indicate a multi-label block
                if text.len() > 50 && !has_subfigure_labels(text) {
                    continue;
                }

                // Must overlap vertically with the figure
                let y_overlap =
                    fig_bbox[3].min(region.bbox[3]) - fig_bbox[1].max(region.bbox[1]);
                if y_overlap <= 0.0 {
                    continue;
                }

                // Must be adjacent (gap ≤ 10pt)
                let x_gap = (region.bbox[0] - fig_bbox[2])
                    .max(fig_bbox[0] - region.bbox[2])
                    .max(0.0);
                let y_gap = (region.bbox[1] - fig_bbox[3])
                    .max(fig_bbox[1] - region.bbox[3])
                    .max(0.0);
                if x_gap > 10.0 || y_gap > 10.0 {
                    continue;
                }

                // Expand figure bbox to include this label
                fig_bbox[0] = fig_bbox[0].min(region.bbox[0]);
                fig_bbox[1] = fig_bbox[1].min(region.bbox[1]);
                fig_bbox[2] = fig_bbox[2].max(region.bbox[2]);
                fig_bbox[3] = fig_bbox[3].max(region.bbox[3]);

                region.consumed = true;
                expanded = true;
            }
        }
        if !expanded {
            break;
        }
    }

}

/// Check if text contains sub-figure label patterns like "(a)", "(b)", "(1)", "(2)".
/// Returns true if ≥2 such patterns are found, indicating a multi-label block
/// (e.g. "(a) Reference (b) AVBD, 5 iterations (c) ...").
fn has_subfigure_labels(text: &str) -> bool {
    let mut count = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len().saturating_sub(2) {
        if bytes[i] == b'(' {
            let c = bytes[i + 1];
            if (c.is_ascii_lowercase() || c.is_ascii_digit()) && bytes[i + 2] == b')' {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if text starts with a caption prefix (Fig., Table, Algorithm, etc.)
fn is_figure_caption_prefix(text: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "fig.", "fig ", "figure",
        "table", "tab.", "tab ", "tbl.",
        "algorithm", "alg.", "alg ",
    ];
    let lower = text.to_lowercase();
    PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Trim whitespace rows from the top and bottom of a figure image.
/// Scans for rows where the average pixel brightness is near-white
/// (>250 for RGB) and crops them out, leaving a small padding.
fn trim_figure_whitespace(img: &image::DynamicImage) -> image::DynamicImage {
    use image::GenericImageView;
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }

    let rgb = img.to_rgb8();
    let ws_threshold = 2; // max non-white pixels per row (< 1% of width)
    let min_ink = (w as usize) / 100;

    // Count non-white pixels per row
    let row_ink: Vec<usize> = (0..h)
        .map(|y| {
            (0..w)
                .filter(|&x| {
                    let p = rgb.get_pixel(x, y);
                    p[0] < 245 || p[1] < 245 || p[2] < 245
                })
                .count()
        })
        .collect();

    // Find first and last rows with content
    let top = row_ink
        .iter()
        .position(|&ink| ink > min_ink.max(ws_threshold))
        .unwrap_or(0);
    let bottom = row_ink
        .iter()
        .rposition(|&ink| ink > min_ink.max(ws_threshold))
        .unwrap_or(h as usize - 1);

    let pad = 4u32;
    let crop_top = (top as u32).saturating_sub(pad);
    let crop_bottom = ((bottom as u32) + pad + 1).min(h);

    if crop_top >= crop_bottom || crop_bottom - crop_top < 20 {
        return img.clone();
    }

    img.crop_imm(0, crop_top, w, crop_bottom - crop_top)
}

fn trim_formula_image(
    img: &image::DynamicImage,
    orig_y_top: u32,
    orig_y_bot: u32,
    pad: u32,
) -> image::DynamicImage {
    use image::GenericImageView;
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }

    let threshold = 240u8;

    // Compute ink density per row — how many non-white pixels in each row
    let row_ink: Vec<u32> = (0..h)
        .map(|y| {
            (0..w)
                .filter(|&x| {
                    let p = img.get_pixel(x, y);
                    p[0] < threshold || p[1] < threshold || p[2] < threshold
                })
                .count() as u32
        })
        .collect();

    // A "whitespace row" has very few ink pixels (< 2% of width)
    let ws_threshold = (w as f32 * 0.02).max(2.0) as u32;

    // Find all whitespace bands (runs of ≥ min_ws_run whitespace rows),
    // then pick the bands nearest the formula bbox to crop.
    //
    // Previous approach scanned outward from the formula CENTER, but for
    // multi-line formulas the center can land in an internal gap between
    // formula lines, causing it to clip the formula instead of the prose.
    let min_ws_run = 3u32;

    let mut bands: Vec<(u32, u32)> = Vec::new(); // (start_y, end_y) inclusive
    let mut band_start: Option<u32> = None;
    for y in 0..h {
        if row_ink[y as usize] <= ws_threshold {
            if band_start.is_none() {
                band_start = Some(y);
            }
        } else {
            if let Some(start) = band_start {
                if y - start >= min_ws_run {
                    bands.push((start, y - 1));
                }
                band_start = None;
            }
        }
    }
    if let Some(start) = band_start {
        if h - start >= min_ws_run {
            bands.push((start, h - 1));
        }
    }

    // Slack for bbox inflation (loose_bounds can inflate by ~14pt = ~28px at 144 DPI).
    let slack = 30u32;

    // crop_top: whitespace band closest to (but above) the formula bbox top.
    // Filter to bands whose bottom edge is near/above the bbox top — this
    // excludes internal formula gaps which are well below orig_y_top.
    let crop_top = bands
        .iter()
        .filter(|(_, end)| *end <= orig_y_top + slack)
        .max_by_key(|(_, end)| *end)
        .map(|(_, end)| end + 1)
        .unwrap_or(0);

    // crop_bot: whitespace band closest to (but below) the formula bbox bottom.
    let crop_bot = bands
        .iter()
        .filter(|(start, _)| *start >= orig_y_bot.saturating_sub(slack))
        .min_by_key(|(start, _)| *start)
        .map(|(start, _)| start - 1)
        .unwrap_or(h.saturating_sub(1));

    // Find left/right ink bounds within the vertical crop
    let mut left = w;
    let mut right = 0u32;
    for y in crop_top..=crop_bot.min(h - 1) {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            if p[0] < threshold || p[1] < threshold || p[2] < threshold {
                left = left.min(x);
                right = right.max(x);
            }
        }
    }

    if left >= right {
        // Fallback: no content found, return as-is
        return img.clone();
    }

    // Apply padding
    let x1 = left.saturating_sub(pad);
    let y1 = crop_top.saturating_sub(pad);
    let x2 = (right + pad + 1).min(w);
    let y2 = (crop_bot + pad + 1).min(h);
    let cw = x2.saturating_sub(x1).max(1);
    let ch = y2.saturating_sub(y1).max(1);

    img.crop_imm(x1, y1, cw, ch)
}

/// Text-only extraction: extract from the PDF text layer using geometric heuristics.
///
/// Skips all ML model loading/inference (layout detection, formula OCR, table OCR).
/// Produces the same `ExtractionResult` format, compatible with existing reflow and
/// ingest pipelines. Uses font-based heading detection for document structure.
pub fn extract_text_only(
    pdf_path: &Path,
    output_dir: &Path,
    options: &ExtractOptions,
) -> Result<ExtractionResult, ExtractError> {
    let start = std::time::Instant::now();

    let pdfium = pdf::load_pdfium(options.pdfium_path.as_deref())?;

    let filename = pdf_path
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("document.pdf")
        .to_string();

    let stem = pdf_path
        .file_stem()
        .and_then(|f| f.to_str())
        .unwrap_or("document");

    // Load the PDF
    let doc = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| ExtractError::Pdf(format!("Failed to load PDF: {e}")))?;
    let total_pages = doc.pages().len() as u32;

    // Extract text layer chars from all pages (for TOC + heading detection)
    eprint!("\r  Extracting text layer...");
    let mut page_chars: Vec<(Vec<pdf::PdfChar>, f32)> = Vec::with_capacity(total_pages as usize);
    let mut page_widths: Vec<f32> = Vec::with_capacity(total_pages as usize);
    for i in 0..total_pages {
        let page = doc.pages().get(i as u16).unwrap();
        let mut chars = pdf::extract_page_chars(&page, i).unwrap_or_default();
        let crop = pdf::crop_offset(&page);
        pdf::apply_crop_offset(&mut chars, crop);
        let height = page.height().value;
        let width = page.width().value;
        page_widths.push(width);
        page_chars.push((chars, height));
    }

    // Parse TOC
    let mut toc_result = toc::parse_toc(&page_chars);
    if let Some(ref mut toc) = toc_result {
        eprintln!(
            "\r  TOC: {} entries from {} pages",
            toc.entries.len(),
            toc.toc_pages.len(),
        );
        output::auto_number_toc_entries(&mut toc.entries);
    }

    // Font-based heading detection (works on raw PdfChars, no models)
    eprint!("\r  Detecting headings...");
    let heading_result = headings::extract_headings(&page_chars);
    eprintln!(
        "\r  Headings: {} detected ({} font groups)",
        heading_result.headings.len(),
        heading_result.font_groups.len(),
    );

    // Build heading font family set for char-level heading partitioning.
    // Heading families that differ from the body font → can identify heading
    // chars by font alone before any layout processing.
    let heading_families: std::collections::HashSet<String> = heading_result
        .font_profile
        .heading_levels
        .iter()
        .map(|l| l.font.clone())
        .filter(|f| *f != heading_result.font_profile.body.font)
        .collect();
    if !heading_families.is_empty() {
        eprintln!(
            "  Heading fonts: {} (body: {})",
            heading_families.iter().cloned().collect::<Vec<_>>().join(", "),
            heading_result.font_profile.body.font,
        );
    }

    // Watermark detection (cross-page char-level)
    let watermarks = text_only::detect_watermark_strings(&page_chars);
    if !watermarks.is_empty() {
        eprintln!("  Watermarks: {} patterns detected", watermarks.len());
    }

    // Determine which pages to process
    let page_indices: Vec<u32> = if let Some(ref pages) = options.pages {
        pages
            .iter()
            .map(|p| p - 1)
            .filter(|&p| p < total_pages)
            .collect()
    } else if options.chapter.is_some() || options.section.is_some() {
        pipeline::resolve_toc_page_range(
            &toc_result,
            total_pages,
            options.chapter.as_deref(),
            options.section.as_deref(),
        )?
    } else if let Some(p) = options.page {
        if p == 0 || p > total_pages {
            return Err(ExtractError::Pdf(format!(
                "Page {p} out of range (document has {total_pages} pages)"
            )));
        }
        vec![p - 1]
    } else {
        (0..total_pages).collect()
    };
    let page_count = page_indices.len() as u32;

    // Process each page
    let mut pages = Vec::with_capacity(page_count as usize);
    for (progress, &page_idx) in page_indices.iter().enumerate() {
        eprint!("\r  Page {}/{page_count}: text blocks", progress + 1);
        let (ref chars, height_pt) = page_chars[page_idx as usize];
        let width_pt = page_widths[page_idx as usize];

        // Filter headings for this page (1-indexed)
        let page_headings: Vec<&headings::DetectedHeading> = heading_result
            .headings
            .iter()
            .filter(|h| h.page == page_idx + 1)
            .collect();

        let regions = text_only::extract_page_text_blocks(
            chars,
            height_pt,
            width_pt,
            page_idx,
            &page_headings,
            &watermarks,
            &heading_families,
        );

        pages.push(Page {
            page: page_idx + 1,
            width_pt,
            height_pt,
            dpi: 0,
            regions,
        });
    }
    eprint!("\r{}\r", " ".repeat(60));

    // Post-processing: filter running headers across pages.
    // Running headers are the topmost ParagraphTitle on each page whose text
    // repeats (or nearly repeats) across many pages.
    text_only::filter_running_headers(&mut pages);

    // ── Visual content extraction ──────────────────────────────────────
    // Detect figures from Form XObject bounds and tables from path rules.
    // Render pages with pdfium and crop at the detected bounds to get
    // composite images showing figures as they appear in the PDF.
    if options.extract_images {
        for (i, &page_idx) in page_indices.iter().enumerate() {
            let page = doc.pages().get(page_idx as u16).unwrap();
            let height_pt = page_chars[page_idx as usize].1;
            let width_pt = page_widths[page_idx as usize];
            let crop = pdf::crop_offset(&page);

            // Detect tables via horizontal rules (path objects).
            let exclude_bboxes: Vec<[f32; 4]> = pages[i]
                .regions
                .iter()
                .filter(|r| !r.consumed && r.kind == RegionKind::Algorithm)
                .map(|r| r.bbox)
                .collect();
            let table_bboxes: Vec<[f32; 4]> = pdf::detect_page_tables(&page, height_pt, width_pt, crop)
                .into_iter()
                .filter(|tbl| {
                    !exclude_bboxes.iter().any(|ex| {
                        let x_overlap = tbl[2].min(ex[2]) - tbl[0].max(ex[0]);
                        let y_overlap = tbl[3].min(ex[3]) - tbl[1].max(ex[1]);
                        x_overlap > 0.0 && y_overlap > 0.0
                    })
                })
                .collect();
            for (tbl_idx, bbox) in table_bboxes.into_iter().enumerate() {
                let id = format!("p{}_tbl_{}", page_idx + 1, tbl_idx);
                let rel_path = format!("images/tables/{id}.png");
                pages[i].regions.push(Region {
                    id, kind: RegionKind::Table, bbox, confidence: 1.0,
                    order: 0, text: None, html: None, latex: None,
                    formula_source: None, ocr_confidence: None,
                    image_path: Some(rel_path), caption: None,
                    chart_type: None, tag: None, items: None, consumed: false,
                });
            }

            // Suppress text inside table bboxes
            let tbl_bboxes: Vec<[f32; 4]> = pages[i].regions.iter()
                .filter(|r| r.kind == RegionKind::Table && !r.consumed)
                .map(|r| r.bbox).collect();
            if !tbl_bboxes.is_empty() {
                for region in &mut pages[i].regions {
                    if region.consumed || region.kind == RegionKind::Table || region.kind.is_caption() {
                        continue;
                    }
                    let cx = (region.bbox[0] + region.bbox[2]) / 2.0;
                    let cy = (region.bbox[1] + region.bbox[3]) / 2.0;
                    if tbl_bboxes.iter().any(|b| cx >= b[0] && cx <= b[2] && cy >= b[1] && cy <= b[3]) {
                        region.consumed = true;
                    }
                }
            }
            figure::associate_captions_tables_only(&mut pages[i].regions);

            // ── Figure detection from Form XObject bounds ──
            // Large Form XObjects (>3 children) are figure containers.
            // Their declared bounds from FPDFPageObj_GetBounds() give the
            // figure's placement on the page. We render the page and crop
            // at these bounds to get composite figure images.
            let mut figure_bboxes = pdf::detect_page_figures(&page, height_pt, width_pt, crop);

            // Absorb adjacent sub-figure labels into figure bboxes.
            // Labels like "(a) Reference" are separate text objects on the page,
            // not inside the Form XObject. This expands figure bboxes to include
            // them and marks them consumed so they don't appear in text output.
            expand_with_labels(&mut figure_bboxes, &mut pages[i].regions);

            // Associate each figure bbox with the nearest FigureTitle caption
            // using reading order: find the FigureTitle whose Y-center is
            // closest to the figure bbox's Y-center in the same column.
            let mut x_centers: Vec<f32> = pages[i].regions.iter()
                .filter(|r| !r.consumed && r.kind != RegionKind::Image)
                .map(|r| (r.bbox[0] + r.bbox[2]) / 2.0)
                .collect();
            x_centers.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let col_boundary = if x_centers.len() >= 4 {
                let mut max_gap = 0.0f32;
                let mut boundary = 0.0f32;
                for w in x_centers.windows(2) {
                    if w[1] - w[0] > max_gap {
                        max_gap = w[1] - w[0];
                        boundary = (w[0] + w[1]) / 2.0;
                    }
                }
                if max_gap > 50.0 { Some(boundary) } else { None }
            } else {
                None
            };

            // Clip figure bboxes to column edges to prevent text bleed.
            if let Some(bnd) = col_boundary {
                let left_x1 = pages[i].regions.iter()
                    .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 < bnd)
                    .map(|r| r.bbox[0]).fold(f32::MAX, f32::min);
                let left_x2 = pages[i].regions.iter()
                    .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 < bnd)
                    .map(|r| r.bbox[2]).fold(f32::MIN, f32::max);
                let right_x1 = pages[i].regions.iter()
                    .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 >= bnd)
                    .map(|r| r.bbox[0]).fold(f32::MAX, f32::min);
                let right_x2 = pages[i].regions.iter()
                    .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 >= bnd)
                    .map(|r| r.bbox[2]).fold(f32::MIN, f32::max);

                for bbox in &mut figure_bboxes {
                    let fig_cx = (bbox[0] + bbox[2]) / 2.0;
                    if fig_cx < bnd {
                        bbox[0] = bbox[0].max(left_x1);
                        bbox[2] = bbox[2].min(left_x2);
                    } else {
                        bbox[0] = bbox[0].max(right_x1);
                        bbox[2] = bbox[2].min(right_x2);
                    }
                }
            }

            // Collect FigureTitle captions
            let fig_captions: Vec<(usize, [f32; 4])> = pages[i].regions.iter()
                .enumerate()
                .filter(|(_, r)| {
                    r.kind == RegionKind::FigureTitle && !r.consumed
                        && r.text.as_deref().unwrap_or("").to_lowercase().starts_with("fig")
                })
                .map(|(idx, r)| (idx, r.bbox))
                .collect();

            let mut used_captions: std::collections::HashSet<usize> = std::collections::HashSet::new();

            for (fig_idx, fig_bbox) in figure_bboxes.iter().enumerate() {
                let fig_cx = (fig_bbox[0] + fig_bbox[2]) / 2.0;
                let fig_cy = (fig_bbox[1] + fig_bbox[3]) / 2.0;
                let fig_is_right = col_boundary.map_or(false, |b| fig_cx > b);

                // Find the nearest unassigned caption in the same column,
                // preferring captions below the figure (natural layout).
                let best_cap = fig_captions.iter()
                    .filter(|(idx, _)| !used_captions.contains(idx))
                    .filter(|(_, cap_bbox)| {
                        let cap_cx = (cap_bbox[0] + cap_bbox[2]) / 2.0;
                        let cap_is_right = col_boundary.map_or(false, |b| cap_cx > b);
                        cap_is_right == fig_is_right
                    })
                    .min_by(|(_, a), (_, b)| {
                        let a_cy = (a[1] + a[3]) / 2.0;
                        let b_cy = (b[1] + b[3]) / 2.0;
                        let da = (a_cy - fig_cy).abs();
                        let db = (b_cy - fig_cy).abs();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| *idx);

                let id = format!("p{}_fig_{}", page_idx + 1, fig_idx);
                let rel_path = format!("images/figures/{id}.png");

                let (caption, order) = if let Some(cap_idx) = best_cap {
                    used_captions.insert(cap_idx);
                    let cap_clone = Box::new(pages[i].regions[cap_idx].clone());
                    let cap_order = pages[i].regions[cap_idx].order;
                    pages[i].regions[cap_idx].consumed = true;
                    (Some(cap_clone), cap_order.saturating_sub(1))
                } else {
                    // No caption — place by Y position
                    let order = pages[i].regions.iter()
                        .filter(|r| !r.consumed && r.kind != RegionKind::Image)
                        .filter(|r| (r.bbox[1] + r.bbox[3]) / 2.0 <= fig_cy)
                        .map(|r| r.order)
                        .max()
                        .unwrap_or(0);
                    (None, order)
                };

                // Suppress text regions inside the figure bbox
                for region in &mut pages[i].regions {
                    if region.consumed || region.kind == RegionKind::Image
                        || region.kind == RegionKind::FigureTitle
                        || region.kind == RegionKind::ParagraphTitle
                    {
                        continue;
                    }
                    let cx = (region.bbox[0] + region.bbox[2]) / 2.0;
                    let cy = (region.bbox[1] + region.bbox[3]) / 2.0;
                    if cx >= fig_bbox[0] && cx <= fig_bbox[2]
                        && cy >= fig_bbox[1] && cy <= fig_bbox[3]
                    {
                        region.consumed = true;
                    }
                }

                pages[i].regions.push(Region {
                    id, kind: RegionKind::Image, bbox: *fig_bbox,
                    confidence: 1.0, order,
                    text: None, html: None, latex: None,
                    formula_source: None, ocr_confidence: None,
                    image_path: Some(rel_path), caption,
                    chart_type: None, tag: None, items: None,
                    consumed: false,
                });
            }

            // Handle FigureTitle captions that didn't match any Form XObject
            // (vector-only figures with no Form XObject container).
            // For these, compute a figure slot from text coordinates.
            for (cap_idx, cap_bbox) in &fig_captions {
                if used_captions.contains(cap_idx) {
                    continue;
                }
                let cap_order = pages[i].regions[*cap_idx].order;
                let cap_cx = (cap_bbox[0] + cap_bbox[2]) / 2.0;
                let cap_is_right = col_boundary.map_or(false, |b| cap_cx > b);

                // Slot: from previous heading/caption/figure to top of this caption.
                // Include consumed regions as boundaries — a consumed FigureTitle
                // or Image (Form XObject figure) should still fence the slot.
                let slot_bottom = cap_bbox[1];
                let slot_top = pages[i].regions.iter()
                    .filter(|r| {
                        r.order < cap_order
                            && (r.kind == RegionKind::ParagraphTitle
                                || r.kind == RegionKind::FigureTitle
                                || r.kind == RegionKind::Image)
                    })
                    .filter(|r| {
                        let cx = (r.bbox[0] + r.bbox[2]) / 2.0;
                        col_boundary.map_or(true, |b| (cx > b) == cap_is_right)
                            && r.bbox[3] < slot_bottom
                    })
                    .map(|r| r.bbox[3])
                    .fold(f32::MIN, f32::max);
                let slot_top = if slot_top == f32::MIN { 35.0 } else { slot_top };

                if slot_bottom - slot_top < 30.0 {
                    continue;
                }

                // Use column edges for left/right
                let (slot_left, slot_right) = if let Some(bnd) = col_boundary {
                    if cap_is_right {
                        let x1 = pages[i].regions.iter()
                            .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 >= bnd)
                            .map(|r| r.bbox[0]).fold(f32::MAX, f32::min);
                        let x2 = pages[i].regions.iter()
                            .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 >= bnd)
                            .map(|r| r.bbox[2]).fold(f32::MIN, f32::max);
                        (x1, x2)
                    } else {
                        let x1 = pages[i].regions.iter()
                            .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 < bnd)
                            .map(|r| r.bbox[0]).fold(f32::MAX, f32::min);
                        let x2 = pages[i].regions.iter()
                            .filter(|r| !r.consumed && (r.bbox[0] + r.bbox[2]) / 2.0 < bnd)
                            .map(|r| r.bbox[2]).fold(f32::MIN, f32::max);
                        (x1, x2)
                    }
                } else {
                    (cap_bbox[0], cap_bbox[2])
                };

                let slot_bbox = [slot_left.max(0.0), slot_top, slot_right.min(width_pt), slot_bottom];
                let fig_id = format!("p{}_figslot_{}", page_idx + 1, cap_idx);
                let rel_path = format!("images/figures/{fig_id}.png");

                let cap_clone = Box::new(pages[i].regions[*cap_idx].clone());
                pages[i].regions[*cap_idx].consumed = true;

                pages[i].regions.push(Region {
                    id: fig_id, kind: RegionKind::Image, bbox: slot_bbox,
                    confidence: 1.0, order: cap_order.saturating_sub(1),
                    text: None, html: None, latex: None,
                    formula_source: None, ocr_confidence: None,
                    image_path: Some(rel_path), caption: Some(cap_clone),
                    chart_type: None, tag: None, items: None,
                    consumed: false,
                });
            }

            pages[i].regions.sort_by_key(|r| r.order);
        }

        let fig_count: usize = pages.iter()
            .flat_map(|p| &p.regions)
            .filter(|r| r.kind == RegionKind::Image && !r.consumed)
            .count();
        let tbl_count: usize = pages.iter()
            .flat_map(|p| &p.regions)
            .filter(|r| r.kind == RegionKind::Table && !r.consumed)
            .count();
        if fig_count > 0 || tbl_count > 0 {
            eprintln!("  Visual: {fig_count} figures, {tbl_count} tables detected");
        }
    }

    // Check for empty extraction (likely scanned PDF)
    let total_regions: usize = pages.iter().map(|p| p.regions.len()).sum();
    if total_regions == 0 {
        eprintln!(
            "  Warning: no text extracted from {} pages. Is this a scanned PDF?",
            page_count
        );
    }

    let extraction_time_ms = start.elapsed().as_millis() as u64;
    // Render pages with display formulas / tables and crop images.
    // This reuses the same code as the layout path's manual formula mode:
    // render the page, crop at formula/table bbox, save as PNG, set image_path.
    let mut page_images: Vec<Option<image::DynamicImage>> = vec![None; pages.len()];
    {
        // Pages that need rendering: formulas, tables, or figure slots.
        let render_pages: std::collections::HashSet<u32> = pages
            .iter()
            .filter(|p| {
                p.regions.iter().any(|r| {
                    (r.kind == RegionKind::DisplayFormula && r.latex.is_none() && !r.consumed)
                        || (r.kind == RegionKind::Table
                            && r.image_path.is_some()
                            && r.text.is_none()
                            && !r.consumed)
                        || (r.kind == RegionKind::Image
                            && r.image_path.is_some()
                            && !r.consumed)
                })
            })
            .map(|p| p.page)
            .collect();

        if !render_pages.is_empty() {
            eprint!("\r  Rendering {} pages for images...", render_pages.len());
            for (i, &page_idx) in page_indices.iter().enumerate() {
                let page_num = page_idx + 1;
                if !render_pages.contains(&page_num) {
                    continue;
                }
                let page = doc.pages().get(page_idx as u16).unwrap();
                let dpi = 144u32;
                match pdf::render_page(&page, dpi) {
                    Ok(img) => {
                        // Set image_path on formula regions and save cropped images.
                        // Create images for formula regions. Filter out tiny
                        // fragments (<12pt high or <80pt wide) but allow
                        // multi-line formulas (no max height — whitespace-band
                        // trimming handles prose separation).
                        for region in &mut pages[i].regions {
                            let bbox_h = region.bbox[3] - region.bbox[1];
                            let bbox_w = region.bbox[2] - region.bbox[0];
                            if region.kind == RegionKind::DisplayFormula
                                && region.latex.is_none()
                                && !region.consumed
                                && bbox_h >= 12.0
                                && bbox_w >= 80.0
                            {
                                region.image_path =
                                    Some(format!("images/formulas/{}.png", region.id));
                                // Clear the garbled text — the image is the source of truth
                                region.text = None;
                            }
                        }
                        pages[i].dpi = dpi;
                        page_images[i] = Some(img);
                    }
                    Err(e) => {
                        eprintln!("\r  Warning: failed to render page {page_num}: {e}");
                    }
                }
            }
        }
    }

    let result = ExtractionResult {
        metadata: Metadata {
            filename,
            page_count,
            extraction_time_ms,
        },
        pages,
    };

    // Write outputs
    std::fs::create_dir_all(output_dir)?;

    // Write formula images for pages that were rendered
    {
        let formula_img_dir = output_dir.join("images/formulas");
        let has_formula_images = page_images.iter().any(|img| img.is_some());
        if has_formula_images {
            std::fs::create_dir_all(&formula_img_dir)?;
            for ((page, maybe_img), &page_idx) in result.pages.iter().zip(page_images.iter()).zip(page_indices.iter()) {
                let Some(img) = maybe_img else { continue };
                let (ref pg_chars, pg_height) = page_chars[page_idx as usize];
                for region in &page.regions {
                    if region.image_path.is_none() || region.kind != RegionKind::DisplayFormula {
                        continue;
                    }
                    // Content-aware crop using rendered pixel scanning.
                    //
                    // Text-layer bboxes and \n positions are both unreliable
                    // (loose_bounds inflates into adjacent lines, \n is at
                    // baselines not whitespace gaps). Instead:
                    // 1. Expand generously from the formula CENTER
                    // 2. Crop the expanded area from the rendered page
                    // 3. Use whitespace-band detection on the PIXELS to
                    //    find the actual gap between formula and prose
                    let y_center = (region.bbox[1] + region.bbox[3]) / 2.0;
                    let bbox_half_h = (region.bbox[3] - region.bbox[1]) / 2.0;
                    let v_expand = bbox_half_h + 15.0; // fixed padding (~1 text line)
                    // Modest horizontal padding — formula fragments have
                    // already been merged into the formula bbox during
                    // block post-processing, so we don't need aggressive
                    // expansion that would capture adjacent column prose.
                    let h_expand = 20.0;
                    let expanded_bbox = [
                        (region.bbox[0] - h_expand).max(0.0),
                        (y_center - v_expand).max(0.0),
                        (region.bbox[2] + h_expand).min(page.width_pt),
                        (y_center + v_expand).min(page.height_pt),
                    ];
                    let wide_crop = figure::crop_region(
                        img,
                        expanded_bbox,
                        page.width_pt,
                        page.height_pt,
                        page.dpi,
                    );
                    // Whitespace-band trim anchored from the center.
                    let scale = page.dpi as f32 / 72.0;
                    let center_px = ((y_center - expanded_bbox[1]) * scale) as u32;
                    let half_h_px = (bbox_half_h * scale) as u32;
                    let orig_y_top = center_px.saturating_sub(half_h_px);
                    let orig_y_bot = center_px + half_h_px;
                    let cropped = trim_formula_image(&wide_crop, orig_y_top, orig_y_bot, 8);
                    let path = formula_img_dir.join(format!("{}.png", region.id));
                    if let Err(e) = cropped.save(&path) {
                        eprintln!("  Warning: failed to save formula image {}: {e}", region.id);
                    }
                }
            }
        }
    }

    // Write table images for pages that were rendered
    {
        let table_img_dir = output_dir.join("images/tables");
        let has_table_images = result.pages.iter().any(|p| {
            p.regions
                .iter()
                .any(|r| r.kind == RegionKind::Table && r.image_path.is_some() && !r.consumed)
        });
        if has_table_images {
            std::fs::create_dir_all(&table_img_dir)?;
            for (page, maybe_img) in result.pages.iter().zip(page_images.iter()) {
                let Some(img) = maybe_img else { continue };
                for region in &page.regions {
                    if region.kind != RegionKind::Table
                        || region.image_path.is_none()
                        || region.consumed
                    {
                        continue;
                    }
                    let cropped = figure::crop_region(
                        img,
                        region.bbox,
                        page.width_pt,
                        page.height_pt,
                        page.dpi,
                    );
                    let path = table_img_dir.join(format!("{}.png", region.id));
                    if let Err(e) = cropped.save(&path) {
                        eprintln!("  Warning: failed to save table image {}: {e}", region.id);
                    }
                }
            }
        }
    }

    // Write figure slot images (cropped from rendered pages)
    {
        let fig_img_dir = output_dir.join("images/figures");
        let has_figs = result.pages.iter().any(|p| {
            p.regions.iter().any(|r| r.kind == RegionKind::Image && r.image_path.is_some() && !r.consumed)
        });
        if has_figs {
            std::fs::create_dir_all(&fig_img_dir)?;
            for (page, maybe_img) in result.pages.iter().zip(page_images.iter()) {
                let Some(img) = maybe_img else { continue };
                for region in &page.regions {
                    if region.kind != RegionKind::Image
                        || region.image_path.is_none()
                        || region.consumed
                    {
                        continue;
                    }
                    let cropped = figure::crop_region(
                        img,
                        region.bbox,
                        page.width_pt,
                        page.height_pt,
                        page.dpi,
                    );
                    // Trim whitespace from top and bottom of the figure.
                    // The slot can extend to the page top when there's no
                    // heading above, capturing empty space and page headers.
                    let trimmed = trim_figure_whitespace(&cropped);
                    let rel_path = region.image_path.as_ref().unwrap();
                    let full_path = output_dir.join(rel_path);
                    if let Err(e) = trimmed.save(&full_path) {
                        eprintln!("  Warning: failed to save figure image {}: {e}", region.id);
                    }
                }
            }
        }
    }

    let json_path = output_dir.join(format!("{stem}.json"));
    output::write_json(&result, &json_path)?;

    let watermarks = output::detect_watermarks(&page_chars);
    let reflow_doc = if let Some(ref toc) = toc_result {
        output::reflow_with_outline(&result, &toc.entries, &toc.toc_pages, total_pages, &watermarks)
    } else {
        output::reflow(&result, &watermarks)
    };
    let reflow_path = output_dir.join(format!("{stem}.reflow.json"));
    output::write_reflow_json(&reflow_doc, &reflow_path)?;

    let md = output::render_markdown_from_reflow(&reflow_doc);
    let md_path = output_dir.join(format!("{stem}.md"));
    std::fs::write(&md_path, md)?;

    let elapsed = start.elapsed();
    eprintln!(
        "  Done: {} pages, {} regions, {:.1}s (text-only)",
        page_count,
        total_regions,
        elapsed.as_secs_f64(),
    );

    Ok(result)
}
