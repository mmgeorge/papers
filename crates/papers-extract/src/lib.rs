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

    // Check for empty extraction (likely scanned PDF)
    let total_regions: usize = pages.iter().map(|p| p.regions.len()).sum();
    if total_regions == 0 {
        eprintln!(
            "  Warning: no text extracted from {} pages. Is this a scanned PDF?",
            page_count
        );
    }

    let extraction_time_ms = start.elapsed().as_millis() as u64;
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
