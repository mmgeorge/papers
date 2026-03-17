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
