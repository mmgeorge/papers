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
