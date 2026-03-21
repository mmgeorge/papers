use std::path::PathBuf;
use std::process;

use clap::{Parser, ValueEnum};
use papers_extract::{DebugMode, ExtractOptions, FormulaModel, FormulaParseMode, parse_page_spec};

#[derive(Parser)]
#[command(
    name = "papers-extract",
    about = "Extract structured content from PDFs using local ONNX models",
    term_width = 100
)]
struct Cli {
    /// Path to the input PDF file
    pdf: PathBuf,

    /// Output directory (default: same directory as the PDF)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Extract only this page (1-indexed)
    #[arg(long, short = 'p', conflicts_with_all = ["pages", "chapter", "section"])]
    page: Option<u32>,

    /// Extract specific pages (e.g. "1-50", "33,36,42", "1-10,15")
    #[arg(long, conflicts_with_all = ["page", "chapter", "section"])]
    pages: Option<String>,

    /// Extract pages for a TOC chapter (e.g. "3" or "Introduction")
    #[arg(long, conflicts_with_all = ["page", "pages", "section"])]
    chapter: Option<String>,

    /// Extract pages for a TOC section (e.g. "1.3.2")
    #[arg(long, conflicts_with_all = ["page", "pages", "chapter"])]
    section: Option<String>,

    /// Re-run reflow from existing extraction JSON (skip model inference)
    #[arg(long)]
    reflow_only: bool,

    /// Text-only extraction: skip all ML models, extract from PDF text layer only
    #[arg(long, conflicts_with_all = ["reflow_only", "write_layout", "formula", "formula_parse_mode"])]
    text_only: bool,

    /// Skip image extraction
    #[arg(long)]
    skip_images: bool,

    /// Formula recognition model: "glm-ocr" (default) or "pp-formulanet"
    #[arg(long, value_name = "MODEL", default_value = "glm-ocr")]
    formula: FormulaModelArg,

    /// Formula parse mode: "hybrid" (default), "manual", or "ocr"
    #[arg(long, value_name = "MODE", default_value = "hybrid")]
    formula_parse_mode: FormulaParseModeArg,

    /// Write layout debug output: "images" for annotated PNGs, "pdf" for PNGs + debug PDF
    #[arg(long, value_name = "MODE")]
    write_layout: Option<LayoutDebugArg>,
}

#[derive(ValueEnum, Clone, Debug)]
enum FormulaModelArg {
    /// PP-FormulaNet encoder/decoder (better at complex fractions)
    PpFormulanet,
    /// GLM-OCR vision-language model (default)
    GlmOcr,
}

#[derive(ValueEnum, Clone, Debug)]
enum FormulaParseModeArg {
    /// Try char-based first, fall back to OCR (default)
    Hybrid,
    /// Char-based only — skip formulas that can't be handled
    Manual,
    /// Run OCR on every formula
    Ocr,
}

#[derive(ValueEnum, Clone, Debug)]
enum LayoutDebugArg {
    /// Write annotated page PNGs to layout/
    Images,
    /// Write annotated page PNGs + a vector-overlay debug PDF
    Pdf,
}

fn main() {
    let cli = Cli::parse();

    let output_dir = cli.output.unwrap_or_else(|| {
        cli.pdf
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    });

    let parsed_pages = cli.pages.as_deref().map(|s| {
        parse_page_spec(s).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            process::exit(1);
        })
    });

    let options = ExtractOptions {
        extract_images: !cli.skip_images,
        page: cli.page,
        pages: parsed_pages,
        reflow_only: cli.reflow_only,
        chapter: cli.chapter.clone(),
        section: cli.section.clone(),
        formula: match cli.formula {
            FormulaModelArg::PpFormulanet => FormulaModel::PpFormulanet,
            FormulaModelArg::GlmOcr => FormulaModel::GlmOcr,
        },
        formula_parse_mode: match cli.formula_parse_mode {
            FormulaParseModeArg::Hybrid => FormulaParseMode::Hybrid,
            FormulaParseModeArg::Manual => FormulaParseMode::Manual,
            FormulaParseModeArg::Ocr => FormulaParseMode::Ocr,
        },
        debug: match cli.write_layout {
            Some(LayoutDebugArg::Images) => DebugMode::Images,
            Some(LayoutDebugArg::Pdf) => DebugMode::Pdf,
            None => DebugMode::Off,
        },
        text_only: cli.text_only,
        ..ExtractOptions::default()
    };

    if cli.text_only {
        eprintln!(
            "Text-only: {} → {}",
            cli.pdf.display(),
            output_dir.display()
        );
        match papers_extract::extract_text_only(&cli.pdf, &output_dir, &options) {
            Ok(result) => {
                eprintln!(
                    "Done: {} pages, {} regions, {:.1}s",
                    result.metadata.page_count,
                    result.pages.iter().map(|p| p.regions.len()).sum::<usize>(),
                    result.metadata.extraction_time_ms as f64 / 1000.0,
                );
            }
            Err(e) => {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        return;
    }

    if cli.reflow_only {
        eprintln!(
            "Reflow-only: {} → {}",
            cli.pdf.display(),
            output_dir.display()
        );
        match papers_extract::reflow_only(&cli.pdf, &output_dir, &options) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        }
        return;
    }

    eprintln!(
        "Extracting {} → {}",
        cli.pdf.display(),
        output_dir.display()
    );

    match papers_extract::extract(&cli.pdf, &output_dir, &options) {
        Ok(result) => {
            eprintln!(
                "Done: {} pages, {} regions, {:.1}s",
                result.metadata.page_count,
                result.pages.iter().map(|p| p.regions.len()).sum::<usize>(),
                result.metadata.extraction_time_ms as f64 / 1000.0,
            );
        }
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}
