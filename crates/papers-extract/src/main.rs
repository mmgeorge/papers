use std::path::PathBuf;
use std::process;

use clap::{Parser, ValueEnum};
use papers_extract::{DebugMode, ExtractOptions, FormulaQuality, Quality};

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

    /// Quality mode for table recognition
    #[arg(long, short, default_value = "fast")]
    quality: QualityArg,

    /// Formula model quality: low (221 MB), med (565 MB), high (700 MB)
    #[arg(long, default_value = "low")]
    formula_quality: FormulaQualityArg,

    /// DPI for page rendering
    #[arg(long, default_value = "144")]
    dpi: u32,

    /// Layout detection confidence threshold (0.0–1.0)
    #[arg(long, default_value = "0.3")]
    confidence: f32,

    /// Extract only this page (1-indexed)
    #[arg(long, short = 'p')]
    page: Option<u32>,

    /// Skip image extraction
    #[arg(long)]
    no_images: bool,

    /// Dump cropped formula images to formulas/ directory
    #[arg(long)]
    dump_formulas: bool,

    /// Path to pdfium library (auto-detected if omitted)
    #[arg(long)]
    pdfium_path: Option<PathBuf>,

    /// Directory for ONNX model cache
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,

    /// Write layout debug output: "images" for annotated PNGs, "pdf" for PNGs + debug PDF
    #[arg(long, value_name = "MODE")]
    write_layout: Option<LayoutDebugArg>,
}

#[derive(ValueEnum, Clone, Debug)]
enum QualityArg {
    /// SLANet-Plus (7 MB)
    Fast,
    /// PP-LCNet classifier + SLANeXt-wired (~358 MB)
    Quality,
}

#[derive(ValueEnum, Clone, Debug)]
enum FormulaQualityArg {
    /// PP-FormulaNet_plus-S (221 MB)
    Low,
    /// PP-FormulaNet_plus-M (565 MB)
    Med,
    /// PP-FormulaNet_plus-L (700 MB)
    High,
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

    let options = ExtractOptions {
        dpi: cli.dpi,
        confidence_threshold: cli.confidence,
        extract_images: !cli.no_images,
        quality: match cli.quality {
            QualityArg::Fast => Quality::Fast,
            QualityArg::Quality => Quality::Quality,
        },
        formula_quality: match cli.formula_quality {
            FormulaQualityArg::Low => FormulaQuality::Low,
            FormulaQualityArg::Med => FormulaQuality::Med,
            FormulaQualityArg::High => FormulaQuality::High,
        },
        pdfium_path: cli.pdfium_path,
        model_cache_dir: cli.model_cache_dir,
        page: cli.page,
        debug: match cli.write_layout {
            Some(LayoutDebugArg::Images) => DebugMode::Images,
            Some(LayoutDebugArg::Pdf) => DebugMode::Pdf,
            None => DebugMode::Off,
        },
        dump_formulas: cli.dump_formulas,
    };

    eprintln!(
        "Extracting {} → {}",
        cli.pdf.display(),
        output_dir.display()
    );

    match papers_extract::extract(&cli.pdf, &output_dir, &options) {
        Ok(result) => {
            eprintln!(
                "Done: {} pages, {} regions, {}ms",
                result.metadata.page_count,
                result.pages.iter().map(|p| p.regions.len()).sum::<usize>(),
                result.metadata.extraction_time_ms,
            );
        }
        Err(e) => {
            eprintln!("Error: {e}");
            process::exit(1);
        }
    }
}
