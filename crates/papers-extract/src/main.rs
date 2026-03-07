use std::path::PathBuf;
use std::process;

use clap::{Parser, ValueEnum};
use papers_extract::{DebugMode, ExtractOptions};

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
    #[arg(long, short = 'p')]
    page: Option<u32>,

    /// Skip image extraction
    #[arg(long)]
    skip_images: bool,

    /// Write layout debug output: "images" for annotated PNGs, "pdf" for PNGs + debug PDF
    #[arg(long, value_name = "MODE")]
    write_layout: Option<LayoutDebugArg>,
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
        extract_images: !cli.skip_images,
        page: cli.page,
        debug: match cli.write_layout {
            Some(LayoutDebugArg::Images) => DebugMode::Images,
            Some(LayoutDebugArg::Pdf) => DebugMode::Pdf,
            None => DebugMode::Off,
        },
        ..ExtractOptions::default()
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
