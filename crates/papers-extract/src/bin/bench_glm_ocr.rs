//! Benchmark GLM-OCR on layout-detected regions from a PDF page.
//!
//! Renders a page, runs layout detection, finds regions of a given type
//! (default: Algorithm), crops them, and benchmarks GLM-OCR prediction.
//!
//! Usage:
//!   bench_glm_ocr data/vbd.pdf --page 9           # find & benchmark Algorithm regions
//!   bench_glm_ocr data/vbd.pdf --page 3 --region-type DisplayFormula
//!   bench_glm_ocr data/vbd.pdf --page 9 --dump    # print OCR output to stdout
//!   bench_glm_ocr data/vbd.pdf --page 9 --save-crop algo.png  # save cropped region

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use image::DynamicImage;
use papers_extract::glm_ocr::GlmOcrConfig;
use papers_extract::layout::LayoutDetector;
use papers_extract::models;
use papers_extract::RegionKind;

const DEFAULT_DPI: u32 = 150;
const DEFAULT_MAX_SEQ: usize = 512;
const DEFAULT_CONFIDENCE: f32 = 0.3;

/// Map region kind to the appropriate GLM-OCR prompt.
fn prompt_for_kind(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::DisplayFormula | RegionKind::InlineFormula => "Formula Recognition:",
        RegionKind::Table => "Table Recognition:",
        _ => "Text Recognition:",
    }
}

/// Parse a region kind from CLI string.
fn parse_region_kind(s: &str) -> RegionKind {
    match s.to_lowercase().as_str() {
        "algorithm" => RegionKind::Algorithm,
        "displayformula" | "display_formula" => RegionKind::DisplayFormula,
        "inlineformula" | "inline_formula" => RegionKind::InlineFormula,
        "table" => RegionKind::Table,
        "text" => RegionKind::Text,
        "title" => RegionKind::Title,
        "image" => RegionKind::Image,
        "abstract" => RegionKind::Abstract,
        _ => panic!("Unknown region type: {s}. Use: Algorithm, DisplayFormula, Table, Text, etc."),
    }
}

#[derive(Parser)]
#[command(name = "bench_glm_ocr", about = "Benchmark GLM-OCR on layout-detected regions")]
struct Cli {
    /// Path to PDF file
    pdf: PathBuf,

    /// Page to benchmark (1-indexed)
    #[arg(long, default_value = "1")]
    page: u32,

    /// Region type to find and benchmark
    #[arg(long, default_value = "Algorithm")]
    region_type: String,

    /// DPI for rendering pages
    #[arg(long, default_value_t = DEFAULT_DPI)]
    dpi: u32,

    /// Layout detection confidence threshold
    #[arg(long, default_value_t = DEFAULT_CONFIDENCE)]
    confidence: f32,

    /// Number of benchmark runs
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Override the prompt (default: auto from region type)
    #[arg(long)]
    prompt: Option<String>,

    /// Maximum decode sequence length
    #[arg(long, default_value_t = DEFAULT_MAX_SEQ)]
    max_seq: usize,

    /// Dump OCR output to stdout
    #[arg(long)]
    dump: bool,

    /// Save the cropped region image to a file
    #[arg(long)]
    save_crop: Option<PathBuf>,

    /// Model cache directory
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,

    /// Path to pdfium library
    #[arg(long)]
    pdfium_path: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    let target_kind = parse_region_kind(&cli.region_type);

    // Render the target page
    eprintln!("Rendering page {} at {} DPI...", cli.page, cli.dpi);
    let page_image = render_pdf_page(&cli.pdf, cli.page, cli.dpi, cli.pdfium_path.as_deref());
    let (w, h) = (page_image.width(), page_image.height());
    eprintln!("  Page image: {}x{}", w, h);

    // Init ORT
    models::init_ort_runtime().expect("ORT runtime init");

    let cache_dir = cli
        .model_cache_dir
        .unwrap_or_else(models::default_cache_dir);

    // Layout detection
    eprintln!("Running layout detection...");
    let paths = models::ensure_models(papers_extract::Quality::Fast, &cache_dir)
        .expect("layout model files");
    let layout = models::build_layout_detector(&paths.layout).expect("layout detector");
    let detected = layout.detect(&page_image, cli.confidence).expect("layout detect");

    eprintln!("  {} regions detected", detected.len());
    for (i, d) in detected.iter().enumerate() {
        eprintln!(
            "    [{:2}] {:?} ({:.0}x{:.0}) conf={:.2}",
            i,
            d.kind,
            d.bbox_px[2] - d.bbox_px[0],
            d.bbox_px[3] - d.bbox_px[1],
            d.confidence,
        );
    }

    // Find target regions
    let targets: Vec<_> = detected
        .iter()
        .filter(|d| d.kind == target_kind)
        .collect();

    if targets.is_empty() {
        eprintln!("\nNo {:?} regions found on page {}!", target_kind, cli.page);
        eprintln!("Available region types:");
        let mut kinds: Vec<_> = detected.iter().map(|d| format!("{:?}", d.kind)).collect();
        kinds.sort();
        kinds.dedup();
        for k in kinds {
            eprintln!("  - {k}");
        }
        std::process::exit(1);
    }

    // Use the largest target region (by area)
    let region = targets
        .iter()
        .max_by(|a, b| {
            let area_a = (a.bbox_px[2] - a.bbox_px[0]) * (a.bbox_px[3] - a.bbox_px[1]);
            let area_b = (b.bbox_px[2] - b.bbox_px[0]) * (b.bbox_px[3] - b.bbox_px[1]);
            area_a.partial_cmp(&area_b).unwrap()
        })
        .unwrap();

    let [x1, y1, x2, y2] = region.bbox_px;
    let crop_w = (x2 - x1) as u32;
    let crop_h = (y2 - y1) as u32;
    eprintln!(
        "\nTarget: {:?} region ({crop_w}x{crop_h} px, conf={:.2})",
        target_kind, region.confidence
    );

    // Crop the region from the page image
    let cropped = page_image.crop_imm(x1 as u32, y1 as u32, crop_w, crop_h);
    eprintln!("  Cropped: {}x{}", cropped.width(), cropped.height());

    if let Some(ref path) = cli.save_crop {
        cropped.save(path).expect("save crop");
        eprintln!("  Saved crop to {}", path.display());
    }

    // Load GLM-OCR
    let prompt = cli
        .prompt
        .clone()
        .unwrap_or_else(|| prompt_for_kind(target_kind).to_string());
    eprintln!(
        "\nLoading GLM-OCR (prompt={:?}, max_seq={})...",
        prompt, cli.max_seq
    );

    let model_paths = models::ensure_glm_ocr_models(&cache_dir).expect("GLM-OCR model files");
    let config = GlmOcrConfig {
        prompt,
        max_seq: cli.max_seq,
    };
    let predictor =
        models::build_glm_ocr_predictor_with_config(&model_paths, config).expect("GLM-OCR init");
    eprintln!("GLM-OCR predictor ready\n");

    // Benchmark
    eprintln!("{:>4}  {:>10}  {:>10}", "Run", "Time", "Tokens");
    eprintln!("{}", "-".repeat(35));

    let mut times = Vec::with_capacity(cli.runs);
    let mut last_output = String::new();

    for run in 1..=cli.runs {
        let t0 = Instant::now();
        let results = predictor.predict(&[cropped.clone()]).expect("predict");
        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let text = &results[0];
        let token_count = text.len(); // character count (more useful than whitespace-split)
        times.push(ms);
        last_output = text.clone();

        eprintln!("{:>4}  {:>8.0}ms  {:>8} ch", run, ms, token_count);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];

    eprintln!("{}", "-".repeat(35));
    eprintln!(
        "Median: {:.0}ms  Min: {:.0}ms  Max: {:.0}ms",
        median, min, max
    );
    eprintln!(
        "{:?} region on page {}, {} runs",
        target_kind, cli.page, cli.runs
    );

    if cli.dump {
        println!("{}", last_output);
    }
}

fn render_pdf_page(
    pdf_path: &PathBuf,
    page_num: u32,
    dpi: u32,
    pdfium_path: Option<&std::path::Path>,
) -> DynamicImage {
    let pdfium = papers_extract::pdf::load_pdfium(pdfium_path).expect("load pdfium");
    let doc = pdfium
        .load_pdf_from_file(pdf_path, None)
        .expect("load PDF");
    let total = doc.pages().len() as u32;
    assert!(
        page_num >= 1 && page_num <= total,
        "Page {page_num} out of range (1..{total})"
    );
    let page = doc.pages().get((page_num - 1) as u16).expect("get page");
    papers_extract::pdf::render_page(&page, dpi).expect("render page")
}
