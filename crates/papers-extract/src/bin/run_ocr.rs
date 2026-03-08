//! Run OCR on detected layout regions from a PDF.
//!
//! Runs layout detection (pp-doclayoutv3) then OCR on matching regions.
//! Supports model selection: GLM-OCR (all region types) or PP-FormulaNet
//! (DisplayFormula + InlineFormula only).
//!
//! Usage:
//!   run_ocr data/avbd.pdf -o .temp/avbd-glm --region-type DisplayFormula
//!   run_ocr data/avbd.pdf -o .temp/avbd-ppfn --model pp-formulanet
//!   run_ocr data/avbd.pdf -o .temp/avbd-glm --region-type "Text,DisplayFormula" --page 3
//!   run_ocr data/avbd.pdf -o .temp/avbd-bench --bench --runs 3

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use image::DynamicImage;
use papers_extract::error::ExtractError;
use papers_extract::figure;
use papers_extract::formula::FormulaPredictor;
use papers_extract::glm_ocr::{Backend, GlmOcrConfig};
use papers_extract::layout::LayoutDetector;
use papers_extract::models;
use papers_extract::pdf;
use papers_extract::RegionKind;
use serde::Serialize;

// ── Model dispatch ───────────────────────────────────────────────────

#[derive(ValueEnum, Clone, Debug)]
enum ModelArg {
    /// GLM-OCR vision-language model (supports all region types)
    GlmOcr,
    /// PP-FormulaNet encoder/decoder (formulas only)
    PpFormulanet,
}

/// Which region kinds a model can handle.
fn supported_kinds(model: &ModelArg) -> &'static [RegionKind] {
    match model {
        ModelArg::GlmOcr => &[
            RegionKind::DisplayFormula,
            RegionKind::InlineFormula,
            RegionKind::Table,
            RegionKind::Text,
            RegionKind::Title,
            RegionKind::Abstract,
            RegionKind::Algorithm,
        ],
        ModelArg::PpFormulanet => &[
            RegionKind::DisplayFormula,
            RegionKind::InlineFormula,
        ],
    }
}

/// Default region types when none specified.
fn default_region_types(model: &ModelArg) -> Vec<RegionKind> {
    match model {
        ModelArg::GlmOcr => vec![RegionKind::DisplayFormula, RegionKind::InlineFormula],
        ModelArg::PpFormulanet => vec![RegionKind::DisplayFormula, RegionKind::InlineFormula],
    }
}

/// Map region kind to GLM-OCR prompt.
fn prompt_for_kind(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::DisplayFormula | RegionKind::InlineFormula => "Formula Recognition:",
        RegionKind::Table => "Table Recognition:",
        _ => "Text Recognition:",
    }
}

enum Predictor {
    GlmOcr {
        model_paths: models::GlmOcrModelPaths,
        backend: Backend,
    },
    PpFormulanet(FormulaPredictor),
}

impl Predictor {
    /// Run OCR on a batch of regions that share a prompt/kind group.
    fn predict(
        &self,
        images: &[DynamicImage],
        kind: RegionKind,
    ) -> Result<Vec<String>, ExtractError> {
        match self {
            Self::GlmOcr { model_paths, backend } => {
                let config = GlmOcrConfig {
                    prompt: prompt_for_kind(kind).to_string(),
                    backend: *backend,
                };
                let predictor =
                    models::build_glm_ocr_predictor_with_config(model_paths, config)?;
                Ok(predictor.predict(images)?.into_iter().map(|fr| fr.latex).collect())
            }
            Self::PpFormulanet(p) => {
                Ok(p.predict(images)?.into_iter().map(|fr| fr.latex).collect())
            }
        }
    }
}

// ── Data types ───────────────────────────────────────────────────────

/// Output result per region.
#[derive(Serialize)]
struct ResultEntry {
    id: String,
    kind: RegionKind,
    page: u32,
    confidence: f32,
    bbox_px: [f32; 4],
    width: u32,
    height: u32,
    latex: String,
}

// ── CLI ──────────────────────────────────────────────────────────────

/// Parse a single region kind from string.
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
        _ => {
            eprintln!("Unknown region type: {s}. Use: Algorithm, DisplayFormula, Table, Text, etc.");
            std::process::exit(1);
        }
    }
}

/// Parse a comma-separated list of region kinds.
fn parse_region_kinds(s: &str) -> Vec<RegionKind> {
    s.split(',')
        .map(|part| parse_region_kind(part.trim()))
        .collect()
}

#[derive(Parser)]
#[command(name = "run_ocr", about = "Run OCR on layout-detected regions from a PDF")]
struct Cli {
    /// Path to the input PDF file
    pdf: PathBuf,

    /// Output directory for results JSON
    #[arg(short, long)]
    output: PathBuf,

    /// OCR model to use
    #[arg(long, value_name = "MODEL", default_value = "glm-ocr")]
    model: ModelArg,

    /// Region type(s) to process (comma-separated, e.g. "DisplayFormula,InlineFormula,Text")
    /// Default: DisplayFormula,InlineFormula
    #[arg(long)]
    region_type: Option<String>,

    /// Filter to a specific page (1-indexed)
    #[arg(long, short = 'p')]
    page: Option<u32>,

    /// Max number of regions to process per kind
    #[arg(long)]
    limit: Option<usize>,

    /// Layout detection confidence threshold
    #[arg(long, default_value = "0.3")]
    confidence: f32,

    /// DPI for rendering pages
    #[arg(long, default_value = "144")]
    dpi: u32,

    /// Enable benchmark mode (multiple timed runs with summary)
    #[arg(long)]
    bench: bool,

    /// Number of timed runs per image in bench mode
    #[arg(long, default_value = "2")]
    runs: usize,

    /// Model cache directory
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,

    /// Inference backend for GLM-OCR: cuda, coreml, cpu, or auto
    #[arg(long, default_value = "auto")]
    backend: String,
}

// ── Helpers ──────────────────────────────────────────────────────────

fn progress(msg: &str) {
    eprint!("\r{:<60}", msg);
    std::io::stderr().flush().ok();
}

fn std_dev(times: &[f64]) -> f64 {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

fn median_of(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

struct Timing {
    kind: RegionKind,
    times_ms: Vec<f64>,
}

/// A detected region with its cropped image.
struct CroppedRegion {
    id: String,
    kind: RegionKind,
    page: u32,
    confidence: f32,
    bbox_px: [f32; 4],
    width: u32,
    height: u32,
    image: DynamicImage,
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    // Determine target region kinds
    let target_kinds = match &cli.region_type {
        Some(s) => parse_region_kinds(s),
        None => default_region_types(&cli.model),
    };

    // Validate region kinds against model capabilities
    let supported = supported_kinds(&cli.model);
    for kind in &target_kinds {
        if !supported.contains(kind) {
            eprintln!(
                "Error: {:?} does not support {:?} regions. Supported: {:?}",
                cli.model, kind, supported
            );
            std::process::exit(1);
        }
    }

    let model_label = match cli.model {
        ModelArg::GlmOcr => "GLM-OCR",
        ModelArg::PpFormulanet => "PP-FormulaNet",
    };

    // Init ORT + pdfium
    models::init_ort_runtime().expect("ORT runtime init");
    let pdfium = pdf::load_pdfium(None).expect("load pdfium");

    // Load layout model
    let cache_dir = models::layout_cache_dir(cli.model_cache_dir.as_deref());
    std::fs::create_dir_all(&cache_dir).expect("create cache dir");
    let layout_path = models::ensure_layout_model(&cache_dir).expect("layout model");
    let layout = models::build_layout_detector(&layout_path).expect("layout detector");

    // Phase 1: Layout detection — render pages, detect regions, crop
    progress("Layout detection...");
    let cropped = run_layout(
        &pdfium,
        &cli.pdf,
        &layout,
        &target_kinds,
        cli.page,
        cli.limit,
        cli.confidence,
        cli.dpi,
    );

    if cropped.is_empty() {
        eprintln!("No {:?} regions found", target_kinds);
        std::process::exit(1);
    }

    eprintln!(
        "\r{} {:?} regions from {} [{}]{}",
        cropped.len(),
        target_kinds,
        cli.pdf.display(),
        model_label,
        " ".repeat(20),
    );

    // Phase 2: Load OCR model
    progress("Loading OCR model...");
    let predictor = match cli.model {
        ModelArg::GlmOcr => {
            let paths = models::ensure_glm_ocr_models_standalone(cli.model_cache_dir.as_deref())
                .expect("GLM-OCR model files");
            Predictor::GlmOcr {
                model_paths: paths,
                backend: Backend::from_str_loose(&cli.backend),
            }
        }
        ModelArg::PpFormulanet => {
            let paths = models::ensure_pp_formulanet_models_standalone(
                cli.model_cache_dir.as_deref(),
            )
            .expect("PP-FormulaNet model files");
            Predictor::PpFormulanet(
                FormulaPredictor::new(&paths.encoder, &paths.decoder, &paths.tokenizer)
                    .expect("FormulaPredictor::new"),
            )
        }
    };

    // Warmup
    {
        progress("warmup...");
        let _ = predictor
            .predict(std::slice::from_ref(&cropped[0].image), cropped[0].kind)
            .expect("warmup predict");
    }

    // Phase 3: Run OCR
    std::fs::create_dir_all(&cli.output).expect("create output dir");
    let mut results: Vec<ResultEntry> = Vec::new();
    let mut timings: Vec<Timing> = Vec::new();

    for (i, region) in cropped.iter().enumerate() {
        progress(&format!("{:?}: {} of {}", region.kind, i + 1, cropped.len()));

        let runs = if cli.bench { cli.runs } else { 1 };
        let mut times_ms = Vec::with_capacity(runs);
        let mut last_output = String::new();

        for _ in 0..runs {
            let t0 = Instant::now();
            let result = predictor
                .predict(std::slice::from_ref(&region.image), region.kind)
                .expect("predict");
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_output = result.into_iter().next().unwrap();
        }

        if cli.bench {
            timings.push(Timing {
                kind: region.kind,
                times_ms,
            });
        }

        results.push(ResultEntry {
            id: region.id.clone(),
            kind: region.kind,
            page: region.page,
            confidence: region.confidence,
            bbox_px: region.bbox_px,
            width: region.width,
            height: region.height,
            latex: last_output,
        });
    }

    progress(&format!("{} regions done.", cropped.len()));
    eprintln!();

    // Write results JSON
    let results_path = cli.output.join("results.json");
    let json = serde_json::to_string_pretty(&results).expect("serialize results");
    std::fs::write(&results_path, &json).expect("write results.json");
    eprintln!("Results: {}", results_path.display());

    if cli.bench {
        print_summary(&timings);
    }
}

// ── Layout detection + cropping ──────────────────────────────────────

fn run_layout(
    pdfium: &pdfium_render::prelude::Pdfium,
    pdf_path: &std::path::Path,
    layout: &LayoutDetector,
    target_kinds: &[RegionKind],
    page_filter: Option<u32>,
    limit: Option<usize>,
    confidence: f32,
    dpi: u32,
) -> Vec<CroppedRegion> {
    let doc = pdfium
        .load_pdf_from_file(pdf_path, None)
        .unwrap_or_else(|e| {
            eprintln!("Failed to load PDF: {e}");
            std::process::exit(1);
        });

    let total_pages = doc.pages().len() as u32;
    let page_indices: Vec<u32> = if let Some(p) = page_filter {
        if p == 0 || p > total_pages {
            eprintln!("Page {p} out of range (document has {total_pages} pages)");
            std::process::exit(1);
        }
        vec![p - 1]
    } else {
        (0..total_pages).collect()
    };

    let scale = dpi as f32 / 72.0;
    let mut cropped = Vec::new();
    let mut kind_counts: HashMap<RegionKind, usize> = HashMap::new();

    for &page_idx in &page_indices {
        let page_num = page_idx + 1;
        progress(&format!("Layout: page {page_num}/{total_pages}"));

        let page = doc
            .pages()
            .get(page_idx as u16)
            .unwrap_or_else(|e| {
                eprintln!("Failed to get page {page_num}: {e}");
                std::process::exit(1);
            });

        let width_pt = page.width().value;
        let height_pt = page.height().value;
        let page_image = pdf::render_page(&page, dpi).expect("render page");
        let detected = layout.detect(&page_image, confidence).expect("layout detect");

        for (det_idx, det) in detected.iter().enumerate() {
            if !target_kinds.contains(&det.kind) {
                continue;
            }
            if let Some(limit) = limit {
                let count = kind_counts.entry(det.kind).or_insert(0);
                if *count >= limit {
                    continue;
                }
                *count += 1;
            }

            let bbox_pt = [
                det.bbox_px[0] / scale,
                det.bbox_px[1] / scale,
                det.bbox_px[2] / scale,
                det.bbox_px[3] / scale,
            ];
            let crop = figure::crop_region(&page_image, bbox_pt, width_pt, height_pt, dpi);
            let (w, h) = (crop.width(), crop.height());

            cropped.push(CroppedRegion {
                id: format!("p{}_{}", page_num, det_idx),
                kind: det.kind,
                page: page_num,
                confidence: det.confidence,
                bbox_px: det.bbox_px,
                width: w,
                height: h,
                image: crop,
            });
        }
    }

    cropped
}

// ── Benchmark summary ────────────────────────────────────────────────

fn print_summary(timings: &[Timing]) {
    eprintln!();

    let mut kind_times: HashMap<RegionKind, Vec<f64>> = HashMap::new();
    for t in timings {
        let median = median_of(&t.times_ms);
        kind_times.entry(t.kind).or_default().push(median);
    }

    eprintln!(
        "{:<20} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Kind", "Count", "Median", "Min", "Max", "StdDev", "Total"
    );
    eprintln!("{}", "-".repeat(80));

    let mut kinds: Vec<_> = kind_times.keys().copied().collect();
    kinds.sort_by_key(|k| format!("{:?}", k));

    let mut grand_total_ms = 0.0;
    let mut grand_count = 0;

    for kind in &kinds {
        let mut times = kind_times[kind].clone();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let count = times.len();
        let med = median_of(&times);
        let min = times[0];
        let max = times[times.len() - 1];
        let sd = std_dev(&times);
        let total: f64 = times.iter().sum();

        grand_total_ms += total;
        grand_count += count;

        eprintln!(
            "{:<20} {:>6} {:>8.0}ms {:>8.0}ms {:>8.0}ms {:>8.1}ms {:>7.1}s",
            format!("{:?}", kind),
            count,
            med,
            min,
            max,
            sd,
            total / 1000.0
        );
    }

    eprintln!("{}", "-".repeat(80));
    eprintln!(
        "{:<20} {:>6} {:>49} {:>7.1}s",
        "Total",
        grand_count,
        "",
        grand_total_ms / 1000.0
    );
}
