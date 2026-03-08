//! Run formula recognition on formula regions from a dump directory.
//!
//! Supports both PP-FormulaNet and GLM-OCR models via --model flag.
//! By default runs each region once and writes results.json.
//! With --bench, runs multiple timed iterations and prints a summary.
//!
//! Usage:
//!   run_formulas data/dumps/vbd -o .temp/vbd-ppfn --model pp-formulanet
//!   run_formulas data/dumps/vbd -o .temp/vbd-glm --model glm-ocr
//!   run_formulas data/dumps/vbd -o .temp/vbd-ppfn --model pp-formulanet --bench --runs 3

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use image::DynamicImage;
use papers_extract::error::ExtractError;
use papers_extract::formula::FormulaPredictor;
use papers_extract::glm_ocr::GlmOcrPredictor;
use papers_extract::models;
use papers_extract::RegionKind;
use serde::{Deserialize, Serialize};

// ── Model dispatch ───────────────────────────────────────────────────

#[derive(ValueEnum, Clone, Debug)]
enum ModelArg {
    /// PP-FormulaNet encoder/decoder
    PpFormulanet,
    /// GLM-OCR vision-language model
    GlmOcr,
}

enum Predictor {
    PpFormulanet(FormulaPredictor),
    GlmOcr(GlmOcrPredictor),
}

impl Predictor {
    fn predict(&self, images: &[DynamicImage]) -> Result<Vec<String>, ExtractError> {
        match self {
            Self::PpFormulanet(p) => p.predict(images),
            Self::GlmOcr(p) => p.predict(images),
        }
    }
}

// ── Data types ───────────────────────────────────────────────────────

/// Layout entry from dump's layout.json.
#[derive(Deserialize)]
#[allow(dead_code)]
struct LayoutEntry {
    id: String,
    kind: RegionKind,
    page: u32,
    confidence: f32,
    bbox_px: [f32; 4],
    width: u32,
    height: u32,
    image: String,
}

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

#[derive(Parser)]
#[command(name = "run_formulas", about = "Run formula recognition on regions from a dump directory")]
struct Cli {
    /// Path to dump directory (created by `dump` binary)
    dump_dir: PathBuf,

    /// Output directory for results JSON
    #[arg(short, long)]
    output: PathBuf,

    /// Formula model to use
    #[arg(long, value_name = "MODEL", default_value = "glm-ocr")]
    model: ModelArg,

    /// Filter to a specific page (1-indexed)
    #[arg(long)]
    page: Option<u32>,

    /// Max number of regions to process
    #[arg(long)]
    limit: Option<usize>,

    /// Enable benchmark mode (multiple timed runs with summary)
    #[arg(long)]
    bench: bool,

    /// Number of timed runs per image in bench mode
    #[arg(long, default_value = "2")]
    runs: usize,

    /// Model cache directory
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,
}

// ── Helpers ──────────────────────────────────────────────────────────

fn progress(msg: &str) {
    eprint!("\r{:<50}", msg);
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

/// Per-image timing record for the summary.
struct Timing {
    kind: RegionKind,
    times_ms: Vec<f64>,
}

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();

    // Load layout.json and filter to formula regions
    let json_path = cli.dump_dir.join("layout.json");
    let json_str = std::fs::read_to_string(&json_path).expect("read layout.json");
    let entries: Vec<LayoutEntry> = serde_json::from_str(&json_str).expect("parse layout.json");

    let mut filtered: Vec<_> = entries
        .into_iter()
        .filter(|e| e.kind == RegionKind::DisplayFormula || e.kind == RegionKind::InlineFormula)
        .filter(|e| cli.page.map_or(true, |p| e.page == p))
        .collect();

    if let Some(limit) = cli.limit {
        filtered.truncate(limit);
    }

    if filtered.is_empty() {
        eprintln!("No formula regions found in {}", cli.dump_dir.display());
        std::process::exit(1);
    }

    let model_label = match cli.model {
        ModelArg::PpFormulanet => "PP-FormulaNet",
        ModelArg::GlmOcr => "GLM-OCR",
    };
    eprintln!("{} formula regions from {} [{}]", filtered.len(), cli.dump_dir.display(), model_label);

    // Init ORT + load model
    models::init_ort_runtime().expect("ORT runtime init");

    let predictor = match cli.model {
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
        ModelArg::GlmOcr => {
            let paths = models::ensure_glm_ocr_models_standalone(
                cli.model_cache_dir.as_deref(),
            )
            .expect("GLM-OCR model files");
            Predictor::GlmOcr(
                models::build_glm_ocr_predictor(&paths).expect("GlmOcrPredictor::new"),
            )
        }
    };

    // Warmup: single prediction to prime ORT/CUDA
    {
        let first = &filtered[0];
        let img_path = cli.dump_dir.join(first.image.trim_start_matches("./"));
        let image = image::open(&img_path).expect("load warmup image");
        progress("warmup...");
        let _ = predictor.predict(std::slice::from_ref(&image)).expect("warmup predict");
    }

    // Create output directory
    std::fs::create_dir_all(&cli.output).expect("create output dir");
    let mut results: Vec<ResultEntry> = Vec::new();
    let mut timings: Vec<Timing> = Vec::new();

    for (i, entry) in filtered.iter().enumerate() {
        let msg = format!("{:?}: {} of {}", entry.kind, i + 1, filtered.len());
        progress(&msg);

        let img_path = cli.dump_dir.join(entry.image.trim_start_matches("./"));
        let image = image::open(&img_path)
            .unwrap_or_else(|err| panic!("load {}: {err}", img_path.display()));

        let runs = if cli.bench { cli.runs } else { 1 };
        let mut times_ms = Vec::with_capacity(runs);
        let mut last_output = String::new();

        for _ in 0..runs {
            let t0 = Instant::now();
            let result = predictor.predict(std::slice::from_ref(&image)).expect("predict");
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_output = result.into_iter().next().unwrap();
        }

        if cli.bench {
            timings.push(Timing {
                kind: entry.kind,
                times_ms,
            });
        }

        results.push(ResultEntry {
            id: entry.id.clone(),
            kind: entry.kind,
            page: entry.page,
            confidence: entry.confidence,
            bbox_px: entry.bbox_px,
            width: entry.width,
            height: entry.height,
            latex: last_output,
        });
    }

    progress(&format!("{} regions done.", filtered.len()));
    eprintln!();

    // Write results JSON
    let results_path = cli.output.join("results.json");
    let json = serde_json::to_string_pretty(&results).expect("serialize results");
    std::fs::write(&results_path, &json).expect("write results.json");
    eprintln!("Results: {}", results_path.display());

    // Print benchmark summary
    if cli.bench {
        print_summary(&timings);
    }
}

fn print_summary(timings: &[Timing]) {
    eprintln!();

    // Per-kind aggregation
    let mut kind_times: std::collections::HashMap<RegionKind, Vec<f64>> =
        std::collections::HashMap::new();
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
