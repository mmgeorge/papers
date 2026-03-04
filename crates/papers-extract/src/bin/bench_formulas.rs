//! Benchmark PP-FormulaNet formula prediction on regions from a dump directory.
//!
//! Usage:
//!   bench_formulas data/dumps/vbd -o data/results/vbd-ppformula
//!   bench_formulas data/dumps/vbd -o data/results/vbd-ppformula --runs 3
//!   bench_formulas data/dumps/vbd -o data/results/vbd-ppformula --page 5

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use papers_extract::formula::FormulaPredictor;
use papers_extract::models;
use papers_extract::RegionKind;
use serde::{Deserialize, Serialize};

/// Layout entry from dump's layout.json.
#[derive(Deserialize)]
#[allow(dead_code)]
struct LayoutEntry {
    id: String,
    kind: RegionKind,
    page: u32,
    confidence: f32,
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
    text: String,
}

#[derive(Parser)]
#[command(name = "bench_formulas", about = "Benchmark PP-FormulaNet on formula regions from a dump")]
struct Cli {
    /// Path to dump directory (created by `dump` binary)
    dump_dir: PathBuf,

    /// Output directory for results JSON
    #[arg(short, long)]
    output: PathBuf,

    /// Number of timed runs (excludes warmup)
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Filter to a specific page (1-indexed)
    #[arg(long)]
    page: Option<u32>,

    /// Model cache directory
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,
}

fn std_dev(times: &[f64]) -> f64 {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

fn main() {
    let cli = Cli::parse();

    // Load layout.json and filter to formula regions
    let json_path = cli.dump_dir.join("layout.json");
    let json_str = std::fs::read_to_string(&json_path).expect("read layout.json");
    let entries: Vec<LayoutEntry> = serde_json::from_str(&json_str).expect("parse layout.json");

    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|e| e.kind == RegionKind::DisplayFormula || e.kind == RegionKind::InlineFormula)
        .filter(|e| cli.page.map_or(true, |p| e.page == p))
        .collect();

    if filtered.is_empty() {
        eprintln!("No formula regions found in {}", cli.dump_dir.display());
        std::process::exit(1);
    }

    let mut paths = Vec::new();
    let mut images = Vec::new();
    for entry in &filtered {
        let img_path = cli.dump_dir.join(entry.image.trim_start_matches("./"));
        let image = image::open(&img_path)
            .unwrap_or_else(|err| panic!("load {}: {err}", img_path.display()));
        paths.push(img_path);
        images.push(image);
    }

    eprintln!("Loaded {} formula images from {}", images.len(), cli.dump_dir.display());

    // Init ORT + load models
    models::init_ort_runtime().expect("ORT runtime init");
    let cache_dir = cli
        .model_cache_dir
        .unwrap_or_else(models::default_cache_dir);
    let model_paths =
        models::ensure_models(papers_extract::Quality::Fast, &cache_dir).expect("Model download");

    eprintln!("Loading formula predictor...");
    let predictor = FormulaPredictor::new(
        &model_paths.formula_encoder,
        &model_paths.formula_decoder,
        &model_paths.formula_tokenizer,
    )
    .expect("FormulaPredictor::new");
    eprintln!("Formula predictor ready\n");

    // Benchmark runs
    eprintln!(
        "{:>4}  {:>10}  {:>10}  {:>10}",
        "Run", "Total", "Per-formula", "Formulas"
    );
    eprintln!("{}", "-".repeat(45));

    let mut times = Vec::with_capacity(cli.runs);
    let mut last_results: Vec<String> = Vec::new();

    for run in 1..=cli.runs {
        let t0 = Instant::now();
        let results = predictor.predict(&images).expect("predict failed");
        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let per_formula = ms / images.len() as f64;
        times.push(ms);
        last_results = results;

        eprintln!(
            "{:>4}  {:>8.0}ms  {:>8.1}ms  {:>10}",
            run, ms, per_formula, last_results.len()
        );
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let sd = std_dev(&times);

    eprintln!("{}", "-".repeat(45));
    eprintln!(
        "Median: {:.0}ms ({:.1}ms/formula)  Min: {:.0}ms  Max: {:.0}ms  StdDev: {:.1}ms",
        median,
        median / images.len() as f64,
        min,
        max,
        sd
    );
    eprintln!("{} formulas, {} runs", images.len(), cli.runs);

    // Write results JSON
    std::fs::create_dir_all(&cli.output).expect("create output dir");
    let results: Vec<ResultEntry> = filtered
        .iter()
        .zip(last_results.iter())
        .map(|(entry, text)| ResultEntry {
            id: entry.id.clone(),
            kind: entry.kind,
            page: entry.page,
            text: text.clone(),
        })
        .collect();
    let results_path = cli.output.join("results.json");
    let json = serde_json::to_string_pretty(&results).expect("serialize results");
    std::fs::write(&results_path, &json).expect("write results.json");
    eprintln!("\nResults: {}", results_path.display());
}
