//! Run TableFormer on table regions from a dump directory.
//!
//! By default runs each region once and writes results.json.
//! With --bench, runs multiple timed iterations and prints a summary.
//!
//! Usage:
//!   run_tableformer data/dumps/vbd -o data/results/vbd-tf
//!   run_tableformer data/dumps/vbd -o data/results/vbd-tf --limit 5
//!   run_tableformer data/dumps/vbd -o data/results/vbd-tf --bench --runs 3

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
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
#[command(name = "run_tableformer", about = "Run TableFormer on table regions from a dump directory")]
struct Cli {
    /// Path to dump directory (created by `dump` binary)
    dump_dir: PathBuf,

    /// Output directory for results JSON
    #[arg(short, long)]
    output: PathBuf,

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

fn progress(msg: &str) {
    eprint!("\r{:<50}", msg);
    std::io::stderr().flush().ok();
}

fn main() {
    let cli = Cli::parse();

    // Load layout.json
    let json_path = cli.dump_dir.join("layout.json");
    let json_str = std::fs::read_to_string(&json_path).expect("read layout.json");
    let entries: Vec<LayoutEntry> = serde_json::from_str(&json_str).expect("parse layout.json");

    let mut filtered: Vec<_> = entries
        .into_iter()
        .filter(|e| e.kind == RegionKind::Table)
        .filter(|e| cli.page.map_or(true, |p| e.page == p))
        .collect();

    if let Some(limit) = cli.limit {
        filtered.truncate(limit);
    }

    if filtered.is_empty() {
        eprintln!("No Table regions found in {}", cli.dump_dir.display());
        std::process::exit(1);
    }

    eprintln!("{} table regions to process", filtered.len());

    // Init ORT runtime
    models::init_ort_runtime().expect("ORT runtime init");
    let cache_dir = cli
        .model_cache_dir
        .unwrap_or_else(models::default_cache_dir);
    let tf_paths = models::ensure_tableformer_models(&cache_dir).expect("TableFormer model files");
    let predictor =
        models::build_tableformer_predictor(&tf_paths).expect("TableFormer init");

    // Warmup
    {
        let first = &filtered[0];
        let img_path = cli.dump_dir.join(first.image.trim_start_matches("./"));
        let image = image::open(&img_path).expect("load warmup image");
        progress("warmup...");
        let _ = predictor.predict_one(&image).expect("warmup predict");
    }

    // Create output directory
    std::fs::create_dir_all(&cli.output).expect("create output dir");
    let mut results: Vec<ResultEntry> = Vec::new();
    let mut timings: Vec<Vec<f64>> = Vec::new();

    for (i, entry) in filtered.iter().enumerate() {
        let msg = format!("Table: {} of {}", i + 1, filtered.len());
        progress(&msg);

        let img_path = cli.dump_dir.join(entry.image.trim_start_matches("./"));
        let image = image::open(&img_path)
            .unwrap_or_else(|err| panic!("load {}: {err}", img_path.display()));

        let runs = if cli.bench { cli.runs } else { 1 };
        let mut times_ms = Vec::with_capacity(runs);
        let mut last_output = String::new();

        for _ in 0..runs {
            let t0 = Instant::now();
            let result = predictor.predict_one(&image).expect("predict");
            times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            last_output = result.html;
        }

        if cli.bench {
            timings.push(times_ms);
        }

        results.push(ResultEntry {
            id: entry.id.clone(),
            kind: entry.kind,
            page: entry.page,
            text: last_output,
        });
    }

    progress(&format!("{} tables done.", filtered.len()));
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

fn print_summary(timings: &[Vec<f64>]) {
    eprintln!();

    let medians: Vec<f64> = timings.iter().map(|t| median_of(t)).collect();
    let count = medians.len();
    let med = median_of(&medians);
    let min = medians.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = medians.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sd = std_dev(&medians);
    let total: f64 = medians.iter().sum();

    eprintln!(
        "{:<10} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Kind", "Count", "Median", "Min", "Max", "StdDev", "Total"
    );
    eprintln!("{}", "-".repeat(70));
    eprintln!(
        "{:<10} {:>6} {:>8.0}ms {:>8.0}ms {:>8.0}ms {:>8.1}ms {:>7.1}s",
        "Table", count, med, min, max, sd, total / 1000.0
    );
}

fn median_of(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn std_dev(times: &[f64]) -> f64 {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}
