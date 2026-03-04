//! Run GLM-OCR on regions from a dump directory.
//!
//! By default runs each region once and writes results.json.
//! With --bench, runs multiple timed iterations and prints a summary.
//!
//! Usage:
//!   run_glm_ocr data/dumps/vbd -o data/results/vbd-glm --region-type DisplayFormula
//!   run_glm_ocr data/dumps/vbd -o data/results/vbd-glm --region-type Text --limit 10
//!   run_glm_ocr data/dumps/vbd -o data/results/vbd-glm --region-type "Text,DisplayFormula" --bench

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use papers_extract::glm_ocr::GlmOcrConfig;
use papers_extract::models;
use papers_extract::RegionKind;
use serde::{Deserialize, Serialize};

/// Map region kind to the appropriate GLM-OCR prompt.
fn prompt_for_kind(kind: RegionKind) -> &'static str {
    match kind {
        RegionKind::DisplayFormula | RegionKind::InlineFormula => "Formula Recognition:",
        RegionKind::Table => "Table Recognition:",
        _ => "Text Recognition:",
    }
}

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
        _ => panic!("Unknown region type: {s}. Use: Algorithm, DisplayFormula, Table, Text, etc."),
    }
}

/// Parse a comma-separated list of region kinds.
fn parse_region_kinds(s: &str) -> Vec<RegionKind> {
    s.split(',')
        .map(|part| parse_region_kind(part.trim()))
        .collect()
}

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
#[command(name = "run_glm_ocr", about = "Run GLM-OCR on regions from a dump directory")]
struct Cli {
    /// Path to dump directory (created by `dump` binary)
    dump_dir: PathBuf,

    /// Output directory for results JSON
    #[arg(short, long)]
    output: PathBuf,

    /// Region type(s) to process (comma-separated, e.g. "Text,DisplayFormula,Algorithm")
    #[arg(long, default_value = "Algorithm")]
    region_type: String,

    /// Filter to a specific page (1-indexed)
    #[arg(long)]
    page: Option<u32>,

    /// Max number of regions to process per kind
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

fn std_dev(times: &[f64]) -> f64 {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

/// Per-image timing record for the summary.
struct Timing {
    kind: RegionKind,
    times_ms: Vec<f64>,
}

fn main() {
    let cli = Cli::parse();
    let target_kinds = parse_region_kinds(&cli.region_type);

    // Load layout.json
    let json_path = cli.dump_dir.join("layout.json");
    let json_str = std::fs::read_to_string(&json_path).expect("read layout.json");
    let entries: Vec<LayoutEntry> = serde_json::from_str(&json_str).expect("parse layout.json");

    let filtered: Vec<_> = entries
        .into_iter()
        .filter(|e| target_kinds.contains(&e.kind))
        .filter(|e| cli.page.map_or(true, |p| e.page == p))
        .collect();

    if filtered.is_empty() {
        eprintln!("No {:?} regions found in {}", target_kinds, cli.dump_dir.display());
        std::process::exit(1);
    }

    // Init ORT
    models::init_ort_runtime().expect("ORT runtime init");
    let cache_dir = cli
        .model_cache_dir
        .unwrap_or_else(models::default_cache_dir);
    let model_paths = models::ensure_glm_ocr_models(&cache_dir).expect("GLM-OCR model files");

    // Group entries by prompt, applying --limit per kind.
    let mut groups: Vec<(&'static str, Vec<&LayoutEntry>)> = Vec::new();
    let mut kind_counts: HashMap<RegionKind, usize> = HashMap::new();
    for entry in &filtered {
        if let Some(limit) = cli.limit {
            let count = kind_counts.entry(entry.kind).or_insert(0);
            if *count >= limit {
                continue;
            }
            *count += 1;
        }
        let prompt = prompt_for_kind(entry.kind);
        if let Some(group) = groups.iter_mut().find(|(p, _)| *p == prompt) {
            group.1.push(entry);
        } else {
            groups.push((prompt, vec![entry]));
        }
    }

    let total: usize = groups.iter().map(|(_, v)| v.len()).sum();

    // Warmup: single prediction to prime ORT/CUDA
    {
        let first = groups[0].1[0];
        let img_path = cli.dump_dir.join(first.image.trim_start_matches("./"));
        let image = image::open(&img_path).expect("load warmup image");
        let prompt = groups[0].0;
        let config = GlmOcrConfig { prompt: prompt.to_string() };
        let predictor = models::build_glm_ocr_predictor_with_config(&model_paths, config)
            .expect("GLM-OCR init");
        progress("warmup...");
        let _ = predictor.predict(std::slice::from_ref(&image)).expect("warmup predict");
    }

    // Create output directory
    std::fs::create_dir_all(&cli.output).expect("create output dir");
    let mut results: Vec<ResultEntry> = Vec::new();
    let mut timings: Vec<Timing> = Vec::new();
    for (prompt, group_entries) in &groups {
        let config = GlmOcrConfig { prompt: prompt.to_string() };
        let predictor = models::build_glm_ocr_predictor_with_config(&model_paths, config)
            .expect("GLM-OCR init");

        for (i, entry) in group_entries.iter().enumerate() {
            let msg = format!("{:?}: {} of {}", entry.kind, i + 1, group_entries.len());
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
                text: last_output,
            });
        }
    }

    progress(&format!("{} regions done.", total));
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
    let mut kind_times: HashMap<RegionKind, Vec<f64>> = HashMap::new();
    for t in timings {
        let median = median_of(&t.times_ms);
        kind_times.entry(t.kind).or_default().push(median);
    }

    eprintln!("{:<20} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}", "Kind", "Count", "Median", "Min", "Max", "StdDev", "Total");
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
            format!("{:?}", kind), count, med, min, max, sd, total / 1000.0
        );
    }

    eprintln!("{}", "-".repeat(80));
    eprintln!(
        "{:<20} {:>6} {:>49} {:>7.1}s",
        "Total", grand_count, "", grand_total_ms / 1000.0
    );
}

fn median_of(times: &[f64]) -> f64 {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}
