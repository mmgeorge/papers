//! Run GLM-OCR on regions from a dump directory.
//!
//! By default runs each region once and writes results.json.
//! With --bench, adds a warmup run and multiple timed iterations with stats.
//!
//! Usage:
//!   run_glm_ocr data/dumps/vbd -o data/results/vbd-glm --region-type DisplayFormula
//!   run_glm_ocr data/dumps/vbd -o data/results/vbd-glm --region-type Text --limit 10
//!   run_glm_ocr data/dumps/vbd -o data/results/vbd-glm --region-type Algorithm --bench --runs 5

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

    /// Max number of regions to process
    #[arg(long)]
    limit: Option<usize>,

    /// Enable benchmark mode (warmup + multiple timed runs with stats)
    #[arg(long)]
    bench: bool,

    /// Number of timed runs per image in bench mode (default 5)
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Print OCR output to stdout
    #[arg(long)]
    dump: bool,

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

    // Group entries by prompt so we build one predictor per prompt,
    // applying --limit per kind.
    let mut groups: Vec<(&'static str, Vec<&LayoutEntry>)> = Vec::new();
    let mut kind_counts: std::collections::HashMap<RegionKind, usize> = std::collections::HashMap::new();
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
    eprintln!(
        "{} regions ({:?}) from {}",
        total,
        target_kinds,
        cli.dump_dir.display()
    );

    // Create output directory
    std::fs::create_dir_all(&cli.output).expect("create output dir");
    let mut results: Vec<ResultEntry> = Vec::new();

    for (prompt, entries) in &groups {
        eprintln!("\n--- prompt={:?} ({} regions) ---", prompt, entries.len());
        let config = GlmOcrConfig { prompt: prompt.to_string() };
        let predictor = models::build_glm_ocr_predictor_with_config(&model_paths, config)
            .expect("GLM-OCR init");

        for entry in entries {
            let img_path = cli.dump_dir.join(entry.image.trim_start_matches("./"));
            let image = image::open(&img_path)
                .unwrap_or_else(|err| panic!("load {}: {err}", img_path.display()));

            eprintln!(
                "=== {} {:?} (p{}, {}x{}, conf={:.2}) ===",
                entry.id, entry.kind, entry.page,
                image.width(), image.height(), entry.confidence,
            );

            let output = if cli.bench {
                run_bench(&predictor, &image, &entry.id, cli.runs, cli.dump)
            } else {
                run_once(&predictor, &image, &entry.id, cli.dump)
            };

            results.push(ResultEntry {
                id: entry.id.clone(),
                kind: entry.kind,
                page: entry.page,
                text: output,
            });

            eprintln!();
        }
    }

    // Write results JSON
    let results_path = cli.output.join("results.json");
    let json = serde_json::to_string_pretty(&results).expect("serialize results");
    std::fs::write(&results_path, &json).expect("write results.json");
    eprintln!("Results: {} ({} regions)", results_path.display(), results.len());
}

/// Single run, no warmup.
fn run_once(
    predictor: &papers_extract::glm_ocr::GlmOcrPredictor,
    image: &image::DynamicImage,
    id: &str,
    dump: bool,
) -> String {
    let t0 = Instant::now();
    let result = predictor.predict(std::slice::from_ref(image)).expect("predict");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let text = &result[0];
    eprintln!("  {:.0}ms  {} ch", ms, text.len());

    if dump {
        println!("--- {} ---", id);
        println!("{}", text);
        println!();
    }

    text.clone()
}

/// Benchmark mode: warmup + multiple timed runs with stats.
fn run_bench(
    predictor: &papers_extract::glm_ocr::GlmOcrPredictor,
    image: &image::DynamicImage,
    id: &str,
    runs: usize,
    dump: bool,
) -> String {
    // Warmup
    eprintln!("  warmup...");
    let _ = predictor.predict(std::slice::from_ref(image)).expect("warmup predict");

    eprintln!("{:>4}  {:>10}  {:>10}", "Run", "Time", "Chars");
    eprintln!("{}", "-".repeat(35));

    let mut times = Vec::with_capacity(runs);
    let mut last_output = String::new();

    for run in 1..=runs {
        let t0 = Instant::now();
        let result = predictor.predict(std::slice::from_ref(image)).expect("predict");
        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let text = &result[0];
        times.push(ms);
        last_output = text.clone();

        eprintln!("{:>4}  {:>8.0}ms  {:>8} ch", run, ms, text.len());
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];
    let sd = std_dev(&times);

    eprintln!("{}", "-".repeat(35));
    eprintln!(
        "Median: {:.0}ms  Min: {:.0}ms  Max: {:.0}ms  StdDev: {:.1}ms",
        median, min, max, sd
    );

    if dump {
        println!("--- {} ---", id);
        println!("{}", last_output);
        println!();
    }

    last_output
}
