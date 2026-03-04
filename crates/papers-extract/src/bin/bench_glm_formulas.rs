//! Benchmark GLM-OCR formula prediction in isolation.
//!
//! Usage:
//!   # First dump formulas from a PDF:
//!   papers-extract data/vbd.pdf -o test-extract/ --dump-formulas
//!
//!   # Then benchmark:
//!   bench_glm_formulas test-extract/formulas/
//!   bench_glm_formulas test-extract/formulas/ --runs 5
//!   bench_glm_formulas test-extract/formulas/ --dump

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use image::DynamicImage;
use papers_extract::models;

#[derive(Parser)]
#[command(name = "bench_glm_formulas", about = "Benchmark GLM-OCR formula prediction")]
struct Cli {
    /// Directory containing formula PNG images
    formulas_dir: PathBuf,

    /// Number of benchmark runs (after warmup)
    #[arg(long, default_value = "5")]
    runs: usize,

    /// Dump all formula outputs (filename\tlatex) to stdout for comparison
    #[arg(long)]
    dump: bool,

    /// Model cache directory
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    // Load formula images
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&cli.formulas_dir)
        .expect("Cannot read formulas directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "png"))
        .collect();
    paths.sort();

    if paths.is_empty() {
        eprintln!("No PNG files found in {}", cli.formulas_dir.display());
        std::process::exit(1);
    }

    let images: Vec<DynamicImage> = paths
        .iter()
        .map(|p| image::open(p).expect(&format!("Cannot open {}", p.display())))
        .collect();

    eprintln!("Loaded {} formula images", images.len());

    // Init ORT
    models::init_ort_runtime().expect("ORT runtime init");

    // Load GLM-OCR models
    let cache_dir = cli
        .model_cache_dir
        .unwrap_or_else(models::default_cache_dir);
    let model_paths =
        models::ensure_glm_ocr_models(&cache_dir).expect("GLM-OCR model files");

    eprintln!("Loading GLM-OCR predictor...");
    let predictor = models::build_glm_ocr_predictor(&model_paths)
        .expect("GlmOcrPredictor::new");
    eprintln!("GLM-OCR predictor ready (includes warmup)\n");

    // Benchmark runs
    eprintln!(
        "{:>4}  {:>10}  {:>10}  {:>10}",
        "Run", "Total", "Per-formula", "Formulas"
    );
    eprintln!("{}", "-".repeat(45));

    let mut times = Vec::with_capacity(cli.runs);

    for run in 1..=cli.runs {
        let t0 = Instant::now();
        let results = predictor.predict(&images).expect("predict failed");
        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        let per_formula = ms / images.len() as f64;
        times.push(ms);

        eprintln!(
            "{:>4}  {:>8.0}ms  {:>8.1}ms  {:>10}",
            run,
            ms,
            per_formula,
            results.len()
        );
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let min = times[0];
    let max = times[times.len() - 1];

    eprintln!("{}", "-".repeat(45));
    eprintln!(
        "Median: {:.0}ms ({:.1}ms/formula)  Min: {:.0}ms  Max: {:.0}ms",
        median,
        median / images.len() as f64,
        min,
        max
    );
    eprintln!(
        "{} formulas, {} runs",
        images.len(),
        cli.runs
    );

    // Dump all outputs or print one sample
    if cli.dump {
        let results = predictor.predict(&images).expect("dump predict");
        for (path, latex) in paths.iter().zip(results.iter()) {
            println!("{}\t{}", path.file_name().unwrap().to_string_lossy(), latex);
        }
    } else {
        let sample = predictor.predict(&images[..1]).expect("sample predict");
        eprintln!("\nSample ({}):", paths[0].file_name().unwrap().to_string_lossy());
        eprintln!("  {}", sample[0]);
    }
}
