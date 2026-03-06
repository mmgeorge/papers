//! Dump layout-detected regions from a PDF as cropped images.
//!
//! Renders each page, runs PP-DocLayout detection, crops each region,
//! and saves images organized by type.  Writes `layout.json` with a
//! flat list of every detected region.
//!
//! Usage:
//!   dump data/vbd.pdf data/dumps/vbd
//!   dump data/vbd.pdf data/dumps/vbd --page 9
//!   dump data/vbd.pdf data/dumps/vbd --dpi 200 --confidence 0.25

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use image::DynamicImage;
use serde::Serialize;

use papers_extract::layout::DetectedRegion;
use papers_extract::models;
use papers_extract::output;
use papers_extract::RegionKind;

const DEFAULT_DPI: u32 = 150;
const DEFAULT_CONFIDENCE: f32 = 0.3;

#[derive(Parser)]
#[command(name = "dump", about = "Dump layout-detected regions as cropped images")]
struct Cli {
    /// Path to PDF file
    pdf: PathBuf,

    /// Output directory (e.g. data/dumps/vbd)
    output: PathBuf,

    /// Process only this page (1-indexed). If omitted, all pages.
    #[arg(long)]
    page: Option<u32>,

    /// DPI for rendering pages
    #[arg(long, default_value_t = DEFAULT_DPI)]
    dpi: u32,

    /// Layout detection confidence threshold
    #[arg(long, default_value_t = DEFAULT_CONFIDENCE)]
    confidence: f32,

    /// Model cache directory
    #[arg(long)]
    model_cache_dir: Option<PathBuf>,

    /// Path to pdfium library
    #[arg(long)]
    pdfium_path: Option<PathBuf>,
}

/// One entry in the flat layout.json list.
#[derive(Serialize)]
struct LayoutEntry {
    /// Unique id within this kind (e.g. "p3_2" for the 3rd region of this kind, on page 3)
    id: String,
    /// Region kind (e.g. "DisplayFormula")
    kind: RegionKind,
    /// 1-indexed page number
    page: u32,
    /// Bounding box in page pixels: [x1, y1, x2, y2]
    bbox_px: [f32; 4],
    /// Detection confidence
    confidence: f32,
    /// Reading order key
    order_key: f32,
    /// Width in pixels
    width: u32,
    /// Height in pixels
    height: u32,
    /// Relative path to cropped image (e.g. "./DisplayFormula/p3_2.png")
    image: String,
}

/// Convert RegionKind to a filesystem-safe directory name.
fn dir_name(kind: RegionKind) -> String {
    format!("{:?}", kind)
}

fn main() {
    let cli = Cli::parse();

    // Init ORT
    models::init_ort_runtime().expect("ORT runtime init");

    let cache_dir = cli
        .model_cache_dir
        .unwrap_or_else(models::default_cache_dir);

    // Load layout model
    eprintln!("Loading layout model...");
    let layout_path = models::ensure_layout_model(&cache_dir).expect("layout model file");
    let layout = models::build_layout_detector(&layout_path).expect("layout detector");

    // Load PDF
    eprintln!("Loading PDF: {}", cli.pdf.display());
    let pdfium =
        papers_extract::pdf::load_pdfium(cli.pdfium_path.as_deref()).expect("load pdfium");
    let doc = pdfium
        .load_pdf_from_file(&cli.pdf, None)
        .expect("load PDF");
    let total_pages = doc.pages().len() as u32;
    eprintln!("  {} pages", total_pages);

    // Determine page range
    let pages: Vec<u32> = if let Some(p) = cli.page {
        assert!(p >= 1 && p <= total_pages, "Page {p} out of range (1..{total_pages})");
        vec![p]
    } else {
        (1..=total_pages).collect()
    };

    // Create output directory
    std::fs::create_dir_all(&cli.output).expect("create output dir");

    let mut entries: Vec<LayoutEntry> = Vec::new();
    // Track per-kind counter across all pages for unique filenames
    let mut kind_counters: HashMap<String, usize> = HashMap::new();

    // Create layout/ dir for annotated page images
    let layout_dir = cli.output.join("layout");
    std::fs::create_dir_all(&layout_dir).expect("create layout dir");

    for &page_num in &pages {
        eprintln!("Page {}/{}...", page_num, total_pages);

        let page = doc
            .pages()
            .get((page_num - 1) as u16)
            .expect("get page");
        let page_image = papers_extract::pdf::render_page(&page, cli.dpi).expect("render page");
        let detected = layout
            .detect(&page_image, cli.confidence)
            .expect("layout detect");

        eprintln!("  {} regions", detected.len());

        // Build annotated page image with bounding boxes
        let mut annotated = page_image.to_rgba8();
        for region in &detected {
            output::draw_region_box(&mut annotated, region.kind, region.bbox_px, 2);
        }
        let annotated_path = layout_dir.join(format!("p{}.png", page_num));
        annotated.save(&annotated_path).expect("save annotated page");

        for region in &detected {
            let kind_dir = dir_name(region.kind);
            let counter = kind_counters.entry(kind_dir.clone()).or_insert(0);
            let id = format!("p{}_{}", page_num, counter);
            let filename = format!("{}.png", id);
            *counter += 1;

            // Ensure type directory exists
            let type_dir = cli.output.join(&kind_dir);
            std::fs::create_dir_all(&type_dir).expect("create type dir");

            // Crop and save
            let cropped = crop_region(&page_image, region);
            let save_path = type_dir.join(&filename);
            cropped.save(&save_path).expect("save crop");

            let rel_path = format!("./{}/{}", kind_dir, filename);
            let [x1, y1, x2, y2] = region.bbox_px;

            entries.push(LayoutEntry {
                id,
                kind: region.kind,
                page: page_num,
                bbox_px: region.bbox_px,
                confidence: region.confidence,
                order_key: region.order_key,
                width: (x2 - x1) as u32,
                height: (y2 - y1) as u32,
                image: rel_path,
            });
        }
    }

    // Write layout.json
    let json_path = cli.output.join("layout.json");
    let json = serde_json::to_string_pretty(&entries).expect("serialize layout");
    std::fs::write(&json_path, &json).expect("write layout.json");

    // Summary
    eprintln!("\nDumped {} regions to {}", entries.len(), cli.output.display());
    let mut summary: HashMap<String, usize> = HashMap::new();
    for e in &entries {
        *summary.entry(dir_name(e.kind)).or_insert(0) += 1;
    }
    let mut summary: Vec<_> = summary.into_iter().collect();
    summary.sort_by(|a, b| b.1.cmp(&a.1));
    for (kind, count) in &summary {
        eprintln!("  {:20} {}", kind, count);
    }
    eprintln!("layout.json: {}", json_path.display());
    eprintln!("layout images: {}", layout_dir.display());
}

fn crop_region(image: &DynamicImage, region: &DetectedRegion) -> DynamicImage {
    let [x1, y1, x2, y2] = region.bbox_px;
    let w = (x2 - x1) as u32;
    let h = (y2 - y1) as u32;
    image.crop_imm(x1 as u32, y1 as u32, w, h)
}
