use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use image::DynamicImage;
use pdfium_render::prelude::*;

use crate::error::ExtractError;
use crate::figure;
use crate::formula::FormulaPredictor;
use crate::glm_ocr::{GlmOcrConfig, GlmOcrPredictor};
use crate::layout::{DetectedRegion, LayoutDetector};
use crate::models;
use crate::output;
use crate::pdf::{self, PdfChar};
use crate::reading_order;
use crate::tableformer::TableFormerPredictor;
use crate::text;
use crate::types::*;
use crate::{ExtractOptions, FormulaModel, TableModel};

// ── Engine dispatch enums ────────────────────────────────────────────

enum FormulaEngine {
    PpFormulanet(FormulaPredictor),
    GlmOcr(GlmOcrPredictor),
}

impl FormulaEngine {
    fn predict(&self, images: &[DynamicImage]) -> Result<Vec<String>, ExtractError> {
        match self {
            Self::PpFormulanet(p) => p.predict(images),
            Self::GlmOcr(p) => p.predict(images),
        }
    }
}

enum TableEngine {
    GlmOcr(GlmOcrPredictor),
    TableFormer(TableFormerPredictor),
}

/// Table prediction result — HTML skeleton + optional per-cell bboxes.
enum TableResult {
    /// HTML with cell text already filled (e.g. from GLM-OCR).
    Html(String),
    /// TableFormer prediction with cell bboxes for text filling.
    Prediction(crate::tableformer::TablePrediction),
}

impl TableEngine {
    fn predict(
        &self,
        entries: &[(usize, DynamicImage)],
    ) -> Result<HashMap<usize, TableResult>, ExtractError> {
        if entries.is_empty() {
            return Ok(HashMap::new());
        }
        match self {
            Self::GlmOcr(predictor) => {
                let crops: Vec<DynamicImage> =
                    entries.iter().map(|(_, img)| img.clone()).collect();
                let results = predictor
                    .predict(&crops)
                    .map_err(|e| ExtractError::Layout(format!("Table prediction failed: {e}")))?;
                Ok(entries
                    .iter()
                    .zip(results)
                    .map(|((det_idx, _), html)| (*det_idx, TableResult::Html(html)))
                    .collect())
            }
            Self::TableFormer(predictor) => {
                let crops: Vec<DynamicImage> =
                    entries.iter().map(|(_, img)| img.clone()).collect();
                let results = predictor
                    .predict(&crops)
                    .map_err(|e| ExtractError::Layout(format!("Table prediction failed: {e}")))?;
                Ok(entries
                    .iter()
                    .zip(results)
                    .map(|((det_idx, _), pred)| (*det_idx, TableResult::Prediction(pred)))
                    .collect())
            }
        }
    }
}

/// Reusable extraction pipeline — load models once, extract many PDFs.
pub struct Pipeline {
    pdfium: Pdfium,
    layout: LayoutDetector,
    formula: FormulaEngine,
    table: TableEngine,
    options: PipelineOptions,
}

/// Internal options carried from ExtractOptions.
struct PipelineOptions {
    dpi: u32,
    confidence_threshold: f32,
    extract_images: bool,
    dump_formulas: bool,
    page: Option<u32>,
    debug: crate::DebugMode,
}

impl Pipeline {
    /// Create a new pipeline, loading models and pdfium.
    pub fn new(options: &ExtractOptions) -> Result<Self, ExtractError> {
        models::init_ort_runtime()?;
        let pdfium = pdf::load_pdfium(options.pdfium_path.as_deref())?;

        let cache_dir = options
            .model_cache_dir
            .clone()
            .unwrap_or_else(models::default_cache_dir);

        let paths = models::ensure_models(options.formula, options.table, &cache_dir)?;
        let layout = models::build_layout_detector(&paths.layout)?;

        let formula = match options.formula {
            FormulaModel::PpFormulanet => {
                FormulaEngine::PpFormulanet(models::build_formula_predictor(&paths)?)
            }
            FormulaModel::GlmOcr => {
                let glm_paths = paths.glm_ocr.as_ref()
                    .ok_or_else(|| ExtractError::Model("GLM-OCR model paths missing".into()))?;
                FormulaEngine::GlmOcr(models::build_glm_ocr_predictor(glm_paths)?)
            }
        };

        let table = match options.table {
            TableModel::GlmOcr => {
                let glm_paths = paths.glm_ocr.as_ref()
                    .ok_or_else(|| ExtractError::Model("GLM-OCR model paths missing".into()))?;
                let config = GlmOcrConfig {
                    prompt: "Table Recognition:".into(),
                    ..GlmOcrConfig::default()
                };
                TableEngine::GlmOcr(
                    models::build_glm_ocr_predictor_with_config(glm_paths, config)?,
                )
            }
            TableModel::TableFormer => {
                let tf_paths = paths.tableformer.as_ref()
                    .ok_or_else(|| ExtractError::Model("TableFormer model paths missing".into()))?;
                TableEngine::TableFormer(models::build_tableformer_predictor(tf_paths)?)
            }
        };

        Ok(Self {
            pdfium,
            layout,
            formula,
            table,
            options: PipelineOptions {
                dpi: options.dpi,
                confidence_threshold: options.confidence_threshold,
                extract_images: options.extract_images,
                dump_formulas: options.dump_formulas,
                page: options.page,
                debug: options.debug,
            },
        })
    }

    /// Extract content from a PDF and write results to the output directory.
    pub fn extract(
        &self,
        pdf_path: &Path,
        output_dir: &Path,
    ) -> Result<ExtractionResult, ExtractError> {
        let start = Instant::now();

        let filename = pdf_path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("document.pdf")
            .to_string();

        let stem = pdf_path
            .file_stem()
            .and_then(|f| f.to_str())
            .unwrap_or("document");

        // Load the PDF
        let doc = self
            .pdfium
            .load_pdf_from_file(pdf_path, None)
            .map_err(|e| ExtractError::Pdf(format!("Failed to load PDF: {e}")))?;

        let total_pages = doc.pages().len() as u32;
        let mut pages = Vec::new();
        let mut page_images = Vec::new();

        // Determine which pages to process
        let page_indices: Vec<u32> = if let Some(p) = self.options.page {
            if p == 0 || p > total_pages {
                return Err(ExtractError::Pdf(format!(
                    "Page {p} out of range (document has {total_pages} pages)"
                )));
            }
            vec![p - 1] // Convert 1-indexed to 0-indexed
        } else {
            (0..total_pages).collect()
        };
        let page_count = page_indices.len() as u32;

        for page_idx in page_indices {
            let page = doc.pages().get(page_idx as u16).map_err(|e| {
                ExtractError::Pdf(format!("Failed to get page {page_idx}: {e}"))
            })?;
            let (result_page, page_img) = self.process_page(&page, page_idx)?;
            page_images.push(page_img);
            pages.push(result_page);
        }

        let extraction_time_ms = start.elapsed().as_millis() as u64;

        let result = ExtractionResult {
            metadata: Metadata {
                filename,
                page_count,
                extraction_time_ms,
            },
            pages,
        };

        // Write output files
        std::fs::create_dir_all(output_dir)?;
        let json_path = output_dir.join(format!("{stem}.json"));
        let md_path = output_dir.join(format!("{stem}.md"));
        let images_dir = output_dir.join("images");

        output::write_json(&result, &json_path)?;
        output::write_markdown(&result, &md_path)?;

        if self.options.extract_images {
            output::write_images(&result.pages, &page_images, &images_dir)?;
        }

        if self.options.dump_formulas {
            let formulas_dir = output_dir.join("formulas");
            output::write_formula_images(&result.pages, &page_images, &formulas_dir)?;
        }

        if self.options.debug.is_enabled() {
            output::write_debug(
                &self.pdfium,
                pdf_path,
                &result.pages,
                output_dir,
                self.options.debug,
            )?;
        }

        Ok(result)
    }

    /// Process a single page: render, detect layout, extract content.
    fn process_page(
        &self,
        page: &PdfPage,
        page_idx: u32,
    ) -> Result<(Page, DynamicImage), ExtractError> {
        let width_pt = page.width().value;
        let height_pt = page.height().value;

        // Render page to image
        let page_image = pdf::render_page(page, self.options.dpi)?;

        // Extract characters from text layer
        let chars = pdf::extract_page_chars(page, page_idx)?;

        // Run direct layout detection (correct bboxes + reading order)
        let detected = self
            .layout
            .detect(&page_image, self.options.confidence_threshold)?;

        let scale = self.options.dpi as f32 / 72.0;

        // Phase A: try char-based extraction for inline formulas (skip expensive ML OCR)
        let mut char_based_latex: HashMap<usize, String> = HashMap::new();
        for (i, d) in detected.iter().enumerate() {
            if d.kind == RegionKind::InlineFormula {
                let bbox_pt = [
                    d.bbox_px[0] / scale,
                    d.bbox_px[1] / scale,
                    d.bbox_px[2] / scale,
                    d.bbox_px[3] / scale,
                ];
                if let Some(latex) = text::try_extract_inline_formula(&chars, bbox_pt, height_pt) {
                    tracing::debug!(
                        "Char-based bypass for inline formula {i}: {latex}"
                    );
                    char_based_latex.insert(i, latex);
                }
            }
        }

        // Crop formula regions (display + inline not already handled) and run batched recognition
        let formula_entries: Vec<(usize, DynamicImage)> = detected
            .iter()
            .enumerate()
            .filter(|(i, d)| {
                (d.kind == RegionKind::DisplayFormula || d.kind == RegionKind::InlineFormula)
                    && !char_based_latex.contains_key(i)
            })
            .map(|(i, d)| {
                let bbox_pt = [
                    d.bbox_px[0] / scale,
                    d.bbox_px[1] / scale,
                    d.bbox_px[2] / scale,
                    d.bbox_px[3] / scale,
                ];
                (
                    i,
                    figure::crop_region(
                        &page_image,
                        bbox_pt,
                        width_pt,
                        height_pt,
                        self.options.dpi,
                    ),
                )
            })
            .collect();

        let mut formula_latex: HashMap<usize, String> = if !formula_entries.is_empty() {
            let crops: Vec<DynamicImage> =
                formula_entries.iter().map(|(_, img)| img.clone()).collect();
            let results = self
                .formula
                .predict(&crops)
                .map_err(|e| ExtractError::Layout(format!("Formula prediction failed: {e}")))?;
            formula_entries
                .iter()
                .zip(results)
                .map(|((det_idx, _), latex)| (*det_idx, latex))
                .collect()
        } else {
            HashMap::new()
        };

        // Merge char-based results into formula_latex
        for (idx, latex) in char_based_latex {
            formula_latex.insert(idx, latex);
        }

        // Crop table regions and run batched recognition
        let table_entries: Vec<(usize, DynamicImage)> = detected
            .iter()
            .enumerate()
            .filter(|(_, d)| d.kind == RegionKind::Table)
            .map(|(i, d)| {
                let bbox_pt = [
                    d.bbox_px[0] / scale,
                    d.bbox_px[1] / scale,
                    d.bbox_px[2] / scale,
                    d.bbox_px[3] / scale,
                ];
                (
                    i,
                    figure::crop_region(
                        &page_image,
                        bbox_pt,
                        width_pt,
                        height_pt,
                        self.options.dpi,
                    ),
                )
            })
            .collect();

        let table_results = self.table.predict(&table_entries)?;

        // Build regions from layout detection + formula/table results
        let mut regions = self.build_regions(
            &detected,
            &formula_latex,
            &table_results,
            &chars,
            &page_image,
            page_idx,
            width_pt,
            height_pt,
        )?;

        // Suppress sub-panel detections contained within a larger visual region
        figure::suppress_contained_visuals(&mut regions);

        // The model provides reading order via order_key; use XY-Cut as fallback
        let has_model_order = regions.iter().any(|r| r.order > 0);
        if !has_model_order && regions.len() > 1 {
            reading_order::xy_cut_order(&mut regions);
        }

        // Sort regions by reading order
        regions.sort_by_key(|r| r.order);

        // Associate captions with their parent regions
        figure::associate_captions(&mut regions);

        // Attach formula numbers to their nearest display formula
        associate_formula_numbers(&mut regions);

        let result_page = Page {
            page: page_idx + 1,
            width_pt,
            height_pt,
            dpi: self.options.dpi,
            regions,
        };

        Ok((result_page, page_image))
    }

    /// Convert DetectedRegion values (from direct ONNX) into our Region types,
    /// augmented with table HTML and formula LaTeX from crop-based prediction.
    fn build_regions(
        &self,
        detected: &[DetectedRegion],
        formula_latex: &HashMap<usize, String>,
        table_results: &HashMap<usize, TableResult>,
        chars: &[PdfChar],
        page_image: &DynamicImage,
        page_idx: u32,
        page_width_pt: f32,
        page_height_pt: f32,
    ) -> Result<Vec<Region>, ExtractError> {
        let scale = self.options.dpi as f32 / 72.0;
        let mut regions = Vec::new();
        // Track which inline formula detection indices have been consumed
        // (spliced into a parent text-bearing region).
        let mut consumed_inline: std::collections::HashSet<usize> = std::collections::HashSet::new();

        // Collect inline formula data for text merging (bbox in PDF points, image-space Y-down)
        let inline_formulas: Vec<(usize, text::InlineFormula)> = detected
            .iter()
            .enumerate()
            .filter(|(_, d)| d.kind == RegionKind::InlineFormula)
            .filter_map(|(idx, d)| {
                formula_latex.get(&idx).map(|latex| {
                    let bbox = [
                        d.bbox_px[0] / scale,
                        d.bbox_px[1] / scale,
                        d.bbox_px[2] / scale,
                        d.bbox_px[3] / scale,
                    ];
                    (idx, text::InlineFormula { bbox, latex: latex.clone() })
                })
            })
            .collect();

        for (idx, det) in detected.iter().enumerate() {
            let kind = det.kind;

            // Convert pixel bboxes to PDF-point top-left-origin
            let bbox = [
                det.bbox_px[0] / scale,
                det.bbox_px[1] / scale,
                det.bbox_px[2] / scale,
                det.bbox_px[3] / scale,
            ];

            let order = idx as u32;
            let id = format!("p{}_{}", page_idx + 1, order);

            let mut region = Region {
                id,
                kind,
                bbox,
                confidence: det.confidence,
                order,
                text: None,
                html: None,
                latex: None,
                image_path: None,
                caption: None,
                chart_type: None,
                tag: None,
                consumed: kind == RegionKind::InlineFormula && consumed_inline.contains(&idx),
            };

            // Populate text content, splicing inline formulas.
            // InlineFormula regions are not text-extracted — they get latex from crop recognition.
            if (kind.is_text_bearing() || kind.is_caption()) && kind != RegionKind::InlineFormula {
                // Find overlapping inline formulas (formula center inside this region bbox)
                let overlapping: Vec<&text::InlineFormula> = inline_formulas
                    .iter()
                    .filter_map(|(det_idx, f)| {
                        let cx = (f.bbox[0] + f.bbox[2]) / 2.0;
                        let cy = (f.bbox[1] + f.bbox[3]) / 2.0;
                        if cx >= bbox[0] && cx <= bbox[2] && cy >= bbox[1] && cy <= bbox[3] {
                            consumed_inline.insert(*det_idx);
                            Some(f)
                        } else {
                            None
                        }
                    })
                    .collect();
                let mode = if kind == RegionKind::Algorithm {
                    text::AssemblyMode::PreserveLayout
                } else {
                    text::AssemblyMode::Reflow
                };
                region.text = Some(text::extract_region_text(
                    chars,
                    bbox,
                    page_height_pt,
                    &overlapping,
                    mode,
                ));
            }

            // Populate formula LaTeX from crop-based prediction
            if kind == RegionKind::DisplayFormula || kind == RegionKind::InlineFormula {
                if let Some(latex) = formula_latex.get(&idx) {
                    region.latex = Some(latex.clone());
                }
            }

            // Populate table HTML + markdown text from crop-based prediction
            if kind == RegionKind::Table {
                if let Some(result) = table_results.get(&idx) {
                    let html = match result {
                        TableResult::Html(html) => html.clone(),
                        TableResult::Prediction(pred) => {
                            crate::tableformer::fill_table_html(
                                &pred.html,
                                &pred.cell_bboxes,
                                chars,
                                bbox,
                                page_height_pt,
                            )
                        }
                    };
                    region.text = crate::html_table::html_table_to_markdown(&html);
                    region.html = Some(html);
                }
            }

            if kind.is_visual() && self.options.extract_images {
                let rel_path = format!("images/p{}_{}.png", page_idx + 1, order);
                region.image_path = Some(rel_path);
                if kind == RegionKind::Chart {
                    let cropped = figure::crop_region(
                        page_image,
                        bbox,
                        page_width_pt,
                        page_height_pt,
                        self.options.dpi,
                    );
                    region.chart_type = Some(figure::classify_chart_type(&cropped).to_string());
                }
            }

            regions.push(region);
        }

        Ok(regions)
    }
}

/// Associate `FormulaNumber` regions with their nearest `DisplayFormula`.
///
/// For each formula number with extracted text, finds the closest display
/// formula by vertical-center distance (in PDF points).  Appends `\tag{…}`
/// to the formula's LaTeX and marks the number region as consumed.
///
/// Logs a warning and discards any formula number that cannot be matched.
fn associate_formula_numbers(regions: &mut [Region]) {
    use std::collections::HashSet;

    let display_indices: Vec<usize> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| r.kind == RegionKind::DisplayFormula && r.latex.is_some() && !r.consumed)
        .map(|(i, _)| i)
        .collect();

    let number_indices: Vec<usize> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| r.kind == RegionKind::FormulaNumber && !r.consumed)
        .map(|(i, _)| i)
        .collect();

    let mut matched_formulas: HashSet<usize> = HashSet::new();

    for &num_idx in &number_indices {
        let num_text = match &regions[num_idx].text {
            Some(t) if !t.trim().is_empty() => t.trim().to_string(),
            _ => {
                tracing::warn!(
                    "FormulaNumber (id={}) has no text content, discarding",
                    regions[num_idx].id,
                );
                regions[num_idx].consumed = true;
                continue;
            }
        };

        let num_bbox = regions[num_idx].bbox;
        let num_cy = (num_bbox[1] + num_bbox[3]) / 2.0;

        // Find the closest unmatched DisplayFormula by vertical center distance
        let mut best: Option<(usize, f32)> = None;
        for &disp_idx in &display_indices {
            if matched_formulas.contains(&disp_idx) {
                continue;
            }
            let disp_bbox = regions[disp_idx].bbox;
            let disp_cy = (disp_bbox[1] + disp_bbox[3]) / 2.0;
            let dist = (num_cy - disp_cy).abs();

            if best.map_or(true, |(_, best_dist)| dist < best_dist) {
                best = Some((disp_idx, dist));
            }
        }

        // Maximum vertical center distance in PDF points (~2 text lines)
        const MAX_DIST_PT: f32 = 50.0;

        match best {
            Some((disp_idx, dist)) if dist <= MAX_DIST_PT => {
                matched_formulas.insert(disp_idx);
                regions[num_idx].consumed = true;

                // Strip outer parentheses: "(10)" → "10"
                let tag = num_text.trim();
                let tag = if tag.starts_with('(') && tag.ends_with(')') {
                    &tag[1..tag.len() - 1]
                } else {
                    tag
                };

                regions[disp_idx].tag = Some(tag.to_string());

                if let Some(ref mut latex) = regions[disp_idx].latex {
                    latex.push_str(&format!(" \\tag{{{}}}", tag));
                }
            }
            Some((_, dist)) => {
                tracing::warn!(
                    "FormulaNumber '{}' (id={}) too far from any DisplayFormula \
                     (closest={:.1}pt), discarding",
                    num_text,
                    regions[num_idx].id,
                    dist,
                );
                regions[num_idx].consumed = true;
            }
            None => {
                tracing::warn!(
                    "FormulaNumber '{}' (id={}) has no DisplayFormula on this page, discarding",
                    num_text,
                    regions[num_idx].id,
                );
                regions[num_idx].consumed = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a minimal Region for testing.
    fn make_region(kind: RegionKind, bbox: [f32; 4]) -> Region {
        Region {
            id: String::new(),
            kind,
            bbox,
            confidence: 0.9,
            order: 0,
            text: None,
            html: None,
            latex: None,
            image_path: None,
            caption: None,
            chart_type: None,
            tag: None,
            consumed: false,
        }
    }

    #[test]
    fn formula_number_basic_association() {
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 100.0, 300.0, 130.0]);
                r.latex = Some("E = mc^2".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 105.0, 340.0, 125.0]);
                r.text = Some("(1)".into());
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        assert_eq!(regions[0].latex.as_deref(), Some("E = mc^2 \\tag{1}"));
        assert_eq!(regions[0].tag.as_deref(), Some("1"));
        assert!(regions[1].consumed);
    }

    #[test]
    fn formula_number_no_parens() {
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 100.0, 300.0, 130.0]);
                r.latex = Some("a + b".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 105.0, 340.0, 125.0]);
                r.text = Some("2a".into());
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        assert_eq!(regions[0].latex.as_deref(), Some("a + b \\tag{2a}"));
        assert_eq!(regions[0].tag.as_deref(), Some("2a"));
        assert!(regions[1].consumed);
    }

    #[test]
    fn formula_number_below_formula() {
        // Formula number slightly below the formula (like VBD p5 eq.10)
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 233.0, 289.0, 265.0]);
                r.latex = Some("x^2".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [280.0, 264.0, 295.0, 275.0]);
                r.text = Some("(10)".into());
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        assert_eq!(regions[0].latex.as_deref(), Some("x^2 \\tag{10}"));
        assert!(regions[1].consumed);
    }

    #[test]
    fn formula_number_stacked_formulas() {
        // Two formulas close together, each with its own number
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 100.0, 300.0, 120.0]);
                r.latex = Some("a = b".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 103.0, 340.0, 117.0]);
                r.text = Some("(13)".into());
                r
            },
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 130.0, 300.0, 160.0]);
                r.latex = Some("c = d".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 138.0, 340.0, 152.0]);
                r.text = Some("(14)".into());
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        assert_eq!(regions[0].latex.as_deref(), Some("a = b \\tag{13}"));
        assert_eq!(regions[2].latex.as_deref(), Some("c = d \\tag{14}"));
        assert!(regions[1].consumed);
        assert!(regions[3].consumed);
    }

    #[test]
    fn formula_number_too_far() {
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 100.0, 300.0, 130.0]);
                r.latex = Some("y = x".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 400.0, 340.0, 420.0]);
                r.id = "p1_5".into();
                r.text = Some("(99)".into());
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        // Formula should be unchanged, number should be consumed (discarded)
        assert_eq!(regions[0].latex.as_deref(), Some("y = x"));
        assert!(regions[1].consumed);
    }

    #[test]
    fn formula_number_no_display_formula() {
        let mut regions = vec![{
            let mut r = make_region(RegionKind::FormulaNumber, [310.0, 105.0, 340.0, 125.0]);
            r.id = "p1_0".into();
            r.text = Some("(7)".into());
            r
        }];

        associate_formula_numbers(&mut regions);

        assert!(regions[0].consumed);
    }

    #[test]
    fn formula_number_empty_text() {
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 100.0, 300.0, 130.0]);
                r.latex = Some("z = w".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 105.0, 340.0, 125.0]);
                r.id = "p1_1".into();
                r.text = Some("  ".into()); // whitespace-only
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        // Formula untouched, number consumed
        assert_eq!(regions[0].latex.as_deref(), Some("z = w"));
        assert!(regions[1].consumed);
    }

    #[test]
    fn formula_number_already_consumed() {
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::DisplayFormula, [50.0, 100.0, 300.0, 130.0]);
                r.latex = Some("p = q".into());
                r
            },
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 105.0, 340.0, 125.0]);
                r.text = Some("(3)".into());
                r.consumed = true; // already consumed by something else
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        // Formula should be unchanged since the number was already consumed
        assert_eq!(regions[0].latex.as_deref(), Some("p = q"));
    }

    #[test]
    fn formula_no_latex_skipped() {
        // DisplayFormula with no LaTeX shouldn't be a match target
        let mut regions = vec![
            make_region(RegionKind::DisplayFormula, [50.0, 100.0, 300.0, 130.0]),
            {
                let mut r = make_region(RegionKind::FormulaNumber, [310.0, 105.0, 340.0, 125.0]);
                r.id = "p1_1".into();
                r.text = Some("(5)".into());
                r
            },
        ];

        associate_formula_numbers(&mut regions);

        // Number should be consumed (discarded) since there's no valid formula
        assert!(regions[1].consumed);
    }
}
