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
use crate::toc;
use crate::types::*;
use crate::{ExtractOptions, FormulaModel, FormulaParseMode, TableModel};

// ── Engine dispatch enums ────────────────────────────────────────────

enum TableEngine {
    GlmOcr(GlmOcrPredictor),
    TableFormer(TableFormerPredictor),
}

/// Table prediction result — HTML skeleton + optional per-cell bboxes.
pub enum TableResult {
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
                    .map(|((det_idx, _), fr)| (*det_idx, TableResult::Html(fr.latex)))
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

enum FormulaEngine {
    PpFormulanet(FormulaPredictor),
    GlmOcr(GlmOcrPredictor),
}

impl FormulaEngine {
    fn predict_with_progress(
        &self,
        images: &[DynamicImage],
        on_progress: impl Fn(usize),
    ) -> Result<Vec<crate::types::FormulaResult>, ExtractError> {
        match self {
            Self::PpFormulanet(p) => {
                let mut results = Vec::with_capacity(images.len());
                for (i, img) in images.iter().enumerate() {
                    results.push(p.predict(&[img.clone()])?.pop().unwrap());
                    on_progress(i + 1);
                }
                Ok(results)
            }
            Self::GlmOcr(p) => p.predict_with_progress(images, on_progress),
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
    formula_parse_mode: FormulaParseMode,
    page: Option<u32>,
    pages: Option<Vec<u32>>,
    chapter: Option<String>,
    section: Option<String>,
    debug: crate::DebugMode,
}

impl Pipeline {
    /// Create a new pipeline, loading models and pdfium.
    pub fn new(options: &ExtractOptions) -> Result<Self, ExtractError> {
        eprint!("\r  Loading models: ort");
        models::init_ort_runtime()?;
        let pdfium = pdf::load_pdfium(options.pdfium_path.as_deref())?;

        let paths = models::ensure_models(options.formula, options.table, options.model_cache_dir.as_deref())?;
        eprint!("\r  Loading models: layout");
        let layout = models::build_layout_detector(&paths.layout)?;

        eprint!("\r  Loading models: formula");
        let formula = match options.formula {
            FormulaModel::PpFormulanet => {
                let fp = paths.formula.as_ref()
                    .ok_or_else(|| ExtractError::Model("PP-FormulaNet model paths missing".into()))?;
                FormulaEngine::PpFormulanet(models::build_formula_predictor(fp)?)
            }
            FormulaModel::GlmOcr => {
                FormulaEngine::GlmOcr(models::build_glm_ocr_predictor(&paths.glm_ocr)?)
            }
        };

        eprint!("\r  Loading models: table");
        let table = match options.table {
            TableModel::GlmOcr => {
                let config = GlmOcrConfig {
                    prompt: "Table Recognition:".into(),
                    ..GlmOcrConfig::default()
                };
                TableEngine::GlmOcr(
                    models::build_glm_ocr_predictor_with_config(&paths.glm_ocr, config)?,
                )
            }
            TableModel::TableFormer => {
                let tf_paths = paths.tableformer.as_ref()
                    .ok_or_else(|| ExtractError::Model("TableFormer model paths missing".into()))?;
                TableEngine::TableFormer(models::build_tableformer_predictor(tf_paths)?)
            }
        };
        eprint!("\r{}\r", " ".repeat(60));

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
                formula_parse_mode: options.formula_parse_mode,
                page: options.page,
                pages: options.pages.clone(),
                chapter: options.chapter.clone(),
                section: options.section.clone(),
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

        // Quick first pass: extract text layer chars from all pages for TOC parsing.
        let page_chars: Vec<(Vec<PdfChar>, f32)> = (0..total_pages)
            .map(|i| {
                let page = doc.pages().get(i as u16).unwrap();
                let chars = pdf::extract_page_chars(&page, i).unwrap_or_default();
                let height = page.height().value;
                (chars, height)
            })
            .collect();
        let mut toc_result = toc::parse_toc(&page_chars);
        if let Some(ref mut toc) = toc_result {
            eprintln!(
                "  TOC: {} entries from {} pages",
                toc.entries.len(),
                toc.toc_pages.len(),
            );
            // Auto-number unnumbered TOC entries so --section works.
            crate::output::auto_number_toc_entries(&mut toc.entries);
        }

        let mut pages = Vec::new();
        let mut page_images = Vec::new();

        // Determine which pages to process
        let page_indices: Vec<u32> = if let Some(ref pages) = self.options.pages {
            pages
                .iter()
                .map(|p| p - 1)
                .filter(|&p| p < total_pages)
                .collect()
        } else if self.options.chapter.is_some() || self.options.section.is_some() {
            resolve_toc_page_range(
                &toc_result,
                total_pages,
                self.options.chapter.as_deref(),
                self.options.section.as_deref(),
            )?
        } else if let Some(p) = self.options.page {
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

        for (i, page_idx) in page_indices.iter().enumerate() {
            eprint!(
                "\r  Page {}/{page_count}: layout",
                i + 1,
            );
            let page = doc.pages().get(*page_idx as u16).map_err(|e| {
                ExtractError::Pdf(format!("Failed to get page {page_idx}: {e}"))
            })?;
            let (result_page, page_img) = self.process_page(&page, *page_idx, page_count, output_dir)?;
            page_images.push(page_img);
            pages.push(result_page);
        }
        // Clear the progress line
        eprint!("\r{}\r", " ".repeat(60));

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
        eprint!("\r  Writing output...");
        let t_out = Instant::now();
        std::fs::create_dir_all(output_dir)?;
        let json_path = output_dir.join(format!("{stem}.json"));
        let md_path = output_dir.join(format!("{stem}.md"));
        let images_dir = output_dir.join("images");

        output::write_json(&result, &json_path)?;

        let watermarks = output::detect_watermarks(&page_chars);
        let reflow_doc = if let Some(ref toc) = toc_result {
            output::reflow_with_outline(&result, &toc.entries, &toc.toc_pages, total_pages, &watermarks)
        } else {
            output::reflow(&result, &watermarks)
        };
        let reflow_path = output_dir.join(format!("{stem}.reflow.json"));
        output::write_reflow_json(&reflow_doc, &reflow_path)?;

        let md = output::render_markdown_from_reflow(&reflow_doc);
        std::fs::write(&md_path, md)?;

        if self.options.extract_images {
            output::write_images(&result.pages, &page_images, &images_dir)?;
        }

        // Save cropped images for formula regions that lack LaTeX (unparsed fallback)
        let has_unresolved_formulas = result.pages.iter().any(|p| {
            p.regions.iter().any(|r| {
                matches!(r.kind, RegionKind::DisplayFormula | RegionKind::InlineFormula)
                    && r.latex.is_none()
                    && !r.consumed
            })
        });
        if has_unresolved_formulas {
            let formula_img_dir = output_dir.join("images/formulas");
            output::write_formula_images(&result.pages, &page_images, &formula_img_dir)?;
        }

        if self.options.dump_formulas {
            let formulas_dir = output_dir.join("formulas");
            output::write_formula_images(&result.pages, &page_images, &formulas_dir)?;
        }

        if self.options.debug.is_enabled() {
            eprint!("\r  Writing debug layout...");
            output::write_debug(
                &self.pdfium,
                pdf_path,
                &result.pages,
                output_dir,
                self.options.debug,
            )?;
        }
        let out_secs = t_out.elapsed().as_secs_f64();
        if out_secs >= 0.5 {
            eprintln!("\r  Output: {out_secs:.1}s{}",
                " ".repeat(40),
            );
        } else {
            // Clear the progress line
            eprint!("\r{}\r", " ".repeat(60));
        }

        Ok(result)
    }

    /// Process a single page: render, detect layout, extract content.
    fn process_page(
        &self,
        page: &PdfPage,
        page_idx: u32,
        page_count: u32,
        output_dir: &Path,
    ) -> Result<(Page, DynamicImage), ExtractError> {
        let page_num = page_idx + 1;
        let width_pt = page.width().value;
        let height_pt = page.height().value;

        // Render page to image
        let page_image = pdf::render_page(page, self.options.dpi)?;

        // Extract characters from text layer and adjust for CropBox offset.
        // Pdfium returns char bboxes in MediaBox coordinates, but the rendered
        // image (and layout model) uses the CropBox area. Apply the offset so
        // char positions align with layout-detected region bboxes.
        let mut chars = pdf::extract_page_chars(page, page_idx)?;
        let crop = pdf::crop_offset(page);
        pdf::apply_crop_offset(&mut chars, crop);

        // Run direct layout detection (correct bboxes + reading order)
        let detected = self
            .layout
            .detect(&page_image, self.options.confidence_threshold)?;

        let scale = self.options.dpi as f32 / 72.0;

        // Split wide text regions that span both columns in two-column layouts.
        let detected = split_cross_column_regions(detected, &chars, width_pt, height_pt, scale);

        // Phase A: try char-based extraction for inline formulas (skip expensive ML OCR).
        // Also compute expanded bboxes (bracket-aware) for formulas that fall through to OCR.
        let parse_mode = self.options.formula_parse_mode;
        let mut char_based_latex: HashMap<usize, String> = HashMap::new();
        let mut expanded_bboxes: HashMap<usize, [f32; 4]> = HashMap::new();
        for (i, d) in detected.iter().enumerate() {
            if d.kind == RegionKind::InlineFormula {
                let bbox_pt = [
                    d.bbox_px[0] / scale,
                    d.bbox_px[1] / scale,
                    d.bbox_px[2] / scale,
                    d.bbox_px[3] / scale,
                ];
                // In OCR mode, skip char-based extraction entirely
                if parse_mode != FormulaParseMode::Ocr {
                    let attempt = text::try_extract_inline_formula(&chars, bbox_pt, height_pt);
                    if let Some(latex) = attempt.latex {
                        tracing::debug!(
                            "Char-based bypass for inline formula {i}: {latex}"
                        );
                        char_based_latex.insert(i, latex);
                        continue;
                    }
                    // Char-based failed — use the adjusted bbox (may be expanded
                    // for brackets or tightened to exclude stray edge chars)
                    if attempt.adjusted_bbox != bbox_pt {
                        expanded_bboxes.insert(i, attempt.adjusted_bbox);
                    }
                    continue;
                }
                // OCR mode — still compute expanded bbox for bracket recovery
                let expanded = text::expand_formula_bbox(&chars, bbox_pt, height_pt);
                if expanded != bbox_pt {
                    expanded_bboxes.insert(i, expanded);
                }
            }
        }

        // Crop formula regions (display + inline not already handled) and run batched recognition.
        // In Manual mode, skip OCR entirely — only char-based results are used.
        // Use expanded bboxes (from bracket expansion) when available.
        let formula_entries: Vec<(usize, DynamicImage)> = if parse_mode == FormulaParseMode::Manual {
            Vec::new()
        } else {
            detected
            .iter()
            .enumerate()
            .filter(|(i, d)| {
                (d.kind == RegionKind::DisplayFormula || d.kind == RegionKind::InlineFormula)
                    && !char_based_latex.contains_key(i)
            })
            .map(|(i, d)| {
                let bbox_pt = expanded_bboxes.get(&i).copied().unwrap_or([
                    d.bbox_px[0] / scale,
                    d.bbox_px[1] / scale,
                    d.bbox_px[2] / scale,
                    d.bbox_px[3] / scale,
                ]);
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
            .collect()
        };

        let mut formula_results: HashMap<usize, crate::types::FormulaResult> =
            if !formula_entries.is_empty() {
                let msg = format!(
                    "  Page {page_num}/{page_count}: formulas 0/{}",
                    formula_entries.len(),
                );
                eprint!("\r{msg:<60}");
                let crops: Vec<DynamicImage> =
                    formula_entries.iter().map(|(_, img)| img.clone()).collect();
                let formula_count = crops.len();
                let results = self
                    .formula
                    .predict_with_progress(&crops, |done| {
                        let msg = format!(
                            "  Page {page_num}/{page_count}: formulas {done}/{formula_count}",
                        );
                        eprint!("\r{msg:<60}");
                    })
                    .map_err(|e| {
                        ExtractError::Layout(format!("Formula prediction failed: {e}"))
                    })?;
                formula_entries
                    .iter()
                    .zip(results)
                    .map(|((det_idx, _), fr)| (*det_idx, fr))
                    .collect()
            } else {
                HashMap::new()
            };

        // Strip leading/trailing $ delimiters from OCR output — models sometimes
        // emit them, but we add our own wrapping in the markdown renderer.
        for fr in formula_results.values_mut() {
            let stripped = fr.latex.trim().trim_matches('$').trim();
            if stripped.len() != fr.latex.len() {
                fr.latex = stripped.to_string();
            }
        }

        // Merge char-based results into formula_results.
        // Char-based extraction has no meaningful confidence signal, so we use NaN
        // as a sentinel — downstream code will skip setting ocr_confidence for these.
        for (idx, latex) in char_based_latex {
            formula_results.insert(
                idx,
                crate::types::FormulaResult {
                    latex,
                    confidence: f32::NAN,
                },
            );
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

        if !table_entries.is_empty() {
            eprint!(
                "\r  Page {page_num}/{page_count}: tables ({})",
                table_entries.len(),
            );
        }
        let table_results = self.table.predict(&table_entries)?;

        // Write table debug overlays if layout debug is enabled
        if self.options.debug.is_enabled() {
            output::write_table_debug(
                output_dir,
                &table_entries,
                &table_results,
                page_idx,
            )?;
        }

        // Build regions from layout detection + formula/table results
        let mut regions = self.build_regions(
            &detected,
            &formula_results,
            &table_results,
            &chars,
            &page_image,
            page_idx,
            width_pt,
            height_pt,
        )?;

        // Suppress sub-panel detections contained within a larger visual region
        figure::suppress_contained_visuals(&mut regions);

        // Suppress duplicate Abstract/Text regions sharing the same bbox
        figure::suppress_duplicate_abstract_text(&mut regions);

        // The model provides reading order via order_key; use XY-Cut as fallback
        let has_model_order = regions.iter().any(|r| r.order > 0);
        if !has_model_order && regions.len() > 1 {
            reading_order::xy_cut_order(&mut regions);
        }

        // Sort regions by reading order
        regions.sort_by_key(|r| r.order);

        // Associate captions with their parent regions
        figure::associate_captions(&mut regions);
        // Group spatially close visual regions into FigureGroups
        figure::group_figure_regions(&mut regions);

        // Expand visual bboxes to include their captions, and consume
        // sub-labels / text regions enclosed within the expanded bounds.
        figure::expand_visual_bounds(&mut regions);

        // Attach formula numbers to their nearest display formula
        associate_formula_numbers(&mut regions);

        // Remove structural/redundant regions from output
        strip_structural_regions(&mut regions);

        // Assign composite image path to FigureGroups, clear member image paths
        if self.options.extract_images {
            for region in &mut regions {
                if region.kind == RegionKind::FigureGroup {
                    region.image_path =
                        Some(format!("images/p{}_{}.png", page_idx + 1, region.order));
                    if let Some(ref mut items) = region.items {
                        for item in items.iter_mut() {
                            item.image_path = None;
                        }
                    }
                }
            }
        }

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
        formula_results: &HashMap<usize, crate::types::FormulaResult>,
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

        // Pre-pass: merge References detections in the same column into a single bbox.
        // The layout model often detects overlapping References regions (one large +
        // many small per-entry regions). Merging by column avoids duplicate text.
        let ref_merged_bbox = merge_references_by_column(detected, scale);

        // Collect inline formula data for text merging (bbox in PDF points, image-space Y-down)
        let inline_formulas: Vec<(usize, text::InlineFormula)> = detected
            .iter()
            .enumerate()
            .filter(|(_, d)| d.kind == RegionKind::InlineFormula)
            .filter_map(|(idx, d)| {
                formula_results.get(&idx).map(|fr| {
                    let bbox = [
                        d.bbox_px[0] / scale,
                        d.bbox_px[1] / scale,
                        d.bbox_px[2] / scale,
                        d.bbox_px[3] / scale,
                    ];
                    (idx, text::InlineFormula { bbox, latex: fr.latex.clone() })
                })
            })
            .collect();

        // Collect unresolved inline formulas (detected but no LaTeX available,
        // e.g. in manual mode when char-based extraction fails).
        let unresolved_inline: Vec<(usize, text::UnresolvedFormula)> = detected
            .iter()
            .enumerate()
            .filter(|(idx, d)| {
                d.kind == RegionKind::InlineFormula && !formula_results.contains_key(idx)
            })
            .map(|(idx, d)| {
                let bbox = [
                    d.bbox_px[0] / scale,
                    d.bbox_px[1] / scale,
                    d.bbox_px[2] / scale,
                    d.bbox_px[3] / scale,
                ];
                let id = format!("p{}_{}", page_idx + 1, idx);
                (idx, text::UnresolvedFormula { bbox, id })
            })
            .collect();

        for (idx, det) in detected.iter().enumerate() {
            let mut kind = det.kind;

            // Convert pixel bboxes to PDF-point top-left-origin
            let mut bbox = [
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
                items: None,
                formula_source: None,
                ocr_confidence: None,
                consumed: kind == RegionKind::InlineFormula && consumed_inline.contains(&idx),
            };

            // For References, check if this index was merged into another region.
            if kind == RegionKind::References {
                if let Some(merged) = ref_merged_bbox.get(&idx) {
                    match merged {
                        Some(merged_bbox) => {
                            // Primary region — extract text from the merged bbox
                            region.bbox = *merged_bbox;
                            region.text = Some(text::extract_region_text(
                                chars,
                                *merged_bbox,
                                page_height_pt,
                                &[],
                                &[],
                                text::AssemblyMode::References,
                            ));
                        }
                        None => {
                            // Consumed duplicate — mark as consumed, skip text extraction
                            region.consumed = true;
                        }
                    }
                }
            }

            // Populate text content, splicing inline formulas.
            // InlineFormula regions are not text-extracted — they get latex from crop recognition.
            if !region.consumed
                && (kind.is_text_bearing() || kind.is_caption())
                && kind != RegionKind::InlineFormula
                && kind != RegionKind::References  // already handled above
            {
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
                // Find overlapping unresolved inline formulas
                let overlapping_unresolved: Vec<&text::UnresolvedFormula> = unresolved_inline
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
                // Decide assembly mode based on content analysis.
                // For Text regions, check if the content looks like code
                // or formatted text (mini-TOC, structured listings) — if so,
                // preserve the layout instead of reflowing.
                let (mode, promote_kind) = if kind == RegionKind::Algorithm {
                    (text::AssemblyMode::PreserveLayout, None)
                } else if kind == RegionKind::Text {
                    let region_chars: Vec<&PdfChar> = chars.iter()
                        .filter(|c| {
                            let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
                            let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
                            let cy_img = page_height_pt - cy;
                            cx >= bbox[0] && cx <= bbox[2] && cy_img >= bbox[1] && cy_img <= bbox[3]
                        })
                        .collect();
                    let sample_text: String = region_chars.iter()
                        .filter(|c| !c.codepoint.is_control())
                        .map(|c| c.codepoint)
                        .take(500)
                        .collect();
                    if is_likely_formula(&region_chars) {
                        // Promote to DisplayFormula — chars are mostly math
                        (text::AssemblyMode::Reflow, Some(RegionKind::DisplayFormula))
                    } else if output::code_score(&sample_text) >= 3 {
                        (text::AssemblyMode::PreserveLayout, Some(RegionKind::Algorithm))
                    } else if is_formatted_text(&region_chars, page_height_pt, bbox) {
                        (text::AssemblyMode::PreserveLayout, Some(RegionKind::FormattedText))
                    } else {
                        (text::AssemblyMode::Reflow, None)
                    }
                } else {
                    (text::AssemblyMode::Reflow, None)
                };
                // Apply kind promotion if detected
                if let Some(new_kind) = promote_kind {
                    kind = new_kind;
                    region.kind = new_kind;
                }
                region.text = Some(text::extract_region_text(
                    chars,
                    bbox,
                    page_height_pt,
                    &overlapping,
                    &overlapping_unresolved,
                    mode,
                ));
            }

            // Populate formula LaTeX + confidence + source from prediction
            if kind == RegionKind::DisplayFormula || kind == RegionKind::InlineFormula {
                if let Some(fr) = formula_results.get(&idx) {
                    // Detect drop caps masquerading as formulas: a single
                    // uppercase letter with the rest of the paragraph as
                    // superscript/subscript (e.g. P^{hysics is a hot...}).
                    if let Some(plain_text) = detect_drop_cap(&fr.latex) {
                        region.kind = RegionKind::Text;
                        kind = RegionKind::Text;
                        region.text = Some(plain_text);
                    } else {
                        region.latex = Some(fr.latex.clone());
                        if fr.confidence.is_finite() {
                            region.formula_source = Some("ocr".into());
                            region.ocr_confidence = Some(fr.confidence);
                        } else {
                            region.formula_source = Some("char".into());
                        }
                    }
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
                    let md = crate::html_table::html_table_to_markdown(&html);

                    // Quality check: if the table markdown has mostly empty
                    // cells (> 60% empty), demote to Image — the table
                    // detection is likely a false positive.
                    // Also detect degenerate decoder output: if any single
                    // row has an unreasonable number of columns (> 30) or
                    // the same cell text repeats excessively, the model
                    // entered a token loop.
                    let is_low_quality = if let Some(ref md_text) = md {
                        let total_cells = md_text.matches('|').count();
                        let empty_cells = md_text.matches("|  |").count()
                            + md_text.matches("| |").count();
                        let mostly_empty = total_cells > 4
                            && empty_cells as f32 / (total_cells as f32 / 2.0) > 0.6;

                        // Check for degenerate row width: split by newline,
                        // count pipe chars per line. Real tables rarely
                        // exceed 20 columns.
                        let max_row_cols = md_text
                            .lines()
                            .map(|line| line.matches('|').count())
                            .max()
                            .unwrap_or(0);
                        let too_wide = max_row_cols > 30;

                        mostly_empty || too_wide
                    } else {
                        true
                    };

                    if is_low_quality {
                        // Demote: keep as image, drop table text
                        region.kind = RegionKind::Image;
                    } else {
                        region.text = md;
                        region.html = Some(html);
                    }
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

            // Set image path for formula regions that lack LaTeX (unparsed fallback)
            if matches!(kind, RegionKind::DisplayFormula | RegionKind::InlineFormula)
                && region.latex.is_none()
                && !region.consumed
            {
                let rel_path = format!("images/formulas/{}.png", region.id);
                region.image_path = Some(rel_path);
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

/// Detect two-column page layout and split text regions that span both columns.
///
/// Many textbooks use two-column layouts. The layout model sometimes produces
/// a single wide bounding box that covers both columns. When text is extracted
/// from such a region, chars from both columns at the same Y position get merged
/// into one garbled line (e.g. "The software systems th This book is about how").
///
/// This function:
/// 1. Detects a vertical gutter by analyzing char X-positions for a gap in the
///    middle third of the page
/// 2. Splits any text-bearing region that spans the gutter into two regions
///    (left column, right column)
fn split_cross_column_regions(
    mut detected: Vec<DetectedRegion>,
    chars: &[PdfChar],
    page_width_pt: f32,
    _page_height_pt: f32,
    scale: f32,
) -> Vec<DetectedRegion> {
    // Need enough chars to analyze
    if chars.len() < 20 || page_width_pt < 100.0 {
        return detected;
    }

    // Collect non-space char X centers in PDF points
    let mut x_positions: Vec<f32> = chars
        .iter()
        .filter(|c| !c.codepoint.is_whitespace() && (c.bbox[2] - c.bbox[0]) > 0.1)
        .map(|c| (c.bbox[0] + c.bbox[2]) / 2.0)
        .collect();
    if x_positions.len() < 20 {
        return detected;
    }
    x_positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Look for a gap in the middle third of the page.
    // Bin X positions and find a region with very few chars.
    let left_bound = page_width_pt * 0.3;
    let right_bound = page_width_pt * 0.7;
    let bin_width = 3.0; // 3pt bins
    let num_bins = ((right_bound - left_bound) / bin_width) as usize;
    if num_bins < 5 {
        return detected;
    }
    let mut bins = vec![0u32; num_bins];
    for &x in &x_positions {
        if x >= left_bound && x < right_bound {
            let idx = ((x - left_bound) / bin_width) as usize;
            if idx < num_bins {
                bins[idx] += 1;
            }
        }
    }

    // Find the widest contiguous run of empty/near-empty bins (≤1 char)
    let mut best_start = 0;
    let mut best_len = 0;
    let mut cur_start = 0;
    let mut cur_len = 0;
    for (i, &count) in bins.iter().enumerate() {
        if count <= 1 {
            if cur_len == 0 {
                cur_start = i;
            }
            cur_len += 1;
        } else {
            if cur_len > best_len {
                best_start = cur_start;
                best_len = cur_len;
            }
            cur_len = 0;
        }
    }
    if cur_len > best_len {
        best_start = cur_start;
        best_len = cur_len;
    }

    // Need at least ~6pt gap (2 bins) to consider it a column gutter
    if best_len < 2 {
        return detected;
    }

    // Gutter center in PDF points
    let gutter_left_pt = left_bound + best_start as f32 * bin_width;
    let gutter_right_pt = gutter_left_pt + best_len as f32 * bin_width;
    let gutter_center_pt = (gutter_left_pt + gutter_right_pt) / 2.0;

    // Verify: enough chars on both sides of the gutter
    let left_chars = x_positions.iter().filter(|&&x| x < gutter_left_pt).count();
    let right_chars = x_positions.iter().filter(|&&x| x > gutter_right_pt).count();
    tracing::debug!(
        gutter_left = gutter_left_pt,
        gutter_right = gutter_right_pt,
        left_chars,
        right_chars,
        "Column gutter candidate"
    );
    if left_chars < 10 || right_chars < 10 {
        return detected;
    }

    // Convert gutter to pixel space for bbox comparison
    let gutter_center_px = gutter_center_pt * scale;
    let gutter_margin_px = (best_len as f32 * bin_width * scale) / 2.0;

    // Split regions that span the gutter
    let mut result = Vec::with_capacity(detected.len() + 10);
    for det in detected.drain(..) {
        let x1 = det.bbox_px[0];
        let x2 = det.bbox_px[2];
        let region_width = x2 - x1;
        let page_width_px = page_width_pt * scale;

        // Only split text-bearing regions that are wider than ~55% of page width
        // and span the gutter
        let spans_gutter = x1 < (gutter_center_px - gutter_margin_px)
            && x2 > (gutter_center_px + gutter_margin_px);
        let is_wide = region_width > page_width_px * 0.55;
        let is_splittable = det.kind.is_text_bearing()
            || det.kind == RegionKind::ParagraphTitle
            || det.kind == RegionKind::Abstract;

        if spans_gutter && is_wide && is_splittable {
            // Split into left and right column regions
            let mut left = DetectedRegion {
                kind: det.kind,
                bbox_px: [x1, det.bbox_px[1], gutter_center_px - gutter_margin_px, det.bbox_px[3]],
                confidence: det.confidence,
                order_key: det.order_key,
            };
            let mut right = DetectedRegion {
                kind: det.kind,
                bbox_px: [gutter_center_px + gutter_margin_px, det.bbox_px[1], x2, det.bbox_px[3]],
                confidence: det.confidence,
                order_key: det.order_key + 0.001, // right column after left
            };
            // Adjust order: left column first (lower Y = earlier), right column second
            // For same Y range, left comes first
            result.push(left);
            result.push(right);
        } else {
            result.push(det);
        }
    }

    result
}

/// Group References detections by column and merge their bounding boxes.
///
/// Returns a map from detection index to:
/// - `Some(merged_bbox)` for the primary (first) region in each column group
/// - `None` for duplicate regions that should be consumed
///
/// Two References regions are in the same column if their horizontal ranges
/// overlap by more than 50% of the narrower region's width.
fn merge_references_by_column(
    detected: &[DetectedRegion],
    scale: f32,
) -> HashMap<usize, Option<[f32; 4]>> {
    let ref_indices: Vec<usize> = detected
        .iter()
        .enumerate()
        .filter(|(_, d)| d.kind == RegionKind::References)
        .map(|(i, _)| i)
        .collect();

    if ref_indices.is_empty() {
        return HashMap::new();
    }

    // Convert to PDF-point bboxes
    let ref_bboxes: Vec<(usize, [f32; 4])> = ref_indices
        .iter()
        .map(|&i| {
            let d = &detected[i];
            (
                i,
                [
                    d.bbox_px[0] / scale,
                    d.bbox_px[1] / scale,
                    d.bbox_px[2] / scale,
                    d.bbox_px[3] / scale,
                ],
            )
        })
        .collect();

    // Group by column using union-find style grouping
    let n = ref_bboxes.len();
    let mut group: Vec<usize> = (0..n).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let (_, a) = ref_bboxes[i];
            let (_, b) = ref_bboxes[j];
            // Horizontal overlap
            let overlap_x1 = a[0].max(b[0]);
            let overlap_x2 = a[2].min(b[2]);
            let overlap = (overlap_x2 - overlap_x1).max(0.0);
            let min_width = (a[2] - a[0]).min(b[2] - b[0]);
            if min_width > 0.0 && overlap / min_width > 0.5 {
                // Same column — merge groups
                let gi = find(&mut group, i);
                let gj = find(&mut group, j);
                group[gi] = gj;
            }
        }
    }

    // Compute merged bbox per group
    let mut group_bbox: HashMap<usize, [f32; 4]> = HashMap::new();
    let mut group_primary: HashMap<usize, usize> = HashMap::new(); // group root → first det index
    for (local_idx, &(det_idx, bbox)) in ref_bboxes.iter().enumerate() {
        let root = find(&mut group, local_idx);
        let merged = group_bbox.entry(root).or_insert(bbox);
        merged[0] = merged[0].min(bbox[0]);
        merged[1] = merged[1].min(bbox[1]);
        merged[2] = merged[2].max(bbox[2]);
        merged[3] = merged[3].max(bbox[3]);
        group_primary.entry(root).or_insert(det_idx);
    }

    // Build result map
    let mut result = HashMap::new();
    for (local_idx, &(det_idx, _)) in ref_bboxes.iter().enumerate() {
        let root = find(&mut group, local_idx);
        let primary = group_primary[&root];
        if det_idx == primary {
            result.insert(det_idx, Some(group_bbox[&root]));
        } else {
            result.insert(det_idx, None);
        }
    }

    result
}

/// Union-find helper
fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Returns true when `inner` bbox is fully contained within `outer` bbox
/// (with a small tolerance for floating-point rounding).
fn bbox_contains(outer: [f32; 4], inner: [f32; 4]) -> bool {
    const EPS: f32 = 1.0; // 1pt tolerance
    inner[0] >= outer[0] - EPS
        && inner[1] >= outer[1] - EPS
        && inner[2] <= outer[2] + EPS
        && inner[3] <= outer[3] + EPS
}

/// Remove structural/redundant regions from the output:
/// - `FormulaNumber` (already captured as `tag` on DisplayFormula)
/// - `PageHeader` / `PageFooter`
/// - Any non-header region whose bbox is fully contained within a PageHeader bbox
///   (e.g. a duplicate Title detection that overlaps the header)
/// Compute the fraction of `a`'s area that overlaps with `b`.
fn bbox_overlap_ratio(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x_overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let y_overlap = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let overlap_area = x_overlap * y_overlap;
    let a_area = (a[2] - a[0]) * (a[3] - a[1]);
    if a_area <= 0.0 {
        return 0.0;
    }
    overlap_area / a_area
}

fn strip_structural_regions(regions: &mut Vec<Region>) {
    // Collect PageHeader bboxes before filtering.
    let header_bboxes: Vec<[f32; 4]> = regions
        .iter()
        .filter(|r| r.kind == RegionKind::PageHeader)
        .map(|r| r.bbox)
        .collect();

    // Collect Image/Figure bboxes to suppress text inside figures.
    let figure_bboxes: Vec<[f32; 4]> = regions
        .iter()
        .filter(|r| {
            matches!(
                r.kind,
                RegionKind::Image | RegionKind::FigureGroup
            )
        })
        .map(|r| r.bbox)
        .collect();

    // Deduplicate overlapping regions. The layout model sometimes detects
    // the same content area as multiple region types (e.g. Table + Text,
    // Algorithm + Text). Keep the more specific/higher-priority kind.
    // Priority: Table > Algorithm > Text (Table has structured extraction,
    // Algorithm preserves layout, Text is the fallback).
    fn region_priority(kind: RegionKind) -> u8 {
        match kind {
            RegionKind::Table => 3,
            RegionKind::Algorithm => 2,
            RegionKind::Image | RegionKind::FigureGroup => 2,
            RegionKind::DisplayFormula | RegionKind::InlineFormula => 2,
            RegionKind::Footnote => 1,
            _ => 0, // Text, ParagraphTitle, etc.
        }
    }

    // Mark indices of regions to drop due to overlap with a higher-priority region.
    // Skip consumed regions — they're already handled (e.g. FigureGroup members).
    let mut drop_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for i in 0..regions.len() {
        if drop_indices.contains(&i) || regions[i].consumed {
            continue;
        }
        for j in (i + 1)..regions.len() {
            if drop_indices.contains(&j) || regions[j].consumed {
                continue;
            }
            let ri = &regions[i];
            let rj = &regions[j];
            // Skip zero bboxes (synthetic/test data)
            if ri.bbox == [0.0; 4] || rj.bbox == [0.0; 4] {
                continue;
            }
            let overlap = bbox_overlap_ratio(ri.bbox, rj.bbox);
            let overlap_rev = bbox_overlap_ratio(rj.bbox, ri.bbox);
            // If either region covers >50% of the other, they're duplicates
            if overlap > 0.5 || overlap_rev > 0.5 {
                let pi = region_priority(ri.kind);
                let pj = region_priority(rj.kind);
                if pi >= pj {
                    drop_indices.insert(j);
                } else {
                    drop_indices.insert(i);
                    break; // i is dropped, no need to compare further
                }
            }
        }
    }

    let mut idx = 0;
    regions.retain(|r| {
        let keep = !drop_indices.contains(&idx);
        // Also apply the original structural filters
        let dominated = matches!(
            r.kind,
            RegionKind::FormulaNumber | RegionKind::PageHeader | RegionKind::PageFooter
        ) || header_bboxes
            .iter()
            .any(|hdr| bbox_contains(*hdr, r.bbox))
            || (r.kind == RegionKind::Text
                && figure_bboxes
                    .iter()
                    .any(|fig| bbox_contains(*fig, r.bbox)));
        idx += 1;
        keep && !dominated
    });
}

/// Resolve `--chapter` or `--section` to a set of 0-indexed PDF page indices.
fn resolve_toc_page_range(
    toc_result: &Option<toc::TocResult>,
    total_pages: u32,
    chapter: Option<&str>,
    section: Option<&str>,
) -> Result<Vec<u32>, ExtractError> {
    let toc = toc_result.as_ref().ok_or_else(|| {
        ExtractError::Pdf("No TOC found in this PDF; use --pages instead".into())
    })?;

    let spec = chapter.or(section).unwrap();
    let entries = &toc.entries;

    // Find matching TOC entry
    let entry_idx = entries
        .iter()
        .position(|e| {
            // Match by number prefix: "3" matches "3 Introduction"
            let title_lower = e.title.to_lowercase();
            let spec_lower = spec.to_lowercase();
            title_lower.starts_with(&format!("{spec_lower} "))
                || title_lower.starts_with(&format!("{spec_lower}."))
                || title_lower == spec_lower
        })
        .ok_or_else(|| {
            ExtractError::Pdf(format!("TOC entry not found for: {spec}"))
        })?;

    let entry = &entries[entry_idx];
    let entry_depth = entry.depth;

    // Find start page
    let start_page_value = entry.page_value;

    // Find next entry at same or shallower depth → end boundary
    let end_page_value = entries[entry_idx + 1..]
        .iter()
        .find(|e| e.depth <= entry_depth)
        .map(|e| e.page_value);

    // Compute page offset using fallback (toc_pages end)
    let offset = toc.toc_pages.last().map(|&p| p as i32 + 1).unwrap_or(0);

    let fm_offset = 0; // front-matter offset not needed for chapter extraction
    let start = output::toc_page_to_pdf_page(start_page_value, offset, fm_offset, total_pages)
        .unwrap_or(0);
    let end = end_page_value
        .and_then(|pv| output::toc_page_to_pdf_page(pv, offset, fm_offset, total_pages))
        .unwrap_or(total_pages);

    eprintln!(
        "  Chapter/section \"{spec}\": pages {}-{} (PDF pages {}-{})",
        start + 1,
        end,
        start,
        end.saturating_sub(1)
    );

    Ok((start..end).collect())
}

/// Detect drop caps masquerading as formulas.
///
/// Pattern: a single uppercase letter followed by `^{...}` or `_{...}` where
/// the braced content is 50+ chars of prose (has spaces, mostly alphabetic).
/// Returns the expanded plain text if detected.
///
/// Example: `"P^{hysics is a hot topic...}"` → `Some("Physics is a hot topic...")`
fn detect_drop_cap(latex: &str) -> Option<String> {
    let t = latex.trim();
    if t.len() < 55 {
        return None;
    }
    let bytes = t.as_bytes();
    // Must start with a single uppercase letter
    let base = bytes[0] as char;
    if !base.is_ascii_uppercase() {
        return None;
    }
    // Followed by ^ or _
    if bytes.len() < 4 {
        return None;
    }
    let marker = bytes[1] as char;
    if marker != '^' && marker != '_' {
        return None;
    }
    // Then an opening brace
    if bytes[2] as char != '{' {
        return None;
    }
    // Find closing brace
    let rest = &t[3..];
    let close = rest.rfind('}')?;
    let content = &rest[..close];
    if content.len() < 50 {
        return None;
    }
    // Must be prose-like: has multiple spaces
    let space_count = content.chars().filter(|c| *c == ' ').count();
    if space_count < 5 {
        return None;
    }
    // Mostly alphabetic prose (>70% alpha or space, no backslashes)
    if content.contains('\\') {
        return None; // LaTeX commands → real formula
    }
    let alpha = content
        .chars()
        .filter(|c| c.is_alphabetic() || *c == ' ')
        .count();
    if (alpha as f32 / content.len() as f32) < 0.7 {
        return None;
    }
    Some(format!("{base}{content}"))
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
            items: None,
            formula_source: None,
            ocr_confidence: None,
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

    // ── Drop cap detection tests ─────────────────────────────────────

    #[test]
    fn test_detect_drop_cap_basic() {
        let latex = "P^{hysics is a hot topic in computer games and these days physics simulations are everywhere}";
        let result = detect_drop_cap(latex);
        assert!(result.is_some());
        assert!(result.unwrap().starts_with("Physics is a hot"));
    }

    #[test]
    fn test_detect_drop_cap_subscript() {
        let latex = "T_{he quick brown fox jumps over the lazy dog and keeps on running endlessly through fields}";
        let result = detect_drop_cap(latex);
        assert!(result.is_some());
        assert!(result.unwrap().starts_with("The quick brown"));
    }

    #[test]
    fn test_detect_drop_cap_short_not_matched() {
        assert!(detect_drop_cap("P^{2}").is_none());
        assert!(detect_drop_cap("x^{n}").is_none());
    }

    #[test]
    fn test_detect_drop_cap_multi_letter_base() {
        let latex = "PQ^{some long text that goes on for quite a while in a paragraph of text about things}";
        assert!(detect_drop_cap(latex).is_none());
    }

    #[test]
    fn test_detect_drop_cap_lowercase_base() {
        let latex = "p^{hysics is a hot topic in computer games and these days physics simulations are everywhere}";
        assert!(detect_drop_cap(latex).is_none());
    }

    #[test]
    fn test_detect_drop_cap_no_spaces() {
        let latex = "A^{bcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyzabcdefghij}";
        assert!(detect_drop_cap(latex).is_none());
    }

    #[test]
    fn test_detect_drop_cap_math_content() {
        let latex = r"X^{\alpha + \beta + \gamma + \delta + \epsilon + \zeta + \eta + \theta + \iota + \kappa}";
        assert!(detect_drop_cap(latex).is_none());
    }

    #[test]
    fn test_detect_drop_cap_legitimate_formulas() {
        assert!(detect_drop_cap("E = mc^{2}").is_none());
        assert!(detect_drop_cap("x^{n} + y^{n} = z^{n}").is_none());
        assert!(detect_drop_cap(r"\sum_{i=0}^{n} x_i").is_none());
    }

    #[test]
    fn test_detect_drop_cap_no_closing_brace() {
        let latex = "P^{hysics is a hot topic in computer games and these days physics simulations";
        assert!(detect_drop_cap(latex).is_none());
    }
}

/// Detect if a Text region likely contains a display formula.
///
/// Formulas misclassified as Text have these characteristics:
/// - Short content (< 100 visible chars)
/// - High ratio of math-like chars (operators, single letters, parens)
/// - Many italic chars (variables rendered in italic)
/// - Few multi-letter words (prose has long words, math has single-char variables)
fn is_likely_formula(chars: &[&PdfChar]) -> bool {
    let visible: Vec<&&PdfChar> = chars.iter()
        .filter(|c| !c.codepoint.is_control() && c.codepoint != ' ')
        .collect();

    if visible.len() < 3 || visible.len() > 80 {
        return false;
    }

    let mut math_chars = 0;
    let mut italic_chars = 0;
    let mut letter_chars = 0;

    for c in &visible {
        let ch = c.codepoint;
        if matches!(ch, '=' | '+' | '-' | '×' | '÷' | '∫' | '∑' | '∏'
            | '≤' | '≥' | '≠' | '∈' | '∉' | '⊂' | '⊃' | '∪' | '∩'
            | '(' | ')' | '[' | ']' | '{' | '}' | '/' | '|' | ','
            | '→' | '←' | '↔' | '∞' | '∂' | '∇')
        {
            math_chars += 1;
        }
        if ch.is_alphabetic() {
            letter_chars += 1;
        }
        if c.is_italic {
            italic_chars += 1;
        }
    }

    let total = visible.len() as f32;
    let math_ratio = math_chars as f32 / total;
    let italic_ratio = italic_chars as f32 / total;

    // High math symbol ratio (> 15%) → likely formula
    if math_ratio > 0.15 && total < 60.0 {
        return true;
    }

    // High italic ratio (> 50%) + math symbols → likely formula with variables
    if italic_ratio > 0.5 && math_chars >= 2 && total < 60.0 {
        return true;
    }

    // Very short with any math operators → likely inline formula promoted to display
    if total < 15.0 && math_chars >= 1 {
        return true;
    }

    false
}

/// Trim an Algorithm region's bbox to exclude trailing body text.
///
/// The layout model sometimes extends the Algorithm bbox too far down,
/// capturing prose text below the code. We detect this by checking for
/// a font-family change: code uses monospace fonts (LucidaSans, Courier,
/// Consolas, etc.) while prose uses serif/sans-serif body fonts.
/// When we find the font change, we trim the bbox to stop above it.
fn trim_algorithm_bbox(chars: &[PdfChar], bbox: [f32; 4], page_height_pt: f32) -> [f32; 4] {
    // Collect chars inside the bbox
    let region_chars: Vec<&PdfChar> = chars.iter()
        .filter(|c| {
            if c.codepoint.is_control() || c.codepoint == ' ' {
                return false;
            }
            let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
            let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
            let cy_img = page_height_pt - cy;
            cx >= bbox[0] && cx <= bbox[2] && cy_img >= bbox[1] && cy_img <= bbox[3]
        })
        .collect();

    if region_chars.len() < 10 {
        return bbox;
    }

    // Find the dominant font (should be the code font)
    let mut font_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for c in &region_chars {
        let family = crate::output::normalize_font_family(&c.font_name);
        if !family.is_empty() {
            *font_counts.entry(family).or_default() += 1;
        }
    }
    // For Algorithm regions, prefer monospace fonts as the "dominant" code font.
    // If a monospace font is present with significant usage, use it even if
    // a body font has more chars (due to bbox overlap with body text).
    let is_mono = |name: &str| -> bool {
        let n = name.to_ascii_lowercase();
        n.contains("lucida") || n.contains("courier") || n.contains("consola")
            || n.contains("mono") || n.contains("typewriter") || n.contains("menlo")
            || n.contains("source code") || n.contains("fira code")
    };
    let mono_font = font_counts.iter()
        .filter(|(name, count)| is_mono(name) && **count >= 10)
        .max_by_key(|(_, count)| *count)
        .map(|(f, _)| f.clone());
    let dominant = if let Some(mono) = mono_font {
        mono
    } else {
        match font_counts.iter().max_by_key(|(_, count)| *count) {
            Some((f, _)) => f.clone(),
            None => return bbox,
        }
    };

    // Scan from the bottom of the region upward. Find the first line
    // (by Y position) where the dominant font switches to a different font.
    // Sort chars by Y (image space: higher Y = lower on page)
    let mut by_y: Vec<(f32, &str)> = region_chars.iter()
        .map(|c| {
            let cy_img = page_height_pt - (c.bbox[1] + c.bbox[3]) / 2.0;
            (cy_img, c.font_name.as_str())
        })
        .collect();
    by_y.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Group into lines by Y proximity
    let avg_h = {
        let heights: Vec<f32> = region_chars.iter()
            .map(|c| (c.bbox[3] - c.bbox[1]).abs())
            .filter(|h| *h > 0.5)
            .collect();
        if heights.is_empty() { return bbox; }
        heights.iter().sum::<f32>() / heights.len() as f32
    };
    let y_thresh = avg_h * 0.5;

    // Group chars into lines by Y proximity (image coords, top-to-bottom)
    let mut lines: Vec<(f32, std::collections::HashMap<String, usize>)> = Vec::new();
    let mut current_y = f32::NAN;
    for &(cy, font_name) in &by_y {
        let family = crate::output::normalize_font_family(font_name);
        if family.is_empty() { continue; }

        if current_y.is_nan() || (current_y - cy).abs() > y_thresh {
            lines.push((cy, std::collections::HashMap::new()));
            current_y = cy;
        }
        *lines.last_mut().unwrap().1.entry(family).or_default() += 1;
    }

    if lines.len() < 2 {
        return bbox;
    }

    // Find the line-level dominant font for each line
    let line_dominants: Vec<(f32, String)> = lines.iter()
        .map(|(y, fonts)| {
            let dom = fonts.iter()
                .max_by_key(|(_, c)| *c)
                .map(|(f, _)| f.clone())
                .unwrap_or_default();
            (*y, dom)
        })
        .collect();

    // Trim from top: skip leading lines with non-code font
    let mut new_y1 = bbox[1];
    for (y, font) in &line_dominants {
        if *font != dominant {
            new_y1 = *y + avg_h * 0.5; // move top below this line
        } else {
            break;
        }
    }

    // Trim from bottom: skip trailing lines with non-code font
    let mut new_y2 = bbox[3];
    for (y, font) in line_dominants.iter().rev() {
        if *font != dominant {
            new_y2 = *y - avg_h * 0.5; // move bottom above this line
        } else {
            break;
        }
    }

    // Only trim if we still have meaningful content
    if new_y2 > new_y1 + avg_h * 2.0 {
        [bbox[0], new_y1, bbox[2], new_y2]
    } else {
        bbox
    }
}

/// Detect if a Text region contains formatted/structured text that should
/// preserve its layout (mini-TOCs, structured listings, decorated blocks).
///
/// Tells:
/// - Multiple lines where many end with digits (page numbers)
/// - Leader dots (". . ." or "...") on multiple lines
/// - Lines significantly shorter than the region width
/// - Consistent repeating visual pattern
fn is_formatted_text(chars: &[&PdfChar], page_height_pt: f32, bbox: [f32; 4]) -> bool {
    if chars.len() < 20 {
        return false;
    }

    // Group chars into lines by Y proximity
    let avg_h = {
        let heights: Vec<f32> = chars.iter()
            .map(|c| (c.bbox[3] - c.bbox[1]).abs())
            .filter(|h| *h > 0.5)
            .collect();
        if heights.is_empty() { return false; }
        heights.iter().sum::<f32>() / heights.len() as f32
    };
    let y_thresh = avg_h * 0.5;

    // Count distinct lines
    let mut line_ys: Vec<f32> = Vec::new();
    for c in chars {
        // Use image-space Y for grouping
        let cy = page_height_pt - (c.bbox[1] + c.bbox[3]) / 2.0;
        if !line_ys.iter().any(|&ly| (ly - cy).abs() < y_thresh) {
            line_ys.push(cy);
        }
    }

    if line_ys.len() < 4 {
        return false;
    }

    // For each line, check the last visible char and count dots
    let mut lines_ending_digit = 0;
    let mut lines_with_dots = 0;

    for &ly in &line_ys {
        // Collect chars on this line
        let line_chars: Vec<&&PdfChar> = chars.iter()
            .filter(|c| {
                let cy = page_height_pt - (c.bbox[1] + c.bbox[3]) / 2.0;
                (cy - ly).abs() < y_thresh
            })
            .collect();

        if line_chars.is_empty() {
            continue;
        }

        // Check last visible char
        let last = line_chars.iter()
            .filter(|c| !c.codepoint.is_control() && c.codepoint != ' ')
            .max_by(|a, b| a.bbox[2].partial_cmp(&b.bbox[2]).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(last_char) = last {
            if last_char.codepoint.is_ascii_digit() {
                lines_ending_digit += 1;
            }
        }

        // Count dots on this line
        let dot_count = line_chars.iter()
            .filter(|c| c.codepoint == '.')
            .count();
        if dot_count >= 5 {
            lines_with_dots += 1;
        }
    }

    let total_lines = line_ys.len();
    let pct_digit = lines_ending_digit as f32 / total_lines as f32;
    let pct_dots = lines_with_dots as f32 / total_lines as f32;

    // Formatted text: many lines end with digits AND have leader dots
    (pct_digit >= 0.4 && pct_dots >= 0.3) || pct_dots >= 0.5
}
