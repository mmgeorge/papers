use std::path::Path;
use std::time::Instant;

use image::DynamicImage;
use oar_ocr::domain::structure::{FormulaResult, LayoutElementType, StructureResult, TableResult};
use oar_ocr::oarocr::structure::OARStructure;
use pdfium_render::prelude::*;

use crate::error::ExtractError;
use crate::figure;
use crate::models;
use crate::output;
use crate::pdf::{self, PdfChar};
use crate::reading_order;
use crate::text;
use crate::types::*;
use crate::ExtractOptions;

/// Reusable extraction pipeline — load models once, extract many PDFs.
pub struct Pipeline {
    pdfium: Pdfium,
    structure: OARStructure,
    options: PipelineOptions,
}

/// Internal options carried from ExtractOptions.
struct PipelineOptions {
    dpi: u32,
    confidence_threshold: f32,
    extract_images: bool,
    page: Option<u32>,
    debug: bool,
}

impl Pipeline {
    /// Create a new pipeline, loading models and pdfium.
    pub fn new(options: &ExtractOptions) -> Result<Self, ExtractError> {
        let pdfium = pdf::load_pdfium(options.pdfium_path.as_deref())?;

        let cache_dir = options
            .model_cache_dir
            .clone()
            .unwrap_or_else(models::default_cache_dir);

        let paths = models::ensure_models(options.quality, &cache_dir)?;
        let structure = models::build_structure(&paths, options.quality)?;

        Ok(Self {
            pdfium,
            structure,
            options: PipelineOptions {
                dpi: options.dpi,
                confidence_threshold: options.confidence_threshold,
                extract_images: options.extract_images,
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

        if self.options.debug {
            output::write_debug(&self.pdfium, pdf_path, &result.pages, output_dir, stem)?;
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

        // Run layout detection + table/formula recognition
        let rgb_image = page_image.to_rgb8();
        let structure_result = self
            .structure
            .predict_image(rgb_image)
            .map_err(|e| ExtractError::Layout(format!("Layout detection failed: {e}")))?;

        // Convert layout elements to our Region types
        let mut regions = self.build_regions(
            &structure_result,
            &chars,
            &page_image,
            page_idx,
            width_pt,
            height_pt,
        )?;

        // If the model didn't provide reading order, use XY-Cut fallback
        let has_model_order = regions.iter().any(|r| r.order > 0);
        if !has_model_order && regions.len() > 1 {
            reading_order::xy_cut_order(&mut regions);
        }

        // Sort regions by reading order
        regions.sort_by_key(|r| r.order);

        // Associate captions with their parent regions
        figure::associate_captions(&mut regions);

        let result_page = Page {
            page: page_idx + 1,
            width_pt,
            height_pt,
            dpi: self.options.dpi,
            regions,
        };

        Ok((result_page, page_image))
    }

    /// Convert StructureResult layout elements into our Region types.
    fn build_regions(
        &self,
        result: &StructureResult,
        chars: &[PdfChar],
        page_image: &DynamicImage,
        page_idx: u32,
        page_width_pt: f32,
        page_height_pt: f32,
    ) -> Result<Vec<Region>, ExtractError> {
        let mut regions = Vec::new();

        for element in &result.layout_elements {
            if element.confidence < self.options.confidence_threshold {
                continue;
            }

            let kind = match map_element_type(&element.element_type) {
                Some(k) => k,
                None => continue,
            };

            // Convert oar-ocr bbox (pixel coords) to PDF points
            let scale = self.options.dpi as f32 / 72.0;
            let bbox = [
                element.bbox.x_min() as f32 / scale,
                element.bbox.y_min() as f32 / scale,
                element.bbox.x_max() as f32 / scale,
                element.bbox.y_max() as f32 / scale,
            ];

            let order = element.order_index.unwrap_or(0);
            let id = format!("p{}_{}", page_idx + 1, order);

            let mut region = Region {
                id,
                kind,
                bbox,
                confidence: element.confidence,
                order,
                text: None,
                html: None,
                latex: None,
                image_path: None,
                caption: None,
                chart_type: None,
            };

            // Populate content based on region kind
            if kind.is_text_bearing() || kind.is_caption() {
                region.text = Some(text::extract_region_text(chars, bbox));
            }

            if kind == RegionKind::Table {
                // Find matching TableResult by bbox overlap (using pixel coords)
                let elem_coords = (
                    element.bbox.x_min() as f32,
                    element.bbox.y_min() as f32,
                    element.bbox.x_max() as f32,
                    element.bbox.y_max() as f32,
                );
                if let Some(table) = find_matching_table(&result.tables, elem_coords) {
                    region.html = table.html_structure.clone();
                }
            }

            if kind == RegionKind::DisplayFormula {
                // Find matching FormulaResult by bbox overlap (using pixel coords)
                let elem_coords = (
                    element.bbox.x_min() as f32,
                    element.bbox.y_min() as f32,
                    element.bbox.x_max() as f32,
                    element.bbox.y_max() as f32,
                );
                if let Some(formula) = find_matching_formula(&result.formulas, elem_coords) {
                    region.latex = Some(formula.latex.clone());
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

/// Map oar-ocr LayoutElementType to our RegionKind.
fn map_element_type(t: &LayoutElementType) -> Option<RegionKind> {
    use LayoutElementType::*;
    match t {
        DocTitle => Some(RegionKind::Title),
        ParagraphTitle => Some(RegionKind::ParagraphTitle),
        Text | Content | List => Some(RegionKind::Text),
        Abstract => Some(RegionKind::Abstract),
        Image => Some(RegionKind::Image),
        Table => Some(RegionKind::Table),
        Chart => Some(RegionKind::Chart),
        Formula => Some(RegionKind::DisplayFormula),
        FigureTitle => Some(RegionKind::FigureTitle),
        TableTitle => Some(RegionKind::TableTitle),
        ChartTitle => Some(RegionKind::ChartTitle),
        FigureTableChartTitle => Some(RegionKind::FigureTableTitle),
        Header | HeaderImage => Some(RegionKind::PageHeader),
        Footer | FooterImage => Some(RegionKind::PageFooter),
        Footnote => Some(RegionKind::Footnote),
        Seal => Some(RegionKind::Seal),
        Number => Some(RegionKind::PageNumber),
        Reference | ReferenceContent => Some(RegionKind::References),
        Algorithm => Some(RegionKind::Algorithm),
        FormulaNumber => Some(RegionKind::FormulaNumber),
        AsideText => Some(RegionKind::SidebarText),
        Region | Other => None,
    }
}

/// Find a TableResult whose bbox overlaps with the given element coordinates (pixels).
fn find_matching_table<'a>(
    tables: &'a [TableResult],
    elem: (f32, f32, f32, f32),
) -> Option<&'a TableResult> {
    let (ex1, ey1, ex2, ey2) = elem;

    tables
        .iter()
        .max_by(|a, b| {
            let iou_a = compute_iou_with_table(ex1, ey1, ex2, ey2, a);
            let iou_b = compute_iou_with_table(ex1, ey1, ex2, ey2, b);
            iou_a
                .partial_cmp(&iou_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|t| compute_iou_with_table(ex1, ey1, ex2, ey2, t) > 0.3)
}

/// Find a FormulaResult whose bbox overlaps with the given element coordinates (pixels).
fn find_matching_formula<'a>(
    formulas: &'a [FormulaResult],
    elem: (f32, f32, f32, f32),
) -> Option<&'a FormulaResult> {
    let (ex1, ey1, ex2, ey2) = elem;

    formulas
        .iter()
        .max_by(|a, b| {
            let iou_a = compute_iou_with_formula(ex1, ey1, ex2, ey2, a);
            let iou_b = compute_iou_with_formula(ex1, ey1, ex2, ey2, b);
            iou_a
                .partial_cmp(&iou_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .filter(|f| compute_iou_with_formula(ex1, ey1, ex2, ey2, f) > 0.3)
}

/// Compute IoU between an element bbox and a TableResult bbox.
fn compute_iou_with_table(
    ex1: f32,
    ey1: f32,
    ex2: f32,
    ey2: f32,
    table: &TableResult,
) -> f32 {
    compute_iou(
        ex1,
        ey1,
        ex2,
        ey2,
        table.bbox.x_min() as f32,
        table.bbox.y_min() as f32,
        table.bbox.x_max() as f32,
        table.bbox.y_max() as f32,
    )
}

/// Compute IoU between an element bbox and a FormulaResult bbox.
fn compute_iou_with_formula(
    ex1: f32,
    ey1: f32,
    ex2: f32,
    ey2: f32,
    formula: &FormulaResult,
) -> f32 {
    compute_iou(
        ex1,
        ey1,
        ex2,
        ey2,
        formula.bbox.x_min() as f32,
        formula.bbox.y_min() as f32,
        formula.bbox.x_max() as f32,
        formula.bbox.y_max() as f32,
    )
}

/// Compute Intersection over Union between two axis-aligned bounding boxes.
fn compute_iou(
    ax1: f32,
    ay1: f32,
    ax2: f32,
    ay2: f32,
    bx1: f32,
    by1: f32,
    bx2: f32,
    by2: f32,
) -> f32 {
    let inter_x1 = ax1.max(bx1);
    let inter_y1 = ay1.max(by1);
    let inter_x2 = ax2.min(bx2);
    let inter_y2 = ay2.min(by2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (ax2 - ax1) * (ay2 - ay1);
    let area_b = (bx2 - bx1) * (by2 - by1);
    let union = area_a + area_b - inter_area;

    if union > 0.0 {
        inter_area / union
    } else {
        0.0
    }
}
