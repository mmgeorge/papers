use std::path::Path;

use image::DynamicImage;
use pdfium_render::prelude::*;

use crate::error::ExtractError;
use crate::pdf;
use crate::types::{ExtractionResult, Page, Region, RegionKind};

/// Write the extraction result as pretty-printed JSON.
pub fn write_json(result: &ExtractionResult, path: &Path) -> Result<(), ExtractError> {
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Write the extraction result as Markdown.
pub fn write_markdown(result: &ExtractionResult, path: &Path) -> Result<(), ExtractError> {
    let md = render_markdown(result);
    std::fs::write(path, md)?;
    Ok(())
}

/// Save cropped region images to the output directory.
pub fn write_images(
    pages: &[Page],
    page_images: &[DynamicImage],
    images_dir: &Path,
) -> Result<(), ExtractError> {
    std::fs::create_dir_all(images_dir)?;

    for (page, page_img) in pages.iter().zip(page_images.iter()) {
        for region in &page.regions {
            if let Some(ref rel_path) = region.image_path {
                let full_path = images_dir
                    .parent()
                    .unwrap_or(images_dir)
                    .join(rel_path);

                let cropped = crate::figure::crop_region(
                    page_img,
                    region.bbox,
                    page.width_pt,
                    page.height_pt,
                    page.dpi,
                );

                cropped.save(&full_path)?;
            }
        }
    }

    Ok(())
}

/// Render the full extraction result as Markdown.
fn render_markdown(result: &ExtractionResult) -> String {
    let mut md = String::new();

    for page in &result.pages {
        for region in &page.regions {
            let section = region_to_markdown(region);
            if !section.is_empty() {
                md.push_str(&section);
                md.push_str("\n\n");
            }
        }
    }

    // Remove trailing whitespace
    md.trim_end().to_string()
}

/// Convert a single region to its Markdown representation.
fn region_to_markdown(region: &Region) -> String {
    match region.kind {
        RegionKind::Title => {
            if let Some(ref text) = region.text {
                format!("# {text}")
            } else {
                String::new()
            }
        }
        RegionKind::ParagraphTitle => {
            if let Some(ref text) = region.text {
                format!("## {text}")
            } else {
                String::new()
            }
        }
        RegionKind::Text
        | RegionKind::VerticalText
        | RegionKind::Abstract
        | RegionKind::SidebarText
        | RegionKind::References => region.text.clone().unwrap_or_default(),

        RegionKind::Table => {
            if let Some(ref html) = region.html {
                html.clone()
            } else {
                String::new()
            }
        }
        RegionKind::DisplayFormula => {
            if let Some(ref latex) = region.latex {
                format!("$${latex}$$")
            } else {
                String::new()
            }
        }
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => {
            let path = region.image_path.as_deref().unwrap_or("");
            let alt = region.caption.as_deref().unwrap_or("");
            format!("![{alt}]({path})")
        }
        RegionKind::Algorithm => {
            if let Some(ref text) = region.text {
                format!("```\n{text}\n```")
            } else {
                String::new()
            }
        }
        RegionKind::Footnote => {
            if let Some(ref text) = region.text {
                format!("[^]: {text}")
            } else {
                String::new()
            }
        }
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => {
            // Captions are associated with parents; skip standalone rendering
            String::new()
        }
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber | RegionKind::TOC => {
            // Skip page structural elements
            String::new()
        }
        RegionKind::FormulaNumber => {
            // Formula numbers are inline metadata, skip
            String::new()
        }
    }
}

// ── Debug visualization ──────────────────────────────────────────────

/// Map a RegionKind to a debug visualization color.
fn region_color(kind: RegionKind) -> PdfColor {
    match kind {
        RegionKind::Title | RegionKind::ParagraphTitle => PdfColor::new(255, 50, 50, 255),
        RegionKind::Text
        | RegionKind::VerticalText
        | RegionKind::Abstract
        | RegionKind::SidebarText => PdfColor::new(50, 100, 255, 255),
        RegionKind::References | RegionKind::Footnote | RegionKind::TOC => {
            PdfColor::new(0, 180, 180, 255)
        }
        RegionKind::Table => PdfColor::new(0, 200, 0, 255),
        RegionKind::DisplayFormula | RegionKind::FormulaNumber => PdfColor::new(200, 0, 200, 255),
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => PdfColor::new(255, 140, 0, 255),
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => PdfColor::new(220, 200, 0, 255),
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber => {
            PdfColor::new(150, 150, 150, 255)
        }
        RegionKind::Algorithm => PdfColor::new(0, 160, 120, 255),
    }
}

/// Write debug visualization: annotate the original PDF with colored bounding boxes
/// and labels, save as `{stem}_debug.pdf`, and render annotated pages as PNGs.
pub fn write_debug(
    pdfium: &Pdfium,
    pdf_path: &Path,
    pages: &[Page],
    output_dir: &Path,
    stem: &str,
) -> Result<(), ExtractError> {
    let mut doc = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| ExtractError::Pdf(format!("Failed to re-open PDF for debug: {e}")))?;

    let font = doc.fonts_mut().courier_bold();

    for page_info in pages {
        let page_idx = page_info.page - 1; // 0-indexed
        let mut page = doc.pages().get(page_idx as u16).map_err(|e| {
            ExtractError::Pdf(format!("Failed to get page {page_idx} for debug: {e}"))
        })?;

        let page_h = page_info.height_pt;

        for region in &page_info.regions {
            let color = region_color(region.kind);
            let [x1, y1, x2, y2] = region.bbox;

            // bbox is in top-left-origin points; convert to PDF bottom-left-origin
            let pdf_left = x1;
            let pdf_right = x2;
            let pdf_bottom = page_h - y2;
            let pdf_top = page_h - y1;

            // Draw bounding box rectangle (stroke only, no fill)
            page.objects_mut()
                .create_path_object_rect(
                    PdfRect::new_from_values(pdf_bottom, pdf_left, pdf_top, pdf_right),
                    Some(color),
                    Some(PdfPoints::new(1.0)),
                    None,
                )
                .map_err(|e| ExtractError::Pdf(format!("Failed to draw debug rect: {e}")))?;

            // Label text
            let label = format!(
                "{:?} #{} {}%",
                region.kind,
                region.order,
                (region.confidence * 100.0) as u32
            );

            let font_size = 7.0;
            // Place label just above the box top edge
            let label_y = if pdf_top + font_size + 1.0 < page_h {
                pdf_top + 1.0
            } else {
                pdf_top - font_size - 1.0
            };

            let mut text_obj = PdfPageTextObject::new(
                &doc,
                &label,
                font,
                PdfPoints::new(font_size),
            )
            .map_err(|e| ExtractError::Pdf(format!("Failed to create debug text: {e}")))?;

            text_obj
                .translate(PdfPoints::new(pdf_left), PdfPoints::new(label_y))
                .map_err(|e| ExtractError::Pdf(format!("Failed to position debug text: {e}")))?;
            text_obj
                .set_fill_color(color)
                .map_err(|e| ExtractError::Pdf(format!("Failed to color debug text: {e}")))?;

            page.objects_mut()
                .add_text_object(text_obj)
                .map_err(|e| ExtractError::Pdf(format!("Failed to add debug text: {e}")))?;
        }
    }

    // Save annotated PDF
    let debug_pdf_path = output_dir.join(format!("{stem}_debug.pdf"));
    doc.save_to_file(&debug_pdf_path)
        .map_err(|e| ExtractError::Pdf(format!("Failed to save debug PDF: {e}")))?;

    // Render annotated pages to PNGs
    let identified_dir = output_dir.join("identified");
    std::fs::create_dir_all(&identified_dir)?;

    for page_info in pages {
        let page_idx = page_info.page - 1;
        let page = doc.pages().get(page_idx as u16).map_err(|e| {
            ExtractError::Pdf(format!("Failed to get page {page_idx} for render: {e}"))
        })?;
        let img = pdf::render_page(&page, page_info.dpi)?;
        let out_path = identified_dir.join(format!("p{}.png", page_info.page));
        img.save(&out_path)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ExtractionResult, Metadata, Page, Region, RegionKind};

    fn make_region(kind: RegionKind) -> Region {
        Region {
            id: String::new(),
            kind,
            bbox: [0.0; 4],
            confidence: 1.0,
            order: 0,
            text: None,
            html: None,
            latex: None,
            image_path: None,
            caption: None,
            chart_type: None,
        }
    }

    #[test]
    fn test_title_to_markdown() {
        let mut r = make_region(RegionKind::Title);
        r.text = Some("My Title".into());
        assert_eq!(region_to_markdown(&r), "# My Title");
    }

    #[test]
    fn test_paragraph_title_to_markdown() {
        let mut r = make_region(RegionKind::ParagraphTitle);
        r.text = Some("Section 1".into());
        assert_eq!(region_to_markdown(&r), "## Section 1");
    }

    #[test]
    fn test_text_to_markdown() {
        let mut r = make_region(RegionKind::Text);
        r.text = Some("Hello world.".into());
        assert_eq!(region_to_markdown(&r), "Hello world.");
    }

    #[test]
    fn test_table_to_markdown() {
        let mut r = make_region(RegionKind::Table);
        r.html = Some("<table><tr><td>A</td></tr></table>".into());
        assert_eq!(
            region_to_markdown(&r),
            "<table><tr><td>A</td></tr></table>"
        );
    }

    #[test]
    fn test_formula_to_markdown() {
        let mut r = make_region(RegionKind::DisplayFormula);
        r.latex = Some("E = mc^2".into());
        assert_eq!(region_to_markdown(&r), "$$E = mc^2$$");
    }

    #[test]
    fn test_image_to_markdown() {
        let mut r = make_region(RegionKind::Image);
        r.image_path = Some("images/p1_7.png".into());
        r.caption = Some("Figure 1".into());
        assert_eq!(region_to_markdown(&r), "![Figure 1](images/p1_7.png)");
    }

    #[test]
    fn test_algorithm_to_markdown() {
        let mut r = make_region(RegionKind::Algorithm);
        r.text = Some("for i in range(n):".into());
        assert_eq!(region_to_markdown(&r), "```\nfor i in range(n):\n```");
    }

    #[test]
    fn test_skip_page_header_footer() {
        let r1 = make_region(RegionKind::PageHeader);
        let r2 = make_region(RegionKind::PageFooter);
        assert!(region_to_markdown(&r1).is_empty());
        assert!(region_to_markdown(&r2).is_empty());
    }

    #[test]
    fn test_multi_page_assembly() {
        let result = ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 2,
                extraction_time_ms: 0,
            },
            pages: vec![
                Page {
                    page: 1,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::Title);
                        r.text = Some("Page 1 Title".into());
                        r.order = 0;
                        r
                    }],
                },
                Page {
                    page: 2,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::Text);
                        r.text = Some("Page 2 text.".into());
                        r.order = 0;
                        r
                    }],
                },
            ],
        };

        let md = render_markdown(&result);
        assert!(md.contains("# Page 1 Title"));
        assert!(md.contains("Page 2 text."));
    }
}
