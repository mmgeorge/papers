use std::path::Path;

use image::DynamicImage;
use pdfium_render::prelude::*;

use crate::error::ExtractError;
use crate::pdf;
use crate::types::{ExtractionResult, Page, Region, RegionKind};
use crate::DebugMode;

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

/// Save a single region's image if it has an `image_path`.
fn save_region_image(
    region: &Region,
    page_img: &DynamicImage,
    page: &Page,
    images_dir: &Path,
) -> Result<(), ExtractError> {
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
            save_region_image(region, page_img, page, images_dir)?;
            // Also save images for FigureGroup member items
            if let Some(ref items) = region.items {
                for item in items {
                    save_region_image(item, page_img, page, images_dir)?;
                }
            }
        }
    }

    Ok(())
}

/// Save cropped formula region images to the output directory.
pub fn write_formula_images(
    pages: &[Page],
    page_images: &[DynamicImage],
    formulas_dir: &Path,
) -> Result<(), ExtractError> {
    std::fs::create_dir_all(formulas_dir)?;

    for (page, page_img) in pages.iter().zip(page_images.iter()) {
        for region in &page.regions {
            if region.kind != RegionKind::DisplayFormula
                && region.kind != RegionKind::InlineFormula
            {
                continue;
            }

            let filename = format!("{}.png", region.id);
            let full_path = formulas_dir.join(&filename);

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

    Ok(())
}

/// Render the full extraction result as Markdown.
fn render_markdown(result: &ExtractionResult) -> String {
    // Collect rendered sections with metadata for cross-region dehyphenation.
    struct Section {
        markdown: String,
        kind: RegionKind,
        is_text: bool,
        is_references: bool,
        /// Display formulas are part of the paragraph flow — text before/after
        /// them should NOT be rearranged across them.
        is_formula: bool,
        /// A short single-line text block that doesn't span the page width.
        /// These are typically author names, affiliations, or other metadata
        /// that the PDF intentionally places on separate lines. They should
        /// not be merged with neighboring text via the mid-sentence heuristic.
        is_short_line: bool,
    }

    let mut sections: Vec<Section> = Vec::new();

    for page in &result.pages {
        for region in &page.regions {
            // Skip regions whose content was spliced into a parent region.
            if region.consumed {
                continue;
            }
            let section = region_to_markdown(region);
            if section.is_empty() {
                continue;
            }
            let is_references = region.kind == RegionKind::References;
            let is_text = is_references
                || matches!(
                    region.kind,
                    RegionKind::Text
                        | RegionKind::VerticalText
                        | RegionKind::Abstract
                        | RegionKind::SidebarText
                );

            // Merge all references sections into a single section, even
            // when non-text regions (page headers/footers) sit between them.
            // When the previous chunk ends mid-sentence (no terminal punctuation),
            // join with a space so split entries like "ACM Transactions on" +
            // "Graphics 31, 5 ..." reconnect properly.
            if is_references {
                if let Some(prev) = sections.iter_mut().rfind(|s| s.is_references) {
                    let prev_trimmed = prev.markdown.trim_end();
                    let ends_mid = prev_trimmed
                        .ends_with(|c: char| !matches!(c, '.' | '!' | '?' | ':' | '"' | '\u{201D}'));
                    if ends_mid {
                        prev.markdown.push(' ');
                    } else {
                        prev.markdown.push_str("\n\n");
                    }
                    prev.markdown.push_str(section.trim());
                    continue;
                }
            }

            let is_formula = region.kind == RegionKind::DisplayFormula;

            // Detect short single-line text blocks: narrow region AND short
            // text content. These are typically author names, affiliations,
            // sub-figure labels, or other metadata that the PDF intentionally
            // places on separate lines — NOT paragraph fragments.
            //
            // Width threshold is 35% of page width. This cleanly separates:
            //   - Author lines / labels: ~20-30% of page width → narrow
            //   - Two-column paragraph blocks: ~40%+ of page width → not narrow
            //   - Single-column paragraphs: ~85%+ → not narrow
            //
            // The text-length check (< 80 chars) further guards against
            // paragraph fragments that happen to be narrow.
            let is_short_line = is_text && {
                let region_width = region.bbox[2] - region.bbox[0];
                let has_bbox = region_width > 0.0;
                let narrow = has_bbox && region_width < page.width_pt * 0.35;
                let short_text = section.trim().len() < 80;
                narrow && short_text
            };

            sections.push(Section {
                markdown: section,
                kind: region.kind,
                is_text,
                is_references,
                is_formula,
                is_short_line,
            });
        }
    }

    // --- Dehyphenation across region boundaries ---
    //
    // When a text region ends with STX (U+0002), the last word was split by
    // a hyphen at a region boundary (e.g., "approx-" at end of one column,
    // "imation" at the top of the next). Move the trailing word fragment
    // to the front of the next text section, rejoining the hyphenated word.
    // Non-text sections between them (figures, formulas) are skipped over.
    let mut i = 0;
    while i < sections.len() {
        if sections[i].is_text && sections[i].markdown.ends_with('\u{0002}') {
            sections[i].markdown.pop(); // remove STX sentinel

            let split_pos = sections[i]
                .markdown
                .rfind(|c: char| c == ' ' || c == '\n')
                .map(|p| p + 1)
                .unwrap_or(0);
            let fragment = sections[i].markdown[split_pos..].to_string();
            sections[i].markdown.truncate(split_pos);

            if !fragment.is_empty() {
                let mut found = false;
                for j in (i + 1)..sections.len() {
                    if sections[j].is_text {
                        sections[j].markdown.insert_str(0, &fragment);
                        found = true;
                        break;
                    }
                }
                if !found {
                    sections[i].markdown.push_str(&fragment);
                }
            }
        }
        i += 1;
    }

    // --- Rejoin paragraphs split across column/page breaks ---
    //
    // In multi-column layouts, a paragraph may be split across two text
    // regions with non-text elements between them (figures, algorithms,
    // footnotes, captions). We detect this by checking if the previous
    // text section ends mid-sentence (no terminal punctuation). If so,
    // the two sections are halves of the same paragraph — merge them.
    //
    // Skip conditions (these should never participate in paragraph merging):
    //   - Abstract regions: structurally distinct from preceding metadata
    //   - Short single-line blocks (bbox < 35% page width AND < 80 chars):
    //     author names, affiliations, sub-figure labels like "(a) Reference"
    //
    // Backward search blockers (structural boundaries):
    //   - Display formulas: part of paragraph flow, text wraps around them
    //   - Titles / paragraph titles: new document sections
    //
    // The backward search also skips over short single-line blocks
    // transparently, so paragraph halves find each other even when
    // separated by sub-figure labels and figures.
    //
    // Examples:
    //   DO merge:  "...method presented superior"  +  "convergence behavior..."
    //   DO merge:  "...constraints" [Footnote] "and produce stable..."
    //   DO merge:  "...shows a" [(a) Ref] [Fig] [(b) AVBD] [Fig] "delicate card..."
    //   NO merge:  "...et al." [DisplayFormula] "1992] where..."
    //   NO merge:  "CHRIS GILES, USA"  +  "ELIE DIAZ, University..."
    {
        let mut i = 1;
        while i < sections.len() {
            if !sections[i].is_text || sections[i].markdown.trim().is_empty() {
                i += 1;
                continue;
            }
            if sections[i].kind == RegionKind::Abstract {
                i += 1;
                continue;
            }
            // Short single-line blocks (sub-figure labels, author lines)
            // are standalone — they should not merge into paragraphs.
            if sections[i].is_short_line {
                i += 1;
                continue;
            }

            // Search backward for the nearest paragraph-like text section,
            // skipping over short single-line blocks (sub-figure labels,
            // author lines, etc.) which are neither merge candidates nor
            // structural boundaries. Stop at actual boundaries (formulas,
            // titles) that mark distinct document sections.
            let mut blocked = false;
            let mut prev_text = None;
            for k in (0..i).rev() {
                if sections[k].is_formula
                    || matches!(
                        sections[k].kind,
                        RegionKind::Title | RegionKind::ParagraphTitle
                    )
                {
                    blocked = true;
                    break;
                }
                if sections[k].is_text
                    && !sections[k].markdown.trim().is_empty()
                    && !sections[k].is_short_line
                {
                    prev_text = Some(k);
                    break;
                }
            }
            if blocked {
                i += 1;
                continue;
            }
            let Some(k) = prev_text else {
                i += 1;
                continue;
            };

            // Check if the previous text ends mid-sentence.
            let prev_trimmed = sections[k].markdown.trim();
            let ends_mid = prev_trimmed
                .ends_with(|c: char| !matches!(c, '.' | '!' | '?' | ':' | '"' | '\u{201D}'));
            if !ends_mid {
                i += 1;
                continue;
            }

            // Two halves of the same paragraph — merge continuation into
            // the previous section and clear the current one.
            let text = sections[k].markdown.trim_end().to_string();
            let next_md = sections[i].markdown.trim().to_string();
            if !next_md.is_empty() {
                sections[k].markdown = format!("{text} {next_md}");
                sections[i].markdown = String::new();
            }
            i += 1;
        }
    }

    // Assemble with paragraph breaks between sections.
    let mut md = String::new();
    for sec in &sections {
        let text = sec.markdown.trim();
        if text.is_empty() {
            continue;
        }
        if !md.is_empty() {
            md.push_str("\n\n");
        }
        md.push_str(text);
    }

    md.trim_end().to_string()
}

/// Bold the label prefix in a figure/table caption.
///
/// Turns `"Fig. 1. Example..."` into `"**Fig. 1.** Example..."` and
/// `"Table 1. Performance..."` into `"**Table 1.** Performance..."`.
/// Leaves text unchanged if no recognized label prefix is found.
fn bold_caption_label(text: &str) -> String {
    use std::fmt::Write;

    // Match patterns: "Fig. N." / "Figure N." / "Table N." / "Algorithm N."
    // where N can be multi-part like "1" or "1a" or absent for sub-captions
    let prefixes = ["Fig.", "Figure", "Table", "Algorithm"];
    for prefix in prefixes {
        if !text.starts_with(prefix) {
            continue;
        }
        // Find the second period after the number: "Fig. 1." or "Table 1."
        let after_prefix = &text[prefix.len()..];
        if let Some(dot_pos) = after_prefix.find('.') {
            let end = prefix.len() + dot_pos + 1; // include the '.'
            let mut result = String::with_capacity(text.len() + 4);
            let _ = write!(result, "**{}**{}", &text[..end], &text[end..]);
            return result;
        }
    }
    text.to_string()
}

/// Convert a single region to its Markdown representation.
/// Strip `$$` or `$` delimiters the model may have included in its output.
fn strip_dollar_delimiters(s: &str) -> &str {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix("$$") {
        inner.strip_suffix("$$").unwrap_or(inner).trim()
    } else if let Some(inner) = s.strip_prefix('$') {
        inner.strip_suffix('$').unwrap_or(inner).trim()
    } else {
        s
    }
}

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
            let table_md = region
                .text
                .as_deref()
                .filter(|s| !s.is_empty())
                .or(region.html.as_deref())
                .unwrap_or_default();
            let caption_text = region
                .caption
                .as_ref()
                .and_then(|c| c.text.as_deref());
            if let Some(cap) = caption_text {
                format!("{table_md}\n\n{}", bold_caption_label(cap))
            } else {
                table_md.to_string()
            }
        }
        RegionKind::DisplayFormula => {
            if let Some(ref latex) = region.latex {
                let s = strip_dollar_delimiters(latex);
                format!("$${s}$$")
            } else {
                String::new()
            }
        }
        RegionKind::InlineFormula => {
            // Orphan inline formula (not consumed by a text region)
            if let Some(ref latex) = region.latex {
                let s = strip_dollar_delimiters(latex);
                format!("${s}$")
            } else {
                String::new()
            }
        }
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => {
            let path = region.image_path.as_deref().unwrap_or("");
            let caption_text = region
                .caption
                .as_ref()
                .and_then(|c| c.text.as_deref());
            if let Some(cap) = caption_text {
                format!("![]({path})\n\n{}", bold_caption_label(cap))
            } else {
                format!("![]({path})")
            }
        }
        RegionKind::Algorithm => {
            if let Some(ref text) = region.text {
                text.clone()
            } else {
                String::new()
            }
        }
        RegionKind::Footnote => {
            // Footnotes are metadata (author addresses, permissions, etc.)
            // — omit from the main markdown body.
            String::new()
        }
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => {
            // Consumed captions are handled by their parent (Image/Table/Chart).
            // Orphan captions that weren't consumed still get rendered with bold labels.
            if region.consumed {
                String::new()
            } else if let Some(ref text) = region.text {
                bold_caption_label(text)
            } else {
                String::new()
            }
        }
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber | RegionKind::TOC => {
            // Skip page structural elements
            String::new()
        }
        RegionKind::FormulaNumber => {
            // Formula numbers are inline metadata, skip
            String::new()
        }
        RegionKind::FigureGroup => {
            let mut parts = Vec::new();
            // Single composite image for the entire figure group
            if let Some(ref path) = region.image_path {
                parts.push(format!("![]({path})"));
            }
            // Group-level caption
            if let Some(ref cap) = region.caption {
                if let Some(ref text) = cap.text {
                    parts.push(bold_caption_label(text));
                }
            }
            parts.join("\n\n")
        }
    }
}

// ── Debug visualization ──────────────────────────────────────────────

/// Map a RegionKind to an (R, G, B) debug visualization color.
pub fn region_color_rgb(kind: RegionKind) -> [u8; 3] {
    match kind {
        RegionKind::Title | RegionKind::ParagraphTitle => [255, 50, 50],
        RegionKind::Text
        | RegionKind::VerticalText
        | RegionKind::Abstract
        | RegionKind::SidebarText => [50, 100, 255],
        RegionKind::References | RegionKind::Footnote | RegionKind::TOC => [0, 180, 180],
        RegionKind::Table => [0, 200, 0],
        RegionKind::DisplayFormula | RegionKind::FormulaNumber => [200, 0, 200],
        RegionKind::InlineFormula => [255, 100, 255],
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => [255, 140, 0],
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => [220, 200, 0],
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber => [150, 150, 150],
        RegionKind::Algorithm => [0, 160, 120],
        RegionKind::FigureGroup => [0, 200, 200],
    }
}

/// Draw a colored bounding box for a detected region onto an RGBA image.
///
/// `bbox_px` is `[x1, y1, x2, y2]` in pixel coordinates matching the image.
pub fn draw_region_box(
    img: &mut image::RgbaImage,
    kind: RegionKind,
    bbox_px: [f32; 4],
    thickness: u32,
) {
    let [r, g, b] = region_color_rgb(kind);
    let color = image::Rgba([r, g, b, 200]);
    let (iw, ih) = (img.width(), img.height());

    let x1 = (bbox_px[0] as u32).min(iw.saturating_sub(1));
    let y1 = (bbox_px[1] as u32).min(ih.saturating_sub(1));
    let x2 = (bbox_px[2] as u32).min(iw.saturating_sub(1));
    let y2 = (bbox_px[3] as u32).min(ih.saturating_sub(1));

    // Draw horizontal lines (top and bottom edges)
    for t in 0..thickness {
        let yt = y1.saturating_add(t).min(y2);
        let yb = y2.saturating_sub(t).max(y1);
        for x in x1..=x2 {
            img.put_pixel(x, yt, color);
            img.put_pixel(x, yb, color);
        }
    }
    // Draw vertical lines (left and right edges)
    for t in 0..thickness {
        let xl = x1.saturating_add(t).min(x2);
        let xr = x2.saturating_sub(t).max(x1);
        for y in y1..=y2 {
            img.put_pixel(xl, y, color);
            img.put_pixel(xr, y, color);
        }
    }
}

/// Draw a colored bounding box on an RGBA image with a specific color.
fn draw_box_rgba(
    img: &mut image::RgbaImage,
    bbox_px: [f32; 4],
    color: image::Rgba<u8>,
    thickness: u32,
) {
    let (iw, ih) = (img.width(), img.height());
    let x1 = (bbox_px[0] as u32).min(iw.saturating_sub(1));
    let y1 = (bbox_px[1] as u32).min(ih.saturating_sub(1));
    let x2 = (bbox_px[2] as u32).min(iw.saturating_sub(1));
    let y2 = (bbox_px[3] as u32).min(ih.saturating_sub(1));

    for t in 0..thickness {
        let yt = y1.saturating_add(t).min(y2);
        let yb = y2.saturating_sub(t).max(y1);
        for x in x1..=x2 {
            img.put_pixel(x, yt, color);
            img.put_pixel(x, yb, color);
        }
    }
    for t in 0..thickness {
        let xl = x1.saturating_add(t).min(x2);
        let xr = x2.saturating_sub(t).max(x1);
        for y in y1..=y2 {
            img.put_pixel(xl, y, color);
            img.put_pixel(xr, y, color);
        }
    }
}

/// Write table crop images with cell bbox overlays to `layout/tables/`.
///
/// For each table with a `TablePrediction`, draws cell bounding boxes on the
/// cropped table image and saves it as a PNG.
pub fn write_table_debug(
    output_dir: &Path,
    table_entries: &[(usize, DynamicImage)],
    table_results: &std::collections::HashMap<usize, crate::pipeline::TableResult>,
    page_idx: u32,
) -> Result<(), ExtractError> {
    use crate::pipeline::TableResult;

    let tables_dir = output_dir.join("layout").join("tables");

    let mut any_written = false;
    for (idx, crop_image) in table_entries {
        let pred = match table_results.get(idx) {
            Some(TableResult::Prediction(p)) => p,
            _ => continue,
        };

        if pred.cell_bboxes.is_empty() {
            continue;
        }

        if !any_written {
            std::fs::create_dir_all(&tables_dir)?;
            any_written = true;
        }

        let mut img = crop_image.to_rgba8();
        let (w, h) = (img.width() as f32, img.height() as f32);

        // Alternate colors for adjacent cells
        let colors = [
            image::Rgba([0u8, 200, 0, 180]),   // green
            image::Rgba([50, 100, 255, 180]),   // blue
            image::Rgba([255, 140, 0, 180]),    // orange
            image::Rgba([200, 0, 200, 180]),    // magenta
        ];

        for (cell_idx, bbox_opt) in pred.cell_bboxes.iter().enumerate() {
            let Some(bbox) = bbox_opt else { continue };
            let px = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h];
            let color = colors[cell_idx % colors.len()];
            draw_box_rgba(&mut img, px, color, 2);
        }

        let path = tables_dir.join(format!("p{}_{}.png", page_idx + 1, idx));
        img.save(&path)
            .map_err(|e| ExtractError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
    }

    Ok(())
}

/// Map a RegionKind to a debug visualization color (pdfium).
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
        RegionKind::InlineFormula => PdfColor::new(255, 100, 255, 255),
        RegionKind::Image | RegionKind::Chart | RegionKind::Seal => PdfColor::new(255, 140, 0, 255),
        RegionKind::FigureTitle
        | RegionKind::TableTitle
        | RegionKind::ChartTitle
        | RegionKind::FigureTableTitle => PdfColor::new(220, 200, 0, 255),
        RegionKind::PageHeader | RegionKind::PageFooter | RegionKind::PageNumber => {
            PdfColor::new(150, 150, 150, 255)
        }
        RegionKind::Algorithm => PdfColor::new(0, 160, 120, 255),
        RegionKind::FigureGroup => PdfColor::new(0, 200, 200, 255),
    }
}

/// Write debug visualization: annotate the original PDF with colored bounding boxes
/// and labels, then save as PNGs and/or a debug PDF under `layout/`.
pub fn write_debug(
    pdfium: &Pdfium,
    pdf_path: &Path,
    pages: &[Page],
    output_dir: &Path,
    mode: DebugMode,
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

    if mode == DebugMode::Images {
        // Render annotated pages to PNGs
        let layout_dir = output_dir.join("layout");
        std::fs::create_dir_all(&layout_dir)?;

        for page_info in pages {
            let page_idx = page_info.page - 1;
            let page = doc.pages().get(page_idx as u16).map_err(|e| {
                ExtractError::Pdf(format!("Failed to get page {page_idx} for render: {e}"))
            })?;
            let img = pdf::render_page(&page, page_info.dpi)?;
            let out_path = layout_dir.join(format!("p{}.png", page_info.page));
            img.save(&out_path)?;
        }
    } else {
        // Remove un-processed pages so the debug PDF only contains annotated pages.
        let keep: std::collections::HashSet<u16> =
            pages.iter().map(|p| (p.page - 1) as u16).collect();
        let total = doc.pages().len();

        for i in (0..total).rev() {
            if !keep.contains(&i) {
                doc.pages()
                    .get(i)
                    .map_err(|e| ExtractError::Pdf(format!("Failed to get page {i} for removal: {e}")))?
                    .delete()
                    .map_err(|e| ExtractError::Pdf(format!("Failed to delete page {i}: {e}")))?;
            }
        }

        let layout_dir = output_dir.join("layout");
        std::fs::create_dir_all(&layout_dir)?;
        let debug_pdf_path = layout_dir.join("layout.pdf");
        doc.save_to_file(&debug_pdf_path)
            .map_err(|e| ExtractError::Pdf(format!("Failed to save debug PDF: {e}")))?;
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
            tag: None,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
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
    fn test_formula_strips_double_dollar() {
        let mut r = make_region(RegionKind::DisplayFormula);
        r.latex = Some("$$\nx^2 + y^2\n$$".into());
        assert_eq!(region_to_markdown(&r), "$$x^2 + y^2$$");
    }

    #[test]
    fn test_formula_strips_single_dollar() {
        let mut r = make_region(RegionKind::InlineFormula);
        r.latex = Some("$x$".into());
        assert_eq!(region_to_markdown(&r), "$x$");
    }

    #[test]
    fn test_image_to_markdown() {
        let mut r = make_region(RegionKind::Image);
        r.image_path = Some("images/p1_7.png".into());
        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Figure 1".into());
        r.caption = Some(Box::new(cap));
        assert_eq!(
            region_to_markdown(&r),
            "![](images/p1_7.png)\n\nFigure 1"
        );
    }

    #[test]
    fn test_image_no_caption_to_markdown() {
        let mut r = make_region(RegionKind::Image);
        r.image_path = Some("images/p1_7.png".into());
        assert_eq!(region_to_markdown(&r), "![](images/p1_7.png)");
    }

    #[test]
    fn test_algorithm_to_markdown() {
        let mut r = make_region(RegionKind::Algorithm);
        r.text = Some("for i in range(n):".into());
        assert_eq!(region_to_markdown(&r), "for i in range(n):");
    }

    #[test]
    fn test_bold_caption_label() {
        assert_eq!(
            bold_caption_label("Fig. 1. Example simulation results"),
            "**Fig. 1.** Example simulation results"
        );
        assert_eq!(
            bold_caption_label("Table 1. Performance results"),
            "**Table 1.** Performance results"
        );
        assert_eq!(
            bold_caption_label("Figure 12. Some caption"),
            "**Figure 12.** Some caption"
        );
        assert_eq!(
            bold_caption_label("Algorithm 1. Pseudocode"),
            "**Algorithm 1.** Pseudocode"
        );
        // No recognized prefix — pass through unchanged
        assert_eq!(
            bold_caption_label("(a) Subfigure label"),
            "(a) Subfigure label"
        );
    }

    #[test]
    fn test_image_with_fig_caption() {
        let mut r = make_region(RegionKind::Image);
        r.image_path = Some("images/p1_5.png".into());
        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Fig. 1. Example simulation results".into());
        r.caption = Some(Box::new(cap));
        assert_eq!(
            region_to_markdown(&r),
            "![](images/p1_5.png)\n\n**Fig. 1.** Example simulation results"
        );
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

    #[test]
    fn test_figure_group_to_markdown() {
        let mut group = make_region(RegionKind::FigureGroup);
        group.image_path = Some("images/p1_0_grp.png".into());

        let item1 = make_region(RegionKind::Image);
        let item2 = make_region(RegionKind::Image);
        group.items = Some(vec![item1, item2]);

        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Fig. 5. Two panels".into());
        group.caption = Some(Box::new(cap));

        let md = region_to_markdown(&group);
        // Single composite image, not individual member images
        assert!(md.contains("![](images/p1_0_grp.png)"));
        assert!(md.contains("**Fig. 5.** Two panels"));
    }

    #[test]
    fn test_figure_group_no_caption() {
        let mut group = make_region(RegionKind::FigureGroup);
        group.image_path = Some("images/p1_0_grp.png".into());

        let item1 = make_region(RegionKind::Image);
        let item2 = make_region(RegionKind::Image);
        group.items = Some(vec![item1, item2]);

        let md = region_to_markdown(&group);
        // Single composite image
        assert!(md.contains("![](images/p1_0_grp.png)"));
        // No caption text
        assert!(!md.contains("Fig"));
    }

    #[test]
    fn test_figure_group_empty_items() {
        let mut group = make_region(RegionKind::FigureGroup);
        group.items = Some(vec![]);
        let mut cap = make_region(RegionKind::FigureTitle);
        cap.text = Some("Fig. 1. Empty group".into());
        group.caption = Some(Box::new(cap));

        let md = region_to_markdown(&group);
        assert!(md.contains("**Fig. 1.** Empty group"));
    }

    #[test]
    fn test_figure_group_consumed_skipped_in_markdown() {
        // FigureGroup renders a single composite image; consumed originals are skipped
        let result = ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 1,
                extraction_time_ms: 0,
            },
            pages: vec![Page {
                page: 1,
                width_pt: 612.0,
                height_pt: 792.0,
                dpi: 144,
                regions: vec![
                    {
                        // Consumed original member (no image_path in new design)
                        let mut r = make_region(RegionKind::Image);
                        r.consumed = true;
                        r.order = 0;
                        r
                    },
                    {
                        // The group with composite image
                        let mut group = make_region(RegionKind::FigureGroup);
                        group.order = 0;
                        group.image_path = Some("images/p1_0.png".into());
                        let item = make_region(RegionKind::Image);
                        group.items = Some(vec![item]);
                        group
                    },
                ],
            }],
        };

        let md = render_markdown(&result);
        // The composite image should appear exactly once
        let count = md.matches("![](images/p1_0.png)").count();
        assert_eq!(count, 1);
    }

    /// Helper to build an ExtractionResult from a list of regions on one page.
    fn make_result(regions: Vec<Region>) -> ExtractionResult {
        ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 1,
                extraction_time_ms: 0,
            },
            pages: vec![Page {
                page: 1,
                width_pt: 612.0,
                height_pt: 792.0,
                dpi: 144,
                regions,
            }],
        }
    }

    #[test]
    fn test_mid_sentence_merge_adjacent_text_sections() {
        // Two adjacent text sections split at a column break
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("method presented superior".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("convergence behavior over prior techniques.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("presented superior convergence behavior"),
            "adjacent text sections should merge mid-sentence, got: {md}"
        );
    }

    #[test]
    fn test_mid_sentence_merge_across_figure() {
        // Text → Figure → Text where the text is split mid-sentence by a figure
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("coloring the nodes of the".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p1_1.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("dual graph, which has more nodes.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(md.contains("![](images/p1_1.png)"), "figure should be present");
        // The trailing fragment moves to the next text section
        assert!(
            md.contains("coloring the nodes of the dual graph"),
            "text should rejoin across figure, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_at_sentence_boundary() {
        // Two text sections where the first ends with a period — no merge
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("This is a complete sentence.".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("This starts a new paragraph.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("sentence.\n\nThis starts"),
            "should have paragraph break, got: {md}"
        );
    }

    #[test]
    fn test_merge_when_prev_ends_mid_sentence_next_starts_uppercase() {
        // Mid-sentence ending and next starts uppercase — still merges
        // because the previous text clearly ended mid-sentence.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("some text ending with comma,".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("Another part of the same paragraph.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("comma, Another part"),
            "should merge when prev ends mid-sentence, got: {md}"
        );
    }

    #[test]
    fn test_references_merge_mid_sentence() {
        // References split across pages where entry is cut mid-sentence
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
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("ACM Transactions on".into());
                        r
                    }],
                },
                Page {
                    page: 2,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("Graphics 31, 5 (Aug. 2012).".into());
                        r
                    }],
                },
            ],
        };
        let md = render_markdown(&result);
        assert!(
            md.contains("Transactions on Graphics"),
            "should join with space across pages, got: {md}"
        );
    }

    #[test]
    fn test_references_merge_at_entry_boundary() {
        // References across pages where first page ends at entry boundary
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
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("First reference entry.".into());
                        r
                    }],
                },
                Page {
                    page: 2,
                    width_pt: 612.0,
                    height_pt: 792.0,
                    dpi: 144,
                    regions: vec![{
                        let mut r = make_region(RegionKind::References);
                        r.text = Some("Second reference entry.".into());
                        r
                    }],
                },
            ],
        };
        let md = render_markdown(&result);
        assert!(
            md.contains("entry.\n\nSecond"),
            "should have paragraph break between complete entries, got: {md}"
        );
    }

    #[test]
    fn test_mid_sentence_merge_across_algorithm() {
        // Text → Algorithm → Text: text split mid-sentence by an algorithm box.
        // The trailing sentence fragment moves to the next text section.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("As expected, VBD".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Algorithm);
                r.text = Some("Algorithm 1: VBD simulation".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("converges slower for stiffer materials.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // The text should be rejoined in the continuation section
        assert!(
            md.contains("As expected, VBD converges slower"),
            "sentence should rejoin across algorithm, got: {md}"
        );
        // Algorithm still appears as its own section
        assert!(
            md.contains("Algorithm 1"),
            "algorithm should be present, got: {md}"
        );
    }

    #[test]
    fn test_mid_sentence_partial_move_across_figure() {
        // Only the incomplete trailing sentence moves, not the whole section.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("First sentence ends here. Second part of the".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p1_1.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("paragraph continues.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // "First sentence ends here." stays in the first section
        assert!(
            md.contains("ends here."),
            "complete sentence should remain, got: {md}"
        );
        // "Second part of the" moves to join "paragraph continues."
        assert!(
            md.contains("Second part of the paragraph continues."),
            "incomplete sentence should rejoin, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_across_display_formula() {
        // Text → DisplayFormula → Text: formulas are part of the paragraph
        // flow, so text should NOT be rearranged across them.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("VBD avoids this by minimizing the variational energy [Simo et al.".into());
                r
            },
            {
                let mut r = make_region(RegionKind::DisplayFormula);
                r.latex = Some("x = argmin E(x)".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("1992] where the timestep size is used.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // Text should NOT be rearranged — formula is inline to the paragraph
        assert!(
            md.contains("[Simo et al.\n\n$$x = argmin E(x)$$\n\n1992]"),
            "text should stay in place around formula, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_authors_into_abstract() {
        // Title → author Text blocks → Image → Abstract: the author lines
        // end without terminal punctuation ("USA"), but should NOT merge
        // with the abstract that follows.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Title);
                r.text = Some("Augmented Vertex Block Descent".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CHRIS GILES, Roblox, USA".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CEM YUKSEL, University of Utah, USA".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p1_4.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Abstract);
                r.text = Some("Vertex Block Descent is a fast physics-based simulation method.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // Authors should remain separate from the abstract
        assert!(
            md.contains("University of Utah, USA\n\n"),
            "authors should not merge into abstract, got: {md}"
        );
        assert!(
            md.contains("Vertex Block Descent is a fast"),
            "abstract should be present, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_across_title_boundary() {
        // Text → Title → Text: a Title between text blocks should prevent
        // backward merge even when the first text ends mid-sentence.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("some text ending mid".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Title);
                r.text = Some("Section 2".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("sentence that continues here.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        // Should NOT merge across the title
        assert!(
            md.contains("some text ending mid\n\n# Section 2"),
            "text before title should stay separate, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_text_into_abstract_even_when_text_kind() {
        // When the abstract is detected as RegionKind::Abstract,
        // it should never absorb the previous text block.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("Author Name, Institution".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Abstract);
                r.text = Some("We present a novel approach to solving.".into());
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("Author Name, Institution\n\nWe present"),
            "abstract should not merge with preceding text, got: {md}"
        );
    }

    #[test]
    fn test_no_merge_short_line_author_blocks() {
        // Three narrow author lines (< 50% of 612pt page) should NOT be merged
        // even though they end without terminal punctuation.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Title);
                r.text = Some("Augmented Vertex Block Descent".into());
                r.bbox = [49.0, 76.0, 289.0, 95.0]; // ~240pt wide
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CHRIS GILES, Roblox, USA".into());
                r.bbox = [49.0, 104.0, 178.0, 117.0]; // ~129pt wide (21% of 612)
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("ELIE DIAZ, University of Utah, USA".into());
                r.bbox = [49.0, 118.0, 211.0, 132.0]; // ~162pt wide (26% of 612)
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("CEM YUKSEL, University of Utah, USA".into());
                r.bbox = [49.0, 132.0, 228.0, 146.0]; // ~179pt wide (29% of 612)
                r
            },
        ]);
        let md = render_markdown(&result);
        // Each author should be a separate paragraph, not merged together
        assert!(
            md.contains("CHRIS GILES, Roblox, USA\n\nELIE DIAZ"),
            "author lines should stay separate, got: {md}"
        );
        assert!(
            md.contains("ELIE DIAZ, University of Utah, USA\n\nCEM YUKSEL"),
            "author lines should stay separate, got: {md}"
        );
    }

    #[test]
    fn test_wide_text_blocks_still_merge() {
        // Two wide text blocks (> 50% of page) that end mid-sentence should
        // still merge — these are paragraph splits, not metadata lines.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("method presented superior".into());
                r.bbox = [49.0, 300.0, 560.0, 320.0]; // ~511pt wide (83% of 612)
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("convergence behavior over prior techniques.".into());
                r.bbox = [49.0, 400.0, 560.0, 420.0]; // ~511pt wide
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("presented superior convergence behavior"),
            "wide text blocks should still merge mid-sentence, got: {md}"
        );
    }

    #[test]
    fn test_merge_across_subfigure_labels() {
        // Paragraph split across multiple figures with short sub-figure labels.
        // The backward search should skip over the short labels to find the
        // real paragraph text and merge the two halves.
        let result = make_result(vec![
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("Figure 6 shows a".into());
                r.bbox = [49.0, 300.0, 300.0, 400.0]; // column-width paragraph
                r
            },
            {
                // Sub-figure label: short + narrow → is_short_line
                let mut r = make_region(RegionKind::Text);
                r.text = Some("(a) Reference".into());
                r.bbox = [49.0, 410.0, 140.0, 425.0]; // narrow, 13 chars
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p8_5.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("(b) AVBD, 5 iter.".into());
                r.bbox = [49.0, 500.0, 160.0, 515.0]; // narrow, 17 chars
                r
            },
            {
                let mut r = make_region(RegionKind::Image);
                r.image_path = Some("images/p8_9.png".into());
                r
            },
            {
                let mut r = make_region(RegionKind::Text);
                r.text = Some("delicate card tower with very lightweight bodies.".into());
                r.bbox = [49.0, 600.0, 300.0, 700.0]; // column-width paragraph
                r
            },
        ]);
        let md = render_markdown(&result);
        assert!(
            md.contains("shows a delicate card tower"),
            "paragraph should merge across sub-figure labels, got: {md}"
        );
    }
}
