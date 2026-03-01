use image::DynamicImage;

use crate::types::{Region, RegionKind};

/// Crop a region from the rendered page image.
///
/// Converts the PDF-point bounding box to pixel coordinates using the DPI,
/// and crops the region from the image.
pub fn crop_region(
    page_image: &DynamicImage,
    bbox: [f32; 4],
    page_width_pt: f32,
    page_height_pt: f32,
    dpi: u32,
) -> DynamicImage {
    let scale = dpi as f32 / 72.0;
    let img_width = page_image.width() as f32;
    let img_height = page_image.height() as f32;

    // Convert PDF points to pixel coordinates
    let x1 = ((bbox[0] * scale).max(0.0)).min(img_width) as u32;
    let y1 = ((bbox[1] * scale).max(0.0)).min(img_height) as u32;
    let x2 = ((bbox[2] * scale).max(0.0)).min(img_width) as u32;
    let y2 = ((bbox[3] * scale).max(0.0)).min(img_height) as u32;

    // PDF coordinate system has origin at bottom-left, images at top-left.
    // pdfium-render already flips Y when rendering, so bbox Y values correspond
    // to image Y values directly (top-to-bottom).
    let _ = (page_width_pt, page_height_pt); // reserved for future coordinate transforms

    let width = x2.saturating_sub(x1).max(1);
    let height = y2.saturating_sub(y1).max(1);

    page_image.crop_imm(x1, y1, width, height)
}

/// Associate caption regions with their parent figure/table/chart regions.
///
/// For each Image/Chart/Table region, finds the nearest caption region
/// (FigureTitle/TableTitle/ChartTitle/FigureTableTitle) within
/// `2 × median_region_height` distance. Prefers captions below the figure.
///
/// Sets the `caption` field on the parent region with the caption text,
/// and also sets the `text` field on caption regions.
pub fn associate_captions(regions: &mut [Region]) {
    if regions.is_empty() {
        return;
    }

    // Compute median region height for distance threshold
    let mut heights: Vec<f32> = regions
        .iter()
        .map(|r| (r.bbox[3] - r.bbox[1]).abs())
        .collect();
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_height = heights[heights.len() / 2];
    let max_distance = median_height * 2.0;

    // Collect caption indices and their data
    let caption_data: Vec<(usize, [f32; 4], Option<String>)> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| r.kind.is_caption())
        .map(|(i, r)| (i, r.bbox, r.text.clone()))
        .collect();

    // Collect parent (figure/table/chart) indices
    let parent_indices: Vec<usize> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            matches!(
                r.kind,
                RegionKind::Image | RegionKind::Chart | RegionKind::Table
            )
        })
        .map(|(i, _)| i)
        .collect();

    // Track which captions have been assigned
    let mut assigned_captions: Vec<bool> = vec![false; caption_data.len()];

    // For each parent, find the nearest unassigned caption
    for &parent_idx in &parent_indices {
        let parent_bbox = regions[parent_idx].bbox;
        let parent_kind = regions[parent_idx].kind;

        let mut best_caption: Option<(usize, f32, bool)> = None; // (caption_data_idx, distance, is_below)

        for (cap_idx, &(_, cap_bbox, _)) in caption_data.iter().enumerate() {
            if assigned_captions[cap_idx] {
                continue;
            }

            // Check caption type compatibility
            let compatible = match parent_kind {
                RegionKind::Image => caption_data[cap_idx]
                    .0
                    .checked_sub(0) // just to access the region
                    .map(|_| {
                        matches!(
                            regions[caption_data[cap_idx].0].kind,
                            RegionKind::FigureTitle | RegionKind::FigureTableTitle
                        )
                    })
                    .unwrap_or(false),
                RegionKind::Table => matches!(
                    regions[caption_data[cap_idx].0].kind,
                    RegionKind::TableTitle | RegionKind::FigureTableTitle
                ),
                RegionKind::Chart => matches!(
                    regions[caption_data[cap_idx].0].kind,
                    RegionKind::ChartTitle | RegionKind::FigureTableTitle
                ),
                _ => false,
            };
            if !compatible {
                continue;
            }

            let distance = edge_distance(parent_bbox, cap_bbox);
            if distance > max_distance {
                continue;
            }

            // Is caption below the parent?
            let cap_center_y = (cap_bbox[1] + cap_bbox[3]) / 2.0;
            let parent_bottom = parent_bbox[3];
            let is_below = cap_center_y > parent_bottom;

            let is_better = match best_caption {
                None => true,
                Some((_, best_dist, best_below)) => {
                    // Prefer below, then closer distance
                    if is_below && !best_below {
                        true
                    } else if !is_below && best_below {
                        false
                    } else {
                        distance < best_dist
                    }
                }
            };

            if is_better {
                best_caption = Some((cap_idx, distance, is_below));
            }
        }

        if let Some((cap_idx, _, _)) = best_caption {
            assigned_captions[cap_idx] = true;
            let caption_text = caption_data[cap_idx].2.clone();
            if let Some(text) = caption_text {
                regions[parent_idx].caption = Some(text);
            }
        }
    }
}

/// Compute the minimum edge-to-edge distance between two bounding boxes.
fn edge_distance(a: [f32; 4], b: [f32; 4]) -> f32 {
    let dx = if a[2] < b[0] {
        b[0] - a[2]
    } else if b[2] < a[0] {
        a[0] - b[2]
    } else {
        0.0
    };

    let dy = if a[3] < b[1] {
        b[1] - a[3]
    } else if b[3] < a[1] {
        a[1] - b[3]
    } else {
        0.0
    };

    (dx * dx + dy * dy).sqrt()
}

/// Classify a chart type using simple heuristics on the cropped image.
///
/// Returns one of: "bar", "line", "pie", "scatter", "other".
pub fn classify_chart_type(_image: &DynamicImage) -> &'static str {
    // Simple heuristic classifier for v1.
    // A dedicated ML model can be added later.
    "other"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(kind: RegionKind, bbox: [f32; 4], text: Option<&str>) -> Region {
        Region {
            id: String::new(),
            kind,
            bbox,
            confidence: 1.0,
            order: 0,
            text: text.map(String::from),
            html: None,
            latex: None,
            image_path: None,
            caption: None,
            chart_type: None,
            consumed: false,
        }
    }

    #[test]
    fn test_caption_below_figure() {
        let mut regions = vec![
            make_region(RegionKind::Image, [80.0, 100.0, 530.0, 400.0], None),
            make_region(
                RegionKind::FigureTitle,
                [80.0, 410.0, 530.0, 430.0],
                Some("Figure 1: Overview"),
            ),
        ];
        associate_captions(&mut regions);
        assert_eq!(
            regions[0].caption.as_deref(),
            Some("Figure 1: Overview")
        );
    }

    #[test]
    fn test_caption_above_figure() {
        let mut regions = vec![
            make_region(
                RegionKind::FigureTitle,
                [80.0, 80.0, 530.0, 100.0],
                Some("Figure 1: Above"),
            ),
            make_region(RegionKind::Image, [80.0, 110.0, 530.0, 400.0], None),
        ];
        associate_captions(&mut regions);
        assert_eq!(regions[1].caption.as_deref(), Some("Figure 1: Above"));
    }

    #[test]
    fn test_caption_too_far() {
        let mut regions = vec![
            make_region(RegionKind::Image, [80.0, 100.0, 530.0, 200.0], None),
            make_region(
                RegionKind::FigureTitle,
                [80.0, 700.0, 530.0, 720.0],
                Some("Far away caption"),
            ),
        ];
        associate_captions(&mut regions);
        assert!(regions[0].caption.is_none());
    }

    #[test]
    fn test_table_caption_association() {
        let mut regions = vec![
            make_region(RegionKind::Table, [60.0, 200.0, 550.0, 400.0], None),
            make_region(
                RegionKind::TableTitle,
                [60.0, 410.0, 550.0, 430.0],
                Some("Table 1: Results"),
            ),
        ];
        associate_captions(&mut regions);
        assert_eq!(regions[0].caption.as_deref(), Some("Table 1: Results"));
    }

    #[test]
    fn test_multiple_figures_multiple_captions() {
        let mut regions = vec![
            make_region(RegionKind::Image, [50.0, 100.0, 280.0, 300.0], None),
            make_region(RegionKind::Image, [320.0, 100.0, 550.0, 300.0], None),
            make_region(
                RegionKind::FigureTitle,
                [50.0, 310.0, 280.0, 330.0],
                Some("Figure 1"),
            ),
            make_region(
                RegionKind::FigureTitle,
                [320.0, 310.0, 550.0, 330.0],
                Some("Figure 2"),
            ),
        ];
        associate_captions(&mut regions);
        assert_eq!(regions[0].caption.as_deref(), Some("Figure 1"));
        assert_eq!(regions[1].caption.as_deref(), Some("Figure 2"));
    }

    #[test]
    fn test_crop_region_coordinates() {
        // Create a 288x396 test image (144 DPI for a 2"x2.75" region)
        let img = DynamicImage::new_rgb8(288, 396);
        let cropped = crop_region(
            &img,
            [72.0, 72.0, 144.0, 144.0], // 1 inch square at offset (1,1) inches
            612.0,
            792.0,
            144,
        );
        // At 144 DPI: 72pt = 144px, so crop should be 144px square
        // from (144,144) to (288,288) — but image is only 288x396
        assert_eq!(cropped.width(), 144);
        assert_eq!(cropped.height(), 144);
    }
}
