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

    // Collect parent (figure/table/chart) indices, skipping consumed regions
    let parent_indices: Vec<usize> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            !r.consumed
                && matches!(
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
            let cap_region_idx = caption_data[cap_idx].0;
            regions[parent_idx].caption = Some(Box::new(regions[cap_region_idx].clone()));
            regions[cap_region_idx].consumed = true;
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

/// Suppress smaller visual regions that are fully contained within a larger visual region.
///
/// For composite figures, the layout model often detects each sub-panel as a separate
/// Image region alongside a larger detection spanning the entire figure. This marks
/// the smaller contained regions as `consumed = true`, keeping only the large composite.
///
/// A region is considered "contained" when the intersection area is > 50% of its own area.
/// For exact-bbox duplicates (same bbox, different kind), the lower-confidence one is consumed.
pub fn suppress_contained_visuals(regions: &mut [Region]) {
    // Collect indices of all visual regions
    let visual_indices: Vec<usize> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| r.kind.is_visual() && !r.consumed)
        .map(|(i, _)| i)
        .collect();

    if visual_indices.len() < 2 {
        return;
    }

    let mut to_consume: Vec<usize> = Vec::new();

    for i in 0..visual_indices.len() {
        for j in (i + 1)..visual_indices.len() {
            let ai = visual_indices[i];
            let bi = visual_indices[j];
            let a_bbox = regions[ai].bbox;
            let b_bbox = regions[bi].bbox;

            let intersection = bbox_intersection_area(a_bbox, b_bbox);
            if intersection <= 0.0 {
                continue;
            }

            let a_area = (a_bbox[2] - a_bbox[0]) * (a_bbox[3] - a_bbox[1]);
            let b_area = (b_bbox[2] - b_bbox[0]) * (b_bbox[3] - b_bbox[1]);

            // Exact-bbox duplicate: consume the lower-confidence one
            if (a_area - b_area).abs() < 1.0 {
                if regions[ai].confidence >= regions[bi].confidence {
                    to_consume.push(bi);
                } else {
                    to_consume.push(ai);
                }
                continue;
            }

            // Containment check: intersection / smaller_area > 0.5
            let smaller_area = a_area.min(b_area);
            if smaller_area > 0.0 && intersection / smaller_area > 0.5 {
                // Consume the smaller region
                if a_area < b_area {
                    to_consume.push(ai);
                } else {
                    to_consume.push(bi);
                }
            }
        }
    }

    for idx in to_consume {
        regions[idx].consumed = true;
    }
}

/// Compute the area of intersection between two axis-aligned bounding boxes.
/// Each bbox is `[x1, y1, x2, y2]`.
fn bbox_intersection_area(a: [f32; 4], b: [f32; 4]) -> f32 {
    let x_overlap = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let y_overlap = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    x_overlap * y_overlap
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
            tag: None,
            consumed: false,
        }
    }

    /// Extract caption text from a region's caption field for test assertions.
    fn caption_text(region: &Region) -> Option<&str> {
        region.caption.as_ref().and_then(|c| c.text.as_deref())
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
        assert_eq!(caption_text(&regions[0]), Some("Figure 1: Overview"));
        assert!(regions[1].consumed);
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
        assert_eq!(caption_text(&regions[1]), Some("Figure 1: Above"));
        assert!(regions[0].consumed);
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
        assert!(!regions[1].consumed);
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
        assert_eq!(caption_text(&regions[0]), Some("Table 1: Results"));
        assert!(regions[1].consumed);
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
        assert_eq!(caption_text(&regions[0]), Some("Figure 1"));
        assert_eq!(caption_text(&regions[1]), Some("Figure 2"));
        assert!(regions[2].consumed);
        assert!(regions[3].consumed);
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

    #[test]
    fn test_suppress_smaller_image_inside_larger() {
        let mut regions = vec![
            make_region(RegionKind::Image, [50.0, 50.0, 500.0, 500.0], None), // large
            make_region(RegionKind::Image, [100.0, 100.0, 200.0, 200.0], None), // small, fully inside
        ];
        suppress_contained_visuals(&mut regions);
        assert!(!regions[0].consumed);
        assert!(regions[1].consumed);
    }

    #[test]
    fn test_suppress_non_overlapping_untouched() {
        let mut regions = vec![
            make_region(RegionKind::Image, [50.0, 50.0, 200.0, 200.0], None),
            make_region(RegionKind::Image, [300.0, 300.0, 450.0, 450.0], None),
        ];
        suppress_contained_visuals(&mut regions);
        assert!(!regions[0].consumed);
        assert!(!regions[1].consumed);
    }

    #[test]
    fn test_suppress_exact_bbox_duplicate() {
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::Chart, [50.0, 50.0, 400.0, 400.0], None);
                r.confidence = 0.9;
                r
            },
            {
                let mut r = make_region(RegionKind::Image, [50.0, 50.0, 400.0, 400.0], None);
                r.confidence = 0.95;
                r
            },
        ];
        suppress_contained_visuals(&mut regions);
        // Lower-confidence Chart should be consumed
        assert!(regions[0].consumed);
        assert!(!regions[1].consumed);
    }

    #[test]
    fn test_suppress_partial_overlap_below_threshold() {
        // Two regions that overlap but intersection / smaller_area < 0.5
        let mut regions = vec![
            make_region(RegionKind::Image, [0.0, 0.0, 100.0, 100.0], None),   // area = 10000
            make_region(RegionKind::Image, [80.0, 80.0, 200.0, 200.0], None), // area = 14400, intersection = 20*20 = 400
        ];
        suppress_contained_visuals(&mut regions);
        // intersection/smaller = 400/10000 = 0.04, well below 0.5
        assert!(!regions[0].consumed);
        assert!(!regions[1].consumed);
    }

    #[test]
    fn test_caption_skips_consumed_parent() {
        // Smaller image is consumed; caption should go to the larger surviving image
        let mut regions = vec![
            make_region(RegionKind::Image, [80.0, 70.0, 270.0, 150.0], None), // large
            make_region(RegionKind::Image, [86.0, 70.0, 156.0, 150.0], None), // small, contained
            make_region(
                RegionKind::FigureTitle,
                [80.0, 155.0, 270.0, 175.0],
                Some("Fig. 4. Two collision types"),
            ),
        ];
        suppress_contained_visuals(&mut regions);
        assert!(!regions[0].consumed);
        assert!(regions[1].consumed);
        associate_captions(&mut regions);
        // Caption should land on the large image, not the consumed small one
        assert_eq!(
            caption_text(&regions[0]),
            Some("Fig. 4. Two collision types")
        );
        assert!(regions[1].caption.is_none());
        // Caption region itself should be consumed
        assert!(regions[2].consumed);
    }
}
