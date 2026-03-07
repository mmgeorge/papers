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

            // Caption–parent compatibility.
            //
            // When the caption text has a clear prefix ("Table …" / "Fig …"),
            // that is the authoritative signal — it overrides the model's kind
            // label, which is often wrong (e.g. FigureTitle on "Table 1").
            // Only when there is no recognisable prefix do we fall back to
            // kind-based matching.
            let cap_region_kind = regions[caption_data[cap_idx].0].kind;
            let cap_text_lower = caption_data[cap_idx]
                .2
                .as_deref()
                .unwrap_or("")
                .to_lowercase();

            let compatible = if cap_text_lower.starts_with("table") {
                parent_kind == RegionKind::Table
            } else if cap_text_lower.starts_with("fig") {
                matches!(parent_kind, RegionKind::Image | RegionKind::Chart)
            } else {
                // No clear text prefix — use kind-based matching
                match parent_kind {
                    RegionKind::Image => matches!(
                        cap_region_kind,
                        RegionKind::FigureTitle | RegionKind::FigureTableTitle
                    ),
                    RegionKind::Table => matches!(
                        cap_region_kind,
                        RegionKind::TableTitle | RegionKind::FigureTableTitle
                    ),
                    RegionKind::Chart => matches!(
                        cap_region_kind,
                        RegionKind::ChartTitle | RegionKind::FigureTableTitle
                    ),
                    _ => false,
                }
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
pub(crate) fn edge_distance(a: [f32; 4], b: [f32; 4]) -> f32 {
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

/// Maximum edge-to-edge gap (in PDF points) for grouping visual regions.
const FIGURE_GROUP_MAX_GAP: f32 = 30.0;

/// Group spatially close visual regions into `FigureGroup` containers.
///
/// Uses connected-component clustering: two groupable regions are connected
/// if their edge-to-edge distance is less than [`FIGURE_GROUP_MAX_GAP`].
/// Each component with 2+ members becomes a FigureGroup.
///
/// Should be called **after** `associate_captions` so that individual items
/// already have their captions attached. The function then promotes a
/// suitable caption (starting with "Fig" or "Table") to the group level
/// and absorbs nearby orphaned captions.
pub fn group_figure_regions(regions: &mut Vec<Region>) {
    // Collect indices of non-consumed groupable regions
    let groupable: Vec<usize> = regions
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.consumed && r.kind.is_groupable())
        .map(|(i, _)| i)
        .collect();

    if groupable.len() < 2 {
        return;
    }

    // Union-Find
    let n = groupable.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    // Build adjacency via pairwise distance
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = edge_distance(regions[groupable[i]].bbox, regions[groupable[j]].bbox);
            if dist < FIGURE_GROUP_MAX_GAP {
                union(&mut parent, i, j);
            }
        }
    }

    // Collect components
    let mut components: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(groupable[i]);
    }

    // Compute median height for orphaned caption absorption threshold
    let median_height = {
        let mut heights: Vec<f32> = regions
            .iter()
            .map(|r| (r.bbox[3] - r.bbox[1]).abs())
            .collect();
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if heights.is_empty() {
            20.0
        } else {
            heights[heights.len() / 2]
        }
    };
    let caption_max_dist = median_height * 2.0;

    let mut new_groups: Vec<Region> = Vec::new();

    for (_root, member_indices) in &components {
        if member_indices.len() < 2 {
            continue;
        }

        // Clone members for the group's items vec
        let mut items: Vec<Region> = member_indices
            .iter()
            .map(|&i| regions[i].clone())
            .collect();

        // Compute union bbox
        let mut union_bbox = items[0].bbox;
        let mut max_conf = items[0].confidence;
        let mut min_order = items[0].order;
        for item in &items[1..] {
            union_bbox[0] = union_bbox[0].min(item.bbox[0]);
            union_bbox[1] = union_bbox[1].min(item.bbox[1]);
            union_bbox[2] = union_bbox[2].max(item.bbox[2]);
            union_bbox[3] = union_bbox[3].max(item.bbox[3]);
            max_conf = max_conf.max(item.confidence);
            min_order = min_order.min(item.order);
        }

        // Caption promotion: find an item whose caption starts with "Fig" or "Table"
        let mut group_caption: Option<Box<Region>> = None;

        // First pass: look for an item whose caption starts with a main label
        let promoted_idx = items.iter().position(|item| {
            item.caption
                .as_ref()
                .and_then(|c| c.text.as_deref())
                .map(|t| {
                    let lower = t.to_lowercase();
                    lower.starts_with("fig") || lower.starts_with("table")
                })
                .unwrap_or(false)
        });

        if let Some(idx) = promoted_idx {
            group_caption = items[idx].caption.take();
        } else {
            // Fallback: if exactly 1 item has a caption, promote it
            let with_caption: Vec<usize> = items
                .iter()
                .enumerate()
                .filter(|(_, item)| item.caption.is_some())
                .map(|(i, _)| i)
                .collect();
            if with_caption.len() == 1 {
                group_caption = items[with_caption[0]].caption.take();
            }
        }

        // Orphaned caption absorption: scan nearby non-consumed caption regions
        if group_caption.is_none() {
            let mut best_orphan: Option<(usize, f32)> = None;
            for (ri, r) in regions.iter().enumerate() {
                if r.consumed || !r.kind.is_caption() {
                    continue;
                }
                // Skip captions that are members of the group
                if member_indices.contains(&ri) {
                    continue;
                }
                let dist = edge_distance(union_bbox, r.bbox);
                if dist > caption_max_dist {
                    continue;
                }
                let text_lower = r.text.as_deref().unwrap_or("").to_lowercase();
                if !text_lower.starts_with("fig") && !text_lower.starts_with("table") {
                    continue;
                }
                if best_orphan.map_or(true, |(_, d)| dist < d) {
                    best_orphan = Some((ri, dist));
                }
            }
            if let Some((orphan_idx, _)) = best_orphan {
                group_caption = Some(Box::new(regions[orphan_idx].clone()));
                regions[orphan_idx].consumed = true;
            }
        }

        // Mark original members as consumed
        for &i in member_indices {
            regions[i].consumed = true;
        }

        let first_id = &regions[member_indices[0]].id;
        let group = Region {
            id: format!("{first_id}_grp"),
            kind: RegionKind::FigureGroup,
            bbox: union_bbox,
            confidence: max_conf,
            order: min_order,
            text: None,
            html: None,
            latex: None,
            image_path: None,
            caption: group_caption,
            chart_type: None,
            tag: None,
            items: Some(items),
            consumed: false,
        };

        new_groups.push(group);
    }

    regions.append(&mut new_groups);
    regions.sort_by_key(|r| r.order);
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
            items: None,
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

    // ── FigureGroup tests ────────────────────────────────────────────

    fn make_groupable(id: &str, kind: RegionKind, bbox: [f32; 4], order: u32) -> Region {
        Region {
            id: id.into(),
            kind,
            bbox,
            confidence: 0.9,
            order,
            text: None,
            html: None,
            latex: None,
            image_path: Some(format!("images/{id}.png")),
            caption: None,
            chart_type: None,
            tag: None,
            items: None,
            consumed: false,
        }
    }

    #[test]
    fn test_group_2x3_grid() {
        // 6 visuals in a 2x3 grid, all < 30pt apart → single FigureGroup
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),  // 10pt gap
            make_groupable("p1_2", RegionKind::Chart, [270.0, 50.0, 370.0, 150.0], 2),  // 10pt gap
            make_groupable("p1_3", RegionKind::Chart, [50.0, 160.0, 150.0, 260.0], 3),  // 10pt vertical gap
            make_groupable("p1_4", RegionKind::Chart, [160.0, 160.0, 260.0, 260.0], 4),
            make_groupable("p1_5", RegionKind::Chart, [270.0, 160.0, 370.0, 260.0], 5),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].items.as_ref().unwrap().len(), 6);
        assert_eq!(groups[0].order, 0);

        // Original members should be consumed
        let consumed_count = regions.iter().filter(|r| r.consumed).count();
        assert_eq!(consumed_count, 6);
    }

    #[test]
    fn test_two_separate_clusters() {
        // Two pairs with > 30pt gap between them → two groups
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),  // 10pt gap
            make_groupable("p1_2", RegionKind::Image, [50.0, 250.0, 150.0, 350.0], 2),  // 90pt gap from row above
            make_groupable("p1_3", RegionKind::Image, [160.0, 250.0, 260.0, 350.0], 3),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 2);
        for g in &groups {
            assert_eq!(g.items.as_ref().unwrap().len(), 2);
        }
    }

    #[test]
    fn test_single_visual_no_group() {
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 0);
        assert!(!regions[0].consumed);
    }

    #[test]
    fn test_caption_promotion() {
        // Two images, one with "Fig. 5" caption → promoted to group level
        let mut regions = vec![
            {
                let mut r = make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0);
                let mut cap = make_region(RegionKind::FigureTitle, [50.0, 155.0, 150.0, 170.0], Some("Fig. 5. Results"));
                cap.consumed = true;
                r.caption = Some(Box::new(cap));
                r
            },
            {
                let mut r = make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1);
                let mut cap = make_region(RegionKind::FigureTitle, [160.0, 155.0, 260.0, 170.0], Some("(a) Detail"));
                cap.consumed = true;
                r.caption = Some(Box::new(cap));
                r
            },
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        // Group caption should be the "Fig. 5" one
        let cap_text = group.caption.as_ref().unwrap().text.as_deref().unwrap();
        assert!(cap_text.starts_with("Fig. 5"));
        // The item that had "Fig. 5" should have its caption removed
        let items = group.items.as_ref().unwrap();
        let fig5_item = items.iter().find(|i| i.id == "p1_0").unwrap();
        assert!(fig5_item.caption.is_none());
        // The "(a) Detail" sub-label should still be on its item
        let detail_item = items.iter().find(|i| i.id == "p1_1").unwrap();
        assert!(detail_item.caption.is_some());
    }

    #[test]
    fn test_orphaned_caption_absorption() {
        // Two images + orphaned caption region near them
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),
            {
                let mut cap = make_region(RegionKind::FigureTitle, [50.0, 160.0, 260.0, 180.0], Some("Fig. 7. Comparison"));
                cap.order = 2;
                cap
            },
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        let cap_text = group.caption.as_ref().unwrap().text.as_deref().unwrap();
        assert!(cap_text.starts_with("Fig. 7"));
        // The orphaned caption should be consumed
        assert!(regions.iter().any(|r| r.kind == RegionKind::FigureTitle && r.consumed));
    }

    #[test]
    fn test_grouping_ignores_text_between() {
        // Two images with a Text region between them in reading order
        // but spatially close → CC clustering still groups them
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            {
                let mut r = make_region(RegionKind::Text, [300.0, 50.0, 500.0, 150.0], Some("Some text"));
                r.order = 1;
                r
            },
            make_groupable("p1_2", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 2),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].items.as_ref().unwrap().len(), 2);
        // Text region should not be consumed
        let text_r = regions.iter().find(|r| r.kind == RegionKind::Text).unwrap();
        assert!(!text_r.consumed);
    }

    #[test]
    fn test_empty_regions_no_panic() {
        let mut regions: Vec<Region> = vec![];
        group_figure_regions(&mut regions);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_all_consumed_visuals_skipped() {
        // Pre-consumed visuals should not be grouped
        let mut regions = vec![
            {
                let mut r = make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0);
                r.consumed = true;
                r
            },
            {
                let mut r = make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1);
                r.consumed = true;
                r
            },
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_mixed_consumed_and_live() {
        // One consumed + two live close together → group of 2
        let mut regions = vec![
            {
                let mut r = make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0);
                r.consumed = true;
                r
            },
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),
            make_groupable("p1_2", RegionKind::Image, [270.0, 50.0, 370.0, 150.0], 2),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].items.as_ref().unwrap().len(), 2);
        // The pre-consumed region should still be consumed but NOT in the group
        assert!(regions[0].consumed);
    }

    #[test]
    fn test_group_union_bbox() {
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [10.0, 20.0, 100.0, 120.0], 0),
            make_groupable("p1_1", RegionKind::Image, [110.0, 30.0, 200.0, 130.0], 1),
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        assert_eq!(group.bbox[0], 10.0);  // min x1
        assert_eq!(group.bbox[1], 20.0);  // min y1
        assert_eq!(group.bbox[2], 200.0); // max x2
        assert_eq!(group.bbox[3], 130.0); // max y2
    }

    #[test]
    fn test_group_confidence_is_max() {
        let mut regions = vec![
            {
                let mut r = make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0);
                r.confidence = 0.8;
                r
            },
            {
                let mut r = make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1);
                r.confidence = 0.95;
                r
            },
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        assert!((group.confidence - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_group_order_is_min() {
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 5),
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 3),
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        assert_eq!(group.order, 3);
    }

    #[test]
    fn test_group_id_format() {
        let mut regions = vec![
            make_groupable("p1_7", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_8", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        assert!(group.id.ends_with("_grp"));
    }

    #[test]
    fn test_exactly_at_threshold_not_grouped() {
        // Two regions exactly 30pt apart should NOT be grouped (< 30, not <=)
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [180.0, 50.0, 280.0, 150.0], 1), // 30pt gap
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_just_under_threshold_grouped() {
        // Two regions 29pt apart → grouped
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [179.0, 50.0, 279.0, 150.0], 1), // 29pt gap
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_chain_connectivity() {
        // A - B - C where A-B < 30pt, B-C < 30pt, but A-C > 30pt
        // CC clustering should group all three via transitivity
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [0.0, 50.0, 100.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [125.0, 50.0, 225.0, 150.0], 1),   // 25pt from A
            make_groupable("p1_2", RegionKind::Image, [250.0, 50.0, 350.0, 150.0], 2),   // 25pt from B, 150pt from A
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].items.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_table_caption_promotion() {
        // Group with Table items: caption starting with "Table" should be promoted
        let mut regions = vec![
            {
                let mut r = make_groupable("p1_0", RegionKind::Table, [50.0, 50.0, 250.0, 200.0], 0);
                let mut cap = make_region(RegionKind::TableTitle, [50.0, 205.0, 250.0, 225.0], Some("Table 3. Comparison"));
                cap.consumed = true;
                r.caption = Some(Box::new(cap));
                r
            },
            make_groupable("p1_1", RegionKind::Table, [260.0, 50.0, 460.0, 200.0], 1),
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        let cap_text = group.caption.as_ref().unwrap().text.as_deref().unwrap();
        assert!(cap_text.starts_with("Table 3"));
    }

    #[test]
    fn test_no_caption_promotion_when_no_fig_or_table_prefix() {
        // Both items have sub-label captions like "(a)" and "(b)" — no promotion
        let mut regions = vec![
            {
                let mut r = make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0);
                let mut cap = make_region(RegionKind::FigureTitle, [50.0, 155.0, 150.0, 170.0], Some("(a) Left panel"));
                cap.consumed = true;
                r.caption = Some(Box::new(cap));
                r
            },
            {
                let mut r = make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1);
                let mut cap = make_region(RegionKind::FigureTitle, [160.0, 155.0, 260.0, 170.0], Some("(b) Right panel"));
                cap.consumed = true;
                r.caption = Some(Box::new(cap));
                r
            },
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        // No group-level caption since none starts with Fig/Table
        assert!(group.caption.is_none());
        // Both items should keep their captions
        let items = group.items.as_ref().unwrap();
        assert!(items[0].caption.is_some());
        assert!(items[1].caption.is_some());
    }

    #[test]
    fn test_single_caption_fallback_promotion() {
        // Only 1 item has a caption (not starting with Fig/Table) → still promoted
        let mut regions = vec![
            {
                let mut r = make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0);
                let mut cap = make_region(RegionKind::FigureTitle, [50.0, 155.0, 150.0, 170.0], Some("(a) Only panel with label"));
                cap.consumed = true;
                r.caption = Some(Box::new(cap));
                r
            },
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        // Single caption should be promoted as fallback
        let cap_text = group.caption.as_ref().unwrap().text.as_deref().unwrap();
        assert!(cap_text.starts_with("(a)"));
    }

    #[test]
    fn test_orphaned_caption_not_absorbed_when_too_far() {
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),
            {
                let mut cap = make_region(RegionKind::FigureTitle, [50.0, 700.0, 260.0, 720.0], Some("Fig. 99. Very far away"));
                cap.order = 2;
                cap
            },
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        assert!(group.caption.is_none());
        // The far-away caption should NOT be consumed
        let cap_r = regions.iter().find(|r| r.kind == RegionKind::FigureTitle).unwrap();
        assert!(!cap_r.consumed);
    }

    #[test]
    fn test_orphaned_caption_non_fig_prefix_not_absorbed() {
        // Orphaned caption without Fig/Table prefix should not be absorbed
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),
            {
                let mut cap = make_region(RegionKind::FigureTitle, [50.0, 160.0, 260.0, 180.0], Some("(a) Some sub-label"));
                cap.order = 2;
                cap
            },
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        assert!(group.caption.is_none());
    }

    #[test]
    fn test_mixed_image_table_chart_seal_grouping() {
        // All groupable kinds should be grouped together
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Chart, [160.0, 50.0, 260.0, 150.0], 1),
            make_groupable("p1_2", RegionKind::Table, [270.0, 50.0, 370.0, 150.0], 2),
            make_groupable("p1_3", RegionKind::Seal, [50.0, 160.0, 150.0, 260.0], 3),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].items.as_ref().unwrap().len(), 4);
    }

    #[test]
    fn test_non_groupable_kinds_ignored() {
        // Text, DisplayFormula etc. are not groupable even if spatially close
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            {
                let mut r = make_region(RegionKind::DisplayFormula, [160.0, 50.0, 260.0, 150.0], None);
                r.order = 1;
                r.id = "p1_1".into();
                r
            },
        ];
        group_figure_regions(&mut regions);

        // No group should be formed (only 1 groupable region)
        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_group_sorted_by_order() {
        // Groups should appear at the correct reading order position
        let mut regions = vec![
            {
                let mut r = make_region(RegionKind::Text, [50.0, 10.0, 500.0, 40.0], Some("Intro text"));
                r.order = 0;
                r.id = "p1_0".into();
                r
            },
            make_groupable("p1_1", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 1),
            make_groupable("p1_2", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 2),
            {
                let mut r = make_region(RegionKind::Text, [50.0, 300.0, 500.0, 330.0], Some("After figure"));
                r.order = 3;
                r.id = "p1_3".into();
                r
            },
        ];
        group_figure_regions(&mut regions);

        // The group should be between the two text regions
        let non_consumed: Vec<&Region> = regions.iter().filter(|r| !r.consumed).collect();
        assert_eq!(non_consumed.len(), 3); // text, group, text
        assert_eq!(non_consumed[0].kind, RegionKind::Text);
        assert_eq!(non_consumed[1].kind, RegionKind::FigureGroup);
        assert_eq!(non_consumed[2].kind, RegionKind::Text);
    }

    #[test]
    fn test_diagonal_proximity() {
        // Two regions diagonally placed — edge_distance uses Euclidean
        // gap_x = 10, gap_y = 10, distance = sqrt(200) ≈ 14.14 < 30 → grouped
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [160.0, 160.0, 260.0, 260.0], 1),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_diagonal_too_far() {
        // gap_x = 25, gap_y = 25, distance = sqrt(1250) ≈ 35.4 > 30 → not grouped
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [175.0, 175.0, 275.0, 275.0], 1),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 0);
    }

    #[test]
    fn test_overlapping_visuals_grouped() {
        // Overlapping regions have edge_distance = 0 → always grouped
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 200.0, 200.0], 0),
            make_groupable("p1_1", RegionKind::Image, [150.0, 150.0, 300.0, 300.0], 1),
        ];
        group_figure_regions(&mut regions);

        let groups: Vec<&Region> = regions
            .iter()
            .filter(|r| r.kind == RegionKind::FigureGroup)
            .collect();
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_caption_text_crosscheck_table_on_image() {
        // FigureTitle with "Table 1..." text should NOT match an Image
        let mut regions = vec![
            make_region(RegionKind::Image, [50.0, 100.0, 250.0, 300.0], None),
            make_region(
                RegionKind::FigureTitle,
                [50.0, 310.0, 250.0, 330.0],
                Some("Table 1. Performance results"),
            ),
        ];
        associate_captions(&mut regions);
        assert!(regions[0].caption.is_none());
        assert!(!regions[1].consumed);
    }

    #[test]
    fn test_caption_text_crosscheck_fig_on_table() {
        // TableTitle with "Fig. 1..." text should NOT match a Table
        let mut regions = vec![
            make_region(RegionKind::Table, [50.0, 100.0, 250.0, 300.0], None),
            make_region(
                RegionKind::TableTitle,
                [50.0, 310.0, 250.0, 330.0],
                Some("Fig. 1. Overview"),
            ),
        ];
        associate_captions(&mut regions);
        assert!(regions[0].caption.is_none());
        assert!(!regions[1].consumed);
    }

    #[test]
    fn test_mislabeled_table_caption_matches_table() {
        // FigureTitle with "Table 1..." text SHOULD match a Table (text overrides kind)
        let mut regions = vec![
            make_region(RegionKind::Table, [50.0, 100.0, 250.0, 300.0], None),
            make_region(
                RegionKind::FigureTitle,
                [50.0, 310.0, 250.0, 330.0],
                Some("Table 1. Performance results"),
            ),
        ];
        associate_captions(&mut regions);
        assert_eq!(caption_text(&regions[0]), Some("Table 1. Performance results"));
        assert!(regions[1].consumed);
    }

    #[test]
    fn test_caption_text_crosscheck_with_nearby_table() {
        // Image and Table both near a "Table 1" FigureTitle caption.
        // Caption should go to the Table, not the Image.
        let mut regions = vec![
            make_region(RegionKind::Image, [50.0, 100.0, 250.0, 300.0], None),
            make_region(
                RegionKind::FigureTitle,
                [50.0, 310.0, 250.0, 330.0],
                Some("Table 1. Performance results"),
            ),
            make_region(RegionKind::Table, [50.0, 340.0, 250.0, 500.0], None),
        ];
        associate_captions(&mut regions);
        assert!(regions[0].caption.is_none());
        assert_eq!(caption_text(&regions[2]), Some("Table 1. Performance results"));
        assert!(regions[1].consumed);
    }

    #[test]
    fn test_items_preserve_image_paths() {
        // Grouped items should retain their image_path for write_images
        let mut regions = vec![
            make_groupable("p1_0", RegionKind::Image, [50.0, 50.0, 150.0, 150.0], 0),
            make_groupable("p1_1", RegionKind::Image, [160.0, 50.0, 260.0, 150.0], 1),
        ];
        group_figure_regions(&mut regions);

        let group = regions.iter().find(|r| r.kind == RegionKind::FigureGroup).unwrap();
        let items = group.items.as_ref().unwrap();
        assert!(items[0].image_path.is_some());
        assert!(items[1].image_path.is_some());
    }
}
