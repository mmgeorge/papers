use crate::types::Region;

/// Assign reading order to regions using XY-Cut algorithm.
///
/// Recursively bisects regions by the largest whitespace gap —
/// compares horizontal and vertical gaps, picks the larger one.
pub fn xy_cut_order(regions: &mut [Region]) {
    let mut order = 0u32;
    let mut indices: Vec<usize> = (0..regions.len()).collect();
    xy_cut_recursive(regions, &mut indices, &mut order);
}

/// A detected gap: position (midpoint) and size.
struct Gap {
    position: f32,
    size: f32,
}

fn xy_cut_recursive(regions: &mut [Region], indices: &mut [usize], order: &mut u32) {
    if indices.is_empty() {
        return;
    }
    if indices.len() == 1 {
        regions[indices[0]].order = *order;
        *order += 1;
        return;
    }

    let y_gap = find_largest_y_gap(regions, indices);
    let x_gap = find_largest_x_gap(regions, indices);

    // Pick the axis with the larger gap
    match (&y_gap, &x_gap) {
        (Some(yg), Some(xg)) => {
            if xg.size >= yg.size {
                // X gap is larger: split into left/right (columns)
                let (left, right) = partition_by_x(regions, indices, xg.position);
                xy_cut_recursive(regions, &mut left.clone(), order);
                xy_cut_recursive(regions, &mut right.clone(), order);
            } else {
                // Y gap is larger: split into top/bottom (rows)
                let (top, bottom) = partition_by_y(regions, indices, yg.position);
                xy_cut_recursive(regions, &mut top.clone(), order);
                xy_cut_recursive(regions, &mut bottom.clone(), order);
            }
        }
        (Some(yg), None) => {
            let (top, bottom) = partition_by_y(regions, indices, yg.position);
            xy_cut_recursive(regions, &mut top.clone(), order);
            xy_cut_recursive(regions, &mut bottom.clone(), order);
        }
        (None, Some(xg)) => {
            let (left, right) = partition_by_x(regions, indices, xg.position);
            xy_cut_recursive(regions, &mut left.clone(), order);
            xy_cut_recursive(regions, &mut right.clone(), order);
        }
        (None, None) => {
            // No gap found — assign order by position (top-to-bottom, left-to-right)
            indices.sort_by(|&a, &b| {
                let ra = &regions[a];
                let rb = &regions[b];
                ra.bbox[1]
                    .partial_cmp(&rb.bbox[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(
                        ra.bbox[0]
                            .partial_cmp(&rb.bbox[0])
                            .unwrap_or(std::cmp::Ordering::Equal),
                    )
            });
            for &idx in indices.iter() {
                regions[idx].order = *order;
                *order += 1;
            }
        }
    }
}

/// Find the largest horizontal whitespace gap (Y-axis) among the given regions.
fn find_largest_y_gap(regions: &[Region], indices: &[usize]) -> Option<Gap> {
    if indices.len() < 2 {
        return None;
    }

    let mut y_ranges: Vec<(f32, f32)> = indices
        .iter()
        .map(|&i| (regions[i].bbox[1], regions[i].bbox[3]))
        .collect();
    y_ranges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_gap = 0.0f32;
    let mut gap_pos = 0.0f32;
    let mut current_bottom = y_ranges[0].1;

    for &(y_min, y_max) in &y_ranges[1..] {
        let gap = y_min - current_bottom;
        if gap > max_gap {
            max_gap = gap;
            gap_pos = (current_bottom + y_min) / 2.0;
        }
        current_bottom = current_bottom.max(y_max);
    }

    let total_min = y_ranges.first().map(|r| r.0).unwrap_or(0.0);
    let total_max = y_ranges.iter().map(|r| r.1).fold(0.0f32, f32::max);
    let total_span = total_max - total_min;
    let threshold = total_span * 0.02;

    if max_gap > threshold && max_gap > 1.0 {
        Some(Gap {
            position: gap_pos,
            size: max_gap,
        })
    } else {
        None
    }
}

/// Find the largest vertical whitespace gap (X-axis) among the given regions.
fn find_largest_x_gap(regions: &[Region], indices: &[usize]) -> Option<Gap> {
    if indices.len() < 2 {
        return None;
    }

    let mut x_ranges: Vec<(f32, f32)> = indices
        .iter()
        .map(|&i| (regions[i].bbox[0], regions[i].bbox[2]))
        .collect();
    x_ranges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut max_gap = 0.0f32;
    let mut gap_pos = 0.0f32;
    let mut current_right = x_ranges[0].1;

    for &(x_min, x_max) in &x_ranges[1..] {
        let gap = x_min - current_right;
        if gap > max_gap {
            max_gap = gap;
            gap_pos = (current_right + x_min) / 2.0;
        }
        current_right = current_right.max(x_max);
    }

    let total_min = x_ranges.first().map(|r| r.0).unwrap_or(0.0);
    let total_max = x_ranges.iter().map(|r| r.1).fold(0.0f32, f32::max);
    let total_span = total_max - total_min;
    let threshold = total_span * 0.02;

    if max_gap > threshold && max_gap > 1.0 {
        Some(Gap {
            position: gap_pos,
            size: max_gap,
        })
    } else {
        None
    }
}

/// Partition region indices into those above and below the split Y coordinate.
fn partition_by_y(
    regions: &[Region],
    indices: &[usize],
    split_y: f32,
) -> (Vec<usize>, Vec<usize>) {
    let mut above = Vec::new();
    let mut below = Vec::new();
    for &i in indices {
        let center_y = (regions[i].bbox[1] + regions[i].bbox[3]) / 2.0;
        if center_y < split_y {
            above.push(i);
        } else {
            below.push(i);
        }
    }
    (above, below)
}

/// Partition region indices into those left and right of the split X coordinate.
fn partition_by_x(
    regions: &[Region],
    indices: &[usize],
    split_x: f32,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in indices {
        let center_x = (regions[i].bbox[0] + regions[i].bbox[2]) / 2.0;
        if center_x < split_x {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    (left, right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RegionKind;

    fn make_region(bbox: [f32; 4]) -> Region {
        Region {
            id: String::new(),
            kind: RegionKind::Text,
            bbox,
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
    fn test_single_column() {
        // Three regions stacked vertically
        let mut regions = vec![
            make_region([50.0, 100.0, 500.0, 150.0]),
            make_region([50.0, 200.0, 500.0, 250.0]),
            make_region([50.0, 300.0, 500.0, 350.0]),
        ];
        xy_cut_order(&mut regions);
        assert_eq!(regions[0].order, 0);
        assert_eq!(regions[1].order, 1);
        assert_eq!(regions[2].order, 2);
    }

    #[test]
    fn test_two_column() {
        // Six regions in a 2-column layout.
        // X gap (200) >> Y gaps (10) → columns detected first.
        let mut regions = vec![
            // Left column
            make_region([50.0, 100.0, 200.0, 140.0]),
            make_region([50.0, 150.0, 200.0, 190.0]),
            make_region([50.0, 200.0, 200.0, 240.0]),
            // Right column
            make_region([400.0, 100.0, 550.0, 140.0]),
            make_region([400.0, 150.0, 550.0, 190.0]),
            make_region([400.0, 200.0, 550.0, 240.0]),
        ];
        xy_cut_order(&mut regions);
        // Left column first (0, 1, 2), then right column (3, 4, 5)
        assert_eq!(regions[0].order, 0);
        assert_eq!(regions[1].order, 1);
        assert_eq!(regions[2].order, 2);
        assert_eq!(regions[3].order, 3);
        assert_eq!(regions[4].order, 4);
        assert_eq!(regions[5].order, 5);
    }

    #[test]
    fn test_header_spanning_columns() {
        // Full-width header, then 2-column body.
        // Y gap between header and body (70) > X gap between columns (200 within body),
        // but header is separated first by the Y gap. Then within body, X gap dominates.
        let mut regions = vec![
            // Full-width header
            make_region([50.0, 50.0, 550.0, 80.0]),
            // Left column body
            make_region([50.0, 150.0, 200.0, 190.0]),
            make_region([50.0, 200.0, 200.0, 240.0]),
            // Right column body
            make_region([400.0, 150.0, 550.0, 190.0]),
            make_region([400.0, 200.0, 550.0, 240.0]),
        ];
        xy_cut_order(&mut regions);
        // Header first
        assert_eq!(regions[0].order, 0);
        // Left column body
        assert_eq!(regions[1].order, 1);
        assert_eq!(regions[2].order, 2);
        // Right column body
        assert_eq!(regions[3].order, 3);
        assert_eq!(regions[4].order, 4);
    }

    #[test]
    fn test_no_regions() {
        let mut regions: Vec<Region> = vec![];
        xy_cut_order(&mut regions);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_single_region() {
        let mut regions = vec![make_region([50.0, 100.0, 500.0, 150.0])];
        xy_cut_order(&mut regions);
        assert_eq!(regions[0].order, 0);
    }
}
