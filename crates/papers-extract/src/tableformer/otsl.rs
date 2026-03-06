//! OTSL (Optimized Table Structure Language) → HTML conversion.
//!
//! Port of `docling_ibm_models/tableformer/otsl.py::otsl_to_html()`.

/// OTSL token IDs (from tm_config.json word_map_tag).
pub const PAD: i64 = 0;
pub const UNK: i64 = 1;
pub const START: i64 = 2;
pub const END: i64 = 3;
pub const ECEL: i64 = 4;
pub const FCEL: i64 = 5;
pub const LCEL: i64 = 6;
pub const UCEL: i64 = 7;
pub const XCEL: i64 = 8;
pub const NL: i64 = 9;
pub const CHED: i64 = 10;
pub const RHED: i64 = 11;
pub const SROW: i64 = 12;

pub const VOCAB_SIZE: usize = 13;

/// Returns true for cell tokens that produce a `<td>` or `<th>`.
pub fn is_cell_token(tag: i64) -> bool {
    matches!(tag, FCEL | ECEL | CHED | RHED | SROW)
}

/// Returns true for tokens that cause the next tag's hidden state to be skipped
/// for bbox collection.
pub fn is_skip_trigger(tag: i64) -> bool {
    matches!(tag, NL | UCEL | XCEL)
}

/// Convert OTSL token IDs to an HTML table skeleton (empty cells).
///
/// Handles colspan (lcel), rowspan (ucel), and 2D spans (xcel).
pub fn otsl_to_html(tokens: &[i64]) -> String {
    // Strip control tokens, convert to string names
    let tags: Vec<i64> = tokens
        .iter()
        .copied()
        .filter(|&t| !matches!(t, PAD | UNK | START | END))
        .collect();

    if tags.is_empty() {
        return String::new();
    }

    // Split on NL → rows
    let mut rows: Vec<Vec<i64>> = Vec::new();
    let mut current_row: Vec<i64> = Vec::new();
    for &tag in &tags {
        if tag == NL {
            if !current_row.is_empty() {
                rows.push(current_row.clone());
            }
            current_row.clear();
        } else {
            current_row.push(tag);
        }
    }
    if !current_row.is_empty() {
        rows.push(current_row);
    }

    if rows.is_empty() {
        return String::new();
    }

    // Pad to square (longest row)
    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    for row in &mut rows {
        row.resize(max_cols, LCEL);
    }

    // Build HTML
    let mut html = String::from("<table>");
    let mut thead_open = false;

    for (row_idx, row) in rows.iter().enumerate() {
        let has_ched = row.contains(&CHED);

        if !thead_open && has_ched {
            html.push_str("<thead>");
            thead_open = true;
        }
        if thead_open && !has_ched {
            html.push_str("</thead>");
            thead_open = false;
        }

        html.push_str("<tr>");

        for (col_idx, &cell) in row.iter().enumerate() {
            if !is_cell_token(cell) {
                continue;
            }

            let mut colspan = 1u32;
            let mut rowspan = 1u32;

            // Check right for lcel → colspan
            let mut c = col_idx + 1;
            while c < row.len() && row[c] == LCEL {
                colspan += 1;
                c += 1;
            }

            // Check right for xcel → 2D span
            if col_idx + 1 < row.len() && row[col_idx + 1] == XCEL {
                colspan = 1;
                c = col_idx + 1;
                while c < row.len() && row[c] == XCEL {
                    colspan += 1;
                    c += 1;
                }
                // Check down for ucel extent
                rowspan = 1;
                let mut r = row_idx + 1;
                while r < rows.len() && rows[r][col_idx] == UCEL {
                    rowspan += 1;
                    r += 1;
                }
            } else {
                // Check down for ucel → rowspan (no xcel)
                let mut r = row_idx + 1;
                while r < rows.len() && rows[r][col_idx] == UCEL {
                    rowspan += 1;
                    r += 1;
                }
            }

            let tag = if cell == CHED { "th" } else { "td" };
            html.push('<');
            html.push_str(tag);
            if colspan > 1 {
                html.push_str(&format!(" colspan=\"{}\"", colspan));
            }
            if rowspan > 1 {
                html.push_str(&format!(" rowspan=\"{}\"", rowspan));
            }
            html.push_str("></");
            html.push_str(tag);
            html.push('>');
        }

        html.push_str("</tr>");
    }

    if thead_open {
        html.push_str("</thead>");
    }
    html.push_str("</table>");
    html
}

/// Union two xyxy bounding boxes.
fn bbox_union(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}

/// Map raw bbox array to per-HTML-cell bboxes in emission order.
///
/// Returns one `Option<[f32; 4]>` per `<td>`/`<th>` in the HTML.
/// Cells that are skipped (first after START/NL) get `None`.
///
/// `raw_bboxes` are xyxy, indexed by bbox_ind from TokenTracker.
/// `bboxes_to_merge` are (span_start, span_end) pairs from TokenTracker.
pub fn assign_cell_bboxes(
    tokens: &[i64],
    raw_bboxes: &[[f32; 4]],
    bboxes_to_merge: &[(usize, usize)],
) -> Vec<Option<[f32; 4]>> {
    // Apply merges: union span-start and span-end bboxes into span-start
    let mut merged = raw_bboxes.to_vec();
    for &(start, end) in bboxes_to_merge {
        if start < merged.len() && end < merged.len() {
            let union = bbox_union(merged[start], merged[end]);
            merged[start] = union;
        }
    }

    // Replay TokenTracker logic to map bbox_ind → HTML cell index
    let tags: Vec<i64> = tokens
        .iter()
        .copied()
        .filter(|&t| !matches!(t, PAD | UNK | START | END))
        .collect();

    // Build the same grid as otsl_to_html
    let mut rows: Vec<Vec<i64>> = Vec::new();
    let mut current_row: Vec<i64> = Vec::new();
    for &tag in &tags {
        if tag == NL {
            if !current_row.is_empty() {
                rows.push(current_row.clone());
            }
            current_row.clear();
        } else {
            current_row.push(tag);
        }
    }
    if !current_row.is_empty() {
        rows.push(current_row);
    }
    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    for row in &mut rows {
        row.resize(max_cols, LCEL);
    }

    // Walk the token stream to track bbox_ind (same as TokenTracker)
    let mut skip_next_tag = true;
    let mut first_lcel = true;
    let mut bbox_ind: usize = 0;

    // Map: (row_idx, col_idx) → bbox_ind for cell tokens
    let mut cell_bbox_map: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();

    let mut row_idx = 0;
    let mut col_idx = 0;

    for &tag in &tags {
        if tag == NL {
            // NL increments bbox_ind if not skipping
            if !skip_next_tag {
                bbox_ind += 1;
            }
            skip_next_tag = true;
            first_lcel = true;
            row_idx += 1;
            col_idx = 0;
            continue;
        }

        if is_cell_token(tag) {
            if !skip_next_tag {
                cell_bbox_map.insert((row_idx, col_idx), bbox_ind);
                bbox_ind += 1;
            }
            // After a cell token, reset first_lcel
            first_lcel = true;
        } else if tag == LCEL {
            if first_lcel {
                // First LCEL in a span — its bbox is the span-start
                if !skip_next_tag {
                    cell_bbox_map.insert((row_idx, col_idx), bbox_ind);
                    bbox_ind += 1;
                }
                first_lcel = false;
            }
            // Subsequent LCELs don't get their own bbox
        } else if tag == UCEL {
            if !skip_next_tag {
                bbox_ind += 1;
            }
        } else if tag == XCEL {
            // XCEL doesn't get a bbox (skip trigger)
        }

        skip_next_tag = is_skip_trigger(tag);
        if !matches!(tag, LCEL) {
            first_lcel = true;
        }
        col_idx += 1;
    }

    // Now walk the grid the same way as otsl_to_html and collect bboxes per HTML cell
    let mut result = Vec::new();
    for (ri, row) in rows.iter().enumerate() {
        for (ci, &cell) in row.iter().enumerate() {
            if !is_cell_token(cell) {
                continue;
            }

            // Look up bbox for this cell
            let bbox = cell_bbox_map
                .get(&(ri, ci))
                .and_then(|&idx| merged.get(idx).copied());
            result.push(bbox);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_2x2() {
        let tokens = vec![START, FCEL, FCEL, NL, FCEL, FCEL, NL, END];
        let html = otsl_to_html(&tokens);
        assert_eq!(
            html,
            "<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>"
        );
    }

    #[test]
    fn colspan_via_lcel() {
        let tokens = vec![START, FCEL, LCEL, NL, FCEL, FCEL, NL, END];
        let html = otsl_to_html(&tokens);
        assert!(html.contains("colspan=\"2\""));
    }

    #[test]
    fn rowspan_via_ucel() {
        let tokens = vec![START, FCEL, FCEL, NL, UCEL, FCEL, NL, END];
        let html = otsl_to_html(&tokens);
        assert!(html.contains("rowspan=\"2\""));
    }

    #[test]
    fn header_detection() {
        let tokens = vec![START, CHED, CHED, NL, FCEL, FCEL, NL, END];
        let html = otsl_to_html(&tokens);
        assert!(html.contains("<thead>"));
        assert!(html.contains("<th>"));
        assert!(html.contains("</thead>"));
    }

    #[test]
    fn empty_tokens() {
        assert_eq!(otsl_to_html(&[]), "");
        assert_eq!(otsl_to_html(&[START, END]), "");
    }

    #[test]
    fn assign_bboxes_2x2() {
        // START FCEL FCEL NL FCEL FCEL NL END
        // TokenTracker: skip(START), FCEL→bbox0, FCEL→bbox1, NL→bbox2,
        //   skip(NL), FCEL→bbox3, FCEL→bbox4, NL→bbox5
        // But after NL, skip_next_tag=true, so first cell of new row is skipped
        // Actually: START sets skip=true. First FCEL is skipped.
        // Then FCEL→bbox0, NL→bbox1 (NL triggers skip).
        // Row2: first FCEL skipped, second FCEL→bbox2
        let tokens = vec![START, FCEL, FCEL, NL, FCEL, FCEL, NL, END];
        let bboxes = vec![
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 1.0],
        ];
        let result = assign_cell_bboxes(&tokens, &bboxes, &[]);
        // 4 HTML cells: first is None (skipped after START), second has bbox0,
        // third is None (skipped after NL), fourth has bbox2
        assert_eq!(result.len(), 4);
        assert!(result[0].is_none()); // first cell skipped
        assert_eq!(result[1], Some([0.0, 0.0, 0.5, 0.5]));
        assert!(result[2].is_none()); // first cell after NL skipped
        assert_eq!(result[3], Some([0.0, 0.5, 0.5, 1.0]));
    }

    #[test]
    fn assign_bboxes_colspan() {
        // START FCEL LCEL NL FCEL FCEL NL END
        // Row 0: FCEL skipped (after START), LCEL: first_lcel=true → bbox0
        // NL → bbox1 (skip), Row 1: FCEL skipped, FCEL → bbox2
        let tokens = vec![START, FCEL, LCEL, NL, FCEL, FCEL, NL, END];
        let bboxes = vec![
            [0.0, 0.0, 1.0, 0.5],  // LCEL span
            [0.5, 0.0, 1.0, 0.5],  // NL
            [0.5, 0.5, 1.0, 1.0],  // second FCEL
        ];
        let result = assign_cell_bboxes(&tokens, &bboxes, &[]);
        // HTML cells: 1 (colspan=2) + 2 = 3 cells
        assert_eq!(result.len(), 3);
        assert!(result[0].is_none()); // FCEL after START
        assert!(result[1].is_none()); // first of row 2
        assert_eq!(result[2], Some([0.5, 0.5, 1.0, 1.0]));
    }
}
