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

    // Pad to square (longest row).
    // Use ECEL (empty cell) — NOT LCEL, which would be interpreted as
    // colspan continuation and inflate every trailing cell's span.
    let max_cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    for row in &mut rows {
        row.resize(max_cols, ECEL);
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
    // Build reduced merged array — matches Python predict_bboxes:
    // - Span-start entries (first LCEL) get merged with span-end entries
    // - Span-end entries are removed entirely
    // - All other entries are kept as-is
    let merge_map: std::collections::HashMap<usize, usize> =
        bboxes_to_merge.iter().copied().collect();
    let span_end_set: std::collections::HashSet<usize> =
        bboxes_to_merge.iter().map(|&(_, end)| end).collect();

    let mut merged = Vec::with_capacity(raw_bboxes.len());
    for i in 0..raw_bboxes.len() {
        if let Some(&end) = merge_map.get(&i) {
            if end < raw_bboxes.len() {
                merged.push(bbox_union(raw_bboxes[i], raw_bboxes[end]));
            } else {
                merged.push(raw_bboxes[i]);
            }
        } else if !span_end_set.contains(&i) {
            merged.push(raw_bboxes[i]);
        }
    }

    // Build the grid (same as otsl_to_html)
    let tags: Vec<i64> = tokens
        .iter()
        .copied()
        .filter(|&t| !matches!(t, PAD | UNK | START | END))
        .collect();

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

    // Assign reduced merged bboxes to HTML cells using 1:1 sequential mapping.
    // The N-th merged bbox corresponds to the N-th HTML cell in emission order.
    let mut result = Vec::new();
    let mut merged_idx = 0;

    for row in &rows {
        for &cell in row {
            if !is_cell_token(cell) {
                continue;
            }
            result.push(merged.get(merged_idx).copied());
            merged_idx += 1;
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
        // TokenTracker collects 4 hidden states → 4 raw bboxes, no merges.
        // 4 HTML cells, 4 reduced entries → 1:1 sequential.
        let tokens = vec![START, FCEL, FCEL, NL, FCEL, FCEL, NL, END];
        let bboxes = vec![
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
        ];
        let result = assign_cell_bboxes(&tokens, &bboxes, &[]);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], Some([0.0, 0.0, 0.5, 0.5]));
        assert_eq!(result[1], Some([0.5, 0.0, 1.0, 0.5]));
        assert_eq!(result[2], Some([0.0, 0.5, 0.5, 1.0]));
        assert_eq!(result[3], Some([0.5, 0.5, 1.0, 1.0]));
    }

    #[test]
    fn assign_bboxes_colspan() {
        // START FCEL LCEL NL FCEL FCEL NL END
        // bboxes_to_merge = [(0, 1)]: LCEL merged with NL
        // Reduced: [union(raw[0],raw[1]), raw[2], raw[3]] = 3 entries
        // 3 HTML cells → 1:1
        let tokens = vec![START, FCEL, LCEL, NL, FCEL, FCEL, NL, END];
        let bboxes = vec![
            [0.0, 0.0, 1.0, 0.25],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.5, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
        ];
        let result = assign_cell_bboxes(&tokens, &bboxes, &[(0, 1)]);
        assert_eq!(result.len(), 3);
        // union([0,0,1,0.25], [0,0,1,0.5]) = [0,0,1,0.5]
        assert_eq!(result[0], Some([0.0, 0.0, 1.0, 0.5]));
        assert_eq!(result[1], Some([0.0, 0.5, 0.5, 1.0]));
        assert_eq!(result[2], Some([0.5, 0.5, 1.0, 1.0]));
    }

    #[test]
    fn assign_bboxes_3col_with_colspan() {
        // CHED CHED LCEL NL FCEL FCEL FCEL NL
        // bboxes_to_merge = [(1, 2)]: LCEL merged with NL
        // Reduced: [raw[0], union(raw[1],raw[2]), raw[3], raw[4], raw[5]] = 5 entries
        // 5 HTML cells → 1:1
        let tokens = vec![START, CHED, CHED, LCEL, NL, FCEL, FCEL, FCEL, NL, END];
        let bboxes = vec![
            [0.3, 0.0, 0.5, 0.1],
            [0.5, 0.0, 1.0, 0.1],
            [0.5, 0.0, 1.0, 0.2],
            [0.0, 0.5, 0.3, 1.0],
            [0.3, 0.5, 0.6, 1.0],
            [0.6, 0.5, 1.0, 1.0],
        ];
        let result = assign_cell_bboxes(&tokens, &bboxes, &[(1, 2)]);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], Some([0.3, 0.0, 0.5, 0.1]));
        // union([0.5,0,1,0.1], [0.5,0,1,0.2]) = [0.5,0,1,0.2]
        assert_eq!(result[1], Some([0.5, 0.0, 1.0, 0.2]));
        assert_eq!(result[2], Some([0.0, 0.5, 0.3, 1.0]));
        assert_eq!(result[3], Some([0.3, 0.5, 0.6, 1.0]));
        assert_eq!(result[4], Some([0.6, 0.5, 1.0, 1.0]));
    }
}
