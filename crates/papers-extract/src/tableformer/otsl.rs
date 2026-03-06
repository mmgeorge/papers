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
}
