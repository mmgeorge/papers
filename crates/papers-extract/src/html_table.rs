//! Convert HTML `<table>` markup to pipe-delimited markdown.
//!
//! Handles colspan/rowspan, multi-row `<thead>` flattening, `<br>` normalization,
//! and preserves inline LaTeX (`$…$`) that appears in cell content.

// ── HTML helpers ─────────────────────────────────────────────────────

/// Strip HTML tags, decode common entities, and normalize whitespace.
fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    let out = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Strip HTML tags but preserve `$…$` LaTeX that GLM-OCR embeds in cells.
fn strip_html_preserve_math(html: &str) -> String {
    // GLM-OCR already uses $…$ for inline math in table cells,
    // so we only need to strip the surrounding HTML tags.
    strip_html(html)
}

// ── Cell / grid parsing ──────────────────────────────────────────────

struct ParsedCell {
    text: String,
    colspan: usize,
    rowspan: usize,
}

/// Parse a `colspan` or `rowspan` attribute value from an opening tag.
fn parse_span_attr(tag_html: &str, attr: &str) -> usize {
    let needle = format!("{}=\"", attr);
    let start = match tag_html.find(&needle) {
        Some(s) => s + needle.len(),
        None => return 1,
    };
    let rest = &tag_html[start..];
    let end = rest.find('"').unwrap_or(0);
    rest[..end].parse::<usize>().unwrap_or(1).max(1)
}

/// Extract cells from a `<tr>` block, including span attributes.
fn extract_row_cells(tr_html: &str) -> Vec<ParsedCell> {
    let mut cells = Vec::new();
    let mut search = 0;
    let lower = tr_html.to_ascii_lowercase();

    while search < lower.len() {
        let th_pos = lower[search..].find("<th");
        let td_pos = lower[search..].find("<td");

        let cell_start = match (th_pos, td_pos) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let cell_abs = search + cell_start;

        let tag_close = match lower[cell_abs..].find('>') {
            Some(p) => cell_abs + p + 1,
            None => break,
        };

        let opening_tag_lower = &lower[cell_abs..tag_close];
        let colspan = parse_span_attr(opening_tag_lower, "colspan");
        let rowspan = parse_span_attr(opening_tag_lower, "rowspan");

        let close_tag = lower[tag_close..]
            .find("</th>")
            .or_else(|| lower[tag_close..].find("</td>"));

        let cell_content_end = match close_tag {
            Some(p) => tag_close + p,
            None => {
                search = tag_close;
                continue;
            }
        };

        let cell_html = &tr_html[tag_close..cell_content_end];
        let text = strip_html_preserve_math(cell_html);

        cells.push(ParsedCell {
            text,
            colspan,
            rowspan,
        });

        search = cell_content_end + 5;
    }

    cells
}

/// Build a 2D grid from parsed rows, filling colspan/rowspan blocks.
/// Returns (grid, thead_row_count).
fn build_table_grid(html: &str) -> (Vec<Vec<Option<String>>>, usize) {
    let lower = html.to_ascii_lowercase();

    let thead_range = lower
        .find("<thead")
        .and_then(|start| lower.find("</thead>").map(|end| (start, end)));

    let mut raw_rows: Vec<(Vec<ParsedCell>, bool)> = Vec::new();
    let mut search_from = 0;
    while let Some(tr_start) = lower[search_from..].find("<tr") {
        let tr_abs = search_from + tr_start;
        let tr_end_tag = lower[tr_abs..].find("</tr>");
        let tr_end = match tr_end_tag {
            Some(e) => tr_abs + e,
            None => {
                search_from = tr_abs + 1;
                continue;
            }
        };
        let tr_html = &html[tr_abs..tr_end];

        let in_thead = thead_range.map_or(false, |(ts, te)| tr_abs >= ts && tr_abs < te);

        let cells = extract_row_cells(tr_html);
        if !cells.is_empty() {
            raw_rows.push((cells, in_thead));
        }

        search_from = tr_end + 5;
    }

    if raw_rows.is_empty() {
        return (Vec::new(), 0);
    }

    let num_rows = raw_rows.len();
    let mut grid: Vec<Vec<Option<String>>> = vec![Vec::new(); num_rows];
    let mut thead_row_count = 0usize;

    for (row_idx, (cells, in_thead)) in raw_rows.iter().enumerate() {
        if *in_thead {
            thead_row_count = row_idx + 1;
        }

        let mut col = 0;
        for cell in cells {
            while col < grid[row_idx].len() && grid[row_idx][col].is_some() {
                col += 1;
            }

            for dr in 0..cell.rowspan {
                let r = row_idx + dr;
                if r >= num_rows {
                    break;
                }
                for dc in 0..cell.colspan {
                    let c = col + dc;
                    while grid[r].len() <= c {
                        grid[r].push(None);
                    }
                    grid[r][c] = Some(cell.text.clone());
                }
            }

            col += cell.colspan;
        }
    }

    let max_cols = grid.iter().map(|r| r.len()).max().unwrap_or(0);
    for row in &mut grid {
        row.resize(max_cols, None);
    }

    (grid, thead_row_count)
}

/// Flatten multi-row header into a single row by deduplicating vertically.
fn flatten_header(grid: &[Vec<Option<String>>], thead_rows: usize) -> Vec<String> {
    if thead_rows == 0 || grid.is_empty() {
        return Vec::new();
    }

    let ncols = grid[0].len();
    let mut header = Vec::with_capacity(ncols);

    for col in 0..ncols {
        let mut parts: Vec<String> = Vec::new();
        for row in grid.iter().take(thead_rows) {
            let val = row.get(col).and_then(|v| v.clone()).unwrap_or_default();
            if parts.last().map_or(true, |last| last != &val) {
                parts.push(val);
            }
        }
        let label = parts
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        header.push(label);
    }

    header
}

// ── Public API ───────────────────────────────────────────────────────

/// Convert an HTML `<table>` to pipe-delimited markdown.
///
/// Returns `None` if the input contains no `<table>` or no parseable rows.
pub fn html_table_to_markdown(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    if !lower.contains("<table") {
        return None;
    }

    let html = html
        .replace("<br/>", " ")
        .replace("<br>", " ")
        .replace("<br />", " ")
        .replace("<BR/>", " ")
        .replace("<BR>", " ")
        .replace("<BR />", " ");

    let (grid, thead_row_count) = build_table_grid(&html);
    if grid.is_empty() {
        return None;
    }

    let ncols = grid[0].len();
    let mut lines: Vec<String> = Vec::new();

    let sep = format!(
        "| {} |",
        (0..ncols).map(|_| "---").collect::<Vec<_>>().join(" | ")
    );

    if thead_row_count > 1 {
        let header = flatten_header(&grid, thead_row_count);
        lines.push(format!("| {} |", header.join(" | ")));
        lines.push(sep);
        for row in &grid[thead_row_count..] {
            let cells: Vec<&str> = row.iter().map(|c| c.as_deref().unwrap_or("")).collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    } else if thead_row_count == 1 {
        let cells: Vec<&str> = grid[0].iter().map(|c| c.as_deref().unwrap_or("")).collect();
        lines.push(format!("| {} |", cells.join(" | ")));
        lines.push(sep);
        for row in &grid[1..] {
            let cells: Vec<&str> = row.iter().map(|c| c.as_deref().unwrap_or("")).collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    } else {
        // No thead — just emit all rows without separator
        for row in &grid {
            let cells: Vec<&str> = row.iter().map(|c| c.as_deref().unwrap_or("")).collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
    }

    Some(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_table() {
        let html = "<table><thead><tr><th>A</th><th>B</th></tr></thead>\
                     <tbody><tr><td>1</td><td>2</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert_eq!(md, "| A | B |\n| --- | --- |\n| 1 | 2 |");
    }

    #[test]
    fn multi_row_header() {
        let html = "<table><thead>\
                     <tr><th rowspan=\"2\">Name</th><th colspan=\"2\">Score</th></tr>\
                     <tr><th>Min</th><th>Max</th></tr>\
                     </thead><tbody>\
                     <tr><td>Alice</td><td>80</td><td>95</td></tr>\
                     </tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[0], "| Name | Score Min | Score Max |");
        assert_eq!(lines[1], "| --- | --- | --- |");
        assert_eq!(lines[2], "| Alice | 80 | 95 |");
    }

    #[test]
    fn colspan_in_body() {
        let html = "<table><thead><tr><th>A</th><th>B</th><th>C</th></tr></thead>\
                     <tbody><tr><td colspan=\"2\">wide</td><td>x</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines[2], "| wide | wide | x |");
    }

    #[test]
    fn preserves_inline_latex() {
        let html = "<table><thead><tr><th>Param</th><th>Value</th></tr></thead>\
                     <tbody><tr><td>$k_c$</td><td>$1e6$</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("$k_c$"));
        assert!(md.contains("$1e6$"));
    }

    #[test]
    fn no_table_returns_none() {
        assert!(html_table_to_markdown("just some text").is_none());
    }

    #[test]
    fn empty_table_returns_none() {
        assert!(html_table_to_markdown("<table></table>").is_none());
    }

    #[test]
    fn br_tags_normalized() {
        let html = "<table><thead><tr><th>A</th></tr></thead>\
                     <tbody><tr><td>line1<br/>line2</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("line1 line2"));
    }

    #[test]
    fn html_entities_decoded() {
        let html = "<table><thead><tr><th>A &amp; B</th><th>&lt;C&gt;</th></tr></thead>\
                     <tbody><tr><td>x &quot;y&quot;</td><td>z</td></tr></tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        assert!(md.contains("A & B"));
        assert!(md.contains("<C>"));
        assert!(md.contains("x \"y\""));
    }

    #[test]
    fn no_thead() {
        let html = "<table><tr><td>a</td><td>b</td></tr>\
                     <tr><td>c</td><td>d</td></tr></table>";
        let md = html_table_to_markdown(html).unwrap();
        // No separator line since there's no thead
        let lines: Vec<&str> = md.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "| a | b |");
        assert_eq!(lines[1], "| c | d |");
    }

    #[test]
    fn orphan_column_from_colspan_mismatch() {
        // "Group" has colspan=2 but sub-row has 3 cells → orphan column preserved
        let html = "<table><thead>\
                     <tr><th rowspan=\"2\">Name</th><th colspan=\"2\">Group</th></tr>\
                     <tr><th>A</th><th>B</th><th>C</th></tr>\
                     </thead><tbody>\
                     <tr><td>x</td><td>1</td><td>2</td><td>3</td></tr>\
                     </tbody></table>";
        let md = html_table_to_markdown(html).unwrap();
        let lines: Vec<&str> = md.lines().collect();
        // Orphan column C is kept — better to preserve data than drop it
        assert_eq!(lines[0], "| Name | Group A | Group B | C |");
        assert_eq!(lines[2], "| x | 1 | 2 | 3 |");
    }
}
