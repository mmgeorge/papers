use crate::pdf::PdfChar;

/// Extract text from pdfium characters that fall within a region bounding box.
///
/// Characters are matched by center point, sorted into lines by Y-proximity,
/// grouped into words by X-gaps, and paragraphs by line spacing.
pub fn extract_region_text(chars: &[PdfChar], region_bbox: [f32; 4]) -> String {
    let matched = match_chars_to_region(chars, region_bbox);
    if matched.is_empty() {
        return String::new();
    }

    let lines = group_into_lines(&matched);
    assemble_text(&lines)
}

/// Filter characters whose center falls within the region bounding box.
fn match_chars_to_region<'a>(chars: &'a [PdfChar], bbox: [f32; 4]) -> Vec<&'a PdfChar> {
    let [x1, y1, x2, y2] = bbox;
    chars
        .iter()
        .filter(|c| {
            let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
            let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
            cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2
        })
        .collect()
}

/// Group matched characters into lines by Y-proximity.
///
/// Characters with center Y within `avg_char_height * 0.5` of each other
/// are grouped into the same line. Lines are sorted top-to-bottom.
fn group_into_lines<'a>(chars: &[&'a PdfChar]) -> Vec<Vec<&'a PdfChar>> {
    if chars.is_empty() {
        return vec![];
    }

    // Sort by Y (top-to-bottom), then X (left-to-right)
    let mut sorted: Vec<&PdfChar> = chars.to_vec();
    sorted.sort_by(|a, b| {
        let ay = (a.bbox[1] + a.bbox[3]) / 2.0;
        let by = (b.bbox[1] + b.bbox[3]) / 2.0;
        ay.partial_cmp(&by)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.bbox[0]
                    .partial_cmp(&b.bbox[0])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    // Compute average character height for threshold
    let avg_height = sorted
        .iter()
        .map(|c| (c.bbox[3] - c.bbox[1]).abs())
        .sum::<f32>()
        / sorted.len() as f32;
    let y_threshold = avg_height * 0.5;

    let mut lines: Vec<Vec<&PdfChar>> = vec![];
    let mut current_line: Vec<&PdfChar> = vec![sorted[0]];
    let mut current_y = (sorted[0].bbox[1] + sorted[0].bbox[3]) / 2.0;

    for &ch in &sorted[1..] {
        let ch_y = (ch.bbox[1] + ch.bbox[3]) / 2.0;
        if (ch_y - current_y).abs() <= y_threshold {
            current_line.push(ch);
        } else {
            // Sort current line by X before pushing
            current_line
                .sort_by(|a, b| a.bbox[0].partial_cmp(&b.bbox[0]).unwrap_or(std::cmp::Ordering::Equal));
            lines.push(current_line);
            current_line = vec![ch];
            current_y = ch_y;
        }
    }
    if !current_line.is_empty() {
        current_line
            .sort_by(|a, b| a.bbox[0].partial_cmp(&b.bbox[0]).unwrap_or(std::cmp::Ordering::Equal));
        lines.push(current_line);
    }

    lines
}

/// Assemble lines of characters into text with word boundaries and paragraph breaks.
fn assemble_text(lines: &[Vec<&PdfChar>]) -> String {
    if lines.is_empty() {
        return String::new();
    }

    // Compute global average character width for word boundary detection
    let all_widths: Vec<f32> = lines
        .iter()
        .flat_map(|line| line.iter().map(|c| (c.bbox[2] - c.bbox[0]).abs()))
        .collect();
    let avg_char_width = if all_widths.is_empty() {
        1.0
    } else {
        all_widths.iter().sum::<f32>() / all_widths.len() as f32
    };
    let word_gap_threshold = avg_char_width * 0.3;

    // Compute line spacings for paragraph detection
    let line_spacings: Vec<f32> = lines
        .windows(2)
        .map(|pair| {
            let y1 = pair[0]
                .iter()
                .map(|c| (c.bbox[1] + c.bbox[3]) / 2.0)
                .sum::<f32>()
                / pair[0].len() as f32;
            let y2 = pair[1]
                .iter()
                .map(|c| (c.bbox[1] + c.bbox[3]) / 2.0)
                .sum::<f32>()
                / pair[1].len() as f32;
            (y2 - y1).abs()
        })
        .collect();

    let median_spacing = if line_spacings.is_empty() {
        0.0
    } else {
        let mut sorted = line_spacings.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[sorted.len() / 2]
    };
    let para_threshold = median_spacing * 1.5;

    let mut result = String::new();

    for (line_idx, line) in lines.iter().enumerate() {
        // Build the line string with word boundaries
        let line_text = build_line_text(line, word_gap_threshold);
        result.push_str(&line_text);

        // Check if we need a paragraph break or just a newline
        if line_idx < lines.len() - 1 {
            if para_threshold > 0.0 && line_idx < line_spacings.len() {
                if line_spacings[line_idx] > para_threshold {
                    result.push_str("\n\n");
                } else {
                    result.push('\n');
                }
            } else {
                result.push('\n');
            }
        }
    }

    result
}

/// Build a single line of text from characters, inserting spaces at word boundaries.
fn build_line_text(chars: &[&PdfChar], word_gap_threshold: f32) -> String {
    if chars.is_empty() {
        return String::new();
    }

    let mut text = String::new();
    text.push(chars[0].codepoint);

    for i in 1..chars.len() {
        let gap = chars[i].bbox[0] - chars[i - 1].bbox[2];
        if gap > word_gap_threshold {
            text.push(' ');
        }
        text.push(chars[i].codepoint);
    }

    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_char(c: char, x: f32, y: f32, w: f32, h: f32) -> PdfChar {
        PdfChar {
            codepoint: c,
            bbox: [x, y, x + w, y + h],
        }
    }

    #[test]
    fn test_point_in_bbox() {
        let chars = vec![
            make_char('A', 100.0, 100.0, 10.0, 12.0), // center: (105, 106) — inside
            make_char('B', 600.0, 100.0, 10.0, 12.0), // center: (605, 106) — outside
        ];
        let bbox = [50.0, 50.0, 550.0, 200.0];
        let matched = match_chars_to_region(&chars, bbox);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].codepoint, 'A');
    }

    #[test]
    fn test_single_line_extraction() {
        // Insert a gap between "Hello" and "World"
        let mut chars_with_gap = Vec::new();
        for (i, c) in "HelloWorld".chars().enumerate() {
            let x = if i < 5 {
                100.0 + i as f32 * 12.0
            } else {
                100.0 + 5.0 * 12.0 + 10.0 + (i - 5) as f32 * 12.0
            };
            chars_with_gap.push(make_char(c, x, 100.0, 10.0, 12.0));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars_with_gap, bbox);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_multi_line_extraction() {
        let mut chars = Vec::new();
        // Line 1 at y=100
        for (i, c) in "First".chars().enumerate() {
            chars.push(make_char(c, 100.0 + i as f32 * 12.0, 100.0, 10.0, 12.0));
        }
        // Line 2 at y=130
        for (i, c) in "Second".chars().enumerate() {
            chars.push(make_char(c, 100.0 + i as f32 * 12.0, 130.0, 10.0, 12.0));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox);
        assert!(text.contains("First"));
        assert!(text.contains("Second"));
        assert!(text.contains('\n'));
    }

    #[test]
    fn test_word_boundary_detection() {
        // Characters with a gap between them
        let chars = vec![
            make_char('A', 100.0, 100.0, 10.0, 12.0),
            make_char('B', 112.0, 100.0, 10.0, 12.0),
            // Gap here (> avg_width * 0.3)
            make_char('C', 140.0, 100.0, 10.0, 12.0),
            make_char('D', 152.0, 100.0, 10.0, 12.0),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox);
        assert!(text.contains(' '), "Expected space in: {text}");
    }

    #[test]
    fn test_paragraph_detection() {
        let mut chars = Vec::new();
        // Line 1 at y=100
        for (i, c) in "Line1".chars().enumerate() {
            chars.push(make_char(c, 100.0 + i as f32 * 12.0, 100.0, 10.0, 12.0));
        }
        // Line 2 at y=115 (normal spacing = 15)
        for (i, c) in "Line2".chars().enumerate() {
            chars.push(make_char(c, 100.0 + i as f32 * 12.0, 115.0, 10.0, 12.0));
        }
        // Line 3 at y=130 (normal spacing = 15)
        for (i, c) in "Line3".chars().enumerate() {
            chars.push(make_char(c, 100.0 + i as f32 * 12.0, 130.0, 10.0, 12.0));
        }
        // Line 4 at y=180 (large gap = 50 → paragraph break)
        for (i, c) in "Line4".chars().enumerate() {
            chars.push(make_char(c, 100.0 + i as f32 * 12.0, 180.0, 10.0, 12.0));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox);
        assert!(
            text.contains("\n\n"),
            "Expected paragraph break in: {text:?}"
        );
    }

    #[test]
    fn test_empty_region() {
        let chars = vec![
            make_char('A', 100.0, 100.0, 10.0, 12.0),
        ];
        // Region bbox that doesn't contain any chars
        let bbox = [500.0, 500.0, 600.0, 600.0];
        let text = extract_region_text(&chars, bbox);
        assert!(text.is_empty());
    }

    #[test]
    fn test_overlapping_characters() {
        let chars = vec![
            make_char('A', 98.0, 100.0, 10.0, 12.0),  // center 103 — just inside
            make_char('B', 96.0, 100.0, 10.0, 12.0),  // center 101 — just inside
            make_char('C', 88.0, 100.0, 10.0, 12.0),  // center 93 — outside
        ];
        let bbox = [100.0, 50.0, 600.0, 200.0];
        let matched = match_chars_to_region(&chars, bbox);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_superscript_grouping() {
        // Normal char at y=100, superscript slightly elevated at y=96
        let chars = vec![
            make_char('x', 100.0, 100.0, 10.0, 12.0),
            make_char('2', 112.0, 96.0, 8.0, 10.0), // slightly elevated
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox);
        // Should be on same line since Y difference < avg_height * 0.5
        assert!(!text.contains('\n'), "Superscript should stay on same line: {text:?}");
    }
}
