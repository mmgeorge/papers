use crate::pdf::PdfChar;

/// A character with coordinates converted to image space (Y-down, top-left origin).
struct ImageChar<'a> {
    codepoint: char,
    /// Bounding box in image-space PDF points: [x1, y1, x2, y2]
    /// where (x1, y1) is top-left and (x2, y2) is bottom-right.
    bbox: [f32; 4],
    _source: &'a PdfChar,
}

/// Convert a PdfChar (PDF Y-up coords) to image space (Y-down coords).
fn to_image_char<'a>(c: &'a PdfChar, page_height_pt: f32) -> ImageChar<'a> {
    // PdfChar.bbox = [left, bottom, right, top] in PDF space (Y-up)
    // Image space = [x1, y1, x2, y2] where y1 = top, y2 = bottom (Y-down)
    ImageChar {
        codepoint: c.codepoint,
        bbox: [
            c.bbox[0],                      // x1 = left (unchanged)
            page_height_pt - c.bbox[3],      // y1 = page_height - top (now top in Y-down)
            c.bbox[2],                       // x2 = right (unchanged)
            page_height_pt - c.bbox[1],      // y2 = page_height - bottom (now bottom in Y-down)
        ],
        _source: c,
    }
}

/// Extract text from pdfium characters that fall within a region bounding box.
///
/// `region_bbox` is in image space (Y-down, top-left origin), in PDF points.
/// `page_height_pt` is used to convert PdfChar coords from PDF space (Y-up) to image space.
pub fn extract_region_text(chars: &[PdfChar], region_bbox: [f32; 4], page_height_pt: f32) -> String {
    let image_chars: Vec<ImageChar> = chars
        .iter()
        .map(|c| to_image_char(c, page_height_pt))
        .collect();

    let matched = match_chars_to_region(&image_chars, region_bbox);
    if matched.is_empty() {
        return String::new();
    }

    let lines = group_into_lines(&matched);
    assemble_text(&lines)
}

/// Filter characters whose center falls within the region bounding box.
fn match_chars_to_region<'a>(chars: &'a [ImageChar], bbox: [f32; 4]) -> Vec<&'a ImageChar<'a>> {
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
/// are grouped into the same line. Lines are sorted top-to-bottom (ascending Y in image space).
fn group_into_lines<'a>(chars: &[&'a ImageChar]) -> Vec<Vec<&'a ImageChar<'a>>> {
    if chars.is_empty() {
        return vec![];
    }

    // Sort by Y (top-to-bottom in image space), then X (left-to-right)
    let mut sorted: Vec<&ImageChar> = chars.to_vec();
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

    let mut lines: Vec<Vec<&ImageChar>> = vec![];
    let mut current_line: Vec<&ImageChar> = vec![sorted[0]];
    let mut current_y = (sorted[0].bbox[1] + sorted[0].bbox[3]) / 2.0;

    for &ch in &sorted[1..] {
        let ch_y = (ch.bbox[1] + ch.bbox[3]) / 2.0;
        if (ch_y - current_y).abs() <= y_threshold {
            current_line.push(ch);
        } else {
            // Sort current line by center-X (handles zero-width pdfium space chars correctly)
            current_line.sort_by(|a, b| {
                let ax = (a.bbox[0] + a.bbox[2]) / 2.0;
                let bx = (b.bbox[0] + b.bbox[2]) / 2.0;
                ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
            });
            lines.push(current_line);
            current_line = vec![ch];
            current_y = ch_y;
        }
    }
    if !current_line.is_empty() {
        current_line.sort_by(|a, b| {
            let ax = (a.bbox[0] + a.bbox[2]) / 2.0;
            let bx = (b.bbox[0] + b.bbox[2]) / 2.0;
            ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
        });
        lines.push(current_line);
    }

    lines
}

/// Assemble lines of characters into text with paragraph breaks.
///
/// Pdfium already embeds space characters at word boundaries, so we just
/// concatenate codepoints and filter control characters. Paragraph breaks
/// are detected from line spacing.
fn assemble_text(lines: &[Vec<&ImageChar>]) -> String {
    if lines.is_empty() {
        return String::new();
    }

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
        let line_text = build_line_text(line);
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

/// Build a single line of text from characters.
///
/// Pdfium embeds space characters (U+0020) with zero-width bboxes at word
/// boundaries, so we just emit each codepoint. Control characters are
/// filtered since line structure comes from Y-proximity grouping.
fn build_line_text(chars: &[&ImageChar]) -> String {
    let mut text = String::new();
    for c in chars {
        if c.codepoint.is_control() {
            continue;
        }
        text.push(c.codepoint);
    }
    // Collapse runs of multiple spaces into a single space
    collapse_spaces(&text)
}

/// Collapse runs of multiple spaces into a single space and trim.
fn collapse_spaces(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch == ' ' {
            if !prev_space {
                result.push(' ');
            }
            prev_space = true;
        } else {
            result.push(ch);
            prev_space = false;
        }
    }
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to make a PdfChar where y is specified in image space (Y-down).
    /// Converts to PDF space internally using page_height_pt.
    fn make_char_image_space(c: char, x: f32, y_top_img: f32, w: f32, h: f32, page_h: f32) -> PdfChar {
        // In image space: y_top_img is the top of the char (small value = near top of page)
        // In PDF space: top = page_h - y_top_img, bottom = page_h - (y_top_img + h)
        let pdf_top = page_h - y_top_img;
        let pdf_bottom = page_h - (y_top_img + h);
        PdfChar {
            codepoint: c,
            bbox: [x, pdf_bottom, x + w, pdf_top],
        }
    }

    const PAGE_H: f32 = 792.0;

    #[test]
    fn test_point_in_bbox() {
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('B', 600.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        // Region bbox in image space (Y-down)
        let bbox = [50.0, 50.0, 550.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert_eq!(text, "A");
    }

    #[test]
    fn test_single_line_extraction() {
        let mut chars = Vec::new();
        // "Hello World" with pdfium-style zero-width space between words
        for (i, c) in "Hello".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 12.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Zero-width space character at the word boundary (as pdfium emits)
        let space_x = 100.0 + 5.0 * 12.0;
        chars.push(make_char_image_space(' ', space_x, 100.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "World".chars().enumerate() {
            chars.push(make_char_image_space(c, space_x + 2.0 + i as f32 * 12.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert_eq!(text, "Hello World");
    }

    #[test]
    fn test_multi_line_extraction() {
        let mut chars = Vec::new();
        // Line 1 at y=100 (image space)
        for (i, c) in "First".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 12.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Line 2 at y=130 (image space)
        for (i, c) in "Second".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 12.0, 130.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert!(text.contains("First"));
        assert!(text.contains("Second"));
        assert!(text.contains('\n'));
    }

    #[test]
    fn test_word_boundary_from_pdfium_space() {
        // Pdfium emits a zero-width space character between words
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('B', 112.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space(' ', 122.0, 100.0, 0.0, 12.0, PAGE_H), // pdfium space
            make_char_image_space('C', 140.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('D', 152.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert_eq!(text, "AB CD");
    }

    #[test]
    fn test_control_chars_filtered() {
        // Control characters like \r, \n, U+0002 should be stripped
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('\u{0002}', 112.0, 100.0, 0.0, 12.0, PAGE_H),
            make_char_image_space('B', 114.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert_eq!(text, "AB");
    }

    #[test]
    fn test_paragraph_detection() {
        let mut chars = Vec::new();
        // Line 1 at y=100
        for (i, c) in "Line1".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 12.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Line 2 at y=115 (normal spacing = 15)
        for (i, c) in "Line2".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 12.0, 115.0, 10.0, 12.0, PAGE_H));
        }
        // Line 3 at y=130 (normal spacing = 15)
        for (i, c) in "Line3".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 12.0, 130.0, 10.0, 12.0, PAGE_H));
        }
        // Line 4 at y=180 (large gap = 50 -> paragraph break)
        for (i, c) in "Line4".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 12.0, 180.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert!(
            text.contains("\n\n"),
            "Expected paragraph break in: {text:?}"
        );
    }

    #[test]
    fn test_empty_region() {
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        // Region bbox that doesn't contain any chars
        let bbox = [500.0, 500.0, 600.0, 600.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert!(text.is_empty());
    }

    #[test]
    fn test_overlapping_characters() {
        let chars = vec![
            make_char_image_space('A', 98.0, 100.0, 10.0, 12.0, PAGE_H),  // center x=103, inside
            make_char_image_space('B', 96.0, 100.0, 10.0, 12.0, PAGE_H),  // center x=101, inside
            make_char_image_space('C', 88.0, 100.0, 10.0, 12.0, PAGE_H),  // center x=93, outside
        ];
        let bbox = [100.0, 50.0, 600.0, 200.0];
        let image_chars: Vec<ImageChar> = chars.iter().map(|c| to_image_char(c, PAGE_H)).collect();
        let matched = match_chars_to_region(&image_chars, bbox);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_superscript_grouping() {
        let chars = vec![
            make_char_image_space('x', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('2', 112.0, 96.0, 8.0, 10.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H);
        assert!(!text.contains('\n'), "Superscript should stay on same line: {text:?}");
    }

    #[test]
    fn test_y_axis_conversion() {
        // Verify that PdfChar Y-up coords are correctly converted to image Y-down
        let page_h = 800.0;
        // A char at the TOP of the page in PDF space: bottom=790, top=800
        let top_char = PdfChar {
            codepoint: 'T',
            bbox: [100.0, 790.0, 110.0, 800.0],
        };
        let img = to_image_char(&top_char, page_h);
        // In image space, this should be near y=0 (top of page)
        assert!((img.bbox[1] - 0.0).abs() < 0.01, "Top char y1 should be ~0, got {}", img.bbox[1]);
        assert!((img.bbox[3] - 10.0).abs() < 0.01, "Top char y2 should be ~10, got {}", img.bbox[3]);

        // A char at the BOTTOM of the page in PDF space: bottom=0, top=10
        let bottom_char = PdfChar {
            codepoint: 'B',
            bbox: [100.0, 0.0, 110.0, 10.0],
        };
        let img = to_image_char(&bottom_char, page_h);
        // In image space, this should be near y=790 (bottom of page)
        assert!((img.bbox[1] - 790.0).abs() < 0.01, "Bottom char y1 should be ~790, got {}", img.bbox[1]);
        assert!((img.bbox[3] - 800.0).abs() < 0.01, "Bottom char y2 should be ~800, got {}", img.bbox[3]);
    }
}
