use crate::pdf::PdfChar;

/// pdfium replaces line-end hyphens with U+0002 (STX) and flags them as
/// FPDFTEXT_CHAR_HYPHEN. This sentinel signals that the word was split
/// across lines and should be joined directly (without space or newline).
///
/// The STX char retains the original hyphen glyph's bounding box, so it
/// groups with the preceding text during Y-proximity line grouping. In the
/// simple case the continuation ("tering") is on the very next line, but
/// inline formula elements can land on an intermediate Y band, pushing the
/// continuation to a non-adjacent line. See [`assemble_text`] for how this
/// is handled.
const PDFIUM_HYPHEN_MARKER: char = '\u{0002}';

/// An inline formula to be spliced into text at its spatial position.
pub struct InlineFormula {
    /// Bounding box in image-space PDF points [x1, y1, x2, y2] (Y-down, top-left origin).
    pub bbox: [f32; 4],
    pub latex: String,
}

/// A character with coordinates converted to image space (Y-down, top-left origin).
struct ImageChar<'a> {
    codepoint: char,
    /// Bounding box in image-space PDF points: [x1, y1, x2, y2]
    /// where (x1, y1) is top-left and (x2, y2) is bottom-right.
    bbox: [f32; 4],
    /// Pre-computed gap threshold for word boundary detection, in PDF points.
    space_threshold: f32,
    _source: &'a PdfChar,
}

/// A line element: either a character or an inline formula placeholder.
enum LineElement<'a> {
    Char(&'a ImageChar<'a>),
    Formula { latex: &'a str, bbox: [f32; 4] },
}

impl LineElement<'_> {
    fn bbox(&self) -> [f32; 4] {
        match self {
            LineElement::Char(c) => c.bbox,
            LineElement::Formula { bbox, .. } => *bbox,
        }
    }

    fn center_x(&self) -> f32 {
        let b = self.bbox();
        (b[0] + b[2]) / 2.0
    }

    fn center_y(&self) -> f32 {
        let b = self.bbox();
        (b[1] + b[3]) / 2.0
    }
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
        space_threshold: c.space_threshold,
        _source: c,
    }
}

/// Check if a point (cx, cy) falls inside a bbox [x1, y1, x2, y2].
fn point_in_bbox(cx: f32, cy: f32, bbox: [f32; 4]) -> bool {
    cx >= bbox[0] && cx <= bbox[2] && cy >= bbox[1] && cy <= bbox[3]
}

/// Controls how grouped lines are assembled into the final text string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyMode {
    /// Reflow: join lines with spaces, detect paragraphs, dehyphenate.
    /// Used for normal prose text regions.
    Reflow,
    /// Preserve layout: keep each line separate (`\n`-joined) and compute
    /// leading indentation from character positions. Used for algorithm /
    /// pseudocode regions where line structure is meaningful.
    PreserveLayout,
}

/// Extract text from pdfium characters that fall within a region bounding box,
/// splicing inline formula LaTeX at the correct spatial positions.
///
/// `region_bbox` is in image space (Y-down, top-left origin), in PDF points.
/// `page_height_pt` is used to convert PdfChar coords from PDF space (Y-up) to image space.
/// `inline_formulas` are inline formulas whose LaTeX should be spliced as `$latex$` into the text.
/// Characters whose center falls inside any inline formula bbox are excluded.
/// `mode` controls whether lines are reflowed into paragraphs or preserved with indentation.
pub fn extract_region_text(
    chars: &[PdfChar],
    region_bbox: [f32; 4],
    page_height_pt: f32,
    inline_formulas: &[&InlineFormula],
    mode: AssemblyMode,
) -> String {
    let image_chars: Vec<ImageChar> = chars
        .iter()
        .map(|c| to_image_char(c, page_height_pt))
        .collect();

    // Collect exclude bboxes from inline formulas
    let exclude_bboxes: Vec<[f32; 4]> = inline_formulas.iter().map(|f| f.bbox).collect();

    let matched = match_chars_to_region(&image_chars, region_bbox, &exclude_bboxes);

    // Build LineElements from matched chars
    let mut elements: Vec<LineElement> = matched.iter().map(|c| LineElement::Char(c)).collect();

    // Add formula elements for inline formulas whose center falls within the region
    for formula in inline_formulas {
        let cx = (formula.bbox[0] + formula.bbox[2]) / 2.0;
        let cy = (formula.bbox[1] + formula.bbox[3]) / 2.0;
        if point_in_bbox(cx, cy, region_bbox) {
            elements.push(LineElement::Formula {
                latex: &formula.latex,
                bbox: formula.bbox,
            });
        }
    }

    if elements.is_empty() {
        return String::new();
    }

    let lines = group_elements_into_lines(&elements);
    match mode {
        AssemblyMode::Reflow => assemble_text(&lines),
        AssemblyMode::PreserveLayout => assemble_preserving_layout(&lines, region_bbox[0]),
    }
}

/// Filter characters whose center falls within the region bounding box,
/// excluding characters whose center falls inside any exclude bbox.
fn match_chars_to_region<'a>(
    chars: &'a [ImageChar],
    bbox: [f32; 4],
    exclude_bboxes: &[[f32; 4]],
) -> Vec<&'a ImageChar<'a>> {
    let [x1, y1, x2, y2] = bbox;
    chars
        .iter()
        .filter(|c| {
            let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
            let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
            // Must be inside region
            if !(cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2) {
                return false;
            }
            // Must not be inside any exclude bbox
            !exclude_bboxes.iter().any(|eb| point_in_bbox(cx, cy, *eb))
        })
        .collect()
}

/// Group line elements into lines by Y-proximity.
///
/// Elements with center Y within `avg_height * 0.5` of each other
/// are grouped into the same line. Lines are sorted top-to-bottom (ascending Y in image space).
fn group_elements_into_lines<'a>(elements: &'a [LineElement<'a>]) -> Vec<Vec<&'a LineElement<'a>>> {
    if elements.is_empty() {
        return vec![];
    }

    // Sort by Y (top-to-bottom), then X (left-to-right)
    let mut sorted: Vec<&LineElement> = elements.iter().collect();
    sorted.sort_by(|a, b| {
        a.center_y()
            .partial_cmp(&b.center_y())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.bbox()[0]
                    .partial_cmp(&b.bbox()[0])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });

    // Compute average element height for threshold
    let avg_height = sorted
        .iter()
        .map(|e| {
            let b = e.bbox();
            (b[3] - b[1]).abs()
        })
        .sum::<f32>()
        / sorted.len() as f32;
    let y_threshold = avg_height * 0.5;

    let mut lines: Vec<Vec<&LineElement>> = vec![];
    let mut current_line: Vec<&LineElement> = vec![sorted[0]];
    let mut current_y = sorted[0].center_y();

    for &elem in &sorted[1..] {
        let elem_y = elem.center_y();
        if (elem_y - current_y).abs() <= y_threshold {
            current_line.push(elem);
        } else {
            // Sort current line by center-X
            current_line.sort_by(|a, b| {
                a.center_x()
                    .partial_cmp(&b.center_x())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            lines.push(current_line);
            current_line = vec![elem];
            current_y = elem_y;
        }
    }
    if !current_line.is_empty() {
        current_line.sort_by(|a, b| {
            a.center_x()
                .partial_cmp(&b.center_x())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        lines.push(current_line);
    }

    lines
}

/// Assemble lines of elements into flowing text with paragraph detection.
///
/// Each pair of consecutive lines is joined with one of three separators:
///
/// - **Paragraph break** (`\n\n`) — when line spacing exceeds 1.5× the median.
/// - **Reflow space** (` `) — normal same-paragraph line break.
/// - **Dehyphenation** (no separator) — when the line contains a
///   [`PDFIUM_HYPHEN_MARKER`] (U+0002), indicating the word was split by a
///   line-end hyphen (e.g. "encoun-\ntering" → "encountering").
///
/// # Non-adjacent line dehyphenation
///
/// The STX marker groups with the preceding text by Y-proximity, but the
/// word continuation can land on a **non-adjacent** line when inline formula
/// elements sit at an intermediate Y position. For example, in a TeX paper
/// the word "acceleration" split as "ac-" / "celeration" across two
/// typographic lines might produce three grouped lines:
///
/// ```text
/// line 0: "...estimated ac"  (has STX)
/// line 1: [formula glyph at intermediate Y]
/// line 2: "celeration term..."
/// ```
///
/// Without propagation, line 0 joins directly with line 1 (correct), but
/// line 1 joins with line 2 via a reflow space, producing "ac celeration"
/// instead of "acceleration". To fix this, `pending_dehyphen` propagates
/// the "join directly" intent through intervening lines until the
/// continuation is reached.
fn assemble_text(lines: &[Vec<&LineElement>]) -> String {
    if lines.is_empty() {
        return String::new();
    }

    // Compute line spacings for paragraph detection
    let line_spacings: Vec<f32> = lines
        .windows(2)
        .map(|pair| {
            let y1 = pair[0].iter().map(|e| e.center_y()).sum::<f32>() / pair[0].len() as f32;
            let y2 = pair[1].iter().map(|e| e.center_y()).sum::<f32>() / pair[1].len() as f32;
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
    // Propagates dehyphenation across non-adjacent lines — see doc comment.
    let mut pending_dehyphen = false;

    for (line_idx, line) in lines.iter().enumerate() {
        let line_text = build_line_text(line);
        result.push_str(&line_text);

        let has_hyphen_marker = line.iter().any(|e| matches!(
            e, LineElement::Char(c) if c.codepoint == PDFIUM_HYPHEN_MARKER
        ));

        // Check if we need a paragraph break, reflow space, or dehyphenation join
        if line_idx < lines.len() - 1 {
            if para_threshold > 0.0
                && line_idx < line_spacings.len()
                && line_spacings[line_idx] > para_threshold
            {
                result.push_str("\n\n"); // paragraph break
                pending_dehyphen = false;
            } else if has_hyphen_marker || pending_dehyphen {
                // Dehyphenate: join directly with the next line (no separator).
                // Propagate forward if this line has its own STX marker, in case
                // the continuation is on a non-adjacent line.
                pending_dehyphen = has_hyphen_marker;
            } else {
                result.push(' '); // reflow: join with space
            }
        } else if has_hyphen_marker {
            // Last line of this region ends with a split word.
            // Append STX sentinel so the output assembler can join
            // this region's text directly with the next region.
            result.push(PDFIUM_HYPHEN_MARKER);
        }
    }

    result
}

/// Assemble lines preserving their layout: each line becomes a separate `\n`-joined
/// output line, with leading spaces computed from the first character's X offset
/// relative to the region's left edge.
///
/// This is used for algorithm / pseudocode regions where line structure and
/// indentation are structurally meaningful.
fn assemble_preserving_layout(lines: &[Vec<&LineElement>], region_left_x: f32) -> String {
    if lines.is_empty() {
        return String::new();
    }

    // Compute median character width across all lines as the indentation unit.
    let mut char_widths: Vec<f32> = lines
        .iter()
        .flat_map(|line| line.iter())
        .filter_map(|e| match e {
            LineElement::Char(c) if !c.codepoint.is_control() && c.codepoint != ' ' => {
                let w = c.bbox[2] - c.bbox[0];
                if w > 0.0 { Some(w) } else { None }
            }
            _ => None,
        })
        .collect();

    let median_char_width = if char_widths.is_empty() {
        1.0 // fallback — avoids division by zero
    } else {
        char_widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        char_widths[char_widths.len() / 2]
    };

    let mut output_lines: Vec<String> = Vec::with_capacity(lines.len());

    for line in lines {
        // Find the leftmost element's X position for indentation
        let first_x = line
            .iter()
            .map(|e| e.bbox()[0])
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(region_left_x);

        let indent_count = ((first_x - region_left_x) / median_char_width)
            .round()
            .max(0.0) as usize;

        let line_text = build_line_text(line);
        let mut indented = " ".repeat(indent_count);
        indented.push_str(&line_text);
        output_lines.push(indented);
    }

    output_lines.join("\n")
}

/// Build a single line of text from elements sorted left-to-right.
///
/// # Word boundary detection
///
/// pdfium's text layer normally includes synthesized space characters at word
/// boundaries. However, for some PDFs — particularly TeX-typeset academic papers —
/// pdfium's per-character API silently drops these generated spaces, causing words
/// to run together (e.g. "suchasmallsystem" instead of "such a small system").
///
/// To fix this, we replicate pdfium's own space-insertion heuristic: when the
/// horizontal gap between consecutive characters exceeds a font-metric-based
/// threshold, we insert a space. Each character carries a pre-computed
/// `space_threshold` derived from the font's actual space character advance width.
///
/// This handles both cases transparently:
/// - When pdfium DOES emit space characters: they pass through as normal, and the
///   gap between surrounding chars is zero (no duplicate space inserted).
/// - When pdfium DOESN'T emit spaces: the geometric gap exceeds the threshold,
///   and we insert a space ourselves.
fn build_line_text(elements: &[&LineElement]) -> String {
    let mut text = String::new();
    let mut prev_right: Option<f32> = None;

    // Compute fallback threshold from average char width, used when space_threshold
    // is zero (no font info available). Excludes space chars and zero-width chars.
    let avg_width = {
        let widths: Vec<f32> = elements.iter()
            .filter_map(|e| match e {
                LineElement::Char(c) if !c.codepoint.is_control()
                    && c.codepoint != ' '
                    && (c.bbox[2] - c.bbox[0]) > 0.0 => Some(c.bbox[2] - c.bbox[0]),
                _ => None,
            })
            .collect();
        if widths.is_empty() { 0.0 }
        else { widths.iter().sum::<f32>() / widths.len() as f32 }
    };

    for elem in elements {
        match elem {
            LineElement::Char(c) => {
                if c.codepoint.is_control() { continue; }

                // Gap-based word boundary detection.
                if c.codepoint != ' ' {
                    if let Some(pr) = prev_right {
                        let gap = c.bbox[0] - pr;
                        let threshold = if c.space_threshold > 0.0 {
                            c.space_threshold
                        } else {
                            // Last-resort fallback when no font info at all
                            avg_width * 0.3
                        };
                        if threshold > 0.0 && gap >= threshold && !text.ends_with(' ') {
                            text.push(' ');
                        }
                    }
                }

                text.push(c.codepoint);
                // Track the right edge of the last non-zero-width character.
                // Zero-width chars (pdfium's generated spaces) are skipped so they
                // don't reset prev_right and interfere with gap measurement.
                let w = c.bbox[2] - c.bbox[0];
                if w > 0.0 {
                    prev_right = Some(c.bbox[2]);
                }
            }
            LineElement::Formula { latex, .. } => {
                if !text.is_empty() && !text.ends_with(' ') {
                    text.push(' ');
                }
                text.push('$');
                text.push_str(latex);
                text.push('$');
                text.push(' ');
                prev_right = None;
            }
        }
    }
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
    /// Uses a default space_threshold of 1.5 (≈ 10pt font × 0.3 ratio / 2).
    fn make_char_image_space(c: char, x: f32, y_top_img: f32, w: f32, h: f32, page_h: f32) -> PdfChar {
        // In image space: y_top_img is the top of the char (small value = near top of page)
        // In PDF space: top = page_h - y_top_img, bottom = page_h - (y_top_img + h)
        let pdf_top = page_h - y_top_img;
        let pdf_bottom = page_h - (y_top_img + h);
        PdfChar {
            codepoint: c,
            bbox: [x, pdf_bottom, x + w, pdf_top],
            space_threshold: 1.5,
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
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "A");
    }

    #[test]
    fn test_single_line_extraction() {
        let mut chars = Vec::new();
        // "Hello World" with pdfium-style zero-width space between words.
        // Chars are 10pt wide with 0.5pt gap (< 1.5pt threshold) — no false spaces.
        for (i, c) in "Hello".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Zero-width space character at the word boundary (as pdfium emits)
        let space_x = 100.0 + 5.0 * 10.5;
        chars.push(make_char_image_space(' ', space_x, 100.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "World".chars().enumerate() {
            chars.push(make_char_image_space(c, space_x + 5.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "Hello World");
    }

    #[test]
    fn test_multi_line_extraction() {
        let mut chars = Vec::new();
        // Line 1 at y=100 (image space), tight spacing (0.5pt gap < 1.5pt threshold)
        for (i, c) in "First".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Line 2 at y=130 (image space)
        for (i, c) in "Second".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        // Same-paragraph lines are reflowed with a space (not \n)
        assert_eq!(text, "First Second");
    }

    #[test]
    fn test_word_boundary_from_pdfium_space() {
        // Pdfium emits a zero-width space character between words.
        // Within-word gap = 0.5pt (< 1.5pt threshold), between-word gap via pdfium space.
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('B', 110.5, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space(' ', 120.5, 100.0, 0.0, 12.0, PAGE_H), // pdfium space
            make_char_image_space('C', 130.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('D', 140.5, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "AB CD");
    }

    #[test]
    fn test_control_chars_filtered() {
        // Control characters like \r, \n should be stripped from line text.
        // B is placed tight to A (gap = 0.5pt < 1.5pt threshold) so no space inserted.
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('\r', 110.0, 100.0, 0.0, 12.0, PAGE_H),
            make_char_image_space('B', 110.5, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "AB");
    }

    #[test]
    fn test_paragraph_detection() {
        let mut chars = Vec::new();
        // Line 1 at y=100, tight spacing (0.5pt gap)
        for (i, c) in "Line1".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Line 2 at y=115 (normal spacing = 15)
        for (i, c) in "Line2".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 115.0, 10.0, 12.0, PAGE_H));
        }
        // Line 3 at y=130 (normal spacing = 15)
        for (i, c) in "Line3".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }
        // Line 4 at y=180 (large gap = 50 -> paragraph break)
        for (i, c) in "Line4".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 180.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
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
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
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
        let matched = match_chars_to_region(&image_chars, bbox, &[]);
        assert_eq!(matched.len(), 2);
    }

    #[test]
    fn test_superscript_grouping() {
        let chars = vec![
            make_char_image_space('x', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('2', 110.5, 96.0, 8.0, 10.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
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
            space_threshold: 0.0,
        };
        let img = to_image_char(&top_char, page_h);
        // In image space, this should be near y=0 (top of page)
        assert!((img.bbox[1] - 0.0).abs() < 0.01, "Top char y1 should be ~0, got {}", img.bbox[1]);
        assert!((img.bbox[3] - 10.0).abs() < 0.01, "Top char y2 should be ~10, got {}", img.bbox[3]);

        // A char at the BOTTOM of the page in PDF space: bottom=0, top=10
        let bottom_char = PdfChar {
            codepoint: 'B',
            bbox: [100.0, 0.0, 110.0, 10.0],
            space_threshold: 0.0,
        };
        let img = to_image_char(&bottom_char, page_h);
        // In image space, this should be near y=790 (bottom of page)
        assert!((img.bbox[1] - 790.0).abs() < 0.01, "Bottom char y1 should be ~790, got {}", img.bbox[1]);
        assert!((img.bbox[3] - 800.0).abs() < 0.01, "Bottom char y2 should be ~800, got {}", img.bbox[3]);
    }

    #[test]
    fn test_inline_formula_spliced_into_text() {
        // "we define x as" with an inline formula for "t" between "define" and "as"
        // Tight spacing: 8pt wide chars with 0.5pt gap (< 1.5pt threshold)
        let mut chars = Vec::new();
        for (i, c) in "we define".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 8.5, 100.0, 8.0, 12.0, PAGE_H));
        }
        // Gap where inline formula "t" lives (at x=200..220)
        // Some glyph chars from pdfium that overlap the formula bbox
        chars.push(make_char_image_space('t', 205.0, 100.0, 8.0, 12.0, PAGE_H));
        // Continue with "as"
        chars.push(make_char_image_space(' ', 222.0, 100.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "as".chars().enumerate() {
            chars.push(make_char_image_space(c, 225.0 + i as f32 * 8.5, 100.0, 8.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 200.0];
        let formula = InlineFormula {
            bbox: [200.0, 96.0, 220.0, 114.0], // covers the 't' glyph
            latex: "t".into(),
        };
        let formulas: Vec<&InlineFormula> = vec![&formula];

        let text = extract_region_text(&chars, bbox, PAGE_H, &formulas, AssemblyMode::Reflow);
        assert!(text.contains("$t$"), "Expected $t$ in: {text:?}");
        // The raw 't' glyph should be excluded (not duplicated)
        assert!(!text.contains("t $t$"), "Glyph 't' should be excluded: {text:?}");
    }

    #[test]
    fn test_inline_formula_chars_excluded() {
        // Chars inside formula bbox should be excluded from text
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('x', 115.0, 100.0, 8.0, 12.0, PAGE_H),  // inside formula bbox
            make_char_image_space('B', 140.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let exclude = [110.0_f32, 96.0, 130.0, 114.0]; // covers 'x'
        let image_chars: Vec<ImageChar> = chars.iter().map(|c| to_image_char(c, PAGE_H)).collect();
        let matched = match_chars_to_region(&image_chars, bbox, &[exclude]);
        assert_eq!(matched.len(), 2); // A and B, not x
        assert_eq!(matched[0].codepoint, 'A');
        assert_eq!(matched[1].codepoint, 'B');
    }

    #[test]
    fn test_no_inline_formulas_backward_compat() {
        // Passing empty inline_formulas should produce identical results to before.
        // Tight spacing (0.5pt gap < 1.5pt threshold) — no false spaces.
        let chars = vec![
            make_char_image_space('H', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('i', 110.5, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "Hi");
    }

    #[test]
    fn test_gap_based_space_insertion() {
        // Simulate pdfium dropping space characters: chars with a large gap between them
        // should get a space inserted. Threshold = 1.5pt (10pt font × 0.3 ratio / 2).
        let chars = vec![
            // "AB" tight (gap = 0)
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('B', 110.0, 100.0, 10.0, 12.0, PAGE_H),
            // Gap of 5pt (> threshold of 1.5) — should insert space
            make_char_image_space('C', 125.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('D', 135.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "AB CD");
    }

    #[test]
    fn test_gap_detection_no_false_positives() {
        // Tight kerning (gap < threshold) should NOT produce spurious spaces.
        // Threshold = 1.5pt, gap between each char = 0.5pt.
        let chars = vec![
            make_char_image_space('H', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('e', 110.5, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('l', 121.0, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('l', 131.5, 100.0, 10.0, 12.0, PAGE_H),
            make_char_image_space('o', 142.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_dehyphenation() {
        // When a line ends with a STX marker (U+0002), pdfium is signaling a
        // hyphenated word split. The lines should be joined directly.
        let mut chars = Vec::new();
        // Line 1: "encoun" + STX marker at y=100
        for (i, c) in "encoun".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // STX marker at the end of line 1 (pdfium's hyphen replacement)
        chars.push(make_char_image_space('\u{0002}', 163.0, 100.0, 3.0, 12.0, PAGE_H));
        // Line 2: "tering" at y=130
        for (i, c) in "tering".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "encountering");
    }

    #[test]
    fn test_reflow_same_paragraph() {
        // Same-paragraph lines without hyphenation should be joined with a space.
        let mut chars = Vec::new();
        // Line 1 at y=100
        for (i, c) in "the".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Line 2 at y=130
        for (i, c) in "system".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "the system");
    }

    #[test]
    fn test_dehyphenation_across_intermediate_line() {
        // Regression test for non-adjacent line dehyphenation.
        //
        // In real TeX PDFs, inline formula glyphs (e.g. from $a_{\mathrm{ext}}$)
        // can sit at a slightly different Y than the surrounding text. After
        // Y-proximity grouping this creates an intermediate line between the
        // STX-marked line and the continuation:
        //
        //   line 0 (y=100): "...estimated ac"  [has STX]
        //   line 1 (y=108): [formula glyph]
        //   line 2 (y=130): "celeration term..."
        //
        // Without pending_dehyphen propagation, line 1→2 gets a reflow space,
        // producing "ac celeration" instead of "acceleration".
        let mut chars = Vec::new();
        // Line 0 at y=100: "encoun" + STX marker
        for (i, c) in "encoun".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // STX marker (pdfium's hyphen replacement)
        chars.push(make_char_image_space('\u{0002}', 163.0, 100.0, 3.0, 12.0, PAGE_H));
        // Line 1 at y=108: intermediate element (formula glyph on different Y band)
        chars.push(make_char_image_space('~', 200.0, 108.0, 5.0, 8.0, PAGE_H));
        // Line 2 at y=130: "tering" continuation
        for (i, c) in "tering".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        // The intermediate '~' is joined directly (no spaces around it),
        // producing "encoun~tering" — the key point is NO space before "tering".
        assert!(
            !text.contains(" tering"),
            "Should not insert space before continuation: {text:?}"
        );
        assert!(
            text.contains("tering"),
            "Should contain the continuation: {text:?}"
        );
    }

    #[test]
    fn test_gap_with_pdfium_space_no_duplicate() {
        // When pdfium DOES emit a zero-width space character, the gap-based detector
        // should not insert a duplicate space.
        let chars = vec![
            make_char_image_space('A', 100.0, 100.0, 10.0, 12.0, PAGE_H),
            // pdfium's zero-width space between words
            make_char_image_space(' ', 110.0, 100.0, 0.0, 12.0, PAGE_H),
            // gap from A's right edge (110) to B's left edge (115) = 5pt > threshold,
            // but the space char already handled it
            make_char_image_space('B', 115.0, 100.0, 10.0, 12.0, PAGE_H),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "A B");
    }

    // ---- PreserveLayout tests ----

    #[test]
    fn test_preserve_layout_lines_separated() {
        // Two lines should be joined with \n, not reflowed with a space.
        let mut chars = Vec::new();
        for (i, c) in "Line1".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        for (i, c) in "Line2".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }
        // Region left edge at 100.0 — same as where the chars start (no indent)
        let bbox = [100.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        assert_eq!(text, "Line1\nLine2");
    }

    #[test]
    fn test_preserve_layout_indentation() {
        // Line 2 is indented ~3 char widths (10pt each) from the region left edge.
        // Region left edge = 50.0, line 1 starts at x=50 (0 indent),
        // line 2 starts at x=80 (≈3 char widths of 10pt).
        let mut chars = Vec::new();
        for (i, c) in "for".chars().enumerate() {
            chars.push(make_char_image_space(c, 50.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        for (i, c) in "x=1".chars().enumerate() {
            chars.push(make_char_image_space(c, 80.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        // First line should have no leading spaces (starts at region edge)
        assert_eq!(lines[0], "for");
        // Second line should have leading spaces for indentation
        assert!(lines[1].starts_with(' '), "Expected indentation: {lines:?}");
        assert_eq!(lines[1].trim(), "x=1");
    }

    #[test]
    fn test_preserve_layout_single_line() {
        // Edge case: single line should work without \n.
        let mut chars = Vec::new();
        for (i, c) in "return x".chars().enumerate() {
            if c == ' ' {
                chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 0.0, 12.0, PAGE_H));
            } else {
                chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
            }
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        assert!(!text.contains('\n'), "Single line should have no newlines: {text:?}");
        assert!(text.contains("return"), "Should contain 'return': {text:?}");
    }

    #[test]
    fn test_preserve_layout_with_inline_formula() {
        // Inline formula in an algorithm line should be spliced correctly.
        let mut chars = Vec::new();
        for (i, c) in "set".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 8.5, 100.0, 8.0, 12.0, PAGE_H));
        }
        // Formula glyph (will be excluded)
        chars.push(make_char_image_space('x', 140.0, 100.0, 8.0, 12.0, PAGE_H));
        // Continue after formula
        for (i, c) in "to".chars().enumerate() {
            chars.push(make_char_image_space(c, 165.0 + i as f32 * 8.5, 100.0, 8.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 200.0];
        let formula = InlineFormula {
            bbox: [135.0, 96.0, 155.0, 114.0],
            latex: "x_0".into(),
        };
        let formulas: Vec<&InlineFormula> = vec![&formula];
        let text = extract_region_text(&chars, bbox, PAGE_H, &formulas, AssemblyMode::PreserveLayout);
        assert!(text.contains("$x_0$"), "Expected $x_0$ in: {text:?}");
    }

    #[test]
    fn test_reflow_mode_unchanged() {
        // Regression: Reflow mode should still join lines with spaces (not \n).
        let mut chars = Vec::new();
        for (i, c) in "Hello".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        for (i, c) in "World".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "Hello World");
    }
}
