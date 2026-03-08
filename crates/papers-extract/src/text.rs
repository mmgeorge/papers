use std::borrow::Cow;

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
    /// Rendered font size in PDF points (from PdfChar).
    font_size: f32,
    /// Whether the character is rendered in an italic font.
    is_italic: bool,
    _source: &'a PdfChar,
}

/// A line element: either a character or an inline formula placeholder.
enum LineElement<'a> {
    Char(&'a ImageChar<'a>),
    Formula { latex: Cow<'a, str>, bbox: [f32; 4] },
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
        font_size: c.font_size,
        is_italic: c.is_italic,
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
    /// Bibliography / reference list mode. Lines are reflowed within each
    /// entry, and entries are separated by paragraph breaks detected via
    /// hanging-indent boundaries (first line flush left, continuations
    /// indented).
    References,
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
                latex: Cow::Borrowed(&formula.latex),
                bbox: formula.bbox,
            });
        }
    }

    if elements.is_empty() {
        return String::new();
    }

    let lines = group_elements_into_lines(&elements);
    match mode {
        AssemblyMode::Reflow => assemble_text(&lines, false),
        AssemblyMode::References => assemble_text(&lines, true),
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

/// Average center Y of all elements in a line.
fn line_center_y(line: &[&LineElement]) -> f32 {
    let sum: f32 = line.iter().map(|e| e.center_y()).sum();
    sum / line.len() as f32
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

    // Post-pass: merge orphan lines (≤2 visible chars) into the nearest adjacent
    // line when the Y gap is within avg_height (a looser threshold than the
    // initial grouping). This rescues stray characters like '←' that are rendered
    // at a slightly different vertical position in the PDF.
    let merge_threshold = avg_height;
    let mut merged: Vec<Vec<&LineElement>> = Vec::with_capacity(lines.len());
    for line in lines {
        let visible_count = line
            .iter()
            .filter(|e| match e {
                LineElement::Char(c) => !c.codepoint.is_whitespace() && !c.codepoint.is_control(),
                LineElement::Formula { .. } => true,
            })
            .count();
        if visible_count <= 2 && !merged.is_empty() {
            // Check Y gap to the previous line
            let prev_y = line_center_y(merged.last().unwrap());
            let this_y = line_center_y(&line);
            if (this_y - prev_y).abs() <= merge_threshold {
                // Merge into previous line, then re-sort by X
                let prev = merged.last_mut().unwrap();
                prev.extend(line);
                prev.sort_by(|a, b| {
                    a.center_x()
                        .partial_cmp(&b.center_x())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                continue;
            }
        }
        merged.push(line);
    }

    merged
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
fn assemble_text(lines: &[Vec<&LineElement>], hanging_indent: bool) -> String {
    if lines.is_empty() {
        return String::new();
    }

    // Pre-compute accent merging across ALL lines (accents and base chars
    // may end up on different lines due to Y-proximity grouping).
    let all_chars: Vec<&ImageChar> = lines
        .iter()
        .flat_map(|line| line.iter())
        .filter_map(|e| match e {
            LineElement::Char(c) if !c.codepoint.is_control() => Some(*c),
            _ => None,
        })
        .collect();
    let accent_merge = compute_accent_merge(&all_chars);

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

    // For hanging-indent mode (bibliographies): compute the left-margin X
    // position and a threshold for detecting de-indentation.  A new entry
    // starts when the next line begins at (near) the left margin.
    //
    // Only enabled when the region actually exhibits indentation variation
    // (some lines flush-left, some indented).
    let (margin_x, indent_threshold, use_hanging) = if hanging_indent {
        // Collect the starting X of each line's first visible character.
        let start_xs: Vec<f32> = lines
            .iter()
            .filter_map(|line| {
                line.iter()
                    .filter_map(|e| match e {
                        LineElement::Char(c) if !c.codepoint.is_control() && c.codepoint != ' ' => {
                            Some(e.bbox()[0])
                        }
                        _ => None,
                    })
                    .next()
            })
            .collect();
        let min_x = start_xs.iter().copied().fold(f32::MAX, f32::min);
        let max_x = start_xs.iter().copied().fold(f32::MIN, f32::max);
        // Median character width → tolerance for "flush left"
        let median_w = {
            let mut widths: Vec<f32> = lines
                .iter()
                .flat_map(|l| l.iter())
                .filter_map(|e| match e {
                    LineElement::Char(c)
                        if !c.codepoint.is_control() && c.codepoint != ' ' =>
                    {
                        let w = c.bbox[2] - c.bbox[0];
                        if w > 0.0 { Some(w) } else { None }
                    }
                    _ => None,
                })
                .collect();
            widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if widths.is_empty() { 4.0 } else { widths[widths.len() / 2] }
        };
        let threshold = median_w * 1.5;
        // Only use hanging-indent detection if there's a meaningful
        // indentation difference (more than the flush-left tolerance).
        let has_indent_variation = (max_x - min_x) > threshold;
        (min_x, threshold, has_indent_variation)
    } else {
        (0.0, 0.0, false)
    };

    let mut result = String::new();
    // Propagates dehyphenation across non-adjacent lines — see doc comment.
    let mut pending_dehyphen = false;

    for (line_idx, line) in lines.iter().enumerate() {
        let processed = detect_scripts(line);
        let processed_refs: Vec<&LineElement> = processed.iter().collect();
        let line_text = build_line_text(&processed_refs, &accent_merge);
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
            } else if has_hyphen_marker {
                // This line ends with a hyphen marker (pdfium's STX).
                // Distinguish compound-word hyphens ("Barrier-augmented")
                // from line-break splits ("encoun-tering"):
                // If the last word starts with uppercase, it's likely a
                // compound word — keep the hyphen. Otherwise, join directly.
                let last_word_start = result.rfind(|c: char| c == ' ' || c == '\n')
                    .map(|p| p + 1)
                    .unwrap_or(0);
                let last_word = &result[last_word_start..];
                if last_word.starts_with(|c: char| c.is_ascii_uppercase()) {
                    result.push('-');
                    // No pending_dehyphen — we kept the hyphen and the
                    // next line will be joined with a normal reflow space.
                } else {
                    pending_dehyphen = true;
                }
            } else if pending_dehyphen {
                // Propagating dehyphenation across non-adjacent lines
                // (e.g., when a formula sits between the two halves of a
                // split word). Intermediate lines have very few visible
                // text characters (1–3 formula glyphs); real text lines
                // have many more. When we see a substantial text line,
                // the continuation was already received on the previous
                // join, so resume normal reflow spacing.
                let text_char_count = line.iter().filter(|e| matches!(
                    e, LineElement::Char(c) if !c.codepoint.is_control() && c.codepoint != ' '
                )).count();
                if text_char_count > 4 {
                    result.push(' ');
                    pending_dehyphen = false;
                }
                // else: short intermediate line (e.g., formula glyph) —
                // keep propagating
            } else if use_hanging && is_hanging_indent_break(
                line, &lines[line_idx + 1], margin_x, indent_threshold,
            ) {
                result.push_str("\n\n"); // bibliography entry break
            } else {
                result.push(' '); // reflow: join with space
            }
        } else if has_hyphen_marker {
            // Last line of this region ends with a split word.
            // Same compound-word check as above: if the last word is
            // capitalized, keep the hyphen instead of propagating the
            // STX sentinel for cross-region dehyphenation.
            let last_word_start = result.rfind(|c: char| c == ' ' || c == '\n')
                .map(|p| p + 1)
                .unwrap_or(0);
            let last_word = &result[last_word_start..];
            if last_word.starts_with(|c: char| c.is_ascii_uppercase()) {
                result.push('-');
            } else {
                result.push(PDFIUM_HYPHEN_MARKER);
            }
        }
    }

    result
}

/// Detect a hanging-indent paragraph break between two lines.
///
/// Returns `true` when the next line starts at (near) the left margin while
/// the current line is indented — indicating a new bibliography entry in a
/// hanging-indent reference list.
fn is_hanging_indent_break(
    _current: &[&LineElement],
    next: &[&LineElement],
    margin_x: f32,
    indent_threshold: f32,
) -> bool {
    // Find the starting X of the next line's first visible character.
    let next_start_x = next
        .iter()
        .filter_map(|e| match e {
            LineElement::Char(c) if !c.codepoint.is_control() && c.codepoint != ' ' => {
                Some(e.bbox()[0])
            }
            _ => None,
        })
        .next();

    match next_start_x {
        Some(x) if (x - margin_x).abs() <= indent_threshold => true,
        _ => false,
    }
}

/// Assemble lines preserving their layout: each line becomes a separate `\n`-joined
/// output line, with leading spaces computed from the content start X offset.
///
/// This is used for algorithm / pseudocode regions where line structure and
/// indentation are structurally meaningful.
///
/// # Line-number handling
///
/// Algorithms often have left-aligned line numbers (e.g. "7 parallel for …").
/// When detected (≥ 2 lines start with digit prefixes), the line numbers are
/// placed at the start of each output line in a right-aligned column, and
/// indentation is computed only for the pseudocode content that follows.
/// Unnumbered lines (comments, titles) get the number column filled with
/// spaces so their content aligns with numbered lines at the same nesting.
///
/// When the algorithm has no line numbers, indentation is computed from the
/// leftmost element on each line.
fn assemble_preserving_layout(lines: &[Vec<&LineElement>], region_left_x: f32) -> String {
    if lines.is_empty() {
        return String::new();
    }

    // Pre-compute accent merging across all lines.
    let all_chars: Vec<&ImageChar> = lines
        .iter()
        .flat_map(|line| line.iter())
        .filter_map(|e| match e {
            LineElement::Char(c) if !c.codepoint.is_control() => Some(*c),
            _ => None,
        })
        .collect();
    let accent_merge = compute_accent_merge(&all_chars);

    let median_char_width = compute_median_char_width(lines);

    // Split each line into optional line-number prefix and content start index.
    let splits: Vec<(Option<String>, usize)> = lines
        .iter()
        .map(|line| split_line_number_prefix(line))
        .collect();

    // Detect numbered algorithm: at least 2 lines have a line-number prefix.
    let numbered_count = splits.iter().filter(|(num, _)| num.is_some()).count();
    let has_line_numbers = numbered_count >= 2;

    // Compute content-start X for each line.
    // Numbered: use first content element (after the number prefix).
    // Unnumbered: use first element on the line.
    let content_starts: Vec<f32> = if has_line_numbers {
        lines
            .iter()
            .zip(splits.iter())
            .map(|(line, (_, content_idx))| {
                line.get(*content_idx)
                    .map(|e| e.bbox()[0])
                    .unwrap_or(region_left_x)
            })
            .collect()
    } else {
        lines
            .iter()
            .map(|line| line.first().map(|e| e.bbox()[0]).unwrap_or(region_left_x))
            .collect()
    };

    let min_content_x = content_starts
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(region_left_x);

    // Max line-number width for right-alignment (0 when unnumbered).
    let max_num_width = if has_line_numbers {
        splits
            .iter()
            .filter_map(|(num, _)| num.as_ref().map(|s| s.len()))
            .max()
            .unwrap_or(0)
    } else {
        0
    };

    let mut output_lines: Vec<String> = Vec::with_capacity(lines.len());

    for i in 0..lines.len() {
        let line = &lines[i];
        let (ref num, content_idx) = splits[i];
        let content_x = content_starts[i];

        let indent_count = ((content_x - min_content_x) / median_char_width)
            .round()
            .max(0.0) as usize;

        let mut formatted = String::new();

        if has_line_numbers {
            // Number column: right-aligned number or spaces for unnumbered lines.
            match num {
                Some(n) => {
                    for _ in 0..(max_num_width - n.len()) {
                        formatted.push(' ');
                    }
                    formatted.push_str(n);
                }
                None => {
                    for _ in 0..max_num_width {
                        formatted.push(' ');
                    }
                }
            }
            formatted.push(' ');
            for _ in 0..indent_count {
                formatted.push(' ');
            }
            let processed = detect_scripts(&line[content_idx..]);
            let processed_refs: Vec<&LineElement> = processed.iter().collect();
            formatted.push_str(&build_line_text(&processed_refs, &accent_merge));
        } else {
            // No line numbers: indent + full line text.
            for _ in 0..indent_count {
                formatted.push(' ');
            }
            let processed = detect_scripts(line);
            let processed_refs: Vec<&LineElement> = processed.iter().collect();
            formatted.push_str(&build_line_text(&processed_refs, &accent_merge));
        }

        // In a numbered algorithm, a very short unnumbered line (after the
        // title) is likely a continuation of the previous numbered line —
        // formula glyphs that landed on a different Y band in the PDF.
        // Only merge when the line has ≤3 visible characters (stray arrows,
        // subscripts, etc.), not legitimate unnumbered lines like comments.
        if has_line_numbers && num.is_none() && !output_lines.is_empty() && i > 0 {
            let visible_chars = line
                .iter()
                .filter(|e| match e {
                    LineElement::Char(c) => {
                        !c.codepoint.is_whitespace() && !c.codepoint.is_control()
                    }
                    LineElement::Formula { .. } => true,
                })
                .count();
            if visible_chars <= 3 {
                let prev = output_lines.last_mut().unwrap();
                let trimmed = formatted.trim();
                if !trimmed.is_empty() {
                    if !prev.ends_with(' ') {
                        prev.push(' ');
                    }
                    prev.push_str(trimmed);
                }
                continue;
            }
        }
        output_lines.push(formatted);
    }

    output_lines.join("\n")
}

/// Compute the median width of visible characters across all lines.
/// Used as the indentation unit for layout preservation.
fn compute_median_char_width(lines: &[Vec<&LineElement>]) -> f32 {
    let mut char_widths: Vec<f32> = lines
        .iter()
        .flat_map(|line| line.iter())
        .filter_map(|e| match e {
            LineElement::Char(c) if !c.codepoint.is_control() && c.codepoint != ' ' => {
                let w = c.bbox[2] - c.bbox[0];
                if w > 0.0 {
                    Some(w)
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    if char_widths.is_empty() {
        1.0 // fallback — avoids division by zero
    } else {
        char_widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        char_widths[char_widths.len() / 2]
    }
}

/// Split a line into an optional line-number prefix and the index where
/// content begins.
///
/// Leading ASCII digits followed by whitespace are treated as a line number.
/// If the digits consume the entire line (no content follows), they are
/// treated as content (not a line number) and `(None, 0)` is returned.
fn split_line_number_prefix(line: &[&LineElement]) -> (Option<String>, usize) {
    let mut idx = 0;
    let mut num_str = String::new();

    // Phase 1: collect leading digits
    while idx < line.len() {
        match &line[idx] {
            LineElement::Char(c) if c.codepoint.is_ascii_digit() => {
                num_str.push(c.codepoint);
                idx += 1;
            }
            _ => break,
        }
    }

    if num_str.is_empty() {
        return (None, 0);
    }

    // Phase 2: skip whitespace after digits
    while idx < line.len() {
        match &line[idx] {
            LineElement::Char(c) if c.codepoint == ' ' || c.codepoint.is_control() => {
                idx += 1;
            }
            _ => break,
        }
    }

    // Phase 3: skip optional ':' separator (common in algorithm pseudocode: "4: x ← ...")
    if idx < line.len() {
        if let LineElement::Char(c) = &line[idx] {
            if c.codepoint == ':' {
                idx += 1;
                // Skip whitespace after the colon
                while idx < line.len() {
                    match &line[idx] {
                        LineElement::Char(c)
                            if c.codepoint == ' ' || c.codepoint.is_control() =>
                        {
                            idx += 1;
                        }
                        _ => break,
                    }
                }
            }
        }
    }

    // If digits consumed entire line, treat as content (not a line number)
    if idx >= line.len() {
        return (None, 0);
    }

    (Some(num_str), idx)
}

/// Maximum font-size ratio (script / median) for a character to be considered a sub/superscript.
const SCRIPT_FONT_RATIO: f32 = 0.85;

/// Detect sub/superscript character patterns on a single line and fold them into
/// `LineElement::Formula` entries.
///
/// A "script run" is detected when a baseline-sized character is immediately followed
/// (horizontally) by one or more smaller-font characters. The vertical shift determines
/// whether it's a subscript (shifted down in Y-down image space) or superscript (shifted up).
///
/// Characters that are already part of a `Formula` element pass through unchanged.
fn detect_scripts<'a>(line: &[&'a LineElement<'a>]) -> Vec<LineElement<'a>> {
    // Compute the baseline (max) font size of visible Char elements on the line.
    // We use max rather than median because script chars can outnumber baseline chars
    // (e.g. `a_{ext}` has 1 baseline + 3 script chars). The max is always the baseline
    // font size since scripts are strictly smaller.
    let baseline_font = line
        .iter()
        .filter_map(|e| match e {
            LineElement::Char(c)
                if !c.codepoint.is_control()
                    && c.codepoint != ' '
                    && (c.bbox[2] - c.bbox[0]) > 0.0
                    && c.font_size > 0.0 =>
            {
                Some(c.font_size)
            }
            _ => None,
        })
        .fold(0.0_f32, f32::max);

    // Need at least 2 visible chars and a positive baseline to detect scripts.
    let visible_count = line
        .iter()
        .filter(|e| matches!(e, LineElement::Char(c)
            if !c.codepoint.is_control()
                && c.codepoint != ' '
                && (c.bbox[2] - c.bbox[0]) > 0.0
                && c.font_size > 0.0))
        .count();

    if visible_count < 2 || baseline_font <= 0.0 {
        return line.iter().map(|e| clone_element(e)).collect();
    }

    let mut result: Vec<LineElement<'a>> = Vec::with_capacity(line.len());
    let mut i = 0;

    while i < line.len() {
        let elem = &line[i];

        // Only try to detect scripts starting from a baseline-sized Char.
        let base_char = match elem {
            LineElement::Char(c)
                if !c.codepoint.is_control()
                    && c.codepoint != ' '
                    && (c.bbox[2] - c.bbox[0]) > 0.0
                    && c.font_size / baseline_font >= SCRIPT_FONT_RATIO =>
            {
                c
            }
            _ => {
                result.push(clone_element(elem));
                i += 1;
                continue;
            }
        };

        // Look ahead for a script run: skip zero-width/control chars, find small-font chars.
        let mut j = i + 1;

        // Skip intervening control/zero-width characters
        while j < line.len() {
            match &line[j] {
                LineElement::Char(c)
                    if c.codepoint.is_control() || (c.bbox[2] - c.bbox[0]) <= 0.0 =>
                {
                    j += 1;
                }
                _ => break,
            }
        }

        // Check if next real char is a script char
        if j >= line.len() {
            result.push(clone_element(elem));
            i += 1;
            continue;
        }

        let first_script = match &line[j] {
            LineElement::Char(c)
                if !c.codepoint.is_control()
                    && c.codepoint != ' '
                    && (c.bbox[2] - c.bbox[0]) > 0.0
                    && c.font_size > 0.0
                    && c.font_size / baseline_font < SCRIPT_FONT_RATIO =>
            {
                c
            }
            _ => {
                result.push(clone_element(elem));
                i += 1;
                continue;
            }
        };

        // Check horizontal adjacency: script char must be close to the base char
        let gap = first_script.bbox[0] - base_char.bbox[2];
        if gap > base_char.space_threshold && base_char.space_threshold > 0.0 {
            // Too far apart — not a script
            result.push(clone_element(elem));
            i += 1;
            continue;
        }

        // Determine sub vs superscript from Y position.
        // In Y-down image space: superscript center_y < base center_y (higher on page).
        let base_cy = (base_char.bbox[1] + base_char.bbox[3]) / 2.0;
        let script_cy = (first_script.bbox[1] + first_script.bbox[3]) / 2.0;
        let is_superscript = script_cy < base_cy;

        // Collect all consecutive small-font chars in the script run
        let script_start = j;
        let mut script_chars = String::new();
        script_chars.push(first_script.codepoint);
        let mut script_end = j + 1;

        while script_end < line.len() {
            match &line[script_end] {
                LineElement::Char(c)
                    if !c.codepoint.is_control()
                        && (c.bbox[2] - c.bbox[0]) > 0.0
                        && c.font_size > 0.0
                        && c.font_size / baseline_font < SCRIPT_FONT_RATIO =>
                {
                    script_chars.push(c.codepoint);
                    script_end += 1;
                }
                // Skip zero-width/control chars within the run
                LineElement::Char(c)
                    if c.codepoint.is_control() || (c.bbox[2] - c.bbox[0]) <= 0.0 =>
                {
                    script_end += 1;
                }
                _ => break,
            }
        }

        // Build the LaTeX string
        let latex = if is_superscript {
            format!("{}^{{{}}}", base_char.codepoint, script_chars)
        } else {
            format!("{}_{{{}}}", base_char.codepoint, script_chars)
        };

        // Compute combined bbox covering base + all script chars
        let last_script_idx = script_end - 1;
        let script_last_bbox = line[last_script_idx].bbox();
        let combined_bbox = [
            base_char.bbox[0],
            base_char.bbox[1].min(line[script_start].bbox()[1]),
            script_last_bbox[2],
            base_char.bbox[3].max(script_last_bbox[3]),
        ];

        result.push(LineElement::Formula {
            latex: Cow::Owned(latex),
            bbox: combined_bbox,
        });

        // Advance past all consumed elements (base + skipped + script run)
        i = script_end;
    }

    result
}

/// Clone a `LineElement` reference into an owned `LineElement`.
fn clone_element<'a>(elem: &LineElement<'a>) -> LineElement<'a> {
    match elem {
        LineElement::Char(c) => LineElement::Char(c),
        LineElement::Formula { latex, bbox } => LineElement::Formula {
            latex: latex.clone(),
            bbox: *bbox,
        },
    }
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
/// Accent-merging results: maps base char pointers to LaTeX commands,
/// and collects accent char pointers to skip during output.
struct AccentMerge<'a> {
    /// Base char → accent command (e.g. "hat", "tilde")
    accented_bases: std::collections::HashMap<*const ImageChar<'a>, &'static str>,
    /// Accent chars to skip
    accent_ptrs: std::collections::HashSet<*const ImageChar<'a>>,
}

/// Scan matched chars for modifier accents overlapping base chars.
/// Uses the same horizontal overlap logic as try_extract_inline_formula.
fn compute_accent_merge<'a>(chars: &[&'a ImageChar<'a>]) -> AccentMerge<'a> {
    let mut accented_bases = std::collections::HashMap::new();
    let mut accent_ptrs = std::collections::HashSet::new();

    for (i, &c) in chars.iter().enumerate() {
        if !is_modifier_accent(c.codepoint) {
            continue;
        }
        let accent_x1 = c.bbox[0];
        let accent_x2 = c.bbox[2];
        let mut best_ptr: Option<*const ImageChar> = None;
        let mut best_overlap = 0.0_f32;
        for (j, &base) in chars.iter().enumerate() {
            if i == j || is_modifier_accent(base.codepoint) {
                continue;
            }
            let overlap_x1 = accent_x1.max(base.bbox[0]);
            let overlap_x2 = accent_x2.min(base.bbox[2]);
            let overlap = (overlap_x2 - overlap_x1).max(0.0);
            if overlap > best_overlap {
                best_overlap = overlap;
                best_ptr = Some(base as *const ImageChar);
            }
        }
        if let Some(ptr) = best_ptr {
            if best_overlap > 0.0 {
                if let Some(cmd) = accent_command(c.codepoint) {
                    accented_bases.insert(ptr, cmd);
                    accent_ptrs.insert(c as *const ImageChar);
                }
            }
        }
    }

    AccentMerge { accented_bases, accent_ptrs }
}

fn build_line_text(elements: &[&LineElement], accent_merge: &AccentMerge) -> String {
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

    let AccentMerge { ref accented_bases, ref accent_ptrs } = *accent_merge;

    let mut in_italic = false;

    // Helper: find the next non-space, non-control element's italic status.
    let next_non_space_italic = |from: usize| -> bool {
        for e in &elements[from..] {
            match e {
                LineElement::Char(c) if !c.codepoint.is_control() && c.codepoint != ' ' => {
                    return c.is_italic;
                }
                LineElement::Formula { .. } => return false,
                _ => continue,
            }
        }
        false
    };

    for (ei, elem) in elements.iter().enumerate() {
        match elem {
            LineElement::Char(c) => {
                if c.codepoint.is_control() { continue; }

                // Skip accent chars (merged into their base char)
                if accent_ptrs.contains(&(*c as *const ImageChar)) {
                    continue;
                }

                // Gap-based word boundary detection.
                if c.codepoint != ' ' {
                    if let Some(pr) = prev_right {
                        let gap = c.bbox[0] - pr;
                        let threshold = if c.space_threshold > 0.0 {
                            c.space_threshold
                        } else {
                            avg_width * 0.3
                        };
                        if threshold > 0.0 && gap >= threshold && !text.ends_with(' ') {
                            if in_italic && !c.is_italic {
                                text.push('*');
                                in_italic = false;
                            }
                            text.push(' ');
                        }
                    }
                }

                // Handle italic transitions.
                if c.codepoint == ' ' {
                    if in_italic && !next_non_space_italic(ei + 1) {
                        // Next real char is not italic — close before the space.
                        text.push('*');
                        in_italic = false;
                    }
                    text.push(' ');
                } else if let Some(cmd) = accented_bases.get(&(*c as *const ImageChar)) {
                    // Accented base char — emit as inline math $\hat{x}$
                    if in_italic {
                        text.push('*');
                        in_italic = false;
                    }
                    text.push_str(&format!("$\\{cmd}{{{}}}", c.codepoint));
                    text.push('$');
                } else {
                    if c.is_italic && !in_italic {
                        text.push('*');
                        in_italic = true;
                    } else if !c.is_italic && in_italic {
                        text.push('*');
                        in_italic = false;
                    }
                    text.push(c.codepoint);
                }

                let w = c.bbox[2] - c.bbox[0];
                if w > 0.0 {
                    prev_right = Some(c.bbox[2]);
                }
            }
            LineElement::Formula { latex, .. } => {
                if in_italic {
                    text.push('*');
                    in_italic = false;
                }
                let trimmed = text.trim_end();
                if trimmed.ends_with('$') && trimmed.len() > 1 {
                    let dollar_pos = trimmed.len() - 1;
                    text.truncate(dollar_pos);
                    text.push(' ');
                    text.push_str(latex);
                    text.push('$');
                    text.push(' ');
                } else {
                    if !text.is_empty() && !text.ends_with(' ') {
                        text.push(' ');
                    }
                    text.push('$');
                    text.push_str(latex);
                    text.push('$');
                    text.push(' ');
                }
                prev_right = None;
            }
        }
    }
    if in_italic {
        text.push('*');
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

// ── Char-based inline formula bypass ────────────────────────────────

/// Map a single character to its LaTeX representation.
///
/// Returns `None` for unknown characters, which causes the entire formula
/// to be rejected to ML OCR (all-or-nothing policy).
fn char_to_latex(c: char) -> Option<&'static str> {
    // First handle Mathematical Alphanumeric Symbols (U+1D400–U+1D7FF).
    // These are used in TeX-typeset PDFs for italic/bold math letters.
    // We map them back to their ASCII equivalents (LaTeX handles styling).
    if let Some(ascii) = math_alphanumeric_to_ascii(c) {
        // Return None to signal "handled by is_known_formula_char / get_latex_for_char"
        // but we need a static str, so we use a lookup table.
        return Some(ascii);
    }
    match c {
        // ASCII letters and digits pass through
        'a'..='z' | 'A'..='Z' | '0'..='9' => None, // handled specially — return char itself
        // Operators
        '+' => Some("+"),
        '-' => Some("-"),
        '=' => Some("="),
        '<' => Some("<"),
        '>' => Some(">"),
        '/' => Some("/"),
        '|' => Some("|"),
        '\u{2212}' => Some("-"),       // minus sign
        '\u{2264}' => Some("\\leq"),
        '\u{2265}' => Some("\\geq"),
        '\u{2260}' => Some("\\neq"),
        '\u{2248}' => Some("\\approx"),
        '\u{2261}' => Some("\\equiv"),
        '\u{00B1}' => Some("\\pm"),
        '\u{00D7}' => Some("\\times"),
        '\u{00F7}' => Some("\\div"),
        '\u{22C5}' => Some("\\cdot"),
        '\u{00B7}' => Some("\\cdot"),  // middle dot
        '\u{221E}' => Some("\\infty"),
        '\u{2225}' => Some("\\|"),     // parallel / double vertical bar (norm)
        // Modifier accents (standalone fallback — normally merged with base char)
        '\u{02C6}' => Some("\\hat{}"),   // ˆ circumflex
        '\u{02DC}' => Some("\\tilde{}"), // ˜ tilde
        '\u{02C7}' => Some("\\check{}"), // ˇ caron
        '\u{02D8}' => Some("\\breve{}"), // ˘ breve
        '\u{02D9}' => Some("\\dot{}"),   // ˙ dot above
        '\u{00AF}' => Some("\\bar{}"),   // ¯ macron
        '\u{00B4}' => Some("\\acute{}"), // ´ acute
        '\u{0060}' => Some("\\grave{}"), // ` grave
        '\u{00A8}' => Some("\\ddot{}"),  // ¨ diaeresis
        // Combining diacriticals (zero-width — normally merged with base char)
        '\u{0302}' => Some("\\hat{}"),   // combining circumflex
        '\u{0303}' => Some("\\tilde{}"), // combining tilde
        '\u{030C}' => Some("\\check{}"), // combining caron
        '\u{0306}' => Some("\\breve{}"), // combining breve
        '\u{0307}' => Some("\\dot{}"),   // combining dot above
        '\u{0304}' => Some("\\bar{}"),   // combining macron
        '\u{0301}' => Some("\\acute{}"), // combining acute
        '\u{0300}' => Some("\\grave{}"), // combining grave
        '\u{0308}' => Some("\\ddot{}"),  // combining diaeresis
        '\u{20D7}' => Some("\\vec{}"),   // combining right arrow above
        // Delimiters
        '(' => Some("("),
        ')' => Some(")"),
        '[' => Some("["),
        ']' => Some("]"),
        '{' => Some("\\{"),
        '}' => Some("\\}"),
        ',' => Some(","),
        '.' => Some("."),
        ':' => Some(":"),
        ';' => Some(";"),
        '!' => Some("!"),
        // Common math symbols
        '\u{2202}' => Some("\\partial"),
        '\u{2207}' => Some("\\nabla"),
        '\u{2208}' => Some("\\in"),
        '\u{2209}' => Some("\\notin"),
        '\u{2229}' => Some("\\cap"),
        '\u{222A}' => Some("\\cup"),
        '\u{2286}' => Some("\\subseteq"),
        '\u{2287}' => Some("\\supseteq"),
        '\u{2192}' => Some("\\to"),
        '\u{2190}' => Some("\\leftarrow"),
        '\u{21D2}' => Some("\\Rightarrow"),
        '\u{2200}' => Some("\\forall"),
        '\u{2203}' => Some("\\exists"),
        '\u{2026}' => Some("\\ldots"),
        '\u{22EF}' => Some("\\cdots"),
        '\'' => Some("'"),
        '\u{2032}' => Some("'"),       // prime
        '\u{2033}' => Some("''"),      // double prime
        // Lowercase Greek (standard Unicode block U+03B1–U+03C9)
        '\u{03B1}' => Some("\\alpha"),
        '\u{03B2}' => Some("\\beta"),
        '\u{03B3}' => Some("\\gamma"),
        '\u{03B4}' => Some("\\delta"),
        '\u{03B5}' => Some("\\epsilon"),
        '\u{03B6}' => Some("\\zeta"),
        '\u{03B7}' => Some("\\eta"),
        '\u{03B8}' => Some("\\theta"),
        '\u{03B9}' => Some("\\iota"),
        '\u{03BA}' => Some("\\kappa"),
        '\u{03BB}' => Some("\\lambda"),
        '\u{03BC}' => Some("\\mu"),
        '\u{03BD}' => Some("\\nu"),
        '\u{03BE}' => Some("\\xi"),
        '\u{03BF}' => Some("o"),       // omicron = latin o
        '\u{03C0}' => Some("\\pi"),
        '\u{03C1}' => Some("\\rho"),
        '\u{03C2}' => Some("\\varsigma"),
        '\u{03C3}' => Some("\\sigma"),
        '\u{03C4}' => Some("\\tau"),
        '\u{03C5}' => Some("\\upsilon"),
        '\u{03C6}' => Some("\\phi"),
        '\u{03C7}' => Some("\\chi"),
        '\u{03C8}' => Some("\\psi"),
        '\u{03C9}' => Some("\\omega"),
        // Uppercase Greek
        '\u{0393}' => Some("\\Gamma"),
        '\u{0394}' => Some("\\Delta"),
        '\u{0398}' => Some("\\Theta"),
        '\u{039B}' => Some("\\Lambda"),
        '\u{039E}' => Some("\\Xi"),
        '\u{03A0}' => Some("\\Pi"),
        '\u{03A3}' => Some("\\Sigma"),
        '\u{03A5}' => Some("\\Upsilon"),
        '\u{03A6}' => Some("\\Phi"),
        '\u{03A8}' => Some("\\Psi"),
        '\u{03A9}' => Some("\\Omega"),
        // Mathematical italic Greek (U+1D6FC–U+1D76F) — used in TeX PDFs
        '\u{1D6FC}' => Some("\\alpha"),
        '\u{1D6FD}' => Some("\\beta"),
        '\u{1D6FE}' => Some("\\gamma"),
        '\u{1D6FF}' => Some("\\delta"),
        '\u{1D700}' => Some("\\epsilon"),
        '\u{1D701}' => Some("\\zeta"),
        '\u{1D702}' => Some("\\eta"),
        '\u{1D703}' => Some("\\theta"),
        '\u{1D704}' => Some("\\iota"),
        '\u{1D705}' => Some("\\kappa"),
        '\u{1D706}' => Some("\\lambda"),
        '\u{1D707}' => Some("\\mu"),
        '\u{1D708}' => Some("\\nu"),
        '\u{1D709}' => Some("\\xi"),
        '\u{1D70B}' => Some("\\pi"),
        '\u{1D70C}' => Some("\\rho"),
        '\u{1D70E}' => Some("\\sigma"),
        '\u{1D70F}' => Some("\\tau"),
        '\u{1D710}' => Some("\\upsilon"),
        '\u{1D711}' => Some("\\phi"),
        '\u{1D712}' => Some("\\chi"),
        '\u{1D713}' => Some("\\psi"),
        '\u{1D714}' => Some("\\omega"),
        '\u{1D715}' => Some("\\partial"),  // math italic partial
        '\u{1D716}' => Some("\\epsilon"),  // math italic epsilon symbol
        '\u{1D717}' => Some("\\vartheta"),
        '\u{1D718}' => Some("\\varkappa"),
        '\u{1D719}' => Some("\\varphi"),
        '\u{1D71A}' => Some("\\varrho"),
        '\u{1D71B}' => Some("\\varpi"),
        // Mathematical italic uppercase Greek (U+1D6E2–U+1D6FB)
        '\u{1D6E4}' => Some("\\Gamma"),
        '\u{1D6E5}' => Some("\\Delta"),
        '\u{1D6E9}' => Some("\\Theta"),
        '\u{1D6EC}' => Some("\\Lambda"),
        '\u{1D6EF}' => Some("\\Xi"),
        '\u{1D6F1}' => Some("\\Pi"),
        '\u{1D6F4}' => Some("\\Sigma"),
        '\u{1D6F6}' => Some("\\Upsilon"),
        '\u{1D6F7}' => Some("\\Phi"),
        '\u{1D6F9}' => Some("\\Psi"),
        '\u{1D6FA}' => Some("\\Omega"),
        _ => None,
    }
}

/// Map Mathematical Alphanumeric Symbols (U+1D400–U+1D7FF) to ASCII equivalents.
/// These are used in TeX-typeset PDFs for italic, bold, script, etc. math letters.
/// Returns a static str of the single ASCII character.
fn math_alphanumeric_to_ascii(c: char) -> Option<&'static str> {
    static ASCII_LETTERS: [&str; 52] = [
        "A","B","C","D","E","F","G","H","I","J","K","L","M",
        "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
        "a","b","c","d","e","f","g","h","i","j","k","l","m",
        "n","o","p","q","r","s","t","u","v","w","x","y","z",
    ];
    let cp = c as u32;
    // Mathematical Bold: A-Z U+1D400..1D419, a-z U+1D41A..1D433
    // Mathematical Italic: A-Z U+1D434..1D44D, a-z U+1D44E..1D467 (h at U+210E)
    // Mathematical Bold Italic: A-Z U+1D468..1D481, a-z U+1D482..1D49B
    // Mathematical Script: A-Z U+1D49C..1D4B5, a-z U+1D4B6..1D4CF
    // Mathematical Bold Script: A-Z U+1D4D0..1D4E9, a-z U+1D4EA..1D503
    // Mathematical Fraktur: A-Z U+1D504..1D51D, a-z U+1D51E..1D537
    // Mathematical Double-Struck: A-Z U+1D538..1D551, a-z U+1D552..1D56B
    // Mathematical Bold Fraktur: A-Z U+1D56C..1D585, a-z U+1D586..1D59F
    // Mathematical Sans-Serif: A-Z U+1D5A0..1D5B9, a-z U+1D5BA..1D5D3
    // Mathematical Sans-Serif Bold: A-Z U+1D5D4..1D5ED, a-z U+1D5EE..1D607
    // Mathematical Sans-Serif Italic: A-Z U+1D608..1D621, a-z U+1D622..1D63B
    // Mathematical Sans-Serif Bold Italic: A-Z U+1D63C..1D655, a-z U+1D656..1D66F
    // Mathematical Monospace: A-Z U+1D670..1D689, a-z U+1D68A..1D6A3
    let ranges: &[(u32, u32)] = &[
        (0x1D400, 0x1D433), // Bold
        (0x1D434, 0x1D467), // Italic
        (0x1D468, 0x1D49B), // Bold Italic
        (0x1D49C, 0x1D4CF), // Script
        (0x1D4D0, 0x1D503), // Bold Script
        (0x1D504, 0x1D537), // Fraktur
        (0x1D538, 0x1D56B), // Double-Struck
        (0x1D56C, 0x1D59F), // Bold Fraktur
        (0x1D5A0, 0x1D5D3), // Sans-Serif
        (0x1D5D4, 0x1D607), // Sans-Serif Bold
        (0x1D608, 0x1D63B), // Sans-Serif Italic
        (0x1D63C, 0x1D66F), // Sans-Serif Bold Italic
        (0x1D670, 0x1D6A3), // Monospace
    ];
    for &(start, end) in ranges {
        if cp >= start && cp <= end {
            let offset = (cp - start) as usize;
            if offset < 52 {
                return Some(ASCII_LETTERS[offset]);
            }
        }
    }
    // Special: Mathematical Italic Small H is at U+210E (Planck constant)
    if cp == 0x210E {
        return Some("h");
    }
    None
}

/// Check if a character is a valid LaTeX formula character.
/// ASCII alphanumerics are always valid; other chars must be in the `char_to_latex` map.
fn is_known_formula_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || char_to_latex(c).is_some()
}

/// Get the LaTeX representation of a character.
/// ASCII alphanumerics return themselves; others use `char_to_latex`.
fn get_latex_for_char(c: char) -> String {
    if c.is_ascii_alphanumeric() {
        c.to_string()
    } else {
        char_to_latex(c).unwrap().to_string()
    }
}

/// Replace raw Unicode codepoints in a LaTeX string with proper commands.
///
/// `detect_scripts()` produces strings like `α_{t}` or `𝐺_{𝑖}` containing raw Unicode.
/// This post-processes them into `\alpha_{t}` or `G_{i}`, inserting spaces where
/// LaTeX requires them (e.g. `\epsilon v` not `\epsilonv`).
fn replace_greek_in_latex(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_was_command = false;
    for c in s.chars() {
        if c.is_ascii() {
            if prev_was_command && c.is_ascii_alphabetic() {
                result.push(' ');
            }
            result.push(c);
            prev_was_command = false;
        } else if let Some(latex) = char_to_latex(c) {
            if prev_was_command && latex.starts_with(|ch: char| ch.is_ascii_alphabetic()) {
                result.push(' ');
            }
            result.push_str(latex);
            prev_was_command = is_latex_command(latex);
        } else {
            // Should not happen if we validated all chars, but be safe
            result.push(c);
            prev_was_command = false;
        }
    }
    result
}

/// Check if a string is a LaTeX command (e.g. `\alpha`, `\nabla`).
/// A command starts with `\` followed by one or more ASCII letters, and nothing else.
fn is_latex_command(s: &str) -> bool {
    s.starts_with('\\') && s.len() > 1 && s[1..].chars().all(|c| c.is_ascii_alphabetic())
}

/// Try to extract a LaTeX string for an inline formula directly from PDF characters.
///
/// Expand formula bbox to include matching brackets that were clipped by layout detection.
///
/// When the bbox clips a bracket expression (e.g. `|det(G)| < ε` detected as `et(G)| < ε`),
/// we look on the same line for the matching bracket and expand the bbox to cover it.
///
/// Handles: `|...|`, `(...)`, `[...]`, `{...}`, `‖...‖` (U+2016).
/// Expand formula bbox horizontally to include matching brackets clipped by layout detection.
///
/// Public so the pipeline can also use the expanded bbox for OCR crops.
pub fn expand_formula_bbox(chars: &[PdfChar], bbox: [f32; 4], page_height_pt: f32) -> [f32; 4] {
    let image_chars: Vec<ImageChar> = chars
        .iter()
        .map(|c| to_image_char(c, page_height_pt))
        .collect();
    expand_for_brackets(&image_chars, bbox)
}

fn expand_for_brackets(chars: &[ImageChar], bbox: [f32; 4]) -> [f32; 4] {
    let matched = match_chars_to_region(chars, bbox, &[]);

    // Filter to visible chars
    let visible: Vec<&ImageChar> = matched
        .into_iter()
        .filter(|c| {
            !c.codepoint.is_control()
                && c.codepoint != ' '
                && (c.bbox[2] - c.bbox[0]) > 0.0
                && c.font_size > 0.0
        })
        .collect();

    if visible.is_empty() {
        return bbox;
    }

    // Compute Y range of the matched chars for "same line" check
    let y_center_min = visible
        .iter()
        .map(|c| (c.bbox[1] + c.bbox[3]) / 2.0)
        .fold(f32::INFINITY, f32::min);
    let y_center_max = visible
        .iter()
        .map(|c| (c.bbox[1] + c.bbox[3]) / 2.0)
        .fold(f32::NEG_INFINITY, f32::max);
    let avg_height = visible
        .iter()
        .map(|c| c.bbox[3] - c.bbox[1])
        .sum::<f32>()
        / visible.len() as f32;
    let y_tolerance = avg_height * 0.75;

    let is_same_line = |c: &ImageChar| {
        let cy = (c.bbox[1] + c.bbox[3]) / 2.0;
        cy >= y_center_min - y_tolerance && cy <= y_center_max + y_tolerance
    };

    // Count brackets
    let mut parens = 0i32; // +1 for '(', -1 for ')'
    let mut squares = 0i32;
    let mut curlies = 0i32;
    let mut pipes = 0i32; // |
    let mut double_pipes = 0i32; // ‖ (U+2016) and ∥ (U+2225)
    for c in &visible {
        match c.codepoint {
            '(' => parens += 1,
            ')' => parens -= 1,
            '[' => squares += 1,
            ']' => squares -= 1,
            '{' => curlies += 1,
            '}' => curlies -= 1,
            '|' => pipes += 1,
            '\u{2016}' | '\u{2225}' => double_pipes += 1,
            _ => {}
        }
    }

    let mut expanded = bbox;

    // For paired brackets: negative balance means missing openers (look left),
    // positive balance means missing closers (look right).
    let bracket_needs: &[(char, char, i32)] = &[
        ('(', ')', parens),
        ('[', ']', squares),
        ('{', '}', curlies),
    ];

    for &(opener, closer, balance) in bracket_needs {
        if balance < 0 {
            // Missing openers — scan left
            let need = (-balance) as usize;
            expanded = scan_for_bracket(chars, expanded, opener, need, true, &is_same_line);
        } else if balance > 0 {
            // Missing closers — scan right
            let need = balance as usize;
            expanded = scan_for_bracket(chars, expanded, closer, need, false, &is_same_line);
        }
    }

    // For | and ‖: odd count means one is missing. Scan left first (more common
    // for clipped left edge), then right if still odd.
    if pipes % 2 != 0 {
        expanded = scan_for_bracket(chars, expanded, '|', 1, true, &is_same_line);
        // Re-check: if we found one on the left, we're balanced. If not, try right.
        let new_matched = match_chars_to_region(chars, expanded, &[]);
        let new_pipes = new_matched.iter().filter(|c| c.codepoint == '|').count();
        if new_pipes % 2 != 0 {
            expanded = scan_for_bracket(chars, expanded, '|', 1, false, &is_same_line);
        }
    }
    if double_pipes % 2 != 0 {
        // Try both U+2016 and U+2225 (∥)
        expanded = scan_for_bracket(chars, expanded, '\u{2016}', 1, true, &is_same_line);
        expanded = scan_for_bracket(chars, expanded, '\u{2225}', 1, true, &is_same_line);
        let new_matched = match_chars_to_region(chars, expanded, &[]);
        let new_dp = new_matched
            .iter()
            .filter(|c| c.codepoint == '\u{2016}' || c.codepoint == '\u{2225}')
            .count();
        if new_dp % 2 != 0 {
            expanded = scan_for_bracket(chars, expanded, '\u{2016}', 1, false, &is_same_line);
            expanded = scan_for_bracket(chars, expanded, '\u{2225}', 1, false, &is_same_line);
        }
    }

    expanded
}

/// Scan for `count` occurrences of `target` char outside `bbox` on the same line.
/// If `look_left` is true, scan chars to the left of bbox; otherwise to the right.
/// Returns the expanded bbox if found.
fn scan_for_bracket(
    chars: &[ImageChar],
    bbox: [f32; 4],
    target: char,
    count: usize,
    look_left: bool,
    is_same_line: &dyn Fn(&ImageChar) -> bool,
) -> [f32; 4] {
    let mut found = 0;
    let mut expanded = bbox;

    // Collect candidates: same-line chars of the target type outside the bbox
    let mut candidates: Vec<&ImageChar> = chars
        .iter()
        .filter(|c| {
            c.codepoint == target
                && is_same_line(c)
                && (c.bbox[2] - c.bbox[0]) > 0.0
                && !c.codepoint.is_control()
        })
        .filter(|c| {
            let cx = (c.bbox[0] + c.bbox[2]) / 2.0;
            if look_left {
                cx < bbox[0]
            } else {
                cx > bbox[2]
            }
        })
        .collect();

    // Sort: if looking left, closest first (descending x); if right, closest first (ascending x)
    if look_left {
        candidates.sort_by(|a, b| {
            let ax = (a.bbox[0] + a.bbox[2]) / 2.0;
            let bx = (b.bbox[0] + b.bbox[2]) / 2.0;
            bx.partial_cmp(&ax).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        candidates.sort_by(|a, b| {
            let ax = (a.bbox[0] + a.bbox[2]) / 2.0;
            let bx = (b.bbox[0] + b.bbox[2]) / 2.0;
            ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    for c in candidates {
        if found >= count {
            break;
        }
        // Expand bbox horizontally only — vertical expansion pulls in other lines
        expanded[0] = expanded[0].min(c.bbox[0]);
        expanded[2] = expanded[2].max(c.bbox[2]);
        found += 1;
    }

    expanded
}

/// Returns true if the character is a modifier accent that should be merged with its base char.
/// Returns the LaTeX accent command for a modifier/combining accent character, if any.
/// e.g. ˆ (U+02C6) → "hat", ˜ (U+02DC) → "tilde", etc.
fn accent_command(c: char) -> Option<&'static str> {
    match c {
        // Modifier letters (spacing)
        '\u{02C6}' => Some("hat"),       // ˆ circumflex
        '\u{02DC}' => Some("tilde"),     // ˜ tilde
        '\u{02C7}' => Some("check"),     // ˇ caron/háček
        '\u{02D8}' => Some("breve"),     // ˘ breve
        '\u{02D9}' => Some("dot"),       // ˙ dot above
        '\u{00AF}' => Some("bar"),       // ¯ macron
        '\u{00B4}' => Some("acute"),     // ´ acute
        '\u{0060}' => Some("grave"),     // ` grave
        '\u{00A8}' => Some("ddot"),      // ¨ diaeresis
        // Combining diacriticals (zero-width, attached to preceding char)
        '\u{0302}' => Some("hat"),       // combining circumflex
        '\u{0303}' => Some("tilde"),     // combining tilde
        '\u{030C}' => Some("check"),     // combining caron
        '\u{0306}' => Some("breve"),     // combining breve
        '\u{0307}' => Some("dot"),       // combining dot above
        '\u{0304}' => Some("bar"),       // combining macron
        '\u{0301}' => Some("acute"),     // combining acute
        '\u{0300}' => Some("grave"),     // combining grave
        '\u{0308}' => Some("ddot"),      // combining diaeresis
        '\u{20D7}' => Some("vec"),       // combining right arrow above
        _ => None,
    }
}

fn is_modifier_accent(c: char) -> bool {
    accent_command(c).is_some()
}

/// Returns `Some(latex)` if every character in the region maps to a known LaTeX token,
/// `None` otherwise (the formula should be sent to ML OCR).
///
/// `formula_bbox` is in image-space Y-down, PDF points.
/// `page_height_pt` is used to convert PdfChar coords from PDF space (Y-up) to image space.
/// Trim stray text characters from the edges of a formula's matched chars.
///
/// The layout model's bounding box sometimes clips a neighboring text character,
/// causing its center to fall just inside the formula region. For example, in
/// "form a 3×3 basis", the bbox might start slightly left of "3×3" and capture
/// the article "a".
///
/// Detection: sort chars left-to-right, then check if the leftmost or rightmost
/// char is separated from its nearest neighbor by a space-sized gap. If so, it's
/// a stray text char — remove it and tighten the bbox to exclude it.
///
/// When chars are trimmed, `bbox` is tightened so downstream OCR crops also
/// exclude the stray char.
fn trim_stray_edge_chars<'a>(
    mut visible: Vec<&'a ImageChar<'a>>,
    bbox: &mut [f32; 4],
) -> Vec<&'a ImageChar<'a>> {
    // Need at least 3 chars: ≥2 real formula chars + 1 stray.
    // With only 2 chars we can't distinguish stray from legitimate content.
    if visible.len() < 3 {
        return visible;
    }

    // Sort left-to-right by center X
    visible.sort_by(|a, b| {
        let ax = (a.bbox[0] + a.bbox[2]) / 2.0;
        let bx = (b.bbox[0] + b.bbox[2]) / 2.0;
        ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute median char width as word-space threshold.
    // A gap ≥ 1 full median character width between an edge char and its
    // neighbor indicates a word boundary — the edge char is likely a stray
    // text character captured by the layout bbox, not part of the formula.
    // Intra-formula spacing (around operators, scripts) is typically < 1 char width.
    let mut widths: Vec<f32> = visible
        .iter()
        .map(|c| (c.bbox[2] - c.bbox[0]).max(0.0))
        .filter(|&w| w > 0.0)
        .collect();
    if widths.is_empty() {
        return visible;
    }
    widths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_width = widths[widths.len() / 2];
    let space_gap = median_width;

    // Check leftmost char: gap between its right edge and the second char's left edge
    let left_gap = visible[1].bbox[0] - visible[0].bbox[2];
    if left_gap > space_gap {
        // Stray char on the left — tighten bbox to start after it
        let stray_right = visible[0].bbox[2];
        // New left edge: midpoint between stray's right edge and next char's left edge
        bbox[0] = (stray_right + visible[1].bbox[0]) / 2.0;
        visible.remove(0);
    }

    // Check rightmost char: gap between second-to-last's right edge and last's left edge
    if visible.len() >= 2 {
        let n = visible.len();
        let right_gap = visible[n - 1].bbox[0] - visible[n - 2].bbox[2];
        if right_gap > space_gap {
            // Stray char on the right — tighten bbox to end before it
            let stray_left = visible[n - 1].bbox[0];
            // New right edge: midpoint between prev char's right edge and stray's left edge
            bbox[2] = (visible[n - 2].bbox[2] + stray_left) / 2.0;
            visible.pop();
        }
    }

    visible
}

/// Result of attempting char-based inline formula extraction.
#[derive(Debug)]
pub struct InlineFormulaAttempt {
    /// The LaTeX string if char-based extraction succeeded.
    pub latex: Option<String>,
    /// The adjusted bounding box — may be expanded (bracket recovery) or
    /// tightened (stray edge chars trimmed). Use this for OCR crops when
    /// `latex` is `None`.
    pub adjusted_bbox: [f32; 4],
}

pub fn try_extract_inline_formula(
    chars: &[PdfChar],
    formula_bbox: [f32; 4],
    page_height_pt: f32,
) -> InlineFormulaAttempt {
    let image_chars: Vec<ImageChar> = chars
        .iter()
        .map(|c| to_image_char(c, page_height_pt))
        .collect();

    // Match chars to the formula bbox, then expand if brackets are unbalanced
    let mut bbox = expand_for_brackets(&image_chars, formula_bbox);

    let matched = match_chars_to_region(&image_chars, bbox, &[]);

    // Helper: return early with no latex but the current adjusted bbox
    macro_rules! reject {
        () => {
            return InlineFormulaAttempt { latex: None, adjusted_bbox: bbox };
        };
    }

    // Reject: CR/LF control chars between visible chars indicate fractions
    // (PDF encodes fraction bars as \r\n with zero-width bboxes)
    if matched.iter().any(|c| c.codepoint == '\r' || c.codepoint == '\n') {
        reject!();
    }

    // Filter out control chars and zero-width chars
    let visible: Vec<&ImageChar> = matched
        .into_iter()
        .filter(|c| {
            !c.codepoint.is_control()
                && c.codepoint != ' '
                && (c.bbox[2] - c.bbox[0]) > 0.0
                && c.font_size > 0.0
        })
        .collect();

    // Reject: no chars matched
    if visible.is_empty() {
        reject!();
    }

    // Trim stray edge chars: if the leftmost or rightmost char is separated
    // from its neighbor by a space-sized gap, it's a text char that the layout
    // bbox accidentally captured. Remove it and tighten the bbox.
    let visible = trim_stray_edge_chars(visible, &mut bbox);

    if visible.is_empty() {
        reject!();
    }

    // Reject: ∂ (partial derivative) — almost always appears in fractions or complex
    // expressions where char-based left-to-right reading produces garbage
    if visible.iter().any(|c| c.codepoint == '\u{2202}' || c.codepoint == '\u{1D715}') {
        reject!();
    }

    // Reject: ∇ (nabla) with more than 2 chars — single "∇f" is fine,
    // but longer expressions like "∇²f" or "∇·F" need spatial understanding
    if visible.len() > 2
        && visible.iter().any(|c| c.codepoint == '\u{2207}')
    {
        reject!();
    }

    // Reject: too many chars (complex formula, send to OCR)
    if visible.len() > 20 {
        reject!();
    }

    // Reject: chars span more than one text line.
    // Use the max char height as threshold — sub/superscripts shift Y centers within
    // a single formula "line" so we need a generous threshold (1.5× max height).
    let max_height = visible
        .iter()
        .map(|c| (c.bbox[3] - c.bbox[1]).abs())
        .fold(0.0_f32, f32::max);

    if max_height > 0.0 {
        let y_min = visible.iter().map(|c| c.bbox[1]).fold(f32::INFINITY, f32::min);
        let y_max = visible.iter().map(|c| c.bbox[3]).fold(f32::NEG_INFINITY, f32::max);
        let vertical_span = y_max - y_min;
        // If the total vertical span exceeds 3× the tallest character, it's multi-line
        if vertical_span > max_height * 3.0 {
            reject!();
        }
    }

    // Merge modifier accents (ˆ U+02C6, etc.) with their overlapping base characters.
    // PDFs store e.g. n̂ as separate 'n' + 'ˆ' chars with overlapping bboxes.
    // We identify which base char each accent belongs to (by horizontal overlap),
    // then skip the accent char and wrap the base in \hat{} during LaTeX assembly.
    let mut accented_bases: std::collections::HashMap<*const ImageChar, &str> = std::collections::HashMap::new();
    let mut accent_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, c) in visible.iter().enumerate() {
        if !is_modifier_accent(c.codepoint) {
            continue;
        }
        // Find the base char with the most horizontal overlap
        let accent_x1 = c.bbox[0];
        let accent_x2 = c.bbox[2];
        let mut best_idx = None;
        let mut best_overlap = 0.0_f32;
        for (j, base) in visible.iter().enumerate() {
            if i == j || is_modifier_accent(base.codepoint) {
                continue;
            }
            let overlap_x1 = accent_x1.max(base.bbox[0]);
            let overlap_x2 = accent_x2.min(base.bbox[2]);
            let overlap = (overlap_x2 - overlap_x1).max(0.0);
            if overlap > best_overlap {
                best_overlap = overlap;
                best_idx = Some(j);
            }
        }
        if let Some(j) = best_idx {
            if best_overlap > 0.0 {
                if let Some(cmd) = accent_command(c.codepoint) {
                    accented_bases.insert(visible[j] as *const ImageChar, cmd);
                    accent_indices.insert(i);
                }
            }
        }
    }

    // Validate every non-accent char maps to a known LaTeX token
    for (i, c) in visible.iter().enumerate() {
        if accent_indices.contains(&i) {
            continue;
        }
        if !is_known_formula_char(c.codepoint) {
            reject!();
        }
    }

    // Build LineElement::Char entries for detect_scripts(), excluding accent chars
    let elements: Vec<LineElement> = visible
        .iter()
        .enumerate()
        .filter(|(i, _)| !accent_indices.contains(i))
        .map(|(_, c)| LineElement::Char(c))
        .collect();

    // Sort left-to-right for line building
    let mut elem_refs: Vec<&LineElement> = elements.iter().collect();
    elem_refs.sort_by(|a, b| {
        a.center_x()
            .partial_cmp(&b.center_x())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Run detect_scripts to fold sub/superscripts
    let processed = detect_scripts(&elem_refs);

    // Assemble final LaTeX string directly (not using build_line_text which adds $..$ wrapping)
    let mut tokens: Vec<String> = Vec::new();
    for elem in &processed {
        let token = match elem {
            LineElement::Char(c) => {
                let base = get_latex_for_char(c.codepoint);
                if let Some(cmd) = accented_bases.get(&(*c as *const ImageChar)) {
                    format!("\\{cmd}{{{base}}}")
                } else {
                    base
                }
            }
            LineElement::Formula { latex: f, .. } => {
                // detect_scripts produces formula elements like "x_{t}" with raw Unicode
                replace_greek_in_latex(f)
            }
        };
        tokens.push(token);
    }

    // Join tokens, inserting spaces where LaTeX requires them.
    // A \command followed by a letter needs a separating space, otherwise
    // LaTeX parses e.g. \alphat as a single undefined command.
    let mut latex = String::new();
    for (i, token) in tokens.iter().enumerate() {
        if i > 0 && is_latex_command(&tokens[i - 1]) && token.starts_with(|c: char| c.is_ascii_alphabetic()) {
            latex.push(' ');
        }
        latex.push_str(token);
    }

    // Final sanity: reject empty results
    let trimmed = latex.trim();
    if trimmed.is_empty() {
        reject!();
    }

    InlineFormulaAttempt {
        latex: Some(trimmed.to_string()),
        adjusted_bbox: bbox,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to make a PdfChar where y is specified in image space (Y-down).
    /// Converts to PDF space internally using page_height_pt.
    /// Uses a default space_threshold of 1.5 (≈ 10pt font × 0.3 ratio / 2).
    fn make_char_image_space(c: char, x: f32, y_top_img: f32, w: f32, h: f32, page_h: f32) -> PdfChar {
        make_char_image_space_italic(c, x, y_top_img, w, h, page_h, false)
    }

    fn make_char_image_space_italic(c: char, x: f32, y_top_img: f32, w: f32, h: f32, page_h: f32, is_italic: bool) -> PdfChar {
        // In image space: y_top_img is the top of the char (small value = near top of page)
        // In PDF space: top = page_h - y_top_img, bottom = page_h - (y_top_img + h)
        let pdf_top = page_h - y_top_img;
        let pdf_bottom = page_h - (y_top_img + h);
        PdfChar {
            codepoint: c,
            bbox: [x, pdf_bottom, x + w, pdf_top],
            space_threshold: 1.5,
            font_name: String::new(),
            font_size: 10.0,
            is_italic,
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
            font_name: String::new(),
            font_size: 10.0,
            is_italic: false,
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
            font_name: String::new(),
            font_size: 10.0,
            is_italic: false,
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

    // ── Italic marker tests ──────────────────────────────────────────

    #[test]
    fn test_italic_run_produces_markers() {
        // "Hello *World*" — World is italic
        let mut chars = Vec::new();
        for (i, c) in "Hello".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // Gap-based space (wide gap > threshold)
        for (i, c) in "World".chars().enumerate() {
            chars.push(make_char_image_space_italic(c, 160.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H, true));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "Hello *World*");
    }

    #[test]
    fn test_italic_whole_line() {
        let mut chars = Vec::new();
        for (i, c) in "AllItalic".chars().enumerate() {
            chars.push(make_char_image_space_italic(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H, true));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "*AllItalic*");
    }

    #[test]
    fn test_no_italic_markers_for_regular_text() {
        let mut chars = Vec::new();
        for (i, c) in "Normal".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "Normal");
    }

    #[test]
    fn test_italic_mixed_with_regular() {
        // "*Heading.* Regular text"
        let mut chars = Vec::new();
        let mut x = 100.0;
        for c in "Heading.".chars() {
            chars.push(make_char_image_space_italic(c, x, 100.0, 10.0, 12.0, PAGE_H, true));
            x += 10.5;
        }
        // Gap-based space
        x += 10.0;
        for c in "Regular".chars() {
            chars.push(make_char_image_space(c, x, 100.0, 10.0, 12.0, PAGE_H));
            x += 10.5;
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "*Heading.* Regular");
    }

    #[test]
    fn test_italic_multi_word_span() {
        // "*Approximating Constraints.* For hard" — italic spans multiple words
        let mut chars = Vec::new();
        let mut x = 100.0;
        for c in "Approximating".chars() {
            chars.push(make_char_image_space_italic(c, x, 100.0, 10.0, 12.0, PAGE_H, true));
            x += 10.5;
        }
        // pdfium zero-width space between italic words
        chars.push(make_char_image_space_italic(' ', x, 100.0, 0.0, 12.0, PAGE_H, true));
        x += 5.0;
        for c in "Constraints.".chars() {
            chars.push(make_char_image_space_italic(c, x, 100.0, 10.0, 12.0, PAGE_H, true));
            x += 10.5;
        }
        // Gap-based space before non-italic text
        x += 10.0;
        for c in "For".chars() {
            chars.push(make_char_image_space(c, x, 100.0, 10.0, 12.0, PAGE_H));
            x += 10.5;
        }
        x += 10.0;
        for c in "hard".chars() {
            chars.push(make_char_image_space(c, x, 100.0, 10.0, 12.0, PAGE_H));
            x += 10.5;
        }
        let bbox = [50.0, 50.0, 800.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "*Approximating Constraints.* For hard");
    }

    #[test]
    fn test_italic_does_not_leak_across_formula() {
        // "*Heading* $x$ regular" — italic closes before formula
        let mut chars = Vec::new();
        let mut x = 100.0;
        for c in "Heading".chars() {
            chars.push(make_char_image_space_italic(c, x, 100.0, 10.0, 12.0, PAGE_H, true));
            x += 10.5;
        }
        // Formula region starting after a gap
        let formula_start = x + 10.0;
        let formula = InlineFormula {
            bbox: [formula_start, 95.0, formula_start + 15.0, 115.0],
            latex: "x".to_string(),
        };
        // Chars under the formula (would be excluded by interleave)
        // Just put regular chars after formula position
        x = formula_start + 25.0;
        for c in "regular".chars() {
            chars.push(make_char_image_space(c, x, 100.0, 10.0, 12.0, PAGE_H));
            x += 10.5;
        }
        let bbox = [50.0, 50.0, 800.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[&formula], AssemblyMode::Reflow);
        assert!(text.contains("*Heading*"), "Expected italic markers around Heading, got: {text}");
        assert!(text.contains("$x$"), "Expected formula, got: {text}");
        // Italic should not leak into formula or regular text
        assert!(!text.contains("*$"), "Italic should not touch formula: {text}");
        assert!(!text.contains("*regular"), "Italic should not leak into regular: {text}");
    }

    #[test]
    fn test_italic_multiple_separate_runs() {
        // "*word1* normal *word2*" — two separate italic runs
        let mut chars = Vec::new();
        let mut x = 100.0;
        for c in "word1".chars() {
            chars.push(make_char_image_space_italic(c, x, 100.0, 10.0, 12.0, PAGE_H, true));
            x += 10.5;
        }
        x += 10.0;
        for c in "normal".chars() {
            chars.push(make_char_image_space(c, x, 100.0, 10.0, 12.0, PAGE_H));
            x += 10.5;
        }
        x += 10.0;
        for c in "word2".chars() {
            chars.push(make_char_image_space_italic(c, x, 100.0, 10.0, 12.0, PAGE_H, true));
            x += 10.5;
        }
        let bbox = [50.0, 50.0, 800.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "*word1* normal *word2*");
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
    fn test_dehyphenation_compound_word_preserved() {
        // "Barrier-augmented": the hyphen is a compound-word hyphen, not a
        // line-break split. "Barrier" starts with uppercase → keep hyphen.
        let mut chars = Vec::new();
        // Line 1: "the Barrier" + STX marker at y=100
        for (i, c) in "the".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // space gap
        for (i, c) in "Barrier".chars().enumerate() {
            chars.push(make_char_image_space(c, 145.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        // STX marker
        chars.push(make_char_image_space('\u{0002}', 218.5, 100.0, 3.0, 12.0, PAGE_H));
        // Line 2: "augmented" at y=130
        for (i, c) in "augmented".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("Barrier-augmented"),
            "compound word hyphen should be preserved, got: {text}"
        );
    }

    #[test]
    fn test_dehyphenation_lowercase_still_joins() {
        // "encoun-tering": lowercase word fragment → join directly.
        let mut chars = Vec::new();
        for (i, c) in "the".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        for (i, c) in "encoun".chars().enumerate() {
            chars.push(make_char_image_space(c, 145.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        chars.push(make_char_image_space('\u{0002}', 208.0, 100.0, 3.0, 12.0, PAGE_H));
        for (i, c) in "tering".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("encountering"),
            "line-break split should be dehyphenated, got: {text}"
        );
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
    fn test_preserve_layout_numbered_lines_at_start() {
        // Numbered algorithm: line numbers should appear at column 0,
        // content indented after a fixed-width number column.
        //   "1 for x"       (number "1" at x=55, content "for" at x=75)
        //   "2   y=0"       (number "2" at x=55, content "y=0" at x=95)
        //   "    // hi"     (no number, comment "//" at x=95, aligned with y=0)
        let mut chars = Vec::new();

        // Line 1: "1" at x=55, then "for x" at x=75
        chars.push(make_char_image_space('1', 55.0, 100.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 100.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "for".chars().enumerate() {
            chars.push(make_char_image_space(c, 75.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        chars.push(make_char_image_space(' ', 105.5, 100.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('x', 110.0, 100.0, 10.0, 12.0, PAGE_H));

        // Line 2: "2" at x=55, then "y=0" at x=95 (indented deeper)
        chars.push(make_char_image_space('2', 55.0, 130.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 130.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "y=0".chars().enumerate() {
            chars.push(make_char_image_space(c, 95.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        // Line 3: "// hi" at x=95 (same X as line 2's content, no line number)
        chars.push(make_char_image_space('/', 95.0, 160.0, 5.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('/', 100.0, 160.0, 5.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 105.0, 160.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "hi".chars().enumerate() {
            chars.push(make_char_image_space(c, 108.0 + i as f32 * 10.5, 160.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        // Line 1: number "1" at start, then content "for x" (least indented → 0 indent)
        assert!(lines[0].starts_with("1 "), "Line 1 should start with number: {:?}", lines[0]);
        assert!(lines[0].contains("for"), "Line 1: {:?}", lines[0]);
        // Line 2: number "2" at start, content "y=0" indented
        assert!(lines[1].starts_with("2 "), "Line 2 should start with number: {:?}", lines[1]);
        assert!(lines[1].contains("y=0"), "Line 2: {:?}", lines[1]);
        // Line 3: no number → spaces for number column, then "//" indented
        assert!(lines[2].starts_with(' '), "Line 3 should start with spaces: {:?}", lines[2]);
        assert!(lines[2].contains("//"), "Line 3: {:?}", lines[2]);
        // Content of lines 2 and 3 should align (same X offset)
        let y_col = lines[1].find("y=0").unwrap();
        let slash_col = lines[2].find("//").unwrap();
        assert_eq!(y_col, slash_col, "Content should align: {:?} vs {:?}", lines[1], lines[2]);
    }

    #[test]
    fn test_preserve_layout_multi_digit_numbers() {
        // Numbers "9" and "10": single-digit should be right-aligned to match "10".
        let mut chars = Vec::new();

        // Line 1: "9" at x=50, content "end" at x=75
        chars.push(make_char_image_space('9', 50.0, 100.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 56.0, 100.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "end".chars().enumerate() {
            chars.push(make_char_image_space(c, 75.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }

        // Line 2: "10" at x=44, content "x=1" at x=75 (same content X → same indent)
        chars.push(make_char_image_space('1', 44.0, 130.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('0', 50.0, 130.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 56.0, 130.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "x=1".chars().enumerate() {
            chars.push(make_char_image_space(c, 75.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [40.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        // " 9" right-aligned to match "10"
        assert!(lines[0].starts_with(" 9"), "Line 1 should right-align: {:?}", lines[0]);
        assert!(lines[1].starts_with("10"), "Line 2 should start with 10: {:?}", lines[1]);
        // Content should be at the same column
        let col1 = lines[0].find("end").unwrap();
        let col2 = lines[1].find("x=1").unwrap();
        assert_eq!(col1, col2, "Content should align: {:?}", lines);
    }

    #[test]
    fn test_preserve_layout_unnumbered_algorithm() {
        // Algorithm without line numbers: indentation from element positions.
        let mut chars = Vec::new();

        // Line 1: "for x" at x=100
        for (i, c) in "for".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }
        chars.push(make_char_image_space(' ', 131.5, 100.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('x', 135.0, 100.0, 10.0, 12.0, PAGE_H));

        // Line 2: "y=0" at x=120 (indented)
        for (i, c) in "y=0".chars().enumerate() {
            chars.push(make_char_image_space(c, 120.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        // Line 3: "end" at x=100 (same as line 1)
        for (i, c) in "end".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 160.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        // Lines 1 and 3: no indent (at min content X)
        assert!(!lines[0].starts_with(' '), "Line 1 no indent: {:?}", lines[0]);
        assert!(!lines[2].starts_with(' '), "Line 3 no indent: {:?}", lines[2]);
        // Line 2: indented
        assert!(lines[1].starts_with(' '), "Line 2 should be indented: {:?}", lines[1]);
        assert!(lines[1].contains("y=0"));
        // No number column: line 1 text starts at column 0
        assert!(lines[0].starts_with("for"), "Should start with content directly: {:?}", lines[0]);
    }

    #[test]
    fn test_preserve_layout_single_digit_not_line_number() {
        // Only 1 line starts with a digit → not treated as numbered algorithm.
        // The digit is part of the content, not a line number.
        let mut chars = Vec::new();

        // Line 1: "3D" at x=100
        chars.push(make_char_image_space('3', 100.0, 100.0, 10.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('D', 110.5, 100.0, 10.0, 12.0, PAGE_H));

        // Line 2: "ok" at x=100
        for (i, c) in "ok".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        // "3D" should be intact — not split into number "3" + content "D"
        assert!(lines[0].starts_with("3D"), "Digit should be content: {:?}", lines[0]);
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

    // ---- Script detection tests ----

    /// Helper to make a PdfChar with a configurable font size.
    /// Like `make_char_image_space` but takes `font_size` as a parameter.
    fn make_char_with_font(
        c: char,
        x: f32,
        y_top_img: f32,
        w: f32,
        h: f32,
        page_h: f32,
        font_size: f32,
    ) -> PdfChar {
        let pdf_top = page_h - y_top_img;
        let pdf_bottom = page_h - (y_top_img + h);
        PdfChar {
            codepoint: c,
            bbox: [x, pdf_bottom, x + w, pdf_top],
            space_threshold: 1.5,
            font_name: String::new(),
            font_size,
            is_italic: false,
        }
    }

    #[test]
    fn test_subscript_detection() {
        // 'a' at 10pt baseline + "ext" at 7.3pt shifted down → $a_{ext}$
        // In Y-down image space, subscript has HIGHER center_y than baseline.
        let chars = vec![
            make_char_with_font('a', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            // Subscript chars: smaller font, shifted down (higher y in image space)
            make_char_with_font('e', 108.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('x', 114.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('t', 120.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("$a_{ext}$"),
            "Expected $a_{{ext}}$ in: {text:?}"
        );
    }

    #[test]
    fn test_superscript_detection() {
        // 'x' at 10pt baseline + '2' at 7.3pt shifted up → $x^{2}$
        // In Y-down image space, superscript has LOWER center_y than baseline.
        // Y shift must be small enough that both chars group on the same line.
        let chars = vec![
            make_char_with_font('x', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            // Superscript: smaller font, shifted up (lower y in image space)
            make_char_with_font('2', 108.0, 98.0, 6.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("$x^{2}$"),
            "Expected $x^{{2}}$ in: {text:?}"
        );
    }

    #[test]
    fn test_no_false_positive_uniform_size() {
        // All chars at the same font size → plain text, no $ wrappers.
        let chars = vec![
            make_char_with_font('a', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('b', 108.5, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('c', 117.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert_eq!(text, "abc");
        assert!(!text.contains('$'), "No formulas expected: {text:?}");
    }

    #[test]
    fn test_script_gap_too_large() {
        // Small-font char far from the base char → no script detection.
        let chars = vec![
            make_char_with_font('a', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            // Far gap: 20pt away (>> space_threshold of 1.5)
            make_char_with_font('x', 128.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(!text.contains('$'), "No formula expected for large gap: {text:?}");
    }

    #[test]
    fn test_multiple_scripts_one_line() {
        // Two separate base+script pairs on the same line.
        // Adjacent formulas with only a space between merge into one $ block.
        let chars = vec![
            // First pair: a_{ext}
            make_char_with_font('a', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('e', 108.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('x', 114.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('t', 120.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
            // Gap between the two pairs (space)
            make_char_with_font(' ', 126.0, 100.0, 0.0, 10.0, PAGE_H, 10.0),
            // Second pair: n_{max}
            make_char_with_font('n', 135.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('m', 143.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('a', 149.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('x', 155.0, 103.0, 6.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("a_{ext}") && text.contains("n_{max}"),
            "Expected both scripts in: {text:?}"
        );
        // Adjacent formulas merge into a single $ block
        assert!(
            text.contains("$a_{ext} n_{max}$"),
            "Expected merged formula in: {text:?}"
        );
    }

    #[test]
    fn test_subscript_in_prose() {
        // Script chars embedded in normal text: "value a_{ext} is used"
        let mut chars = Vec::new();
        // "value " — baseline text
        for (i, c) in "value".chars().enumerate() {
            chars.push(make_char_with_font(c, 100.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }
        chars.push(make_char_with_font(' ', 142.5, 100.0, 0.0, 10.0, PAGE_H, 10.0));
        // "a" base + "ext" subscript
        chars.push(make_char_with_font('a', 150.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        chars.push(make_char_with_font('e', 158.0, 103.0, 6.0, 7.3, PAGE_H, 7.3));
        chars.push(make_char_with_font('x', 164.0, 103.0, 6.0, 7.3, PAGE_H, 7.3));
        chars.push(make_char_with_font('t', 170.0, 103.0, 6.0, 7.3, PAGE_H, 7.3));
        // " is used"
        chars.push(make_char_with_font(' ', 176.0, 100.0, 0.0, 10.0, PAGE_H, 10.0));
        for (i, c) in "is".chars().enumerate() {
            chars.push(make_char_with_font(c, 183.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }
        chars.push(make_char_with_font(' ', 200.0, 100.0, 0.0, 10.0, PAGE_H, 10.0));
        for (i, c) in "used".chars().enumerate() {
            chars.push(make_char_with_font(c, 207.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }

        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("$a_{ext}$"),
            "Expected $a_{{ext}}$ in: {text:?}"
        );
        // Surrounding text should be preserved
        assert!(text.contains("value"), "Expected 'value' in: {text:?}");
        assert!(text.contains("is"), "Expected 'is' in: {text:?}");
        assert!(text.contains("used"), "Expected 'used' in: {text:?}");
    }

    #[test]
    fn test_superscript_single_char() {
        // Common case: h^2
        let chars = vec![
            make_char_with_font('h', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('2', 108.0, 98.0, 6.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("$h^{2}$"),
            "Expected $h^{{2}}$ in: {text:?}"
        );
    }

    #[test]
    fn test_script_detection_preserves_existing_formulas() {
        // When an inline formula is already present, script detection should not interfere.
        let mut chars = Vec::new();
        for (i, c) in "set".chars().enumerate() {
            chars.push(make_char_with_font(c, 100.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }
        // Formula glyph (will be excluded by the formula bbox)
        chars.push(make_char_with_font('x', 140.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        // Continue after formula
        for (i, c) in "to".chars().enumerate() {
            chars.push(make_char_with_font(c, 165.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }

        let bbox = [50.0, 50.0, 600.0, 200.0];
        let formula = InlineFormula {
            bbox: [135.0, 96.0, 155.0, 114.0],
            latex: "x_0".into(),
        };
        let formulas: Vec<&InlineFormula> = vec![&formula];

        let text = extract_region_text(&chars, bbox, PAGE_H, &formulas, AssemblyMode::Reflow);
        assert!(text.contains("$x_0$"), "Expected $x_0$ in: {text:?}");
    }

    #[test]
    fn test_script_detection_only_one_small_char() {
        // Single small char adjacent to a base char should still be detected.
        let chars = vec![
            make_char_with_font('n', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('i', 108.0, 103.0, 4.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("$n_{i}$"),
            "Expected $n_{{i}}$ in: {text:?}"
        );
    }

    #[test]
    fn test_script_with_preserve_layout() {
        // Script detection should also work in PreserveLayout mode.
        let chars = vec![
            make_char_with_font('x', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('2', 108.0, 98.0, 6.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        assert!(
            text.contains("$x^{2}$"),
            "Expected $x^{{2}}$ in PreserveLayout: {text:?}"
        );
    }

    #[test]
    fn test_mixed_super_and_subscript() {
        // Two different script types on one line: x^2 and a_{ext}
        // Y shifts are kept small to ensure all chars group on the same line.
        // Adjacent formulas merge into one $ block.
        let chars = vec![
            // x^2: superscript (shifted up by 1pt)
            make_char_with_font('x', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('2', 108.0, 99.0, 6.0, 7.3, PAGE_H, 7.3),
            // space
            make_char_with_font(' ', 114.0, 100.0, 0.0, 10.0, PAGE_H, 10.0),
            // a_{ext}: subscript (shifted down by 2pt)
            make_char_with_font('a', 125.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('e', 133.0, 102.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('x', 139.0, 102.0, 6.0, 7.3, PAGE_H, 7.3),
            make_char_with_font('t', 145.0, 102.0, 6.0, 7.3, PAGE_H, 7.3),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(
            text.contains("x^{2}") && text.contains("a_{ext}"),
            "Expected both scripts in: {text:?}"
        );
        assert!(
            text.contains("$x^{2} a_{ext}$"),
            "Expected merged formula in: {text:?}"
        );
    }

    #[test]
    fn test_script_detection_skips_space_chars() {
        // Space characters between base and script should not prevent detection.
        // But a literal space char (with codepoint ' ') at baseline size should not
        // become part of a script run.
        let chars = vec![
            make_char_with_font('x', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            // zero-width space (pdfium artifact)
            make_char_with_font(' ', 108.0, 100.0, 0.0, 10.0, PAGE_H, 10.0),
            // normal text continues at baseline size — not a script
            make_char_with_font('y', 112.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(!text.contains('$'), "No formula expected: {text:?}");
    }

    #[test]
    fn test_nearly_same_size_not_detected() {
        // Chars with only a tiny size difference (ratio > 0.85) should NOT be scripts.
        let chars = vec![
            make_char_with_font('a', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0),
            make_char_with_font('b', 108.5, 101.0, 8.0, 9.5, PAGE_H, 9.0), // 9.0/10.0 = 0.9 > 0.85
        ];
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(!text.contains('$'), "No formula for near-equal sizes: {text:?}");
    }

    #[test]
    fn test_adjacent_formulas_merged() {
        // When an inline formula from the model is immediately followed by a
        // script-detected formula, they should merge into a single $...$ block.
        let mut chars = Vec::new();
        // "set" text before
        for (i, c) in "set".chars().enumerate() {
            chars.push(make_char_with_font(c, 100.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }
        // Formula glyph region (chars excluded by formula bbox)
        chars.push(make_char_with_font('y', 140.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        // Script chars right after the formula bbox: subscript "ext"
        chars.push(make_char_with_font('a', 170.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        chars.push(make_char_with_font('e', 178.0, 103.0, 6.0, 7.3, PAGE_H, 7.3));
        chars.push(make_char_with_font('x', 184.0, 103.0, 6.0, 7.3, PAGE_H, 7.3));
        chars.push(make_char_with_font('t', 190.0, 103.0, 6.0, 7.3, PAGE_H, 7.3));

        let bbox = [50.0, 50.0, 600.0, 200.0];
        let formula = InlineFormula {
            bbox: [135.0, 96.0, 155.0, 114.0],
            latex: "h^{2}".into(),
        };
        let formulas: Vec<&InlineFormula> = vec![&formula];

        let text = extract_region_text(&chars, bbox, PAGE_H, &formulas, AssemblyMode::Reflow);
        // The inline formula $h^{2}$ and the detected $a_{ext}$ should merge
        // into a single $h^{2} a_{ext}$ block.
        assert!(
            text.contains("$h^{2} a_{ext}$"),
            "Expected merged $h^{{2}} a_{{ext}}$ in: {text:?}"
        );
        // Should NOT have two separate $ blocks
        assert!(
            !text.contains("$ $"),
            "Should not have adjacent separate formulas: {text:?}"
        );
    }

    #[test]
    fn test_adjacent_formulas_both_inline() {
        // Two inline formulas from the model that are spatially adjacent should merge.
        let mut chars = Vec::new();
        // Some text before
        for (i, c) in "if".chars().enumerate() {
            chars.push(make_char_with_font(c, 100.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }
        // Formula glyphs (excluded)
        chars.push(make_char_with_font('x', 130.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        chars.push(make_char_with_font('y', 160.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        // Text after
        for (i, c) in "then".chars().enumerate() {
            chars.push(make_char_with_font(c, 190.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }

        let bbox = [50.0, 50.0, 600.0, 200.0];
        let f1 = InlineFormula {
            bbox: [125.0, 96.0, 145.0, 114.0],
            latex: "x>0".into(),
        };
        let f2 = InlineFormula {
            bbox: [155.0, 96.0, 175.0, 114.0],
            latex: "\\land y<1".into(),
        };
        let formulas: Vec<&InlineFormula> = vec![&f1, &f2];

        let text = extract_region_text(&chars, bbox, PAGE_H, &formulas, AssemblyMode::Reflow);
        assert!(
            text.contains("$x>0 \\land y<1$"),
            "Expected merged formula in: {text:?}"
        );
    }

    #[test]
    fn test_non_adjacent_formulas_stay_separate() {
        // Two formulas separated by text should NOT be merged.
        let mut chars = Vec::new();
        // Formula glyph 1
        chars.push(make_char_with_font('x', 100.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        // Text between formulas: "and"
        for (i, c) in "and".chars().enumerate() {
            chars.push(make_char_with_font(c, 130.0 + i as f32 * 8.5, 100.0, 8.0, 10.0, PAGE_H, 10.0));
        }
        // Formula glyph 2
        chars.push(make_char_with_font('y', 170.0, 100.0, 8.0, 10.0, PAGE_H, 10.0));

        let bbox = [50.0, 50.0, 600.0, 200.0];
        let f1 = InlineFormula {
            bbox: [95.0, 96.0, 115.0, 114.0],
            latex: "x".into(),
        };
        let f2 = InlineFormula {
            bbox: [165.0, 96.0, 185.0, 114.0],
            latex: "y".into(),
        };
        let formulas: Vec<&InlineFormula> = vec![&f1, &f2];

        let text = extract_region_text(&chars, bbox, PAGE_H, &formulas, AssemblyMode::Reflow);
        // Should have two separate formulas with "and" between
        assert!(
            text.contains("$x$") && text.contains("$y$") && text.contains("and"),
            "Expected separate formulas with text between: {text:?}"
        );
    }

    // ── Inline formula char-based bypass tests ─────────────────────────

    /// Helper: make a PdfChar with explicit font_size, positioned in image-space.
    fn make_formula_char(c: char, x: f32, y_top: f32, w: f32, h: f32, font_size: f32) -> PdfChar {
        let pdf_top = PAGE_H - y_top;
        let pdf_bottom = PAGE_H - (y_top + h);
        PdfChar {
            codepoint: c,
            bbox: [x, pdf_bottom, x + w, pdf_top],
            space_threshold: 1.5,
            font_name: String::new(),
            font_size,
            is_italic: false,
        }
    }

    #[test]
    fn test_bypass_single_variable() {
        let chars = vec![make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x".to_string()));
    }

    #[test]
    fn test_bypass_greek_letter() {
        let chars = vec![make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha".to_string()));
    }

    #[test]
    fn test_bypass_subscript() {
        // 'x' at baseline font size, 't' at smaller font size below
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('t', 108.0, 105.0, 5.0, 7.0, 7.0), // smaller, shifted down
        ];
        let bbox = [95.0, 95.0, 120.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x_{t}".to_string()));
    }

    #[test]
    fn test_bypass_superscript() {
        // 'x' at baseline, '2' smaller and shifted up
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('2', 108.0, 95.0, 5.0, 7.0, 7.0), // smaller, shifted up
        ];
        let bbox = [95.0, 90.0, 120.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x^{2}".to_string()));
    }

    #[test]
    fn test_bypass_simple_expression() {
        // x+y — all same font size, no scripts
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 110.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 120.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 135.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x+y".to_string()));
    }

    #[test]
    fn test_bypass_greek_with_subscript() {
        // α with subscript t
        let chars = vec![
            make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('t', 108.0, 105.0, 5.0, 7.0, 7.0),
        ];
        let bbox = [95.0, 95.0, 120.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha_{t}".to_string()));
    }

    #[test]
    fn test_bypass_unknown_char_rejects() {
        // PUA codepoint — should reject entire formula
        let chars = vec![
            make_formula_char('\u{E000}', 100.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None);
    }

    #[test]
    fn test_bypass_empty_region() {
        let chars: Vec<PdfChar> = vec![];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None);
    }

    #[test]
    fn test_bypass_too_many_chars() {
        // 21 chars should be rejected
        let chars: Vec<PdfChar> = (0..21)
            .map(|i| {
                make_formula_char('a', 100.0 + i as f32 * 10.0, 100.0, 8.0, 10.0, 10.0)
            })
            .collect();
        let bbox = [95.0, 95.0, 400.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None);
    }

    // ── Edge case tests ────────────────────────────────────────────────

    #[test]
    fn test_bypass_exactly_20_chars_passes() {
        // Exactly 20 chars should be accepted (boundary)
        let chars: Vec<PdfChar> = (0..20)
            .map(|i| {
                make_formula_char('a', 100.0 + i as f32 * 10.0, 100.0, 8.0, 10.0, 10.0)
            })
            .collect();
        let bbox = [95.0, 95.0, 400.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "20 chars should be accepted");
        assert_eq!(result.latex.unwrap().len(), 20); // 20 'a's
    }

    #[test]
    fn test_bypass_chars_outside_bbox_ignored() {
        // Char 'x' inside bbox, chars 'A' and 'B' outside — only 'x' should be extracted
        let chars = vec![
            make_formula_char('A', 10.0, 100.0, 8.0, 10.0, 10.0),  // far left, outside
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0), // inside bbox
            make_formula_char('B', 300.0, 100.0, 8.0, 10.0, 10.0), // far right, outside
        ];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x".to_string()));
    }

    #[test]
    fn test_bypass_two_separate_bboxes_from_same_chars() {
        // Simulate two formula regions pulling from the same chars array.
        // Chars at positions: a(50), b(60), +(150), c(160), d(170)
        // Formula 1 bbox covers a,b; Formula 2 bbox covers c,d
        let chars = vec![
            make_formula_char('a', 50.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('b', 60.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('c', 150.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('d', 160.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox1 = [45.0, 95.0, 75.0, 115.0];
        let bbox2 = [145.0, 95.0, 175.0, 115.0];

        let r1 = try_extract_inline_formula(&chars, bbox1, PAGE_H);
        let r2 = try_extract_inline_formula(&chars, bbox2, PAGE_H);
        assert_eq!(r1.latex, Some("ab".to_string()));
        assert_eq!(r2.latex, Some("cd".to_string()));
    }

    #[test]
    fn test_bypass_space_chars_filtered_out() {
        // pdfium may include space chars in the region — they should be filtered
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char(' ', 108.0, 100.0, 4.0, 10.0, 10.0), // space
            make_formula_char('y', 115.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Space is filtered; x and y are adjacent, no script detection → "xy"
        assert_eq!(result.latex, Some("xy".to_string()));
    }

    #[test]
    fn test_bypass_control_chars_filtered_out() {
        // Control chars (e.g., pdfium hyphen marker U+0002) should be filtered
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0002}', 108.0, 100.0, 0.0, 10.0, 10.0), // zero-width control
            make_formula_char('y', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("xy".to_string()));
    }

    #[test]
    fn test_bypass_single_digit() {
        let chars = vec![make_formula_char('0', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("0".to_string()));
    }

    #[test]
    fn test_bypass_parenthesized_expression() {
        // (x+y)
        let chars = vec![
            make_formula_char('(', 100.0, 100.0, 5.0, 10.0, 10.0),
            make_formula_char('x', 105.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 113.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 121.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char(')', 129.0, 100.0, 5.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 140.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("(x+y)".to_string()));
    }

    #[test]
    fn test_bypass_greek_operator_expression() {
        // α+β
        let chars = vec![
            make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{03B2}', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha+\\beta".to_string()));
    }

    #[test]
    fn test_bypass_multi_char_subscript() {
        // x_{12} — base 'x' with two small subscript chars '1' and '2'
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('1', 108.0, 105.0, 5.0, 7.0, 7.0),
            make_formula_char('2', 113.0, 105.0, 5.0, 7.0, 7.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x_{12}".to_string()));
    }

    #[test]
    fn test_bypass_two_subscripted_vars() {
        // a_{i}b_{j} — two base chars each with subscript
        // Need enough gap between 'i' and 'b' so 'b' is recognized as a new base char
        let chars = vec![
            make_formula_char('a', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('i', 108.0, 105.0, 4.0, 7.0, 7.0),  // subscript of a
            make_formula_char('b', 116.0, 100.0, 8.0, 10.0, 10.0), // new base char
            make_formula_char('j', 124.0, 105.0, 4.0, 7.0, 7.0),  // subscript of b
        ];
        let bbox = [95.0, 95.0, 135.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("a_{i}b_{j}".to_string()));
    }

    #[test]
    fn test_bypass_prime_symbol() {
        // x' (x prime)
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2032}', 108.0, 98.0, 3.0, 6.0, 6.0), // prime, small and up
        ];
        let bbox = [95.0, 93.0, 118.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Prime is small font → detect_scripts sees it as superscript
        assert!(result.latex.is_some(), "Prime should be handled");
        let latex = result.latex.unwrap();
        // Either x^{'} or x' depending on detect_scripts behavior
        assert!(
            latex.contains("x") && latex.contains("'"),
            "Should contain x and prime: {latex:?}"
        );
    }

    #[test]
    fn test_bypass_nabla_f() {
        // ∇f
        let chars = vec![
            make_formula_char('\u{2207}', 100.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('f', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\nabla f".to_string()));
    }

    #[test]
    fn test_bypass_partial_derivative() {
        // ∂x
        let chars = vec![
            make_formula_char('\u{2202}', 100.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('x', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // ∂ always goes to OCR now (needs spatial understanding for fractions)
        assert_eq!(result.latex, None);
    }

    #[test]
    fn test_bypass_mixed_unknown_rejects_all() {
        // 'x', unknown PUA char, 'y' — entire formula rejected even though x and y are valid
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{E001}', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "Any unknown char should reject the whole formula");
    }

    #[test]
    fn test_bypass_only_operators() {
        // Just '+' alone
        let chars = vec![make_formula_char('+', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("+".to_string()));
    }

    #[test]
    fn test_bypass_leq_geq_symbols() {
        // x≤y
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2264}', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x\\leq y".to_string()));
    }

    #[test]
    fn test_bypass_multi_line_rejects() {
        // Two chars on very different Y lines — should be rejected
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 100.0, 200.0, 8.0, 10.0, 10.0), // 100pt below
        ];
        let bbox = [95.0, 95.0, 115.0, 215.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "Multi-line should be rejected");
    }

    #[test]
    fn test_bypass_zero_width_chars_only() {
        // Only zero-width chars in the region — should return None
        let chars = vec![
            make_formula_char(' ', 100.0, 100.0, 0.0, 10.0, 10.0),
            make_formula_char('\u{0002}', 100.0, 100.0, 0.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None);
    }

    #[test]
    fn test_bypass_zero_font_size_filtered() {
        // Char with font_size=0 should be filtered out, leaving only 'y'
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 0.0), // zero font size
            make_formula_char('y', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("y".to_string()));
    }

    #[test]
    fn test_bypass_uppercase_greek() {
        // Ω
        let chars = vec![make_formula_char('\u{03A9}', 100.0, 100.0, 10.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\Omega".to_string()));
    }

    #[test]
    fn test_bypass_chars_unsorted_in_array() {
        // Chars appear in reverse order in the array but should be sorted left-to-right
        let chars = vec![
            make_formula_char('y', 120.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 110.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 135.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x+y".to_string()));
    }

    #[test]
    fn test_bypass_bbox_tight_misses_subscript() {
        // bbox only covers 'x', the subscript 't' is outside — should get just "x"
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('t', 108.0, 105.0, 5.0, 7.0, 7.0), // subscript, center outside bbox
        ];
        // Tight bbox: only covers 'x' region
        let bbox = [95.0, 95.0, 107.0, 112.0]; // 't' center at (110.5, 108.5) is outside
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Should get just "x" — partial but not wrong LaTeX
        assert_eq!(result.latex, Some("x".to_string()));
    }

    #[test]
    fn test_bypass_infty_symbol() {
        // ∞
        let chars = vec![make_formula_char('\u{221E}', 100.0, 100.0, 10.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\infty".to_string()));
    }

    // ── LaTeX spacing and more edge cases ──────────────────────────────

    #[test]
    fn test_bypass_greek_followed_by_letter_gets_space() {
        // α followed by t (same font size, no script) → "\alpha t" not "\alphat"
        let chars = vec![
            make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('t', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha t".to_string()));
    }

    #[test]
    fn test_bypass_greek_followed_by_digit_no_space() {
        // α followed by 0 (same font size) → "\alpha0" — no space needed before digit
        // (LaTeX parses \alpha0 correctly since 0 is not a letter)
        let chars = vec![
            make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('0', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Digits don't need space after commands
        assert_eq!(result.latex, Some("\\alpha0".to_string()));
    }

    #[test]
    fn test_bypass_greek_followed_by_operator_no_space() {
        // α+β → "\alpha+\beta" — no space needed between command and operator
        let chars = vec![
            make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{03B2}', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha+\\beta".to_string()));
    }

    #[test]
    fn test_bypass_two_greek_adjacent() {
        // αβ → "\alpha\beta" — no space needed (\ starts next command)
        let chars = vec![
            make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{03B2}', 108.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha\\beta".to_string()));
    }

    #[test]
    fn test_bypass_command_before_subscript_no_space() {
        // α with subscript t → "\alpha_{t}" — the _{} handles separation
        // (This is already tested in test_bypass_greek_with_subscript, but verifying
        // the space logic doesn't interfere with detect_scripts output)
        let chars = vec![
            make_formula_char('\u{03B1}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('t', 108.0, 105.0, 5.0, 7.0, 7.0),
        ];
        let bbox = [95.0, 95.0, 120.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha_{t}".to_string()));
    }

    #[test]
    fn test_bypass_minus_sign_unicode() {
        // x − y (Unicode minus U+2212)
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2212}', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x-y".to_string()));
    }

    #[test]
    fn test_bypass_char_center_on_bbox_boundary() {
        // Char center exactly on bbox edge — should be included (>= check)
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 10.0, 10.0, 10.0),
            // center_x = 105.0, center_y = 105.0
        ];
        // bbox right edge = 105.0, bottom edge = 105.0 — center is exactly on boundary
        let bbox = [100.0, 100.0, 105.0, 105.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x".to_string()));
    }

    #[test]
    fn test_bypass_char_center_just_outside_bbox() {
        // Char center just outside bbox — should NOT be included
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 10.0, 10.0, 10.0),
            // center_x = 105.0, center_y = 105.0
        ];
        // bbox right edge = 104.9 — center is outside
        let bbox = [100.0, 100.0, 104.9, 104.9];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "Char center outside bbox should not be matched");
    }

    #[test]
    fn test_bypass_subscript_gap_too_large() {
        // 'x' and subscript 't' far apart horizontally — detect_scripts should NOT
        // fold them, so we get "xt" instead of "x_{t}"
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('t', 130.0, 105.0, 5.0, 7.0, 7.0), // far away
        ];
        let bbox = [95.0, 95.0, 140.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Gap = 130.0 - 108.0 = 22.0 > space_threshold(1.5), so no script detection
        assert!(result.latex.is_some());
        let latex = result.latex.unwrap();
        // Should be "xt" (no script folding) since gap is too large
        assert_eq!(latex, "xt", "Large gap should prevent script detection");
    }

    #[test]
    fn test_bypass_in_element_in_set() {
        // x∈S
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2208}', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('S', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x\\in S".to_string()));
    }

    #[test]
    fn test_bypass_math_italic_chars() {
        // Mathematical Italic 𝑡 (U+1D461) and 𝑖 (U+1D456) — common in TeX PDFs
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D461}', 108.0, 105.0, 5.0, 7.0, 7.0), // math italic t, subscript
        ];
        let bbox = [95.0, 95.0, 120.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x_{t}".to_string()));
    }

    #[test]
    fn test_bypass_math_italic_expression() {
        // 𝐺𝑖 (U+1D43A U+1D456) — Mathematical Italic G with italic i subscript
        let chars = vec![
            make_formula_char('\u{1D43A}', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D456}', 108.0, 105.0, 5.0, 7.0, 7.0),
        ];
        let bbox = [95.0, 95.0, 120.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("G_{i}".to_string()));
    }

    #[test]
    fn test_bypass_math_italic_greek() {
        // 𝛿 (U+1D6FF) — Mathematical Italic Small Delta
        let chars = vec![make_formula_char('\u{1D6FF}', 100.0, 100.0, 10.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\delta".to_string()));
    }

    #[test]
    fn test_bypass_planck_constant_h() {
        // ℎ (U+210E) — Planck constant (Mathematical Italic h)
        let chars = vec![make_formula_char('\u{210E}', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("h".to_string()));
    }

    #[test]
    fn test_bypass_norm_bars() {
        // ∥x∥ — norm notation
        let chars = vec![
            make_formula_char('\u{2225}', 100.0, 100.0, 5.0, 10.0, 10.0),
            make_formula_char('x', 105.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2225}', 113.0, 100.0, 5.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\|x\\|".to_string()));
    }

    #[test]
    fn test_bypass_double_prime() {
        // x″ (double prime U+2033)
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2033}', 108.0, 98.0, 4.0, 6.0, 6.0),
        ];
        let bbox = [95.0, 93.0, 118.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "Double prime should be handled");
    }

    // ── Math-alphanumeric and spacing edge cases ───────────────────────

    #[test]
    fn test_bypass_math_bold_letter() {
        // 𝐀 (U+1D400) — Mathematical Bold Capital A → "A"
        let chars = vec![make_formula_char('\u{1D400}', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("A".to_string()));
    }

    #[test]
    fn test_bypass_math_sans_serif_letter() {
        // 𝖠 (U+1D5A0) — Mathematical Sans-Serif Capital A → "A"
        let chars = vec![make_formula_char('\u{1D5A0}', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("A".to_string()));
    }

    #[test]
    fn test_bypass_math_italic_subscript_with_greek_and_letter() {
        // x with subscript 𝜖𝑣 → x_{\epsilon v}
        // This tests the replace_greek_in_latex spacing: \epsilon followed by v needs space
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D716}', 108.0, 105.0, 5.0, 7.0, 7.0), // math italic epsilon
            make_formula_char('\u{1D463}', 113.0, 105.0, 4.0, 7.0, 7.0), // math italic v
        ];
        let bbox = [95.0, 95.0, 125.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x_{\\epsilon v}".to_string()));
    }

    #[test]
    fn test_bypass_subscript_two_math_italic_greeks() {
        // x with subscript 𝛿𝜖 → x_{\delta\epsilon}
        // Two commands: no space between them (\d is followed by \, not a letter)
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D6FF}', 108.0, 105.0, 5.0, 7.0, 7.0), // math italic delta
            make_formula_char('\u{1D716}', 113.0, 105.0, 5.0, 7.0, 7.0), // math italic epsilon
        ];
        let bbox = [95.0, 95.0, 125.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x_{\\delta\\epsilon}".to_string()));
    }

    #[test]
    fn test_bypass_math_italic_partial_followed_by_letter() {
        // 𝜕x → \partial x (math italic partial U+1D715 needs space before letter)
        let chars = vec![
            make_formula_char('\u{1D715}', 100.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('x', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // 𝜕 (math italic partial) always goes to OCR now
        assert_eq!(result.latex, None);
    }

    #[test]
    fn test_bypass_math_italic_expression_e_of_x() {
        // 𝐸(𝑥) → E(x) — common in the VBD paper
        let chars = vec![
            make_formula_char('\u{1D438}', 100.0, 100.0, 8.0, 10.0, 10.0), // math italic E
            make_formula_char('(', 108.0, 100.0, 5.0, 10.0, 10.0),
            make_formula_char('\u{1D465}', 113.0, 100.0, 8.0, 10.0, 10.0), // math italic x
            make_formula_char(')', 121.0, 100.0, 5.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 132.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("E(x)".to_string()));
    }

    #[test]
    fn test_bypass_mixed_ascii_and_math_italic_subscript() {
        // F with subscript 𝑖 — F is plain ASCII, 𝑖 is math italic
        let chars = vec![
            make_formula_char('F', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D456}', 108.0, 105.0, 4.0, 7.0, 7.0), // math italic i
        ];
        let bbox = [95.0, 95.0, 118.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("F_{i}".to_string()));
    }

    #[test]
    fn test_bypass_double_struck_r_rejects() {
        // ℝ (U+211D) — double-struck R, not in our map → should reject
        let chars = vec![make_formula_char('\u{211D}', 100.0, 100.0, 10.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "Double-struck R should reject (not in map)");
    }

    #[test]
    fn test_bypass_asterisk_rejects() {
        // * is not in our map → should reject
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('*', 108.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 122.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "Asterisk should reject");
    }

    #[test]
    fn test_bypass_combining_mark_merges() {
        // U+0300 (combining grave accent) overlapping x → \grave{x}
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0300}', 100.0, 94.0, 4.0, 6.0, 5.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\grave{x}".to_string()), "Combining marks should merge with base");
    }

    #[test]
    fn test_bypass_arrow_symbol() {
        // x→y using U+2192
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2192}', 108.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('y', 118.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 132.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x\\to y".to_string()));
    }

    #[test]
    fn test_bypass_forall_x() {
        // ∀x
        let chars = vec![
            make_formula_char('\u{2200}', 100.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('x', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\forall x".to_string()));
    }

    #[test]
    fn test_bypass_ellipsis() {
        // … (U+2026)
        let chars = vec![make_formula_char('\u{2026}', 100.0, 100.0, 12.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 118.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\ldots".to_string()));
    }

    #[test]
    fn test_bypass_colon_in_formula() {
        // f:A — colon is a delimiter, not a command, so no spacing issues
        let chars = vec![
            make_formula_char('f', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char(':', 108.0, 100.0, 4.0, 10.0, 10.0),
            make_formula_char('A', 112.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("f:A".to_string()));
    }

    #[test]
    fn test_bypass_h_in_r_3_from_real_pdf() {
        // Simulates "H_{i}∈R^{3×3}" from real PDF data
        // H is ASCII, 𝑖 is math italic, ∈, R is ASCII, 3 digits, × operator
        let chars = vec![
            make_formula_char('H', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D456}', 108.0, 105.0, 4.0, 7.0, 7.0), // math italic i (sub)
            make_formula_char('\u{2208}', 116.0, 100.0, 8.0, 10.0, 10.0), // ∈
            make_formula_char('R', 124.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('3', 132.0, 96.0, 5.0, 7.0, 7.0),  // superscript
            make_formula_char('\u{00D7}', 137.0, 96.0, 5.0, 7.0, 7.0), // × (same size as 3)
            make_formula_char('3', 142.0, 96.0, 5.0, 7.0, 7.0),  // superscript
        ];
        let bbox = [95.0, 91.0, 152.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "Complex expression should be handled: {:?}", result);
        let latex = result.latex.unwrap();
        // Should contain H_{i}, ∈, R^{...}
        assert!(latex.contains("H_{i}"), "Should have H_{{i}}: {latex}");
        assert!(latex.contains("\\in"), "Should have \\in: {latex}");
        assert!(latex.contains("R"), "Should have R: {latex}");
    }

    #[test]
    fn test_bypass_planck_h_with_comma() {
        // ℎ, — from real PDF (vbd.pdf page 3 formula #21)
        let chars = vec![
            make_formula_char('\u{210E}', 100.0, 100.0, 8.0, 10.0, 10.0), // Planck h
            make_formula_char(',', 108.0, 100.0, 3.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 118.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("h,".to_string()));
    }

    #[test]
    fn test_bypass_norm_with_math_italic() {
        // ∥𝑢∥ — norm bars around math italic u
        let chars = vec![
            make_formula_char('\u{2225}', 100.0, 100.0, 5.0, 10.0, 10.0), // ∥
            make_formula_char('\u{1D462}', 105.0, 100.0, 8.0, 10.0, 10.0), // math italic u
            make_formula_char('\u{2225}', 113.0, 100.0, 5.0, 10.0, 10.0), // ∥
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\|u\\|".to_string()));
    }

    #[test]
    fn test_bypass_delta_x_subscript_c() {
        // 𝛿x𝑐 — math italic delta, plain x, math italic c as subscript
        // Should produce \delta x_{c}
        let chars = vec![
            make_formula_char('\u{1D6FF}', 100.0, 100.0, 8.0, 10.0, 10.0), // math italic delta
            make_formula_char('x', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D450}', 116.0, 105.0, 4.0, 7.0, 7.0), // math italic c (sub)
        ];
        let bbox = [95.0, 95.0, 127.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // \delta is a command, x is a letter → space. Then x_{c} from detect_scripts.
        // But detect_scripts sees x (baseline) followed by c (subscript) → Formula("x_{𝑐}")
        // After replace_greek_in_latex: "x_{c}"
        // And the \delta before it: is_latex_command("\delta") = true, "x" starts with letter → space
        // Result: "\delta x_{c}"
        assert_eq!(result.latex, Some("\\delta x_{c}".to_string()));
    }

    #[test]
    fn test_bypass_v_equals_expression_from_pdf() {
        // v_{i}=(x_{i}-x_{ti})/h — a realistic formula from vbd.pdf
        // v baseline, 𝑖 subscript, =, (, x baseline, 𝑖 subscript, -, x baseline,
        // 𝑡 subscript, 𝑖 subscript (part of same run), ), /, h
        let chars = vec![
            make_formula_char('v', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D456}', 108.0, 105.0, 4.0, 7.0, 7.0), // 𝑖 sub
            make_formula_char('=', 116.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('(', 124.0, 100.0, 5.0, 10.0, 10.0),
            make_formula_char('x', 129.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D456}', 137.0, 105.0, 4.0, 7.0, 7.0), // 𝑖 sub
            make_formula_char('-', 145.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char('x', 151.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{1D461}', 159.0, 105.0, 4.0, 7.0, 7.0), // 𝑡 sub
            make_formula_char('\u{1D456}', 163.0, 105.0, 4.0, 7.0, 7.0), // 𝑖 sub (same run)
            make_formula_char(')', 170.0, 100.0, 5.0, 10.0, 10.0),
            make_formula_char('/', 175.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char('\u{210E}', 181.0, 100.0, 8.0, 10.0, 10.0), // Planck h
        ];
        let bbox = [95.0, 95.0, 195.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "Should handle realistic formula");
        let latex = result.latex.unwrap();
        assert_eq!(latex, "v_{i}=(x_{i}-x_{ti})/h");
    }

    #[test]
    fn test_line_number_colon_separator_stripped() {
        // Algorithm format "4: x ← ..." — the colon after the line number
        // should be stripped so indentation is measured from the content after
        // the colon, not from the colon itself.
        let mut chars = Vec::new();

        // Line 1: "1: for x" at (number at x=55, colon at x=65, content at x=75)
        chars.push(make_char_image_space('1', 55.0, 100.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 100.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(':', 65.0, 100.0, 4.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 69.0, 100.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "for".chars().enumerate() {
            chars.push(make_char_image_space(c, 75.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }

        // Line 2: "2:   y=0" at (number at x=55, colon at x=65, content at x=95 — indented)
        chars.push(make_char_image_space('2', 55.0, 130.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 130.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(':', 65.0, 130.0, 4.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 69.0, 130.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "y=0".chars().enumerate() {
            chars.push(make_char_image_space(c, 95.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "Should have 2 lines: {text:?}");
        // Colon should NOT appear in output
        assert!(
            !lines[0].contains(':'),
            "Colon should be stripped from line 1: {:?}",
            lines[0]
        );
        assert!(
            !lines[1].contains(':'),
            "Colon should be stripped from line 2: {:?}",
            lines[1]
        );
        // Line 2 should have indentation (content at x=95 vs x=75)
        let for_col = lines[0].find("for").unwrap();
        let y_col = lines[1].find("y=0").unwrap();
        assert!(
            y_col > for_col,
            "Line 2 should be indented deeper than line 1: {:?} vs {:?}",
            lines[0],
            lines[1]
        );
    }

    #[test]
    fn test_orphan_arrow_merged_into_line() {
        // A stray '←' at a slightly different Y should merge back into the
        // nearest line rather than forming its own output line.
        let mut chars = Vec::new();

        // Line 1: "1: x" at y=100
        chars.push(make_char_image_space('1', 55.0, 100.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 100.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(':', 65.0, 100.0, 4.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 69.0, 100.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('x', 75.0, 100.0, 10.0, 12.0, PAGE_H));

        // Stray '←' at slightly different Y (y=104, within avg_height but outside 0.5*avg_height)
        chars.push(make_char_image_space('←', 90.0, 104.0, 10.0, 12.0, PAGE_H));

        // "init" continues on line 1's Y
        for (i, c) in "init".chars().enumerate() {
            chars.push(make_char_image_space(c, 105.0 + i as f32 * 10.5, 100.0, 10.0, 12.0, PAGE_H));
        }

        // Line 2: "2: end" at y=130
        chars.push(make_char_image_space('2', 55.0, 130.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 130.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(':', 65.0, 130.0, 4.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 69.0, 130.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "end".chars().enumerate() {
            chars.push(make_char_image_space(c, 75.0 + i as f32 * 10.5, 130.0, 10.0, 12.0, PAGE_H));
        }

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "Arrow should merge, not create extra line: {text:?}");
        assert!(
            lines[0].contains('←'),
            "Line 1 should contain the arrow: {:?}",
            lines[0]
        );
    }

    #[test]
    fn test_unnumbered_formula_continuation_merged() {
        // When a Formula element lands on a different Y band in a numbered
        // algorithm, it should be merged into the preceding numbered line
        // rather than creating a separate unnumbered output line.
        let mut chars = Vec::new();

        // Line 1: "1: x" at y=100
        chars.push(make_char_image_space('1', 55.0, 100.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 100.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(':', 65.0, 100.0, 4.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 69.0, 100.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('x', 75.0, 100.0, 10.0, 12.0, PAGE_H));

        // Line 2: "2: end" at y=160 (well-separated)
        chars.push(make_char_image_space('2', 55.0, 160.0, 6.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 61.0, 160.0, 0.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(':', 65.0, 160.0, 4.0, 12.0, PAGE_H));
        chars.push(make_char_image_space(' ', 69.0, 160.0, 0.0, 12.0, PAGE_H));
        for (i, c) in "end".chars().enumerate() {
            chars.push(make_char_image_space(c, 75.0 + i as f32 * 10.5, 160.0, 10.0, 12.0, PAGE_H));
        }

        // InlineFormula at y=130 (between the two text lines — different Y band)
        let formula = InlineFormula {
            bbox: [85.0, 126.0, 150.0, 142.0], // center y ≈ 134 (image space)
            latex: "f(x)".into(),
        };
        let formulas: Vec<&InlineFormula> = vec![&formula];

        let bbox = [50.0, 50.0, 600.0, 300.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &formulas, AssemblyMode::PreserveLayout);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines.len(),
            2,
            "Formula should merge into line 1, not create extra line: {text:?}"
        );
        assert!(
            lines[0].contains("$f(x)$"),
            "Line 1 should contain the formula: {:?}",
            lines[0]
        );
    }

    // --- Rejection rule tests: chars that force OCR ---

    #[test]
    fn test_reject_fraction_crlf() {
        // PDF encodes fraction bars as \r\n control chars with zero-width bboxes.
        // numerator chars, then \r\n (fraction bar), then denominator chars.
        let chars = vec![
            make_formula_char('E', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\r', 108.0, 100.0, 0.0, 0.0, 10.0),
            make_formula_char('\n', 108.0, 100.0, 0.0, 0.0, 10.0),
            make_formula_char('x', 108.0, 110.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 125.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "CR/LF fraction chars should force OCR");
    }

    #[test]
    fn test_reject_partial_u2202() {
        // ∂ (U+2202) always goes to OCR
        let chars = vec![
            make_formula_char('\u{2202}', 100.0, 100.0, 10.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "∂ should always force OCR");
    }

    #[test]
    fn test_reject_math_italic_partial_u1d715() {
        // 𝜕 (U+1D715) always goes to OCR
        let chars = vec![
            make_formula_char('\u{1D715}', 100.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('f', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "𝜕 (math italic partial) should always force OCR");
    }

    #[test]
    fn test_reject_nabla_complex_expression() {
        // ∇²f — nabla with >2 visible chars goes to OCR
        let chars = vec![
            make_formula_char('\u{2207}', 100.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('\u{00B2}', 110.0, 96.0, 6.0, 7.0, 10.0), // superscript 2
            make_formula_char('f', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 90.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, None, "∇ with >2 chars should force OCR");
    }

    #[test]
    fn test_allow_nabla_simple() {
        // ∇f — nabla with exactly 2 chars stays on fast path
        let chars = vec![
            make_formula_char('\u{2207}', 100.0, 100.0, 10.0, 10.0, 10.0),
            make_formula_char('f', 110.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\nabla f".to_string()), "∇f should stay on fast path");
    }

    // --- Modifier accent (hat) merging tests ---
    //
    // PDFs store accented chars like n̂ as separate 'n' + 'ˆ' (U+02C6) characters
    // with overlapping bounding boxes. We need to merge them into \hat{n}.

    #[test]
    fn test_hat_over_n() {
        // n̂ — circumflex overlaps with n
        let chars = vec![
            make_formula_char('n', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02C6}', 101.0, 94.0, 6.0, 6.0, 10.0), // ˆ above n
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\hat{n}".to_string()));
    }

    #[test]
    fn test_hat_over_b() {
        // b̂ — circumflex overlaps with b (from the avbd paper: ˆb)
        let chars = vec![
            make_formula_char('\u{02C6}', 101.0, 94.0, 6.0, 6.0, 10.0), // ˆ comes first in PDF
            make_formula_char('b', 100.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\hat{b}".to_string()));
    }

    #[test]
    fn test_hat_n_hat_t_comma_separated() {
        // n̂, t̂ — two hatted chars with commas (from avbd p4_44)
        let chars = vec![
            make_formula_char('n', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02C6}', 101.0, 94.0, 6.0, 6.0, 10.0),
            make_formula_char(',', 109.0, 100.0, 3.0, 10.0, 10.0),
            make_formula_char('\u{02C6}', 115.0, 94.0, 6.0, 6.0, 10.0),
            make_formula_char('t', 114.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char(',', 121.0, 100.0, 3.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\hat{n},\\hat{t},".to_string()));
    }

    #[test]
    fn test_hat_not_overlapping() {
        // ˆ that doesn't overlap any base char — should produce \hat{} as before
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02C6}', 130.0, 94.0, 6.0, 6.0, 10.0), // far away from x
        ];
        let bbox = [95.0, 89.0, 145.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // No overlap: ˆ stays as standalone \hat{}
        assert_eq!(result.latex, Some("x\\hat{}".to_string()));
    }

    #[test]
    fn test_hat_over_greek() {
        // α̂ — hat over a Greek letter
        let chars = vec![
            make_formula_char('\u{1D6FC}', 100.0, 100.0, 8.0, 10.0, 10.0), // math italic alpha
            make_formula_char('\u{02C6}', 101.0, 94.0, 6.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\hat{\\alpha}".to_string()));
    }

    #[test]
    fn test_hat_b_comma() {
        // b̂, — hatted b followed by comma (from avbd p4_53)
        let chars = vec![
            make_formula_char('\u{02C6}', 101.0, 94.0, 6.0, 6.0, 10.0),
            make_formula_char('b', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char(',', 109.0, 100.0, 3.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 118.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\hat{b},".to_string()));
    }

    #[test]
    fn test_hat_between_other_chars() {
        // x + n̂ + y — hat in the middle of an expression
        let chars = vec![
            make_formula_char('x', 80.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 92.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('n', 104.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02C6}', 105.0, 94.0, 6.0, 6.0, 10.0),
            make_formula_char('+', 116.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 128.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [75.0, 89.0, 142.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x+\\hat{n}+y".to_string()));
    }

    #[test]
    fn test_hat_combining_circumflex() {
        // Combining circumflex U+0302 (zero-width, attached to preceding char)
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0302}', 101.0, 94.0, 6.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\hat{x}".to_string()));
    }

    #[test]
    fn test_tilde_modifier() {
        // ˜ (U+02DC) modifier tilde over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02DC}', 101.0, 94.0, 6.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\tilde{x}".to_string()));
    }

    #[test]
    fn test_tilde_combining() {
        // Combining tilde U+0303
        let chars = vec![
            make_formula_char('n', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0303}', 101.0, 94.0, 6.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\tilde{n}".to_string()));
    }

    #[test]
    fn test_bar_modifier() {
        // ¯ (U+00AF) macron over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{00AF}', 100.0, 94.0, 8.0, 3.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\bar{x}".to_string()));
    }

    #[test]
    fn test_bar_combining() {
        // Combining macron U+0304
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0304}', 100.0, 94.0, 8.0, 3.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\bar{x}".to_string()));
    }

    #[test]
    fn test_dot_modifier() {
        // ˙ (U+02D9) dot above over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02D9}', 102.0, 94.0, 4.0, 4.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\dot{x}".to_string()));
    }

    #[test]
    fn test_dot_combining() {
        // Combining dot above U+0307
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0307}', 102.0, 94.0, 4.0, 4.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\dot{x}".to_string()));
    }

    #[test]
    fn test_ddot_modifier() {
        // ¨ (U+00A8) diaeresis over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{00A8}', 101.0, 94.0, 6.0, 4.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\ddot{x}".to_string()));
    }

    #[test]
    fn test_ddot_combining() {
        // Combining diaeresis U+0308
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0308}', 101.0, 94.0, 6.0, 4.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\ddot{x}".to_string()));
    }

    #[test]
    fn test_check_modifier() {
        // ˇ (U+02C7) caron over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02C7}', 101.0, 94.0, 6.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\check{x}".to_string()));
    }

    #[test]
    fn test_breve_modifier() {
        // ˘ (U+02D8) breve over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{02D8}', 101.0, 94.0, 6.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\breve{x}".to_string()));
    }

    #[test]
    fn test_acute_modifier() {
        // ´ (U+00B4) acute over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{00B4}', 104.0, 94.0, 4.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\acute{x}".to_string()));
    }

    #[test]
    fn test_grave_modifier() {
        // ` (U+0060) grave over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{0060}', 100.0, 94.0, 4.0, 6.0, 10.0),
        ];
        let bbox = [95.0, 89.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\grave{x}".to_string()));
    }

    #[test]
    fn test_vec_combining() {
        // Combining right arrow U+20D7 over x
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{20D7}', 100.0, 93.0, 8.0, 5.0, 10.0),
        ];
        let bbox = [95.0, 88.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\vec{x}".to_string()));
    }

    // --- Bracket expansion tests ---

    #[test]
    fn test_expand_pipe_left() {
        // |det(G)| < ε — bbox clips the left |, expansion should recover it
        // Chars: | d e t ( G ) | < ε
        let chars = vec![
            make_formula_char('|', 80.0, 100.0, 3.0, 16.0, 10.0),
            make_formula_char('d', 84.0, 100.0, 7.0, 10.0, 10.0),
            make_formula_char('e', 91.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char('t', 97.0, 100.0, 5.0, 10.0, 10.0),
            make_formula_char('(', 102.0, 100.0, 4.0, 16.0, 10.0),
            make_formula_char('G', 106.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char(')', 114.0, 100.0, 4.0, 16.0, 10.0),
            make_formula_char('|', 118.0, 100.0, 3.0, 16.0, 10.0),
            make_formula_char('<', 124.0, 100.0, 7.0, 10.0, 10.0),
        ];
        // Bbox starts at x=90 — clips | at x=80 and d at x=84
        let bbox = [90.0, 95.0, 135.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "should expand left to find matching |");
        let latex = result.latex.unwrap();
        assert!(latex.contains("det"), "should include 'det': got {latex}");
        assert!(latex.starts_with('|'), "should start with |: got {latex}");
    }

    #[test]
    fn test_expand_paren_left() {
        // (x+1) — bbox clips the opening (
        let chars = vec![
            make_formula_char('(', 80.0, 100.0, 4.0, 16.0, 10.0),
            make_formula_char('x', 85.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 94.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('1', 103.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char(')', 110.0, 100.0, 4.0, 16.0, 10.0),
        ];
        // Bbox starts at x=83 — clips ( at x=80
        let bbox = [83.0, 95.0, 118.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "should expand left to find matching (");
        let latex = result.latex.unwrap();
        assert!(latex.starts_with('('), "should start with (: got {latex}");
        assert!(latex.ends_with(')'), "should end with ): got {latex}");
    }

    #[test]
    fn test_expand_paren_right() {
        // (x+1) — bbox clips the closing )
        let chars = vec![
            make_formula_char('(', 80.0, 100.0, 4.0, 16.0, 10.0),
            make_formula_char('x', 85.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 94.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('1', 103.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char(')', 110.0, 100.0, 4.0, 16.0, 10.0),
        ];
        // Bbox ends at x=109 — clips ) at x=110
        let bbox = [75.0, 95.0, 109.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "should expand right to find matching )");
        let latex = result.latex.unwrap();
        assert!(latex.ends_with(')'), "should end with ): got {latex}");
    }

    #[test]
    fn test_expand_no_false_match_other_line() {
        // | on a different line should NOT be matched
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('|', 108.0, 100.0, 3.0, 16.0, 10.0),
            // | on a completely different line (y=200)
            make_formula_char('|', 80.0, 200.0, 3.0, 16.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 115.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Should NOT expand to the other-line |, so | count stays odd → 2 chars, just "x|"
        assert!(result.latex.is_some());
        let latex = result.latex.unwrap();
        assert_eq!(latex, "x|");
    }

    #[test]
    fn test_expand_balanced_no_change() {
        // (x+1) fully inside bbox — no expansion needed
        let chars = vec![
            make_formula_char('(', 100.0, 100.0, 4.0, 16.0, 10.0),
            make_formula_char('x', 105.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char(')', 114.0, 100.0, 4.0, 16.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 125.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("(x)".to_string()));
    }

    #[test]
    fn test_expand_square_brackets() {
        // [0,1] — bbox clips the opening [
        let chars = vec![
            make_formula_char('[', 80.0, 100.0, 4.0, 16.0, 10.0),
            make_formula_char('0', 85.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char(',', 92.0, 100.0, 3.0, 10.0, 10.0),
            make_formula_char('1', 96.0, 100.0, 6.0, 10.0, 10.0),
            make_formula_char(']', 103.0, 100.0, 4.0, 16.0, 10.0),
        ];
        // Bbox starts at x=83 — clips [ at x=80
        let bbox = [83.0, 95.0, 112.0, 120.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert!(result.latex.is_some(), "should expand left to find matching [");
        let latex = result.latex.unwrap();
        assert!(latex.starts_with('['), "should start with [: got {latex}");
    }

    // --- Text-region accent merging tests ---
    // These test that modifier accents in regular text (not formula regions)
    // are merged with their base chars and emitted as $\hat{x}$ etc.

    #[test]
    fn test_text_accent_hat_before_base() {
        // ˆb in text → $\hat{b}$
        let mut chars = Vec::new();
        // "the " at y=100
        for (i, c) in "the ".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        // ˆ overlapping with b (both at x=140)
        chars.push(make_char_image_space('\u{02C6}', 140.0, 95.0, 8.0, 5.0, PAGE_H));
        chars.push(make_char_image_space('b', 140.0, 100.0, 8.0, 12.0, PAGE_H));
        // " end"
        for (i, c) in " end".chars().enumerate() {
            chars.push(make_char_image_space(c, 150.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(text.contains("$\\hat{b}$"), "expected $\\hat{{b}}$ in: {text}");
        // Should NOT contain raw ˆ
        assert!(!text.contains('\u{02C6}'), "should not contain raw ˆ in: {text}");
    }

    #[test]
    fn test_text_accent_hat_after_base() {
        // nˆ in text → $\hat{n}$
        let mut chars = Vec::new();
        for (i, c) in "the ".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        // n then ˆ overlapping
        chars.push(make_char_image_space('n', 140.0, 100.0, 8.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('\u{02C6}', 140.0, 95.0, 8.0, 5.0, PAGE_H));
        for (i, c) in " end".chars().enumerate() {
            chars.push(make_char_image_space(c, 150.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(text.contains("$\\hat{n}$"), "expected $\\hat{{n}}$ in: {text}");
        assert!(!text.contains('\u{02C6}'), "should not contain raw ˆ in: {text}");
    }

    #[test]
    fn test_text_accent_tilde_in_prose() {
        // ˜x in text → $\tilde{x}$
        let mut chars = Vec::new();
        for (i, c) in "the ".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        chars.push(make_char_image_space('\u{02DC}', 140.0, 95.0, 8.0, 5.0, PAGE_H));
        chars.push(make_char_image_space('x', 140.0, 100.0, 8.0, 12.0, PAGE_H));
        for (i, c) in " end".chars().enumerate() {
            chars.push(make_char_image_space(c, 150.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(text.contains("$\\tilde{x}$"), "expected $\\tilde{{x}}$ in: {text}");
    }

    #[test]
    fn test_text_accent_no_overlap_no_merge() {
        // ˆ far from any base char — should appear raw (no merge)
        let mut chars = Vec::new();
        for (i, c) in "the ".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        // ˆ at x=200, base 'b' at x=140 — no overlap
        chars.push(make_char_image_space('\u{02C6}', 200.0, 95.0, 8.0, 5.0, PAGE_H));
        chars.push(make_char_image_space('b', 140.0, 100.0, 8.0, 12.0, PAGE_H));
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        // No merge — both appear separately
        assert!(!text.contains("$\\hat{b}$"), "should NOT merge without overlap: {text}");
    }

    #[test]
    fn test_text_accent_combining_circumflex() {
        // U+0302 combining circumflex overlapping 'n' → $\hat{n}$
        let mut chars = Vec::new();
        for (i, c) in "the ".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        chars.push(make_char_image_space('n', 140.0, 100.0, 8.0, 12.0, PAGE_H));
        // Combining marks are zero-width, positioned at same x
        chars.push(make_char_image_space('\u{0302}', 140.0, 95.0, 8.0, 5.0, PAGE_H));
        for (i, c) in " end".chars().enumerate() {
            chars.push(make_char_image_space(c, 150.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(text.contains("$\\hat{n}$"), "expected $\\hat{{n}}$ in: {text}");
    }

    #[test]
    fn test_text_accent_vec_arrow() {
        // U+20D7 combining right arrow over 'v' → $\vec{v}$
        let mut chars = Vec::new();
        for (i, c) in "the ".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        chars.push(make_char_image_space('v', 140.0, 100.0, 8.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('\u{20D7}', 140.0, 95.0, 8.0, 5.0, PAGE_H));
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        assert!(text.contains("$\\vec{v}$"), "expected $\\vec{{v}}$ in: {text}");
    }

    #[test]
    fn test_text_accent_closes_italic() {
        // Italic text followed by accented char should close italic before $\hat{}$
        let mut chars = Vec::new();
        // "the " normal
        for (i, c) in "the ".chars().enumerate() {
            chars.push(make_char_image_space(c, 100.0 + i as f32 * 10.0, 100.0, 10.0, 12.0, PAGE_H));
        }
        // italic 'x'
        chars.push(make_char_image_space_italic('x', 140.0, 100.0, 8.0, 12.0, PAGE_H, true));
        // ˆ overlapping 'n' (not italic)
        chars.push(make_char_image_space('n', 160.0, 100.0, 8.0, 12.0, PAGE_H));
        chars.push(make_char_image_space('\u{02C6}', 160.0, 95.0, 8.0, 5.0, PAGE_H));
        let bbox = [50.0, 50.0, 600.0, 200.0];
        let text = extract_region_text(&chars, bbox, PAGE_H, &[], AssemblyMode::Reflow);
        // Should have *x* then $\hat{n}$, not *x$\hat{n}$*
        assert!(text.contains("$\\hat{n}$"), "expected $\\hat{{n}}$ in: {text}");
        // Italic x should be closed before the hat
        assert!(!text.contains("*$"), "italic should close before $ in: {text}");
    }

    // ---------------------------------------------------------------
    // trim_stray_edge_chars tests
    //
    // These verify that stray text characters accidentally captured by
    // the layout bbox are trimmed, while legitimate formula content
    // (even with spacing) is preserved.
    // ---------------------------------------------------------------

    #[test]
    fn test_trim_stray_left_char_in_a_3x3() {
        // "a 3×3" — 'a' is body text captured by the formula bbox.
        // Chars: a(100-108), 3(120-128), ×(128-136), 3(136-144)
        // Gap a→3 = 12pt, median_width = 8pt → trim 'a'
        let chars = vec![
            make_formula_char('a', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('3', 120.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{00D7}', 128.0, 100.0, 8.0, 10.0, 10.0), // ×
            make_formula_char('3', 136.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 150.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("3\\times3".to_string()));
        // Bbox should be tightened to exclude 'a'
        assert!(result.adjusted_bbox[0] > 108.0, "left edge should be tightened past 'a'");
    }

    #[test]
    fn test_trim_stray_right_char() {
        // "3×3 b" — 'b' is body text captured on the right.
        // Chars: 3(100-108), ×(108-116), 3(116-124), b(136-144)
        // Gap 3→b = 12pt, median_width = 8pt → trim 'b'
        let chars = vec![
            make_formula_char('3', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{00D7}', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('3', 116.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('b', 136.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 150.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("3\\times3".to_string()));
        // Bbox should be tightened to exclude 'b'
        assert!(result.adjusted_bbox[2] < 136.0, "right edge should be tightened before 'b'");
    }

    #[test]
    fn test_trim_stray_both_edges() {
        // "a 3×3 b" — stray chars on both sides.
        // Chars: a(80-88), 3(100-108), ×(108-116), 3(116-124), b(136-144)
        let chars = vec![
            make_formula_char('a', 80.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('3', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{00D7}', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('3', 116.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('b', 136.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [75.0, 95.0, 150.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("3\\times3".to_string()));
        assert!(result.adjusted_bbox[0] > 88.0, "left edge tightened past 'a'");
        assert!(result.adjusted_bbox[2] < 136.0, "right edge tightened before 'b'");
    }

    #[test]
    fn test_trim_no_trim_when_only_two_chars() {
        // Two chars with a gap — can't tell which is stray, so don't trim either.
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 120.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 135.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("xy".to_string()));
    }

    #[test]
    fn test_trim_no_trim_when_all_chars_close() {
        // "xyz" — all chars adjacent, no stray.
        // Gaps: x→y = 0pt, y→z = 0pt. No trimming.
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 108.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('z', 116.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("xyz".to_string()));
    }

    #[test]
    fn test_trim_no_trim_when_evenly_spaced() {
        // "x + y" — even spacing around operator, no outlier gap.
        // x(100-108), +(114-122), y(128-136) — gaps are 6pt each, median_width=8pt
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 114.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('y', 128.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 142.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x+y".to_string()));
    }

    #[test]
    fn test_trim_preserves_subscript_with_small_gap() {
        // "x_t" — subscript t is smaller and shifted, but gap < median_width.
        // x(100-108), t(110-115) — gap = 2pt, median_width ≈ 6.5pt → no trim
        let chars = vec![
            make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('t', 110.0, 105.0, 5.0, 7.0, 7.0),
            make_formula_char('y', 118.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 130.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // All 3 chars preserved (t is subscript of x)
        assert!(result.latex.is_some());
        let latex = result.latex.unwrap();
        assert!(latex.contains("x"), "should contain x");
        assert!(latex.contains("t"), "should contain t");
        assert!(latex.contains("y"), "should contain y");
    }

    #[test]
    fn test_trim_bbox_midpoint_placement() {
        // Verify the tightened bbox left edge is at the midpoint between
        // the stray char's right edge and the next char's left edge.
        // stray 'a'(100-108), formula '3'(120-128), '+'(128-136), '5'(136-144)
        let chars = vec![
            make_formula_char('a', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('3', 120.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('+', 128.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('5', 136.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 150.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Midpoint between stray right (108) and next left (120) = 114
        let expected_left = (108.0 + 120.0) / 2.0;
        assert!(
            (result.adjusted_bbox[0] - expected_left).abs() < 0.1,
            "left edge should be at midpoint {expected_left}, got {}",
            result.adjusted_bbox[0]
        );
    }

    #[test]
    fn test_trim_single_char_formula_not_trimmed() {
        // Single char formula — only 1 visible char, no trimming possible.
        let chars = vec![make_formula_char('x', 100.0, 100.0, 8.0, 10.0, 10.0)];
        let bbox = [95.0, 95.0, 115.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("x".to_string()));
    }

    #[test]
    fn test_trim_stray_char_with_tight_formula() {
        // "a αβ" — stray 'a' before tightly-packed Greek formula.
        // a(100-108), α(120-128), β(128-136) — gap a→α = 12 > median_width 8 → trim
        let chars = vec![
            make_formula_char('a', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{03B1}', 120.0, 100.0, 8.0, 10.0, 10.0), // α
            make_formula_char('\u{03B2}', 128.0, 100.0, 8.0, 10.0, 10.0), // β
        ];
        let bbox = [95.0, 95.0, 142.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("\\alpha\\beta".to_string()));
    }

    #[test]
    fn test_trim_adjusted_bbox_used_for_ocr_fallback() {
        // When char-based extraction fails (unknown char), the adjusted_bbox
        // should still be tightened from stray edge chars.
        // stray 'a'(100-108), unknown(120-128), '+'(128-136), unknown(136-144)
        let chars = vec![
            make_formula_char('a', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2603}', 120.0, 100.0, 8.0, 10.0, 10.0), // snowman - unknown
            make_formula_char('+', 128.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('\u{2603}', 136.0, 100.0, 8.0, 10.0, 10.0), // snowman - unknown
        ];
        let bbox = [95.0, 95.0, 150.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        // Char-based fails (unknown chars), but bbox should still be tightened
        assert!(result.latex.is_none(), "should fail char-based extraction");
        assert!(
            result.adjusted_bbox[0] > 108.0,
            "bbox should be tightened even when extraction fails, got left={}",
            result.adjusted_bbox[0]
        );
    }

    #[test]
    fn test_trim_gap_exactly_at_threshold_no_trim() {
        // Edge gap exactly equals median_width — should NOT trim (need strictly greater).
        // a(100-108), b(116-124), c(124-132) — gap a→b = 8 = median_width(8). No trim.
        let chars = vec![
            make_formula_char('a', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('b', 116.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('c', 124.0, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 138.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("abc".to_string()));
    }

    #[test]
    fn test_trim_gap_just_above_threshold_trims() {
        // Edge gap just above median_width → trim.
        // a(100-108), b(116.5-124.5), c(124.5-132.5) — gap a→b = 8.5 > median_width(8) → trim
        let chars = vec![
            make_formula_char('a', 100.0, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('b', 116.5, 100.0, 8.0, 10.0, 10.0),
            make_formula_char('c', 124.5, 100.0, 8.0, 10.0, 10.0),
        ];
        let bbox = [95.0, 95.0, 138.0, 115.0];
        let result = try_extract_inline_formula(&chars, bbox, PAGE_H);
        assert_eq!(result.latex, Some("bc".to_string()));
    }
}
