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
        let processed = detect_scripts(line);
        let processed_refs: Vec<&LineElement> = processed.iter().collect();
        let line_text = build_line_text(&processed_refs);
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
            formatted.push_str(&build_line_text(&processed_refs));
        } else {
            // No line numbers: indent + full line text.
            for _ in 0..indent_count {
                formatted.push(' ');
            }
            let processed = detect_scripts(line);
            let processed_refs: Vec<&LineElement> = processed.iter().collect();
            formatted.push_str(&build_line_text(&processed_refs));
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
            font_name: String::new(),
            font_size: 10.0,
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
        // Two separate base+script pairs on the same line: "a_{ext} n_{max}"
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
            text.contains("$a_{ext}$"),
            "Expected $a_{{ext}}$ in: {text:?}"
        );
        assert!(
            text.contains("$n_{max}$"),
            "Expected $n_{{max}}$ in: {text:?}"
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
            text.contains("$x^{2}$"),
            "Expected $x^{{2}}$ in: {text:?}"
        );
        assert!(
            text.contains("$a_{ext}$"),
            "Expected $a_{{ext}}$ in: {text:?}"
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
}
