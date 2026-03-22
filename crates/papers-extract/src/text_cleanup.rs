//! Shared text post-processing utilities used by both the layout pipeline
//! and the text-only extraction path.

/// Detect drop caps masquerading as formulas.
///
/// Pattern: a single uppercase letter followed by `^{...}` or `_{...}` where
/// the braced content is 50+ chars of prose (has spaces, mostly alphabetic).
/// Returns the expanded plain text if detected.
///
/// Example: `"P^{hysics is a hot topic...}"` → `Some("Physics is a hot topic...")`
pub fn detect_drop_cap(latex: &str) -> Option<String> {
    let t = latex.trim();
    if t.len() < 55 {
        return None;
    }
    let bytes = t.as_bytes();
    // Must start with a single uppercase letter
    let base = bytes[0] as char;
    if !base.is_ascii_uppercase() {
        return None;
    }
    // Followed by ^ or _
    if bytes.len() < 4 {
        return None;
    }
    let marker = bytes[1] as char;
    if marker != '^' && marker != '_' {
        return None;
    }
    // Then an opening brace
    if bytes[2] as char != '{' {
        return None;
    }
    // Find closing brace
    let rest = &t[3..];
    let close = rest.rfind('}')?;
    let content = &rest[..close];
    if content.len() < 50 {
        return None;
    }
    // Must be prose-like: has multiple spaces
    let space_count = content.chars().filter(|c| *c == ' ').count();
    if space_count < 5 {
        // Fallback: no spaces but >50 chars and mostly alpha → camelCase drop cap
        // e.g. "$B^{eforewelookatsimulatingthephysicsofparticles...}$"
        let alpha = content
            .chars()
            .filter(|c| c.is_alphabetic())
            .count();
        if alpha as f32 / content.len().max(1) as f32 > 0.8 && content.len() > 50 {
            // Insert spaces before uppercase letters (rough word splitting)
            let mut result = String::with_capacity(content.len() + 20);
            result.push(base);
            for ch in content.chars() {
                if ch.is_uppercase() && !result.is_empty() {
                    let last = result.chars().last().unwrap_or(' ');
                    if last != ' ' && last.is_lowercase() {
                        result.push(' ');
                    }
                }
                result.push(ch);
            }
            return Some(result);
        }
        return None;
    }
    // Mostly alphabetic prose (>70% alpha or space, no backslashes)
    if content.contains('\\') {
        return None; // LaTeX commands → real formula
    }
    let alpha = content
        .chars()
        .filter(|c| c.is_alphabetic() || *c == ' ')
        .count();
    if (alpha as f32 / content.len() as f32) < 0.7 {
        return None;
    }
    Some(format!("{base}{content}"))
}

/// Fix doubled ligature sequences from PDFs that emit both the ligature
/// glyph and the individual characters.
///
/// `"fifirst"` → `"first"`, `"flflags"` → `"flags"`, `"confifigure"` → `"configure"`
pub fn fix_doubled_ligatures(text: &str) -> String {
    if !text.contains("fifi") && !text.contains("flfl") && !text.contains("ffff") {
        return text.to_string();
    }
    let mut result = text.to_string();
    // Order matters: longer patterns first
    result = result.replace("ffiffi", "ffi");
    result = result.replace("fflffl", "ffl");
    result = result.replace("fifi", "fi");
    result = result.replace("flfl", "fl");
    result = result.replace("ffff", "ff");
    result
}

/// Strip InDesign production metadata from text.
///
/// Removes patterns like `"53292_ch08_ptg01.indd 206 9/14/17 10:10 AM"`
pub fn strip_indesign_metadata(text: &str) -> String {
    if !text.contains(".indd") {
        return text.to_string();
    }
    // Pattern: word_with_underscores.indd followed by numbers and optional timestamp
    let mut result = String::with_capacity(text.len());
    let mut remaining = text;
    while let Some(pos) = remaining.find(".indd") {
        // Find the start of the filename (scan backwards for whitespace or start)
        let before = &remaining[..pos];
        let fname_start = before
            .rfind(|c: char| c.is_whitespace())
            .map(|i| i + 1)
            .unwrap_or(0);
        result.push_str(&remaining[..fname_start]);

        // Find the end: skip past .indd and any following digits/slashes/colons/AM/PM
        let after_indd = &remaining[pos + 5..];
        let mut end = 0;
        let chars: Vec<char> = after_indd.chars().collect();
        while end < chars.len()
            && (chars[end].is_ascii_digit()
                || chars[end] == '/'
                || chars[end] == ':'
                || chars[end] == ' '
                || chars[end] == 'A'
                || chars[end] == 'P'
                || chars[end] == 'M')
        {
            end += 1;
        }
        // Trim trailing whitespace from the consumed metadata
        let consumed_end = pos + 5 + after_indd[..end].len();
        remaining = &remaining[consumed_end..];
    }
    result.push_str(remaining);
    result
}

/// Remove consecutive duplicate paragraphs within a text block.
///
/// Some PDFs (especially TeX-generated) have overlapping text objects that
/// produce the same text twice. After extraction, a block's text may contain
/// consecutive identical paragraphs.
pub fn dedup_within_block(text: &str) -> String {
    if !text.contains("\n\n") {
        return text.to_string();
    }
    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    if paragraphs.len() <= 1 {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::with_capacity(paragraphs.len());
    for para in &paragraphs {
        let trimmed = para.trim();
        if trimmed.is_empty() {
            continue;
        }
        // Check if this paragraph duplicates the previous one
        if let Some(prev) = result.last() {
            let prev_trimmed = prev.trim();
            // Exact duplicate
            if prev_trimmed == trimmed {
                continue;
            }
            // One is a substring of the other (>80% overlap)
            let shorter = prev_trimmed.len().min(trimmed.len());
            let longer = prev_trimmed.len().max(trimmed.len());
            if shorter >= 20 && longer > 0 {
                let is_subset = prev_trimmed.starts_with(trimmed)
                    || trimmed.starts_with(prev_trimmed);
                if is_subset && (shorter as f32 / longer as f32) > 0.8 {
                    // Keep the longer one
                    if trimmed.len() > prev_trimmed.len() {
                        *result.last_mut().unwrap() = para;
                    }
                    continue;
                }
            }
        }
        result.push(para);
    }
    result.join("\n\n")
}

// ── Shared label/caption constants ───────────────────────────────────

/// Table caption prefixes — blocks starting with these become TableTitle.
pub const TABLE_CAPTION_PREFIXES: &[&str] = &[
    "table", "tab.", "tab", "tbl.",
];

/// All recognized label prefixes for captions and math environments.
/// Sorted longest-first within each group to avoid "fig" matching before "figure".
pub const ALL_LABEL_PREFIXES: &[&str] = &[
    // Visual element captions
    "figure", "fig.", "fig",
    "table", "tab.", "tab", "tbl.",
    "algorithm", "alg.", "alg",
    "listing", "pseudocode", "procedure", "code",
    "plate", "scheme", "chart", "graph", "diagram",
    // Math environment labels
    "definition", "defn.", "def.",
    "proposition", "prop.",
    "corollary", "corol.", "cor.",
    "conjecture", "conj.",
    "assumption",
    "observation",
    "theorem", "thm.",
    "example", "ex.",
    "exercise",
    "property",
    "remark", "rem.",
    "lemma", "lem.",
    "claim",
    "axiom",
    "proof",
    "tip",
];

/// Check if text starts with a recognized caption/label prefix.
/// Returns the matching prefix if found.
pub fn match_label_prefix(text: &str) -> Option<&'static str> {
    let lower = text.trim_start().to_lowercase();
    for &prefix in ALL_LABEL_PREFIXES {
        if lower.starts_with(prefix) {
            let rest = &lower[prefix.len()..];
            if rest.is_empty() {
                // Bare label like "Proof" — only valid for short blocks
                if text.trim().len() < 80 {
                    return Some(prefix);
                }
            } else {
                let next = rest.chars().next().unwrap();
                if next == ' ' || next == '.' || next == '-'
                    || next == ':' || next.is_ascii_digit()
                {
                    return Some(prefix);
                }
            }
        }
    }
    None
}

/// Classify a label prefix into a RegionKind.
pub fn label_to_region_kind(prefix: &str) -> crate::types::RegionKind {
    if TABLE_CAPTION_PREFIXES.contains(&prefix) {
        crate::types::RegionKind::TableTitle
    } else {
        crate::types::RegionKind::FigureTitle
    }
}

// ── Formula detection ───────────────────────────────────────────────

/// Check if a character is a math symbol or operator.
pub(crate) fn is_math_char(c: char) -> bool {
    matches!(
        c,
        '=' | '+' | '−' | '×' | '÷' | '∫' | '∑' | '∏'
            | '≤' | '≥' | '≠' | '∈' | '∉' | '⊂' | '⊃' | '∪' | '∩'
            | '→' | '←' | '↔' | '∞' | '∂' | '∇'
            | '±' | '∓' | '·' | '∘' | '⊗' | '⊕'
            // Greek lowercase
            | 'α' | 'β' | 'γ' | 'δ' | 'ε' | 'ζ' | 'η' | 'θ'
            | 'ι' | 'κ' | 'λ' | 'μ' | 'ν' | 'ξ' | 'π' | 'ρ'
            | 'σ' | 'τ' | 'υ' | 'φ' | 'χ' | 'ψ' | 'ω'
            // Greek uppercase
            | 'Γ' | 'Δ' | 'Θ' | 'Λ' | 'Ξ' | 'Π' | 'Σ' | 'Φ' | 'Ψ' | 'Ω'
    ) || is_math_italic_unicode(c)
}

/// Check if a character is a mathematical italic Unicode character (U+1D400 block).
/// These are used in PDFs for math variables: 𝑥, 𝑦, 𝑓, 𝑔, 𝐸, 𝐺, etc.
pub(crate) fn is_math_italic_unicode(c: char) -> bool {
    let cp = c as u32;
    // Mathematical Bold (1D400-1D433), Italic (1D434-1D467),
    // Bold Italic (1D468-1D49B), Script (1D49C-1D4CF),
    // Bold Script (1D4D0-1D503), Fraktur (1D504-1D537),
    // Double-struck (1D538-1D56B), Bold Fraktur (1D56C-1D59F),
    // Sans-serif (1D5A0-1D5D3), Sans-serif Bold (1D5D4-1D607),
    // Sans-serif Italic (1D608-1D63B), Sans-serif Bold Italic (1D63C-1D66F),
    // Monospace (1D670-1D6A3), plus Greek variants
    (0x1D400..=0x1D7FF).contains(&cp)
}

/// Check if extracted text looks like a display formula (math expression).
///
/// Uses character-level heuristics: math symbol density, `$` markers,
/// mathematical italic Unicode, and absence of prose words.
pub fn is_likely_formula_text(text: &str) -> bool {
    let trimmed = text.trim();
    let total = trimmed.chars().count();
    // Minimum 5 chars — anything shorter is an inline fragment, not a display formula
    if total < 5 || total > 200 {
        return false;
    }

    // Reject pseudocode/algorithm lines — these contain math but are code, not formulas.
    // ← alone is NOT a pseudocode signal (it's used in math: x_i ← argmin ...),
    // but ← combined with pseudocode keywords IS.
    let lower = trimmed.to_lowercase();
    let pseudocode_keywords = ["for ", "end ", "if ", "then", "else", "while ",
                                "return ", "repeat", "until", "do "];
    let has_pseudocode_keyword = pseudocode_keywords.iter().any(|kw| lower.contains(kw));
    if has_pseudocode_keyword {
        return false;
    }
    // ← with \\leftarrow LaTeX command is pseudocode when combined with other signals
    if trimmed.contains("\\leftarrow") {
        return false;
    }
    // Line numbers at start (e.g., "15: f_i ← ...")
    if trimmed.starts_with(|c: char| c.is_ascii_digit()) {
        if let Some(colon_pos) = trimmed.find(':') {
            if colon_pos <= 3 && trimmed[..colon_pos].chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
        }
    }

    // Reject prose definitions — lines starting with "where", "such that", etc.
    if lower.starts_with("where ") || lower.starts_with("such that")
        || lower.starts_with("and ") || lower.starts_with("is the ")
    {
        return false;
    }

    let math_chars = trimmed.chars().filter(|c| is_math_char(*c)).count();
    let math_italic_chars = trimmed.chars().filter(|c| is_math_italic_unicode(*c)).count();
    let dollar_signs = trimmed.chars().filter(|c| *c == '$').count();

    // Must have at least one math OPERATOR (=, +, -, ≤, ∑, etc.), not just variables.
    // A line of just math italic variables without operators is a definition, not a formula.
    let has_operator = trimmed.chars().any(|c| {
        matches!(c, '=' | '+' | '−' | '×' | '÷' | '∫' | '∑' | '∏'
            | '≤' | '≥' | '≠' | '∈' | '⊂' | '→' | '←' | '∂' | '∇'
            | '±' | '·')
    }) || trimmed.contains("argmin") || trimmed.contains("argmax");

    // Count "prose words": ≥4 ASCII alphabetic letters.
    let prose_words = trimmed
        .split_whitespace()
        .filter(|w| {
            w.len() >= 4
                && w.chars().all(|c| c.is_ascii_alphabetic())
                && !w.starts_with('$')
        })
        .count();

    // Count ASCII alphabetic chars — high ratio means prose (even garbled
    // prose with no spaces like "updateofEquation12when𝜆min").
    let ascii_alpha = trimmed.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let ascii_alpha_ratio = ascii_alpha as f32 / total.max(1) as f32;

    let math_ratio = math_chars as f32 / total.max(1) as f32;

    // High math italic density = garbled math from PDF text layer → almost
    // certainly a formula, BUT only if the line isn't dominated by ASCII
    // text (which would indicate prose with inline math).
    if math_italic_chars >= 2 && prose_words == 0 && total < 80
        && ascii_alpha_ratio < 0.5
    {
        return true;
    }

    if !has_operator && dollar_signs < 2 && math_italic_chars < 2 {
        return false;
    }

    // Reject lines dominated by ASCII text — these are prose with inline
    // math (possibly garbled with no spaces between words).
    if ascii_alpha_ratio > 0.5 && math_ratio < 0.3 {
        return false;
    }

    // High math symbol density + few prose words → formula
    if math_ratio > 0.15 && prose_words <= 1 && total < 120 {
        return true;
    }

    // Very short with math operators and no prose → formula
    if total < 40 && (math_chars >= 1 || math_italic_chars >= 1) && prose_words == 0 && has_operator {
        return true;
    }

    // Contains $ markers (from text layer LaTeX) with math content and an operator
    if dollar_signs >= 2 && has_operator && prose_words <= 1 && total < 120 {
        return true;
    }

    false
}

/// Extract a formula tag like "(3.2)" from the end of text.
///
/// Returns `(formula_text, Some(tag))` if a trailing tag is found,
/// or `(original_text, None)` if no tag detected.
pub fn extract_formula_tag(text: &str) -> (String, Option<String>) {
    let trimmed = text.trim_end();
    if let Some(paren_start) = trimmed.rfind('(') {
        let after_paren = &trimmed[paren_start + 1..];
        if let Some(paren_end) = after_paren.find(')') {
            let tag_content = after_paren[..paren_end].trim();
            // Tag must look like a number: "3", "3.2", "3.2a", "A.1"
            if !tag_content.is_empty()
                && tag_content.len() <= 10
                && tag_content
                    .chars()
                    .next()
                    .map_or(false, |c| c.is_alphanumeric())
                && tag_content
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '.')
            {
                let formula_text = trimmed[..paren_start].trim_end().to_string();
                if !formula_text.is_empty() {
                    return (formula_text, Some(tag_content.to_string()));
                }
            }
        }
    }
    (text.to_string(), None)
}

/// Apply all text cleanup passes to a block's extracted text.
pub fn clean_block_text(text: &str) -> String {
    let mut result = text.to_string();

    // 1. Drop cap repair (check the raw text for formula-like patterns)
    if result.contains("^{") || result.contains("_{") {
        if let Some(expanded) = detect_drop_cap(&result) {
            result = expanded;
        }
    }

    // 2. Ligature dedup
    result = fix_doubled_ligatures(&result);

    // 3. InDesign metadata strip
    result = strip_indesign_metadata(&result);

    // 4. Within-block paragraph dedup
    result = dedup_within_block(&result);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Drop cap tests (moved from pipeline.rs) ──

    #[test]
    fn test_detect_drop_cap_basic() {
        let latex = "P^{hysics is a hot topic and this is the beginning of a chapter about physics}";
        let result = detect_drop_cap(latex);
        assert!(result.is_some());
        assert!(result.unwrap().starts_with("Physics"));
    }

    #[test]
    fn test_detect_drop_cap_subscript() {
        let latex = "I_{n this chapter we will introduce the key concepts and definitions used}";
        let result = detect_drop_cap(latex);
        assert!(result.is_some());
        assert!(result.unwrap().starts_with("In this"));
    }

    #[test]
    fn test_detect_drop_cap_short_not_matched() {
        assert!(detect_drop_cap("P^{2}").is_none());
        assert!(detect_drop_cap("x^{n}").is_none());
    }

    #[test]
    fn test_detect_drop_cap_multi_letter_base() {
        let latex = "PQ^{hysics is a hot topic and this is some long text that should not match}";
        assert!(detect_drop_cap(latex).is_none());
    }

    #[test]
    fn test_detect_drop_cap_lowercase_base() {
        let latex = "p^{hysics is a hot topic and this is the beginning of a chapter about physics}";
        assert!(detect_drop_cap(latex).is_none());
    }

    #[test]
    fn test_detect_drop_cap_no_spaces_camelcase() {
        // Fallback: camelCase splitting for no-space drop caps
        let latex = "I^{nPartIWeBuiltAParticlePhysicsEngineThatIncludedTheForceOfGravityAndWe}";
        let result = detect_drop_cap(latex);
        assert!(result.is_some(), "Should detect camelCase drop cap");
        let expanded = result.unwrap();
        assert!(expanded.starts_with("In"), "Got: {expanded}");
        assert!(expanded.contains(" "), "Should have spaces: {expanded}");
    }

    #[test]
    fn test_detect_drop_cap_no_spaces_all_lower() {
        // All lowercase no-space content — still detected as drop cap
        let latex = "B^{eforewelookatsimulatingthephysicsofparticlesandthischapterreviewsthreemath}";
        let result = detect_drop_cap(latex);
        assert!(result.is_some(), "Should detect all-lowercase drop cap");
        let expanded = result.unwrap();
        assert!(expanded.starts_with("B"), "Got: {expanded}");
    }

    #[test]
    fn test_detect_drop_cap_math_content() {
        let latex = r"F^{\sum_{i=0}^{n} x_i \cdot y_i + \alpha \beta \gamma \delta \epsilon \zeta}";
        assert!(detect_drop_cap(latex).is_none());
    }

    #[test]
    fn test_detect_drop_cap_legitimate_formulas() {
        assert!(detect_drop_cap("E = mc^{2}").is_none());
        assert!(detect_drop_cap("x^{n} + y^{n} = z^{n}").is_none());
        assert!(detect_drop_cap(r"\sum_{i=0}^{n} x_i").is_none());
    }

    // ── Ligature tests ──

    #[test]
    fn test_ligature_fi_dedup() {
        assert_eq!(fix_doubled_ligatures("fifirst"), "first");
        assert_eq!(fix_doubled_ligatures("confifigure"), "configure");
    }

    #[test]
    fn test_ligature_fl_dedup() {
        assert_eq!(fix_doubled_ligatures("flflags"), "flags");
    }

    #[test]
    fn test_ligature_no_false_positive() {
        assert_eq!(fix_doubled_ligatures("WiFi signal"), "WiFi signal");
        assert_eq!(fix_doubled_ligatures("normal text"), "normal text");
    }

    // ── InDesign metadata tests ──

    #[test]
    fn test_indesign_strip() {
        let input = "some text 53292_ch08_ptg01.indd 206 9/14/17 10:10 AM more text";
        let result = strip_indesign_metadata(input);
        assert!(!result.contains(".indd"), "Should strip .indd metadata: {result}");
        assert!(result.contains("some text"));
        assert!(result.contains("more text"));
    }

    #[test]
    fn test_indesign_no_match() {
        let input = "normal text without metadata";
        assert_eq!(strip_indesign_metadata(input), input);
    }

    // ── Within-block dedup tests ──

    #[test]
    fn test_dedup_exact_duplicate() {
        let input = "first paragraph\n\nfirst paragraph\n\nsecond paragraph";
        let result = dedup_within_block(input);
        assert_eq!(
            result.matches("first paragraph").count(),
            1,
            "Should deduplicate: {result}"
        );
        assert!(result.contains("second paragraph"));
    }

    #[test]
    fn test_dedup_subset() {
        let short = "the quick brown fox jumps";
        let long = "the quick brown fox jumps over the lazy dog";
        let input = format!("{long}\n\n{short}");
        let result = dedup_within_block(&input);
        assert!(result.contains("lazy dog"), "Should keep longer version: {result}");
    }

    #[test]
    fn test_dedup_no_false_positive() {
        let input = "paragraph one\n\nparagraph two\n\nparagraph three";
        assert_eq!(dedup_within_block(input), input);
    }

    // ── clean_block_text integration ──

    #[test]
    fn test_clean_block_text_drop_cap() {
        let input = "I^{n this chapter we will introduce the key concepts and fundamental definitions}";
        let result = clean_block_text(input);
        assert!(result.starts_with("In this chapter"), "Got: {result}");
    }

    #[test]
    fn test_clean_block_text_ligature() {
        let input = "The fifirst step is to confifigure the system";
        let result = clean_block_text(input);
        assert_eq!(result, "The first step is to configure the system");
    }

    // ── Label/caption tests ──

    #[test]
    fn test_match_label_theorem() {
        assert_eq!(match_label_prefix("Theorem 3.1 If f is continuous"), Some("theorem"));
    }

    #[test]
    fn test_match_label_lemma() {
        assert_eq!(match_label_prefix("Lemma 2. For any epsilon > 0"), Some("lemma"));
    }

    #[test]
    fn test_match_label_proof_bare() {
        assert_eq!(match_label_prefix("Proof."), Some("proof"));
    }

    #[test]
    fn test_match_label_proof_with_period() {
        // "Proof." or "Proof. We proceed..." — valid proof label
        assert_eq!(match_label_prefix("Proof. We proceed by induction on n."), Some("proof"));
    }

    #[test]
    fn test_match_label_not_mid_sentence() {
        // Words that happen to contain a label prefix but aren't labels
        assert!(match_label_prefix("profitable business venture").is_none());
        assert!(match_label_prefix("graphically enhanced display").is_none());
    }

    #[test]
    fn test_match_label_definition() {
        assert_eq!(match_label_prefix("Definition 1.3. A set S is convex"), Some("definition"));
    }

    #[test]
    fn test_match_label_listing() {
        assert_eq!(match_label_prefix("Listing 4.2 CUDA kernel"), Some("listing"));
    }

    #[test]
    fn test_match_label_figure() {
        assert_eq!(match_label_prefix("Figure 1. Simulation results"), Some("figure"));
    }

    #[test]
    fn test_match_label_fig_dot() {
        assert_eq!(match_label_prefix("Fig. 3 Stress test"), Some("fig."));
    }

    #[test]
    fn test_match_label_table() {
        assert_eq!(match_label_prefix("Table 2 Performance comparison"), Some("table"));
    }

    #[test]
    fn test_match_label_algorithm() {
        assert_eq!(match_label_prefix("Algorithm 1 Vertex Block Descent"), Some("algorithm"));
    }

    #[test]
    fn test_match_label_corollary() {
        assert_eq!(match_label_prefix("Corollary 2.3. Under the same assumptions"), Some("corollary"));
    }

    #[test]
    fn test_match_label_example() {
        assert_eq!(match_label_prefix("Example 5.1. Consider the function"), Some("example"));
    }

    #[test]
    fn test_match_label_remark() {
        assert_eq!(match_label_prefix("Remark 3. This result generalizes"), Some("remark"));
    }

    #[test]
    fn test_match_label_proposition() {
        assert_eq!(match_label_prefix("Proposition 4.1 For any bounded set"), Some("proposition"));
    }

    #[test]
    fn test_match_label_not_prose() {
        assert!(match_label_prefix("the theorem states that").is_none());
        assert!(match_label_prefix("In this example we show").is_none());
        assert!(match_label_prefix("A remarkable result").is_none());
    }

    #[test]
    fn test_match_label_exercise() {
        assert_eq!(match_label_prefix("Exercise 7. Prove that"), Some("exercise"));
    }

    #[test]
    fn test_label_to_kind_table() {
        assert_eq!(label_to_region_kind("table"), crate::types::RegionKind::TableTitle);
        assert_eq!(label_to_region_kind("tab."), crate::types::RegionKind::TableTitle);
        assert_eq!(label_to_region_kind("tbl."), crate::types::RegionKind::TableTitle);
    }

    #[test]
    fn test_label_to_kind_figure() {
        assert_eq!(label_to_region_kind("figure"), crate::types::RegionKind::FigureTitle);
        assert_eq!(label_to_region_kind("theorem"), crate::types::RegionKind::FigureTitle);
        assert_eq!(label_to_region_kind("algorithm"), crate::types::RegionKind::FigureTitle);
        assert_eq!(label_to_region_kind("proof"), crate::types::RegionKind::FigureTitle);
    }

    // ── Formula detection tests ──

    #[test]
    fn test_formula_simple_equation() {
        assert!(is_likely_formula_text("E = mc²"));
    }

    #[test]
    fn test_formula_greek() {
        assert!(is_likely_formula_text("α + β = γ"));
    }

    #[test]
    fn test_formula_operators() {
        assert!(is_likely_formula_text("x ≤ y + z"));
    }

    #[test]
    fn test_formula_not_prose() {
        assert!(!is_likely_formula_text(
            "The sum of the squares equals the total variance in the sample."
        ));
    }

    #[test]
    fn test_formula_not_long_text() {
        assert!(!is_likely_formula_text(
            "This is a normal paragraph of text that discusses mathematical concepts but is not itself a formula."
        ));
    }

    #[test]
    fn test_formula_short_with_op() {
        assert!(is_likely_formula_text("x + y"));
    }

    // ── Formula tag extraction tests ──

    #[test]
    fn test_formula_tag_simple() {
        let (text, tag) = extract_formula_tag("f(x) = 0    (12)");
        assert_eq!(tag, Some("12".to_string()));
        assert_eq!(text, "f(x) = 0");
    }

    #[test]
    fn test_formula_tag_dotted() {
        let (text, tag) = extract_formula_tag("x + y = z    (A.3)");
        assert_eq!(tag, Some("A.3".to_string()));
        assert_eq!(text, "x + y = z");
    }

    #[test]
    fn test_formula_tag_multi_digit() {
        let (text, tag) = extract_formula_tag("L = -sum log p    (3.14)");
        assert_eq!(tag, Some("3.14".to_string()));
        assert!(text.starts_with("L = "));
    }

    #[test]
    fn test_formula_no_tag() {
        let (text, tag) = extract_formula_tag("x² + y² = r²");
        assert!(tag.is_none());
        assert_eq!(text, "x² + y² = r²");
    }

    #[test]
    fn test_formula_tag_not_prose_parens() {
        // "(see Section 3)" has spaces → not a valid tag
        let (_, tag) = extract_formula_tag("some text (see Section 3)");
        assert!(tag.is_none());
    }

    #[test]
    fn test_formula_tag_letter_suffix() {
        let (_, tag) = extract_formula_tag("expression (2a)");
        assert_eq!(tag, Some("2a".to_string()));
    }
}
