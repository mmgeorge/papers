//! Font-glyph-name recovery.
//!
//! pdfium maps each glyph to a Unicode codepoint via the PDF's `/ToUnicode`
//! CMap. When that CMap is missing or broken, pdfium returns a wrong or
//! *collapsed* codepoint (a math epsilon as U+000F; four distinct glyphs all as
//! U+0000). pdfium exposes no glyph name, glyph index, or char code, so this
//! cannot be repaired from pdfium alone.
//!
//! This module re-reads the PDF independently with `lopdf`. For each font it
//! builds a `char-code -> Unicode` map from the font's `/Encoding /Differences`
//! (resolving each glyph NAME through the Adobe Glyph List), then re-tokenizes a
//! page's content stream to recover the ordered char-code of every drawn glyph.
//!
//! It overrides ONLY pdfium chars whose codepoint is a control character (i.e.
//! already broken), and only when the recovered glyph sequence aligns 1:1 with
//! pdfium's non-generated chars — so correctly-extracted text can never be
//! corrupted (a misalignment degrades to "no change", guarded by the count).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use lopdf::content::Content;
use lopdf::{Document, Object, ObjectId};

use crate::pdf::PdfChar;

static GLYPHLIST_TXT: &str = include_str!("../assets/glyphlist.txt");

/// Sentinel for a content-stream glyph we could not map to a character. It keeps
/// the positional alignment intact without ever overriding a pdfium char.
const NO_RECOVERY: char = '\u{FFFF}';

/// The Adobe Glyph List (`glyph name -> char`), parsed once, plus a small
/// supplement of TeX/math glyph names absent from the base list.
fn agl() -> &'static HashMap<String, char> {
    static AGL: OnceLock<HashMap<String, char>> = OnceLock::new();
    AGL.get_or_init(|| {
        let mut m: HashMap<String, char> = HashMap::new();
        for line in GLYPHLIST_TXT.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((name, hex)) = line.split_once(';') {
                // A ligature decomposes to several codepoints; take the first.
                let first = hex.split_whitespace().next().unwrap_or("");
                if let Ok(cp) = u32::from_str_radix(first, 16) {
                    if let Some(c) = char::from_u32(cp) {
                        m.insert(name.to_string(), c);
                    }
                }
            }
        }
        // Math / TeX glyph names not in the base AGL.
        for (n, cp) in [
            ("ell", 0x2113u32),
            ("lscript", 0x2113),
            ("epsilon1", 0x03B5),
            ("Rdblstruck", 0x211D),
            ("Cdblstruck", 0x2102),
            ("Ndblstruck", 0x2115),
            ("Zdblstruck", 0x2124),
            ("Qdblstruck", 0x211A),
            ("Hdblstruck", 0x210D),
            ("Pdblstruck", 0x2119),
            ("summationtext", 0x2211),
            ("producttext", 0x220F),
            ("integraltext", 0x222B),
            ("greatermuch", 0x226B),
            ("lessmuch", 0x226A),
        ] {
            if let Some(c) = char::from_u32(cp) {
                m.entry(n.to_string()).or_insert(c);
            }
        }
        m
    })
}

/// Resolve a glyph name to a character: AGL, a `.suffix`-stripped retry, then the
/// `uniXXXX` / `uXXXXXX` algorithmic conventions.
fn glyph_name_to_char(name: &str) -> Option<char> {
    if let Some(&c) = agl().get(name) {
        return Some(c);
    }
    if let Some((base, _)) = name.split_once('.') {
        if let Some(&c) = agl().get(base) {
            return Some(c);
        }
    }
    if let Some(rest) = name.strip_prefix("uni") {
        if rest.len() >= 4 && rest[..4].chars().all(|c| c.is_ascii_hexdigit()) {
            if let Ok(cp) = u32::from_str_radix(&rest[..4], 16) {
                return char::from_u32(cp);
            }
        }
    }
    if let Some(rest) = name.strip_prefix('u') {
        if (4..=6).contains(&rest.len()) && rest.chars().all(|c| c.is_ascii_hexdigit()) {
            if let Ok(cp) = u32::from_str_radix(rest, 16) {
                return char::from_u32(cp);
            }
        }
    }
    None
}

/// A loaded PDF used to repair broken codepoints in pdfium's extracted chars.
pub struct GlyphRecovery {
    doc: Document,
    font_cache: RefCell<HashMap<ObjectId, HashMap<u8, char>>>,
}

impl GlyphRecovery {
    /// Load the PDF for glyph recovery. Returns `None` if it can't be parsed —
    /// the caller then proceeds with pdfium's chars unchanged.
    pub fn open(path: &Path) -> Option<Self> {
        let doc = Document::load(path).ok()?;
        Some(Self {
            doc,
            font_cache: RefCell::new(HashMap::new()),
        })
    }

    fn resolve<'a>(&'a self, obj: Option<&'a Object>) -> Option<&'a Object> {
        match obj {
            Some(Object::Reference(id)) => self.doc.get_object(*id).ok(),
            other => other,
        }
    }

    /// `char code -> char` from a font's `/Encoding /Differences`, cached.
    fn font_code_map(&self, font_id: ObjectId) -> HashMap<u8, char> {
        if let Some(m) = self.font_cache.borrow().get(&font_id) {
            return m.clone();
        }
        let map = self.build_font_code_map(font_id);
        self.font_cache.borrow_mut().insert(font_id, map.clone());
        map
    }

    fn build_font_code_map(&self, font_id: ObjectId) -> HashMap<u8, char> {
        let mut map = HashMap::new();
        let Ok(font) = self.doc.get_object(font_id).and_then(|o| o.as_dict()) else {
            return map;
        };
        let enc = self.resolve(font.get(b"Encoding").ok());
        let Some(enc) = enc else { return map };
        let Ok(enc_dict) = enc.as_dict() else { return map };
        let Ok(Object::Array(diffs)) = enc_dict.get(b"Differences") else {
            return map;
        };
        let mut code: i64 = 0;
        for el in diffs {
            match el {
                Object::Integer(n) => code = *n,
                Object::Name(name) => {
                    let nm = String::from_utf8_lossy(name);
                    if (0..=255).contains(&code) {
                        if let Some(c) = glyph_name_to_char(&nm) {
                            map.insert(code as u8, c);
                        }
                    }
                    code += 1;
                }
                _ => {}
            }
        }
        map
    }

    /// Resource font dictionary for a page (`resource name -> font object id`),
    /// walking up the `/Parent` chain for inherited `/Resources`.
    fn page_fonts(&self, page_id: ObjectId) -> HashMap<Vec<u8>, ObjectId> {
        let mut out = HashMap::new();
        let mut resources: Option<&Object> = None;
        let mut cur = Some(page_id);
        let mut guard = 0;
        while let Some(id) = cur {
            guard += 1;
            if guard > 50 {
                break;
            }
            let Ok(dict) = self.doc.get_object(id).and_then(|o| o.as_dict()) else {
                break;
            };
            if let Ok(r) = dict.get(b"Resources") {
                resources = Some(r);
                break;
            }
            cur = match dict.get(b"Parent") {
                Ok(Object::Reference(pid)) => Some(*pid),
                _ => None,
            };
        }
        let Some(res) = self.resolve(resources) else {
            return out;
        };
        let Ok(res_dict) = res.as_dict() else { return out };
        let Some(fonts) = self.resolve(res_dict.get(b"Font").ok()) else {
            return out;
        };
        let Ok(fonts_dict) = fonts.as_dict() else { return out };
        for (name, val) in fonts_dict.iter() {
            if let Object::Reference(fid) = val {
                out.insert(name.clone(), *fid);
            }
        }
        out
    }

    /// Repair broken (control-char) codepoints in `chars` for the given page,
    /// using the recovered content-stream glyph sequence. No-op unless the
    /// recovered sequence aligns exactly with pdfium's non-generated chars.
    pub fn recover_page(&self, page_idx: u32, chars: &mut [PdfChar]) {
        let pages = self.doc.get_pages();
        let Some(&page_id) = pages.get(&(page_idx + 1)) else {
            return;
        };
        let fonts = self.page_fonts(page_id);
        if fonts.is_empty() {
            return;
        }
        let Ok(content_bytes) = self.doc.get_page_content(page_id) else {
            return;
        };
        let Ok(content) = Content::decode(&content_bytes) else {
            return;
        };

        // One entry per drawn glyph, in draw order (a char, or NO_RECOVERY).
        let mut glyphs: Vec<char> = Vec::new();
        let mut cur_map: HashMap<u8, char> = HashMap::new();
        for op in &content.operations {
            match op.operator.as_str() {
                "Tf" => {
                    if let Some(Object::Name(n)) = op.operands.first() {
                        cur_map = fonts
                            .get(n)
                            .map(|&fid| self.font_code_map(fid))
                            .unwrap_or_default();
                    }
                }
                "Tj" | "'" | "\"" => {
                    if let Some(Object::String(s, _)) = op.operands.last() {
                        push_codes(s, &cur_map, &mut glyphs);
                    }
                }
                "TJ" => {
                    if let Some(Object::Array(arr)) = op.operands.first() {
                        for el in arr {
                            if let Object::String(s, _) = el {
                                push_codes(s, &cur_map, &mut glyphs);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Align against pdfium's non-generated chars (pdfium injects synthetic
        // spaces that are absent from the content stream).
        let pdf_idx: Vec<usize> = chars
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.is_generated)
            .map(|(i, _)| i)
            .collect();
        if glyphs.is_empty() || pdf_idx.len() != glyphs.len() {
            return; // not a clean 1:1 alignment — leave pdfium's chars untouched
        }
        for (k, &i) in pdf_idx.iter().enumerate() {
            let rc = glyphs[k];
            if rc == NO_RECOVERY {
                continue;
            }
            // Repair a broken pdfium char with the font's real glyph. "Broken" =
            // a control code, or a stray grave accent (U+0060), which broken math
            // ToUnicode maps the script-ell and similar glyphs to. The recovered
            // glyph is the font's actual character for this code, so when pdfium
            // already had it right the override is a no-op.
            let broken = chars[i].codepoint.is_control() || chars[i].codepoint == '\u{0060}';
            if broken && !rc.is_control() && rc != ' ' {
                chars[i].codepoint = rc;
            }
        }
    }
}

fn push_codes(s: &[u8], map: &HashMap<u8, char>, out: &mut Vec<char>) {
    for &code in s {
        out.push(map.get(&code).copied().unwrap_or(NO_RECOVERY));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agl_resolves_common_and_math_names() {
        assert_eq!(glyph_name_to_char("epsilon"), Some('ε'));
        assert_eq!(glyph_name_to_char("lambda"), Some('λ'));
        assert_eq!(glyph_name_to_char("ffi"), Some('ﬃ'));
        assert_eq!(glyph_name_to_char("lscript"), Some('ℓ'));
        assert_eq!(glyph_name_to_char("Rdblstruck"), Some('ℝ'));
        assert_eq!(glyph_name_to_char("uni03B5"), Some('ε'));
        assert_eq!(glyph_name_to_char("a.sc"), Some('a'));
        assert_eq!(glyph_name_to_char("not_a_glyph_name_xyz"), None);
    }
}
