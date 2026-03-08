use serde::{Deserialize, Serialize};

/// Per-formula prediction result with LaTeX and confidence score.
pub struct FormulaResult {
    pub latex: String,
    /// Sequence-level confidence: `exp(mean(token_log_probs))`.
    /// Range (0, 1] — higher means the model was more certain per-token on average.
    pub confidence: f32,
}

/// Full extraction result for a PDF document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub metadata: Metadata,
    pub pages: Vec<Page>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub filename: String,
    pub page_count: u32,
    pub extraction_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Page {
    pub page: u32,
    pub width_pt: f32,
    pub height_pt: f32,
    pub dpi: u32,
    pub regions: Vec<Region>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    pub id: String,
    pub kind: RegionKind,
    pub bbox: [f32; 4],
    pub confidence: f32,
    pub order: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub html: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latex: Option<String>,
    /// How this formula's LaTeX was produced: "char" (char-based) or "ocr" (ML model).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formula_source: Option<String>,
    /// OCR model confidence for formula LaTeX (sequence-level, 0–1).
    /// Only present when `formula_source` is "ocr".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr_confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<Box<Region>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chart_type: Option<String>,
    /// Equation number tag for display formulas (e.g. "1", "2a").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    /// Member regions for a `FigureGroup`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Vec<Region>>,
    /// True when this region's content has been spliced into a parent region
    /// (e.g. an InlineFormula consumed by an Algorithm or Text region).
    /// Consumed regions are kept in the JSON for debug/layout but skipped
    /// in markdown output to avoid duplication.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub consumed: bool,
}

impl FormulaResult {
    /// Compute sequence-level confidence from per-token log-probabilities.
    ///
    /// Returns `exp(mean(log_probs))` — the geometric mean of per-token probabilities.
    /// If `token_log_probs` is empty, returns 0.0.
    pub fn sequence_confidence(token_log_probs: &[f32]) -> f32 {
        if token_log_probs.is_empty() {
            return 0.0;
        }
        let mean_log_prob: f32 =
            token_log_probs.iter().sum::<f32>() / token_log_probs.len() as f32;
        mean_log_prob.exp()
    }
}

/// All 24 DocLayout V3 region classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionKind {
    Title,
    ParagraphTitle,
    Text,
    VerticalText,
    PageNumber,
    Abstract,
    TOC,
    References,
    Footnote,
    PageHeader,
    PageFooter,
    Algorithm,
    DisplayFormula,
    InlineFormula,
    FormulaNumber,
    Image,
    Table,
    FigureTableTitle,
    FigureTitle,
    TableTitle,
    ChartTitle,
    Seal,
    Chart,
    SidebarText,
    /// Synthetic group of spatially close visual regions sharing one caption.
    FigureGroup,
}

impl RegionKind {
    /// Map from a label string (model output or oar-ocr label) to RegionKind.
    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            "Title" | "doc_title" => Some(Self::Title),
            "ParagraphTitle" | "paragraph_title" | "title" => Some(Self::ParagraphTitle),
            "Text" | "text" | "content" | "list" => Some(Self::Text),
            "VerticalText" | "vertical_text" => Some(Self::VerticalText),
            "PageNumber" | "page_number" | "number" => Some(Self::PageNumber),
            "Abstract" | "abstract" => Some(Self::Abstract),
            "TOC" | "toc" => Some(Self::TOC),
            "References" | "references" | "reference" | "reference_content" => {
                Some(Self::References)
            }
            "Footnote" | "footnote" | "vision_footnote" => Some(Self::Footnote),
            "PageHeader" | "page_header" | "header" | "header_image" => Some(Self::PageHeader),
            "PageFooter" | "page_footer" | "footer" | "footer_image" => Some(Self::PageFooter),
            "Algorithm" | "algorithm" => Some(Self::Algorithm),
            "DisplayFormula" | "display_formula" | "formula" => Some(Self::DisplayFormula),
            "InlineFormula" | "inline_formula" => Some(Self::InlineFormula),
            "FormulaNumber" | "formula_number" => Some(Self::FormulaNumber),
            "Image" | "image" => Some(Self::Image),
            "Table" | "table" => Some(Self::Table),
            "FigureTableTitle" | "figure_table_title" | "figure_table_chart_title" => {
                Some(Self::FigureTableTitle)
            }
            "FigureTitle" | "figure_title" => Some(Self::FigureTitle),
            "TableTitle" | "table_title" => Some(Self::TableTitle),
            "ChartTitle" | "chart_title" => Some(Self::ChartTitle),
            "Seal" | "seal" => Some(Self::Seal),
            "Chart" | "chart" => Some(Self::Chart),
            "SidebarText" | "sidebar_text" | "aside_text" => Some(Self::SidebarText),
            _ => None,
        }
    }

    /// Whether this region kind should have text content populated.
    pub fn is_text_bearing(&self) -> bool {
        matches!(
            self,
            Self::Title
                | Self::ParagraphTitle
                | Self::Text
                | Self::VerticalText
                | Self::Abstract
                | Self::References
                | Self::Footnote
                | Self::Algorithm
                | Self::SidebarText
                | Self::TOC
                | Self::PageNumber
                | Self::FormulaNumber
                | Self::PageHeader
                | Self::PageFooter
        )
    }

    /// Whether this region kind is a caption type.
    pub fn is_caption(&self) -> bool {
        matches!(
            self,
            Self::FigureTitle | Self::TableTitle | Self::FigureTableTitle | Self::ChartTitle
        )
    }

    /// Whether this region kind represents a visual element that should be cropped.
    pub fn is_visual(&self) -> bool {
        matches!(self, Self::Image | Self::Chart | Self::Seal)
    }

    /// Whether this region kind can be grouped into a `FigureGroup`.
    ///
    /// Includes Table (which `is_visual()` excludes) because composite figures
    /// sometimes contain table sub-panels alongside images/charts.
    pub fn is_groupable(&self) -> bool {
        matches!(
            self,
            Self::Image | Self::Chart | Self::Table | Self::Seal
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_kind_roundtrip() {
        let kinds = [
            RegionKind::Title,
            RegionKind::ParagraphTitle,
            RegionKind::Text,
            RegionKind::VerticalText,
            RegionKind::PageNumber,
            RegionKind::Abstract,
            RegionKind::TOC,
            RegionKind::References,
            RegionKind::Footnote,
            RegionKind::PageHeader,
            RegionKind::PageFooter,
            RegionKind::Algorithm,
            RegionKind::DisplayFormula,
            RegionKind::InlineFormula,
            RegionKind::FormulaNumber,
            RegionKind::Image,
            RegionKind::Table,
            RegionKind::FigureTableTitle,
            RegionKind::FigureTitle,
            RegionKind::TableTitle,
            RegionKind::ChartTitle,
            RegionKind::Seal,
            RegionKind::Chart,
            RegionKind::SidebarText,
            RegionKind::FigureGroup,
        ];

        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let roundtrip: RegionKind = serde_json::from_str(&json).unwrap();
            assert_eq!(*kind, roundtrip);
        }
    }

    #[test]
    fn test_region_json_text() {
        let region = Region {
            id: "p1_0".into(),
            kind: RegionKind::Title,
            bbox: [72.0, 50.0, 540.0, 85.0],
            confidence: 0.97,
            order: 0,
            text: Some("A Novel Approach".into()),
            html: None,
            latex: None,
            image_path: None,
            caption: None,
            chart_type: None,
            tag: None,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
        };

        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(json["text"], "A Novel Approach");
        assert!(json.get("html").is_none());
        assert!(json.get("latex").is_none());
    }

    #[test]
    fn test_region_json_table() {
        let region = Region {
            id: "p1_3".into(),
            kind: RegionKind::Table,
            bbox: [60.0, 200.0, 550.0, 400.0],
            confidence: 0.95,
            order: 3,
            text: None,
            html: Some("<table><tr><td>A</td></tr></table>".into()),
            latex: None,
            image_path: None,
            caption: None,
            chart_type: None,
            tag: None,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
        };

        let json = serde_json::to_value(&region).unwrap();
        assert!(json["html"].as_str().unwrap().contains("<table>"));
        assert!(json.get("text").is_none());
    }

    #[test]
    fn test_region_json_formula() {
        let region = Region {
            id: "p1_5".into(),
            kind: RegionKind::DisplayFormula,
            bbox: [100.0, 420.0, 500.0, 460.0],
            confidence: 0.92,
            order: 5,
            text: None,
            html: None,
            latex: Some("E = mc^2".into()),
            image_path: None,
            caption: None,
            chart_type: None,
            tag: None,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
        };

        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(json["latex"], "E = mc^2");
    }

    #[test]
    fn test_region_json_image() {
        let cap = Region {
            id: "p1_6".into(),
            kind: RegionKind::FigureTitle,
            bbox: [80.0, 710.0, 530.0, 730.0],
            confidence: 0.91,
            order: 6,
            text: Some("Figure 1: Overview".into()),
            html: None,
            latex: None,
            image_path: None,
            caption: None,
            chart_type: None,
            tag: None,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
        };
        let region = Region {
            id: "p1_7".into(),
            kind: RegionKind::Image,
            bbox: [80.0, 500.0, 530.0, 700.0],
            confidence: 0.89,
            order: 7,
            text: None,
            html: None,
            latex: None,
            image_path: Some("images/p1_7.png".into()),
            caption: Some(Box::new(cap)),
            chart_type: None,
            tag: None,
            items: None,
            formula_source: None,
            ocr_confidence: None,
            consumed: false,
        };

        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(json["image_path"], "images/p1_7.png");
        assert_eq!(json["caption"]["text"], "Figure 1: Overview");
        assert_eq!(json["caption"]["kind"], "FigureTitle");
    }

    #[test]
    fn test_extraction_result_roundtrip() {
        let result = ExtractionResult {
            metadata: Metadata {
                filename: "test.pdf".into(),
                page_count: 1,
                extraction_time_ms: 100,
            },
            pages: vec![Page {
                page: 1,
                width_pt: 612.0,
                height_pt: 792.0,
                dpi: 144,
                regions: vec![Region {
                    id: "p1_0".into(),
                    kind: RegionKind::Title,
                    bbox: [72.0, 50.0, 540.0, 85.0],
                    confidence: 0.97,
                    order: 0,
                    text: Some("Test Title".into()),
                    html: None,
                    latex: None,
                    image_path: None,
                    caption: None,
                    chart_type: None,
                    tag: None,
                    items: None,
                    formula_source: None,
                    ocr_confidence: None,
                    consumed: false,
                }],
            }],
        };

        let json = serde_json::to_string_pretty(&result).unwrap();
        let roundtrip: ExtractionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.metadata.filename, "test.pdf");
        assert_eq!(roundtrip.pages.len(), 1);
        assert_eq!(roundtrip.pages[0].regions[0].kind, RegionKind::Title);
    }

    #[test]
    fn test_region_kind_from_label() {
        // oar-ocr label strings
        assert_eq!(RegionKind::from_label("doc_title"), Some(RegionKind::Title));
        assert_eq!(
            RegionKind::from_label("paragraph_title"),
            Some(RegionKind::ParagraphTitle)
        );
        assert_eq!(RegionKind::from_label("text"), Some(RegionKind::Text));
        assert_eq!(RegionKind::from_label("content"), Some(RegionKind::Text));
        assert_eq!(RegionKind::from_label("list"), Some(RegionKind::Text));
        assert_eq!(
            RegionKind::from_label("abstract"),
            Some(RegionKind::Abstract)
        );
        assert_eq!(RegionKind::from_label("table"), Some(RegionKind::Table));
        assert_eq!(
            RegionKind::from_label("formula"),
            Some(RegionKind::DisplayFormula)
        );
        assert_eq!(
            RegionKind::from_label("aside_text"),
            Some(RegionKind::SidebarText)
        );
        assert_eq!(RegionKind::from_label("header"), Some(RegionKind::PageHeader));
        assert_eq!(RegionKind::from_label("footer"), Some(RegionKind::PageFooter));
        assert_eq!(RegionKind::from_label("number"), Some(RegionKind::PageNumber));
        assert_eq!(
            RegionKind::from_label("reference"),
            Some(RegionKind::References)
        );
        assert_eq!(
            RegionKind::from_label("figure_table_chart_title"),
            Some(RegionKind::FigureTableTitle)
        );

        // PascalCase variants
        assert_eq!(RegionKind::from_label("Title"), Some(RegionKind::Title));
        assert_eq!(RegionKind::from_label("Table"), Some(RegionKind::Table));
        assert_eq!(RegionKind::from_label("Chart"), Some(RegionKind::Chart));

        // PP-DocLayoutV3 class 15 and 24
        assert_eq!(
            RegionKind::from_label("inline_formula"),
            Some(RegionKind::InlineFormula)
        );
        assert_eq!(
            RegionKind::from_label("vision_footnote"),
            Some(RegionKind::Footnote)
        );

        // Unknown
        assert_eq!(RegionKind::from_label("unknown"), None);
        assert_eq!(RegionKind::from_label("region"), None);
    }

    #[test]
    fn test_sequence_confidence() {
        // Empty → 0.0
        assert_eq!(FormulaResult::sequence_confidence(&[]), 0.0);

        // All zero log-probs → exp(0) = 1.0
        assert!((FormulaResult::sequence_confidence(&[0.0, 0.0, 0.0]) - 1.0).abs() < 1e-6);

        // Uniform negative log-probs → exp(mean) = exp(-0.5) ≈ 0.6065
        let conf = FormulaResult::sequence_confidence(&[-0.5, -0.5]);
        assert!((conf - 0.6065).abs() < 0.001);

        // Mixed log-probs → exp(mean(-0.1, -0.9)) = exp(-0.5) ≈ 0.6065
        let conf = FormulaResult::sequence_confidence(&[-0.1, -0.9]);
        assert!((conf - 0.6065).abs() < 0.001);
    }

    #[test]
    fn test_region_json_ocr_confidence() {
        // With ocr_confidence set
        let region = Region {
            id: "p1_5".into(),
            kind: RegionKind::DisplayFormula,
            bbox: [100.0, 420.0, 500.0, 460.0],
            confidence: 0.92,
            order: 5,
            text: None,
            html: None,
            latex: Some("E = mc^2".into()),
            formula_source: Some("ocr".into()),
            ocr_confidence: Some(0.85),
            image_path: None,
            caption: None,
            chart_type: None,
            tag: None,
            items: None,
            consumed: false,
        };

        let json = serde_json::to_value(&region).unwrap();
        let ocr_conf = json["ocr_confidence"].as_f64().unwrap();
        assert!((ocr_conf - 0.85).abs() < 1e-6);

        // Without ocr_confidence — should be omitted
        let region2 = Region {
            ocr_confidence: None,
            ..region.clone()
        };
        let json2 = serde_json::to_value(&region2).unwrap();
        assert!(json2.get("ocr_confidence").is_none());
    }
}
