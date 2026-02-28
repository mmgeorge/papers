use serde::{Deserialize, Serialize};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chart_type: Option<String>,
}

/// All 23 DocLayout V3 region classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
            "Footnote" | "footnote" => Some(Self::Footnote),
            "PageHeader" | "page_header" | "header" | "header_image" => Some(Self::PageHeader),
            "PageFooter" | "page_footer" | "footer" | "footer_image" => Some(Self::PageFooter),
            "Algorithm" | "algorithm" => Some(Self::Algorithm),
            "DisplayFormula" | "display_formula" | "formula" => Some(Self::DisplayFormula),
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
        };

        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(json["latex"], "E = mc^2");
    }

    #[test]
    fn test_region_json_image() {
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
            caption: Some("Figure 1: Overview".into()),
            chart_type: None,
        };

        let json = serde_json::to_value(&region).unwrap();
        assert_eq!(json["image_path"], "images/p1_7.png");
        assert_eq!(json["caption"], "Figure 1: Overview");
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

        // Unknown
        assert_eq!(RegionKind::from_label("unknown"), None);
        assert_eq!(RegionKind::from_label("region"), None);
    }
}
