use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSummary {
    pub chunk_id: String,
    pub text_preview: String,
    pub depth: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferencedFigure {
    pub figure_id: String,
    pub figure_type: String,
    pub caption: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionContext {
    pub total_chunks_in_section: u32,
    pub total_sections_in_chapter: u32,
    pub total_chapters_in_paper: u32,
    pub is_first_in_section: bool,
    pub is_last_in_section: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkWithPosition {
    pub chunk_id: String,
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    pub text: String,
    pub chapter_title: String,
    pub chapter_idx: u16,
    pub section_title: String,
    pub section_idx: u16,
    pub chunk_idx: u16,
    pub depth: String,
    pub figure_ids: Vec<String>,
    pub referenced_figures: Vec<ReferencedFigure>,
    pub position: PositionContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: ChunkWithPosition,
    pub prev: Option<ChunkSummary>,
    pub next: Option<ChunkSummary>,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResult {
    pub chunk: ChunkWithPosition,
    pub prev: Option<ChunkSummary>,
    pub next: Option<ChunkSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionResult {
    pub paper_id: String,
    pub chapter_title: String,
    pub section_title: String,
    pub chunks: Vec<ChunkWithPosition>,
    pub total_chunks: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterSection {
    pub section_idx: u16,
    pub section_title: String,
    pub chunks: Vec<ChunkWithPosition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterResult {
    pub paper_id: String,
    pub chapter_title: String,
    pub chapter_idx: u16,
    pub sections: Vec<ChapterSection>,
    pub total_chunks: usize,
    pub figure_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigureResult {
    pub figure_id: String,
    pub paper_id: String,
    pub figure_type: String,
    pub caption: String,
    pub description: String,
    pub image_path: Option<String>,
    pub page: Option<u16>,
    pub referenced_by: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlineSection {
    pub section_idx: u16,
    pub section_title: String,
    pub chunk_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlineChapter {
    pub chapter_idx: u16,
    pub chapter_title: String,
    pub sections: Vec<OutlineSection>,
    pub figure_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperOutline {
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    pub tags: Vec<String>,
    pub chapters: Vec<OutlineChapter>,
    pub total_chunks: usize,
    pub total_figures: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSummary {
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    pub tags: Vec<String>,
    pub chunk_count: usize,
    pub figure_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagSummary {
    pub tag: String,
    pub paper_count: usize,
}

/// Input parameters for search queries.
pub struct SearchParams {
    pub query: String,
    pub paper_ids: Option<Vec<String>>,
    pub chapter_idx: Option<u16>,
    pub section_idx: Option<u16>,
    pub filter_year_min: Option<u16>,
    pub filter_year_max: Option<u16>,
    pub filter_venue: Option<String>,
    pub filter_tags: Option<Vec<String>>,
    pub filter_depth: Option<String>,
    pub limit: u16,
}

/// Input parameters for figure search.
pub struct SearchFiguresParams {
    pub query: String,
    pub paper_ids: Option<Vec<String>>,
    pub filter_figure_type: Option<String>,
    pub limit: u16,
}

/// Input parameters for list_papers.
pub struct ListPapersParams {
    pub paper_ids: Option<Vec<String>>,
    pub filter_year_min: Option<u16>,
    pub filter_year_max: Option<u16>,
    pub filter_venue: Option<String>,
    pub filter_tags: Option<Vec<String>>,
    pub filter_authors: Option<Vec<String>>,
    pub sort_by: Option<String>,
    pub limit: u16,
}

/// Input parameters for list_tags.
pub struct ListTagsParams {
    pub paper_ids: Option<Vec<String>>,
}

pub struct IngestStats {
    pub chunks_added: usize,
    pub figures_added: usize,
}
