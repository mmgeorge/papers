use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkSummary {
    pub chunk_id: String,
    pub text_preview: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferencedExhibit {
    pub exhibit_id: String,
    pub exhibit_type: String,
    pub caption: String,
    pub description: Option<String>,
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
    pub block_type: String,
    pub exhibit_ids: Vec<String>,
    pub referenced_exhibits: Vec<ReferencedExhibit>,
    pub position: PositionContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchChunkResult {
    pub chunk_id: String,
    pub paper_id: String,
    pub paper_title: String,
    pub block_type: String,
    pub text: String,
    pub chapter_title: String,
    pub section_title: String,
    pub chunk_idx: u16,
    pub exhibit_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub chunk: SearchChunkResult,
    pub prev: Option<ChunkSummary>,
    pub next: Option<ChunkSummary>,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExhibitSearchResult {
    pub exhibit_id: String,
    pub paper_id: String,
    pub exhibit_type: String,
    pub caption: String,
    pub description: Option<String>,
    pub image_path: Option<String>,
    pub content: Option<String>,
    pub page: Option<u16>,
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
    pub exhibit_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExhibitResult {
    pub exhibit_id: String,
    pub paper_id: String,
    pub exhibit_type: String,
    pub caption: String,
    pub description: Option<String>,
    pub image_path: Option<String>,
    pub content: Option<String>,
    pub page: Option<u16>,
    pub referenced_by: Vec<String>,
    pub first_ref_chunk_id: Option<String>,
    pub ref_count: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlineSection {
    pub section_idx: u16,
    pub section_title: String,
    pub chunk_count: usize,
    /// First sentence of the first paragraph (when `--contents` is used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlineChapter {
    pub chapter_idx: u16,
    pub chapter_title: String,
    pub sections: Vec<OutlineSection>,
    pub exhibit_count: usize,
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
    pub total_exhibits: usize,
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
    pub exhibit_count: usize,
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

/// Input parameters for exhibit search.
pub struct SearchExhibitsParams {
    pub query: String,
    pub paper_ids: Option<Vec<String>>,
    pub filter_exhibit_type: Option<String>,
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
    pub exhibits_added: usize,
}

/// Input parameters for work-level semantic search.
pub struct SearchWorksParams {
    pub query: String,
    pub paper_ids: Option<Vec<String>>,
    pub filter_year_min: Option<u16>,
    pub filter_year_max: Option<u16>,
    pub filter_venue: Option<String>,
    pub filter_tags: Option<Vec<String>>,
    pub limit: u16,
}

/// Input parameters for listing chunks in a paper.
pub struct ListChunksParams {
    pub paper_id: Option<String>,
    pub chapter_idx: Option<u16>,
    pub section_idx: Option<u16>,
    pub limit: u16,
}

/// Input parameters for section-level semantic search.
///
/// Use `depth` to control granularity:
/// - `Some(1)` — chapter-level results (one per chapter)
/// - `Some(2)` — section-level results (one per section)
/// - `None`    — all depths (default, both chapters and sections)
pub struct SearchSectionsParams {
    pub query: String,
    pub paper_ids: Option<Vec<String>>,
    pub chapter_idx: Option<u16>,
    pub depth: Option<u16>,
    pub filter_year_min: Option<u16>,
    pub filter_year_max: Option<u16>,
    pub filter_venue: Option<String>,
    pub filter_tags: Option<Vec<String>>,
    pub limit: u16,
}

/// Input parameters for listing sections in a paper.
///
/// Use `depth` to filter: `Some(1)` for chapters only, `Some(2)` for sections only.
pub struct ListSectionsParams {
    pub paper_id: Option<String>,
    pub depth: Option<u16>,
}

/// Input parameters for chapter-level semantic search.
pub struct SearchChaptersParams {
    pub query: String,
    pub paper_ids: Option<Vec<String>>,
    pub filter_year_min: Option<u16>,
    pub filter_year_max: Option<u16>,
    pub filter_venue: Option<String>,
    pub filter_tags: Option<Vec<String>>,
    pub limit: u16,
}

/// Input parameters for listing chapters in a paper.
pub struct ListChaptersParams {
    pub paper_id: Option<String>,
}

/// Metadata for a single indexed work (paper).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkMetadata {
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    pub tags: Vec<String>,
    pub chunk_count: usize,
    pub exhibit_count: usize,
}

/// One entry in a work-level search result list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkSearchResult {
    pub paper_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub year: Option<u16>,
    pub venue: Option<String>,
    pub score: f32,
    pub chunk_count: usize,
    pub top_chunk: String,
}

/// One row in a chunk list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkListItem {
    pub chunk_id: String,
    pub chapter_idx: u16,
    pub chapter_title: String,
    pub section_idx: u16,
    pub section_title: String,
    pub chunk_idx: u16,
    pub depth: String,
    pub block_type: String,
    pub text_preview: String,
}

/// One entry in a section-level search result list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionSearchResult {
    pub paper_id: String,
    pub paper_title: String,
    pub chapter_idx: u16,
    pub chapter_title: String,
    pub section_idx: u16,
    pub section_title: String,
    pub score: f32,
    pub chunk_count: usize,
    pub top_chunk: String,
}

/// One row in a section list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionListItem {
    pub chapter_idx: u16,
    pub chapter_title: String,
    pub section_idx: u16,
    pub section_title: String,
    pub chunk_count: usize,
}

/// One entry in a chapter-level search result list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterSearchResult {
    pub paper_id: String,
    pub paper_title: String,
    pub chapter_idx: u16,
    pub chapter_title: String,
    pub score: f32,
    pub section_count: usize,
    pub chunk_count: usize,
    pub top_chunk: String,
}

/// One row in a chapter list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterListItem {
    pub chapter_idx: u16,
    pub chapter_title: String,
    pub section_count: usize,
    pub chunk_count: usize,
}
