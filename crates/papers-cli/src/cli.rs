use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

/// Quality level for DataLab Marker API extraction.
///
/// Maps directly to DataLab's processing modes:
/// - `fast`     — quickest turnaround, lower layout accuracy
/// - `balanced` — default DataLab mode, good quality/speed trade-off
/// - `accurate` — highest quality markdown with full layout reconstruction (slowest)
#[derive(ValueEnum, Clone, Debug)]
pub enum AdvancedMode {
    Fast,
    Balanced,
    Accurate,
}

/// Layout debug output mode for the extract command.
#[derive(ValueEnum, Clone, Debug)]
pub enum LayoutDebugArg {
    /// Write annotated page PNGs to layout/
    Images,
    /// Write annotated page PNGs + a vector-overlay debug PDF
    Pdf,
}

#[derive(Parser)]
#[command(
    name = "papers",
    about = "Search, manage, and explore academic papers from the terminal",
    term_width = 100
)]
pub struct Cli {
    #[command(subcommand)]
    pub entity: EntityCommand,
}

#[derive(Subcommand)]
pub enum EntityCommand {
    /// Scholarly works: articles, preprints, datasets, and more
    Work {
        #[command(subcommand)]
        cmd: WorkCommand,
    },
    /// Disambiguated researcher profiles
    Author {
        #[command(subcommand)]
        cmd: AuthorCommand,
    },
    /// Publishing venues: journals, repositories, conferences
    Source {
        #[command(subcommand)]
        cmd: SourceCommand,
    },
    /// Research organizations: universities, hospitals, companies
    Institution {
        #[command(subcommand)]
        cmd: InstitutionCommand,
    },
    /// Research topic hierarchy (domain → field → subfield → topic)
    Topic {
        #[command(subcommand)]
        cmd: TopicCommand,
    },
    /// Publishing organizations (e.g. Elsevier, Springer Nature)
    Publisher {
        #[command(subcommand)]
        cmd: PublisherCommand,
    },
    /// Grant-making organizations (e.g. NIH, NSF, ERC)
    Funder {
        #[command(subcommand)]
        cmd: FunderCommand,
    },
    /// Research domains (broadest level of topic hierarchy, 4 total)
    Domain {
        #[command(subcommand)]
        cmd: DomainCommand,
    },
    /// Academic fields (second level of topic hierarchy, 26 total)
    Field {
        #[command(subcommand)]
        cmd: FieldCommand,
    },
    /// Research subfields (third level of topic hierarchy, ~252 total)
    Subfield {
        #[command(subcommand)]
        cmd: SubfieldCommand,
    },
    /// Your personal Zotero reference library
    Zotero {
        #[command(subcommand)]
        cmd: ZoteroCommand,
    },
    /// Named groups of papers for focused work sessions
    Selection {
        #[command(subcommand)]
        cmd: SelectionCommand,
    },
    /// Local DB index: semantic search over your indexed papers
    Db {
        #[command(subcommand)]
        cmd: DbCommand,
    },
    /// Extract structured content from a PDF using local ONNX models
    Extract {
        /// Path to the input PDF file
        pdf: PathBuf,

        /// Output directory (default: same directory as the PDF)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Extract only this page (1-indexed)
        #[arg(long, short = 'p')]
        page: Option<u32>,

        /// Skip image extraction
        #[arg(long)]
        skip_images: bool,

        /// Write layout debug output: "images" for annotated PNGs, "pdf" for PNGs + debug PDF
        #[arg(long, value_name = "MODE")]
        write_layout: Option<LayoutDebugArg>,
    },
    /// Manage papers CLI configuration
    Config {
        #[command(subcommand)]
        cmd: ConfigCommand,
    },
    /// MCP server for use with AI assistants
    Mcp {
        #[command(subcommand)]
        cmd: McpCommand,
    },
}

#[derive(Subcommand)]
pub enum McpCommand {
    /// Start the MCP server
    Start {
        /// Run with stdio transport (required by MCP clients)
        #[arg(long)]
        stdio: bool,
    },
}

#[derive(Subcommand)]
pub enum DbCommand {
    /// Text chunks: search and retrieve indexed content
    Chunk {
        #[command(subcommand)]
        cmd: DbChunkCommand,
    },
    /// Exhibits: figures, tables, algorithms, and diagrams
    Exhibit {
        #[command(subcommand)]
        cmd: DbExhibitCommand,
    },
    /// Indexed papers: list, add, search, and inspect
    Work {
        #[command(subcommand)]
        cmd: DbWorkCommand,
    },
    /// Sections: read full section content in order
    Section {
        #[command(subcommand)]
        cmd: DbSectionCommand,
    },
    /// Chapters: read full chapter content in order
    Chapter {
        #[command(subcommand)]
        cmd: DbChapterCommand,
    },
    /// Tags across indexed papers
    Tag {
        #[command(subcommand)]
        cmd: DbTagCommand,
    },
    /// Manage the on-disk embedding cache
    Embed {
        #[command(subcommand)]
        cmd: DbEmbedCommand,
    },
}

#[derive(Subcommand)]
pub enum DbChunkCommand {
    /// Semantic search over indexed paper chunks
    Search {
        /// Natural language search query
        query: String,
        /// Scope to papers in a named selection
        #[arg(long)]
        selection: Option<String>,
        /// Scope to a specific paper (DOI, item key, or title search)
        #[arg(long)]
        work: Option<String>,
        /// Scope to a specific chapter (1-based; requires --work)
        #[arg(long)]
        chapter_idx: Option<u16>,
        /// Scope to a specific section (1-based; requires --work and --chapter-idx)
        #[arg(long)]
        section_idx: Option<u16>,
        /// Minimum publication year
        #[arg(long)]
        year_min: Option<u16>,
        /// Maximum publication year
        #[arg(long)]
        year_max: Option<u16>,
        /// Filter by venue name
        #[arg(long)]
        venue: Option<String>,
        /// Filter by tag (repeatable)
        #[arg(long)]
        tag: Option<Vec<String>>,
        /// Granularity: chapter | section | paragraph
        #[arg(long)]
        depth: Option<String>,
        /// Maximum number of results
        #[arg(long, short = 'n', default_value = "5")]
        limit: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Retrieve a specific chunk by ID with neighboring context
    Get {
        /// Chunk ID (e.g. YFACFA8C/ch1/s2/p3)
        chunk_id: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List all chunks in reading order (all papers, or scoped with --work)
    List {
        /// Scope to a specific paper (DOI, item key, or title search)
        #[arg(long)]
        work: Option<String>,
        /// Scope to a chapter (1-based)
        #[arg(long)]
        chapter_idx: Option<u16>,
        /// Scope to a section (1-based within chapter; requires --chapter-idx)
        #[arg(long)]
        section_idx: Option<u16>,
        /// Maximum number of results
        #[arg(long, short = 'n', default_value = "50")]
        limit: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum DbExhibitCommand {
    /// Search for exhibits (figures, tables, algorithms)
    Search {
        /// Natural language description of the exhibit to find
        query: String,
        /// Scope to papers in a named selection
        #[arg(long)]
        selection: Option<String>,
        /// Scope to a specific paper (DOI, item key, or title search)
        #[arg(long)]
        work: Option<String>,
        /// Filter by type: "figure", "table", or "algorithm"
        #[arg(long)]
        exhibit_type: Option<String>,
        /// Maximum number of results
        #[arg(long, short = 'n', default_value = "5")]
        limit: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get exhibit details by ID
    Get {
        /// Exhibit ID (e.g. YFACFA8C/fig3)
        exhibit_id: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum DbWorkCommand {
    /// List indexed papers with optional filters
    List {
        /// Scope to papers in a named selection
        #[arg(long)]
        selection: Option<String>,
        /// Minimum publication year
        #[arg(long)]
        year_min: Option<u16>,
        /// Maximum publication year
        #[arg(long)]
        year_max: Option<u16>,
        /// Filter by venue name
        #[arg(long)]
        venue: Option<String>,
        /// Filter by tag (repeatable)
        #[arg(long)]
        tag: Option<Vec<String>>,
        /// Filter by author name (repeatable, post-filter)
        #[arg(long)]
        author: Option<Vec<String>>,
        /// Sort by: "year" (default) or "title"
        #[arg(long)]
        sort: Option<String>,
        /// Maximum number of results
        #[arg(long, short = 'n', default_value = "50")]
        limit: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get metadata for a single indexed paper (title, authors, year, chunk/exhibit counts)
    Get {
        /// Paper: DOI, item key, or title search
        paper_id: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Semantic search returning one result per matching paper
    Search {
        /// Natural language search query
        query: String,
        /// Scope to papers in a named selection
        #[arg(long)]
        selection: Option<String>,
        /// Minimum publication year
        #[arg(long)]
        year_min: Option<u16>,
        /// Maximum publication year
        #[arg(long)]
        year_max: Option<u16>,
        /// Filter by venue name
        #[arg(long)]
        venue: Option<String>,
        /// Filter by tag (repeatable)
        #[arg(long)]
        tag: Option<Vec<String>>,
        /// Maximum number of results
        #[arg(long, short = 'n', default_value = "5")]
        limit: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Index a paper (or all papers) from local DataLab cache into the RAG database.
    /// If the DataLab cache is missing for a paper, extraction runs automatically
    /// via DataLab Marker API (requires DATALAB_API_KEY and a local Zotero PDF).
    Add {
        /// Paper: item key (e.g. LF4MJWZK), DOI, or title search; omit with --all
        work: Option<String>,
        /// Index all papers in the DataLab cache
        #[arg(long)]
        all: bool,
        /// Add tags to this paper (repeatable)
        #[arg(long)]
        tag: Option<Vec<String>>,
        /// Force re-index even if already indexed
        #[arg(long)]
        force: bool,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
        /// Quality level for DataLab extraction when cache is missing (ignored if already cached)
        #[arg(long, short = 'm', default_value = "balanced")]
        mode: AdvancedMode,
        /// Re-run DataLab extraction even if a local cache already exists
        #[arg(long)]
        force_extract: bool,
    },
    /// Remove a paper from the RAG index (deletes all chunks and exhibits)
    Remove {
        /// Paper: DOI, item key, or title search
        paper_id: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Show the table of contents for an indexed paper
    Outline {
        /// Paper: DOI, item key, or title search
        paper_id: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Print cached DataLab extraction (markdown by default; use --json for structured JSON)
    Extract {
        /// Paper: item key (e.g. LF4MJWZK), DOI, or title search
        work: String,
        /// Output structured JSON instead of markdown
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum DbSectionCommand {
    /// Semantic search returning one result per matching section
    Search {
        /// Natural language search query
        query: String,
        /// Scope to papers in a named selection
        #[arg(long)]
        selection: Option<String>,
        /// Scope to a specific paper (DOI, item key, or title search)
        #[arg(long)]
        work: Option<String>,
        /// Scope to a chapter (1-based; requires --work)
        #[arg(long)]
        chapter_idx: Option<u16>,
        /// Minimum publication year
        #[arg(long)]
        year_min: Option<u16>,
        /// Maximum publication year
        #[arg(long)]
        year_max: Option<u16>,
        /// Filter by venue name
        #[arg(long)]
        venue: Option<String>,
        /// Filter by tag (repeatable)
        #[arg(long)]
        tag: Option<Vec<String>>,
        /// Maximum number of results
        #[arg(long, short = 'n', default_value = "5")]
        limit: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List all sections as a flat outline (all papers, or scoped with --work)
    List {
        /// Scope to a specific paper (DOI, item key, or title search)
        #[arg(long)]
        work: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Fetch all chunks in a section in reading order
    Get {
        /// Paper: DOI, item key, or title search
        paper_id: String,
        /// Chapter index (1-based)
        #[arg(long)]
        chapter_idx: u16,
        /// Section index (1-based within chapter)
        #[arg(long)]
        section_idx: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum DbChapterCommand {
    /// Semantic search returning one result per matching chapter
    Search {
        /// Natural language search query
        query: String,
        /// Scope to papers in a named selection
        #[arg(long)]
        selection: Option<String>,
        /// Scope to a specific paper (DOI, item key, or title search)
        #[arg(long)]
        work: Option<String>,
        /// Minimum publication year
        #[arg(long)]
        year_min: Option<u16>,
        /// Maximum publication year
        #[arg(long)]
        year_max: Option<u16>,
        /// Filter by venue name
        #[arg(long)]
        venue: Option<String>,
        /// Filter by tag (repeatable)
        #[arg(long)]
        tag: Option<Vec<String>>,
        /// Maximum number of results
        #[arg(long, short = 'n', default_value = "5")]
        limit: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List all chapters as a flat outline (all papers, or scoped with --work)
    List {
        /// Scope to a specific paper (DOI, item key, or title search)
        #[arg(long)]
        work: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Fetch all chunks in a chapter in reading order
    Get {
        /// Paper: DOI, item key, or title search
        paper_id: String,
        /// Chapter index (1-based)
        #[arg(long)]
        chapter_idx: u16,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum DbTagCommand {
    /// List all tags across indexed papers with counts
    List {
        /// Scope to papers in a named selection
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum SelectionCommand {
    /// List all selections (marks active with *)
    List {
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Activate a selection and print a one-line summary
    Set {
        /// Selection name or 1-based index (omit to use active selection)
        name: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Create a new selection and activate it
    Create {
        /// Selection name (alphanumeric, - and _ only)
        name: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Delete a selection (deactivates if currently active)
    Delete {
        /// Selection name or 1-based index
        name: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Add a paper to a selection (Zotero optional; stores rich metadata)
    Add {
        /// Paper identifier: Zotero key, DOI, OpenAlex ID, or title
        paper: String,
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Remove a paper from a selection (accepts 1-based index from `status`)
    Remove {
        /// Paper identifier: Zotero key, DOI, OpenAlex ID, title, or 1-based index
        paper: String,
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Per-paper status: Zotero, PDF, extracted, DB
    Status {
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Download open-access PDFs for entries missing a PDF
    Find {
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Open DOI URLs in browser for papers without OA PDF
        #[arg(long)]
        open: bool,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Sync selection to Zotero (create items, upload PDFs, upload extractions, add to collection)
    Sync {
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Skip confirmation prompt
        #[arg(long, short = 'y')]
        yes: bool,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Batch DB operations on the active selection
    Db {
        #[command(subcommand)]
        cmd: SelectionDbCommand,
    },
    /// Zotero collection operations on the active selection
    Collection {
        #[command(subcommand)]
        cmd: SelectionCollectionCommand,
    },
    /// Merge another selection's entries into the active selection
    Merge {
        /// Source selection name or 1-based index
        source: String,
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Rename the active selection
    Rename {
        /// New selection name (alphanumeric, - and _ only)
        new_name: String,
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum SelectionDbCommand {
    /// Batch ingest all selection entries into the DB
    Add {
        /// Skip entries without extraction cache instead of erroring
        #[arg(long)]
        allow_skip: bool,
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Batch remove all selection entries from the DB
    Remove {
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum SelectionCollectionCommand {
    /// Import a Zotero collection into the active selection
    Add {
        /// Zotero collection key (8 chars) or name search
        collection: String,
        /// Target selection name or index (default: active selection)
        #[arg(long)]
        selection: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

/// Shared args for all list commands
#[derive(Args, Clone)]
pub struct ListArgs {
    /// Filter expression (comma-separated AND conditions, pipe for OR)
    #[arg(long, short = 'f')]
    pub filter: Option<String>,

    /// Sort field with optional :desc (e.g. "cited_by_count:desc")
    #[arg(long)]
    pub sort: Option<String>,

    /// Results per page
    #[arg(long, short = 'n', default_value = "10")]
    pub per_page: u32,

    /// Page number for offset pagination
    #[arg(long)]
    pub page: Option<u32>,

    /// Cursor for cursor-based pagination (use "*" to start)
    #[arg(long)]
    pub cursor: Option<String>,

    /// Random sample of N results
    #[arg(long)]
    pub sample: Option<u32>,

    /// Seed for reproducible sampling
    #[arg(long)]
    pub seed: Option<u32>,

    /// Output raw JSON instead of formatted text
    #[arg(long)]
    pub json: bool,
}

/// Shorthand filter flags for `work list`.
///
/// These resolve to real OpenAlex filter expressions. ID-based filters accept
/// either an OpenAlex entity ID or a search string (resolved to the top result
/// by citation count).
#[derive(Args, Clone, Default)]
pub struct WorkFilterArgs {
    /// Filter by author name or OpenAlex author ID (e.g. "einstein", "Albert Einstein", or "A5108093963")
    #[arg(long)]
    pub author: Option<String>,

    /// Filter by topic name or OpenAlex topic ID (e.g. "deep learning",
    /// "computer graphics and visualization techniques", "advanced numerical analysis techniques",
    /// or "T10320"). Run `papers topic list -s <query>` to browse topics.
    #[arg(long)]
    pub topic: Option<String>,

    /// Filter by domain name or ID. The 4 domains: 1 Life Sciences, 2 Social Sciences,
    /// 3 Physical Sciences, 4 Health Sciences (e.g. "physical sciences" or "3")
    #[arg(long)]
    pub domain: Option<String>,

    /// Filter by field name or ID (e.g. "computer science", "engineering", "mathematics", or "17").
    /// Run `papers field list` to browse all 26 fields.
    #[arg(long)]
    pub field: Option<String>,

    /// Filter by subfield name or ID (e.g. "artificial intelligence", "computer graphics",
    /// "computational geometry", or "1702"). Run `papers subfield list -s <query>` or
    /// `papers subfield autocomplete <query>` to discover subfields.
    #[arg(long)]
    pub subfield: Option<String>,

    /// Filter by publisher name or ID. Supports pipe-separated OR (e.g. "acm", "acm|ieee", "P4310319798")
    #[arg(long)]
    pub publisher: Option<String>,

    /// Filter by source (journal/conference) name or ID (e.g. "siggraph", "nature", or "S131921510")
    #[arg(long)]
    pub source: Option<String>,

    /// Filter by institution name or ID. Uses lineage for broad matching (e.g. "mit" or "I136199984")
    #[arg(long)]
    pub institution: Option<String>,

    /// Filter by publication year (e.g. "2024", ">2008", "2008-2024", "2020|2021")
    #[arg(long)]
    pub year: Option<String>,

    /// Filter by citation count (e.g. ">100", "10-50")
    #[arg(long)]
    pub citations: Option<String>,

    /// Filter by country code of author institutions (e.g. "US", "GB")
    #[arg(long)]
    pub country: Option<String>,

    /// Filter by continent of author institutions (e.g. "europe", "asia")
    #[arg(long)]
    pub continent: Option<String>,

    /// Filter by work type (e.g. "article", "preprint", "dataset")
    #[arg(long = "type")]
    pub entity_type: Option<String>,

    /// Filter for open access works only
    #[arg(long)]
    pub open: bool,
}

/// Shorthand filter flags for `author list`.
#[derive(Args, Clone, Default)]
pub struct AuthorFilterArgs {
    /// Filter by institution name or ID (e.g. "harvard", "mit", or "I136199984")
    #[arg(long)]
    pub institution: Option<String>,

    /// Filter by country code of last known institution (e.g. "US", "GB")
    #[arg(long)]
    pub country: Option<String>,

    /// Filter by continent of last known institution (e.g. "europe", "asia")
    #[arg(long)]
    pub continent: Option<String>,

    /// Filter by citation count (e.g. ">1000", "100-500")
    #[arg(long)]
    pub citations: Option<String>,

    /// Filter by works count (e.g. ">500", "100-200")
    #[arg(long)]
    pub works: Option<String>,

    /// Filter by h-index (e.g. ">50", "10-20"). The h-index measures sustained
    /// research impact: an author with h-index h has h works each cited at least
    /// h times.
    #[arg(long)]
    pub h_index: Option<String>,
}

/// Shorthand filter flags for `source list`.
#[derive(Args, Clone, Default)]
pub struct SourceFilterArgs {
    /// Filter by publisher name or ID (e.g. "springer", "P4310319798")
    #[arg(long)]
    pub publisher: Option<String>,

    /// Filter by country code (e.g. "US", "GB")
    #[arg(long)]
    pub country: Option<String>,

    /// Filter by continent (e.g. "europe")
    #[arg(long)]
    pub continent: Option<String>,

    /// Filter by source type (e.g. "journal", "repository", "conference")
    #[arg(long = "type")]
    pub entity_type: Option<String>,

    /// Filter for open access sources only
    #[arg(long)]
    pub open: bool,

    /// Filter by citation count (e.g. ">10000")
    #[arg(long)]
    pub citations: Option<String>,

    /// Filter by works count (e.g. ">100000")
    #[arg(long)]
    pub works: Option<String>,
}

/// Shorthand filter flags for `institution list`.
#[derive(Args, Clone, Default)]
pub struct InstitutionFilterArgs {
    /// Filter by country code (e.g. "US", "GB")
    #[arg(long)]
    pub country: Option<String>,

    /// Filter by continent (e.g. "europe", "asia")
    #[arg(long)]
    pub continent: Option<String>,

    /// Filter by institution type (e.g. "education", "healthcare", "company")
    #[arg(long = "type")]
    pub entity_type: Option<String>,

    /// Filter by citation count (e.g. ">100000")
    #[arg(long)]
    pub citations: Option<String>,

    /// Filter by works count (e.g. ">100000")
    #[arg(long)]
    pub works: Option<String>,
}

/// Shorthand filter flags for `topic list`.
#[derive(Args, Clone, Default)]
pub struct TopicFilterArgs {
    /// Filter by domain name or ID (e.g. "life sciences", "3")
    #[arg(long)]
    pub domain: Option<String>,

    /// Filter by field name or ID (e.g. "computer science", "17")
    #[arg(long)]
    pub field: Option<String>,

    /// Filter by subfield name or ID (e.g. "artificial intelligence", "1702")
    #[arg(long)]
    pub subfield: Option<String>,

    /// Filter by citation count (e.g. ">1000")
    #[arg(long)]
    pub citations: Option<String>,

    /// Filter by works count (e.g. ">1000")
    #[arg(long)]
    pub works: Option<String>,
}

/// Shorthand filter flags for `publisher list`.
#[derive(Args, Clone, Default)]
pub struct PublisherFilterArgs {
    /// Filter by country code (e.g. "US", "GB"). Note: uses `country_codes` (plural).
    #[arg(long)]
    pub country: Option<String>,

    /// Filter by continent (e.g. "europe")
    #[arg(long)]
    pub continent: Option<String>,

    /// Filter by citation count (e.g. ">10000")
    #[arg(long)]
    pub citations: Option<String>,

    /// Filter by works count (e.g. ">1000000")
    #[arg(long)]
    pub works: Option<String>,
}

/// Shorthand filter flags for `funder list`.
#[derive(Args, Clone, Default)]
pub struct FunderFilterArgs {
    /// Filter by country code (e.g. "US", "GB")
    #[arg(long)]
    pub country: Option<String>,

    /// Filter by continent (e.g. "europe")
    #[arg(long)]
    pub continent: Option<String>,

    /// Filter by citation count (e.g. ">10000")
    #[arg(long)]
    pub citations: Option<String>,

    /// Filter by works count (e.g. ">100000")
    #[arg(long)]
    pub works: Option<String>,
}

/// Shorthand filter flags for `domain list`.
#[derive(Args, Clone, Default)]
pub struct DomainFilterArgs {
    /// Filter by works count (e.g. ">100000000")
    #[arg(long)]
    pub works: Option<String>,
}

/// Shorthand filter flags for `field list`.
#[derive(Args, Clone, Default)]
pub struct FieldFilterArgs {
    /// Filter by domain name or ID (e.g. "life sciences", "3")
    #[arg(long)]
    pub domain: Option<String>,

    /// Filter by works count (e.g. ">1000000")
    #[arg(long)]
    pub works: Option<String>,
}

/// Shorthand filter flags for `subfield list`.
#[derive(Args, Clone, Default)]
pub struct SubfieldFilterArgs {
    /// Filter by domain name or ID (e.g. "physical sciences", "3")
    #[arg(long)]
    pub domain: Option<String>,

    /// Filter by field name or ID (e.g. "computer science", "17")
    #[arg(long)]
    pub field: Option<String>,

    /// Filter by works count (e.g. ">1000000")
    #[arg(long)]
    pub works: Option<String>,
}

#[derive(Subcommand)]
pub enum WorkCommand {
    /// List works with optional filter/sort
    #[command(
        after_help = "Advanced filtering: https://docs.openalex.org/api-entities/works/filter-works"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        work_filters: WorkFilterArgs,
    },
    /// Full-text search for works (title, abstract, etc.)
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        work_filters: WorkFilterArgs,
    },
    /// Get a single work by ID (OpenAlex ID, DOI, PMID, or PMCID)
    Get {
        /// Work ID
        id: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Type-ahead search for works by title
    Autocomplete {
        /// Search query
        query: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// AI semantic search for similar works (requires OPENALEX_KEY)
    Find {
        /// Text to find similar works for
        query: String,
        /// Number of results (1-100)
        #[arg(long, short = 'n')]
        count: Option<u32>,
        /// Filter expression (https://docs.openalex.org/api-entities/works/filter-works)
        #[arg(long, short = 'f')]
        filter: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum AuthorCommand {
    /// List authors with optional filter/sort
    #[command(
        after_help = "Advanced filtering: https://docs.openalex.org/api-entities/authors/filter-authors"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: AuthorFilterArgs,
    },
    /// Full-text search for authors
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: AuthorFilterArgs,
    },
    /// Get a single author by ID (OpenAlex ID or ORCID)
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Type-ahead search for authors
    Autocomplete {
        query: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum SourceCommand {
    /// List sources with optional filter/sort
    #[command(
        after_help = "Advanced filtering: https://docs.openalex.org/api-entities/sources/filter-sources"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: SourceFilterArgs,
    },
    /// Full-text search for sources
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: SourceFilterArgs,
    },
    /// Get a single source by ID (OpenAlex ID or ISSN)
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Type-ahead search for sources
    Autocomplete {
        query: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum InstitutionCommand {
    /// List institutions with optional filter/sort
    #[command(
        after_help = "Advanced filtering: https://docs.openalex.org/api-entities/institutions/filter-institutions"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: InstitutionFilterArgs,
    },
    /// Full-text search for institutions
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: InstitutionFilterArgs,
    },
    /// Get a single institution by ID (OpenAlex ID or ROR)
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Type-ahead search for institutions
    Autocomplete {
        query: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum TopicCommand {
    /// List topics with optional filter/sort
    #[command(
        after_help = "Advanced filtering: https://docs.openalex.org/api-entities/topics/filter-topics"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: TopicFilterArgs,
    },
    /// Full-text search for topics
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: TopicFilterArgs,
    },
    /// Get a single topic by OpenAlex ID
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum PublisherCommand {
    /// List publishers with optional filter/sort
    #[command(
        after_help = "Advanced filtering: https://docs.openalex.org/api-entities/publishers/filter-publishers"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: PublisherFilterArgs,
    },
    /// Full-text search for publishers
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: PublisherFilterArgs,
    },
    /// Get a single publisher by OpenAlex ID
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Type-ahead search for publishers
    Autocomplete {
        query: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum FunderCommand {
    /// List funders with optional filter/sort
    #[command(
        after_help = "Advanced filtering: https://docs.openalex.org/api-entities/funders/filter-funders"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: FunderFilterArgs,
    },
    /// Full-text search for funders
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: FunderFilterArgs,
    },
    /// Get a single funder by OpenAlex ID
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Type-ahead search for funders
    Autocomplete {
        query: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum DomainCommand {
    /// List domains with optional filter/sort
    #[command(
        after_help = "Example filters: works_count:>100000000, display_name.search:physical\nFilter docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: DomainFilterArgs,
    },
    /// Full-text search for domains
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: DomainFilterArgs,
    },
    /// Get a single domain by numeric ID (1-4)
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum FieldCommand {
    /// List fields with optional filter/sort
    #[command(
        after_help = "Example filters: domain.id:domains/3, works_count:>1000000\nFilter docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: FieldFilterArgs,
    },
    /// Full-text search for fields
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: FieldFilterArgs,
    },
    /// Get a single field by numeric ID (e.g. 17)
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum SubfieldCommand {
    /// List subfields with optional filter/sort
    #[command(
        after_help = "Example filters: field.id:fields/17, works_count:>100000\nFilter docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists"
    )]
    List {
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: SubfieldFilterArgs,
    },
    /// Full-text search for subfields
    Search {
        /// Search query
        query: String,
        #[command(flatten)]
        args: ListArgs,
        #[command(flatten)]
        filters: SubfieldFilterArgs,
    },
    /// Get a single subfield by numeric ID (e.g. 1702)
    Get {
        id: String,
        #[arg(long)]
        json: bool,
    },
    /// Type-ahead search for subfields
    Autocomplete {
        query: String,
        #[arg(long)]
        json: bool,
    },
}

// ── Zotero commands ────────────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum ZoteroCommand {
    /// Bibliographic items (journalArticle, book, conferencePaper, etc.)
    Work {
        #[command(subcommand)]
        cmd: ZoteroWorkCommand,
    },
    /// File attachments (PDFs, snapshots, links)
    Attachment {
        #[command(subcommand)]
        cmd: ZoteroAttachmentCommand,
    },
    /// PDF reader highlights, comments, and marks
    Annotation {
        #[command(subcommand)]
        cmd: ZoteroAnnotationCommand,
    },
    /// User-written text notes
    Note {
        #[command(subcommand)]
        cmd: ZoteroNoteCommand,
    },
    /// Collections (folders for organizing items)
    Collection {
        #[command(subcommand)]
        cmd: ZoteroCollectionCommand,
    },
    /// Tags (labels applied to items)
    Tag {
        #[command(subcommand)]
        cmd: ZoteroTagCommand,
    },
    /// Saved searches
    Search {
        #[command(subcommand)]
        cmd: ZoteroSearchCommand,
    },
    /// Zotero groups (shared libraries)
    Group {
        #[command(subcommand)]
        cmd: ZoteroGroupCommand,
    },
    /// Library settings (tagColors, feeds, etc.)
    Setting {
        #[command(subcommand)]
        cmd: ZoteroSettingCommand,
    },
    /// Objects deleted from the library
    Deleted {
        #[command(subcommand)]
        cmd: ZoteroDeletedCommand,
    },
    /// API key identity and permissions
    Permission {
        #[command(subcommand)]
        cmd: ZoteroPermissionCommand,
    },
}

#[derive(Subcommand)]
pub enum ZoteroWorkCommand {
    /// List bibliographic items (excludes notes, attachments, annotations)
    List {
        /// Filter by tag; use || for OR, - prefix for NOT
        #[arg(long, short = 't')]
        tag: Option<String>,
        /// Filter by bibliographic type (e.g. journalArticle, book, conferencePaper)
        #[arg(long = "type")]
        type_: Option<String>,
        /// Sort field (dateAdded, dateModified, title, creator, date, publisher, publicationTitle)
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Only items modified after this library version
        #[arg(long)]
        since: Option<u64>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Search bibliographic items by title, creator, year, etc.
    Search {
        /// Search query (title, creator, year)
        query: String,
        /// Expand search to all fields
        #[arg(long)]
        everything: bool,
        /// Filter by tag; use || for OR, - prefix for NOT
        #[arg(long, short = 't')]
        tag: Option<String>,
        /// Filter by bibliographic type (e.g. journalArticle, book, conferencePaper)
        #[arg(long = "type")]
        type_: Option<String>,
        /// Sort field
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a single bibliographic item by Zotero key or title search
    Get {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List collections the work belongs to
    Collections {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List notes attached to a work
    Notes {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Results per page
        #[arg(long, short = 'n')]
        limit: Option<u32>,
        /// Pagination offset
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List attachments of a work
    Attachments {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Results per page
        #[arg(long, short = 'n')]
        limit: Option<u32>,
        /// Pagination offset
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List all annotations across all PDFs of a work
    Annotations {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List tags attached to a work
    Tags {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Filter tags by name (substring match)
        #[arg(long, short = 'q')]
        search: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n')]
        limit: Option<u32>,
        /// Pagination offset
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get Zotero's indexed full text for a work's primary PDF attachment
    Text {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get the CDN view URL for a work's primary PDF attachment
    ViewUrl {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
    },
    /// Download the PDF for a work's primary attachment (writes to file or stdout)
    View {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Output path (use - for stdout)
        #[arg(long, short = 'o', required = true)]
        output: String,
    },
}

#[derive(Subcommand)]
pub enum ZoteroAttachmentCommand {
    /// List all attachment items in the library
    List {
        /// Sort field (dateAdded, dateModified, title, accessDate)
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Search attachments by filename or title
    Search {
        /// Search query (filename or title)
        query: String,
        /// Sort field (dateAdded, dateModified, title, accessDate)
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a single attachment item by key or title search
    Get {
        /// Attachment key (e.g. LF4MJWZK) or a title/filename search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Download the file for an attachment (only works for imported_file/imported_url)
    File {
        /// Attachment key (e.g. LF4MJWZK) or a title/filename search string
        key: String,
        /// Output path (use - for stdout)
        #[arg(long, short = 'o', required = true)]
        output: String,
    },
    /// Get the CDN view URL for an attachment
    Url {
        /// Attachment key (e.g. LF4MJWZK) or a title/filename search string
        key: String,
    },
}

#[derive(Subcommand)]
pub enum ZoteroAnnotationCommand {
    /// List all annotation items in the library
    List {
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a single annotation by key or search string
    Get {
        /// Annotation key (e.g. LF4MJWZK) or a search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroNoteCommand {
    /// List all note items in the library
    List {
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Search note items by content
    Search {
        /// Search query (note content)
        query: String,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a single note by key or search string
    Get {
        /// Note key (e.g. LF4MJWZK) or a search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroCollectionCommand {
    /// List collections in the library
    List {
        /// Sort field (title, dateAdded, dateModified)
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Only root-level collections
        #[arg(long)]
        top: bool,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a single collection by key or name search
    Get {
        /// Collection key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List bibliographic works in a collection
    Works {
        /// Collection key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Quick text search
        #[arg(long, short = 's')]
        search: Option<String>,
        /// Expand search to all fields (default: title/creator/year only)
        #[arg(long)]
        everything: bool,
        /// Filter by tag
        #[arg(long, short = 't')]
        tag: Option<String>,
        /// Filter by bibliographic type
        #[arg(long = "type")]
        type_: Option<String>,
        /// Sort field
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List attachments in a collection
    Attachments {
        /// Collection key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Sort field
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List notes in a collection
    Notes {
        /// Collection key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Search note content
        #[arg(long, short = 's')]
        search: Option<String>,
        /// Sort field
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List annotations on PDFs in a collection
    Annotations {
        /// Collection key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List sub-collections of a collection
    Subcollections {
        /// Collection key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Sort field (title, dateAdded, dateModified)
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// List tags on items within a collection
    Tags {
        /// Collection key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Filter tags by name (substring match)
        #[arg(long, short = 'q')]
        search: Option<String>,
        /// Results per page
        #[arg(long, short = 'n')]
        limit: Option<u32>,
        /// Pagination offset
        #[arg(long)]
        start: Option<u32>,
        /// Only tags on top-level items in the collection
        #[arg(long)]
        top: bool,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroTagCommand {
    /// List tags from the global library tag index (with per-tag item counts)
    List {
        /// Sort field
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Only tags appearing on top-level items
        #[arg(long)]
        top: bool,
        /// Only tags appearing on trashed items
        #[arg(long)]
        trash: bool,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Search tags by name (substring match)
    Search {
        /// Tag name query (substring match)
        query: String,
        /// Sort field
        #[arg(long)]
        sort: Option<String>,
        /// Sort direction: asc or desc
        #[arg(long)]
        direction: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Pagination offset (0-based)
        #[arg(long)]
        start: Option<u32>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a specific tag by name
    Get {
        /// Tag name
        name: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroSearchCommand {
    /// List all saved searches in the library
    List {
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a single saved search by key or name search
    Get {
        /// Search key (e.g. AB12CDEF) or a name search string
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroGroupCommand {
    /// List all Zotero groups accessible to the current user
    List {
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroSettingCommand {
    /// List all library settings (tagColors, etc.)
    List {
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Get a single library setting by key
    Get {
        /// Setting key (e.g. tagColors)
        key: String,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroDeletedCommand {
    /// List objects deleted from the library since a given version
    List {
        /// Library version to sync from (0 = all deletions)
        #[arg(long, default_value = "0")]
        since: u64,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroPermissionCommand {
    /// List permissions and identity for the current API key
    List {
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub enum DbEmbedCommand {
    /// List cached embeddings for a paper (or all papers when work is omitted)
    List {
        /// Paper identifier (Zotero key)
        work: Option<String>,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Compute and cache embeddings for a paper (or all papers)
    Add {
        /// Paper identifier (Zotero key)
        work: Option<String>,
        /// Embedding model (defaults to configured model)
        #[arg(long)]
        model: Option<String>,
        /// Re-embed even if the cache already exists
        #[arg(long)]
        force: bool,
    },
    /// Remove cached embeddings for a paper (or all papers)
    Delete {
        /// Paper identifier (Zotero key)
        work: Option<String>,
        /// Embedding model (defaults to configured model)
        #[arg(long)]
        model: Option<String>,
    },
}

#[derive(Subcommand)]
pub enum ConfigCommand {
    /// Set a configuration value
    Set {
        #[command(subcommand)]
        cmd: ConfigSetCommand,
    },
}

#[derive(Subcommand)]
pub enum ConfigSetCommand {
    /// Set the default embedding model
    Model {
        /// Model name (e.g. embedding-gemma-300m)
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(args).expect("parse failed")
    }

    #[test]
    fn test_parse_config_set_model_valid() {
        let cli = parse(&["papers", "config", "set", "model", "embedding-gemma-300m"]);
        match cli.entity {
            EntityCommand::Config {
                cmd:
                    ConfigCommand::Set {
                        cmd: ConfigSetCommand::Model { name },
                    },
            } => assert_eq!(name, "embedding-gemma-300m"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_embed_list_no_args() {
        let cli = parse(&["papers", "db", "embed", "list"]);
        match cli.entity {
            EntityCommand::Db {
                cmd:
                    DbCommand::Embed {
                        cmd: DbEmbedCommand::List { work, json: _ },
                    },
            } => assert!(work.is_none()),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_embed_list_with_work() {
        let cli = parse(&["papers", "db", "embed", "list", "ABC123"]);
        match cli.entity {
            EntityCommand::Db {
                cmd:
                    DbCommand::Embed {
                        cmd: DbEmbedCommand::List { work, json: _ },
                    },
            } => assert_eq!(work.as_deref(), Some("ABC123")),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_embed_add_no_model() {
        let cli = parse(&["papers", "db", "embed", "add", "KEY1"]);
        match cli.entity {
            EntityCommand::Db {
                cmd:
                    DbCommand::Embed {
                        cmd: DbEmbedCommand::Add { work, model, force },
                    },
            } => {
                assert_eq!(work.as_deref(), Some("KEY1"));
                assert!(model.is_none());
                assert!(!force);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_embed_add_explicit_model() {
        let cli = parse(&[
            "papers",
            "db",
            "embed",
            "add",
            "KEY1",
            "--model",
            "embedding-gemma-300m",
        ]);
        match cli.entity {
            EntityCommand::Db {
                cmd:
                    DbCommand::Embed {
                        cmd: DbEmbedCommand::Add { model, .. },
                    },
            } => assert_eq!(model.as_deref(), Some("embedding-gemma-300m")),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_embed_add_force_flag() {
        let cli = parse(&["papers", "db", "embed", "add", "KEY1", "--force"]);
        match cli.entity {
            EntityCommand::Db {
                cmd:
                    DbCommand::Embed {
                        cmd: DbEmbedCommand::Add { force, .. },
                    },
            } => assert!(force),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_embed_delete() {
        let cli = parse(&["papers", "db", "embed", "delete", "KEY1"]);
        match cli.entity {
            EntityCommand::Db {
                cmd:
                    DbCommand::Embed {
                        cmd: DbEmbedCommand::Delete { work, model },
                    },
            } => {
                assert_eq!(work.as_deref(), Some("KEY1"));
                assert!(model.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_embed_delete_with_model() {
        let cli = parse(&[
            "papers",
            "db",
            "embed",
            "delete",
            "KEY1",
            "--model",
            "embedding-gemma-300m",
        ]);
        match cli.entity {
            EntityCommand::Db {
                cmd:
                    DbCommand::Embed {
                        cmd: DbEmbedCommand::Delete { model, .. },
                    },
            } => assert_eq!(model.as_deref(), Some("embedding-gemma-300m")),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_chunk_search() {
        let cli = parse(&["papers", "db", "chunk", "search", "neural rendering"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Chunk { cmd: DbChunkCommand::Search { query, .. } },
            } => assert_eq!(query, "neural rendering"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_chunk_get() {
        let cli = parse(&["papers", "db", "chunk", "get", "YFACFA8C/ch1/s2/p3"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Chunk { cmd: DbChunkCommand::Get { chunk_id, .. } },
            } => assert_eq!(chunk_id, "YFACFA8C/ch1/s2/p3"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_work_list() {
        let cli = parse(&["papers", "db", "work", "list"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Work { cmd: DbWorkCommand::List { .. } },
            } => {}
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_work_add_single() {
        let cli = parse(&["papers", "db", "work", "add", "YFACFA8C"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Work { cmd: DbWorkCommand::Add { work, all, .. } },
            } => {
                assert_eq!(work.as_deref(), Some("YFACFA8C"));
                assert!(!all);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_work_add_all() {
        let cli = parse(&["papers", "db", "work", "add", "--all"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Work { cmd: DbWorkCommand::Add { work, all, .. } },
            } => {
                assert!(work.is_none());
                assert!(all);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_work_add_with_mode() {
        let cli = parse(&["papers", "db", "work", "add", "YFACFA8C", "-m", "accurate"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Work { cmd: DbWorkCommand::Add { work, mode, .. } },
            } => {
                assert_eq!(work.as_deref(), Some("YFACFA8C"));
                assert!(matches!(mode, AdvancedMode::Accurate));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_work_add_force_extract() {
        let cli = parse(&["papers", "db", "work", "add", "YFACFA8C", "--force-extract"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Work { cmd: DbWorkCommand::Add { work, force_extract, .. } },
            } => {
                assert_eq!(work.as_deref(), Some("YFACFA8C"));
                assert!(force_extract);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_work_add_mode_default_balanced() {
        let cli = parse(&["papers", "db", "work", "add", "YFACFA8C"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Work { cmd: DbWorkCommand::Add { mode, .. } },
            } => {
                assert!(matches!(mode, AdvancedMode::Balanced));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_parse_db_tag_list() {
        let cli = parse(&["papers", "db", "tag", "list"]);
        match cli.entity {
            EntityCommand::Db {
                cmd: DbCommand::Tag { cmd: DbTagCommand::List { .. } },
            } => {}
            _ => panic!("wrong variant"),
        }
    }
}
