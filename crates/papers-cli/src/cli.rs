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

#[derive(Parser)]
#[command(name = "papers", about = "Query the OpenAlex academic research database", term_width = 100)]
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
}

#[derive(Subcommand)]
pub enum SelectionCommand {
    /// List all selections (marks active with *)
    List {
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Show selection info and items; activate it (defaults to active selection)
    Get {
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
    /// Remove a paper from a selection
    Remove {
        /// Paper identifier: Zotero key, DOI, OpenAlex ID, or title
        paper: String,
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
    /// Full-text search query
    #[arg(long, short = 's')]
    pub search: Option<String>,

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
    /// List works with optional search/filter/sort
    #[command(after_help = "Advanced filtering: https://docs.openalex.org/api-entities/works/filter-works")]
    List {
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
    /// Extract full text from a work's PDF (tries Zotero, open-access URLs, OpenAlex content API)
    Text {
        /// Work ID (OpenAlex ID, DOI, PMID, or PMCID)
        id: String,
        /// Output raw JSON (includes source metadata)
        #[arg(long)]
        json: bool,
        /// Skip interactive prompt when no PDF is found
        #[arg(long)]
        no_prompt: bool,
        /// Use DataLab Marker API for markdown extraction instead of local pdfium.
        /// Requires DATALAB_API_KEY. Quality: fast | balanced | accurate (default: balanced).
        #[arg(long, value_name = "QUALITY")]
        advanced: Option<AdvancedMode>,
    },
}

#[derive(Subcommand)]
pub enum AuthorCommand {
    /// List authors with optional search/filter/sort
    #[command(after_help = "Advanced filtering: https://docs.openalex.org/api-entities/authors/filter-authors")]
    List {
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
    /// List sources with optional search/filter/sort
    #[command(after_help = "Advanced filtering: https://docs.openalex.org/api-entities/sources/filter-sources")]
    List {
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
    /// List institutions with optional search/filter/sort
    #[command(after_help = "Advanced filtering: https://docs.openalex.org/api-entities/institutions/filter-institutions")]
    List {
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
    /// List topics with optional search/filter/sort
    #[command(after_help = "Advanced filtering: https://docs.openalex.org/api-entities/topics/filter-topics")]
    List {
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
    /// List publishers with optional search/filter/sort
    #[command(after_help = "Advanced filtering: https://docs.openalex.org/api-entities/publishers/filter-publishers")]
    List {
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
    /// List funders with optional search/filter/sort
    #[command(after_help = "Advanced filtering: https://docs.openalex.org/api-entities/funders/filter-funders")]
    List {
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
    /// List domains with optional search/filter/sort
    #[command(after_help = "Example filters: works_count:>100000000, display_name.search:physical\nFilter docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists")]
    List {
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
    /// List fields with optional search/filter/sort
    #[command(after_help = "Example filters: domain.id:domains/3, works_count:>1000000\nFilter docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists")]
    List {
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
    /// List subfields with optional search/filter/sort
    #[command(after_help = "Example filters: field.id:fields/17, works_count:>100000\nFilter docs: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists")]
    List {
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
pub enum ZoteroExtractCommand {
    /// List Zotero items with optional search, showing which have cached extractions
    List {
        /// Quick text search (title, creator, year)
        #[arg(long, short = 's')]
        search: Option<String>,
        /// Results per page (1-100, default 25)
        #[arg(long, short = 'n', default_value = "25")]
        limit: u32,
        /// Output raw JSON
        #[arg(long)]
        json: bool,
    },
    /// Print cached markdown for a paper (must already be extracted)
    Text {
        /// Item key (e.g. LF4MJWZK), DOI, or title search
        query: String,
    },
    /// Print cached structured JSON for a paper (must already be extracted)
    Json {
        /// Item key (e.g. LF4MJWZK), DOI, or title search
        query: String,
    },
    /// Print the local cache directory path for a paper's extraction
    Get {
        /// Item key (e.g. LF4MJWZK), DOI, or title search
        query: String,
    },
    /// Upload all locally-cached extractions that are missing from Zotero
    Upload {
        /// Show what would be uploaded without actually uploading
        #[arg(long)]
        dry_run: bool,
    },
    /// Download all Zotero-backed extractions that are missing from local cache
    Download {
        /// Show what would be downloaded without actually downloading
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
pub enum ZoteroCommand {
    /// Bibliographic items (journalArticle, book, conferencePaper, etc.)
    Work {
        #[command(subcommand)]
        cmd: ZoteroWorkCommand,
    },
    /// DataLab PDF extractions (cached markdown/JSON)
    Extract {
        #[command(subcommand)]
        cmd: ZoteroExtractCommand,
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
        /// Quick text search (title, creator, year)
        #[arg(long, short = 's')]
        search: Option<String>,
        /// Expand search to all fields (default: title/creator/year only)
        #[arg(long)]
        everything: bool,
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
    /// Extract full text from a work's PDF using DataLab Marker (requires DATALAB_API_KEY).
    /// Results are cached locally; subsequent calls return instantly.
    Extract {
        /// Item key (e.g. LF4MJWZK) or a title/creator search string
        key: String,
        /// Quality level: fast, balanced (default), or accurate
        #[arg(long, short = 'm', default_value = "balanced")]
        mode: AdvancedMode,
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
        /// Search by filename or title
        #[arg(long, short = 's')]
        search: Option<String>,
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
        /// Search note content
        #[arg(long, short = 's')]
        search: Option<String>,
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
        /// Search tag names (substring match)
        #[arg(long, short = 'q')]
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
