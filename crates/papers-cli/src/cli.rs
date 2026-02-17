use clap::{Args, Parser, Subcommand};

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

