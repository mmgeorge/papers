use serde::Deserialize;

// ── Shared nested types ────────────────────────────────────────────────

/// Impact metrics summary for an entity (author, source, institution, publisher,
/// or funder).
///
/// ```json
/// {"2yr_mean_citedness": 4.065, "h_index": 20, "i10_index": 30}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct SummaryStats {
    /// Mean number of citations received by this entity's works in the last two
    /// years. Serialized as `"2yr_mean_citedness"` in JSON.
    #[serde(rename = "2yr_mean_citedness")]
    pub two_yr_mean_citedness: Option<f64>,

    /// The [h-index](https://en.wikipedia.org/wiki/H-index): the largest number
    /// *h* such that *h* works have each been cited at least *h* times.
    pub h_index: Option<i64>,

    /// The [i10-index](https://en.wikipedia.org/wiki/Author-level_metrics#i-10-index):
    /// the number of works with at least 10 citations.
    pub i10_index: Option<i64>,
}

/// Annual publication and citation counts for an entity.
///
/// Returned in the `counts_by_year` array on authors, sources, institutions,
/// publishers, funders, and works. Each entry represents one calendar year.
///
/// ```json
/// {"year": 2023, "works_count": 42, "oa_works_count": 30, "cited_by_count": 150}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CountsByYear {
    /// Calendar year.
    pub year: i32,

    /// Number of works published in this year.
    pub works_count: Option<i64>,

    /// Number of open-access works published in this year. Present on authors,
    /// sources, institutions, and funders.
    pub oa_works_count: Option<i64>,

    /// Number of times this entity's works were cited in this year.
    pub cited_by_count: Option<i64>,
}

/// A role that an organization plays in the research ecosystem.
///
/// Some organizations serve multiple roles: the same entity can appear as a
/// funder, publisher, and institution. For example, the Wellcome Trust is both a
/// funder and a publisher.
///
/// ```json
/// {"role": "funder", "id": "https://openalex.org/F4320332161", "works_count": 3253779}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Role {
    /// Role type: one of `"institution"`, `"funder"`, or `"publisher"`.
    pub role: Option<String>,

    /// OpenAlex ID of this entity in the given role.
    pub id: Option<String>,

    /// Number of works associated with this entity in this role.
    pub works_count: Option<i64>,
}

/// A location where a work is hosted or published.
///
/// Works can have multiple locations (e.g. publisher site, preprint server,
/// institutional repository). The `primary_location` is the best available
/// version, and `best_oa_location` is the best open-access version.
///
/// ```json
/// {
///   "is_oa": true,
///   "landing_page_url": "https://doi.org/10.7717/peerj.4375",
///   "pdf_url": "https://peerj.com/articles/4375.pdf",
///   "source": {"id": "https://openalex.org/S1983995261", "display_name": "PeerJ", ...},
///   "license": "cc-by",
///   "version": "publishedVersion",
///   "is_accepted": true,
///   "is_published": true
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Location {
    /// Whether this location provides open-access to the full text.
    pub is_oa: Option<bool>,

    /// URL of the landing page for this work at this location.
    pub landing_page_url: Option<String>,

    /// Direct URL to the PDF, if available.
    pub pdf_url: Option<String>,

    /// The source (journal, repository, etc.) hosting this location.
    pub source: Option<DehydratedSource>,

    /// License identifier (e.g. `"cc-by"`, `"cc-by-nc"`).
    pub license: Option<String>,

    /// Machine-readable license URL or identifier.
    pub license_id: Option<String>,

    /// Version of the work at this location: `"publishedVersion"` (final),
    /// `"acceptedVersion"` (post-peer-review), or `"submittedVersion"`
    /// (preprint).
    pub version: Option<String>,

    /// Whether this is an accepted (post-peer-review) version.
    pub is_accepted: Option<bool>,

    /// Whether this is the published (final) version.
    pub is_published: Option<bool>,
}

/// A compact representation of a [`Source`](crate::Source) embedded in other
/// entities (e.g. in [`Location`]).
///
/// Contains core identifying fields but omits detailed metrics and topic data.
///
/// ```json
/// {
///   "id": "https://openalex.org/S1983995261",
///   "display_name": "PeerJ",
///   "issn_l": "2167-8359",
///   "issn": ["2167-8359"],
///   "is_oa": true,
///   "is_in_doaj": true,
///   "host_organization": "https://openalex.org/P4310320104",
///   "host_organization_name": "PeerJ, Inc.",
///   "type": "journal"
/// }
/// ```
///
/// # Note
///
/// The `host_organization_lineage` array may contain `null` elements. This is
/// an API quirk — the field is typed as `Option<Vec<Option<String>>>` to handle
/// this.
#[derive(Debug, Clone, Deserialize)]
pub struct DehydratedSource {
    /// OpenAlex ID (e.g. `"https://openalex.org/S1983995261"`).
    pub id: Option<String>,

    /// Human-readable source name.
    pub display_name: Option<String>,

    /// Linking ISSN (ISSN-L) that groups all ISSNs for this source.
    pub issn_l: Option<String>,

    /// All ISSNs associated with this source.
    pub issn: Option<Vec<String>>,

    /// Whether this source is open-access.
    pub is_oa: Option<bool>,

    /// Whether this source is indexed in the
    /// [DOAJ](https://doaj.org/) (Directory of Open Access Journals).
    pub is_in_doaj: Option<bool>,

    /// Whether this source is indexed in
    /// [CORE](https://core.ac.uk/).
    pub is_core: Option<bool>,

    /// OpenAlex ID of the host organization (publisher or platform).
    pub host_organization: Option<String>,

    /// Display name of the host organization.
    pub host_organization_name: Option<String>,

    /// OpenAlex IDs of the host organization's lineage (parent organizations).
    /// May contain `null` elements — this is a known API quirk.
    pub host_organization_lineage: Option<Vec<Option<String>>>,

    /// Display names of the host organization's lineage. May contain `null`
    /// elements.
    pub host_organization_lineage_names: Option<Vec<Option<String>>>,

    /// Source type: `"journal"`, `"repository"`, `"conference"`,
    /// `"ebook platform"`, `"book series"`, `"metadata"`, or `"other"`.
    pub r#type: Option<String>,
}

/// Open-access status information for a work.
///
/// ```json
/// {"is_oa": true, "oa_status": "gold", "oa_url": "https://doi.org/...", "any_repository_has_fulltext": true}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct OpenAccess {
    /// Whether any location provides open-access to this work.
    pub is_oa: Option<bool>,

    /// Open-access status: `"diamond"` (free, no APC), `"gold"` (OA journal
    /// with APC), `"green"` (repository copy), `"hybrid"` (OA article in
    /// subscription journal), `"bronze"` (free to read but no open license),
    /// or `"closed"` (paywalled).
    pub oa_status: Option<String>,

    /// URL to the best open-access version.
    pub oa_url: Option<String>,

    /// Whether any repository location has the full text.
    pub any_repository_has_fulltext: Option<bool>,
}

/// An author's contribution to a work, including position, institutional
/// affiliations, and countries.
///
/// ```json
/// {
///   "author_position": "first",
///   "author": {"id": "https://openalex.org/A5023888391", "display_name": "Jason Priem", "orcid": "..."},
///   "institutions": [{"id": "https://openalex.org/I4200000001", "display_name": "OurResearch", ...}],
///   "countries": ["US"],
///   "is_corresponding": true,
///   "raw_author_name": "Jason Priem"
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Authorship {
    /// Position in the author list: `"first"`, `"middle"`, or `"last"`.
    pub author_position: Option<String>,

    /// The author entity.
    pub author: Option<DehydratedAuthor>,

    /// Institutions affiliated with the author for this work.
    pub institutions: Option<Vec<DehydratedInstitution>>,

    /// ISO 3166-1 alpha-2 country codes of the author's affiliations.
    pub countries: Option<Vec<String>>,

    /// Whether this author is the corresponding author.
    pub is_corresponding: Option<bool>,

    /// Author name as it appears in the original source.
    pub raw_author_name: Option<String>,

    /// Raw affiliation strings as they appear in the original source.
    pub raw_affiliation_strings: Option<Vec<String>>,

    /// Parsed affiliation data linking raw strings to institution IDs.
    pub affiliations: Option<Vec<AuthorshipAffiliation>>,
}

/// A parsed affiliation within an [`Authorship`], linking a raw affiliation
/// string to resolved institution IDs.
///
/// ```json
/// {"raw_affiliation_string": "Dept of Biology, MIT", "institution_ids": ["https://openalex.org/I63966007"]}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AuthorshipAffiliation {
    /// The affiliation string as it appears in the source.
    pub raw_affiliation_string: Option<String>,

    /// OpenAlex institution IDs resolved from this affiliation string.
    pub institution_ids: Option<Vec<String>>,
}

/// A compact representation of an [`Author`](crate::Author) embedded in other
/// entities (e.g. in [`Authorship`]).
///
/// ```json
/// {"id": "https://openalex.org/A5023888391", "display_name": "Jason Priem", "orcid": "https://orcid.org/0000-0001-6187-6610"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct DehydratedAuthor {
    /// OpenAlex ID (e.g. `"https://openalex.org/A5023888391"`).
    pub id: Option<String>,

    /// Author's display name.
    pub display_name: Option<String>,

    /// ORCID URL (e.g. `"https://orcid.org/0000-0001-6187-6610"`).
    pub orcid: Option<String>,
}

/// A compact representation of an [`Institution`](crate::Institution) embedded
/// in other entities (e.g. in [`Authorship`], [`Affiliation`]).
///
/// ```json
/// {
///   "id": "https://openalex.org/I136199984",
///   "display_name": "Harvard University",
///   "ror": "https://ror.org/03vek6s52",
///   "country_code": "US",
///   "type": "education",
///   "lineage": ["https://openalex.org/I136199984"]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct DehydratedInstitution {
    /// OpenAlex ID (e.g. `"https://openalex.org/I136199984"`).
    pub id: Option<String>,

    /// Institution's display name.
    pub display_name: Option<String>,

    /// ROR identifier URL (e.g. `"https://ror.org/03vek6s52"`).
    pub ror: Option<String>,

    /// ISO 3166-1 alpha-2 country code (e.g. `"US"`).
    pub country_code: Option<String>,

    /// Institution type from the ROR vocabulary: `"Education"`, `"Healthcare"`,
    /// `"Company"`, `"Archive"`, `"Nonprofit"`, `"Government"`, `"Facility"`,
    /// or `"Other"`.
    pub r#type: Option<String>,

    /// OpenAlex IDs of this institution and its parent organizations.
    pub lineage: Option<Vec<String>>,
}

/// A compact representation of a [`Funder`](crate::Funder) embedded in work
/// entities.
///
/// ```json
/// {"id": "https://openalex.org/F4320332161", "display_name": "National Institutes of Health", "ror": "https://ror.org/01cwqze88"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct DehydratedFunder {
    /// OpenAlex ID (e.g. `"https://openalex.org/F4320332161"`).
    pub id: Option<String>,

    /// Funder's display name.
    pub display_name: Option<String>,

    /// ROR identifier URL.
    pub ror: Option<String>,
}

/// A grant or award associated with a work.
///
/// ```json
/// {
///   "id": "https://openalex.org/G...",
///   "display_name": "Research grant",
///   "funder_award_id": "R01-GM12345",
///   "funder_id": "https://openalex.org/F4320332161",
///   "funder_display_name": "National Institutes of Health"
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Award {
    /// OpenAlex ID of the award/grant.
    pub id: Option<String>,

    /// Display name of the award/grant.
    pub display_name: Option<String>,

    /// Funder's internal award identifier (e.g. grant number).
    pub funder_award_id: Option<String>,

    /// OpenAlex ID of the funder.
    pub funder_id: Option<String>,

    /// Display name of the funder.
    pub funder_display_name: Option<String>,

    /// DOI of the award, if available.
    pub doi: Option<String>,
}

/// Bibliographic information for a work (journal volume, issue, page numbers).
///
/// ```json
/// {"volume": "6", "issue": "3", "first_page": "e4375", "last_page": "e4375"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Biblio {
    /// Journal or book volume number.
    pub volume: Option<String>,

    /// Journal issue number.
    pub issue: Option<String>,

    /// First page of the work in the publication.
    pub first_page: Option<String>,

    /// Last page of the work in the publication.
    pub last_page: Option<String>,
}

/// Article processing charge (APC) information for a work.
///
/// Represents either the list price (`apc_list`) or the amount actually paid
/// (`apc_paid`).
///
/// ```json
/// {"value": 1620, "currency": "USD", "value_usd": 1620, "provenance": "doaj"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Apc {
    /// APC amount in the original currency.
    pub value: Option<i64>,

    /// ISO 4217 currency code (e.g. `"USD"`, `"EUR"`, `"GBP"`).
    pub currency: Option<String>,

    /// APC amount converted to US dollars.
    pub value_usd: Option<i64>,

    /// Source of the APC data (e.g. `"doaj"`).
    pub provenance: Option<String>,
}

/// A price entry in a source's APC price list.
///
/// Sources may have multiple APC prices in different currencies.
///
/// ```json
/// {"price": 1620, "currency": "USD"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ApcPrice {
    /// APC amount.
    pub price: Option<i64>,

    /// ISO 4217 currency code.
    pub currency: Option<String>,
}

/// Geographic location data for an institution.
///
/// ```json
/// {
///   "city": "Cambridge",
///   "geonames_city_id": "4931972",
///   "region": "Massachusetts",
///   "country_code": "US",
///   "country": "United States",
///   "latitude": 42.3751,
///   "longitude": -71.1056
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Geo {
    /// City name.
    pub city: Option<String>,

    /// [GeoNames](https://www.geonames.org/) city identifier.
    pub geonames_city_id: Option<String>,

    /// State, province, or region name.
    pub region: Option<String>,

    /// ISO 3166-1 alpha-2 country code (e.g. `"US"`).
    pub country_code: Option<String>,

    /// Full country name.
    pub country: Option<String>,

    /// Geographic latitude in decimal degrees.
    pub latitude: Option<f64>,

    /// Geographic longitude in decimal degrees.
    pub longitude: Option<f64>,
}

/// A research topic associated with an entity, including a relevance score.
///
/// Works are assigned up to 3 topics with relevance scores. Authors, sources,
/// and institutions list their top topics by publication count.
///
/// Each topic is placed in a 3-level hierarchy: domain > field > subfield >
/// topic.
///
/// ```json
/// {
///   "id": "https://openalex.org/T10001",
///   "display_name": "Malaria Control and Epidemiology",
///   "score": 0.9987,
///   "subfield": {"id": "2725", "display_name": "Infectious Diseases"},
///   "field": {"id": "27", "display_name": "Medicine"},
///   "domain": {"id": "4", "display_name": "Health Sciences"}
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct TopicWithScore {
    /// OpenAlex topic ID (e.g. `"https://openalex.org/T10001"`).
    pub id: Option<String>,

    /// Human-readable topic name.
    pub display_name: Option<String>,

    /// Relevance score (0.0–1.0) for how well this topic matches the entity.
    /// Higher is more relevant.
    pub score: Option<f64>,

    /// Number of works in this topic (present on author/source/institution
    /// topic lists).
    pub count: Option<i64>,

    /// The subfield this topic belongs to in the hierarchy.
    pub subfield: Option<TopicHierarchyLevel>,

    /// The field this topic belongs to in the hierarchy.
    pub field: Option<TopicHierarchyLevel>,

    /// The domain this topic belongs to in the hierarchy.
    pub domain: Option<TopicHierarchyLevel>,
}

/// A research topic with a share value representing the fraction of an entity's
/// works in that topic.
///
/// Used in `topic_share` arrays on authors, sources, and institutions.
///
/// ```json
/// {
///   "id": "https://openalex.org/T10001",
///   "display_name": "Malaria Control and Epidemiology",
///   "value": 0.15,
///   "count": 42,
///   "subfield": {"id": "2725", "display_name": "Infectious Diseases"},
///   "field": {"id": "27", "display_name": "Medicine"},
///   "domain": {"id": "4", "display_name": "Health Sciences"}
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct TopicShare {
    /// OpenAlex topic ID.
    pub id: Option<String>,

    /// Human-readable topic name.
    pub display_name: Option<String>,

    /// Fraction of the entity's works in this topic (0.0–1.0).
    pub value: Option<f64>,

    /// Number of works in this topic.
    pub count: Option<i64>,

    /// The subfield this topic belongs to.
    pub subfield: Option<TopicHierarchyLevel>,

    /// The field this topic belongs to.
    pub field: Option<TopicHierarchyLevel>,

    /// The domain this topic belongs to.
    pub domain: Option<TopicHierarchyLevel>,
}

/// A level in the topic hierarchy (domain, field, or subfield).
///
/// ```json
/// {"id": "27", "display_name": "Medicine"}
/// ```
///
/// # Note
///
/// The `id` field is typed as [`serde_json::Value`] because the OpenAlex API
/// returns an integer for topics within the [`Topic`](crate::Topic) entity but
/// a string when topics appear nested in [`Work`](crate::Work) entities.
#[derive(Debug, Clone, Deserialize)]
pub struct TopicHierarchyLevel {
    /// Hierarchy level ID. May be an integer or string depending on context
    /// (see struct-level docs).
    pub id: Option<serde_json::Value>,

    /// Human-readable name for this hierarchy level.
    pub display_name: Option<String>,
}

/// A sibling topic at the same level in the topic hierarchy.
///
/// Returned in the `siblings` array on [`Topic`](crate::Topic) entities.
///
/// ```json
/// {"id": "https://openalex.org/T10002", "display_name": "HIV/AIDS Prevention and Treatment"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct TopicSibling {
    /// OpenAlex topic ID.
    pub id: Option<String>,

    /// Human-readable topic name.
    pub display_name: Option<String>,
}

/// A deprecated concept tag associated with a work or author.
///
/// Concepts are the older tagging system, replaced by [`TopicWithScore`].
/// Still present in API responses for backwards compatibility.
///
/// ```json
/// {
///   "id": "https://openalex.org/C86803240",
///   "wikidata": "https://www.wikidata.org/wiki/Q420",
///   "display_name": "Biology",
///   "level": 0,
///   "score": 0.86
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Concept {
    /// OpenAlex concept ID.
    pub id: Option<String>,

    /// Wikidata URL for this concept.
    pub wikidata: Option<String>,

    /// Human-readable concept name.
    pub display_name: Option<String>,

    /// Hierarchy level (0 = broadest, 5 = most specific).
    pub level: Option<i32>,

    /// Relevance score (0.0–1.0) for how well this concept matches the entity.
    pub score: Option<f64>,
}

/// A keyword extracted from a work, with a relevance score.
///
/// ```json
/// {"id": "https://openalex.org/keywords/open-access", "display_name": "Open Access", "score": 0.95}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Keyword {
    /// OpenAlex keyword ID.
    pub id: Option<String>,

    /// The keyword text.
    pub display_name: Option<String>,

    /// Relevance score (0.0–1.0).
    pub score: Option<f64>,
}

/// A [MeSH](https://www.nlm.nih.gov/mesh/) (Medical Subject Headings) term
/// assigned to a work.
///
/// MeSH terms are assigned by the National Library of Medicine to biomedical
/// literature indexed in PubMed/MEDLINE.
///
/// ```json
/// {
///   "descriptor_ui": "D017712",
///   "descriptor_name": "Peer Review, Research",
///   "qualifier_ui": "",
///   "qualifier_name": "",
///   "is_major_topic": false
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct MeshTerm {
    /// MeSH descriptor unique identifier (e.g. `"D017712"`).
    pub descriptor_ui: Option<String>,

    /// Human-readable descriptor name.
    pub descriptor_name: Option<String>,

    /// MeSH qualifier unique identifier (empty string if no qualifier).
    pub qualifier_ui: Option<String>,

    /// Human-readable qualifier name.
    pub qualifier_name: Option<String>,

    /// Whether this is a major topic of the work.
    pub is_major_topic: Option<bool>,
}

/// A [UN Sustainable Development Goal](https://sdgs.un.org/goals) (SDG) tag
/// assigned to a work, with a relevance score.
///
/// ```json
/// {"id": "https://metadata.un.org/sdg/3", "display_name": "Good health and well-being", "score": 0.85}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct SdgTag {
    /// SDG identifier URL.
    pub id: Option<String>,

    /// SDG name (e.g. `"Good health and well-being"`).
    pub display_name: Option<String>,

    /// Relevance score (0.0–1.0).
    pub score: Option<f64>,
}

/// Citation percentile ranking for a work relative to works published in the
/// same year.
///
/// ```json
/// {"value": 99.5, "is_in_top_1_percent": true, "is_in_top_10_percent": true}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CitationPercentile {
    /// Percentile value (0–100). A value of 95 means this work is cited more
    /// than 95% of works published in the same year.
    pub value: Option<f64>,

    /// Whether this work is in the top 1% of cited works for its publication
    /// year.
    pub is_in_top_1_percent: Option<bool>,

    /// Whether this work is in the top 10% of cited works for its publication
    /// year.
    pub is_in_top_10_percent: Option<bool>,
}

/// Min/max citation count percentile for a work's publication year.
///
/// ```json
/// {"min": 143, "max": 143}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct CitedByPercentileYear {
    /// Minimum citation count at the work's percentile for its year.
    pub min: Option<i32>,

    /// Maximum citation count at the work's percentile for its year.
    pub max: Option<i32>,
}

/// Availability of full-text content for a work.
///
/// ```json
/// {"pdf": true, "grobid_xml": true}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct HasContent {
    /// Whether a PDF version is available.
    pub pdf: Option<bool>,

    /// Whether a GROBID-parsed XML version is available.
    pub grobid_xml: Option<bool>,
}

/// An author's affiliation with an institution over a range of years.
///
/// Used in [`Author::affiliations`](crate::Author::affiliations) to show
/// institutional history.
///
/// ```json
/// {
///   "institution": {"id": "https://openalex.org/I4200000001", "display_name": "OurResearch", ...},
///   "years": [2022, 2021, 2020]
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct Affiliation {
    /// The affiliated institution.
    pub institution: Option<DehydratedInstitution>,

    /// Calendar years during which the author was affiliated with this
    /// institution.
    pub years: Option<Vec<i32>>,
}

/// An institution related to another institution (parent, child, or affiliated).
///
/// Returned in
/// [`Institution::associated_institutions`](crate::Institution::associated_institutions).
///
/// ```json
/// {
///   "id": "https://openalex.org/I205783295",
///   "ror": "https://ror.org/...",
///   "display_name": "Harvard Medical School",
///   "country_code": "US",
///   "type": "education",
///   "relationship": "child"
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AssociatedInstitution {
    /// OpenAlex ID.
    pub id: Option<String>,

    /// ROR identifier URL.
    pub ror: Option<String>,

    /// Institution's display name.
    pub display_name: Option<String>,

    /// ISO 3166-1 alpha-2 country code.
    pub country_code: Option<String>,

    /// Institution type from the ROR vocabulary.
    pub r#type: Option<String>,

    /// Relationship to the parent institution: `"parent"` (governing
    /// organization), `"child"` (subsidiary), or `"related"` (affiliated).
    pub relationship: Option<String>,
}

/// An author's name parsed into components.
///
/// ```json
/// {"first": "Jason", "middle": null, "last": "Priem", "suffix": null, "nickname": null}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct ParsedName {
    /// First/given name.
    pub first: Option<String>,

    /// Middle name or initial.
    pub middle: Option<String>,

    /// Last/family name.
    pub last: Option<String>,

    /// Name suffix (e.g. `"Jr."`, `"III"`).
    pub suffix: Option<String>,

    /// Nickname or preferred name.
    pub nickname: Option<String>,
}

// ── Per-entity ID types ────────────────────────────────────────────────

/// External identifiers for a [`Work`](crate::Work).
///
/// ```json
/// {
///   "openalex": "https://openalex.org/W2741809807",
///   "doi": "https://doi.org/10.7717/peerj.4375",
///   "mag": "2741809807",
///   "pmid": "https://pubmed.ncbi.nlm.nih.gov/29456894",
///   "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/5815332"
/// }
/// ```
///
/// # Note
///
/// The `mag` field is a string, not an integer, despite containing numeric
/// values. This is a known API quirk.
#[derive(Debug, Clone, Deserialize)]
pub struct WorkIds {
    /// OpenAlex ID URL.
    pub openalex: Option<String>,

    /// DOI URL (e.g. `"https://doi.org/10.7717/peerj.4375"`).
    pub doi: Option<String>,

    /// Microsoft Academic Graph identifier (string, not integer).
    pub mag: Option<String>,

    /// PubMed identifier URL.
    pub pmid: Option<String>,

    /// PubMed Central identifier URL.
    pub pmcid: Option<String>,
}

/// External identifiers for an [`Author`](crate::Author).
///
/// ```json
/// {"openalex": "https://openalex.org/A5023888391", "orcid": "https://orcid.org/0000-0001-6187-6610", "scopus": "56462225600"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct AuthorIds {
    /// OpenAlex ID URL.
    pub openalex: Option<String>,

    /// ORCID URL.
    pub orcid: Option<String>,

    /// Scopus author identifier.
    pub scopus: Option<String>,
}

/// External identifiers for a [`Source`](crate::Source).
///
/// ```json
/// {
///   "openalex": "https://openalex.org/S137773608",
///   "issn_l": "0028-0836",
///   "issn": ["1476-4687", "0028-0836"],
///   "mag": "137773608",
///   "wikidata": "https://www.wikidata.org/wiki/Q180445"
/// }
/// ```
///
/// # Note
///
/// The `mag` field is a string, not an integer.
#[derive(Debug, Clone, Deserialize)]
pub struct SourceIds {
    /// OpenAlex ID URL.
    pub openalex: Option<String>,

    /// Linking ISSN (ISSN-L).
    pub issn_l: Option<String>,

    /// All ISSNs associated with this source.
    pub issn: Option<Vec<String>>,

    /// Microsoft Academic Graph identifier (string, not integer).
    pub mag: Option<String>,

    /// Wikidata URL.
    pub wikidata: Option<String>,
}

/// External identifiers for an [`Institution`](crate::Institution).
///
/// ```json
/// {
///   "openalex": "https://openalex.org/I136199984",
///   "ror": "https://ror.org/03vek6s52",
///   "grid": "grid.38142.3c",
///   "mag": "136199984",
///   "wikipedia": "https://en.wikipedia.org/wiki/Harvard%20University",
///   "wikidata": "https://www.wikidata.org/wiki/Q13371"
/// }
/// ```
///
/// # Note
///
/// The `mag` field is a string, not an integer.
#[derive(Debug, Clone, Deserialize)]
pub struct InstitutionIds {
    /// OpenAlex ID URL.
    pub openalex: Option<String>,

    /// ROR identifier URL.
    pub ror: Option<String>,

    /// GRID identifier (deprecated, replaced by ROR).
    pub grid: Option<String>,

    /// Microsoft Academic Graph identifier (string, not integer).
    pub mag: Option<String>,

    /// Wikipedia article URL.
    pub wikipedia: Option<String>,

    /// Wikidata URL.
    pub wikidata: Option<String>,
}

/// External identifiers for a [`Topic`](crate::Topic).
///
/// ```json
/// {"openalex": "https://openalex.org/T10001", "wikipedia": "https://en.wikipedia.org/wiki/Malaria"}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct TopicIds {
    /// OpenAlex ID URL.
    pub openalex: Option<String>,

    /// Wikipedia article URL.
    pub wikipedia: Option<String>,
}

/// External identifiers for a [`Publisher`](crate::Publisher).
///
/// ```json
/// {"openalex": "https://openalex.org/P4310319965", "ror": "https://ror.org/...", "wikidata": "https://www.wikidata.org/wiki/Q..."}
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct PublisherIds {
    /// OpenAlex ID URL.
    pub openalex: Option<String>,

    /// ROR identifier URL.
    pub ror: Option<String>,

    /// Wikidata URL.
    pub wikidata: Option<String>,
}

/// External identifiers for a [`Funder`](crate::Funder).
///
/// ```json
/// {
///   "openalex": "https://openalex.org/F4320332161",
///   "ror": "https://ror.org/01cwqze88",
///   "wikidata": "https://www.wikidata.org/wiki/Q390551",
///   "crossref": "100000002",
///   "doi": "https://doi.org/10.13039/100000002"
/// }
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct FunderIds {
    /// OpenAlex ID URL.
    pub openalex: Option<String>,

    /// ROR identifier URL.
    pub ror: Option<String>,

    /// Wikidata URL.
    pub wikidata: Option<String>,

    /// Crossref funder registry identifier.
    pub crossref: Option<String>,

    /// DOI URL for this funder.
    pub doi: Option<String>,
}
