use serde::Deserialize;

use super::common::*;

/// A research institution: university, hospital, company, government agency, or
/// other research organization.
///
/// Linked to [ROR](https://ror.org/) identifiers. Includes geographic location,
/// parent/child relationships, affiliated repositories, and research output
/// metrics.
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/I136199984",
///   "ror": "https://ror.org/03vek6s52",
///   "display_name": "Harvard University",
///   "country_code": "US",
///   "type": "education",
///   "geo": {"city": "Cambridge", "country": "United States", "latitude": 42.3751, ...},
///   "works_count": 800000,
///   "cited_by_count": 40000000
/// }
/// ```
///
/// # ID formats
///
/// Institutions can be retrieved by OpenAlex ID (`I...`) or ROR.
#[derive(Debug, Clone, Deserialize)]
pub struct Institution {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/I136199984"`).
    pub id: String,

    /// ROR identifier URL (e.g. `"https://ror.org/03vek6s52"`).
    pub ror: Option<String>,

    /// Human-readable institution name (e.g. `"Harvard University"`).
    pub display_name: Option<String>,

    /// ISO 3166-1 alpha-2 country code (e.g. `"US"`).
    pub country_code: Option<String>,

    /// Institution type from the ROR vocabulary: `"Education"`, `"Healthcare"`,
    /// `"Company"`, `"Archive"`, `"Nonprofit"`, `"Government"`, `"Facility"`,
    /// or `"Other"`.
    pub r#type: Option<String>,

    /// Machine-readable ROR type identifier.
    pub type_id: Option<String>,

    /// OpenAlex IDs of this institution and its parent organizations in the
    /// hierarchy.
    pub lineage: Option<Vec<String>>,

    /// URL of the institution's homepage.
    pub homepage_url: Option<String>,

    /// URL of the institution's logo or image.
    pub image_url: Option<String>,

    /// URL of a thumbnail version of the institution's image.
    pub image_thumbnail_url: Option<String>,

    /// Acronyms for the institution name (e.g. `["MIT"]`).
    pub display_name_acronyms: Option<Vec<String>>,

    /// Alternative names for the institution (translations,
    /// historical names).
    pub display_name_alternatives: Option<Vec<String>>,

    /// Internationalized display names (structure varies by locale).
    pub international: Option<serde_json::Value>,

    /// Repositories hosted by this institution (structure varies).
    pub repositories: Option<Vec<serde_json::Value>>,

    /// Total number of works affiliated with this institution.
    pub works_count: Option<i64>,

    /// Total number of citations received by works from this institution.
    pub cited_by_count: Option<i64>,

    /// Impact metrics: h-index, i10-index, and 2-year mean citedness.
    pub summary_stats: Option<SummaryStats>,

    /// External identifiers (OpenAlex, ROR, GRID, MAG, Wikipedia, Wikidata).
    pub ids: Option<InstitutionIds>,

    /// Geographic location: city, region, country, and coordinates.
    pub geo: Option<Geo>,

    /// Related institutions (parent, child, and affiliated organizations).
    pub associated_institutions: Option<Vec<AssociatedInstitution>>,

    /// Publication and citation counts broken down by year.
    pub counts_by_year: Option<Vec<CountsByYear>>,

    /// Roles this organization plays in the research ecosystem (institution,
    /// funder, publisher).
    pub roles: Option<Vec<Role>>,

    /// Top research topics for this institution, ranked by relevance or work
    /// count.
    pub topics: Option<Vec<TopicWithScore>>,

    /// Research topics as a share of this institution's total works.
    pub topic_share: Option<Vec<TopicShare>>,

    /// Whether this institution is a "super system" (top-level umbrella
    /// organization).
    pub is_super_system: Option<bool>,

    /// API URL to retrieve this institution's works.
    pub works_api_url: Option<String>,

    /// ISO 8601 timestamp of the last update to this record.
    pub updated_date: Option<String>,

    /// ISO 8601 date when this record was first created.
    pub created_date: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_institution() {
        let json = include_str!("../../tests/fixtures/institution.json");
        let inst: Institution =
            serde_json::from_str(json).expect("Failed to deserialize Institution");
        assert_eq!(inst.id, "https://openalex.org/I136199984");
        assert_eq!(inst.display_name.as_deref(), Some("Harvard University"));
        assert!(inst.ror.is_some());
        assert!(inst.geo.is_some());
        assert!(inst.r#type.is_some());
    }
}
