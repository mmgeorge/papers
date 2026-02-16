use serde::Deserialize;

use super::common::*;

/// A publishing organization (e.g. Elsevier, Springer Nature, Wiley).
///
/// Includes parent/child hierarchy, country of origin, linked sources, and
/// citation metrics. Some publishers also act as funders or institutions (see
/// [`roles`](Publisher::roles)).
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/P4310319965",
///   "display_name": "Springer Nature",
///   "hierarchy_level": 0,
///   "country_codes": ["DE"],
///   "works_count": 2750825,
///   "cited_by_count": 75000000,
///   "summary_stats": {"2yr_mean_citedness": 5.1, "h_index": 1500, ...}
/// }
/// ```
///
/// # ID formats
///
/// Publishers can only be retrieved by OpenAlex ID (`P...`).
#[derive(Debug, Clone, Deserialize)]
pub struct Publisher {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/P4310319965"`).
    pub id: String,

    /// Human-readable publisher name (e.g. `"Springer Nature"`).
    pub display_name: Option<String>,

    /// Alternative names or name variants.
    pub alternate_titles: Option<Vec<String>>,

    /// Level in the publisher hierarchy. `0` = top-level publisher, `1` =
    /// subsidiary, etc.
    pub hierarchy_level: Option<i32>,

    /// Parent publisher (if this is a subsidiary). Structure varies — may be a
    /// string ID or an object.
    pub parent_publisher: Option<serde_json::Value>,

    /// Publisher lineage from this publisher up to the top-level parent.
    /// Elements may be string IDs or objects.
    pub lineage: Option<Vec<serde_json::Value>>,

    /// ISO 3166-1 alpha-2 country codes of the publisher's country/countries of
    /// origin.
    pub country_codes: Option<Vec<String>>,

    /// URL of the publisher's homepage.
    pub homepage_url: Option<String>,

    /// URL of the publisher's logo or image.
    pub image_url: Option<String>,

    /// URL of a thumbnail version of the publisher's image.
    pub image_thumbnail_url: Option<String>,

    /// Total number of works published by this publisher.
    pub works_count: Option<i64>,

    /// Total number of citations received by works from this publisher.
    pub cited_by_count: Option<i64>,

    /// Impact metrics: h-index, i10-index, and 2-year mean citedness.
    pub summary_stats: Option<SummaryStats>,

    /// External identifiers (OpenAlex, ROR, Wikidata).
    pub ids: Option<PublisherIds>,

    /// Publication and citation counts broken down by year.
    pub counts_by_year: Option<Vec<CountsByYear>>,

    /// Roles this organization plays in the research ecosystem (institution,
    /// funder, publisher).
    pub roles: Option<Vec<Role>>,

    /// API URL to retrieve sources published by this publisher.
    pub sources_api_url: Option<String>,

    /// ISO 8601 timestamp of the last update to this record.
    pub updated_date: Option<String>,

    /// ISO 8601 date when this record was first created.
    pub created_date: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_publisher() {
        let json = include_str!("../../tests/fixtures/publisher.json");
        let publisher: Publisher =
            serde_json::from_str(json).expect("Failed to deserialize Publisher");
        assert_eq!(publisher.id, "https://openalex.org/P4310319965");
        assert_eq!(
            publisher.display_name.as_deref(),
            Some("Springer Nature")
        );
        assert!(publisher.hierarchy_level.is_some());
        assert!(publisher.country_codes.is_some());
    }
}
