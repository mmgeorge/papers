use serde::Deserialize;

use super::common::*;

/// A research funding organization (e.g. NIH, NSF, ERC).
///
/// Linked to the [Crossref funder registry](https://www.crossref.org/services/funder-registry/).
/// Includes grant counts, funded works, country of origin, and research impact
/// metrics.
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/F4320332161",
///   "display_name": "National Institutes of Health",
///   "country_code": "US",
///   "description": "Agency of the United States...",
///   "awards_count": 500000,
///   "works_count": 3253779,
///   "cited_by_count": 150000000,
///   "summary_stats": {"2yr_mean_citedness": 8.2, "h_index": 2000, ...}
/// }
/// ```
///
/// # ID formats
///
/// Funders can only be retrieved by OpenAlex ID (`F...`).
#[derive(Debug, Clone, Deserialize)]
pub struct Funder {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/F4320332161"`).
    pub id: String,

    /// Human-readable funder name (e.g. `"National Institutes of Health"`).
    pub display_name: Option<String>,

    /// Alternative names or abbreviations (e.g. `["NIH"]`).
    pub alternate_titles: Option<Vec<String>>,

    /// ISO 3166-1 alpha-2 country code of the funder's country (e.g. `"US"`).
    pub country_code: Option<String>,

    /// Wikidata description of the funder.
    pub description: Option<String>,

    /// URL of the funder's homepage.
    pub homepage_url: Option<String>,

    /// URL of the funder's logo or image.
    pub image_url: Option<String>,

    /// URL of a thumbnail version of the funder's image.
    pub image_thumbnail_url: Option<String>,

    /// Total number of grants/awards issued by this funder.
    pub awards_count: Option<i64>,

    /// Total number of works funded by this funder.
    pub works_count: Option<i64>,

    /// Total number of citations received by works funded by this funder.
    pub cited_by_count: Option<i64>,

    /// Impact metrics: h-index, i10-index, and 2-year mean citedness.
    pub summary_stats: Option<SummaryStats>,

    /// External identifiers (OpenAlex, ROR, Wikidata, Crossref, DOI).
    pub ids: Option<FunderIds>,

    /// Funding and citation counts broken down by year.
    pub counts_by_year: Option<Vec<CountsByYear>>,

    /// Roles this organization plays in the research ecosystem (institution,
    /// funder, publisher).
    pub roles: Option<Vec<Role>>,

    /// ISO 8601 timestamp of the last update to this record.
    pub updated_date: Option<String>,

    /// ISO 8601 date when this record was first created.
    pub created_date: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_funder() {
        let json = include_str!("../../tests/fixtures/funder.json");
        let funder: Funder = serde_json::from_str(json).expect("Failed to deserialize Funder");
        assert_eq!(funder.id, "https://openalex.org/F4320332161");
        assert_eq!(
            funder.display_name.as_deref(),
            Some("National Institutes of Health")
        );
        assert_eq!(funder.country_code.as_deref(), Some("US"));
        assert!(funder.awards_count.is_some());
    }
}
