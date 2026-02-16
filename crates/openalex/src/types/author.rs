use serde::Deserialize;

use super::common::*;

/// A disambiguated author profile in OpenAlex.
///
/// OpenAlex contains 110M+ author profiles. Each author has a unified identity
/// across name variants, with linked ORCID, institutional affiliations over
/// time, publication history, citation metrics, and topic expertise.
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/A5023888391",
///   "orcid": "https://orcid.org/0000-0001-6187-6610",
///   "display_name": "Jason Priem",
///   "works_count": 49,
///   "cited_by_count": 4215,
///   "summary_stats": {"2yr_mean_citedness": 4.065, "h_index": 20, "i10_index": 30},
///   "affiliations": [{"institution": {...}, "years": [2022, 2021, 2020]}]
/// }
/// ```
///
/// # ID formats
///
/// Authors can be retrieved by OpenAlex ID (`A...`) or ORCID.
#[derive(Debug, Clone, Deserialize)]
pub struct Author {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/A5023888391"`).
    pub id: String,

    /// ORCID URL (e.g. `"https://orcid.org/0000-0001-6187-6610"`).
    pub orcid: Option<String>,

    /// Primary display name for this author.
    pub display_name: Option<String>,

    /// Alternative name variants (e.g. maiden names, transliterations).
    pub display_name_alternatives: Option<Vec<String>>,

    /// Total number of works by this author.
    pub works_count: Option<i64>,

    /// Total number of citations received by this author's works.
    pub cited_by_count: Option<i64>,

    /// Impact metrics: h-index, i10-index, and 2-year mean citedness.
    pub summary_stats: Option<SummaryStats>,

    /// External identifiers (OpenAlex, ORCID, Scopus).
    pub ids: Option<AuthorIds>,

    /// Institutional affiliations over time. Each entry links an institution to
    /// the years the author was affiliated.
    pub affiliations: Option<Vec<Affiliation>>,

    /// Most recently known institutional affiliation(s).
    pub last_known_institutions: Option<Vec<DehydratedInstitution>>,

    /// Top research topics for this author, ranked by relevance or work count.
    pub topics: Option<Vec<TopicWithScore>>,

    /// Research topics as a share of this author's total works.
    pub topic_share: Option<Vec<TopicShare>>,

    /// Deprecated concept tags (replaced by topics). Prefixed with `x_` to
    /// indicate deprecation.
    pub x_concepts: Option<Vec<Concept>>,

    /// Publication and citation counts broken down by year.
    pub counts_by_year: Option<Vec<CountsByYear>>,

    /// API URL to retrieve this author's works (e.g.
    /// `"https://api.openalex.org/works?filter=author.id:A5023888391"`).
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
    fn test_deserialize_author() {
        let json = include_str!("../../tests/fixtures/author.json");
        let author: Author = serde_json::from_str(json).expect("Failed to deserialize Author");
        assert_eq!(author.id, "https://openalex.org/A5023888391");
        assert_eq!(author.display_name.as_deref(), Some("Jason Priem"));
        assert_eq!(
            author.orcid.as_deref(),
            Some("https://orcid.org/0000-0001-6187-6610")
        );
        assert!(author.works_count.unwrap() > 0);

        let stats = author.summary_stats.as_ref().unwrap();
        assert!(stats.h_index.is_some());
    }
}
