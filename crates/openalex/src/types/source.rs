use serde::Deserialize;

use super::common::*;

/// A publishing venue: journal, repository, conference, ebook platform, or book
/// series.
///
/// Sources include ISSN identifiers, open-access status, APC pricing, host
/// organization, and impact metrics.
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/S137773608",
///   "display_name": "Nature",
///   "issn_l": "0028-0836",
///   "type": "journal",
///   "is_oa": false,
///   "works_count": 450000,
///   "cited_by_count": 25000000,
///   "summary_stats": {"2yr_mean_citedness": 50.2, "h_index": 1200, ...}
/// }
/// ```
///
/// # ID formats
///
/// Sources can be retrieved by OpenAlex ID (`S...`) or ISSN.
///
/// # Note
///
/// The `host_organization_lineage` array may contain `null` elements. This is a
/// known API quirk — the field is typed as `Option<Vec<Option<String>>>`.
#[derive(Debug, Clone, Deserialize)]
pub struct Source {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/S137773608"`).
    pub id: String,

    /// Linking ISSN (ISSN-L) that groups all ISSNs for this source.
    pub issn_l: Option<String>,

    /// All ISSNs associated with this source.
    pub issn: Option<Vec<String>>,

    /// Human-readable source name (e.g. `"Nature"`).
    pub display_name: Option<String>,

    /// Abbreviated title (e.g. `"Nat."`).
    pub abbreviated_title: Option<String>,

    /// Alternative titles or name variants.
    pub alternate_titles: Option<Vec<String>>,

    /// OpenAlex ID of the host organization (publisher or platform).
    pub host_organization: Option<String>,

    /// Display name of the host organization.
    pub host_organization_name: Option<String>,

    /// OpenAlex IDs of the host organization's lineage (parent organizations).
    /// May contain `null` elements — this is a known API quirk.
    pub host_organization_lineage: Option<Vec<Option<String>>>,

    /// Total number of works published in this source.
    pub works_count: Option<i64>,

    /// Total number of citations received by works in this source.
    pub cited_by_count: Option<i64>,

    /// Impact metrics: h-index, i10-index, and 2-year mean citedness.
    pub summary_stats: Option<SummaryStats>,

    /// Whether this source is open-access.
    pub is_oa: Option<bool>,

    /// Whether this source is indexed in the
    /// [DOAJ](https://doaj.org/) (Directory of Open Access Journals).
    pub is_in_doaj: Option<bool>,

    /// Whether this source is indexed in [CORE](https://core.ac.uk/).
    pub is_core: Option<bool>,

    /// Whether this source has a high proportion of open-access works.
    pub is_high_oa_rate: Option<bool>,

    /// Whether this source is indexed in [SciELO](https://scielo.org/).
    pub is_in_scielo: Option<bool>,

    /// Whether this source uses [Open Journal Systems](https://pkp.sfu.ca/software/ojs/).
    pub is_ojs: Option<bool>,

    /// Year the source flipped to open-access, if applicable.
    pub oa_flip_year: Option<i32>,

    /// Year of the earliest publication in this source.
    pub first_publication_year: Option<i32>,

    /// Year of the most recent publication in this source.
    pub last_publication_year: Option<i32>,

    /// External identifiers (OpenAlex, ISSN-L, ISSNs, MAG, Wikidata).
    pub ids: Option<SourceIds>,

    /// URL of the source's homepage.
    pub homepage_url: Option<String>,

    /// Article processing charge prices in various currencies.
    pub apc_prices: Option<Vec<ApcPrice>>,

    /// APC amount in US dollars.
    pub apc_usd: Option<i64>,

    /// ISO 3166-1 alpha-2 country code of the source's country of origin.
    pub country_code: Option<String>,

    /// Learned societies associated with this source (structure varies).
    pub societies: Option<Vec<serde_json::Value>>,

    /// Source type: `"journal"`, `"repository"`, `"conference"`,
    /// `"ebook platform"`, `"book series"`, `"metadata"`, or `"other"`.
    pub r#type: Option<String>,

    /// Top research topics for this source, ranked by relevance or work count.
    pub topics: Option<Vec<TopicWithScore>>,

    /// Research topics as a share of this source's total works.
    pub topic_share: Option<Vec<TopicShare>>,

    /// Publication and citation counts broken down by year.
    pub counts_by_year: Option<Vec<CountsByYear>>,

    /// API URL to retrieve this source's works.
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
    fn test_deserialize_source() {
        let json = include_str!("../../tests/fixtures/source.json");
        let source: Source = serde_json::from_str(json).expect("Failed to deserialize Source");
        assert_eq!(source.id, "https://openalex.org/S137773608");
        assert_eq!(source.display_name.as_deref(), Some("Nature"));
        assert!(source.issn_l.is_some());
        assert!(source.is_oa.is_some());
        assert!(source.r#type.is_some());
    }
}
