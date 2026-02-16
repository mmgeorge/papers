use serde::Deserialize;
use std::collections::HashMap;

use super::common::*;

/// A scholarly work: an article, book, dataset, preprint, or other research
/// output.
///
/// OpenAlex contains 240M+ works. Each work includes metadata about its title,
/// authors, publication venue, open-access status, citations, topics, funding,
/// and more.
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/W2741809807",
///   "doi": "https://doi.org/10.7717/peerj.4375",
///   "display_name": "The state of OA: a large-scale analysis...",
///   "publication_year": 2018,
///   "type": "article",
///   "cited_by_count": 1234,
///   "open_access": {"is_oa": true, "oa_status": "gold", ...},
///   "authorships": [{"author_position": "first", "author": {...}, ...}],
///   "primary_topic": {"id": "https://openalex.org/T...", "score": 0.99, ...}
/// }
/// ```
///
/// # ID formats
///
/// Works can be retrieved by OpenAlex ID (`W...`), DOI, PMID (`pmid:...`),
/// PMCID (`pmcid:...`), or MAG ID (`mag:...`).
///
/// # Abstract
///
/// The abstract is stored as an inverted index in
/// [`abstract_inverted_index`](Work::abstract_inverted_index): a
/// `HashMap<String, Vec<u32>>` mapping each word to its position(s) in the
/// abstract text. This format is more compact than storing the full text.
#[derive(Debug, Clone, Deserialize)]
pub struct Work {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/W2741809807"`).
    pub id: String,

    /// DOI URL (e.g. `"https://doi.org/10.7717/peerj.4375"`).
    pub doi: Option<String>,

    /// Original title of the work.
    pub title: Option<String>,

    /// Display-ready title (same as `title` in most cases).
    pub display_name: Option<String>,

    /// Year of publication.
    pub publication_year: Option<i32>,

    /// Full publication date as an ISO 8601 string (e.g. `"2018-02-13"`).
    pub publication_date: Option<String>,

    /// ISO 639-1 two-letter language code (e.g. `"en"`, `"zh"`).
    pub language: Option<String>,

    /// Simplified work type: `"article"`, `"preprint"`, `"paratext"`,
    /// `"letter"`, `"editorial"`, `"erratum"`, `"libguides"`,
    /// `"supplementary-materials"`, or `"review"`.
    pub r#type: Option<String>,

    /// Crossref type (more granular than `type`): `"journal-article"`,
    /// `"proceedings-article"`, `"posted-content"`, `"book-chapter"`,
    /// `"dissertation"`, `"dataset"`, and 24 other values from the Crossref
    /// controlled vocabulary.
    pub type_crossref: Option<String>,

    /// External indexes that include this work: `"arxiv"`, `"crossref"`,
    /// `"doaj"`, `"pubmed"`.
    pub indexed_in: Option<Vec<String>>,

    /// External identifiers for this work (DOI, MAG, PMID, PMCID).
    pub ids: Option<WorkIds>,

    /// The primary (best) location for this work — typically the publisher's
    /// site.
    pub primary_location: Option<Location>,

    /// All known locations where this work is available.
    pub locations: Option<Vec<Location>>,

    /// Total number of locations.
    pub locations_count: Option<i64>,

    /// The best open-access location, if any.
    pub best_oa_location: Option<Location>,

    /// Open-access status: whether the work is OA, OA type, and OA URL.
    pub open_access: Option<OpenAccess>,

    /// List of authors and their affiliations for this work.
    pub authorships: Option<Vec<Authorship>>,

    /// Number of distinct countries represented by the authors' affiliations.
    pub countries_distinct_count: Option<i64>,

    /// Number of distinct institutions represented by the authors'
    /// affiliations.
    pub institutions_distinct_count: Option<i64>,

    /// OpenAlex IDs of the corresponding author(s).
    pub corresponding_author_ids: Option<Vec<String>>,

    /// OpenAlex IDs of the corresponding author(s)' institution(s).
    pub corresponding_institution_ids: Option<Vec<String>>,

    /// Article processing charge list price.
    pub apc_list: Option<Apc>,

    /// Article processing charge actually paid.
    pub apc_paid: Option<Apc>,

    /// Field-weighted citation impact. A value of 1.0 means average for the
    /// field; above 1.0 means more cited than average.
    pub fwci: Option<f64>,

    /// Whether full text is available for this work.
    pub has_fulltext: Option<bool>,

    /// Total number of times this work has been cited.
    pub cited_by_count: Option<i64>,

    /// Citation percentile ranking relative to works published in the same
    /// year.
    pub citation_normalized_percentile: Option<CitationPercentile>,

    /// Min/max citation count at this work's percentile for its publication
    /// year.
    pub cited_by_percentile_year: Option<CitedByPercentileYear>,

    /// Bibliographic details: volume, issue, and page numbers.
    pub biblio: Option<Biblio>,

    /// Whether this work has been retracted.
    pub is_retracted: Option<bool>,

    /// Whether this work is paratext (content about a venue, e.g. table of
    /// contents, cover page).
    pub is_paratext: Option<bool>,

    /// Whether this work is cross-publisher article content.
    #[serde(default)]
    pub is_xpac: Option<bool>,

    /// The most relevant topic assigned to this work, with a relevance score.
    pub primary_topic: Option<TopicWithScore>,

    /// Up to 3 topics assigned to this work, each with a relevance score.
    pub topics: Option<Vec<TopicWithScore>>,

    /// Keywords extracted from this work.
    pub keywords: Option<Vec<Keyword>>,

    /// Deprecated concept tags (replaced by topics).
    pub concepts: Option<Vec<Concept>>,

    /// MeSH terms assigned to this work (biomedical literature only).
    pub mesh: Option<Vec<MeshTerm>>,

    /// UN Sustainable Development Goal tags.
    pub sustainable_development_goals: Option<Vec<SdgTag>>,

    /// Funding organizations that supported this work.
    pub funders: Option<Vec<DehydratedFunder>>,

    /// Grants and awards associated with this work.
    pub awards: Option<Vec<Award>>,

    /// Availability of full-text content (PDF, GROBID XML).
    pub has_content: Option<HasContent>,

    /// URLs for accessing full-text content. Structure varies.
    pub content_urls: Option<serde_json::Value>,

    /// Number of works cited by this work.
    pub referenced_works_count: Option<i64>,

    /// OpenAlex IDs of works cited in this work's references.
    pub referenced_works: Option<Vec<String>>,

    /// OpenAlex IDs of works related to this one (by topic similarity).
    pub related_works: Option<Vec<String>>,

    /// Abstract stored as an inverted index: a map from each word to its
    /// position(s) in the abstract text. Reconstruct the abstract by sorting
    /// words by their positions.
    pub abstract_inverted_index: Option<HashMap<String, Vec<u32>>>,

    /// Citation and publication counts broken down by year.
    pub counts_by_year: Option<Vec<CountsByYear>>,

    /// ISO 8601 timestamp of the last update to this record.
    pub updated_date: Option<String>,

    /// ISO 8601 date when this record was first created.
    pub created_date: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_work() {
        let json = include_str!("../../tests/fixtures/work.json");
        let work: Work = serde_json::from_str(json).expect("Failed to deserialize Work");
        assert_eq!(work.id, "https://openalex.org/W2741809807");
        assert_eq!(
            work.display_name.as_deref(),
            Some("The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles")
        );
        assert_eq!(work.publication_year, Some(2018));
        assert!(work.r#type.is_some());

        let authorships = work.authorships.as_ref().unwrap();
        assert!(!authorships.is_empty());

        let oa = work.open_access.as_ref().unwrap();
        assert_eq!(oa.is_oa, Some(true));

        if let Some(topics) = &work.topics {
            assert!(!topics.is_empty());
        }
    }
}
