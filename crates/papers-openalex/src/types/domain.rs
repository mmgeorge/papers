use serde::{Deserialize, Serialize};

use super::common::*;

/// A top-level research domain in OpenAlex's topic hierarchy.
///
/// Domains are the broadest level: **domain > field > subfield > topic**.
/// There are 4 domains: Life Sciences, Social Sciences, Physical Sciences,
/// and Health Sciences.
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/domains/3",
///   "display_name": "Physical Sciences",
///   "description": "branch of natural science that studies non-living systems",
///   "fields": [{"id": "https://openalex.org/fields/17", "display_name": "Computer Science"}, ...],
///   "works_count": 134263529
/// }
/// ```
///
/// # ID formats
///
/// Domains use numeric IDs: `1` (Life Sciences), `2` (Social Sciences),
/// `3` (Physical Sciences), `4` (Health Sciences).
///
/// # Note
///
/// Domains do **not** support autocomplete (`/autocomplete/domains` returns 404).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/domains/3"`).
    pub id: String,

    /// Human-readable domain name (e.g. `"Physical Sciences"`).
    pub display_name: Option<String>,

    /// Brief description of the domain's scope.
    pub description: Option<String>,

    /// External identifiers (OpenAlex, Wikidata, Wikipedia).
    pub ids: Option<HierarchyIds>,

    /// Alternative names or name variants.
    pub display_name_alternatives: Option<Vec<String>>,

    /// Academic fields within this domain.
    pub fields: Option<Vec<HierarchyEntity>>,

    /// Other domains at the same level in the hierarchy.
    pub siblings: Option<Vec<HierarchyEntity>>,

    /// Total number of works in this domain.
    pub works_count: Option<i64>,

    /// Total number of citations received by works in this domain.
    pub cited_by_count: Option<i64>,

    /// API URL to retrieve works in this domain.
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
    fn test_deserialize_domain() {
        let json = include_str!("../../tests/fixtures/domain.json");
        let domain: Domain = serde_json::from_str(json).expect("Failed to deserialize Domain");
        assert_eq!(domain.id, "https://openalex.org/domains/3");
        assert_eq!(domain.display_name.as_deref(), Some("Physical Sciences"));
        assert!(domain.fields.is_some());
        if let Some(fields) = &domain.fields {
            assert!(!fields.is_empty());
        }
        assert!(domain.siblings.is_some());
    }
}
