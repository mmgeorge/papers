use serde::{Deserialize, Serialize};

use super::common::*;

/// A research subfield in OpenAlex's topic hierarchy.
///
/// Subfields are the third level: **domain > field > subfield > topic**.
/// There are ~252 subfields (e.g. Artificial Intelligence, Organic Chemistry).
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/subfields/1702",
///   "display_name": "Artificial Intelligence",
///   "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
///   "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
///   "topics": [{"id": "https://openalex.org/T10028", "display_name": "Topic Modeling"}, ...],
///   "works_count": 9059921
/// }
/// ```
///
/// # ID formats
///
/// Subfields use numeric IDs (e.g. `1702` for Artificial Intelligence).
///
/// # Note
///
/// Subfields **do** support autocomplete (unlike domains and fields).
/// However, the autocomplete response returns `entity_type: null` and
/// `short_id: "Nones/..."` — these are known API quirks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subfield {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/subfields/1702"`).
    pub id: String,

    /// Human-readable subfield name (e.g. `"Artificial Intelligence"`).
    pub display_name: Option<String>,

    /// Brief description of the subfield's scope.
    pub description: Option<String>,

    /// External identifiers (OpenAlex, Wikidata, Wikipedia).
    pub ids: Option<HierarchyIds>,

    /// Alternative names or name variants.
    pub display_name_alternatives: Option<Vec<String>>,

    /// The parent field this subfield belongs to.
    pub field: Option<HierarchyEntity>,

    /// The parent domain this subfield belongs to (grandparent in hierarchy).
    pub domain: Option<HierarchyEntity>,

    /// Research topics within this subfield.
    pub topics: Option<Vec<HierarchyEntity>>,

    /// Other subfields at the same level in the hierarchy.
    pub siblings: Option<Vec<HierarchyEntity>>,

    /// Total number of works in this subfield.
    pub works_count: Option<i64>,

    /// Total number of citations received by works in this subfield.
    pub cited_by_count: Option<i64>,

    /// API URL to retrieve works in this subfield.
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
    fn test_deserialize_subfield() {
        let json = include_str!("../../tests/fixtures/subfield.json");
        let subfield: Subfield =
            serde_json::from_str(json).expect("Failed to deserialize Subfield");
        assert_eq!(subfield.id, "https://openalex.org/subfields/1702");
        assert_eq!(
            subfield.display_name.as_deref(),
            Some("Artificial Intelligence")
        );
        assert!(subfield.field.is_some());
        assert!(subfield.domain.is_some());
        assert!(subfield.topics.is_some());
        if let Some(topics) = &subfield.topics {
            assert!(!topics.is_empty());
        }
        assert!(subfield.siblings.is_some());
    }
}
