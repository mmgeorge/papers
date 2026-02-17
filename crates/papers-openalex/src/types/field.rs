use serde::{Deserialize, Serialize};

use super::common::*;

/// An academic field in OpenAlex's topic hierarchy.
///
/// Fields are the second level: **domain > field > subfield > topic**.
/// There are 26 fields (e.g. Computer Science, Medicine, Mathematics).
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/fields/17",
///   "display_name": "Computer Science",
///   "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
///   "subfields": [{"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"}, ...],
///   "works_count": 22038624
/// }
/// ```
///
/// # ID formats
///
/// Fields use numeric IDs (e.g. `17` for Computer Science).
///
/// # Note
///
/// Fields do **not** support autocomplete (`/autocomplete/fields` returns 404).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/fields/17"`).
    pub id: String,

    /// Human-readable field name (e.g. `"Computer Science"`).
    pub display_name: Option<String>,

    /// Brief description of the field's scope.
    pub description: Option<String>,

    /// External identifiers (OpenAlex, Wikidata, Wikipedia).
    pub ids: Option<HierarchyIds>,

    /// Alternative names or name variants.
    pub display_name_alternatives: Option<Vec<String>>,

    /// The parent domain this field belongs to.
    pub domain: Option<HierarchyEntity>,

    /// Research subfields within this field.
    pub subfields: Option<Vec<HierarchyEntity>>,

    /// Other fields at the same level in the hierarchy.
    pub siblings: Option<Vec<HierarchyEntity>>,

    /// Total number of works in this field.
    pub works_count: Option<i64>,

    /// Total number of citations received by works in this field.
    pub cited_by_count: Option<i64>,

    /// API URL to retrieve works in this field.
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
    fn test_deserialize_field() {
        let json = include_str!("../../tests/fixtures/field.json");
        let field: Field = serde_json::from_str(json).expect("Failed to deserialize Field");
        assert_eq!(field.id, "https://openalex.org/fields/17");
        assert_eq!(field.display_name.as_deref(), Some("Computer Science"));
        assert!(field.domain.is_some());
        assert!(field.subfields.is_some());
        if let Some(subfields) = &field.subfields {
            assert!(!subfields.is_empty());
        }
        assert!(field.siblings.is_some());
    }
}
