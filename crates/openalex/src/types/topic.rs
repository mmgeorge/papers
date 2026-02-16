use serde::Deserialize;

use super::common::*;

/// A research topic in OpenAlex's 3-level topic hierarchy.
///
/// Topics are organized as: **domain > field > subfield > topic**. There are
/// ~4,500 topics. Each topic has an AI-generated description and keywords. Works
/// are assigned up to 3 topics with relevance scores.
///
/// # Example
///
/// ```json
/// {
///   "id": "https://openalex.org/T10001",
///   "display_name": "Malaria Control and Epidemiology",
///   "description": "This cluster of papers focuses on...",
///   "keywords": ["malaria", "plasmodium falciparum", "antimalarial drugs", ...],
///   "subfield": {"id": 2725, "display_name": "Infectious Diseases"},
///   "field": {"id": 27, "display_name": "Medicine"},
///   "domain": {"id": 4, "display_name": "Health Sciences"},
///   "works_count": 143562
/// }
/// ```
///
/// # ID formats
///
/// Topics can only be retrieved by OpenAlex ID (`T...`).
///
/// # Note
///
/// Within `Topic` entities, the hierarchy level `id` fields are integers. When
/// topics appear nested in `Work` entities, these same fields are strings. The
/// [`TopicHierarchyLevel::id`] field uses `serde_json::Value` to handle this.
#[derive(Debug, Clone, Deserialize)]
pub struct Topic {
    /// OpenAlex ID URI (e.g. `"https://openalex.org/T10001"`).
    pub id: String,

    /// Human-readable topic name (e.g. `"Malaria Control and Epidemiology"`).
    pub display_name: Option<String>,

    /// AI-generated description of this research topic.
    pub description: Option<String>,

    /// Keywords associated with this topic, useful for understanding its scope.
    pub keywords: Option<Vec<String>>,

    /// External identifiers (OpenAlex, Wikipedia).
    pub ids: Option<TopicIds>,

    /// The subfield this topic belongs to in the hierarchy.
    pub subfield: Option<TopicHierarchyLevel>,

    /// The field this topic belongs to in the hierarchy.
    pub field: Option<TopicHierarchyLevel>,

    /// The domain this topic belongs to in the hierarchy.
    pub domain: Option<TopicHierarchyLevel>,

    /// Other topics at the same level in the hierarchy (siblings under the same
    /// subfield).
    pub siblings: Option<Vec<TopicSibling>>,

    /// Total number of works assigned to this topic.
    pub works_count: Option<i64>,

    /// Total number of citations received by works in this topic.
    pub cited_by_count: Option<i64>,

    /// API URL to retrieve works in this topic.
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
    fn test_deserialize_topic() {
        let json = include_str!("../../tests/fixtures/topic.json");
        let topic: Topic = serde_json::from_str(json).expect("Failed to deserialize Topic");
        assert_eq!(topic.id, "https://openalex.org/T10001");
        assert!(topic.display_name.is_some());
        assert!(topic.domain.is_some());
        assert!(topic.subfield.is_some());
        if let Some(keywords) = &topic.keywords {
            assert!(!keywords.is_empty());
        }
    }
}
