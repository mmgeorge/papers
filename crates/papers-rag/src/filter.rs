use crate::error::RagError;

/// Builds a SQL WHERE clause from optional filter components.
pub struct FilterBuilder {
    clauses: Vec<String>,
}

impl FilterBuilder {
    pub fn new() -> Self {
        Self { clauses: Vec::new() }
    }

    pub fn paper_ids(mut self, ids: &[String]) -> Self {
        if ids.is_empty() {
            return self;
        }
        let list = ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(", ");
        self.clauses.push(format!("paper_id IN ({list})"));
        self
    }

    pub fn chapter_idx(mut self, idx: u16) -> Self {
        self.clauses.push(format!("chapter_idx = {idx}"));
        self
    }

    pub fn section_idx(mut self, idx: u16) -> Self {
        self.clauses.push(format!("section_idx = {idx}"));
        self
    }

    pub fn eq_str(mut self, col: &str, val: &str) -> Self {
        let escaped = val.replace('\'', "''");
        self.clauses.push(format!("{col} = '{escaped}'"));
        self
    }

    pub fn eq_u16(mut self, col: &str, val: u16) -> Self {
        self.clauses.push(format!("{col} = {val}"));
        self
    }

    pub fn year_range(mut self, min: Option<u16>, max: Option<u16>) -> Self {
        if let Some(min) = min {
            self.clauses.push(format!("year >= {min}"));
        }
        if let Some(max) = max {
            self.clauses.push(format!("year <= {max}"));
        }
        self
    }

    /// Filter: tags list contains any of the given tags.
    pub fn tags_any(mut self, tags: &[String]) -> Self {
        if tags.is_empty() {
            return self;
        }
        // Use OR-joined array_contains for each tag
        let conditions: Vec<String> = tags
            .iter()
            .map(|t| {
                let escaped = t.replace('\'', "''");
                format!("array_has(tags, '{escaped}')")
            })
            .collect();
        if conditions.len() == 1 {
            self.clauses.push(conditions.into_iter().next().unwrap());
        } else {
            self.clauses
                .push(format!("({})", conditions.join(" OR ")));
        }
        self
    }

    /// Build the final WHERE clause string, or None if no filters were added.
    pub fn build(self) -> Option<String> {
        if self.clauses.is_empty() {
            None
        } else {
            Some(self.clauses.join(" AND "))
        }
    }
}

/// Validate that scope parameters form a valid hierarchy.
/// section_idx requires chapter_idx; chapter_idx requires paper_id.
pub fn validate_scope(
    chapter_idx: Option<u16>,
    section_idx: Option<u16>,
    paper_id: Option<&str>,
) -> Result<(), RagError> {
    if section_idx.is_some() && chapter_idx.is_none() {
        return Err(RagError::Scope(
            "section_idx requires chapter_idx".into(),
        ));
    }
    if chapter_idx.is_some() && paper_id.is_none() {
        return Err(RagError::Scope(
            "chapter_idx requires paper_id".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── FilterBuilder ────────────────────────────────────────────────────────

    #[test]
    fn empty_filter_returns_none() {
        assert_eq!(FilterBuilder::new().build(), None);
    }

    #[test]
    fn single_paper_id() {
        let ids = vec!["abc123".to_string()];
        let f = FilterBuilder::new().paper_ids(&ids).build().unwrap();
        assert_eq!(f, "paper_id IN ('abc123')");
    }

    #[test]
    fn multiple_paper_ids() {
        let ids = vec!["id1".to_string(), "id2".to_string(), "id3".to_string()];
        let f = FilterBuilder::new().paper_ids(&ids).build().unwrap();
        assert_eq!(f, "paper_id IN ('id1', 'id2', 'id3')");
    }

    #[test]
    fn paper_id_with_single_quote_is_escaped() {
        let ids = vec!["it's".to_string()];
        let f = FilterBuilder::new().paper_ids(&ids).build().unwrap();
        assert!(f.contains("it''s"), "got: {f}");
    }

    #[test]
    fn empty_paper_ids_slice_adds_no_clause() {
        let f = FilterBuilder::new().paper_ids(&[]).build();
        assert_eq!(f, None);
    }

    #[test]
    fn chapter_idx_clause() {
        let f = FilterBuilder::new().chapter_idx(3).build().unwrap();
        assert_eq!(f, "chapter_idx = 3");
    }

    #[test]
    fn section_idx_clause() {
        let f = FilterBuilder::new().section_idx(7).build().unwrap();
        assert_eq!(f, "section_idx = 7");
    }

    #[test]
    fn year_range_both_bounds() {
        let f = FilterBuilder::new()
            .year_range(Some(2020), Some(2024))
            .build()
            .unwrap();
        assert_eq!(f, "year >= 2020 AND year <= 2024");
    }

    #[test]
    fn year_range_min_only() {
        let f = FilterBuilder::new()
            .year_range(Some(2021), None)
            .build()
            .unwrap();
        assert_eq!(f, "year >= 2021");
    }

    #[test]
    fn year_range_max_only() {
        let f = FilterBuilder::new()
            .year_range(None, Some(2022))
            .build()
            .unwrap();
        assert_eq!(f, "year <= 2022");
    }

    #[test]
    fn year_range_both_none_adds_no_clause() {
        let f = FilterBuilder::new().year_range(None, None).build();
        assert_eq!(f, None);
    }

    #[test]
    fn tags_any_single_tag() {
        let tags = vec!["GPU".to_string()];
        let f = FilterBuilder::new().tags_any(&tags).build().unwrap();
        assert_eq!(f, "array_has(tags, 'GPU')");
    }

    #[test]
    fn tags_any_multiple_tags_uses_or() {
        let tags = vec!["GPU".to_string(), "rendering".to_string()];
        let f = FilterBuilder::new().tags_any(&tags).build().unwrap();
        assert_eq!(f, "(array_has(tags, 'GPU') OR array_has(tags, 'rendering'))");
    }

    #[test]
    fn tags_any_quote_in_tag_is_escaped() {
        let tags = vec!["can't".to_string()];
        let f = FilterBuilder::new().tags_any(&tags).build().unwrap();
        assert!(f.contains("can''t"), "got: {f}");
    }

    #[test]
    fn tags_any_empty_slice_adds_no_clause() {
        let f = FilterBuilder::new().tags_any(&[]).build();
        assert_eq!(f, None);
    }

    #[test]
    fn eq_str_clause() {
        let f = FilterBuilder::new().eq_str("venue", "SIGGRAPH").build().unwrap();
        assert_eq!(f, "venue = 'SIGGRAPH'");
    }

    #[test]
    fn eq_str_escapes_single_quote() {
        let f = FilterBuilder::new().eq_str("venue", "it's").build().unwrap();
        assert_eq!(f, "venue = 'it''s'");
    }

    #[test]
    fn multiple_clauses_joined_with_and() {
        let ids = vec!["p1".to_string()];
        let f = FilterBuilder::new()
            .paper_ids(&ids)
            .chapter_idx(2)
            .eq_str("depth", "paragraph")
            .build()
            .unwrap();
        assert_eq!(f, "paper_id IN ('p1') AND chapter_idx = 2 AND depth = 'paragraph'");
    }

    // ── validate_scope ───────────────────────────────────────────────────────

    #[test]
    fn validate_no_scope_ok() {
        validate_scope(None, None, None).unwrap();
    }

    #[test]
    fn validate_paper_only_ok() {
        validate_scope(None, None, Some("p1")).unwrap();
    }

    #[test]
    fn validate_paper_and_chapter_ok() {
        validate_scope(Some(1), None, Some("p1")).unwrap();
    }

    #[test]
    fn validate_paper_chapter_section_ok() {
        validate_scope(Some(1), Some(2), Some("p1")).unwrap();
    }

    #[test]
    fn validate_section_without_chapter_is_err() {
        let err = validate_scope(None, Some(1), Some("p1")).unwrap_err();
        assert!(err.to_string().contains("chapter_idx"), "got: {err}");
    }

    #[test]
    fn validate_chapter_without_paper_is_err() {
        let err = validate_scope(Some(1), None, None).unwrap_err();
        assert!(err.to_string().contains("paper_id"), "got: {err}");
    }

    #[test]
    fn validate_section_without_chapter_or_paper_is_err() {
        let err = validate_scope(None, Some(2), None).unwrap_err();
        assert!(err.to_string().contains("chapter_idx"), "got: {err}");
    }
}
