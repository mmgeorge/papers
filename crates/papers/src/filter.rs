use papers_openalex::{ListParams, OpenAlexClient, OpenAlexError};

// ── Alias specification types ────────────────────────────────────────────

/// Alias definition mapping a user-facing name to its OpenAlex filter key.
pub(crate) struct AliasSpec {
    pub name: &'static str,
    pub filter_key: &'static str,
    pub entity_type: &'static str,
    pub kind: AliasKind,
}

pub(crate) enum AliasKind {
    /// Resolve ID-based values (search strings → entity IDs via API).
    Entity,
    /// Pass value through directly (no API resolution needed).
    Direct,
    /// Boolean flag — when present and true, emits `filter_key:true`.
    Boolean,
}

// ── Error type ───────────────────────────────────────────────────────────

/// Error type for filter resolution.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    #[error("Conflict: '{alias}' alias maps to '{filter_key}' which is already in --filter")]
    Conflict {
        alias: &'static str,
        filter_key: &'static str,
    },
    #[error("No {entity_type} found matching \"{query}\"")]
    NotFound {
        entity_type: &'static str,
        query: String,
    },
    #[error(transparent)]
    Api(#[from] OpenAlexError),
}

// ── Alias constant arrays ────────────────────────────────────────────────

pub(crate) const WORK_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "author",      filter_key: "authorships.author.id",                     entity_type: "authors",       kind: AliasKind::Entity },
    AliasSpec { name: "topic",       filter_key: "primary_topic.id",                          entity_type: "topics",        kind: AliasKind::Entity },
    AliasSpec { name: "domain",      filter_key: "primary_topic.domain.id",                   entity_type: "domains",       kind: AliasKind::Entity },
    AliasSpec { name: "field",       filter_key: "primary_topic.field.id",                    entity_type: "fields",        kind: AliasKind::Entity },
    AliasSpec { name: "subfield",    filter_key: "primary_topic.subfield.id",                 entity_type: "subfields",     kind: AliasKind::Entity },
    AliasSpec { name: "publisher",   filter_key: "primary_location.source.publisher_lineage", entity_type: "publishers",    kind: AliasKind::Entity },
    AliasSpec { name: "source",      filter_key: "primary_location.source.id",                entity_type: "sources",       kind: AliasKind::Entity },
    AliasSpec { name: "institution", filter_key: "authorships.institutions.lineage",          entity_type: "institutions",  kind: AliasKind::Entity },
    AliasSpec { name: "year",        filter_key: "publication_year",                          entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "citations",   filter_key: "cited_by_count",                            entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "country",     filter_key: "authorships.institutions.country_code",     entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "continent",   filter_key: "authorships.institutions.continent",        entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "type",        filter_key: "type",                                      entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "open",        filter_key: "is_oa",                                     entity_type: "",              kind: AliasKind::Boolean },
];

pub(crate) const AUTHOR_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "institution", filter_key: "last_known_institutions.id",                entity_type: "institutions",  kind: AliasKind::Entity },
    AliasSpec { name: "country",     filter_key: "last_known_institutions.country_code",      entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "continent",   filter_key: "last_known_institutions.continent",         entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "citations",   filter_key: "cited_by_count",                            entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "h_index",     filter_key: "summary_stats.h_index",                     entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const SOURCE_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "publisher",   filter_key: "host_organization_lineage",                 entity_type: "publishers",    kind: AliasKind::Entity },
    AliasSpec { name: "country",     filter_key: "country_code",                              entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "continent",   filter_key: "continent",                                 entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "type",        filter_key: "type",                                      entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "open",        filter_key: "is_oa",                                     entity_type: "",              kind: AliasKind::Boolean },
    AliasSpec { name: "citations",   filter_key: "cited_by_count",                            entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const INSTITUTION_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "country",     filter_key: "country_code",                              entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "continent",   filter_key: "continent",                                 entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "type",        filter_key: "type",                                      entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "citations",   filter_key: "cited_by_count",                            entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const TOPIC_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "domain",      filter_key: "domain.id",                                 entity_type: "domains",       kind: AliasKind::Entity },
    AliasSpec { name: "field",       filter_key: "field.id",                                  entity_type: "fields",        kind: AliasKind::Entity },
    AliasSpec { name: "subfield",    filter_key: "subfield.id",                               entity_type: "subfields",     kind: AliasKind::Entity },
    AliasSpec { name: "citations",   filter_key: "cited_by_count",                            entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const PUBLISHER_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "country",     filter_key: "country_codes",                             entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "continent",   filter_key: "continent",                                 entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "citations",   filter_key: "cited_by_count",                            entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const FUNDER_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "country",     filter_key: "country_code",                              entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "continent",   filter_key: "continent",                                 entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "citations",   filter_key: "cited_by_count",                            entity_type: "",              kind: AliasKind::Direct },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const DOMAIN_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const FIELD_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "domain",      filter_key: "domain.id",                                 entity_type: "domains",       kind: AliasKind::Entity },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

pub(crate) const SUBFIELD_ALIASES: &[AliasSpec] = &[
    AliasSpec { name: "domain",      filter_key: "domain.id",                                 entity_type: "domains",       kind: AliasKind::Entity },
    AliasSpec { name: "field",       filter_key: "field.id",                                  entity_type: "fields",        kind: AliasKind::Entity },
    AliasSpec { name: "works",       filter_key: "works_count",                               entity_type: "",              kind: AliasKind::Direct },
];

// ── Generic resolve_filters ──────────────────────────────────────────────

/// Resolves filter aliases + raw filter into a single filter string.
///
/// Each alias is resolved (IDs passed through, search strings resolved via API),
/// then combined with the raw filter using comma-separated AND logic.
///
/// Returns `None` if no filters are active.
pub(crate) async fn resolve_filters(
    client: &OpenAlexClient,
    alias_specs: &[AliasSpec],
    alias_values: &[Option<String>],
    raw_filter: Option<&str>,
) -> Result<Option<String>, FilterError> {
    assert_eq!(alias_specs.len(), alias_values.len());

    // Build list of active aliases for overlap checking
    let active: Vec<(&'static str, &'static str)> = alias_specs
        .iter()
        .zip(alias_values.iter())
        .filter(|(_, v)| v.is_some())
        .map(|(spec, _)| (spec.name, spec.filter_key))
        .collect();

    // Check for overlap with raw filter
    if let Some(raw) = raw_filter {
        if !raw.is_empty() {
            check_filter_overlap(raw, &active)?;
        }
    }

    // Resolve each active alias
    let mut conditions: Vec<String> = Vec::new();

    for (spec, value) in alias_specs.iter().zip(alias_values.iter()) {
        if let Some(val) = value {
            let resolved_value = match spec.kind {
                AliasKind::Direct => val.clone(),
                AliasKind::Entity => {
                    resolve_alias_value(client, val, spec.entity_type).await?
                }
                AliasKind::Boolean => "true".to_string(),
            };
            conditions.push(format!("{}:{}", spec.filter_key, resolved_value));
        }
    }

    // Append raw filter conditions
    if let Some(raw) = raw_filter {
        if !raw.is_empty() {
            conditions.push(raw.to_string());
        }
    }

    if conditions.is_empty() {
        Ok(None)
    } else {
        Ok(Some(conditions.join(",")))
    }
}

// ── WorkListParams ───────────────────────────────────────────────────────

/// Combined parameters for `work_list`, including both standard list parameters
/// and shorthand filter aliases.
///
/// Standard list fields (`filter`, `search`, `sort`, etc.) work like `ListParams`.
/// Alias fields (`author`, `topic`, `year`, etc.) are resolved to OpenAlex filter
/// expressions — ID values pass through, search strings are resolved to the top
/// result by citation count.
#[derive(Debug, Default, Clone)]
pub struct WorkListParams {
    // ── Standard list parameters ─────────────────────────────────────
    pub filter: Option<String>,
    pub search: Option<String>,
    pub sort: Option<String>,
    pub per_page: Option<u32>,
    pub page: Option<u32>,
    pub cursor: Option<String>,
    pub sample: Option<u32>,
    pub seed: Option<u32>,
    pub select: Option<String>,
    pub group_by: Option<String>,
    // ── Filter aliases ───────────────────────────────────────────────
    pub author: Option<String>,
    pub topic: Option<String>,
    pub domain: Option<String>,
    pub field: Option<String>,
    pub subfield: Option<String>,
    pub publisher: Option<String>,
    pub source: Option<String>,
    pub institution: Option<String>,
    pub year: Option<String>,
    pub citations: Option<String>,
    pub country: Option<String>,
    pub continent: Option<String>,
    pub r#type: Option<String>,
    pub open: Option<bool>,
}

impl WorkListParams {
    pub(crate) fn into_aliases_and_list_params(&self) -> (Vec<Option<String>>, ListParams) {
        let alias_values = vec![
            self.author.clone(),
            self.topic.clone(),
            self.domain.clone(),
            self.field.clone(),
            self.subfield.clone(),
            self.publisher.clone(),
            self.source.clone(),
            self.institution.clone(),
            self.year.clone(),
            self.citations.clone(),
            self.country.clone(),
            self.continent.clone(),
            self.r#type.clone(),
            if self.open == Some(true) { Some("true".to_string()) } else { None },
        ];
        let list_params = ListParams {
            filter: self.filter.clone(),
            search: self.search.clone(),
            sort: self.sort.clone(),
            per_page: self.per_page,
            page: self.page,
            cursor: self.cursor.clone(),
            sample: self.sample,
            seed: self.seed,
            select: self.select.clone(),
            group_by: self.group_by.clone(),
        };
        (alias_values, list_params)
    }
}

/// Resolves work filter aliases + raw filter.
pub async fn resolve_work_filters(
    client: &OpenAlexClient,
    aliases: &WorkFilterAliases,
    raw_filter: Option<&str>,
) -> Result<Option<String>, FilterError> {
    let alias_values = vec![
        aliases.author.clone(),
        aliases.topic.clone(),
        aliases.domain.clone(),
        aliases.field.clone(),
        aliases.subfield.clone(),
        aliases.publisher.clone(),
        aliases.source.clone(),
        aliases.institution.clone(),
        aliases.year.clone(),
        aliases.citations.clone(),
        aliases.country.clone(),
        aliases.continent.clone(),
        aliases.r#type.clone(),
        if aliases.open == Some(true) { Some("true".to_string()) } else { None },
    ];
    resolve_filters(client, WORK_ALIASES, &alias_values, raw_filter).await
}

/// Alias fields for work list filters (kept for backward compatibility with tests).
#[derive(Debug, Default, Clone)]
pub struct WorkFilterAliases {
    pub author: Option<String>,
    pub topic: Option<String>,
    pub domain: Option<String>,
    pub field: Option<String>,
    pub subfield: Option<String>,
    pub publisher: Option<String>,
    pub source: Option<String>,
    pub institution: Option<String>,
    pub year: Option<String>,
    pub citations: Option<String>,
    pub country: Option<String>,
    pub continent: Option<String>,
    pub r#type: Option<String>,
    pub open: Option<bool>,
}

// ── Macro for entity list params ─────────────────────────────────────────

macro_rules! entity_list_params {
    (
        $name:ident,
        aliases: $aliases:ident,
        fields: [ $( $field:ident : $kind:ident ),* $(,)? ]
    ) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            // ── Standard list parameters ─────────────────────────────
            pub filter: Option<String>,
            pub search: Option<String>,
            pub sort: Option<String>,
            pub per_page: Option<u32>,
            pub page: Option<u32>,
            pub cursor: Option<String>,
            pub sample: Option<u32>,
            pub seed: Option<u32>,
            pub select: Option<String>,
            pub group_by: Option<String>,
            // ── Filter aliases ───────────────────────────────────────
            $( pub $field: entity_list_params!(@field_type $kind), )*
        }

        impl $name {
            pub(crate) fn into_aliases_and_list_params(&self) -> (Vec<Option<String>>, ListParams) {
                let alias_values = vec![
                    $( entity_list_params!(@to_option_string self, $field, $kind), )*
                ];
                let list_params = ListParams {
                    filter: self.filter.clone(),
                    search: self.search.clone(),
                    sort: self.sort.clone(),
                    per_page: self.per_page,
                    page: self.page,
                    cursor: self.cursor.clone(),
                    sample: self.sample,
                    seed: self.seed,
                    select: self.select.clone(),
                    group_by: self.group_by.clone(),
                };
                (alias_values, list_params)
            }

            pub(crate) fn alias_specs() -> &'static [AliasSpec] {
                $aliases
            }
        }
    };

    (@field_type string) => { Option<String> };
    (@field_type bool) => { Option<bool> };

    (@to_option_string $self:ident, $field:ident, string) => {
        $self.$field.clone()
    };
    (@to_option_string $self:ident, $field:ident, bool) => {
        if $self.$field == Some(true) { Some("true".to_string()) } else { None }
    };
}

entity_list_params!(AuthorListParams, aliases: AUTHOR_ALIASES, fields: [
    institution: string,
    country: string,
    continent: string,
    citations: string,
    works: string,
    h_index: string,
]);

entity_list_params!(SourceListParams, aliases: SOURCE_ALIASES, fields: [
    publisher: string,
    country: string,
    continent: string,
    r#type: string,
    open: bool,
    citations: string,
    works: string,
]);

entity_list_params!(InstitutionListParams, aliases: INSTITUTION_ALIASES, fields: [
    country: string,
    continent: string,
    r#type: string,
    citations: string,
    works: string,
]);

entity_list_params!(TopicListParams, aliases: TOPIC_ALIASES, fields: [
    domain: string,
    field: string,
    subfield: string,
    citations: string,
    works: string,
]);

entity_list_params!(PublisherListParams, aliases: PUBLISHER_ALIASES, fields: [
    country: string,
    continent: string,
    citations: string,
    works: string,
]);

entity_list_params!(FunderListParams, aliases: FUNDER_ALIASES, fields: [
    country: string,
    continent: string,
    citations: string,
    works: string,
]);

entity_list_params!(DomainListParams, aliases: DOMAIN_ALIASES, fields: [
    works: string,
]);

entity_list_params!(FieldListParams, aliases: FIELD_ALIASES, fields: [
    domain: string,
    works: string,
]);

entity_list_params!(SubfieldListParams, aliases: SUBFIELD_ALIASES, fields: [
    domain: string,
    field: string,
    works: string,
]);

// ── Helper functions ─────────────────────────────────────────────────────

/// Returns true if `value` looks like an OpenAlex ID for the given entity type.
fn is_openalex_id(value: &str, entity_type: &str) -> bool {
    // Full URL: https://openalex.org/...
    if value.starts_with("https://openalex.org/") {
        return true;
    }

    match entity_type {
        "authors" => is_prefixed_id(value, 'A'),
        "topics" => is_prefixed_id(value, 'T'),
        "sources" => is_prefixed_id(value, 'S'),
        "publishers" => is_prefixed_id(value, 'P'),
        "institutions" => is_prefixed_id(value, 'I'),
        "funders" => is_prefixed_id(value, 'F'),
        "domains" => is_hierarchy_id(value, "domains"),
        "fields" => is_hierarchy_id(value, "fields"),
        "subfields" => is_hierarchy_id(value, "subfields"),
        _ => false,
    }
}

/// Check if value matches letter-prefix + digits pattern (e.g. A123, P456).
fn is_prefixed_id(value: &str, prefix: char) -> bool {
    if let Some(rest) = value.strip_prefix(prefix) {
        !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
    } else {
        false
    }
}

/// Check if value matches hierarchy ID patterns:
/// - `domains/3`, `fields/17`, `subfields/1702`
/// - Bare digits: `3`, `17`, `1702`
fn is_hierarchy_id(value: &str, prefix: &str) -> bool {
    // Prefixed path: domains/3
    if let Some(rest) = value.strip_prefix(prefix).and_then(|s| s.strip_prefix('/')) {
        return !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit());
    }
    // Bare digits
    !value.is_empty() && value.chars().all(|c| c.is_ascii_digit())
}

/// Normalize an ID to short form:
/// 1. Strip `https://openalex.org/` prefix if present.
/// 2. For hierarchy entities with bare digits, prepend the path prefix.
fn normalize_id(raw_id: &str, entity_type: &str) -> String {
    let id = raw_id
        .strip_prefix("https://openalex.org/")
        .unwrap_or(raw_id);

    // If bare digits for hierarchy entities, prepend prefix
    if id.chars().all(|c| c.is_ascii_digit()) && !id.is_empty() {
        match entity_type {
            "domains" => return format!("domains/{id}"),
            "fields" => return format!("fields/{id}"),
            "subfields" => return format!("subfields/{id}"),
            "publishers" => return format!("P{id}"),
            _ => {}
        }
    }

    id.to_string()
}

/// Resolve a search string to an entity ID by querying the list endpoint.
///
/// For most entities: `GET /{entity_type}?filter=display_name.search:{query}&sort=cited_by_count:desc&per_page=1&select=id`
/// For publishers: `GET /publishers?search={query}&sort=cited_by_count:desc&per_page=1&select=id`
async fn resolve_entity_id(
    client: &OpenAlexClient,
    query: &str,
    entity_type: &'static str,
) -> Result<String, FilterError> {
    let params = if entity_type == "publishers" {
        // Publishers use `search` param to match alternate titles
        ListParams {
            search: Some(query.to_string()),
            sort: Some("cited_by_count:desc".to_string()),
            per_page: Some(1),
            select: Some("id,display_name".to_string()),
            ..Default::default()
        }
    } else {
        ListParams {
            filter: Some(format!("display_name.search:{query}")),
            sort: Some("cited_by_count:desc".to_string()),
            per_page: Some(1),
            select: Some("id,display_name".to_string()),
            ..Default::default()
        }
    };

    // Use serde_json::Value for generic deserialization since we only need the ID
    let result: papers_openalex::ListResponse<serde_json::Value> = match entity_type {
        "authors" => transmute_list(client.list_authors(&params).await?),
        "topics" => transmute_list(client.list_topics(&params).await?),
        "domains" => transmute_list(client.list_domains(&params).await?),
        "fields" => transmute_list(client.list_fields(&params).await?),
        "subfields" => transmute_list(client.list_subfields(&params).await?),
        "publishers" => transmute_list(client.list_publishers(&params).await?),
        "sources" => transmute_list(client.list_sources(&params).await?),
        "institutions" => transmute_list(client.list_institutions(&params).await?),
        _ => unreachable!("unknown entity type: {entity_type}"),
    };

    let first = result.results.into_iter().next().ok_or_else(|| FilterError::NotFound {
        entity_type,
        query: query.to_string(),
    })?;

    let id = first["id"]
        .as_str()
        .ok_or_else(|| FilterError::NotFound {
            entity_type,
            query: query.to_string(),
        })?;

    Ok(normalize_id(id, entity_type))
}

/// Convert a typed ListResponse into a ListResponse<serde_json::Value>.
fn transmute_list<T: serde::Serialize>(
    resp: papers_openalex::ListResponse<T>,
) -> papers_openalex::ListResponse<serde_json::Value> {
    papers_openalex::ListResponse {
        meta: resp.meta,
        results: resp
            .results
            .into_iter()
            .map(|r| serde_json::to_value(r).unwrap())
            .collect(),
        group_by: resp.group_by,
    }
}

/// Check that alias filter keys don't overlap with keys in the raw filter string.
fn check_filter_overlap(
    raw_filter: &str,
    active_aliases: &[(&'static str, &'static str)],
) -> Result<(), FilterError> {
    for condition in raw_filter.split(',') {
        let condition = condition.trim();
        if condition.is_empty() {
            continue;
        }
        // Extract the key: everything before the first `:`
        // Strip leading `!` for negated filters
        let key = condition
            .strip_prefix('!')
            .unwrap_or(condition)
            .split(':')
            .next()
            .unwrap_or("");

        for &(alias_name, filter_key) in active_aliases {
            if key == filter_key {
                return Err(FilterError::Conflict {
                    alias: alias_name,
                    filter_key,
                });
            }
        }
    }
    Ok(())
}

/// Resolve a pipe-separated alias value where each segment may be an ID or search string.
async fn resolve_alias_value(
    client: &OpenAlexClient,
    value: &str,
    entity_type: &'static str,
) -> Result<String, FilterError> {
    let mut resolved = Vec::new();
    for segment in value.split('|') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        if is_openalex_id(segment, entity_type) {
            resolved.push(normalize_id(segment, entity_type));
        } else {
            resolved.push(resolve_entity_id(client, segment, entity_type).await?);
        }
    }
    Ok(resolved.join("|"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── is_openalex_id tests ────────────────────────────────────────────

    #[test]
    fn test_is_id_author() {
        assert!(is_openalex_id("A5083138872", "authors"));
        assert!(!is_openalex_id("einstein", "authors"));
        assert!(!is_openalex_id("A", "authors")); // no digits
    }

    #[test]
    fn test_is_id_publisher() {
        assert!(is_openalex_id("P4310319798", "publishers"));
        assert!(!is_openalex_id("acm", "publishers"));
    }

    #[test]
    fn test_is_id_institution() {
        assert!(is_openalex_id("I136199984", "institutions"));
        assert!(!is_openalex_id("mit", "institutions"));
        assert!(!is_openalex_id("I", "institutions"));
    }

    #[test]
    fn test_is_id_funder() {
        assert!(is_openalex_id("F1234567", "funders"));
        assert!(!is_openalex_id("nih", "funders"));
    }

    #[test]
    fn test_is_id_domain() {
        assert!(is_openalex_id("3", "domains"));
        assert!(is_openalex_id("domains/3", "domains"));
        assert!(!is_openalex_id("physical", "domains"));
    }

    #[test]
    fn test_is_id_field() {
        assert!(is_openalex_id("17", "fields"));
        assert!(is_openalex_id("fields/17", "fields"));
        assert!(!is_openalex_id("computer science", "fields"));
    }

    #[test]
    fn test_is_id_subfield() {
        assert!(is_openalex_id("1702", "subfields"));
        assert!(is_openalex_id("subfields/1702", "subfields"));
        assert!(!is_openalex_id("artificial intelligence", "subfields"));
    }

    #[test]
    fn test_is_id_topic() {
        assert!(is_openalex_id("T11636", "topics"));
        assert!(!is_openalex_id("machine learning", "topics"));
    }

    #[test]
    fn test_is_id_source() {
        assert!(is_openalex_id("S131921510", "sources"));
        assert!(!is_openalex_id("siggraph", "sources"));
    }

    #[test]
    fn test_is_id_full_url() {
        assert!(is_openalex_id("https://openalex.org/A5083138872", "authors"));
        assert!(is_openalex_id("https://openalex.org/P123", "publishers"));
        assert!(is_openalex_id("https://openalex.org/domains/3", "domains"));
    }

    // ── normalize_id tests ──────────────────────────────────────────────

    #[test]
    fn test_normalize_id_strip_url() {
        assert_eq!(normalize_id("https://openalex.org/A123", "authors"), "A123");
        assert_eq!(
            normalize_id("https://openalex.org/domains/3", "domains"),
            "domains/3"
        );
    }

    #[test]
    fn test_normalize_id_bare_digits_publisher() {
        assert_eq!(normalize_id("4310319798", "publishers"), "P4310319798");
    }

    #[test]
    fn test_normalize_id_bare_digits_domain() {
        assert_eq!(normalize_id("3", "domains"), "domains/3");
    }

    #[test]
    fn test_normalize_id_bare_digits_field() {
        assert_eq!(normalize_id("17", "fields"), "fields/17");
    }

    #[test]
    fn test_normalize_id_bare_digits_subfield() {
        assert_eq!(normalize_id("1702", "subfields"), "subfields/1702");
    }

    #[test]
    fn test_normalize_id_already_short() {
        assert_eq!(normalize_id("A123", "authors"), "A123");
        assert_eq!(normalize_id("domains/3", "domains"), "domains/3");
    }

    // ── check_filter_overlap tests ──────────────────────────────────────

    #[test]
    fn test_overlap_detected() {
        let active = vec![("year", "publication_year")];
        let result = check_filter_overlap("publication_year:>2020", &active);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("year"));
        assert!(err.contains("publication_year"));
    }

    #[test]
    fn test_overlap_with_negation() {
        let active = vec![("year", "publication_year")];
        let result = check_filter_overlap("!publication_year:2020", &active);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_overlap() {
        let active = vec![("year", "publication_year")];
        let result = check_filter_overlap("is_oa:true", &active);
        assert!(result.is_ok());
    }

    #[test]
    fn test_no_overlap_empty_filter() {
        let active = vec![("year", "publication_year")];
        let result = check_filter_overlap("", &active);
        assert!(result.is_ok());
    }

    // ── resolve_work_filters unit tests (no API calls) ──────────────────

    #[tokio::test]
    async fn test_direct_value_year() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            year: Some(">2008".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(result.as_deref(), Some("publication_year:>2008"));
    }

    #[tokio::test]
    async fn test_direct_value_citations() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            citations: Some("100-500".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(result.as_deref(), Some("cited_by_count:100-500"));
    }

    #[tokio::test]
    async fn test_id_passthrough() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            author: Some("A5083138872".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("authorships.author.id:A5083138872")
        );
    }

    #[tokio::test]
    async fn test_combine_aliases_and_raw() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            year: Some("2024".to_string()),
            citations: Some(">100".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, Some("is_oa:true"))
            .await
            .unwrap();
        let filter = result.unwrap();
        assert!(filter.contains("publication_year:2024"));
        assert!(filter.contains("cited_by_count:>100"));
        assert!(filter.contains("is_oa:true"));
    }

    #[tokio::test]
    async fn test_pipe_separated_ids() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            author: Some("A123|A456".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("authorships.author.id:A123|A456")
        );
    }

    #[tokio::test]
    async fn test_no_filters_returns_none() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases::default();
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_raw_filter_only() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases::default();
        let result = resolve_work_filters(&client, &aliases, Some("is_oa:true"))
            .await
            .unwrap();
        assert_eq!(result.as_deref(), Some("is_oa:true"));
    }

    #[tokio::test]
    async fn test_overlap_error() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            year: Some("2024".to_string()),
            ..Default::default()
        };
        let result =
            resolve_work_filters(&client, &aliases, Some("publication_year:>2020")).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("year"));
        assert!(err.contains("publication_year"));
    }

    #[tokio::test]
    async fn test_domain_id_bare_digits() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            domain: Some("3".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("primary_topic.domain.id:domains/3")
        );
    }

    #[tokio::test]
    async fn test_field_id_bare_digits() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            field: Some("17".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("primary_topic.field.id:fields/17")
        );
    }

    #[tokio::test]
    async fn test_subfield_id_bare_digits() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            subfield: Some("1702".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("primary_topic.subfield.id:subfields/1702")
        );
    }

    #[tokio::test]
    async fn test_topic_id_passthrough() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            topic: Some("T11636".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(result.as_deref(), Some("primary_topic.id:T11636"));
    }

    #[tokio::test]
    async fn test_source_id_passthrough() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            source: Some("S131921510".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("primary_location.source.id:S131921510")
        );
    }

    #[tokio::test]
    async fn test_publisher_id_passthrough() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            publisher: Some("P4310319798".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("primary_location.source.publisher_lineage:P4310319798")
        );
    }

    #[tokio::test]
    async fn test_full_url_id() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            author: Some("https://openalex.org/A5083138872".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("authorships.author.id:A5083138872")
        );
    }

    // ── New alias tests ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_institution_id_passthrough() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            institution: Some("I136199984".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("authorships.institutions.lineage:I136199984")
        );
    }

    #[tokio::test]
    async fn test_country_direct() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            country: Some("US".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("authorships.institutions.country_code:US")
        );
    }

    #[tokio::test]
    async fn test_continent_direct() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            continent: Some("europe".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(
            result.as_deref(),
            Some("authorships.institutions.continent:europe")
        );
    }

    #[tokio::test]
    async fn test_type_direct() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            r#type: Some("article".to_string()),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(result.as_deref(), Some("type:article"));
    }

    #[tokio::test]
    async fn test_open_boolean() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            open: Some(true),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert_eq!(result.as_deref(), Some("is_oa:true"));
    }

    #[tokio::test]
    async fn test_open_false_is_noop() {
        let client = OpenAlexClient::new();
        let aliases = WorkFilterAliases {
            open: Some(false),
            ..Default::default()
        };
        let result = resolve_work_filters(&client, &aliases, None).await.unwrap();
        assert!(result.is_none());
    }

    // ── Per-entity alias unit tests ─────────────────────────────────────

    #[tokio::test]
    async fn test_author_citations_direct() {
        let client = OpenAlexClient::new();
        let params = AuthorListParams {
            citations: Some(">1000".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, AuthorListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("cited_by_count:>1000"));
    }

    #[tokio::test]
    async fn test_author_h_index_direct() {
        let client = OpenAlexClient::new();
        let params = AuthorListParams {
            h_index: Some(">50".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, AuthorListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("summary_stats.h_index:>50"));
    }

    #[tokio::test]
    async fn test_author_works_direct() {
        let client = OpenAlexClient::new();
        let params = AuthorListParams {
            works: Some(">500".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, AuthorListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("works_count:>500"));
    }

    #[tokio::test]
    async fn test_author_institution_id_passthrough() {
        let client = OpenAlexClient::new();
        let params = AuthorListParams {
            institution: Some("I136199984".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, AuthorListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("last_known_institutions.id:I136199984"));
    }

    #[tokio::test]
    async fn test_source_open_boolean() {
        let client = OpenAlexClient::new();
        let params = SourceListParams {
            open: Some(true),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, SourceListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("is_oa:true"));
    }

    #[tokio::test]
    async fn test_source_type_direct() {
        let client = OpenAlexClient::new();
        let params = SourceListParams {
            r#type: Some("journal".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, SourceListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("type:journal"));
    }

    #[tokio::test]
    async fn test_institution_country_direct() {
        let client = OpenAlexClient::new();
        let params = InstitutionListParams {
            country: Some("US".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, InstitutionListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("country_code:US"));
    }

    #[tokio::test]
    async fn test_institution_type_direct() {
        let client = OpenAlexClient::new();
        let params = InstitutionListParams {
            r#type: Some("education".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, InstitutionListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("type:education"));
    }

    #[tokio::test]
    async fn test_topic_domain_id() {
        let client = OpenAlexClient::new();
        let params = TopicListParams {
            domain: Some("3".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, TopicListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("domain.id:domains/3"));
    }

    #[tokio::test]
    async fn test_topic_field_id() {
        let client = OpenAlexClient::new();
        let params = TopicListParams {
            field: Some("17".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, TopicListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("field.id:fields/17"));
    }

    #[tokio::test]
    async fn test_topic_subfield_id() {
        let client = OpenAlexClient::new();
        let params = TopicListParams {
            subfield: Some("1702".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, TopicListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("subfield.id:subfields/1702"));
    }

    #[tokio::test]
    async fn test_publisher_country_direct() {
        let client = OpenAlexClient::new();
        let params = PublisherListParams {
            country: Some("US".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, PublisherListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("country_codes:US"));
    }

    #[tokio::test]
    async fn test_funder_country_direct() {
        let client = OpenAlexClient::new();
        let params = FunderListParams {
            country: Some("US".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, FunderListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("country_code:US"));
    }

    #[tokio::test]
    async fn test_domain_works_direct() {
        let client = OpenAlexClient::new();
        let params = DomainListParams {
            works: Some(">100000000".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, DomainListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("works_count:>100000000"));
    }

    #[tokio::test]
    async fn test_field_domain_id() {
        let client = OpenAlexClient::new();
        let params = FieldListParams {
            domain: Some("3".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, FieldListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("domain.id:domains/3"));
    }

    #[tokio::test]
    async fn test_subfield_field_id() {
        let client = OpenAlexClient::new();
        let params = SubfieldListParams {
            field: Some("17".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, SubfieldListParams::alias_specs(), &values, list_params.filter.as_deref()).await.unwrap();
        assert_eq!(result.as_deref(), Some("field.id:fields/17"));
    }

    #[tokio::test]
    async fn test_overlap_author_cited_by_count() {
        let client = OpenAlexClient::new();
        let params = AuthorListParams {
            citations: Some(">100".to_string()),
            filter: Some("cited_by_count:>50".to_string()),
            ..Default::default()
        };
        let (values, list_params) = params.into_aliases_and_list_params();
        let result = resolve_filters(&client, AuthorListParams::alias_specs(), &values, list_params.filter.as_deref()).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("citations"));
        assert!(err.contains("cited_by_count"));
    }
}
