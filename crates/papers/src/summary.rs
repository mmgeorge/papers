use openalex::{Author, Funder, Institution, ListMeta, ListResponse, Publisher, Source, Topic, Work};
use openalex::OpenAlexError;
use serde::Serialize;

/// Slim wrapper returned by all list functions — keeps meta but drops group_by
/// and maps full entities to their summary equivalents.
#[derive(Serialize)]
pub struct SlimListResponse<S: Serialize> {
    pub meta: ListMeta,
    pub results: Vec<S>,
}

pub fn summary_list_result<T, S: Serialize>(
    result: Result<ListResponse<T>, OpenAlexError>,
    f: impl Fn(T) -> S,
) -> Result<SlimListResponse<S>, OpenAlexError> {
    result.map(|r| SlimListResponse {
        meta: r.meta,
        results: r.results.into_iter().map(f).collect(),
    })
}

// ── WorkSummary ───────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct WorkSummary {
    pub id: String,
    pub title: Option<String>,
    pub doi: Option<String>,
    pub publication_year: Option<i32>,
    pub r#type: Option<String>,
    pub authors: Vec<String>,
    pub journal: Option<String>,
    pub is_oa: Option<bool>,
    pub oa_url: Option<String>,
    pub cited_by_count: Option<i64>,
    pub primary_topic: Option<String>,
    pub abstract_text: Option<String>,
}

impl From<Work> for WorkSummary {
    fn from(w: Work) -> Self {
        let authors = w
            .authorships
            .unwrap_or_default()
            .into_iter()
            .filter_map(|a| a.author.and_then(|au| au.display_name))
            .collect();

        let journal = w
            .primary_location
            .as_ref()
            .and_then(|l| l.source.as_ref())
            .and_then(|s| s.display_name.clone());

        let is_oa = w.open_access.as_ref().and_then(|oa| oa.is_oa);
        let oa_url = w.open_access.and_then(|oa| oa.oa_url);

        let primary_topic = w
            .primary_topic
            .and_then(|t| t.display_name);

        WorkSummary {
            id: w.id,
            title: w.display_name,
            doi: w.doi,
            publication_year: w.publication_year,
            r#type: w.r#type,
            authors,
            journal,
            is_oa,
            oa_url,
            cited_by_count: w.cited_by_count,
            primary_topic,
            abstract_text: w.abstract_text,
        }
    }
}

// ── AuthorSummary ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct AuthorSummary {
    pub id: String,
    pub display_name: Option<String>,
    pub orcid: Option<String>,
    pub works_count: Option<i64>,
    pub cited_by_count: Option<i64>,
    pub h_index: Option<i64>,
    pub last_known_institutions: Vec<String>,
    pub top_topics: Vec<String>,
}

impl From<Author> for AuthorSummary {
    fn from(a: Author) -> Self {
        let h_index = a.summary_stats.as_ref().and_then(|s| s.h_index);

        let last_known_institutions = a
            .last_known_institutions
            .unwrap_or_default()
            .into_iter()
            .filter_map(|i| i.display_name)
            .collect();

        let top_topics = a
            .topics
            .unwrap_or_default()
            .into_iter()
            .take(3)
            .filter_map(|t| t.display_name)
            .collect();

        AuthorSummary {
            id: a.id,
            display_name: a.display_name,
            orcid: a.orcid,
            works_count: a.works_count,
            cited_by_count: a.cited_by_count,
            h_index,
            last_known_institutions,
            top_topics,
        }
    }
}

// ── SourceSummary ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct SourceSummary {
    pub id: String,
    pub display_name: Option<String>,
    pub issn_l: Option<String>,
    pub r#type: Option<String>,
    pub is_oa: Option<bool>,
    pub is_in_doaj: Option<bool>,
    pub works_count: Option<i64>,
    pub cited_by_count: Option<i64>,
    pub h_index: Option<i64>,
    pub host_organization_name: Option<String>,
}

impl From<Source> for SourceSummary {
    fn from(s: Source) -> Self {
        let h_index = s.summary_stats.as_ref().and_then(|st| st.h_index);

        SourceSummary {
            id: s.id,
            display_name: s.display_name,
            issn_l: s.issn_l,
            r#type: s.r#type,
            is_oa: s.is_oa,
            is_in_doaj: s.is_in_doaj,
            works_count: s.works_count,
            cited_by_count: s.cited_by_count,
            h_index,
            host_organization_name: s.host_organization_name,
        }
    }
}

// ── InstitutionSummary ────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct InstitutionSummary {
    pub id: String,
    pub display_name: Option<String>,
    pub ror: Option<String>,
    pub country_code: Option<String>,
    pub r#type: Option<String>,
    pub city: Option<String>,
    pub works_count: Option<i64>,
    pub cited_by_count: Option<i64>,
    pub h_index: Option<i64>,
}

impl From<Institution> for InstitutionSummary {
    fn from(i: Institution) -> Self {
        let h_index = i.summary_stats.as_ref().and_then(|s| s.h_index);
        let city = i.geo.and_then(|g| g.city);

        InstitutionSummary {
            id: i.id,
            display_name: i.display_name,
            ror: i.ror,
            country_code: i.country_code,
            r#type: i.r#type,
            city,
            works_count: i.works_count,
            cited_by_count: i.cited_by_count,
            h_index,
        }
    }
}

// ── TopicSummary ──────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct TopicSummary {
    pub id: String,
    pub display_name: Option<String>,
    pub description: Option<String>,
    pub subfield: Option<String>,
    pub field: Option<String>,
    pub domain: Option<String>,
    pub works_count: Option<i64>,
    pub cited_by_count: Option<i64>,
}

impl From<Topic> for TopicSummary {
    fn from(t: Topic) -> Self {
        let subfield = t.subfield.and_then(|s| s.display_name);
        let field = t.field.and_then(|f| f.display_name);
        let domain = t.domain.and_then(|d| d.display_name);

        TopicSummary {
            id: t.id,
            display_name: t.display_name,
            description: t.description,
            subfield,
            field,
            domain,
            works_count: t.works_count,
            cited_by_count: t.cited_by_count,
        }
    }
}

// ── PublisherSummary ──────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct PublisherSummary {
    pub id: String,
    pub display_name: Option<String>,
    pub hierarchy_level: Option<i32>,
    pub country_codes: Option<Vec<String>>,
    pub works_count: Option<i64>,
    pub cited_by_count: Option<i64>,
}

impl From<Publisher> for PublisherSummary {
    fn from(p: Publisher) -> Self {
        PublisherSummary {
            id: p.id,
            display_name: p.display_name,
            hierarchy_level: p.hierarchy_level,
            country_codes: p.country_codes,
            works_count: p.works_count,
            cited_by_count: p.cited_by_count,
        }
    }
}

// ── FunderSummary ─────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct FunderSummary {
    pub id: String,
    pub display_name: Option<String>,
    pub country_code: Option<String>,
    pub description: Option<String>,
    pub awards_count: Option<i64>,
    pub works_count: Option<i64>,
    pub cited_by_count: Option<i64>,
}

impl From<Funder> for FunderSummary {
    fn from(f: Funder) -> Self {
        FunderSummary {
            id: f.id,
            display_name: f.display_name,
            country_code: f.country_code,
            description: f.description,
            awards_count: f.awards_count,
            works_count: f.works_count,
            cited_by_count: f.cited_by_count,
        }
    }
}
