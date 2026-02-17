use papers_openalex::{
    Author, AutocompleteResponse, Domain, Field, FindWorksParams, FindWorksResponse, Funder,
    GetParams, Institution, ListParams, OpenAlexClient, OpenAlexError, Publisher, Source, Subfield,
    Topic, Work,
};

use crate::summary::{
    AuthorSummary, DomainSummary, FieldSummary, FunderSummary, InstitutionSummary,
    PublisherSummary, SlimListResponse, SourceSummary, SubfieldSummary, TopicSummary, WorkSummary,
    summary_list_result,
};

// ── List ─────────────────────────────────────────────────────────────────

pub async fn work_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<WorkSummary>, OpenAlexError> {
    summary_list_result(client.list_works(params).await, WorkSummary::from)
}

pub async fn author_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<AuthorSummary>, OpenAlexError> {
    summary_list_result(client.list_authors(params).await, AuthorSummary::from)
}

pub async fn source_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<SourceSummary>, OpenAlexError> {
    summary_list_result(client.list_sources(params).await, SourceSummary::from)
}

pub async fn institution_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<InstitutionSummary>, OpenAlexError> {
    summary_list_result(client.list_institutions(params).await, InstitutionSummary::from)
}

pub async fn topic_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<TopicSummary>, OpenAlexError> {
    summary_list_result(client.list_topics(params).await, TopicSummary::from)
}

pub async fn publisher_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<PublisherSummary>, OpenAlexError> {
    summary_list_result(client.list_publishers(params).await, PublisherSummary::from)
}

pub async fn funder_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<FunderSummary>, OpenAlexError> {
    summary_list_result(client.list_funders(params).await, FunderSummary::from)
}

pub async fn domain_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<DomainSummary>, OpenAlexError> {
    summary_list_result(client.list_domains(params).await, DomainSummary::from)
}

pub async fn field_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<FieldSummary>, OpenAlexError> {
    summary_list_result(client.list_fields(params).await, FieldSummary::from)
}

pub async fn subfield_list(
    client: &OpenAlexClient,
    params: &ListParams,
) -> Result<SlimListResponse<SubfieldSummary>, OpenAlexError> {
    summary_list_result(client.list_subfields(params).await, SubfieldSummary::from)
}

// ── Get ──────────────────────────────────────────────────────────────────

pub async fn work_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Work, OpenAlexError> {
    client.get_work(id, params).await
}

pub async fn author_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Author, OpenAlexError> {
    client.get_author(id, params).await
}

pub async fn source_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Source, OpenAlexError> {
    client.get_source(id, params).await
}

pub async fn institution_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Institution, OpenAlexError> {
    client.get_institution(id, params).await
}

pub async fn topic_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Topic, OpenAlexError> {
    client.get_topic(id, params).await
}

pub async fn publisher_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Publisher, OpenAlexError> {
    client.get_publisher(id, params).await
}

pub async fn funder_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Funder, OpenAlexError> {
    client.get_funder(id, params).await
}

pub async fn domain_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Domain, OpenAlexError> {
    client.get_domain(id, params).await
}

pub async fn field_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Field, OpenAlexError> {
    client.get_field(id, params).await
}

pub async fn subfield_get(
    client: &OpenAlexClient,
    id: &str,
    params: &GetParams,
) -> Result<Subfield, OpenAlexError> {
    client.get_subfield(id, params).await
}

// ── Autocomplete ─────────────────────────────────────────────────────────

pub async fn work_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_works(q).await
}

pub async fn author_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_authors(q).await
}

pub async fn source_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_sources(q).await
}

pub async fn institution_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_institutions(q).await
}

pub async fn concept_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_concepts(q).await
}

pub async fn publisher_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_publishers(q).await
}

pub async fn funder_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_funders(q).await
}

pub async fn subfield_autocomplete(
    client: &OpenAlexClient,
    q: &str,
) -> Result<AutocompleteResponse, OpenAlexError> {
    client.autocomplete_subfields(q).await
}

// ── Find ─────────────────────────────────────────────────────────────────

/// AI semantic search for works by conceptual similarity.
/// Automatically uses POST for queries longer than 2048 characters.
pub async fn work_find(
    client: &OpenAlexClient,
    params: &FindWorksParams,
) -> Result<FindWorksResponse, OpenAlexError> {
    if params.query.len() > 2048 {
        client.find_works_post(params).await
    } else {
        client.find_works(params).await
    }
}
