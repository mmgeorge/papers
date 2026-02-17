use papers_openalex::{
    Author, AutocompleteResponse, Domain, Field, FindWorksParams, FindWorksResponse, Funder,
    GetParams, Institution, OpenAlexClient, OpenAlexError, Publisher, Source, Subfield,
    Topic, Work,
};

use crate::filter::{
    AuthorListParams, DomainListParams, FieldListParams, FilterError, FunderListParams,
    InstitutionListParams, PublisherListParams, SourceListParams, SubfieldListParams,
    TopicListParams, WorkListParams, resolve_filters, WORK_ALIASES,
};
use crate::summary::{
    AuthorSummary, DomainSummary, FieldSummary, FunderSummary, InstitutionSummary,
    PublisherSummary, SlimListResponse, SourceSummary, SubfieldSummary, TopicSummary, WorkSummary,
    summary_list_result,
};

// ── List ─────────────────────────────────────────────────────────────────

pub async fn work_list(
    client: &OpenAlexClient,
    params: &WorkListParams,
) -> Result<SlimListResponse<WorkSummary>, FilterError> {
    let (alias_values, mut list_params) = params.into_aliases_and_list_params();
    list_params.filter = resolve_filters(client, WORK_ALIASES, &alias_values, list_params.filter.as_deref()).await?;
    Ok(summary_list_result(client.list_works(&list_params).await, WorkSummary::from)?)
}

macro_rules! entity_list_fn {
    ($fn_name:ident, $params_type:ident, $summary_type:ident, $client_method:ident) => {
        pub async fn $fn_name(
            client: &OpenAlexClient,
            params: &$params_type,
        ) -> Result<SlimListResponse<$summary_type>, FilterError> {
            let (alias_values, mut list_params) = params.into_aliases_and_list_params();
            list_params.filter = resolve_filters(
                client,
                $params_type::alias_specs(),
                &alias_values,
                list_params.filter.as_deref(),
            ).await?;
            Ok(summary_list_result(client.$client_method(&list_params).await, $summary_type::from)?)
        }
    };
}

entity_list_fn!(author_list, AuthorListParams, AuthorSummary, list_authors);
entity_list_fn!(source_list, SourceListParams, SourceSummary, list_sources);
entity_list_fn!(institution_list, InstitutionListParams, InstitutionSummary, list_institutions);
entity_list_fn!(topic_list, TopicListParams, TopicSummary, list_topics);
entity_list_fn!(publisher_list, PublisherListParams, PublisherSummary, list_publishers);
entity_list_fn!(funder_list, FunderListParams, FunderSummary, list_funders);
entity_list_fn!(domain_list, DomainListParams, DomainSummary, list_domains);
entity_list_fn!(field_list, FieldListParams, FieldSummary, list_fields);
entity_list_fn!(subfield_list, SubfieldListParams, SubfieldSummary, list_subfields);

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
