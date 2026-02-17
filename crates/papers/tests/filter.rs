use papers::filter::{WorkFilterAliases, resolve_work_filters};
use papers::{
    AuthorListParams, FieldListParams, FunderListParams, InstitutionListParams, OpenAlexClient,
    PublisherListParams, SourceListParams, SubfieldListParams, TopicListParams,
};
use papers::api;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_client(mock: &MockServer) -> OpenAlexClient {
    OpenAlexClient::new().with_base_url(mock.uri())
}

/// Minimal list response with one result containing the given ID.
fn search_result_json(id: &str) -> String {
    format!(
        r#"{{
            "meta": {{"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null}},
            "results": [{{"id": "{id}", "display_name": "Test Entity"}}],
            "group_by": []
        }}"#
    )
}

/// Empty list response (no results).
fn empty_search_result_json() -> String {
    r#"{
        "meta": {"count": 0, "db_response_time_ms": 5, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null},
        "results": [],
        "group_by": []
    }"#
    .to_string()
}

#[tokio::test]
async fn test_resolve_author_search() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors"))
        .and(query_param("filter", "display_name.search:einstein"))
        .and(query_param("sort", "cited_by_count:desc"))
        .and(query_param("per-page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/A5083138872")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        author: Some("einstein".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("authorships.author.id:A5083138872")
    );
}

#[tokio::test]
async fn test_resolve_publisher_search() {
    let mock = MockServer::start().await;
    // Publishers use `search` param, NOT `filter=display_name.search:`
    Mock::given(method("GET"))
        .and(path("/publishers"))
        .and(query_param("search", "acm"))
        .and(query_param("sort", "cited_by_count:desc"))
        .and(query_param("per-page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/P4310319798")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        publisher: Some("acm".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("primary_location.source.publisher_lineage:P4310319798")
    );
}

#[tokio::test]
async fn test_resolve_source_search() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources"))
        .and(query_param("filter", "display_name.search:siggraph"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/S131921510")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        source: Some("siggraph".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("primary_location.source.id:S131921510")
    );
}

#[tokio::test]
async fn test_resolve_topic_search() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/topics"))
        .and(query_param("filter", "display_name.search:machine learning"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/T11636")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        topic: Some("machine learning".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(result.as_deref(), Some("primary_topic.id:T11636"));
}

#[tokio::test]
async fn test_resolve_domain_search() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/domains"))
        .and(query_param("filter", "display_name.search:physical"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/domains/3")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        domain: Some("physical".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("primary_topic.domain.id:domains/3")
    );
}

#[tokio::test]
async fn test_resolve_field_search() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/fields"))
        .and(query_param(
            "filter",
            "display_name.search:computer science",
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/fields/17")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        field: Some("computer science".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("primary_topic.field.id:fields/17")
    );
}

#[tokio::test]
async fn test_resolve_subfield_search() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/subfields"))
        .and(query_param(
            "filter",
            "display_name.search:artificial intelligence",
        ))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/subfields/1702")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        subfield: Some("artificial intelligence".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("primary_topic.subfield.id:subfields/1702")
    );
}

#[tokio::test]
async fn test_resolve_mixed_id_and_search() {
    let mock = MockServer::start().await;
    // "acm" should trigger a search, "P4310320595" should pass through
    Mock::given(method("GET"))
        .and(path("/publishers"))
        .and(query_param("search", "acm"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/P4310319798")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        publisher: Some("acm|P4310320595".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None)
        .await
        .unwrap();
    assert_eq!(
        result.as_deref(),
        Some("primary_location.source.publisher_lineage:P4310319798|P4310320595")
    );
}

#[tokio::test]
async fn test_resolve_not_found_error() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors"))
        .respond_with(ResponseTemplate::new(200).set_body_string(empty_search_result_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        author: Some("nonexistent_person_xyz".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, None).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("nonexistent_person_xyz"));
    assert!(err.contains("authors"));
}

/// Empty list response body for mocking entity list endpoints.
fn empty_list_response_json() -> &'static str {
    r#"{"meta": {"count": 0, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null}, "results": [], "group_by": []}"#
}

#[tokio::test]
async fn test_resolve_overlap_error() {
    let mock = MockServer::start().await;
    // No mocks needed — overlap is checked before any API calls
    let client = make_client(&mock);
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
async fn test_resolve_all_aliases_combined() {
    let mock = MockServer::start().await;

    // Mock author search
    Mock::given(method("GET"))
        .and(path("/authors"))
        .and(query_param("filter", "display_name.search:einstein"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/A5083138872")),
        )
        .mount(&mock)
        .await;

    // Mock source search
    Mock::given(method("GET"))
        .and(path("/sources"))
        .and(query_param("filter", "display_name.search:nature"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/S123456")),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let aliases = WorkFilterAliases {
        author: Some("einstein".to_string()),
        topic: Some("T11636".to_string()),
        domain: Some("3".to_string()),
        source: Some("nature".to_string()),
        year: Some(">2020".to_string()),
        citations: Some(">100".to_string()),
        ..Default::default()
    };
    let result = resolve_work_filters(&client, &aliases, Some("is_oa:true"))
        .await
        .unwrap();
    let filter = result.unwrap();

    assert!(filter.contains("authorships.author.id:A5083138872"));
    assert!(filter.contains("primary_topic.id:T11636"));
    assert!(filter.contains("primary_topic.domain.id:domains/3"));
    assert!(filter.contains("primary_location.source.id:S123456"));
    assert!(filter.contains("publication_year:>2020"));
    assert!(filter.contains("cited_by_count:>100"));
    assert!(filter.contains("is_oa:true"));
}

// ── Entity-specific filter alias integration tests (via api::*_list) ────

#[tokio::test]
async fn test_author_list_institution_search() {
    let mock = MockServer::start().await;

    // Mock institution search: resolve "harvard" → I123
    Mock::given(method("GET"))
        .and(path("/institutions"))
        .and(query_param("filter", "display_name.search:harvard"))
        .and(query_param("sort", "cited_by_count:desc"))
        .and(query_param("per-page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/I123")),
        )
        .mount(&mock)
        .await;

    // Mock author list with resolved filter
    Mock::given(method("GET"))
        .and(path("/authors"))
        .and(query_param("filter", "last_known_institutions.id:I123"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = AuthorListParams {
        institution: Some("harvard".into()),
        ..Default::default()
    };
    let result = api::author_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_author_list_country_direct() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/authors"))
        .and(query_param(
            "filter",
            "last_known_institutions.country_code:US",
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = AuthorListParams {
        country: Some("US".into()),
        ..Default::default()
    };
    let result = api::author_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_author_list_citations_direct() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/authors"))
        .and(query_param("filter", "cited_by_count:>1000"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = AuthorListParams {
        citations: Some(">1000".into()),
        ..Default::default()
    };
    let result = api::author_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_source_list_publisher_search() {
    let mock = MockServer::start().await;

    // Mock publisher search: resolve "springer" → P123 (uses `search` param)
    Mock::given(method("GET"))
        .and(path("/publishers"))
        .and(query_param("search", "springer"))
        .and(query_param("sort", "cited_by_count:desc"))
        .and(query_param("per-page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/P123")),
        )
        .mount(&mock)
        .await;

    // Mock source list with resolved filter
    Mock::given(method("GET"))
        .and(path("/sources"))
        .and(query_param("filter", "host_organization_lineage:P123"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = SourceListParams {
        publisher: Some("springer".into()),
        ..Default::default()
    };
    let result = api::source_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_source_list_open_flag() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/sources"))
        .and(query_param("filter", "is_oa:true"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = SourceListParams {
        open: Some(true),
        ..Default::default()
    };
    let result = api::source_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_source_list_type_direct() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/sources"))
        .and(query_param("filter", "type:journal"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = SourceListParams {
        r#type: Some("journal".into()),
        ..Default::default()
    };
    let result = api::source_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_institution_list_country_type() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/institutions"))
        .and(query_param("filter", "country_code:US,type:education"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = InstitutionListParams {
        country: Some("US".into()),
        r#type: Some("education".into()),
        ..Default::default()
    };
    let result = api::institution_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_topic_list_domain_field() {
    let mock = MockServer::start().await;

    // Mock domain search: resolve "physical" → domains/3
    Mock::given(method("GET"))
        .and(path("/domains"))
        .and(query_param("filter", "display_name.search:physical"))
        .and(query_param("sort", "cited_by_count:desc"))
        .and(query_param("per-page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/domains/3")),
        )
        .mount(&mock)
        .await;

    // Mock field search: resolve "computer science" → fields/17
    Mock::given(method("GET"))
        .and(path("/fields"))
        .and(query_param(
            "filter",
            "display_name.search:computer science",
        ))
        .and(query_param("sort", "cited_by_count:desc"))
        .and(query_param("per-page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/fields/17")),
        )
        .mount(&mock)
        .await;

    // Mock topic list with both resolved filters
    Mock::given(method("GET"))
        .and(path("/topics"))
        .and(query_param(
            "filter",
            "domain.id:domains/3,field.id:fields/17",
        ))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = TopicListParams {
        domain: Some("physical".into()),
        field: Some("computer science".into()),
        ..Default::default()
    };
    let result = api::topic_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_field_list_domain_search() {
    let mock = MockServer::start().await;

    // Mock domain search: resolve "life sciences" → domains/1
    Mock::given(method("GET"))
        .and(path("/domains"))
        .and(query_param("filter", "display_name.search:life sciences"))
        .and(query_param("sort", "cited_by_count:desc"))
        .and(query_param("per-page", "1"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/domains/1")),
        )
        .mount(&mock)
        .await;

    // Mock field list with resolved filter
    Mock::given(method("GET"))
        .and(path("/fields"))
        .and(query_param("filter", "domain.id:domains/1"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = FieldListParams {
        domain: Some("life sciences".into()),
        ..Default::default()
    };
    let result = api::field_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_subfield_list_field_id() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/subfields"))
        .and(query_param("filter", "field.id:fields/17"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = SubfieldListParams {
        field: Some("17".into()),
        ..Default::default()
    };
    let result = api::subfield_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_publisher_list_country_direct() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/publishers"))
        .and(query_param("filter", "country_codes:US"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = PublisherListParams {
        country: Some("US".into()),
        ..Default::default()
    };
    let result = api::publisher_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_funder_list_works_direct() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/funders"))
        .and(query_param("filter", "works_count:>100000"))
        .respond_with(
            ResponseTemplate::new(200).set_body_string(empty_list_response_json()),
        )
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = FunderListParams {
        works: Some(">100000".into()),
        ..Default::default()
    };
    let result = api::funder_list(&client, &params).await.unwrap();
    assert_eq!(result.meta.count, 0);
}

#[tokio::test]
async fn test_author_list_overlap_error() {
    // No mock server needed — overlap is checked before any API calls
    let mock = MockServer::start().await;
    let client = make_client(&mock);

    let params = AuthorListParams {
        country: Some("US".into()),
        filter: Some("last_known_institutions.country_code:GB".into()),
        ..Default::default()
    };
    let result = api::author_list(&client, &params).await;
    assert!(result.is_err());
    let err = match result {
        Err(e) => e.to_string(),
        Ok(_) => panic!("expected error but got Ok"),
    };
    assert!(err.contains("country"));
    assert!(err.contains("last_known_institutions.country_code"));
}
