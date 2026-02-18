use papers_core::api;
use papers_core::text;
use papers_core::{
    AuthorListParams, DomainListParams, FieldListParams, FindWorksParams, FunderListParams,
    GetParams, InstitutionListParams, OpenAlexClient, PublisherListParams, SourceListParams,
    SubfieldListParams, TopicListParams, WorkListParams,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_client(mock: &MockServer) -> OpenAlexClient {
    OpenAlexClient::new().with_base_url(mock.uri())
}

// ── Fixture helpers ───────────────────────────────────────────────────────

fn list_response(results_json: &str) -> String {
    format!(
        r#"{{"meta": {{"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null}}, "results": [{results_json}], "group_by": []}}"#
    )
}

fn work_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/W1",
        "display_name": "A Great Paper",
        "doi": "https://doi.org/10.1234/test",
        "publication_year": 2020,
        "type": "article",
        "cited_by_count": 42,
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "Alice"}, "author_position": "first"}],
        "primary_location": {"source": {"id": "https://openalex.org/S1", "display_name": "Nature"}, "is_oa": true, "version": null, "license": null, "landing_page_url": null, "pdf_url": null},
        "open_access": {"is_oa": true, "oa_status": "gold", "oa_url": "https://example.com/oa", "any_repository_has_fulltext": false},
        "primary_topic": {"id": "https://openalex.org/T1", "display_name": "Machine Learning", "score": 0.9, "subfield": null, "field": null, "domain": null},
        "referenced_works": ["https://openalex.org/W2"],
        "counts_by_year": [{"year": 2020, "cited_by_count": 10}]
    }"#
}

fn author_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/A1",
        "display_name": "Alice Smith",
        "orcid": "https://orcid.org/0000-0001-2345-6789",
        "works_count": 100,
        "cited_by_count": 5000,
        "summary_stats": {"h_index": 30, "i10_index": 50, "cited_by_count": 5000, "2yr_mean_citedness": 2.5, "oa_percent": 60.0, "works_count": 100},
        "last_known_institutions": [{"id": "https://openalex.org/I1", "display_name": "MIT", "ror": null, "country_code": null, "type": null}],
        "topics": [
            {"id": "https://openalex.org/T1", "display_name": "Machine Learning", "count": 20, "subfield": null, "field": null, "domain": null},
            {"id": "https://openalex.org/T2", "display_name": "Deep Learning", "count": 15, "subfield": null, "field": null, "domain": null},
            {"id": "https://openalex.org/T3", "display_name": "Computer Vision", "count": 10, "subfield": null, "field": null, "domain": null},
            {"id": "https://openalex.org/T4", "display_name": "NLP", "count": 5, "subfield": null, "field": null, "domain": null}
        ],
        "affiliations": [{"institution": {"id": "https://openalex.org/I1", "display_name": "MIT", "ror": null, "country_code": null, "type": null}, "years": [2020, 2021]}],
        "counts_by_year": [{"year": 2020, "cited_by_count": 100, "works_count": 5}]
    }"#
}

fn source_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/S1",
        "display_name": "Nature",
        "issn_l": "0028-0836",
        "type": "journal",
        "is_oa": false,
        "is_in_doaj": false,
        "works_count": 50000,
        "cited_by_count": 2000000,
        "summary_stats": {"h_index": 900, "i10_index": null, "cited_by_count": 2000000, "2yr_mean_citedness": 50.0, "oa_percent": 10.0, "works_count": 50000},
        "host_organization_name": "Springer Nature",
        "apc_prices": [{"price": 11690, "currency": "USD"}],
        "topics": []
    }"#
}

fn institution_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/I1",
        "display_name": "MIT",
        "ror": "https://ror.org/042nb2s44",
        "country_code": "US",
        "type": "education",
        "geo": {"city": "Cambridge", "region": "Massachusetts", "country_code": "US", "country": "United States", "latitude": 42.36, "longitude": -71.09},
        "works_count": 200000,
        "cited_by_count": 10000000,
        "summary_stats": {"h_index": 500, "i10_index": null, "cited_by_count": 10000000, "2yr_mean_citedness": 5.0, "oa_percent": 40.0, "works_count": 200000},
        "associated_institutions": [{"id": "https://openalex.org/I2", "display_name": "MIT Lincoln Lab", "ror": null, "country_code": null, "type": null, "relationship": "related"}]
    }"#
}

fn topic_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/T1",
        "display_name": "Machine Learning",
        "description": "Study of algorithms that improve through experience",
        "subfield": {"id": 1, "display_name": "Artificial Intelligence"},
        "field": {"id": 2, "display_name": "Computer Science"},
        "domain": {"id": 3, "display_name": "Physical Sciences"},
        "works_count": 500000,
        "cited_by_count": 20000000,
        "keywords": ["neural networks", "deep learning"],
        "siblings": [{"id": "https://openalex.org/T2", "display_name": "Deep Learning"}]
    }"#
}

fn publisher_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/P1",
        "display_name": "Elsevier",
        "hierarchy_level": 0,
        "country_codes": ["NL"],
        "works_count": 5000000,
        "cited_by_count": 100000000,
        "lineage": ["https://openalex.org/P1"],
        "alternate_titles": ["Elsevier BV"],
        "parent_publisher": null
    }"#
}

fn funder_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/F1",
        "display_name": "NIH",
        "country_code": "US",
        "description": "US federal agency for biomedical research",
        "awards_count": 100000,
        "works_count": 500000,
        "cited_by_count": 20000000,
        "alternate_titles": ["National Institutes of Health"],
        "counts_by_year": [{"year": 2020, "cited_by_count": 1000000, "works_count": 50000}]
    }"#
}

fn domain_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/domains/3",
        "display_name": "Physical Sciences",
        "description": "branch of natural science that studies non-living systems",
        "fields": [
            {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
            {"id": "https://openalex.org/fields/22", "display_name": "Engineering"}
        ],
        "siblings": [{"id": "https://openalex.org/domains/1", "display_name": "Life Sciences"}],
        "display_name_alternatives": [],
        "works_count": 134263529,
        "cited_by_count": 1500000000,
        "works_api_url": "https://api.openalex.org/works?filter=primary_topic.domain.id:domains/3"
    }"#
}

fn field_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/fields/17",
        "display_name": "Computer Science",
        "description": "study of computation and information",
        "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
        "subfields": [
            {"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"},
            {"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory and Mathematics"}
        ],
        "siblings": [{"id": "https://openalex.org/fields/22", "display_name": "Engineering"}],
        "display_name_alternatives": [],
        "works_count": 22038624,
        "cited_by_count": 500000000,
        "works_api_url": "https://api.openalex.org/works?filter=primary_topic.field.id:fields/17"
    }"#
}

fn subfield_json() -> &'static str {
    r#"{
        "id": "https://openalex.org/subfields/1702",
        "display_name": "Artificial Intelligence",
        "description": "study of intelligent agents",
        "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
        "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
        "topics": [
            {"id": "https://openalex.org/T10028", "display_name": "Topic Modeling"},
            {"id": "https://openalex.org/T10029", "display_name": "Neural Architecture Search"}
        ],
        "siblings": [{"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory"}],
        "display_name_alternatives": [],
        "works_count": 9059921,
        "cited_by_count": 200000000,
        "works_api_url": "https://api.openalex.org/works?filter=primary_topic.subfield.id:subfields/1702"
    }"#
}

fn autocomplete_response() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10},
        "results": [{"id": "https://openalex.org/W1", "short_id": "works/W1", "display_name": "A Great Paper", "hint": "Alice Smith", "cited_by_count": 42, "works_count": null, "entity_type": "work", "external_id": null, "filter_key": "openalex"}]
    }"#.to_string()
}

fn find_response() -> String {
    r#"{"meta": null, "results": [{"id": "https://openalex.org/W1", "display_name": "A Great Paper", "score": 0.95}]}"#.to_string()
}

// ── List tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_work_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(work_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::work_list(&client, &WorkListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();

    // Slim fields present
    assert!(json.contains("\"id\""));
    assert!(json.contains("A Great Paper"));
    assert!(json.contains("\"cited_by_count\""));
    // Dropped fields absent
    assert!(!json.contains("referenced_works"));
    assert!(!json.contains("counts_by_year"));
    assert!(!json.contains("authorships"));
}

#[tokio::test]
async fn test_work_list_abstract_preserved() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(work_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::work_list(&client, &WorkListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("abstract_text"));
    assert!(json.contains("Hello world"));
}

#[tokio::test]
async fn test_work_list_authors_extracted() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(work_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::work_list(&client, &WorkListParams::default()).await.unwrap();
    assert_eq!(result.results[0].authors, vec!["Alice"]);
}

#[tokio::test]
async fn test_work_list_drops_group_by() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(work_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::work_list(&client, &WorkListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("group_by"));
}

#[tokio::test]
async fn test_work_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/W1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let work = api::work_get(&client, "W1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&work).unwrap();
    assert!(json.contains("referenced_works"));
    assert!(json.contains("counts_by_year"));
}

#[tokio::test]
async fn test_author_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(author_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::author_list(&client, &AuthorListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.h_index, Some(30));
    assert_eq!(s.last_known_institutions, vec!["MIT"]);
    assert_eq!(s.top_topics.len(), 3); // max 3 topics
    assert!(!s.top_topics.contains(&"NLP".to_string())); // 4th topic dropped

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("affiliations"));
    assert!(!json.contains("counts_by_year"));
}

#[tokio::test]
async fn test_author_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors/A1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(author_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let author = api::author_get(&client, "A1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&author).unwrap();
    assert!(json.contains("affiliations"));
    assert!(json.contains("counts_by_year"));
}

#[tokio::test]
async fn test_source_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(source_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::source_list(&client, &SourceListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.h_index, Some(900));
    assert_eq!(s.host_organization_name.as_deref(), Some("Springer Nature"));

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("apc_prices"));
    assert!(!json.contains("topics"));
}

#[tokio::test]
async fn test_source_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources/S1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(source_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let source = api::source_get(&client, "S1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&source).unwrap();
    assert!(json.contains("apc_prices"));
}

#[tokio::test]
async fn test_institution_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/institutions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(institution_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::institution_list(&client, &InstitutionListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.city.as_deref(), Some("Cambridge"));
    assert_eq!(s.h_index, Some(500));

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("associated_institutions"));
}

#[tokio::test]
async fn test_institution_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/institutions/I1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(institution_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let inst = api::institution_get(&client, "I1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&inst).unwrap();
    assert!(json.contains("associated_institutions"));
}

#[tokio::test]
async fn test_topic_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/topics"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(topic_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::topic_list(&client, &TopicListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.subfield.as_deref(), Some("Artificial Intelligence"));
    assert_eq!(s.field.as_deref(), Some("Computer Science"));
    assert_eq!(s.domain.as_deref(), Some("Physical Sciences"));

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("siblings"));
    assert!(!json.contains("keywords"));
}

#[tokio::test]
async fn test_topic_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/topics/T1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(topic_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let topic = api::topic_get(&client, "T1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&topic).unwrap();
    assert!(json.contains("siblings"));
    assert!(json.contains("keywords"));
}

#[tokio::test]
async fn test_publisher_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/publishers"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(publisher_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::publisher_list(&client, &PublisherListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.hierarchy_level, Some(0));
    assert_eq!(s.country_codes.as_deref(), Some(["NL".to_string()].as_slice()));

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("lineage"));
    assert!(!json.contains("alternate_titles"));
}

#[tokio::test]
async fn test_publisher_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/publishers/P1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(publisher_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let pub_ = api::publisher_get(&client, "P1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&pub_).unwrap();
    assert!(json.contains("lineage"));
    assert!(json.contains("alternate_titles"));
}

#[tokio::test]
async fn test_funder_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/funders"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(funder_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::funder_list(&client, &FunderListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.awards_count, Some(100000));
    assert_eq!(s.description.as_deref(), Some("US federal agency for biomedical research"));

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("alternate_titles"));
    assert!(!json.contains("counts_by_year"));
}

#[tokio::test]
async fn test_funder_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/funders/F1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(funder_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let funder = api::funder_get(&client, "F1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&funder).unwrap();
    assert!(json.contains("counts_by_year"));
    assert!(json.contains("alternate_titles"));
}

#[tokio::test]
async fn test_work_autocomplete_passthrough() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/autocomplete/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(autocomplete_response()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::work_autocomplete(&client, "great").await.unwrap();
    assert_eq!(result.results.len(), 1);
    assert_eq!(result.results[0].display_name, "A Great Paper");
}

#[tokio::test]
async fn test_work_find_uses_get_for_short_query() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/find/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(find_response()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = FindWorksParams::builder().query("machine learning").build();
    let result = api::work_find(&client, &params).await.unwrap();
    assert_eq!(result.results.len(), 1);
}

#[tokio::test]
async fn test_work_find_uses_post_for_long_query() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/find/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(find_response()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let long_query = "x".repeat(2049);
    let params = FindWorksParams::builder().query(long_query).build();
    let result = api::work_find(&client, &params).await.unwrap();
    assert_eq!(result.results.len(), 1);
}

#[tokio::test]
async fn test_domain_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/domains"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(domain_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::domain_list(&client, &DomainListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.fields, vec!["Computer Science", "Engineering"]);

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("siblings"));
    assert!(!json.contains("display_name_alternatives"));
    assert!(!json.contains("works_api_url"));
}

#[tokio::test]
async fn test_domain_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/domains/3"))
        .respond_with(ResponseTemplate::new(200).set_body_string(domain_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let domain = api::domain_get(&client, "3", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&domain).unwrap();
    assert!(json.contains("siblings"));
    assert!(json.contains("works_api_url"));
}

#[tokio::test]
async fn test_field_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/fields"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(field_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::field_list(&client, &FieldListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.domain.as_deref(), Some("Physical Sciences"));
    assert_eq!(s.subfield_count, 2);

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("siblings"));
    assert!(!json.contains("display_name_alternatives"));
    assert!(!json.contains("subfields"));
}

#[tokio::test]
async fn test_field_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/fields/17"))
        .respond_with(ResponseTemplate::new(200).set_body_string(field_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let field = api::field_get(&client, "17", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&field).unwrap();
    assert!(json.contains("siblings"));
    assert!(json.contains("subfields"));
}

#[tokio::test]
async fn test_subfield_list_applies_summary() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/subfields"))
        .respond_with(ResponseTemplate::new(200).set_body_string(list_response(subfield_json())))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::subfield_list(&client, &SubfieldListParams::default()).await.unwrap();
    let s = &result.results[0];

    assert_eq!(s.field.as_deref(), Some("Computer Science"));
    assert_eq!(s.domain.as_deref(), Some("Physical Sciences"));

    let json = serde_json::to_string(&result).unwrap();
    assert!(!json.contains("siblings"));
    assert!(!json.contains("topics"));
    assert!(!json.contains("display_name_alternatives"));
}

#[tokio::test]
async fn test_subfield_get_returns_full() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/subfields/1702"))
        .respond_with(ResponseTemplate::new(200).set_body_string(subfield_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let subfield = api::subfield_get(&client, "1702", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&subfield).unwrap();
    assert!(json.contains("siblings"));
    assert!(json.contains("topics"));
}

#[tokio::test]
async fn test_api_error_propagated() {
    let mock = MockServer::start().await;
    // W99999 is a valid OpenAlex ID format — passes through directly, hits /works/W99999
    Mock::given(method("GET"))
        .and(path("/works/W99999"))
        .respond_with(ResponseTemplate::new(404).set_body_string(r#"{"error": "not found"}"#))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::work_get(&client, "W99999", &GetParams::default()).await;
    assert!(result.is_err());
}

// ── Smart ID resolution: search-then-get flow ─────────────────────────────

fn slim_list_response(id: &str, name: &str) -> String {
    format!(
        r#"{{"meta": {{"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null}}, "results": [{{"id": "{id}", "display_name": "{name}"}}], "group_by": []}}"#
    )
}

#[tokio::test]
async fn test_work_get_resolves_search_query() {
    let mock = MockServer::start().await;
    // First call: search for work by name
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            slim_list_response("https://openalex.org/W1", "A Great Paper")
        ))
        .mount(&mock)
        .await;
    // Second call: fetch the resolved ID
    Mock::given(method("GET"))
        .and(path("/works/W1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let work = api::work_get(&client, "A Great Paper", &GetParams::default()).await.unwrap();
    assert_eq!(work.display_name, Some("A Great Paper".to_string()));
}

#[tokio::test]
async fn test_author_get_resolves_search_query() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            slim_list_response("https://openalex.org/A1", "Alice Smith")
        ))
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/authors/A1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(author_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let author = api::author_get(&client, "Alice Smith", &GetParams::default()).await.unwrap();
    assert_eq!(author.display_name, Some("Alice Smith".to_string()));
}

#[tokio::test]
async fn test_source_get_resolves_search_query() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            slim_list_response("https://openalex.org/S1", "Nature")
        ))
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/sources/S1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(source_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let source = api::source_get(&client, "Nature", &GetParams::default()).await.unwrap();
    assert_eq!(source.display_name, Some("Nature".to_string()));
}

#[tokio::test]
async fn test_work_get_by_doi_passes_through() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/https://doi.org/10.1109/test"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_json()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let work = api::work_get(&client, "https://doi.org/10.1109/test", &GetParams::default()).await.unwrap();
    assert_eq!(work.display_name, Some("A Great Paper".to_string()));
}

#[tokio::test]
async fn test_work_get_not_found_error() {
    let mock = MockServer::start().await;
    // Return empty results for the search
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{"meta": {"count": 0, "db_response_time_ms": 5, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null}, "results": [], "group_by": []}"#
        ))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = api::work_get(&client, "nonexistent paper title xyz", &GetParams::default()).await;
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("nonexistent paper title xyz"));
}

// ── work_text tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_work_text_no_pdf_found() {
    let mock = MockServer::start().await;
    // Work has no pdf_url in any location and no has_content.pdf
    Mock::given(method("GET"))
        .and(path("/works/W1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{
                "id": "https://openalex.org/W1",
                "display_name": "A Paper Without PDF",
                "doi": null,
                "primary_location": null,
                "locations": [],
                "best_oa_location": null,
                "has_content": null
            }"#,
        ))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = text::work_text(&client, None, "W1").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    match err {
        text::WorkTextError::NoPdfFound { ref work_id, ref title, .. } => {
            assert_eq!(work_id, "https://openalex.org/W1");
            assert_eq!(title.as_deref(), Some("A Paper Without PDF"));
        }
        other => panic!("Expected NoPdfFound, got: {other}"),
    }
    assert!(err.to_string().contains("A Paper Without PDF"));
}

#[tokio::test]
async fn test_work_text_no_pdf_found_with_non_whitelisted_url() {
    let mock = MockServer::start().await;
    // Work has a pdf_url but on a non-whitelisted domain
    Mock::given(method("GET"))
        .and(path("/works/W2"))
        .respond_with(ResponseTemplate::new(200).set_body_string(
            r#"{
                "id": "https://openalex.org/W2",
                "display_name": "Publisher Paper",
                "doi": null,
                "primary_location": {"pdf_url": "https://publisher.com/paper.pdf"},
                "locations": [],
                "best_oa_location": null,
                "has_content": null
            }"#,
        ))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = text::work_text(&client, None, "W2").await;
    assert!(result.is_err());
    match result.unwrap_err() {
        text::WorkTextError::NoPdfFound { .. } => {}
        other => panic!("Expected NoPdfFound, got: {other}"),
    }
}
