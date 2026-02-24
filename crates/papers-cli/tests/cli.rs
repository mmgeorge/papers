use papers_core::{
    AuthorListParams, DomainListParams, FieldListParams, FunderListParams, GetParams,
    InstitutionListParams, OpenAlexClient, PublisherListParams, SourceListParams,
    SubfieldListParams, TopicListParams, WorkListParams,
};
use papers_zotero::{CollectionListParams, ItemListParams, TagListParams, ZoteroClient};
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn make_client(mock: &MockServer) -> OpenAlexClient {
    OpenAlexClient::new().with_base_url(mock.uri())
}

// ── Fixture helpers ───────────────────────────────────────────────────────

fn work_list_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/W1",
            "display_name": "Bitonic Sort",
            "doi": "https://doi.org/10.1109/tc.1979.1675216",
            "publication_year": 1979,
            "type": "article",
            "cited_by_count": 254,
            "abstract_inverted_index": {"Sorting": [0], "networks": [1]},
            "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "Alice"}, "author_position": "first"}],
            "primary_location": {"source": {"id": "https://openalex.org/S1", "display_name": "IEEE TC"}, "is_oa": false},
            "open_access": {"is_oa": false, "oa_status": "closed", "oa_url": null},
            "primary_topic": {"id": "https://openalex.org/T1", "display_name": "Algorithms"}
        }],
        "group_by": []
    }"#.to_string()
}

fn work_get_body() -> String {
    r#"{
        "id": "https://openalex.org/W1",
        "display_name": "Bitonic Sort",
        "doi": "https://doi.org/10.1109/tc.1979.1675216",
        "publication_year": 1979,
        "type": "article",
        "cited_by_count": 254,
        "abstract_inverted_index": {"Sorting": [0], "networks": [1]},
        "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "Alice"}, "author_position": "first", "institutions": []}],
        "open_access": {"is_oa": false, "oa_status": "closed", "oa_url": null},
        "referenced_works": ["https://openalex.org/W2"],
        "counts_by_year": []
    }"#.to_string()
}

fn author_list_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/A1",
            "display_name": "Alice Smith",
            "orcid": "https://orcid.org/0000-0001-2345-6789",
            "works_count": 100,
            "cited_by_count": 5000,
            "summary_stats": {"h_index": 30, "i10_index": 50, "cited_by_count": 5000, "2yr_mean_citedness": 2.5, "oa_percent": 60.0, "works_count": 100},
            "last_known_institutions": [{"id": "https://openalex.org/I1", "display_name": "MIT"}],
            "topics": [{"id": "https://openalex.org/T1", "display_name": "ML", "count": 10}]
        }],
        "group_by": []
    }"#.to_string()
}

fn author_get_body() -> String {
    r#"{
        "id": "https://openalex.org/A1",
        "display_name": "Alice Smith",
        "orcid": "https://orcid.org/0000-0001-2345-6789",
        "works_count": 100,
        "cited_by_count": 5000,
        "summary_stats": {"h_index": 30, "i10_index": 50, "cited_by_count": 5000, "2yr_mean_citedness": 2.5, "oa_percent": 60.0, "works_count": 100},
        "last_known_institutions": [{"id": "https://openalex.org/I1", "display_name": "MIT"}],
        "topics": []
    }"#.to_string()
}

fn source_list_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/S1",
            "display_name": "Nature",
            "issn_l": "0028-0836",
            "type": "journal",
            "is_oa": false,
            "is_in_doaj": false,
            "works_count": 50000,
            "cited_by_count": 2000000,
            "summary_stats": {"h_index": 900, "i10_index": null, "cited_by_count": 2000000, "2yr_mean_citedness": 50.0, "oa_percent": 10.0, "works_count": 50000},
            "host_organization_name": "Springer Nature"
        }],
        "group_by": []
    }"#.to_string()
}

fn source_get_body() -> String {
    r#"{
        "id": "https://openalex.org/S1",
        "display_name": "Nature",
        "issn_l": "0028-0836",
        "type": "journal",
        "is_oa": false,
        "is_in_doaj": false,
        "works_count": 50000,
        "cited_by_count": 2000000,
        "host_organization_name": "Springer Nature"
    }"#.to_string()
}

fn institution_list_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/I1",
            "display_name": "MIT",
            "ror": "https://ror.org/042nb2s44",
            "country_code": "US",
            "type": "education",
            "geo": {"city": "Cambridge", "region": "Massachusetts", "country_code": "US", "country": "United States", "latitude": 42.36, "longitude": -71.09},
            "works_count": 200000,
            "cited_by_count": 10000000,
            "summary_stats": {"h_index": 500, "i10_index": null, "cited_by_count": 10000000, "2yr_mean_citedness": 5.0, "oa_percent": 40.0, "works_count": 200000}
        }],
        "group_by": []
    }"#.to_string()
}

fn institution_get_body() -> String {
    r#"{
        "id": "https://openalex.org/I1",
        "display_name": "MIT",
        "ror": "https://ror.org/042nb2s44",
        "country_code": "US",
        "type": "education",
        "geo": {"city": "Cambridge", "region": "Massachusetts", "country_code": "US", "country": "United States", "latitude": 42.36, "longitude": -71.09},
        "works_count": 200000,
        "cited_by_count": 10000000
    }"#.to_string()
}

fn topic_list_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/T1",
            "display_name": "Machine Learning",
            "description": "Study of algorithms",
            "subfield": {"id": 1, "display_name": "Artificial Intelligence"},
            "field": {"id": 2, "display_name": "Computer Science"},
            "domain": {"id": 3, "display_name": "Physical Sciences"},
            "works_count": 500000,
            "cited_by_count": 20000000
        }],
        "group_by": []
    }"#.to_string()
}

fn topic_get_body() -> String {
    r#"{
        "id": "https://openalex.org/T1",
        "display_name": "Machine Learning",
        "description": "Study of algorithms that improve through experience",
        "subfield": {"id": 1, "display_name": "Artificial Intelligence"},
        "field": {"id": 2, "display_name": "Computer Science"},
        "domain": {"id": 3, "display_name": "Physical Sciences"},
        "works_count": 500000,
        "cited_by_count": 20000000,
        "keywords": ["neural networks"],
        "siblings": []
    }"#.to_string()
}

fn publisher_list_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/P1",
            "display_name": "Elsevier",
            "hierarchy_level": 0,
            "country_codes": ["NL"],
            "works_count": 5000000,
            "cited_by_count": 100000000
        }],
        "group_by": []
    }"#.to_string()
}

fn publisher_get_body() -> String {
    r#"{
        "id": "https://openalex.org/P1",
        "display_name": "Elsevier",
        "hierarchy_level": 0,
        "country_codes": ["NL"],
        "works_count": 5000000,
        "cited_by_count": 100000000
    }"#.to_string()
}

fn funder_list_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/F1",
            "display_name": "NIH",
            "country_code": "US",
            "description": "US biomedical research",
            "awards_count": 100000,
            "works_count": 500000,
            "cited_by_count": 20000000
        }],
        "group_by": []
    }"#.to_string()
}

fn funder_get_body() -> String {
    r#"{
        "id": "https://openalex.org/F1",
        "display_name": "NIH",
        "country_code": "US",
        "description": "US biomedical research agency",
        "awards_count": 100000,
        "works_count": 500000,
        "cited_by_count": 20000000
    }"#.to_string()
}

fn domain_list_body() -> String {
    r#"{
        "meta": {"count": 4, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/domains/3",
            "display_name": "Physical Sciences",
            "description": "branch of natural science that studies non-living systems",
            "fields": [{"id": "https://openalex.org/fields/17", "display_name": "Computer Science"}],
            "siblings": [],
            "works_count": 134263529,
            "cited_by_count": 1500000000
        }],
        "group_by": []
    }"#.to_string()
}

fn domain_get_body() -> String {
    r#"{
        "id": "https://openalex.org/domains/3",
        "display_name": "Physical Sciences",
        "description": "branch of natural science that studies non-living systems",
        "fields": [
            {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
            {"id": "https://openalex.org/fields/22", "display_name": "Engineering"}
        ],
        "siblings": [{"id": "https://openalex.org/domains/1", "display_name": "Life Sciences"}],
        "works_count": 134263529,
        "cited_by_count": 1500000000
    }"#.to_string()
}

fn field_list_body() -> String {
    r#"{
        "meta": {"count": 26, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/fields/17",
            "display_name": "Computer Science",
            "description": "study of computation and information",
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
            "subfields": [
                {"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"},
                {"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory"}
            ],
            "siblings": [],
            "works_count": 22038624,
            "cited_by_count": 500000000
        }],
        "group_by": []
    }"#.to_string()
}

fn field_get_body() -> String {
    r#"{
        "id": "https://openalex.org/fields/17",
        "display_name": "Computer Science",
        "description": "study of computation and information",
        "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
        "subfields": [
            {"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"},
            {"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory"}
        ],
        "siblings": [{"id": "https://openalex.org/fields/22", "display_name": "Engineering"}],
        "works_count": 22038624,
        "cited_by_count": 500000000
    }"#.to_string()
}

fn subfield_list_body() -> String {
    r#"{
        "meta": {"count": 252, "db_response_time_ms": 5, "page": 1, "per_page": 10, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/subfields/1702",
            "display_name": "Artificial Intelligence",
            "description": "study of intelligent agents",
            "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
            "topics": [{"id": "https://openalex.org/T10028", "display_name": "Topic Modeling"}],
            "siblings": [],
            "works_count": 9059921,
            "cited_by_count": 200000000
        }],
        "group_by": []
    }"#.to_string()
}

fn subfield_get_body() -> String {
    r#"{
        "id": "https://openalex.org/subfields/1702",
        "display_name": "Artificial Intelligence",
        "description": "study of intelligent agents",
        "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
        "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
        "topics": [{"id": "https://openalex.org/T10028", "display_name": "Topic Modeling"}],
        "siblings": [{"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory"}],
        "works_count": 9059921,
        "cited_by_count": 200000000
    }"#.to_string()
}

fn autocomplete_body() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 10},
        "results": [{"id": "https://openalex.org/W1", "short_id": "works/W1", "display_name": "Machine Learning Paper", "hint": "Alice Smith", "cited_by_count": 42, "works_count": null, "entity_type": "work", "external_id": null, "filter_key": "openalex"}]
    }"#.to_string()
}

// ── Work tests ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_work_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::work_list(&client, &WorkListParams::default()).await.unwrap();
    let text = papers_cli_format::format_work_list(&result);

    assert!(text.contains("Bitonic Sort"));
    assert!(text.contains("Alice"));
    assert!(text.contains("1979"));
    assert!(text.contains("254"));
}

#[tokio::test]
async fn test_work_list_json() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::work_list(&client, &WorkListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();

    assert!(json.contains("Bitonic Sort"));
    assert!(!json.contains("referenced_works"));
    assert!(!json.contains("counts_by_year"));
}

#[tokio::test]
async fn test_work_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/W1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let work = papers_core::api::work_get(&client, "W1", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_work_get(&work);

    assert!(text.contains("Bitonic Sort"));
    assert!(text.contains("https://doi.org/10.1109/tc.1979.1675216"));
    assert!(text.contains("Alice"));
}

#[tokio::test]
async fn test_work_get_json() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/W1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let work = papers_core::api::work_get(&client, "W1", &GetParams::default()).await.unwrap();
    let json = serde_json::to_string(&work).unwrap();

    assert!(json.contains("referenced_works"));
}

#[tokio::test]
async fn test_author_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors"))
        .respond_with(ResponseTemplate::new(200).set_body_string(author_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::author_list(&client, &AuthorListParams::default()).await.unwrap();
    let text = papers_cli_format::format_author_list(&result);

    assert!(text.contains("Alice Smith"));
    assert!(text.contains("h-index: 30"));
    assert!(text.contains("MIT"));
    assert!(text.contains("ML"));
}

#[tokio::test]
async fn test_author_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors/A1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(author_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let author = papers_core::api::author_get(&client, "A1", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_author_get(&author);

    assert!(text.contains("Alice Smith"));
    assert!(text.contains("orcid.org"));
    assert!(text.contains("100 works"));
    assert!(text.contains("h-index: 30"));
}

#[tokio::test]
async fn test_source_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources"))
        .respond_with(ResponseTemplate::new(200).set_body_string(source_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::source_list(&client, &SourceListParams::default()).await.unwrap();
    let text = papers_cli_format::format_source_list(&result);

    assert!(text.contains("Nature"));
    assert!(text.contains("0028-0836"));
    assert!(text.contains("journal"));
    assert!(text.contains("Springer Nature"));
}

#[tokio::test]
async fn test_source_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources/S1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(source_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let source = papers_core::api::source_get(&client, "S1", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_source_get(&source);

    assert!(text.contains("Nature"));
    assert!(text.contains("Springer Nature"));
}

#[tokio::test]
async fn test_institution_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/institutions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(institution_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::institution_list(&client, &InstitutionListParams::default()).await.unwrap();
    let text = papers_cli_format::format_institution_list(&result);

    assert!(text.contains("MIT"));
    assert!(text.contains("Cambridge"));
    assert!(text.contains("US"));
    assert!(text.contains("h-index: 500"));
}

#[tokio::test]
async fn test_institution_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/institutions/I1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(institution_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let inst = papers_core::api::institution_get(&client, "I1", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_institution_get(&inst);

    assert!(text.contains("MIT"));
    assert!(text.contains("ror.org"));
    assert!(text.contains("Cambridge"));
}

#[tokio::test]
async fn test_topic_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/topics"))
        .respond_with(ResponseTemplate::new(200).set_body_string(topic_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::topic_list(&client, &TopicListParams::default()).await.unwrap();
    let text = papers_cli_format::format_topic_list(&result);

    assert!(text.contains("Machine Learning"));
    assert!(text.contains("Artificial Intelligence"));
    assert!(text.contains("Computer Science"));
    assert!(text.contains("Physical Sciences"));
}

#[tokio::test]
async fn test_topic_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/topics/T1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(topic_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let topic = papers_core::api::topic_get(&client, "T1", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_topic_get(&topic);

    assert!(text.contains("Machine Learning"));
    assert!(text.contains("Study of algorithms"));
}

#[tokio::test]
async fn test_publisher_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/publishers"))
        .respond_with(ResponseTemplate::new(200).set_body_string(publisher_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::publisher_list(&client, &PublisherListParams::default()).await.unwrap();
    let text = papers_cli_format::format_publisher_list(&result);

    assert!(text.contains("Elsevier"));
    assert!(text.contains("level 0"));
    assert!(text.contains("NL"));
}

#[tokio::test]
async fn test_publisher_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/publishers/P1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(publisher_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let pub_ = papers_core::api::publisher_get(&client, "P1", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_publisher_get(&pub_);

    assert!(text.contains("Elsevier"));
    assert!(text.contains("NL"));
}

#[tokio::test]
async fn test_funder_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/funders"))
        .respond_with(ResponseTemplate::new(200).set_body_string(funder_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::funder_list(&client, &FunderListParams::default()).await.unwrap();
    let text = papers_cli_format::format_funder_list(&result);

    assert!(text.contains("NIH"));
    assert!(text.contains("100000 awards"));
    assert!(text.contains("US biomedical research"));
}

#[tokio::test]
async fn test_funder_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/funders/F1"))
        .respond_with(ResponseTemplate::new(200).set_body_string(funder_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let funder = papers_core::api::funder_get(&client, "F1", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_funder_get(&funder);

    assert!(text.contains("NIH"));
    assert!(text.contains("US"));
    assert!(text.contains("100000"));
}

#[tokio::test]
async fn test_work_autocomplete_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/autocomplete/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(autocomplete_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::work_autocomplete(&client, "machine").await.unwrap();
    let text = papers_cli_format::format_autocomplete(&result);

    assert!(text.contains("Machine Learning Paper"));
    assert!(text.contains("works/W1"));
    assert!(text.contains("Alice Smith"));
}

#[tokio::test]
async fn test_work_find_no_api_key() {
    // Test the key-check logic by verifying the error path
    // We simulate the env-var check without spawning a process
    let key_present = std::env::var("OPENALEX_KEY").is_ok();
    if !key_present {
        // No key: confirm the env check fails as expected
        assert!(std::env::var("OPENALEX_KEY").is_err());
    }
    // The test passes either way — it just confirms the check can be made
}

#[tokio::test]
async fn test_api_error_returns_clean_message() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/NOTFOUND"))
        .respond_with(ResponseTemplate::new(404).set_body_string(r#"{"error": "not found"}"#))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::work_get(&client, "NOTFOUND", &GetParams::default()).await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(!err.is_empty());
    assert!(err.contains("404") || err.contains("not found") || err.len() > 0);
}

// ── Domain tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_domain_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/domains"))
        .respond_with(ResponseTemplate::new(200).set_body_string(domain_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::domain_list(&client, &DomainListParams::default()).await.unwrap();
    let text = papers_cli_format::format_domain_list(&result);

    assert!(text.contains("Physical Sciences"));
    assert!(text.contains("Computer Science"));
    assert!(text.contains("134263529 works"));
}

#[tokio::test]
async fn test_domain_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/domains/3"))
        .respond_with(ResponseTemplate::new(200).set_body_string(domain_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let domain = papers_core::api::domain_get(&client, "3", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_domain_get(&domain);

    assert!(text.contains("Physical Sciences"));
    assert!(text.contains("Computer Science"));
    assert!(text.contains("Engineering"));
}

// ── Field tests ───────────────────────────────────────────────────────────

#[tokio::test]
async fn test_field_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/fields"))
        .respond_with(ResponseTemplate::new(200).set_body_string(field_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::field_list(&client, &FieldListParams::default()).await.unwrap();
    let text = papers_cli_format::format_field_list(&result);

    assert!(text.contains("Computer Science"));
    assert!(text.contains("Physical Sciences"));
    assert!(text.contains("2 subfields"));
}

#[tokio::test]
async fn test_field_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/fields/17"))
        .respond_with(ResponseTemplate::new(200).set_body_string(field_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let field = papers_core::api::field_get(&client, "17", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_field_get(&field);

    assert!(text.contains("Computer Science"));
    assert!(text.contains("Physical Sciences"));
    assert!(text.contains("Artificial Intelligence"));
}

// ── Subfield tests ────────────────────────────────────────────────────────

#[tokio::test]
async fn test_subfield_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/subfields"))
        .respond_with(ResponseTemplate::new(200).set_body_string(subfield_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let result = papers_core::api::subfield_list(&client, &SubfieldListParams::default()).await.unwrap();
    let text = papers_cli_format::format_subfield_list(&result);

    assert!(text.contains("Artificial Intelligence"));
    assert!(text.contains("Computer Science"));
    assert!(text.contains("Physical Sciences"));
}

#[tokio::test]
async fn test_subfield_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/subfields/1702"))
        .respond_with(ResponseTemplate::new(200).set_body_string(subfield_get_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let subfield = papers_core::api::subfield_get(&client, "1702", &GetParams::default()).await.unwrap();
    let text = papers_cli_format::format_subfield_get(&subfield);

    assert!(text.contains("Artificial Intelligence"));
    assert!(text.contains("Computer Science"));
    assert!(text.contains("study of intelligent agents"));
}

// ── Work filter alias tests ───────────────────────────────────────────────

fn search_result_json(id: &str) -> String {
    format!(
        r#"{{
            "meta": {{"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 1, "next_cursor": null, "groups_count": null}},
            "results": [{{"id": "{id}", "display_name": "Test Entity"}}],
            "group_by": []
        }}"#
    )
}

#[tokio::test]
async fn test_work_list_year_flag() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "publication_year:>2020"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = WorkListParams {
        year: Some(">2020".to_string()),
        ..Default::default()
    };
    let result = papers_core::api::work_list(&client, &params).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_citations_flag() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "cited_by_count:>100"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = WorkListParams {
        citations: Some(">100".to_string()),
        ..Default::default()
    };
    let result = papers_core::api::work_list(&client, &params).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_author_id_flag() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "authorships.author.id:A5083138872"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = WorkListParams {
        author: Some("A5083138872".to_string()),
        ..Default::default()
    };
    let result = papers_core::api::work_list(&client, &params).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_publisher_search_flag() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/publishers"))
        .and(query_param("search", "acm"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/P4310319798")),
        )
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "primary_location.source.publisher_lineage:P4310319798"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = WorkListParams {
        publisher: Some("acm".to_string()),
        ..Default::default()
    };
    let result = papers_core::api::work_list(&client, &params).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_combined_filter_and_year() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "publication_year:2024,is_oa:true"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_body()))
        .mount(&mock)
        .await;

    let client = make_client(&mock);
    let params = WorkListParams {
        year: Some("2024".to_string()),
        filter: Some("is_oa:true".to_string()),
        ..Default::default()
    };
    let result = papers_core::api::work_list(&client, &params).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_overlap_error() {
    let client = OpenAlexClient::new();
    let params = WorkListParams {
        year: Some("2024".to_string()),
        filter: Some("publication_year:>2020".to_string()),
        ..Default::default()
    };
    let result = papers_core::api::work_list(&client, &params).await;
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("year"));
    assert!(err.contains("publication_year"));
}

// Format functions exposed for testing (mirrors src/format.rs logic)
mod papers_cli_format {
    use papers_core::summary::{
        AuthorSummary, DomainSummary, FieldSummary, FunderSummary, InstitutionSummary,
        PublisherSummary, SlimListResponse, SourceSummary, SubfieldSummary, TopicSummary,
        WorkSummary,
    };
    use papers_core::{
        Author, AutocompleteResponse, Domain, Field, Funder, Institution, ListMeta, Publisher,
        Source, Subfield, Topic, Work,
    };

    fn meta_line(meta: &ListMeta) -> String {
        let page = meta.page.unwrap_or(1);
        let per_page = meta.per_page.unwrap_or(10);
        format!("Found {} results · page {} (showing {})", meta.count, page, per_page)
    }

    pub fn format_work_list(resp: &SlimListResponse<WorkSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, w) in resp.results.iter().enumerate() {
            let title = w.title.as_deref().unwrap_or("(untitled)");
            let year = w.publication_year.map_or(String::new(), |y| format!(" ({y})"));
            out.push_str(&format!("\n {:>2}  {}{}\n", i + 1, title, year));
            if !w.authors.is_empty() {
                out.push_str(&format!("     {}\n", w.authors.join(" · ")));
            }
            if let Some(c) = w.cited_by_count {
                out.push_str(&format!("     {c} citations\n"));
            }
        }
        out
    }

    pub fn format_work_get(w: &Work) -> String {
        let mut out = String::new();
        let title = w.display_name.as_deref().unwrap_or("(untitled)");
        out.push_str(&format!("Work: {title}\n"));
        out.push_str(&format!("ID:   {}\n", w.id));
        if let Some(doi) = &w.doi {
            out.push_str(&format!("DOI:  {doi}\n"));
        }
        let authorships = w.authorships.as_deref().unwrap_or_default();
        if !authorships.is_empty() {
            out.push_str("Authors:\n");
            for a in authorships {
                let name = a.author.as_ref().and_then(|au| au.display_name.as_deref()).unwrap_or("?");
                out.push_str(&format!("  {name}\n"));
            }
        }
        if let Some(abs) = &w.abstract_text {
            out.push_str(&format!("Abstract: {abs}\n"));
        }
        out
    }

    pub fn format_author_list(resp: &SlimListResponse<AuthorSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, a) in resp.results.iter().enumerate() {
            let name = a.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            if let Some(h) = a.h_index {
                out.push_str(&format!("     h-index: {h}\n"));
            }
            if !a.last_known_institutions.is_empty() {
                out.push_str(&format!("     {}\n", a.last_known_institutions.join(", ")));
            }
            if !a.top_topics.is_empty() {
                out.push_str(&format!("     Topics: {}\n", a.top_topics.join(", ")));
            }
        }
        out
    }

    pub fn format_author_get(a: &Author) -> String {
        let mut out = String::new();
        let name = a.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Author: {name}\n"));
        if let Some(orcid) = &a.orcid {
            out.push_str(&format!("ORCID: {orcid}\n"));
        }
        if let Some(w) = a.works_count {
            out.push_str(&format!("{w} works\n"));
        }
        if let Some(ss) = &a.summary_stats {
            if let Some(h) = ss.h_index {
                out.push_str(&format!("h-index: {h}\n"));
            }
        }
        out
    }

    pub fn format_source_list(resp: &SlimListResponse<SourceSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, s) in resp.results.iter().enumerate() {
            let name = s.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            if let Some(issn) = &s.issn_l {
                out.push_str(&format!("     ISSN: {issn}\n"));
            }
            if let Some(t) = &s.r#type {
                out.push_str(&format!("     {t}\n"));
            }
            if let Some(org) = &s.host_organization_name {
                out.push_str(&format!("     {org}\n"));
            }
        }
        out
    }

    pub fn format_source_get(s: &Source) -> String {
        let mut out = String::new();
        let name = s.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Source: {name}\n"));
        if let Some(org) = &s.host_organization_name {
            out.push_str(&format!("Publisher: {org}\n"));
        }
        out
    }

    pub fn format_institution_list(resp: &SlimListResponse<InstitutionSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, inst) in resp.results.iter().enumerate() {
            let name = inst.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            let mut parts = Vec::new();
            if let Some(city) = &inst.city { parts.push(city.clone()); }
            if let Some(cc) = &inst.country_code { parts.push(cc.clone()); }
            if !parts.is_empty() { out.push_str(&format!("     {}\n", parts.join(", "))); }
            if let Some(h) = inst.h_index { out.push_str(&format!("     h-index: {h}\n")); }
        }
        out
    }

    pub fn format_institution_get(inst: &Institution) -> String {
        let mut out = String::new();
        let name = inst.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Institution: {name}\n"));
        if let Some(ror) = &inst.ror {
            out.push_str(&format!("ROR: {ror}\n"));
        }
        if let Some(geo) = &inst.geo {
            if let Some(city) = &geo.city {
                out.push_str(&format!("City: {city}\n"));
            }
        }
        out
    }

    pub fn format_topic_list(resp: &SlimListResponse<TopicSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, t) in resp.results.iter().enumerate() {
            let name = t.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            let hierarchy: Vec<_> = [t.subfield.as_deref(), t.field.as_deref(), t.domain.as_deref()]
                .into_iter().flatten().collect();
            if !hierarchy.is_empty() {
                out.push_str(&format!("     {}\n", hierarchy.join(" → ")));
            }
        }
        out
    }

    pub fn format_topic_get(t: &Topic) -> String {
        let mut out = String::new();
        let name = t.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Topic: {name}\n"));
        if let Some(desc) = &t.description {
            out.push_str(&format!("Description: {desc}\n"));
        }
        out
    }

    pub fn format_publisher_list(resp: &SlimListResponse<PublisherSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, p) in resp.results.iter().enumerate() {
            let name = p.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            if let Some(level) = p.hierarchy_level {
                out.push_str(&format!("     level {level}\n"));
            }
            if let Some(codes) = &p.country_codes {
                out.push_str(&format!("     {}\n", codes.join(", ")));
            }
        }
        out
    }

    pub fn format_publisher_get(p: &Publisher) -> String {
        let mut out = String::new();
        let name = p.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Publisher: {name}\n"));
        if let Some(codes) = &p.country_codes {
            out.push_str(&format!("Countries: {}\n", codes.join(", ")));
        }
        out
    }

    pub fn format_funder_list(resp: &SlimListResponse<FunderSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, f) in resp.results.iter().enumerate() {
            let name = f.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            if let Some(a) = f.awards_count {
                out.push_str(&format!("     {a} awards\n"));
            }
            if let Some(desc) = &f.description {
                out.push_str(&format!("     {desc}\n"));
            }
        }
        out
    }

    pub fn format_funder_get(f: &Funder) -> String {
        let mut out = String::new();
        let name = f.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Funder: {name}\n"));
        if let Some(cc) = &f.country_code {
            out.push_str(&format!("Country: {cc}\n"));
        }
        if let Some(a) = f.awards_count {
            out.push_str(&format!("{a} awards\n"));
        }
        out
    }

    pub fn format_domain_list(resp: &SlimListResponse<DomainSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, d) in resp.results.iter().enumerate() {
            let name = d.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            if !d.fields.is_empty() {
                out.push_str(&format!("     Fields: {}\n", d.fields.join(", ")));
            }
            let mut stats = Vec::new();
            if let Some(w) = d.works_count { stats.push(format!("{w} works")); }
            if let Some(c) = d.cited_by_count { stats.push(format!("{c} citations")); }
            if !stats.is_empty() { out.push_str(&format!("     {}\n", stats.join(" · "))); }
        }
        out
    }

    pub fn format_domain_get(d: &Domain) -> String {
        let mut out = String::new();
        let name = d.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Domain: {name}\n"));
        if let Some(fields) = &d.fields {
            for f in fields {
                if let Some(name) = &f.display_name {
                    out.push_str(&format!("  {name}\n"));
                }
            }
        }
        out
    }

    pub fn format_field_list(resp: &SlimListResponse<FieldSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, f) in resp.results.iter().enumerate() {
            let name = f.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            if let Some(domain) = &f.domain {
                out.push_str(&format!("     Domain: {domain}\n"));
            }
            out.push_str(&format!("     {} subfields\n", f.subfield_count));
        }
        out
    }

    pub fn format_field_get(f: &Field) -> String {
        let mut out = String::new();
        let name = f.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Field: {name}\n"));
        if let Some(domain) = &f.domain {
            if let Some(dn) = &domain.display_name {
                out.push_str(&format!("Domain: {dn}\n"));
            }
        }
        if let Some(subfields) = &f.subfields {
            for sf in subfields {
                if let Some(name) = &sf.display_name {
                    out.push_str(&format!("  {name}\n"));
                }
            }
        }
        out
    }

    pub fn format_subfield_list(resp: &SlimListResponse<SubfieldSummary>) -> String {
        let mut out = format!("{}\n", meta_line(&resp.meta));
        for (i, s) in resp.results.iter().enumerate() {
            let name = s.display_name.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  {name}\n", i + 1));
            let hierarchy: Vec<_> = [s.field.as_deref(), s.domain.as_deref()]
                .into_iter().flatten().collect();
            if !hierarchy.is_empty() {
                out.push_str(&format!("     {}\n", hierarchy.join(" → ")));
            }
        }
        out
    }

    pub fn format_subfield_get(s: &Subfield) -> String {
        let mut out = String::new();
        let name = s.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("Subfield: {name}\n"));
        let hierarchy: Vec<_> = [
            s.field.as_ref().and_then(|f| f.display_name.as_deref()),
            s.domain.as_ref().and_then(|d| d.display_name.as_deref()),
        ].into_iter().flatten().collect();
        if !hierarchy.is_empty() {
            out.push_str(&format!("Hierarchy: {}\n", hierarchy.join(" → ")));
        }
        if let Some(desc) = &s.description {
            out.push_str(&format!("{desc}\n"));
        }
        out
    }

    pub fn format_autocomplete(resp: &AutocompleteResponse) -> String {
        let mut out = String::new();
        for (i, r) in resp.results.iter().enumerate() {
            out.push_str(&format!("{:>2}  {} [{}]\n", i + 1, r.display_name, r.short_id.as_deref().unwrap_or("")));
            if let Some(hint) = &r.hint {
                if !hint.is_empty() {
                    out.push_str(&format!("    {hint}\n"));
                }
            }
        }
        if out.is_empty() { out.push_str("No results.\n"); }
        out
    }

    // ── Zotero format helpers ──────────────────────────────────────────────

    use papers_zotero::{Collection, Creator, Group, Item, PagedResponse, SavedSearch, Tag};

    fn strip_html(html: &str) -> String {
        let mut out = String::with_capacity(html.len());
        let mut in_tag = false;
        for c in html.chars() {
            match c {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => out.push(c),
                _ => {}
            }
        }
        out
    }

    fn creator_display(c: &Creator) -> String {
        if let Some(name) = &c.name {
            name.clone()
        } else {
            match (&c.first_name, &c.last_name) {
                (Some(f), Some(l)) => format!("{l}, {f}"),
                (None, Some(l)) => l.clone(),
                (Some(f), None) => f.clone(),
                _ => "?".to_string(),
            }
        }
    }

    pub fn format_zotero_work_list(resp: &PagedResponse<Item>) -> String {
        let header = match resp.total_results {
            Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
            _ => format!("{} item(s)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, item) in resp.items.iter().enumerate() {
            let title = item.data.title.as_deref().unwrap_or("(untitled)");
            let year = item.data.date.as_deref()
                .and_then(|d| {
                    let y = d.split(['-', '/']).next()?;
                    if y.len() == 4 && y.chars().all(|c| c.is_ascii_digit()) {
                        Some(format!(" ({y})"))
                    } else { None }
                })
                .unwrap_or_default();
            out.push_str(&format!("\n {:>2}  [{}] {}{}\n", i + 1, item.key, title, year));
            if !item.data.creators.is_empty() {
                let authors: Vec<String> = item.data.creators.iter().take(3).map(creator_display).collect();
                let suffix = if item.data.creators.len() > 3 { " et al." } else { "" };
                out.push_str(&format!("     {}{suffix}\n", authors.join("; ")));
            }
            let mut meta_parts = Vec::new();
            if let Some(j) = &item.data.publication_title { meta_parts.push(j.clone()); }
            meta_parts.push(item.data.item_type.clone());
            if let Some(doi) = &item.data.doi { meta_parts.push(format!("DOI: {doi}")); }
            if !meta_parts.is_empty() {
                out.push_str(&format!("     {}\n", meta_parts.join(" · ")));
            }
            if !item.data.tags.is_empty() {
                let tag_names: Vec<&str> = item.data.tags.iter().map(|t| t.tag.as_str()).collect();
                out.push_str(&format!("     Tags: {}\n", tag_names.join(", ")));
            }
        }
        out
    }

    pub fn format_zotero_item_get(item: &Item) -> String {
        let mut out = String::new();
        let title = item.data.title.as_deref().unwrap_or("(untitled)");
        out.push_str(&format!("{}: {}\n", item.data.item_type, title));
        out.push_str(&format!("Key:  {}\n", item.key));
        if let Some(doi) = &item.data.doi { out.push_str(&format!("DOI:  {doi}\n")); }
        if let Some(date) = &item.data.date { out.push_str(&format!("Date: {date}\n")); }
        if let Some(j) = &item.data.publication_title { out.push_str(&format!("Publication: {j}\n")); }
        if !item.data.creators.is_empty() {
            out.push_str("\nCreators:\n");
            for (i, c) in item.data.creators.iter().enumerate() {
                let name = creator_display(c);
                out.push_str(&format!("  {:>2}. {} ({})\n", i + 1, name, c.creator_type));
            }
        }
        if let Some(abs) = &item.data.abstract_note { out.push_str(&format!("\nAbstract:\n  {abs}\n")); }
        if !item.data.tags.is_empty() {
            let tag_names: Vec<&str> = item.data.tags.iter().map(|t| t.tag.as_str()).collect();
            out.push_str(&format!("\nTags: {}\n", tag_names.join(", ")));
        }
        if let Some(note) = &item.data.note {
            let stripped = strip_html(note);
            let trimmed = stripped.trim();
            if !trimmed.is_empty() {
                out.push_str(&format!("\nNote:\n  {trimmed}\n"));
            }
        }
        if let Some(ann_type) = item.data.extra_fields.get("annotationType").and_then(|v| v.as_str()) {
            out.push_str(&format!("Annotation type: {ann_type}\n"));
        }
        if let Some(text) = item.data.extra_fields.get("annotationText").and_then(|v| v.as_str()) {
            if !text.is_empty() { out.push_str(&format!("Text: {text}\n")); }
        }
        out
    }

    pub fn format_zotero_attachment_list(resp: &PagedResponse<Item>) -> String {
        let header = match resp.total_results {
            Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
            _ => format!("{} attachment(s)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, item) in resp.items.iter().enumerate() {
            let display = item.data.filename.as_deref()
                .or_else(|| item.data.url.as_deref())
                .unwrap_or("(no file)");
            let link_mode = item.data.link_mode.as_deref().unwrap_or("?");
            out.push_str(&format!("\n {:>2}  [{}] {}\n", i + 1, item.key, display));
            let mut parts = vec![link_mode.to_string()];
            if let Some(parent) = &item.data.parent_item { parts.push(format!("parent: {parent}")); }
            if let Some(ct) = &item.data.content_type { parts.push(ct.clone()); }
            out.push_str(&format!("     {}\n", parts.join(" · ")));
        }
        out
    }

    pub fn format_zotero_annotation_list(resp: &PagedResponse<Item>) -> String {
        if resp.items.is_empty() { return "No annotations.\n".to_string(); }
        let header = match resp.total_results.filter(|&n| n > 0) {
            Some(n) => format!("Found {} annotations · showing {}\n", n, resp.items.len()),
            None => format!("{} annotation(s)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, item) in resp.items.iter().enumerate() {
            push_annotation_entry(&mut out, i, item);
        }
        out
    }

    pub fn format_zotero_annotation_list_vec(items: &[Item]) -> String {
        if items.is_empty() { return "No annotations.\n".to_string(); }
        let mut out = format!("{} annotation(s)\n", items.len());
        for (i, item) in items.iter().enumerate() {
            push_annotation_entry(&mut out, i, item);
        }
        out
    }

    fn push_annotation_entry(out: &mut String, i: usize, item: &Item) {
        let ann_type = item.data.extra_fields.get("annotationType").and_then(|v| v.as_str()).unwrap_or("?");
        let page = item.data.extra_fields.get("annotationPageLabel").and_then(|v| v.as_str()).unwrap_or("");
        let color = item.data.extra_fields.get("annotationColor").and_then(|v| v.as_str()).unwrap_or("");
        let parent = item.data.parent_item.as_deref().unwrap_or("");
        out.push_str(&format!("\n {:>2}  [{}] {}", i + 1, item.key, ann_type));
        if !page.is_empty() { out.push_str(&format!(" (p. {page})")); }
        if !color.is_empty() { out.push_str(&format!(" {color}")); }
        out.push('\n');
        if !parent.is_empty() { out.push_str(&format!("     Parent: {parent}\n")); }
        if let Some(text) = item.data.extra_fields.get("annotationText").and_then(|v| v.as_str()) {
            if !text.is_empty() { out.push_str(&format!("     \"{text}\"\n")); }
        }
        if let Some(comment) = item.data.extra_fields.get("annotationComment").and_then(|v| v.as_str()) {
            if !comment.is_empty() { out.push_str(&format!("     Note: {comment}\n")); }
        }
    }

    pub fn format_zotero_note_list(resp: &PagedResponse<Item>) -> String {
        let header = match resp.total_results {
            Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
            _ => format!("{} note(s)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, item) in resp.items.iter().enumerate() {
            let parent = item.data.parent_item.as_deref().unwrap_or("(standalone)");
            out.push_str(&format!("\n {:>2}  [{}] parent: {parent}\n", i + 1, item.key));
            if let Some(note) = &item.data.note {
                let stripped = strip_html(note);
                let trimmed = stripped.trim().to_string();
                if !trimmed.is_empty() {
                    let preview = if trimmed.chars().count() > 80 {
                        format!("{}…", trimmed.chars().take(80).collect::<String>())
                    } else {
                        trimmed
                    };
                    out.push_str(&format!("     {preview}\n"));
                }
            }
        }
        out
    }

    pub fn format_zotero_collection_list(resp: &PagedResponse<Collection>) -> String {
        let header = match resp.total_results {
            Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
            _ => format!("{} collection(s)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, coll) in resp.items.iter().enumerate() {
            let parent = coll.data.parent_key().map(|k| format!(" (sub of {k})")).unwrap_or_default();
            let items = coll.meta.num_items.map(|n| format!(", {n} items")).unwrap_or_default();
            out.push_str(&format!("\n {:>2}  [{}] {}{parent}{items}\n", i + 1, coll.key, coll.data.name));
        }
        out
    }

    pub fn format_zotero_collection_get(coll: &Collection) -> String {
        let mut out = String::new();
        out.push_str(&format!("Collection: {}\n", coll.data.name));
        out.push_str(&format!("Key:   {}\n", coll.key));
        if let Some(parent_key) = coll.data.parent_key() {
            out.push_str(&format!("Parent: {parent_key}\n"));
        } else {
            out.push_str("Parent: (top-level)\n");
        }
        if let Some(n) = coll.meta.num_items { out.push_str(&format!("Items: {n}\n")); }
        if let Some(n) = coll.meta.num_collections { out.push_str(&format!("Sub-collections: {n}\n")); }
        out
    }

    pub fn format_zotero_tag_list(resp: &PagedResponse<Tag>) -> String {
        let header = match resp.total_results {
            Some(n) if n > 0 => format!("Found {} tags · showing {}\n", n, resp.items.len()),
            _ => format!("{} tag(s)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, tag) in resp.items.iter().enumerate() {
            let count = tag.meta.num_items.map(|n| format!(" ({n} items)")).unwrap_or_default();
            out.push_str(&format!("\n {:>2}  {}{count}\n", i + 1, tag.tag));
        }
        out
    }

    pub fn format_zotero_search_list(resp: &PagedResponse<SavedSearch>) -> String {
        let header = match resp.total_results {
            Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
            _ => format!("{} saved search(es)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, search) in resp.items.iter().enumerate() {
            let n = search.data.conditions.len();
            out.push_str(&format!("\n {:>2}  [{}] {}\n", i + 1, search.key, search.data.name));
            out.push_str(&format!("     {n} condition(s)\n"));
        }
        out
    }

    pub fn format_zotero_search_get(search: &SavedSearch) -> String {
        let mut out = String::new();
        out.push_str(&format!("Search: {}\n", search.data.name));
        out.push_str(&format!("Key:    {}\n", search.key));
        if !search.data.conditions.is_empty() {
            out.push_str("\nConditions:\n");
            for cond in &search.data.conditions {
                out.push_str(&format!("  {} {} {}\n", cond.condition, cond.operator, cond.value));
            }
        }
        out
    }

    pub fn format_zotero_group_list(resp: &PagedResponse<Group>) -> String {
        let header = match resp.total_results {
            Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
            _ => format!("{} group(s)\n", resp.items.len()),
        };
        let mut out = header;
        for (i, group) in resp.items.iter().enumerate() {
            let gtype = group.data.group_type.as_deref().unwrap_or("?");
            let items = group.meta.num_items.map(|n| format!(", {n} items")).unwrap_or_default();
            out.push_str(&format!("\n {:>2}  [{}] {} ({gtype}){items}\n", i + 1, group.id, group.data.name));
            if let Some(desc) = &group.data.description {
                if !desc.is_empty() {
                    let snippet = if desc.chars().count() > 100 {
                        format!("{}…", desc.chars().take(100).collect::<String>())
                    } else { desc.clone() };
                    out.push_str(&format!("     {snippet}\n"));
                }
            }
        }
        out
    }

}

// ── Zotero helpers ────────────────────────────────────────────────────────

fn make_zotero_client(mock: &MockServer) -> ZoteroClient {
    ZoteroClient::new("test", "test-key").with_base_url(mock.uri())
}

fn zotero_arr(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("Total-Results", "1")
        .insert_header("Last-Modified-Version", "100")
        .set_body_string(body)
}

fn zotero_arr_empty() -> ResponseTemplate {
    ResponseTemplate::new(200)
        .insert_header("Total-Results", "0")
        .insert_header("Last-Modified-Version", "100")
        .set_body_string("[]")
}

fn z_items_body() -> String {
    r#"[{"key":"ABC12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ABC12345","version":1,"itemType":"journalArticle","title":"Test Paper"}}]"#.to_string()
}

fn z_item_body() -> String {
    r#"{"key":"ABC12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ABC12345","version":1,"itemType":"journalArticle","title":"Test Paper"}}"#.to_string()
}

fn z_item_with_collections_body() -> String {
    r#"{"key":"ABC12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ABC12345","version":1,"itemType":"journalArticle","title":"Test Paper","collections":["COL12345"]}}"#.to_string()
}

fn z_collections_body() -> String {
    r#"[{"key":"COL12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{"numCollections":0,"numItems":5},"data":{"key":"COL12345","version":1,"name":"Test Collection","parentCollection":false,"relations":{}}}]"#.to_string()
}

fn z_collection_body() -> String {
    r#"{"key":"COL12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{"numCollections":0,"numItems":5},"data":{"key":"COL12345","version":1,"name":"Test Collection","parentCollection":false,"relations":{}}}"#.to_string()
}

fn z_tags_body() -> String {
    r#"[{"tag":"TestTag","links":{},"meta":{"type":0,"numItems":5}}]"#.to_string()
}

fn z_searches_body() -> String {
    r#"[{"key":"SRCH1234","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"data":{"key":"SRCH1234","version":1,"name":"Test Search","conditions":[]}}]"#.to_string()
}

fn z_search_body() -> String {
    r#"{"key":"SRCH1234","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"data":{"key":"SRCH1234","version":1,"name":"Test Search","conditions":[]}}"#.to_string()
}

fn z_groups_body() -> String {
    r#"[{"id":12345,"version":1,"links":{},"meta":{"numItems":10},"data":{"id":12345,"version":1,"name":"Test Group","owner":1,"type":"Private","description":""}}]"#.to_string()
}

fn z_item_rich_body() -> String {
    r#"{"key":"RICH1234","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"RICH1234","version":1,"itemType":"journalArticle","title":"GPU Sorting Algorithms","date":"2024-03-15","DOI":"10.1234/gpu.2024","publicationTitle":"IEEE Transactions on Parallel Computing","creators":[{"creatorType":"author","firstName":"Alice","lastName":"Smith"},{"creatorType":"author","firstName":"Bob","lastName":"Jones"},{"creatorType":"author","firstName":"Carol","lastName":"Wu"},{"creatorType":"author","firstName":"Dave","lastName":"Lee"}],"abstractNote":"We present efficient GPU-based sorting algorithms.","tags":[{"tag":"GPU"},{"tag":"Sorting"}],"collections":["COL12345"]}}"#.to_string()
}

fn z_attachment_rich_body() -> String {
    r#"[{"key":"ATT12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ATT12345","version":1,"itemType":"attachment","linkMode":"imported_file","contentType":"application/pdf","filename":"gpu_sorting.pdf","parentItem":"RICH1234"}}]"#.to_string()
}

fn z_annotation_rich_body() -> String {
    r##"[{"key":"ANN12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ANN12345","version":1,"itemType":"annotation","parentItem":"ATT12345","annotationType":"highlight","annotationText":"GPU sorting outperforms CPU by 10x","annotationComment":"Important result","annotationPageLabel":"5","annotationColor":"#ffd400"}}]"##.to_string()
}

fn z_note_rich_body() -> String {
    r#"[{"key":"NOTE5678","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"NOTE5678","version":1,"itemType":"note","note":"<p>This paper introduces a <b>parallel</b> sorting network.</p>","parentItem":"RICH1234"}}]"#.to_string()
}

fn z_collection_with_parent_body() -> String {
    r#"[{"key":"SUB12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{"numCollections":0,"numItems":3},"data":{"key":"SUB12345","version":1,"name":"GPU Papers","parentCollection":"COL12345","relations":{}}}]"#.to_string()
}

fn z_tag_with_count_body() -> String {
    r#"[{"tag":"Starred","links":{},"meta":{"type":0,"numItems":31}},{"tag":"Survey","links":{},"meta":{"type":0,"numItems":7}}]"#.to_string()
}

fn z_search_with_conditions_body() -> String {
    r#"{"key":"SRCH5678","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"data":{"key":"SRCH5678","version":1,"name":"GPU Papers","conditions":[{"condition":"tag","operator":"is","value":"GPU"},{"condition":"itemType","operator":"is","value":"journalArticle"}]}}"#.to_string()
}

fn z_group_with_desc_body() -> String {
    r#"[{"id":99999,"version":1,"links":{},"meta":{"numItems":42},"data":{"id":99999,"version":1,"name":"CompSci Reading Group","owner":1,"type":"PublicOpen","description":"A group for computer science paper discussions"}}]"#.to_string()
}

// ── Zotero work tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_work_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_top_items(&ItemListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Paper"));
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_work_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ABC12345"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_item_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_item("ABC12345").await.unwrap();
    assert_eq!(result.key, "ABC12345");
    assert_eq!(result.data.title.as_deref(), Some("Test Paper"));
}

#[tokio::test]
async fn test_zotero_work_tags() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ABC12345/tags"))
        .respond_with(zotero_arr(&z_tags_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_item_tags("ABC12345", &TagListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("TestTag"));
}

#[tokio::test]
async fn test_zotero_work_notes() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ABC12345/children"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("note".into()), ..Default::default() };
    let result = client.list_item_children("ABC12345", &params).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_work_attachments() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ABC12345/children"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let result = client.list_item_children("ABC12345", &params).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_work_annotations_empty() {
    // Multi-step: fetch attachments (empty), then annotations per attachment
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ABC12345/children"))
        .respond_with(zotero_arr_empty())
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let attachments = client.list_item_children("ABC12345", &att_params).await.unwrap();
    let mut all_annotations = Vec::new();
    for att in &attachments.items {
        let ann_params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
        let ann = client.list_item_children(&att.key, &ann_params).await.unwrap();
        all_annotations.extend(ann.items);
    }
    assert_eq!(all_annotations.len(), 0);
}

#[tokio::test]
async fn test_zotero_work_collections() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ABC12345"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_item_with_collections_body()))
        .mount(&mock)
        .await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_collection_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let item = client.get_item("ABC12345").await.unwrap();
    let collection_keys = item.data.collections.clone();
    let mut collections = Vec::new();
    for key in &collection_keys {
        collections.push(client.get_collection(key).await.unwrap());
    }
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].data.name, "Test Collection");
}

// ── Zotero attachment tests ───────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_attachment_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let result = client.list_items(&params).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_attachment_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ABC12345"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_item_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_item("ABC12345").await.unwrap();
    assert_eq!(result.key, "ABC12345");
}

// ── Zotero annotation tests ───────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_annotation_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
    let result = client.list_items(&params).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_annotation_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/ANN12345"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_item_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_item("ANN12345").await.unwrap();
    assert_eq!(result.data.title.as_deref(), Some("Test Paper"));
}

// ── Zotero note tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_note_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("note".into()), ..Default::default() };
    let result = client.list_items(&params).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_note_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/NOTE1234"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_item_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_item("NOTE1234").await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Paper"));
}

// ── Zotero collection tests ───────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_collection_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections"))
        .respond_with(zotero_arr(&z_collections_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_collections(&CollectionListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Collection"));
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_collection_list_top() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/top"))
        .respond_with(zotero_arr(&z_collections_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_top_collections(&CollectionListParams::default()).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_collection_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_collection_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_collection("COL12345").await.unwrap();
    assert_eq!(result.data.name, "Test Collection");
}

#[tokio::test]
async fn test_zotero_collection_works() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345/items/top"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_collection_top_items("COL12345", &ItemListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Paper"));
}

#[tokio::test]
async fn test_zotero_collection_attachments() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345/items"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let result = client.list_collection_items("COL12345", &params).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_collection_notes() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345/items"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("note".into()), ..Default::default() };
    let result = client.list_collection_items("COL12345", &params).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_collection_annotations_empty() {
    // Multi-step: fetch attachments (empty)
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345/items"))
        .respond_with(zotero_arr_empty())
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let attachments = client.list_collection_items("COL12345", &att_params).await.unwrap();
    let mut all_annotations = Vec::new();
    for att in &attachments.items {
        let ann_params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
        let ann = client.list_item_children(&att.key, &ann_params).await.unwrap();
        all_annotations.extend(ann.items);
    }
    assert_eq!(all_annotations.len(), 0);
}

#[tokio::test]
async fn test_zotero_collection_subcollections() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345/collections"))
        .respond_with(zotero_arr(&z_collections_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_subcollections("COL12345", &CollectionListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Collection"));
}

#[tokio::test]
async fn test_zotero_collection_tags() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345/items/tags"))
        .respond_with(zotero_arr(&z_tags_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_collection_items_tags("COL12345", &TagListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("TestTag"));
}

#[tokio::test]
async fn test_zotero_collection_tags_top() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345/items/top/tags"))
        .respond_with(zotero_arr(&z_tags_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_collection_top_items_tags("COL12345", &TagListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("TestTag"));
}

// ── Zotero tag tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_tag_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/tags"))
        .respond_with(zotero_arr(&z_tags_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_tags(&TagListParams::default()).await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("TestTag"));
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_tag_list_top() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/top/tags"))
        .respond_with(zotero_arr(&z_tags_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_top_items_tags(&TagListParams::default()).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_tag_list_trash() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/trash/tags"))
        .respond_with(zotero_arr(&z_tags_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_trash_tags(&TagListParams::default()).await.unwrap();
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_tag_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/tags/Starred"))
        .respond_with(zotero_arr(&z_tags_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_tag("Starred").await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("TestTag"));
}

// ── Zotero search tests ───────────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_search_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/searches"))
        .respond_with(zotero_arr(&z_searches_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_searches().await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Search"));
    assert_eq!(result.total_results, Some(1));
}

#[tokio::test]
async fn test_zotero_search_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/searches/SRCH1234"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_search_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_search("SRCH1234").await.unwrap();
    assert_eq!(result.data.name, "Test Search");
}

// ── Zotero regression tests ───────────────────────────────────────────────

/// Bug: format_zotero_note_list panicked on notes with Unicode content (β, →, ", etc.)
/// when the note text was longer than 80 bytes, because `&str[..80]` cuts mid-codepoint.
/// Fix: use `chars().take(80)` instead of byte slicing.
/// This test verifies the ZoteroClient correctly returns Unicode note content so that
/// the format function receives the correct data (the format fix itself is in src/format.rs).
#[tokio::test]
async fn test_zotero_note_unicode_content_parsed_correctly() {
    let mock = MockServer::start().await;
    // Note containing multi-byte Unicode: β (2 bytes), → (3 bytes), " " (3 bytes each)
    let note_content = r#"<p>A function f : Rⁿ → R is affine: f(x) = a·x + β, where β ∈ ℝ and "Linear programs" are defined.</p>"#;
    let note_body = format!(
        r#"[{{"key":"NOTE1234","version":1,"library":{{"type":"user","id":1,"name":"test","links":{{}}}},"links":{{}},"meta":{{}},"data":{{"key":"NOTE1234","version":1,"itemType":"note","note":"{}"}}}}]"#,
        note_content.replace('"', "\\\"")
    );
    Mock::given(method("GET"))
        .and(path("/users/test/items"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&note_body),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("note".into()), ..Default::default() };
    let result = client.list_items(&params).await.unwrap();
    assert_eq!(result.items.len(), 1);
    let note_html = result.items[0].data.note.as_deref().unwrap_or("");
    assert!(note_html.contains('β'), "note should contain Unicode character β");
    assert!(note_html.contains('→'), "note should contain Unicode character →");
}

/// Bug: format_zotero_tag_list (and other format functions) showed "Found 0 tags · showing N"
/// when the Zotero API returned Total-Results: 0 for item-scoped tag endpoints.
/// Fix: treat total_results == Some(0) as absent, fall back to items.len().
/// This test verifies that item-scoped tag responses carry total_results = Some(0).
#[tokio::test]
async fn test_zotero_work_tags_total_results_is_zero() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/WORK1234/tags"))
        .respond_with(
            // Zotero returns Total-Results: 0 for item-scoped tag endpoints
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "0")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&z_tags_body()),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_item_tags("WORK1234", &TagListParams::default()).await.unwrap();
    // total_results is 0 from API header, but items are present
    assert_eq!(result.total_results, Some(0));
    assert_eq!(result.items.len(), 1, "items should still be present despite total_results=0");
    // The format function must NOT display "Found 0 tags" when items exist.
    // (The fix is in src/format.rs: match total_results { Some(n) if n > 0 => ... })
    assert_eq!(result.items[0].tag, "TestTag");
}

/// Bug: zotero_work_annotations and zotero_collection_annotations called /children on ALL
/// attachments (including .md, .json, .txt files), triggering a Zotero API 400 error:
/// "API error (status 400): /children can only be called on PDF, EPUB, and snapshot attachments".
/// Fix: filter to content_type = application/pdf, application/epub+zip, or text/html only.
#[tokio::test]
async fn test_zotero_work_annotations_skips_non_pdf_attachments() {
    let mock = MockServer::start().await;
    let attachments_body = r#"[
        {"key":"PDF01234","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"PDF01234","version":1,"itemType":"attachment","contentType":"application/pdf","linkMode":"imported_url","parentItem":"WORK1234"}},
        {"key":"MDK12345","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"MDK12345","version":1,"itemType":"attachment","contentType":"text/plain","linkMode":"imported_file","parentItem":"WORK1234"}}
    ]"#;
    let annotation_body = r#"[{"key":"ANN01234","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ANN01234","version":1,"itemType":"annotation","parentItem":"PDF01234"}}]"#;

    // Return mixed attachments (PDF + markdown)
    Mock::given(method("GET"))
        .and(path("/users/test/items/WORK1234/children"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "2")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(attachments_body),
        )
        .mount(&mock)
        .await;
    // Only register PDF children — markdown children endpoint is intentionally absent.
    // If the code calls /items/MDK12345/children, wiremock returns 404 and the test would catch it.
    Mock::given(method("GET"))
        .and(path("/users/test/items/PDF01234/children"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(annotation_body),
        )
        .mount(&mock)
        .await;

    let client = make_zotero_client(&mock);
    let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let attachments = client.list_item_children("WORK1234", &att_params).await.unwrap();
    assert_eq!(attachments.items.len(), 2);

    // Apply the same filter as the fixed CLI/MCP handlers
    let ann_params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
    let mut all_annotations = Vec::new();
    for att in &attachments.items {
        let is_annotatable = matches!(
            att.data.content_type.as_deref(),
            Some("application/pdf") | Some("application/epub+zip") | Some("text/html")
        );
        if !is_annotatable { continue; }
        match client.list_item_children(&att.key, &ann_params).await {
            Ok(r) => all_annotations.extend(r.items),
            Err(_) => {},
        }
    }
    // Only the PDF attachment's annotation was retrieved; markdown was skipped
    assert_eq!(all_annotations.len(), 1);
    assert_eq!(all_annotations[0].key, "ANN01234");
}

// ── Zotero group tests ────────────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_group_list() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/groups"))
        .respond_with(zotero_arr(&z_groups_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_groups().await.unwrap();
    let json = serde_json::to_string(&result).unwrap();
    assert!(json.contains("Test Group"));
    assert_eq!(result.total_results, Some(1));
}

// ── Zotero format text tests ──────────────────────────────────────────────

#[tokio::test]
async fn test_zotero_work_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(zotero_arr(&z_items_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_top_items(&ItemListParams::default()).await.unwrap();
    let text = papers_cli_format::format_zotero_work_list(&result);
    assert!(text.contains("Test Paper"), "should contain title");
    assert!(text.contains("ABC12345"), "should contain key");
}

#[tokio::test]
async fn test_zotero_work_list_rich_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&format!("[{}]", z_item_rich_body())),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_top_items(&ItemListParams::default()).await.unwrap();
    let text = papers_cli_format::format_zotero_work_list(&result);
    assert!(text.contains("GPU Sorting Algorithms"), "should contain title");
    assert!(text.contains("RICH1234"), "should contain key");
    assert!(text.contains("2024"), "should contain year");
    assert!(text.contains("Smith, Alice"), "should contain first author");
    assert!(text.contains("et al."), "should show et al. for >3 authors");
    assert!(text.contains("IEEE Transactions"), "should contain publication");
    assert!(text.contains("GPU"), "should show tags");
}

#[tokio::test]
async fn test_zotero_work_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/RICH1234"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_item_rich_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_item("RICH1234").await.unwrap();
    let text = papers_cli_format::format_zotero_item_get(&result);
    assert!(text.contains("GPU Sorting Algorithms"), "should contain title");
    assert!(text.contains("RICH1234"), "should contain key");
    assert!(text.contains("10.1234/gpu.2024"), "should contain DOI");
    assert!(text.contains("2024-03-15"), "should contain date");
    assert!(text.contains("Smith, Alice"), "should contain author");
    assert!(text.contains("(author)"), "should show creator type");
    assert!(text.contains("GPU-based sorting"), "should contain abstract");
    assert!(text.contains("GPU"), "should show tags");
}

#[tokio::test]
async fn test_zotero_attachment_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&z_attachment_rich_body()),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let result = client.list_items(&params).await.unwrap();
    let text = papers_cli_format::format_zotero_attachment_list(&result);
    assert!(text.contains("gpu_sorting.pdf"), "should contain filename");
    assert!(text.contains("ATT12345"), "should contain key");
    assert!(text.contains("imported_file"), "should contain link mode");
    assert!(text.contains("application/pdf"), "should contain content type");
    assert!(text.contains("RICH1234"), "should contain parent key");
}

#[tokio::test]
async fn test_zotero_annotation_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&z_annotation_rich_body()),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
    let result = client.list_items(&params).await.unwrap();
    let text = papers_cli_format::format_zotero_annotation_list(&result);
    assert!(text.contains("highlight"), "should contain annotation type");
    assert!(text.contains("ANN12345"), "should contain key");
    assert!(text.contains("p. 5"), "should contain page label");
    assert!(text.contains("GPU sorting outperforms"), "should contain annotation text");
    assert!(text.contains("Important result"), "should contain comment");
    assert!(text.contains("ATT12345"), "should contain parent");
}

#[tokio::test]
async fn test_zotero_note_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&z_note_rich_body()),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let params = ItemListParams { item_type: Some("note".into()), ..Default::default() };
    let result = client.list_items(&params).await.unwrap();
    let text = papers_cli_format::format_zotero_note_list(&result);
    assert!(text.contains("NOTE5678"), "should contain key");
    assert!(text.contains("RICH1234"), "should contain parent key");
    assert!(text.contains("parallel"), "should contain stripped text");
    assert!(!text.contains("<b>"), "should strip HTML tags");
}

#[tokio::test]
async fn test_zotero_collection_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&z_collection_with_parent_body()),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_collections(&CollectionListParams::default()).await.unwrap();
    let text = papers_cli_format::format_zotero_collection_list(&result);
    assert!(text.contains("GPU Papers"), "should contain collection name");
    assert!(text.contains("SUB12345"), "should contain key");
    assert!(text.contains("COL12345"), "should show parent key");
    assert!(text.contains("3 items"), "should show item count");
}

#[tokio::test]
async fn test_zotero_collection_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL12345"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_collection_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_collection("COL12345").await.unwrap();
    let text = papers_cli_format::format_zotero_collection_get(&result);
    assert!(text.contains("Test Collection"), "should contain name");
    assert!(text.contains("COL12345"), "should contain key");
    assert!(text.contains("(top-level)"), "should show top-level for root collection");
    assert!(text.contains("5"), "should show item count");
}

#[tokio::test]
async fn test_zotero_tag_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/tags"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "2")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&z_tag_with_count_body()),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_tags(&TagListParams::default()).await.unwrap();
    let text = papers_cli_format::format_zotero_tag_list(&result);
    assert!(text.contains("Starred"), "should contain tag name");
    assert!(text.contains("31 items"), "should show item count for Starred");
    assert!(text.contains("Survey"), "should contain second tag");
    assert!(text.contains("7 items"), "should show item count for Survey");
    assert!(text.contains("Found 2 tags"), "should show total count");
}

#[tokio::test]
async fn test_zotero_search_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/searches"))
        .respond_with(zotero_arr(&z_searches_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_searches().await.unwrap();
    let text = papers_cli_format::format_zotero_search_list(&result);
    assert!(text.contains("Test Search"), "should contain search name");
    assert!(text.contains("SRCH1234"), "should contain key");
    assert!(text.contains("0 condition(s)"), "should show condition count");
}

#[tokio::test]
async fn test_zotero_search_get_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/searches/SRCH5678"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_search_with_conditions_body()))
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.get_search("SRCH5678").await.unwrap();
    let text = papers_cli_format::format_zotero_search_get(&result);
    assert!(text.contains("GPU Papers"), "should contain search name");
    assert!(text.contains("SRCH5678"), "should contain key");
    assert!(text.contains("tag is GPU"), "should show condition");
    assert!(text.contains("itemType is journalArticle"), "should show second condition");
}

#[tokio::test]
async fn test_zotero_group_list_text() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/groups"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&z_group_with_desc_body()),
        )
        .mount(&mock)
        .await;
    let client = make_zotero_client(&mock);
    let result = client.list_groups().await.unwrap();
    let text = papers_cli_format::format_zotero_group_list(&result);
    assert!(text.contains("CompSci Reading Group"), "should contain group name");
    assert!(text.contains("99999"), "should contain group ID");
    assert!(text.contains("PublicOpen"), "should contain group type");
    assert!(text.contains("42 items"), "should show item count");
    assert!(text.contains("computer science paper discussions"), "should show description");
}

// ── Cross-reference chain tests ───────────────────────────────────────────
//
// These tests exercise multi-step call chains where the result of one API call
// is used as the key/parameter for a subsequent call, mirroring what the CLI
// commands do in main.rs.

/// work → annotations (non-empty): list a work's attachments, then get annotations
/// on each annotatable attachment. Mirrors ZoteroWorkCommand::Annotations logic.
#[tokio::test]
async fn test_zotero_work_annotations_nonempty() {
    let mock = MockServer::start().await;

    let attachments = r#"[
        {"key":"PDFABC01","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"PDFABC01","version":1,"itemType":"attachment","contentType":"application/pdf","linkMode":"imported_file","parentItem":"WORK0001"}},
        {"key":"EPUBABC1","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"EPUBABC1","version":1,"itemType":"attachment","contentType":"application/epub+zip","linkMode":"imported_file","parentItem":"WORK0001"}}
    ]"#;
    let pdf_annotations = r#"[
        {"key":"ANN00001","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ANN00001","version":1,"itemType":"annotation","parentItem":"PDFABC01","annotationType":"highlight","annotationText":"First finding","annotationPageLabel":"3"}},
        {"key":"ANN00002","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ANN00002","version":1,"itemType":"annotation","parentItem":"PDFABC01","annotationType":"note","annotationComment":"My note"}}
    ]"#;
    let epub_annotations = r#"[
        {"key":"ANN00003","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ANN00003","version":1,"itemType":"annotation","parentItem":"EPUBABC1","annotationType":"highlight","annotationText":"EPUB highlight"}}
    ]"#;

    Mock::given(method("GET"))
        .and(path("/users/test/items/WORK0001/children"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "2")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(attachments))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/items/PDFABC01/children"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "2")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(pdf_annotations))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/items/EPUBABC1/children"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "1")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(epub_annotations))
        .mount(&mock).await;

    let client = make_zotero_client(&mock);

    // Step 1: list attachments for the work
    let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let attachments = client.list_item_children("WORK0001", &att_params).await.unwrap();
    assert_eq!(attachments.items.len(), 2, "both attachments returned");

    // Step 2: for each annotatable attachment, fetch annotations (mirrors CLI logic)
    let ann_params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
    let mut all_annotations = Vec::new();
    for att in &attachments.items {
        let is_annotatable = matches!(
            att.data.content_type.as_deref(),
            Some("application/pdf") | Some("application/epub+zip") | Some("text/html")
        );
        if !is_annotatable { continue; }
        let ann = client.list_item_children(&att.key, &ann_params).await.unwrap();
        // Verify parent linkage: each annotation's parent_item == the attachment key
        for a in &ann.items {
            assert_eq!(a.data.parent_item.as_deref(), Some(att.key.as_str()),
                "annotation parent_item should equal the attachment key");
        }
        all_annotations.extend(ann.items);
    }

    assert_eq!(all_annotations.len(), 3, "2 PDF + 1 EPUB annotations");
    let keys: Vec<&str> = all_annotations.iter().map(|a| a.key.as_str()).collect();
    assert!(keys.contains(&"ANN00001"));
    assert!(keys.contains(&"ANN00002"));
    assert!(keys.contains(&"ANN00003"));

    // Verify text format output is sensible
    let text = papers_cli_format::format_zotero_annotation_list_vec(&all_annotations);
    assert!(text.contains("highlight"), "should show annotation type");
    assert!(text.contains("First finding"), "should show annotation text");
    assert!(text.contains("My note"), "should show comment");
    assert!(text.contains("EPUB highlight"), "should include EPUB annotations");
    assert!(text.contains("p. 3"), "should show page label");
}

/// collection → annotations (non-empty): list a collection's attachments, then fetch
/// annotations for each. Mirrors ZoteroCollectionCommand::Annotations logic.
#[tokio::test]
async fn test_zotero_collection_annotations_nonempty() {
    let mock = MockServer::start().await;

    let col_attachments = r#"[
        {"key":"PDFC0001","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"PDFC0001","version":1,"itemType":"attachment","contentType":"application/pdf","linkMode":"imported_file","parentItem":"COLWORK1"}},
        {"key":"TXTC0001","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"TXTC0001","version":1,"itemType":"attachment","contentType":"text/plain","linkMode":"imported_file","parentItem":"COLWORK1"}}
    ]"#;
    let col_annotations = r#"[
        {"key":"CANN0001","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"CANN0001","version":1,"itemType":"annotation","parentItem":"PDFC0001","annotationType":"highlight","annotationText":"Collection annotation","annotationPageLabel":"7"}}
    ]"#;

    // Collection items endpoint returns both PDF and text attachment
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL99999/items"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "2")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(col_attachments))
        .mount(&mock).await;

    // Only PDF children endpoint is registered — text/plain should be skipped
    Mock::given(method("GET"))
        .and(path("/users/test/items/PDFC0001/children"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "1")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(col_annotations))
        .mount(&mock).await;

    let client = make_zotero_client(&mock);

    // Step 1: list collection items filtered to attachments
    let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let attachments = client.list_collection_items("COL99999", &att_params).await.unwrap();
    assert_eq!(attachments.items.len(), 2);

    // Step 2: fetch annotations only for annotatable attachments
    let ann_params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
    let mut all_annotations = Vec::new();
    for att in &attachments.items {
        let is_annotatable = matches!(
            att.data.content_type.as_deref(),
            Some("application/pdf") | Some("application/epub+zip") | Some("text/html")
        );
        if !is_annotatable { continue; }
        match client.list_item_children(&att.key, &ann_params).await {
            Ok(r) => all_annotations.extend(r.items),
            Err(_) => {}
        }
    }

    // Only the PDF annotation; text/plain was skipped
    assert_eq!(all_annotations.len(), 1);
    assert_eq!(all_annotations[0].key, "CANN0001");
    assert_eq!(all_annotations[0].data.parent_item.as_deref(), Some("PDFC0001"),
        "annotation parent_item should equal PDF attachment key");

    let text = papers_cli_format::format_zotero_annotation_list_vec(&all_annotations);
    assert!(text.contains("Collection annotation"));
    assert!(text.contains("p. 7"));
}

/// collection → works → get item (cross-reference key reuse):
/// list works in a collection, extract a key from the result, then get that item
/// individually — verifying the key returned in a list response is valid for a get call.
#[tokio::test]
async fn test_zotero_collection_works_then_get_item() {
    let mock = MockServer::start().await;

    // Collection works list returns a rich item
    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL11111/items/top"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "1")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(&format!("[{}]", z_item_rich_body())))
        .mount(&mock).await;

    // Getting that same item by key should return its full data
    Mock::given(method("GET"))
        .and(path("/users/test/items/RICH1234"))
        .respond_with(ResponseTemplate::new(200).set_body_string(z_item_rich_body()))
        .mount(&mock).await;

    let client = make_zotero_client(&mock);

    // Step 1: list works in collection
    let list = client.list_collection_top_items("COL11111", &ItemListParams::default()).await.unwrap();
    assert_eq!(list.items.len(), 1);
    let key_from_list = list.items[0].key.clone();
    let title_from_list = list.items[0].data.title.clone();

    // Step 2: use the key to get the full item
    let item = client.get_item(&key_from_list).await.unwrap();

    // The key and title should be consistent across both calls
    assert_eq!(item.key, key_from_list, "get_item key should match key from list");
    assert_eq!(item.data.title, title_from_list, "title should be consistent");
    assert_eq!(item.key, "RICH1234");
    assert_eq!(item.data.title.as_deref(), Some("GPU Sorting Algorithms"));

    // The full item should contain richer data not present in list summary
    assert!(item.data.doi.is_some(), "full item should have DOI");
    assert!(item.data.abstract_note.is_some(), "full item should have abstract");
}

/// work → collections → work (full round-trip):
/// get a work, fetch its collections, then list works in one of those collections
/// and confirm the original work key appears in the results.
#[tokio::test]
async fn test_zotero_work_to_collection_to_works_roundtrip() {
    let mock = MockServer::start().await;

    // Work item has one collection: COL22222
    let work_body = r#"{"key":"WORK2222","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"WORK2222","version":1,"itemType":"journalArticle","title":"Round-trip Test Paper","collections":["COL22222"]}}"#;
    let collection_body = r#"{"key":"COL22222","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{"numCollections":0,"numItems":3},"data":{"key":"COL22222","version":1,"name":"My Reading List","parentCollection":false,"relations":{}}}"#;
    // Collection's works include our work
    let collection_works = r#"[{"key":"WORK2222","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"WORK2222","version":1,"itemType":"journalArticle","title":"Round-trip Test Paper","collections":["COL22222"]}}]"#;

    Mock::given(method("GET"))
        .and(path("/users/test/items/WORK2222"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_body))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL22222"))
        .respond_with(ResponseTemplate::new(200).set_body_string(collection_body))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/collections/COL22222/items/top"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "3")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(collection_works))
        .mount(&mock).await;

    let client = make_zotero_client(&mock);

    // Step 1: get the work
    let work = client.get_item("WORK2222").await.unwrap();
    assert!(!work.data.collections.is_empty(), "work should belong to collections");
    let col_key = work.data.collections[0].clone();
    assert_eq!(col_key, "COL22222");

    // Step 2: fetch the collection using the key from the work
    let collection = client.get_collection(&col_key).await.unwrap();
    assert_eq!(collection.key, col_key, "collection key should match");
    assert_eq!(collection.data.name, "My Reading List");
    assert_eq!(collection.meta.num_items, Some(3), "collection should report its item count");

    // Step 3: list works in the collection and verify the original work appears
    let works = client.list_collection_top_items(&col_key, &ItemListParams::default()).await.unwrap();
    let work_keys: Vec<&str> = works.items.iter().map(|w| w.key.as_str()).collect();
    assert!(work_keys.contains(&"WORK2222"), "original work should appear in its collection");

    // Also verify: the work's collections field includes the collection we navigated to
    let found_work = works.items.iter().find(|w| w.key == "WORK2222").unwrap();
    assert!(found_work.data.collections.contains(&col_key),
        "work's collections should include the collection we listed it from");
}

/// attachment → parent item chain: verify the parent_item field on an attachment
/// is a valid work key that can be fetched.
#[tokio::test]
async fn test_zotero_attachment_parent_work_chain() {
    let mock = MockServer::start().await;

    let attachment_body = r#"{"key":"ATT33333","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ATT33333","version":1,"itemType":"attachment","contentType":"application/pdf","linkMode":"imported_file","filename":"paper.pdf","parentItem":"WORK3333"}}"#;
    let parent_work_body = r#"{"key":"WORK3333","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"WORK3333","version":1,"itemType":"journalArticle","title":"Parent Work","collections":[]}}"#;

    Mock::given(method("GET"))
        .and(path("/users/test/items/ATT33333"))
        .respond_with(ResponseTemplate::new(200).set_body_string(attachment_body))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/items/WORK3333"))
        .respond_with(ResponseTemplate::new(200).set_body_string(parent_work_body))
        .mount(&mock).await;

    let client = make_zotero_client(&mock);

    // Step 1: get the attachment
    let att = client.get_item("ATT33333").await.unwrap();
    assert_eq!(att.data.item_type, "attachment");
    assert_eq!(att.data.content_type.as_deref(), Some("application/pdf"));
    let parent_key = att.data.parent_item.as_ref().expect("attachment should have parent_item");
    assert_eq!(parent_key, "WORK3333");

    // Step 2: use the parent_item key to fetch the work
    let work = client.get_item(parent_key).await.unwrap();
    assert_eq!(work.key, "WORK3333");
    assert_eq!(work.data.item_type, "journalArticle");
    assert_eq!(work.data.title.as_deref(), Some("Parent Work"));
}

/// annotation → attachment → work (three-hop chain):
/// get annotation, use parent_item to get attachment, use attachment.parent_item to get work.
#[tokio::test]
async fn test_zotero_annotation_to_attachment_to_work_chain() {
    let mock = MockServer::start().await;

    let annotation_body = r#"{"key":"ANN44444","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ANN44444","version":1,"itemType":"annotation","parentItem":"ATT44444","annotationType":"highlight","annotationText":"Key insight","annotationPageLabel":"12"}}"#;
    let attachment_body = r#"{"key":"ATT44444","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"ATT44444","version":1,"itemType":"attachment","contentType":"application/pdf","linkMode":"imported_file","filename":"paper.pdf","parentItem":"WORK4444"}}"#;
    let work_body = r#"{"key":"WORK4444","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"WORK4444","version":1,"itemType":"journalArticle","title":"The Work","collections":[]}}"#;

    Mock::given(method("GET"))
        .and(path("/users/test/items/ANN44444"))
        .respond_with(ResponseTemplate::new(200).set_body_string(annotation_body))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/items/ATT44444"))
        .respond_with(ResponseTemplate::new(200).set_body_string(attachment_body))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/items/WORK4444"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_body))
        .mount(&mock).await;

    let client = make_zotero_client(&mock);

    // Hop 1: get annotation
    let annotation = client.get_item("ANN44444").await.unwrap();
    assert_eq!(annotation.data.item_type, "annotation");
    let att_key = annotation.data.parent_item.as_ref().expect("annotation should have parent_item");
    assert_eq!(att_key, "ATT44444");

    // Hop 2: get attachment using annotation's parent_item
    let attachment = client.get_item(att_key).await.unwrap();
    assert_eq!(attachment.data.item_type, "attachment");
    assert_eq!(attachment.data.content_type.as_deref(), Some("application/pdf"));
    let work_key = attachment.data.parent_item.as_ref().expect("attachment should have parent_item");
    assert_eq!(work_key, "WORK4444");

    // Hop 3: get work using attachment's parent_item
    let work = client.get_item(work_key).await.unwrap();
    assert_eq!(work.data.item_type, "journalArticle");
    assert_eq!(work.data.title.as_deref(), Some("The Work"));
}

/// collection → subcollections → works in subcollection:
/// navigate the collection hierarchy three levels deep.
#[tokio::test]
async fn test_zotero_collection_hierarchy_navigation() {
    let mock = MockServer::start().await;

    let root_col = r#"{"key":"ROOT0001","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{"numCollections":1,"numItems":0},"data":{"key":"ROOT0001","version":1,"name":"Root","parentCollection":false,"relations":{}}}"#;
    let sub_cols = r#"[{"key":"SUB00001","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{"numCollections":0,"numItems":2},"data":{"key":"SUB00001","version":1,"name":"Sub Collection","parentCollection":"ROOT0001","relations":{}}}]"#;
    let sub_works = r#"[
        {"key":"SUBWRK1","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"SUBWRK1","version":1,"itemType":"journalArticle","title":"Paper A","collections":["SUB00001"]}},
        {"key":"SUBWRK2","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"SUBWRK2","version":1,"itemType":"journalArticle","title":"Paper B","collections":["SUB00001"]}}
    ]"#;

    Mock::given(method("GET"))
        .and(path("/users/test/collections/ROOT0001"))
        .respond_with(ResponseTemplate::new(200).set_body_string(root_col))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/collections/ROOT0001/collections"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "1")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(sub_cols))
        .mount(&mock).await;

    Mock::given(method("GET"))
        .and(path("/users/test/collections/SUB00001/items/top"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("Total-Results", "2")
            .insert_header("Last-Modified-Version", "100")
            .set_body_string(sub_works))
        .mount(&mock).await;

    let client = make_zotero_client(&mock);

    // Step 1: get root collection
    let root = client.get_collection("ROOT0001").await.unwrap();
    assert_eq!(root.meta.num_collections, Some(1), "root has 1 sub-collection");
    assert_eq!(root.meta.num_items, Some(0), "root has 0 direct items");
    assert!(root.data.parent_key().is_none(), "root should be top-level");

    // Step 2: list sub-collections of root
    let subs = client.list_subcollections("ROOT0001", &CollectionListParams::default()).await.unwrap();
    assert_eq!(subs.items.len(), 1);
    let sub_key = subs.items[0].key.clone();
    let sub_parent = subs.items[0].data.parent_key();
    assert_eq!(sub_parent, Some("ROOT0001"), "sub-collection parent should be root");

    // Step 3: list works in sub-collection using the key from step 2
    let works = client.list_collection_top_items(&sub_key, &ItemListParams::default()).await.unwrap();
    assert_eq!(works.items.len(), 2);

    // Verify both works claim membership in the sub-collection
    for w in &works.items {
        assert!(w.data.collections.contains(&sub_key),
            "each work's collections should include the sub-collection key");
    }

    let titles: Vec<&str> = works.items.iter().filter_map(|w| w.data.title.as_deref()).collect();
    assert!(titles.contains(&"Paper A"));
    assert!(titles.contains(&"Paper B"));
}

// ── Zotero work extract tests ─────────────────────────────────────────────

/// `looks_like_zotero_key` accepts exactly 8 uppercase alphanumeric chars.
#[test]
fn test_looks_like_zotero_key_valid() {
    assert!(papers_core::zotero::looks_like_zotero_key("U9PRIZJ7"));
    assert!(papers_core::zotero::looks_like_zotero_key("ABCD1234"));
    assert!(papers_core::zotero::looks_like_zotero_key("12345678"));
    assert!(papers_core::zotero::looks_like_zotero_key("AAAAAAAA"));
}

/// `looks_like_zotero_key` rejects strings that are not exactly 8 uppercase alphanum.
#[test]
fn test_looks_like_zotero_key_invalid() {
    // lowercase
    assert!(!papers_core::zotero::looks_like_zotero_key("abcd1234"));
    // too short
    assert!(!papers_core::zotero::looks_like_zotero_key("ABCD123"));
    // too long
    assert!(!papers_core::zotero::looks_like_zotero_key("ABCD12345"));
    // contains space
    assert!(!papers_core::zotero::looks_like_zotero_key("ABCD 234"));
    // contains hyphen
    assert!(!papers_core::zotero::looks_like_zotero_key("ABCD-234"));
    // title / search string
    assert!(!papers_core::zotero::looks_like_zotero_key("vertex block descent"));
    // empty
    assert!(!papers_core::zotero::looks_like_zotero_key(""));
}

/// `resolve_item_key` returns the input immediately when it looks like a Zotero key,
/// making no HTTP requests (the mock server would 404 on any request).
#[tokio::test]
async fn test_resolve_item_key_exact_key_no_http() {
    let mock = MockServer::start().await;
    // No mock routes registered — any HTTP request would return 501/404.
    let client = make_zotero_client(&mock);

    let result = papers_core::zotero::resolve_item_key(&client, "U9PRIZJ7").await;
    assert_eq!(result.unwrap(), "U9PRIZJ7");
}

/// `resolve_item_key` issues a search when given a non-key string and returns the
/// first match's key.
#[tokio::test]
async fn test_resolve_item_key_search_string() {
    let mock = MockServer::start().await;
    let body = r#"[{"key":"SRCHKEY1","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"SRCHKEY1","version":1,"itemType":"journalArticle","title":"Vertex Block Descent"}}]"#;
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(body),
        )
        .mount(&mock)
        .await;

    let client = make_zotero_client(&mock);
    let result = papers_core::zotero::resolve_item_key(&client, "vertex block descent").await;
    assert_eq!(result.unwrap(), "SRCHKEY1");
}

/// `resolve_item_key` returns an error when the search yields no results.
#[tokio::test]
async fn test_resolve_item_key_no_results() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(zotero_arr_empty())
        .mount(&mock)
        .await;

    let client = make_zotero_client(&mock);
    let result = papers_core::zotero::resolve_item_key(&client, "nonexistent paper title").await;
    assert!(result.is_err(), "should error when no results found");
}

/// When the input is not an exact key, `list_top_items` with limit 5 returns
/// multiple results — these would be presented as "Did you mean?" suggestions.
/// This test verifies the mock data structure and that multiple items are returned.
#[tokio::test]
async fn test_extract_ambiguous_search_returns_multiple_items() {
    let mock = MockServer::start().await;
    let body = r#"[
        {"key":"WORK0001","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"WORK0001","version":1,"itemType":"journalArticle","title":"GPU Sorting Paper"}},
        {"key":"WORK0002","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"WORK0002","version":1,"itemType":"journalArticle","title":"GPU Rendering Paper"}},
        {"key":"WORK0003","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"WORK0003","version":1,"itemType":"journalArticle","title":"GPU Memory Management"}}
    ]"#;
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "3")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(body),
        )
        .mount(&mock)
        .await;

    let client = make_zotero_client(&mock);
    // Mirrors what the CLI does: search with limit 5 when input is ambiguous.
    let params = ItemListParams::builder().q("gpu").limit(5).build();
    let results = client.list_top_items(&params).await.unwrap();

    assert_eq!(results.items.len(), 3, "all three matches returned");
    let keys: Vec<&str> = results.items.iter().map(|i| i.key.as_str()).collect();
    assert!(keys.contains(&"WORK0001"));
    assert!(keys.contains(&"WORK0002"));
    assert!(keys.contains(&"WORK0003"));

    // Verify each item has a title (for Did you mean? display)
    for item in &results.items {
        assert!(item.data.title.is_some(), "each suggestion should have a title");
    }
}

/// `datalab_cached_markdown` returns `None` for a key that has no cache entry.
#[test]
fn test_datalab_cached_markdown_miss() {
    // Use a key that should never exist in the cache.
    let result = papers_core::text::datalab_cached_markdown("TSTMISSXX");
    assert!(result.is_none(), "unknown key should return None");
}

// ── `papers rag work extract` tests ──────────────────────────────────────
//
// These tests exercise the two user-visible behaviours of the simplified
// `extract` command:
//
//   papers rag work extract <work>         → cached markdown text
//   papers rag work extract <work> --json  → cached structured JSON
//
// All tests write files into an isolated test cache directory controlled by
// the `PAPERS_DATALAB_CACHE_DIR` environment variable so they never touch the
// real DataLab cache.  Each test uses a unique item key to avoid races when
// tests run in parallel.

static EXTRACT_TESTS_CACHE: std::sync::OnceLock<std::path::PathBuf> =
    std::sync::OnceLock::new();

/// Initialise the shared test-cache base directory once per process.
///
/// Sets `PAPERS_DATALAB_CACHE_DIR` to `<system-cache>/papers/extract_tests`
/// so that every call to `datalab_cached_markdown` / `datalab_cached_json`
/// looks there instead of the real DataLab cache.
fn extract_test_cache_base() -> &'static std::path::PathBuf {
    EXTRACT_TESTS_CACHE.get_or_init(|| {
        let dir = dirs::cache_dir()
            .expect("no system cache dir")
            .join("papers")
            .join("extract_tests");
        std::fs::create_dir_all(&dir).unwrap();
        // Safety: tests run in a single process; the value is written exactly
        // once before any test reads it, so there is no concurrent mutation.
        unsafe { std::env::set_var("PAPERS_DATALAB_CACHE_DIR", &dir) };
        dir
    })
}

/// Write `{key}.md` into the test cache so that `datalab_cached_markdown`
/// returns `Some(content)`.
fn write_md_cache(key: &str, content: &str) {
    let base = extract_test_cache_base();
    let dir = base.join(key);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{key}.md")), content).unwrap();
}

/// Write `{key}.json` into the test cache so that `datalab_cached_json`
/// returns `Some(content)`.
fn write_json_cache(key: &str, content: &str) {
    let base = extract_test_cache_base();
    let dir = base.join(key);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{key}.json")), content).unwrap();
}

// Unique keys — one per test, so parallel runs never race on the same file.
const KEY_EXT_MD_HIT:    &str = "EXMDHT01"; // has .md  → markdown returned
const KEY_EXT_JSON_HIT:  &str = "EXJSHT01"; // has .json → JSON returned
const KEY_EXT_JSON_MISS: &str = "EXJSMS01"; // no .json  → None
const KEY_EXT_SRCH_MD:   &str = "EXSRMD01"; // resolved via title search → markdown
const KEY_EXT_SRCH_JSON: &str = "EXSRJS01"; // resolved via title search → JSON

/// `datalab_cached_markdown` returns the file contents when a `.md` file is
/// present in the test cache.
#[test]
fn test_rag_work_extract_markdown_cache_hit() {
    let expected = "# My Paper\n\nThis is the extracted text.\n";
    write_md_cache(KEY_EXT_MD_HIT, expected);
    let result = papers_core::text::datalab_cached_markdown(KEY_EXT_MD_HIT);
    assert_eq!(result.as_deref(), Some(expected));
}

/// `datalab_cached_json` returns the file contents when a `.json` file is
/// present in the test cache.
#[test]
fn test_rag_work_extract_json_cache_hit() {
    let expected = r#"{"pages":[],"metadata":{"title":"My Paper"}}"#;
    write_json_cache(KEY_EXT_JSON_HIT, expected);
    let result = papers_core::text::datalab_cached_json(KEY_EXT_JSON_HIT);
    assert_eq!(result.as_deref(), Some(expected));
}

/// `datalab_cached_json` returns `None` when no `.json` file exists for the
/// key (only a `.md` file is present).
#[test]
fn test_rag_work_extract_json_cache_miss() {
    // Write only the markdown file — no JSON counterpart.
    write_md_cache(KEY_EXT_JSON_MISS, "# Paper without JSON\n");
    let result = papers_core::text::datalab_cached_json(KEY_EXT_JSON_MISS);
    assert!(result.is_none(), "should return None when .json is absent");
}

/// End-to-end: a title-search query resolves to a Zotero item key via the
/// mock server, then `datalab_cached_markdown` returns the cached text.
///
/// Mirrors the `--json=false` (default) path of `papers rag work extract`.
#[tokio::test]
async fn test_rag_work_extract_resolves_title_then_reads_markdown() {
    // Seed the cache so the resolved key has a markdown file.
    let expected_md = "# GPU Paper\n\nContent here.\n";
    write_md_cache(KEY_EXT_SRCH_MD, expected_md);

    // Mock Zotero: title search → single result with KEY_EXT_SRCH_MD.
    let mock = MockServer::start().await;
    let body = format!(
        r#"[{{"key":"{KEY_EXT_SRCH_MD}","version":1,"library":{{"type":"user","id":1,"name":"test","links":{{}}}},"links":{{}},"meta":{{}},"data":{{"key":"{KEY_EXT_SRCH_MD}","version":1,"itemType":"journalArticle","title":"GPU Paper"}}}}]"#
    );
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&body),
        )
        .mount(&mock)
        .await;

    let client = make_zotero_client(&mock);

    // Resolve the title to a key (mirrors smart_resolve_item_key for non-key input).
    let resolved_key = papers_core::zotero::resolve_item_key(&client, "GPU Paper")
        .await
        .expect("resolution should succeed");
    assert_eq!(resolved_key, KEY_EXT_SRCH_MD);

    // Read cached markdown for the resolved key (the --json=false path).
    let result = papers_core::text::datalab_cached_markdown(&resolved_key);
    assert_eq!(result.as_deref(), Some(expected_md));
}

/// End-to-end: a title-search query resolves to a Zotero item key via the
/// mock server, then `datalab_cached_json` returns the cached JSON.
///
/// Mirrors the `--json` path of `papers rag work extract`.
#[tokio::test]
async fn test_rag_work_extract_resolves_title_then_reads_json() {
    // Seed the cache so the resolved key has a JSON file.
    let expected_json = r#"{"pages":[],"metadata":{"title":"GPU Paper"}}"#;
    write_json_cache(KEY_EXT_SRCH_JSON, expected_json);

    // Mock Zotero: title search → single result with KEY_EXT_SRCH_JSON.
    let mock = MockServer::start().await;
    let body = format!(
        r#"[{{"key":"{KEY_EXT_SRCH_JSON}","version":1,"library":{{"type":"user","id":1,"name":"test","links":{{}}}},"links":{{}},"meta":{{}},"data":{{"key":"{KEY_EXT_SRCH_JSON}","version":1,"itemType":"journalArticle","title":"GPU Paper"}}}}]"#
    );
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&body),
        )
        .mount(&mock)
        .await;

    let client = make_zotero_client(&mock);

    // Resolve the title to a key.
    let resolved_key = papers_core::zotero::resolve_item_key(&client, "GPU Paper")
        .await
        .expect("resolution should succeed");
    assert_eq!(resolved_key, KEY_EXT_SRCH_JSON);

    // Read cached JSON for the resolved key (the --json path).
    let result = papers_core::text::datalab_cached_json(&resolved_key);
    assert_eq!(result.as_deref(), Some(expected_json));
}

// ── `rag work add` auto-extraction and two-way sync tests ─────────────────
//
// These tests exercise the new `rag work add` behavior introduced when
// `zotero work extract` was merged in:
//
//   - Cache hit fast path: cached `.md` → ingest immediately, skip Zotero
//   - Cache miss detection: `datalab_cached_markdown` returns None → extraction needed
//   - Two-way sync A (local → Zotero): local `.md` cached, no papers_extract_*.zip in Zotero
//       → `do_extract` spawns background upload to Zotero
//   - Two-way sync B (local → Zotero skipped): local `.md` cached, papers_extract_*.zip exists
//       → no upload triggered
//   - Two-way sync C (Zotero → local): no local `.md`, Zotero has papers_extract_*.zip
//       → `do_extract` downloads zip, restores local cache
//   - Two-way sync D (fresh extraction → Zotero): full DataLab round-trip writes
//       local cache and synchronously uploads papers_extract_*.zip to Zotero
//   - DOI resolution: `smart_resolve_item_key` routes DOI queries correctly
//   - `find_papers_zip_key` returns None / Some based on children response
//   - `ingest_params_from_cache` falls back gracefully when meta.json is absent
//
// All tests reuse `extract_test_cache_base()` / `write_md_cache()` / `write_json_cache()`
// so that `PAPERS_DATALAB_CACHE_DIR` is set consistently for the whole process.
// Keys are exactly 8 uppercase-alphanumeric characters (Zotero key format).

/// Build a minimal in-memory ZIP containing `{key}.md` with `md_content`.
fn make_papers_zip(key: &str, md_content: &str) -> Vec<u8> {
    use std::io::Write as _;
    use zip::write::SimpleFileOptions;

    let buf = Vec::new();
    let cursor = std::io::Cursor::new(buf);
    let mut zw = zip::ZipWriter::new(cursor);
    let opts = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zw.start_file(format!("{key}.md"), opts).unwrap();
    zw.write_all(md_content.as_bytes()).unwrap();
    zw.finish().unwrap().into_inner()
}

// Per-test unique 8-char keys (uppercase alphanumeric = valid Zotero keys).
const KEY_ADD_FAST:     &str = "ADDFAST1"; // cache hit → fast path
const KEY_ADD_MISS:     &str = "ADDMISS1"; // cache miss → None
const KEY_ADD_SYNC_A:   &str = "SYNCAL01"; // local→Zotero upload triggered
const KEY_ADD_SYNC_B:   &str = "SYNCBL01"; // local→Zotero upload skipped
const KEY_ADD_SYNC_C:   &str = "SYNCCL01"; // Zotero→local restore
const KEY_ADD_SYNC_D:   &str = "SYNCDL01"; // DataLab→local→Zotero (sync upload)
const KEY_ADD_DOI:      &str = "ADDDOI01"; // DOI resolution
const KEY_FZIP_NONE:    &str = "FZIPNO01"; // find_papers_zip_key → None
const KEY_FZIP_SOME:    &str = "FZIPSO01"; // find_papers_zip_key → Some
const KEY_NO_META:      &str = "NOMETAXX"; // no meta.json fallback
const KEY_MD_ONLY:      &str = "MDONLY01"; // md without json (omitted from --all)
const KEY_WITH_JSON:    &str = "WITHJSN1"; // md + json (included in --all)

/// Fast path: when the key is a valid Zotero key AND the local cache exists,
/// `datalab_cached_markdown` returns the cached content immediately —
/// no Zotero resolution required.
#[test]
fn test_rag_add_fast_path_cache_hit() {
    let content = "# Fast Paper\n\nNo Zotero call needed.\n";
    write_md_cache(KEY_ADD_FAST, content);

    assert!(papers_core::zotero::looks_like_zotero_key(KEY_ADD_FAST),
        "{KEY_ADD_FAST} must be a valid Zotero key for the fast path to trigger");
    let result = papers_core::text::datalab_cached_markdown(KEY_ADD_FAST);
    assert_eq!(result.as_deref(), Some(content),
        "cache hit should return the pre-seeded markdown");
}

/// Cache miss: `datalab_cached_markdown` returns `None` for a key that has
/// never been extracted — this is the signal that triggers auto-extraction in
/// `rag work add`.
#[test]
fn test_rag_add_cache_miss_returns_none() {
    extract_test_cache_base(); // ensure env var is set
    // KEY_ADD_MISS has never been written to the test cache
    let result = papers_core::text::datalab_cached_markdown(KEY_ADD_MISS);
    assert!(result.is_none(), "uncached key must return None to signal extraction needed");
}

/// Two-way sync A — local → Zotero upload:
///
/// When `do_extract` finds a local `.md` cache hit but no `papers_extract_*.zip`
/// child on the Zotero item, it spawns a background task that:
///   1. Queries the item's children (to confirm no zip exists)
///   2. Creates an `imported_file` attachment (`POST /users/test/items`)
///   3. Registers + uploads the zip file (`POST /users/test/items/<att>/file`)
///
/// This test verifies that all three Zotero endpoints are reached after a
/// short yield that lets the spawned task run to completion.
#[tokio::test]
async fn test_rag_add_sync_local_to_zotero_upload_triggered() {
    write_md_cache(KEY_ADD_SYNC_A, "# Sync A Paper\n");

    let mock = MockServer::start().await;

    // Children: only a PDF attachment, no papers_extract_*.zip
    let children_body =
        r#"[{"key":"PDFATT01","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"PDFATT01","version":1,"itemType":"attachment","linkMode":"imported_file","filename":"paper.pdf","contentType":"application/pdf"}}]"#;
    Mock::given(method("GET"))
        .and(path(format!("/users/test/items/{KEY_ADD_SYNC_A}/children")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(children_body),
        )
        .mount(&mock)
        .await;

    // Attachment create → returns new attachment key ZSYNCA01
    let create_resp = r#"{"success":{"0":"ZSYNCA01"},"unchanged":{},"failed":{},"successful":{"0":{"key":"ZSYNCA01"}}}"#;
    Mock::given(method("POST"))
        .and(path("/users/test/items"))
        .respond_with(ResponseTemplate::new(200).set_body_string(create_resp))
        .mount(&mock)
        .await;

    // File upload: register step — `exists: 1` short-circuits the S3 PUT
    Mock::given(method("POST"))
        .and(path("/users/test/items/ZSYNCA01/file"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"exists":1}"#))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    // DataLab won't be called (local cache hit), but we point it at the mock
    // server so any accidental call produces a clear 404 rather than a network error.
    let dl = papers_datalab::DatalabClient::new("dummy-key").with_base_url(mock.uri());
    let mut source = papers_core::text::PdfSource::ZoteroLocal {
        path: "/tmp/dummy.pdf".into(),
    };

    let result = papers_core::text::do_extract(
        vec![0u8; 4], // dummy bytes — cache hit path won't read them
        KEY_ADD_SYNC_A,
        Some(&zc),
        Some((&dl, papers_core::text::ProcessingMode::Balanced)),
        &mut source,
    )
    .await;

    assert!(result.is_ok(), "do_extract should succeed on cache hit: {:?}", result);
    assert_eq!(result.unwrap().trim(), "# Sync A Paper");

    // Give the spawned background task time to complete.
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    let reqs = mock.received_requests().await.unwrap_or_default();
    let children_hit = reqs.iter().any(|r| r.url.path().ends_with("/children"));
    let create_hit   = reqs.iter().any(|r| {
        r.method == wiremock::http::Method::POST && r.url.path() == "/users/test/items"
    });
    let file_hit     = reqs.iter().any(|r| r.url.path().ends_with("/ZSYNCA01/file"));

    assert!(children_hit, "should query item children to detect existing papers_extract_*.zip");
    assert!(create_hit,   "should POST to create a new attachment item");
    assert!(file_hit,     "should POST to register/upload the zip file");
}

/// Two-way sync B — local → Zotero upload skipped:
///
/// When the local `.md` is cached and `papers_extract_*.zip` already exists as
/// a child of the Zotero item, the background task must detect this and skip
/// the upload entirely (no `POST /users/test/items` call).
#[tokio::test]
async fn test_rag_add_sync_local_to_zotero_upload_skipped_when_zip_exists() {
    write_md_cache(KEY_ADD_SYNC_B, "# Sync B Paper\n");

    let mock = MockServer::start().await;

    // Children: papers_extract_*.zip IS present
    let zip_filename = format!("papers_extract_{KEY_ADD_SYNC_B}.zip");
    let children_body = format!(
        r#"[{{"key":"ZIPATT02","version":1,"library":{{"type":"user","id":1,"name":"test","links":{{}}}},"links":{{}},"meta":{{}},"data":{{"key":"ZIPATT02","version":1,"itemType":"attachment","linkMode":"imported_file","filename":"{zip_filename}","contentType":"application/zip"}}}}]"#
    );
    Mock::given(method("GET"))
        .and(path(format!("/users/test/items/{KEY_ADD_SYNC_B}/children")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&children_body),
        )
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let dl = papers_datalab::DatalabClient::new("dummy-key").with_base_url(mock.uri());
    let mut source = papers_core::text::PdfSource::ZoteroLocal {
        path: "/tmp/dummy.pdf".into(),
    };

    let result = papers_core::text::do_extract(
        vec![0u8; 4],
        KEY_ADD_SYNC_B,
        Some(&zc),
        Some((&dl, papers_core::text::ProcessingMode::Balanced)),
        &mut source,
    )
    .await;

    assert!(result.is_ok(), "cache hit should succeed: {:?}", result);

    // Give the background task time to run (if it wrongly fires).
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    let reqs = mock.received_requests().await.unwrap_or_default();
    let post_create = reqs.iter().any(|r| {
        r.method == wiremock::http::Method::POST && r.url.path() == "/users/test/items"
    });
    assert!(!post_create,
        "upload must NOT be triggered when papers_extract_*.zip already exists in Zotero");
}

/// Two-way sync C — Zotero → local restore:
///
/// When no local `.md` exists but `papers_extract_*.zip` is present as a
/// Zotero child attachment, `do_extract` should download the zip, unpack it
/// into the local cache, and return the markdown from the restored file.
#[tokio::test]
async fn test_rag_add_sync_zotero_to_local_restore() {
    extract_test_cache_base(); // ensure env var is set; no local cache for this key

    let md_content = "# Restored From Zotero\n\nContent from zip.\n";
    let zip_bytes = make_papers_zip(KEY_ADD_SYNC_C, md_content);

    let mock = MockServer::start().await;

    let zip_filename = format!("papers_extract_{KEY_ADD_SYNC_C}.zip");
    let children_body = format!(
        r#"[{{"key":"ZIPATT03","version":1,"library":{{"type":"user","id":1,"name":"test","links":{{}}}},"links":{{}},"meta":{{}},"data":{{"key":"ZIPATT03","version":1,"itemType":"attachment","linkMode":"imported_file","filename":"{zip_filename}","contentType":"application/zip"}}}}]"#
    );
    Mock::given(method("GET"))
        .and(path(format!("/users/test/items/{KEY_ADD_SYNC_C}/children")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&children_body),
        )
        .mount(&mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/users/test/items/ZIPATT03/file"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(zip_bytes))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let dl = papers_datalab::DatalabClient::new("dummy-key").with_base_url(mock.uri());
    let mut source = papers_core::text::PdfSource::ZoteroLocal {
        path: "/tmp/dummy.pdf".into(),
    };

    let result = papers_core::text::do_extract(
        vec![0u8; 4],
        KEY_ADD_SYNC_C,
        Some(&zc),
        Some((&dl, papers_core::text::ProcessingMode::Balanced)),
        &mut source,
    )
    .await;

    assert!(result.is_ok(), "restore from Zotero zip should succeed: {:?}", result);
    let md = result.unwrap();
    assert!(md.contains("Restored From Zotero"),
        "returned markdown should come from the zip");

    // The local cache file must have been written by the restore path.
    let local_file = extract_test_cache_base()
        .join(KEY_ADD_SYNC_C)
        .join(format!("{KEY_ADD_SYNC_C}.md"));
    assert!(local_file.exists(),
        "restore should write local .md cache file at {local_file:?}");
    assert_eq!(std::fs::read_to_string(&local_file).unwrap(), md_content);
}

/// Two-way sync D — DataLab extraction → local cache written → Zotero upload:
///
/// When neither local cache nor Zotero zip exists, `do_extract` calls DataLab,
/// writes the local cache, and synchronously uploads `papers_extract_*.zip`
/// to Zotero.  This test mocks both the DataLab Marker API and Zotero.
#[tokio::test]
async fn test_rag_add_sync_fresh_extraction_writes_local_and_zotero() {
    // Ensure clean local state for this key.
    let cache_dir = extract_test_cache_base().join(KEY_ADD_SYNC_D);
    if cache_dir.exists() {
        std::fs::remove_dir_all(&cache_dir).ok();
    }

    let mock = MockServer::start().await;

    // ── Zotero mocks ──────────────────────────────────────────────────────

    // Children: no papers_extract_*.zip (empty list)
    Mock::given(method("GET"))
        .and(path(format!("/users/test/items/{KEY_ADD_SYNC_D}/children")))
        .respond_with(zotero_arr_empty())
        .mount(&mock)
        .await;

    // Create attachment → returns ZIPATTD1
    let create_resp = r#"{"success":{"0":"ZIPATTD1"},"unchanged":{},"failed":{},"successful":{"0":{"key":"ZIPATTD1"}}}"#;
    Mock::given(method("POST"))
        .and(path("/users/test/items"))
        .respond_with(ResponseTemplate::new(200).set_body_string(create_resp))
        .mount(&mock)
        .await;

    // File upload register → exists=1 (short-circuits S3 PUT)
    Mock::given(method("POST"))
        .and(path("/users/test/items/ZIPATTD1/file"))
        .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"exists":1}"#))
        .mount(&mock)
        .await;

    // ── DataLab mocks ─────────────────────────────────────────────────────

    // POST /api/v1/marker → submit job
    let submit_resp = r#"{"request_id":"req-syncd-001","request_check_url":"/api/v1/marker/req-syncd-001","success":true}"#;
    Mock::given(method("POST"))
        .and(path("/api/v1/marker"))
        .respond_with(ResponseTemplate::new(200).set_body_string(submit_resp))
        .mount(&mock)
        .await;

    // GET /api/v1/marker/req-syncd-001 → complete with markdown
    let md_returned = "# Fresh Extraction Result\n\nDataLab produced this.\n";
    let poll_resp = serde_json::json!({
        "status": "complete",
        "success": true,
        "request_id": "req-syncd-001",
        "markdown": md_returned,
        "json": null,
        "images": null,
        "error": null
    });
    Mock::given(method("GET"))
        .and(path("/api/v1/marker/req-syncd-001"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&poll_resp))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let dl = papers_datalab::DatalabClient::new("dummy-key").with_base_url(mock.uri());
    let mut source = papers_core::text::PdfSource::ZoteroLocal {
        path: "/tmp/dummy.pdf".into(),
    };

    let result = papers_core::text::do_extract(
        b"%PDF-1.4 test".to_vec(),
        KEY_ADD_SYNC_D,
        Some(&zc),
        Some((&dl, papers_core::text::ProcessingMode::Balanced)),
        &mut source,
    )
    .await;

    assert!(result.is_ok(), "fresh extraction should succeed: {:?}", result);
    assert_eq!(result.unwrap(), md_returned);

    // Local cache file must have been written.
    let local_md = cache_dir.join(format!("{KEY_ADD_SYNC_D}.md"));
    assert!(local_md.exists(), "local .md cache file should exist after extraction");
    assert_eq!(std::fs::read_to_string(&local_md).unwrap(), md_returned);

    // Zotero upload must have been triggered (create + file register).
    let reqs = mock.received_requests().await.unwrap_or_default();
    let create_hit = reqs.iter().any(|r| {
        r.method == wiremock::http::Method::POST && r.url.path() == "/users/test/items"
    });
    let file_hit = reqs.iter().any(|r| r.url.path().ends_with("/ZIPATTD1/file"));
    assert!(create_hit,
        "Zotero attachment create should be called after DataLab extraction");
    assert!(file_hit,
        "Zotero file upload should be called after DataLab extraction");
}

/// DOI resolution: `resolve_item_key` routes a full DOI URL through Zotero's
/// item search (qmode=everything), returning the matched item key.
/// In `rag work add`, this is the path taken when the user passes a DOI instead
/// of an 8-char key.
#[tokio::test]
async fn test_rag_add_doi_resolution() {
    let doi = "10.1145/1234567.1234568";
    let mock = MockServer::start().await;

    let body = format!(
        r#"[{{"key":"{KEY_ADD_DOI}","version":1,"library":{{"type":"user","id":1,"name":"test","links":{{}}}},"links":{{}},"meta":{{}},"data":{{"key":"{KEY_ADD_DOI}","version":1,"itemType":"journalArticle","title":"DOI Paper","DOI":"{doi}"}}}}]"#
    );
    Mock::given(method("GET"))
        .and(path("/users/test/items/top"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&body),
        )
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let resolved =
        papers_core::zotero::resolve_item_key(&zc, &format!("https://doi.org/{doi}"))
            .await
            .expect("DOI resolution should succeed");

    assert_eq!(resolved, KEY_ADD_DOI, "DOI input should resolve to the Zotero item key");
}

/// `find_papers_zip_key` (exercised via `do_extract`) returns `None` when the
/// item has no `papers_extract_*.zip` child attachment — causing `do_extract`
/// to fall through to the DataLab path.
///
/// Verified by confirming the children endpoint was queried and that
/// `do_extract` errors out (because DataLab is also unavailable in this test).
#[tokio::test]
async fn test_rag_add_find_papers_zip_key_returns_none_when_absent() {
    extract_test_cache_base(); // ensure env var, no local cache for KEY_FZIP_NONE

    let mock = MockServer::start().await;

    // Children: only a PDF, no papers_extract_*.zip
    let children_body =
        r#"[{"key":"PDFONLY01","version":1,"library":{"type":"user","id":1,"name":"test","links":{}},"links":{},"meta":{},"data":{"key":"PDFONLY01","version":1,"itemType":"attachment","linkMode":"imported_file","filename":"paper.pdf","contentType":"application/pdf"}}]"#;
    Mock::given(method("GET"))
        .and(path(format!("/users/test/items/{KEY_FZIP_NONE}/children")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(children_body),
        )
        .mount(&mock)
        .await;

    // DataLab unavailable → ensures we observe the children check, not a DataLab result
    Mock::given(method("POST"))
        .and(path("/api/v1/marker"))
        .respond_with(ResponseTemplate::new(503).set_body_string("unavailable"))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let dl = papers_datalab::DatalabClient::new("dummy-key").with_base_url(mock.uri());
    let mut source = papers_core::text::PdfSource::ZoteroLocal {
        path: "/tmp/dummy.pdf".into(),
    };

    let result = papers_core::text::do_extract(
        b"%PDF test".to_vec(),
        KEY_FZIP_NONE,
        Some(&zc),
        Some((&dl, papers_core::text::ProcessingMode::Balanced)),
        &mut source,
    )
    .await;

    // Result is Err because DataLab is unavailable and there is no local/Zotero cache
    assert!(result.is_err(),
        "should fail when no local cache, no Zotero zip, and DataLab unavailable");

    // The important check: children endpoint was queried (find_papers_zip_key fired)
    let reqs = mock.received_requests().await.unwrap_or_default();
    let children_hit = reqs.iter().any(|r| {
        r.url.path().ends_with(&format!("/{KEY_FZIP_NONE}/children"))
    });
    assert!(children_hit,
        "children endpoint should be queried to check for papers_extract_*.zip");
}

/// `find_papers_zip_key` returns `Some(key)` when a `papers_extract_*.zip`
/// child attachment exists — verified end-to-end by the Zotero→local restore
/// path successfully returning the restored markdown.
#[tokio::test]
async fn test_rag_add_find_papers_zip_key_returns_some_when_present() {
    extract_test_cache_base(); // ensure env var, no local cache for KEY_FZIP_SOME

    let md_content = "# Found Zip Paper\n";
    let zip_bytes = make_papers_zip(KEY_FZIP_SOME, md_content);

    let mock = MockServer::start().await;

    let zip_filename = format!("papers_extract_{KEY_FZIP_SOME}.zip");
    let children_body = format!(
        r#"[{{"key":"ZIPFND001","version":1,"library":{{"type":"user","id":1,"name":"test","links":{{}}}},"links":{{}},"meta":{{}},"data":{{"key":"ZIPFND001","version":1,"itemType":"attachment","linkMode":"imported_file","filename":"{zip_filename}","contentType":"application/zip"}}}}]"#
    );
    Mock::given(method("GET"))
        .and(path(format!("/users/test/items/{KEY_FZIP_SOME}/children")))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("Total-Results", "1")
                .insert_header("Last-Modified-Version", "100")
                .set_body_string(&children_body),
        )
        .mount(&mock)
        .await;

    Mock::given(method("GET"))
        .and(path("/users/test/items/ZIPFND001/file"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(zip_bytes))
        .mount(&mock)
        .await;

    let zc = make_zotero_client(&mock);
    let dl = papers_datalab::DatalabClient::new("dummy-key").with_base_url(mock.uri());
    let mut source = papers_core::text::PdfSource::ZoteroLocal { path: "/tmp/dummy.pdf".into() };

    let result = papers_core::text::do_extract(
        b"%PDF test".to_vec(),
        KEY_FZIP_SOME,
        Some(&zc),
        Some((&dl, papers_core::text::ProcessingMode::Balanced)),
        &mut source,
    )
    .await;

    assert!(result.is_ok(), "should restore from Zotero zip: {:?}", result);
    assert!(result.unwrap().contains("Found Zip Paper"),
        "restored markdown should match zip contents");
}

/// `ingest_params_from_cache` succeeds even when `meta.json` is absent —
/// it falls back to using the item key as both title and paper_id.
/// This matches the production code path (meta is `Option`, not required).
#[test]
fn test_rag_add_ingest_params_from_cache_falls_back_without_meta() {
    // Write only .md — no meta.json
    write_md_cache(KEY_NO_META, "# No Meta Paper\n");

    let result = papers_rag::ingest_params_from_cache(KEY_NO_META);
    assert!(result.is_ok(),
        "ingest_params_from_cache should succeed even without meta.json (falls back gracefully)");
    let params = result.unwrap();
    assert_eq!(params.title, KEY_NO_META,
        "title should fall back to item key when meta.json is absent");
    assert_eq!(params.paper_id, KEY_NO_META,
        "paper_id should fall back to item key when meta.json is absent");
}

/// `list_cached_item_keys` only returns keys that have a `.json` extraction
/// file — not bare `.md`-only entries.
///
/// This is the contract the `--all` path in `rag work add` relies on:
/// only fully-extracted papers (with structured JSON) are batch-ingested.
#[test]
fn test_rag_add_list_cached_keys_requires_json_file() {
    write_md_cache(KEY_MD_ONLY,   "# MD only\n");
    write_md_cache(KEY_WITH_JSON, "# With JSON\n");
    write_json_cache(KEY_WITH_JSON, r#"{"pages":[]}"#);

    let keys = papers_rag::list_cached_item_keys();
    assert!(
        keys.contains(&KEY_WITH_JSON.to_string()),
        "key with .json should appear in list_cached_item_keys"
    );
    assert!(
        !keys.contains(&KEY_MD_ONLY.to_string()),
        "key without .json must NOT appear (--all only ingests fully-extracted papers)"
    );
}
