use papers::OpenAlexClient;
use papers_mcp::server::PapersMcp;
use rmcp::handler::server::wrapper::Parameters;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn minimal_list_json() -> String {
    r#"{
        "meta": {"count": 42, "db_response_time_ms": 10, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [],
        "group_by": []
    }"#
    .to_string()
}

fn minimal_work_json() -> String {
    r#"{
        "id": "https://openalex.org/W2741809807",
        "doi": "https://doi.org/10.7717/peerj.4375",
        "title": "The state of OA",
        "display_name": "The state of OA",
        "publication_year": 2018,
        "type": "article",
        "cited_by_count": 1234
    }"#
    .to_string()
}

fn minimal_autocomplete_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 10, "page": 1, "per_page": 10},
        "results": [{
            "id": "https://openalex.org/W123",
            "short_id": "works/W123",
            "display_name": "Test Work",
            "hint": "Author Name",
            "cited_by_count": 100,
            "works_count": 1,
            "entity_type": "work",
            "external_id": null,
            "filter_key": "openalex"
        }]
    }"#
    .to_string()
}

fn minimal_find_json() -> String {
    r#"{
        "meta": null,
        "results": []
    }"#
    .to_string()
}

fn make_server(mock_server: &MockServer) -> PapersMcp {
    let client = OpenAlexClient::new().with_base_url(mock_server.uri());
    PapersMcp::with_client(client)
}

// ── List tool tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_list_works_tool() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    let text = result.unwrap();
    assert!(text.contains("\"count\": 42"));
}

#[tokio::test]
async fn test_list_works_with_params() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("search", "machine learning"))
        .and(query_param("per-page", "5"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({
        "search": "machine learning",
        "per_page": 5
    }))
    .unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

// ── Get tool tests ───────────────────────────────────────────────────

#[tokio::test]
async fn test_get_work_tool() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/W2741809807"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_work_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"id": "W2741809807"})).unwrap();
    let result = server.work_get(Parameters(params)).await;
    let text = result.unwrap();
    assert!(text.contains("The state of OA"));
}

#[tokio::test]
async fn test_get_work_with_select() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/W123"))
        .and(query_param("select", "id,display_name"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_work_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params =
        serde_json::from_value(serde_json::json!({"id": "W123", "select": "id,display_name"}))
            .unwrap();
    let result = server.work_get(Parameters(params)).await;
    assert!(result.is_ok());
}

// ── Autocomplete tool tests ──────────────────────────────────────────

#[tokio::test]
async fn test_autocomplete_works_tool() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/autocomplete/works"))
        .and(query_param("q", "machine"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_autocomplete_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"q": "machine"})).unwrap();
    let result = server.work_autocomplete(Parameters(params)).await;
    let text = result.unwrap();
    assert!(text.contains("Test Work"));
}

// ── Find works tool tests ────────────────────────────────────────────

#[tokio::test]
async fn test_find_works_get() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/find/works"))
        .and(query_param("query", "drug discovery"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_find_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"query": "drug discovery"})).unwrap();
    let result = server.work_find(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_find_works_post_for_long_query() {
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/find/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_find_json()))
        .mount(&mock)
        .await;

    let long_query = "a ".repeat(1500); // >2048 chars
    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"query": long_query})).unwrap();
    let result = server.work_find(Parameters(params)).await;
    assert!(result.is_ok());
}

// ── Error handling tests ─────────────────────────────────────────────

#[tokio::test]
async fn test_api_error_returns_error_result() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works/invalid"))
        .respond_with(ResponseTemplate::new(404).set_body_string("Not found"))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"id": "invalid"})).unwrap();
    let result = server.work_get(Parameters(params)).await;

    // API errors should be returned as Err, not panics
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("404"));
}

// ── Tool listing tests ───────────────────────────────────────────────

#[test]
fn test_tool_router_has_29_tools() {
    let router = PapersMcp::tool_router();
    let tools = router.list_all();
    assert_eq!(tools.len(), 29);
}

#[test]
fn test_all_tool_names_present() {
    let router = PapersMcp::tool_router();
    let tools = router.list_all();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();

    let expected = [
        "work_list",
        "author_list",
        "source_list",
        "institution_list",
        "topic_list",
        "publisher_list",
        "funder_list",
        "domain_list",
        "field_list",
        "subfield_list",
        "work_get",
        "author_get",
        "source_get",
        "institution_get",
        "topic_get",
        "publisher_get",
        "funder_get",
        "domain_get",
        "field_get",
        "subfield_get",
        "work_autocomplete",
        "author_autocomplete",
        "source_autocomplete",
        "institution_autocomplete",
        "concept_autocomplete",
        "publisher_autocomplete",
        "funder_autocomplete",
        "subfield_autocomplete",
        "work_find",
    ];

    for name in &expected {
        assert!(names.contains(name), "Missing tool: {name}");
    }
}

#[test]
fn test_all_tools_have_descriptions() {
    let router = PapersMcp::tool_router();
    let tools = router.list_all();

    for tool in &tools {
        assert!(
            tool.description.is_some(),
            "Tool {} is missing a description",
            tool.name
        );
    }
}

// ── Summary mapping unit tests ────────────────────────────────────────

mod summary_unit {
    use papers::summary::{
        AuthorSummary, DomainSummary, FieldSummary, FunderSummary, InstitutionSummary,
        PublisherSummary, SourceSummary, SubfieldSummary, TopicSummary, WorkSummary,
    };
    use papers::{Author, Domain, Field, Funder, Institution, Publisher, Source, Subfield, Topic, Work};

    fn minimal_work() -> Work {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1234/test",
            "display_name": "Test Work",
            "publication_year": 2023,
            "type": "article",
            "cited_by_count": 42,
            "open_access": {"is_oa": true, "oa_status": "gold", "oa_url": "https://example.com/oa"},
            "authorships": [
                {"author": {"id": "https://openalex.org/A1", "display_name": "Alice"}},
                {"author": {"id": "https://openalex.org/A2", "display_name": "Bob"}}
            ],
            "primary_location": {
                "source": {"id": "https://openalex.org/S1", "display_name": "Nature"}
            },
            "primary_topic": {"id": "https://openalex.org/T1", "display_name": "Machine Learning"}
        }"#).unwrap()
    }

    fn minimal_author() -> Author {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/A1",
            "display_name": "Alice Smith",
            "orcid": "https://orcid.org/0000-0001-2345-6789",
            "works_count": 50,
            "cited_by_count": 1000,
            "summary_stats": {"2yr_mean_citedness": 3.5, "h_index": 15, "i10_index": 20},
            "last_known_institutions": [
                {"id": "https://openalex.org/I1", "display_name": "MIT"}
            ],
            "topics": [
                {"id": "https://openalex.org/T1", "display_name": "AI"},
                {"id": "https://openalex.org/T2", "display_name": "ML"},
                {"id": "https://openalex.org/T3", "display_name": "NLP"},
                {"id": "https://openalex.org/T4", "display_name": "CV"}
            ]
        }"#).unwrap()
    }

    fn minimal_source() -> Source {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/S1",
            "display_name": "Nature",
            "issn_l": "0028-0836",
            "type": "journal",
            "is_oa": false,
            "is_in_doaj": false,
            "works_count": 450000,
            "cited_by_count": 25000000,
            "summary_stats": {"2yr_mean_citedness": 50.2, "h_index": 1200, "i10_index": 50000},
            "host_organization_name": "Springer Nature"
        }"#).unwrap()
    }

    fn minimal_institution() -> Institution {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/I1",
            "display_name": "Harvard University",
            "ror": "https://ror.org/03vek6s52",
            "country_code": "US",
            "type": "education",
            "works_count": 800000,
            "cited_by_count": 40000000,
            "summary_stats": {"2yr_mean_citedness": 10.0, "h_index": 800, "i10_index": 200000},
            "geo": {"city": "Cambridge", "country": "United States", "latitude": 42.37, "longitude": -71.11}
        }"#).unwrap()
    }

    fn minimal_topic() -> Topic {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/T1",
            "display_name": "Machine Learning",
            "description": "Research on machine learning algorithms and applications.",
            "subfield": {"id": 17, "display_name": "Artificial Intelligence"},
            "field": {"id": 1, "display_name": "Computer Science"},
            "domain": {"id": 1, "display_name": "Physical Sciences"},
            "works_count": 500000,
            "cited_by_count": 10000000
        }"#).unwrap()
    }

    fn minimal_publisher() -> Publisher {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/P1",
            "display_name": "Springer Nature",
            "hierarchy_level": 0,
            "country_codes": ["DE"],
            "works_count": 2750825,
            "cited_by_count": 75000000
        }"#).unwrap()
    }

    fn minimal_funder() -> Funder {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/F1",
            "display_name": "National Institutes of Health",
            "country_code": "US",
            "description": "US federal biomedical research agency",
            "awards_count": 500000,
            "works_count": 3253779,
            "cited_by_count": 150000000
        }"#).unwrap()
    }

    #[test]
    fn work_summary_maps_essential_fields() {
        let s = WorkSummary::from(minimal_work());
        assert_eq!(s.id, "https://openalex.org/W1");
        assert_eq!(s.title.as_deref(), Some("Test Work"));
        assert_eq!(s.doi.as_deref(), Some("https://doi.org/10.1234/test"));
        assert_eq!(s.publication_year, Some(2023));
        assert_eq!(s.r#type.as_deref(), Some("article"));
        assert_eq!(s.cited_by_count, Some(42));
        assert_eq!(s.is_oa, Some(true));
        assert_eq!(s.oa_url.as_deref(), Some("https://example.com/oa"));
        assert_eq!(s.journal.as_deref(), Some("Nature"));
        assert_eq!(s.primary_topic.as_deref(), Some("Machine Learning"));
        assert_eq!(s.authors, vec!["Alice", "Bob"]);
    }

    #[test]
    fn work_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&WorkSummary::from(minimal_work())).unwrap();
        // Verbose fields that should NOT appear
        assert!(!json.contains("referenced_works"));
        assert!(!json.contains("counts_by_year"));
        assert!(!json.contains("locations"));
        assert!(!json.contains("mesh"));
        assert!(!json.contains("concepts"));
        // Essential fields that SHOULD appear
        assert!(json.contains("cited_by_count"));
        assert!(json.contains("publication_year"));
    }

    #[test]
    fn author_summary_maps_essential_fields() {
        let s = AuthorSummary::from(minimal_author());
        assert_eq!(s.id, "https://openalex.org/A1");
        assert_eq!(s.display_name.as_deref(), Some("Alice Smith"));
        assert_eq!(s.orcid.as_deref(), Some("https://orcid.org/0000-0001-2345-6789"));
        assert_eq!(s.works_count, Some(50));
        assert_eq!(s.cited_by_count, Some(1000));
        assert_eq!(s.h_index, Some(15));
        assert_eq!(s.last_known_institutions, vec!["MIT"]);
        // Only first 3 topics kept
        assert_eq!(s.top_topics, vec!["AI", "ML", "NLP"]);
        assert_eq!(s.top_topics.len(), 3);
    }

    #[test]
    fn author_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&AuthorSummary::from(minimal_author())).unwrap();
        assert!(!json.contains("affiliations"));
        assert!(!json.contains("counts_by_year"));
        assert!(!json.contains("x_concepts"));
        assert!(json.contains("h_index"));
        assert!(json.contains("works_count"));
    }

    #[test]
    fn source_summary_maps_essential_fields() {
        let s = SourceSummary::from(minimal_source());
        assert_eq!(s.id, "https://openalex.org/S1");
        assert_eq!(s.display_name.as_deref(), Some("Nature"));
        assert_eq!(s.issn_l.as_deref(), Some("0028-0836"));
        assert_eq!(s.r#type.as_deref(), Some("journal"));
        assert_eq!(s.is_oa, Some(false));
        assert_eq!(s.is_in_doaj, Some(false));
        assert_eq!(s.works_count, Some(450000));
        assert_eq!(s.cited_by_count, Some(25000000));
        assert_eq!(s.h_index, Some(1200));
        assert_eq!(s.host_organization_name.as_deref(), Some("Springer Nature"));
    }

    #[test]
    fn source_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&SourceSummary::from(minimal_source())).unwrap();
        assert!(!json.contains("apc_prices"));
        assert!(!json.contains("counts_by_year"));
        assert!(!json.contains("topics"));
        assert!(json.contains("h_index"));
        assert!(json.contains("issn_l"));
    }

    #[test]
    fn institution_summary_maps_essential_fields() {
        let s = InstitutionSummary::from(minimal_institution());
        assert_eq!(s.id, "https://openalex.org/I1");
        assert_eq!(s.display_name.as_deref(), Some("Harvard University"));
        assert_eq!(s.ror.as_deref(), Some("https://ror.org/03vek6s52"));
        assert_eq!(s.country_code.as_deref(), Some("US"));
        assert_eq!(s.r#type.as_deref(), Some("education"));
        assert_eq!(s.city.as_deref(), Some("Cambridge"));
        assert_eq!(s.works_count, Some(800000));
        assert_eq!(s.cited_by_count, Some(40000000));
        assert_eq!(s.h_index, Some(800));
    }

    #[test]
    fn institution_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&InstitutionSummary::from(minimal_institution())).unwrap();
        assert!(!json.contains("associated_institutions"));
        assert!(!json.contains("counts_by_year"));
        assert!(!json.contains("repositories"));
        assert!(json.contains("h_index"));
        assert!(json.contains("country_code"));
    }

    #[test]
    fn topic_summary_maps_essential_fields() {
        let s = TopicSummary::from(minimal_topic());
        assert_eq!(s.id, "https://openalex.org/T1");
        assert_eq!(s.display_name.as_deref(), Some("Machine Learning"));
        assert!(s.description.as_deref().unwrap().contains("machine learning"));
        assert_eq!(s.subfield.as_deref(), Some("Artificial Intelligence"));
        assert_eq!(s.field.as_deref(), Some("Computer Science"));
        assert_eq!(s.domain.as_deref(), Some("Physical Sciences"));
        assert_eq!(s.works_count, Some(500000));
        assert_eq!(s.cited_by_count, Some(10000000));
    }

    #[test]
    fn topic_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&TopicSummary::from(minimal_topic())).unwrap();
        assert!(!json.contains("keywords"));
        assert!(!json.contains("siblings"));
        assert!(!json.contains("works_api_url"));
        assert!(json.contains("description"));
        assert!(json.contains("works_count"));
    }

    #[test]
    fn publisher_summary_maps_essential_fields() {
        let s = PublisherSummary::from(minimal_publisher());
        assert_eq!(s.id, "https://openalex.org/P1");
        assert_eq!(s.display_name.as_deref(), Some("Springer Nature"));
        assert_eq!(s.hierarchy_level, Some(0));
        assert_eq!(s.country_codes.as_deref(), Some(["DE".to_string()].as_slice()));
        assert_eq!(s.works_count, Some(2750825));
        assert_eq!(s.cited_by_count, Some(75000000));
    }

    #[test]
    fn publisher_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&PublisherSummary::from(minimal_publisher())).unwrap();
        assert!(!json.contains("lineage"));
        assert!(!json.contains("counts_by_year"));
        assert!(!json.contains("alternate_titles"));
        assert!(json.contains("hierarchy_level"));
        assert!(json.contains("country_codes"));
    }

    #[test]
    fn funder_summary_maps_essential_fields() {
        let s = FunderSummary::from(minimal_funder());
        assert_eq!(s.id, "https://openalex.org/F1");
        assert_eq!(s.display_name.as_deref(), Some("National Institutes of Health"));
        assert_eq!(s.country_code.as_deref(), Some("US"));
        assert_eq!(s.description.as_deref(), Some("US federal biomedical research agency"));
        assert_eq!(s.awards_count, Some(500000));
        assert_eq!(s.works_count, Some(3253779));
        assert_eq!(s.cited_by_count, Some(150000000));
    }

    #[test]
    fn funder_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&FunderSummary::from(minimal_funder())).unwrap();
        assert!(!json.contains("alternate_titles"));
        assert!(!json.contains("counts_by_year"));
        assert!(!json.contains("roles"));
        assert!(json.contains("awards_count"));
        assert!(json.contains("country_code"));
    }

    fn minimal_domain() -> Domain {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/domains/3",
            "display_name": "Physical Sciences",
            "description": "branch of natural science",
            "fields": [
                {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
                {"id": "https://openalex.org/fields/22", "display_name": "Engineering"}
            ],
            "siblings": [{"id": "https://openalex.org/domains/1", "display_name": "Life Sciences"}],
            "works_count": 134263529,
            "cited_by_count": 1500000000
        }"#).unwrap()
    }

    fn minimal_field() -> Field {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/fields/17",
            "display_name": "Computer Science",
            "description": "study of computation",
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
            "subfields": [
                {"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"},
                {"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory"}
            ],
            "siblings": [{"id": "https://openalex.org/fields/22", "display_name": "Engineering"}],
            "works_count": 22038624,
            "cited_by_count": 500000000
        }"#).unwrap()
    }

    fn minimal_subfield() -> Subfield {
        serde_json::from_str(r#"{
            "id": "https://openalex.org/subfields/1702",
            "display_name": "Artificial Intelligence",
            "description": "study of intelligent agents",
            "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
            "topics": [{"id": "https://openalex.org/T10028", "display_name": "Topic Modeling"}],
            "siblings": [{"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory"}],
            "works_count": 9059921,
            "cited_by_count": 200000000
        }"#).unwrap()
    }

    #[test]
    fn domain_summary_maps_essential_fields() {
        let s = DomainSummary::from(minimal_domain());
        assert_eq!(s.id, "https://openalex.org/domains/3");
        assert_eq!(s.display_name.as_deref(), Some("Physical Sciences"));
        assert_eq!(s.description.as_deref(), Some("branch of natural science"));
        assert_eq!(s.fields, vec!["Computer Science", "Engineering"]);
        assert_eq!(s.works_count, Some(134263529));
        assert_eq!(s.cited_by_count, Some(1500000000));
    }

    #[test]
    fn domain_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&DomainSummary::from(minimal_domain())).unwrap();
        assert!(!json.contains("siblings"));
        assert!(!json.contains("display_name_alternatives"));
        assert!(!json.contains("works_api_url"));
        assert!(json.contains("fields"));
        assert!(json.contains("works_count"));
    }

    #[test]
    fn field_summary_maps_essential_fields() {
        let s = FieldSummary::from(minimal_field());
        assert_eq!(s.id, "https://openalex.org/fields/17");
        assert_eq!(s.display_name.as_deref(), Some("Computer Science"));
        assert_eq!(s.domain.as_deref(), Some("Physical Sciences"));
        assert_eq!(s.subfield_count, 2);
        assert_eq!(s.works_count, Some(22038624));
        assert_eq!(s.cited_by_count, Some(500000000));
    }

    #[test]
    fn field_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&FieldSummary::from(minimal_field())).unwrap();
        assert!(!json.contains("siblings"));
        assert!(!json.contains("display_name_alternatives"));
        assert!(!json.contains("subfields"));
        assert!(json.contains("subfield_count"));
        assert!(json.contains("domain"));
    }

    #[test]
    fn subfield_summary_maps_essential_fields() {
        let s = SubfieldSummary::from(minimal_subfield());
        assert_eq!(s.id, "https://openalex.org/subfields/1702");
        assert_eq!(s.display_name.as_deref(), Some("Artificial Intelligence"));
        assert_eq!(s.field.as_deref(), Some("Computer Science"));
        assert_eq!(s.domain.as_deref(), Some("Physical Sciences"));
        assert_eq!(s.works_count, Some(9059921));
        assert_eq!(s.cited_by_count, Some(200000000));
    }

    #[test]
    fn subfield_summary_serializes_without_verbose_fields() {
        let json = serde_json::to_string(&SubfieldSummary::from(minimal_subfield())).unwrap();
        assert!(!json.contains("siblings"));
        assert!(!json.contains("topics"));
        assert!(!json.contains("display_name_alternatives"));
        assert!(json.contains("\"field\""));
        assert!(json.contains("\"domain\""));
    }
}

// ── Summary integration tests (wiremock) ─────────────────────────────

fn work_list_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 10, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/W1",
            "display_name": "Bitonic Sort",
            "doi": "https://doi.org/10.1234/test",
            "publication_year": 2020,
            "type": "article",
            "cited_by_count": 99,
            "open_access": {"is_oa": true, "oa_status": "gold", "oa_url": "https://example.com/oa"},
            "authorships": [{"author": {"id": "https://openalex.org/A1", "display_name": "Alice"}}],
            "primary_location": {"source": {"id": "https://openalex.org/S1", "display_name": "JACM"}},
            "primary_topic": {"id": "https://openalex.org/T1", "display_name": "Algorithms"},
            "abstract_inverted_index": {"Sorting": [0], "networks": [1]},
            "referenced_works": ["https://openalex.org/W2"],
            "counts_by_year": [{"year": 2020, "works_count": 1, "cited_by_count": 50}],
            "locations": [{"is_oa": true}],
            "mesh": [{"descriptor_name": "Algorithms"}]
        }],
        "group_by": []
    }"#.to_string()
}

fn author_list_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/A1",
            "display_name": "Alice",
            "orcid": "https://orcid.org/0000-0001-0000-0001",
            "works_count": 10,
            "cited_by_count": 200,
            "summary_stats": {"2yr_mean_citedness": 2.0, "h_index": 8, "i10_index": 5},
            "last_known_institutions": [{"id": "https://openalex.org/I1", "display_name": "MIT"}],
            "topics": [{"id": "https://openalex.org/T1", "display_name": "AI"}],
            "affiliations": [{"institution": {"id": "https://openalex.org/I1", "display_name": "MIT"}, "years": [2020]}],
            "counts_by_year": [{"year": 2020, "works_count": 2, "cited_by_count": 50}]
        }],
        "group_by": []
    }"#.to_string()
}

fn source_list_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/S1",
            "display_name": "Nature",
            "issn_l": "0028-0836",
            "type": "journal",
            "is_oa": false,
            "is_in_doaj": false,
            "works_count": 450000,
            "cited_by_count": 25000000,
            "summary_stats": {"2yr_mean_citedness": 50.0, "h_index": 1200, "i10_index": 50000},
            "host_organization_name": "Springer Nature",
            "apc_prices": [{"price": 5000, "currency": "USD"}],
            "counts_by_year": [{"year": 2020, "works_count": 1000, "cited_by_count": 500000}]
        }],
        "group_by": []
    }"#.to_string()
}

fn institution_list_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/I1",
            "display_name": "Harvard University",
            "ror": "https://ror.org/03vek6s52",
            "country_code": "US",
            "type": "education",
            "works_count": 800000,
            "cited_by_count": 40000000,
            "summary_stats": {"2yr_mean_citedness": 10.0, "h_index": 800, "i10_index": 200000},
            "geo": {"city": "Cambridge", "country": "United States", "latitude": 42.37, "longitude": -71.11},
            "associated_institutions": [{"id": "https://openalex.org/I2", "display_name": "HMS", "relationship": "child"}],
            "counts_by_year": [{"year": 2020, "works_count": 10000, "cited_by_count": 1000000}]
        }],
        "group_by": []
    }"#.to_string()
}

fn topic_list_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/T1",
            "display_name": "Machine Learning",
            "description": "Research on ML algorithms.",
            "keywords": ["neural networks", "deep learning"],
            "subfield": {"id": 17, "display_name": "Artificial Intelligence"},
            "field": {"id": 1, "display_name": "Computer Science"},
            "domain": {"id": 1, "display_name": "Physical Sciences"},
            "works_count": 500000,
            "cited_by_count": 10000000,
            "siblings": [{"id": "https://openalex.org/T2", "display_name": "Deep Learning"}]
        }],
        "group_by": []
    }"#.to_string()
}

fn publisher_list_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/P1",
            "display_name": "Springer Nature",
            "hierarchy_level": 0,
            "country_codes": ["DE"],
            "works_count": 2750825,
            "cited_by_count": 75000000,
            "alternate_titles": ["Springer"],
            "lineage": [{"id": "https://openalex.org/P1"}],
            "counts_by_year": [{"year": 2020, "works_count": 100000, "cited_by_count": 5000000}]
        }],
        "group_by": []
    }"#.to_string()
}

fn funder_list_json() -> String {
    r#"{
        "meta": {"count": 1, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/F1",
            "display_name": "National Institutes of Health",
            "country_code": "US",
            "description": "US federal biomedical research agency",
            "awards_count": 500000,
            "works_count": 3253779,
            "cited_by_count": 150000000,
            "alternate_titles": ["NIH"],
            "counts_by_year": [{"year": 2020, "works_count": 200000, "cited_by_count": 10000000}]
        }],
        "group_by": []
    }"#.to_string()
}

#[tokio::test]
async fn test_list_works_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .respond_with(ResponseTemplate::new(200).set_body_string(work_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.work_list(Parameters(params)).await.unwrap();

    // Essential fields present
    assert!(text.contains("Bitonic Sort"), "title missing");
    assert!(text.contains("\"cited_by_count\""), "cited_by_count missing");
    assert!(text.contains("Alice"), "author name missing");
    assert!(text.contains("JACM"), "journal missing");
    assert!(text.contains("Algorithms"), "primary_topic missing");
    assert!(text.contains("Sorting networks"), "abstract_text missing");
    // Verbose fields absent
    assert!(!text.contains("referenced_works"), "referenced_works should be absent");
    assert!(!text.contains("counts_by_year"), "counts_by_year should be absent");
    assert!(!text.contains("\"locations\""), "locations should be absent");
    assert!(!text.contains("mesh"), "mesh should be absent");
    // group_by absent
    assert!(!text.contains("group_by"), "group_by should be absent");
}

#[tokio::test]
async fn test_list_authors_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/authors"))
        .respond_with(ResponseTemplate::new(200).set_body_string(author_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.author_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Alice"));
    assert!(text.contains("h_index"));
    assert!(text.contains("MIT"));
    assert!(!text.contains("affiliations"), "affiliations should be absent");
    assert!(!text.contains("counts_by_year"), "counts_by_year should be absent");
}

#[tokio::test]
async fn test_list_sources_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources"))
        .respond_with(ResponseTemplate::new(200).set_body_string(source_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.source_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Nature"));
    assert!(text.contains("h_index"));
    assert!(text.contains("Springer Nature"));
    assert!(!text.contains("apc_prices"), "apc_prices should be absent");
    assert!(!text.contains("counts_by_year"), "counts_by_year should be absent");
}

#[tokio::test]
async fn test_list_institutions_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/institutions"))
        .respond_with(ResponseTemplate::new(200).set_body_string(institution_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.institution_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Harvard University"));
    assert!(text.contains("Cambridge"));
    assert!(text.contains("h_index"));
    assert!(!text.contains("associated_institutions"), "associated_institutions should be absent");
    assert!(!text.contains("counts_by_year"), "counts_by_year should be absent");
}

#[tokio::test]
async fn test_list_topics_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/topics"))
        .respond_with(ResponseTemplate::new(200).set_body_string(topic_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.topic_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Machine Learning"));
    assert!(text.contains("Artificial Intelligence"));
    assert!(text.contains("Computer Science"));
    assert!(!text.contains("keywords"), "keywords should be absent");
    assert!(!text.contains("siblings"), "siblings should be absent");
}

#[tokio::test]
async fn test_list_publishers_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/publishers"))
        .respond_with(ResponseTemplate::new(200).set_body_string(publisher_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.publisher_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Springer Nature"));
    assert!(text.contains("hierarchy_level"));
    assert!(!text.contains("alternate_titles"), "alternate_titles should be absent");
    assert!(!text.contains("counts_by_year"), "counts_by_year should be absent");
    assert!(!text.contains("lineage"), "lineage should be absent");
}

#[tokio::test]
async fn test_list_funders_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/funders"))
        .respond_with(ResponseTemplate::new(200).set_body_string(funder_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.funder_list(Parameters(params)).await.unwrap();

    assert!(text.contains("National Institutes of Health"));
    assert!(text.contains("awards_count"));
    assert!(!text.contains("alternate_titles"), "alternate_titles should be absent");
    assert!(!text.contains("counts_by_year"), "counts_by_year should be absent");
}

fn domain_list_json() -> String {
    r#"{
        "meta": {"count": 4, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/domains/3",
            "display_name": "Physical Sciences",
            "description": "branch of natural science",
            "fields": [{"id": "https://openalex.org/fields/17", "display_name": "Computer Science"}],
            "siblings": [{"id": "https://openalex.org/domains/1", "display_name": "Life Sciences"}],
            "display_name_alternatives": [],
            "works_count": 134263529,
            "cited_by_count": 1500000000,
            "works_api_url": "https://api.openalex.org/works?filter=primary_topic.domain.id:domains/3"
        }],
        "group_by": []
    }"#.to_string()
}

fn field_list_json() -> String {
    r#"{
        "meta": {"count": 26, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/fields/17",
            "display_name": "Computer Science",
            "description": "study of computation",
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
            "subfields": [{"id": "https://openalex.org/subfields/1702", "display_name": "Artificial Intelligence"}],
            "siblings": [{"id": "https://openalex.org/fields/22", "display_name": "Engineering"}],
            "display_name_alternatives": [],
            "works_count": 22038624,
            "cited_by_count": 500000000,
            "works_api_url": "https://api.openalex.org/works?filter=primary_topic.field.id:fields/17"
        }],
        "group_by": []
    }"#.to_string()
}

fn subfield_list_json() -> String {
    r#"{
        "meta": {"count": 252, "db_response_time_ms": 5, "page": 1, "per_page": 25, "next_cursor": null, "groups_count": null},
        "results": [{
            "id": "https://openalex.org/subfields/1702",
            "display_name": "Artificial Intelligence",
            "description": "study of intelligent agents",
            "field": {"id": "https://openalex.org/fields/17", "display_name": "Computer Science"},
            "domain": {"id": "https://openalex.org/domains/3", "display_name": "Physical Sciences"},
            "topics": [{"id": "https://openalex.org/T10028", "display_name": "Topic Modeling"}],
            "siblings": [{"id": "https://openalex.org/subfields/1703", "display_name": "Computational Theory"}],
            "display_name_alternatives": [],
            "works_count": 9059921,
            "cited_by_count": 200000000,
            "works_api_url": "https://api.openalex.org/works?filter=primary_topic.subfield.id:subfields/1702"
        }],
        "group_by": []
    }"#.to_string()
}

#[tokio::test]
async fn test_list_domains_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/domains"))
        .respond_with(ResponseTemplate::new(200).set_body_string(domain_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.domain_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Physical Sciences"));
    assert!(text.contains("Computer Science")); // field display_name kept
    assert!(!text.contains("siblings"), "siblings should be absent");
    assert!(!text.contains("works_api_url"), "works_api_url should be absent");
    assert!(!text.contains("display_name_alternatives"), "display_name_alternatives should be absent");
}

#[tokio::test]
async fn test_list_fields_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/fields"))
        .respond_with(ResponseTemplate::new(200).set_body_string(field_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.field_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Computer Science"));
    assert!(text.contains("Physical Sciences")); // domain display_name kept
    assert!(text.contains("subfield_count")); // count instead of full list
    assert!(!text.contains("siblings"), "siblings should be absent");
    assert!(!text.contains("Artificial Intelligence"), "subfields list should be absent");
}

#[tokio::test]
async fn test_list_subfields_returns_slim_response() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/subfields"))
        .respond_with(ResponseTemplate::new(200).set_body_string(subfield_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({})).unwrap();
    let text = server.subfield_list(Parameters(params)).await.unwrap();

    assert!(text.contains("Artificial Intelligence"));
    assert!(text.contains("Computer Science")); // field display_name kept
    assert!(text.contains("Physical Sciences")); // domain display_name kept
    assert!(!text.contains("siblings"), "siblings should be absent");
    assert!(!text.contains("topics"), "topics should be absent");
}

// ── Schema tests ─────────────────────────────────────────────────────

#[test]
fn test_tool_params_schema() {
    use papers_mcp::params::ListToolParams;
    let schema = schemars::schema_for!(ListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    assert_eq!(json["type"], "object");
    let props = json["properties"].as_object().unwrap();
    assert!(props.contains_key("filter"));
    assert!(props.contains_key("search"));
    assert!(props.contains_key("sort"));
    assert!(props.contains_key("per_page"));
    assert!(props.contains_key("page"));
    assert!(props.contains_key("cursor"));
    assert!(props.contains_key("sample"));
    assert!(props.contains_key("seed"));
    assert!(props.contains_key("select"));
    assert!(props.contains_key("group_by"));
}

// ── Work list filter alias tests ─────────────────────────────────────

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
async fn test_work_list_with_year_alias() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "publication_year:>2020"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"year": ">2020"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_citations_alias() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "cited_by_count:>100"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"citations": ">100"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_author_id() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "authorships.author.id:A5083138872"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params =
        serde_json::from_value(serde_json::json!({"author": "A5083138872"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_author_search() {
    let mock = MockServer::start().await;
    // Mock the author search resolution
    Mock::given(method("GET"))
        .and(path("/authors"))
        .and(query_param("filter", "display_name.search:einstein"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/A5083138872")),
        )
        .mount(&mock)
        .await;
    // Mock the works list with resolved filter
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "authorships.author.id:A5083138872"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"author": "einstein"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_publisher_search() {
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
        .and(query_param(
            "filter",
            "primary_location.source.publisher_lineage:P4310319798",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"publisher": "acm"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_source_search() {
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
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param(
            "filter",
            "primary_location.source.id:S131921510",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"source": "siggraph"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_topic_id() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "primary_topic.id:T11636"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"topic": "T11636"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_domain_id() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "primary_topic.domain.id:domains/3"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"domain": "3"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_field_id() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param("filter", "primary_topic.field.id:fields/17"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"field": "17"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_with_subfield_id() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param(
            "filter",
            "primary_topic.subfield.id:subfields/1702",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params =
        serde_json::from_value(serde_json::json!({"subfield": "subfields/1702"})).unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_combined_aliases_and_filter() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/works"))
        .and(query_param(
            "filter",
            "publication_year:2024,cited_by_count:>100,is_oa:true",
        ))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({
        "year": "2024",
        "citations": ">100",
        "filter": "is_oa:true"
    }))
    .unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_work_list_overlap_error() {
    let mock = MockServer::start().await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({
        "year": "2024",
        "filter": "publication_year:>2020"
    }))
    .unwrap();
    let result = server.work_list(Parameters(params)).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.contains("year"));
    assert!(err.contains("publication_year"));
}

#[test]
fn test_work_list_schema_includes_aliases() {
    use papers_mcp::params::WorkListToolParams;
    let schema = schemars::schema_for!(WorkListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    // Standard list params
    assert!(props.contains_key("filter"));
    assert!(props.contains_key("search"));
    assert!(props.contains_key("sort"));
    assert!(props.contains_key("per_page"));

    // Alias params
    assert!(props.contains_key("author"), "missing author");
    assert!(props.contains_key("topic"), "missing topic");
    assert!(props.contains_key("domain"), "missing domain");
    assert!(props.contains_key("field"), "missing field");
    assert!(props.contains_key("subfield"), "missing subfield");
    assert!(props.contains_key("publisher"), "missing publisher");
    assert!(props.contains_key("source"), "missing source");
    assert!(props.contains_key("year"), "missing year");
    assert!(props.contains_key("citations"), "missing citations");
}

#[test]
fn test_work_list_schema_includes_new_aliases() {
    use papers_mcp::params::WorkListToolParams;
    let schema = schemars::schema_for!(WorkListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("institution"), "missing institution");
    assert!(props.contains_key("country"), "missing country");
    assert!(props.contains_key("continent"), "missing continent");
    assert!(props.contains_key("type"), "missing type");
    assert!(props.contains_key("open"), "missing open");
}

#[test]
fn test_author_list_schema() {
    use papers_mcp::params::AuthorListToolParams;
    let schema = schemars::schema_for!(AuthorListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("search"), "missing search");
    assert!(props.contains_key("institution"), "missing institution");
    assert!(props.contains_key("country"), "missing country");
    assert!(props.contains_key("continent"), "missing continent");
    assert!(props.contains_key("citations"), "missing citations");
    assert!(props.contains_key("works"), "missing works");
    assert!(props.contains_key("h_index"), "missing h_index");
}

#[test]
fn test_source_list_schema() {
    use papers_mcp::params::SourceListToolParams;
    let schema = schemars::schema_for!(SourceListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("publisher"), "missing publisher");
    assert!(props.contains_key("country"), "missing country");
    assert!(props.contains_key("continent"), "missing continent");
    assert!(props.contains_key("type"), "missing type");
    assert!(props.contains_key("open"), "missing open");
    assert!(props.contains_key("citations"), "missing citations");
    assert!(props.contains_key("works"), "missing works");
}

#[test]
fn test_institution_list_schema() {
    use papers_mcp::params::InstitutionListToolParams;
    let schema = schemars::schema_for!(InstitutionListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("country"), "missing country");
    assert!(props.contains_key("continent"), "missing continent");
    assert!(props.contains_key("type"), "missing type");
    assert!(props.contains_key("citations"), "missing citations");
    assert!(props.contains_key("works"), "missing works");
}

#[test]
fn test_topic_list_schema() {
    use papers_mcp::params::TopicListToolParams;
    let schema = schemars::schema_for!(TopicListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("domain"), "missing domain");
    assert!(props.contains_key("field"), "missing field");
    assert!(props.contains_key("subfield"), "missing subfield");
    assert!(props.contains_key("citations"), "missing citations");
    assert!(props.contains_key("works"), "missing works");
}

#[test]
fn test_publisher_list_schema() {
    use papers_mcp::params::PublisherListToolParams;
    let schema = schemars::schema_for!(PublisherListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("country"), "missing country");
    assert!(props.contains_key("continent"), "missing continent");
    assert!(props.contains_key("citations"), "missing citations");
    assert!(props.contains_key("works"), "missing works");
}

#[test]
fn test_funder_list_schema() {
    use papers_mcp::params::FunderListToolParams;
    let schema = schemars::schema_for!(FunderListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("country"), "missing country");
    assert!(props.contains_key("continent"), "missing continent");
    assert!(props.contains_key("citations"), "missing citations");
    assert!(props.contains_key("works"), "missing works");
}

#[test]
fn test_domain_list_schema() {
    use papers_mcp::params::DomainListToolParams;
    let schema = schemars::schema_for!(DomainListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("works"), "missing works");
}

#[test]
fn test_field_list_schema() {
    use papers_mcp::params::FieldListToolParams;
    let schema = schemars::schema_for!(FieldListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("domain"), "missing domain");
    assert!(props.contains_key("works"), "missing works");
}

#[test]
fn test_subfield_list_schema() {
    use papers_mcp::params::SubfieldListToolParams;
    let schema = schemars::schema_for!(SubfieldListToolParams);
    let json = serde_json::to_value(&schema).unwrap();
    let props = json["properties"].as_object().unwrap();

    assert!(props.contains_key("filter"), "missing filter");
    assert!(props.contains_key("domain"), "missing domain");
    assert!(props.contains_key("field"), "missing field");
    assert!(props.contains_key("works"), "missing works");
}

#[tokio::test]
async fn test_author_list_with_institution_search() {
    let mock = MockServer::start().await;
    // Mock the institution search resolution
    Mock::given(method("GET"))
        .and(path("/institutions"))
        .and(query_param("filter", "display_name.search:mit"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(search_result_json("https://openalex.org/I136199984")),
        )
        .mount(&mock)
        .await;
    // Mock the authors list with resolved filter
    Mock::given(method("GET"))
        .and(path("/authors"))
        .and(query_param("filter", "last_known_institutions.id:I136199984"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"institution": "mit"})).unwrap();
    let result = server.author_list(Parameters(params)).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_source_list_with_open_flag() {
    let mock = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/sources"))
        .and(query_param("filter", "is_oa:true"))
        .respond_with(ResponseTemplate::new(200).set_body_string(minimal_list_json()))
        .mount(&mock)
        .await;

    let server = make_server(&mock);
    let params = serde_json::from_value(serde_json::json!({"open": true})).unwrap();
    let result = server.source_list(Parameters(params)).await;
    assert!(result.is_ok());
}
