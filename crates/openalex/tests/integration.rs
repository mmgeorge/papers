use openalex::{GetParams, ListParams, OpenAlexClient};

fn client() -> OpenAlexClient {
    OpenAlexClient::new()
}

// ── Live list tests ────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_list_works() {
    let params = ListParams {
        per_page: Some(1),
        ..Default::default()
    };
    let resp = client().list_works(&params).await.unwrap();
    assert!(resp.meta.count > 0);
    assert_eq!(resp.results.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_authors() {
    let params = ListParams {
        per_page: Some(1),
        ..Default::default()
    };
    let resp = client().list_authors(&params).await.unwrap();
    assert!(resp.meta.count > 0);
    assert_eq!(resp.results.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_sources() {
    let params = ListParams {
        per_page: Some(1),
        ..Default::default()
    };
    let resp = client().list_sources(&params).await.unwrap();
    assert!(resp.meta.count > 0);
    assert_eq!(resp.results.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_institutions() {
    let params = ListParams {
        per_page: Some(1),
        ..Default::default()
    };
    let resp = client().list_institutions(&params).await.unwrap();
    assert!(resp.meta.count > 0);
    assert_eq!(resp.results.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_topics() {
    let params = ListParams {
        per_page: Some(1),
        ..Default::default()
    };
    let resp = client().list_topics(&params).await.unwrap();
    assert!(resp.meta.count > 0);
    assert_eq!(resp.results.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_publishers() {
    let params = ListParams {
        per_page: Some(1),
        ..Default::default()
    };
    let resp = client().list_publishers(&params).await.unwrap();
    assert!(resp.meta.count > 0);
    assert_eq!(resp.results.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_funders() {
    let params = ListParams {
        per_page: Some(1),
        ..Default::default()
    };
    let resp = client().list_funders(&params).await.unwrap();
    assert!(resp.meta.count > 0);
    assert_eq!(resp.results.len(), 1);
}

// ── Live get tests ─────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_get_work() {
    let work = client()
        .get_work("W2741809807", &GetParams::default())
        .await
        .unwrap();
    assert_eq!(work.id, "https://openalex.org/W2741809807");
}

#[tokio::test]
#[ignore]
async fn test_live_get_author() {
    let author = client()
        .get_author("A5023888391", &GetParams::default())
        .await
        .unwrap();
    assert_eq!(author.id, "https://openalex.org/A5023888391");
}

#[tokio::test]
#[ignore]
async fn test_live_get_source() {
    let source = client()
        .get_source("S137773608", &GetParams::default())
        .await
        .unwrap();
    assert_eq!(source.id, "https://openalex.org/S137773608");
}

#[tokio::test]
#[ignore]
async fn test_live_get_institution() {
    let inst = client()
        .get_institution("I136199984", &GetParams::default())
        .await
        .unwrap();
    assert_eq!(inst.id, "https://openalex.org/I136199984");
}

#[tokio::test]
#[ignore]
async fn test_live_get_topic() {
    let topic = client()
        .get_topic("T10001", &GetParams::default())
        .await
        .unwrap();
    assert_eq!(topic.id, "https://openalex.org/T10001");
}

#[tokio::test]
#[ignore]
async fn test_live_get_publisher() {
    let publisher = client()
        .get_publisher("P4310319965", &GetParams::default())
        .await
        .unwrap();
    assert_eq!(publisher.id, "https://openalex.org/P4310319965");
}

#[tokio::test]
#[ignore]
async fn test_live_get_funder() {
    let funder = client()
        .get_funder("F4320332161", &GetParams::default())
        .await
        .unwrap();
    assert_eq!(funder.id, "https://openalex.org/F4320332161");
}

// ── Live autocomplete tests ────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_autocomplete_works() {
    let resp = client().autocomplete_works("machine").await.unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].entity_type, "work");
}

#[tokio::test]
#[ignore]
async fn test_live_autocomplete_authors() {
    let resp = client().autocomplete_authors("einstein").await.unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].entity_type, "author");
}

#[tokio::test]
#[ignore]
async fn test_live_autocomplete_sources() {
    let resp = client().autocomplete_sources("nature").await.unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].entity_type, "source");
}

#[tokio::test]
#[ignore]
async fn test_live_autocomplete_institutions() {
    let resp = client().autocomplete_institutions("mit").await.unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].entity_type, "institution");
}

#[tokio::test]
#[ignore]
async fn test_live_autocomplete_concepts() {
    let resp = client().autocomplete_concepts("physics").await.unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].entity_type, "concept");
}

#[tokio::test]
#[ignore]
async fn test_live_autocomplete_publishers() {
    let resp = client().autocomplete_publishers("elsevier").await.unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].entity_type, "publisher");
}

#[tokio::test]
#[ignore]
async fn test_live_autocomplete_funders() {
    let resp = client().autocomplete_funders("nsf").await.unwrap();
    assert!(!resp.results.is_empty());
    assert_eq!(resp.results[0].entity_type, "funder");
}

// ── Live find works test ───────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_find_works() {
    // This test requires OPENALEX_KEY to be set
    if std::env::var("OPENALEX_KEY").is_err() {
        eprintln!("Skipping find_works test: OPENALEX_KEY not set");
        return;
    }
    let params = openalex::FindWorksParams::builder()
        .query("machine learning for drug discovery")
        .count(2)
        .build();
    let resp = client().find_works(&params).await.unwrap();
    assert!(!resp.results.is_empty());
}
