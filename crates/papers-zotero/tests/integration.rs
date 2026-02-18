use papers_zotero::{
    CollectionListParams, ItemListParams, TagListParams, ZoteroClient,
};

fn client() -> ZoteroClient {
    let user_id =
        std::env::var("ZOTERO_USER_ID").expect("ZOTERO_USER_ID must be set for live tests");
    let api_key =
        std::env::var("ZOTERO_API_KEY").expect("ZOTERO_API_KEY must be set for live tests");
    ZoteroClient::new(user_id, api_key)
}

// ── Live item tests ──────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_list_items() {
    let params = ItemListParams::builder().limit(1).build();
    let resp = client().list_items(&params).await.unwrap();
    assert!(resp.total_results.unwrap_or(0) > 0);
    assert_eq!(resp.items.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_top_items() {
    let params = ItemListParams::builder().limit(1).build();
    let resp = client().list_top_items(&params).await.unwrap();
    assert!(resp.total_results.unwrap_or(0) > 0);
    assert_eq!(resp.items.len(), 1);
}

#[tokio::test]
#[ignore]
async fn test_live_list_trash_items() {
    let params = ItemListParams::builder().limit(1).build();
    let resp = client().list_trash_items(&params).await.unwrap();
    // Trash may be empty, that's OK
    assert!(resp.total_results.is_some());
}

#[tokio::test]
#[ignore]
async fn test_live_get_item() {
    // First list to get a key, then get that item
    let params = ItemListParams::builder().limit(1).build();
    let list_resp = client().list_items(&params).await.unwrap();
    assert!(!list_resp.items.is_empty());
    let key = &list_resp.items[0].key;
    let item = client().get_item(key).await.unwrap();
    assert_eq!(&item.key, key);
}

#[tokio::test]
#[ignore]
async fn test_live_list_item_children() {
    // Find an item that has children
    let params = ItemListParams::builder()
        .item_type("-attachment || note")
        .limit(5)
        .build();
    let list_resp = client().list_items(&params).await.unwrap();
    if let Some(parent) = list_resp
        .items
        .iter()
        .find(|i| i.meta.num_children.unwrap_or(0) > 0)
    {
        let children = client()
            .list_item_children(&parent.key, &ItemListParams::default())
            .await
            .unwrap();
        assert!(!children.items.is_empty());
    }
}

#[tokio::test]
#[ignore]
async fn test_live_search_items() {
    let params = ItemListParams::builder().q("rendering").limit(5).build();
    let resp = client().list_items(&params).await.unwrap();
    assert!(!resp.items.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_live_search_xz_ordering() {
    let zotero = client();

    // Test DOI-based search (what try_zotero actually uses)
    let doi = "10.1007/3-540-48482-5_7";
    println!("\n=== DOI search: {:?} ===", doi);
    let doi_params = ItemListParams::builder().q(doi).qmode("everything").limit(5).build();
    let doi_resp = zotero.list_items(&doi_params).await.unwrap();
    println!("DOI search results: {:?}", doi_resp.total_results);
    for item in &doi_resp.items {
        println!("  key={} doi={:?} title={:?}", item.key, item.data.doi, item.data.title);
    }

    for query in &["XZ ordering", "XZ-Ordering", "GeoMesa"] {
        println!("\n=== Search: {:?} ===", query);
        let params = ItemListParams::builder().q(*query).limit(10).build();
        let resp = zotero.list_items(&params).await.unwrap();
        println!("Total results: {:?}", resp.total_results);

        for item in &resp.items {
            let title = item.data.title.as_deref().unwrap_or("<no title>");
            let doi = item.data.doi.as_deref().unwrap_or("<no doi>");
            println!("  key={} doi={:?} title={:?}", item.key, doi, title);

            // Check for PDF children
            let children = zotero
                .list_item_children(&item.key, &ItemListParams::default())
                .await
                .unwrap();
            for child in &children.items {
                let content_type = child.data.content_type.as_deref().unwrap_or("");
                let link_mode = child.data.link_mode.as_deref().unwrap_or("");
                let filename = child.data.filename.as_deref().unwrap_or("");
                println!(
                    "    child={} content_type={:?} link_mode={:?} filename={:?}",
                    child.key, content_type, link_mode, filename
                );
                // Check local file path
                if !filename.is_empty() {
                    let home = dirs::home_dir().unwrap_or_default();
                    let local = home.join("Zotero").join("storage").join(&child.key).join(filename);
                    println!("    local_path={} exists={}", local.display(), local.exists());
                }
            }
        }
    }
}

// ── Live collection tests ────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_list_collections() {
    let params = CollectionListParams::builder().limit(5).build();
    let resp = client().list_collections(&params).await.unwrap();
    assert!(resp.total_results.unwrap_or(0) > 0);
    assert!(!resp.items.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_live_list_top_collections() {
    let resp = client()
        .list_top_collections(&CollectionListParams::default())
        .await
        .unwrap();
    assert!(!resp.items.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_live_get_collection() {
    let list_resp = client()
        .list_collections(&CollectionListParams::builder().limit(1).build())
        .await
        .unwrap();
    assert!(!list_resp.items.is_empty());
    let key = &list_resp.items[0].key;
    let coll = client().get_collection(key).await.unwrap();
    assert_eq!(&coll.key, key);
}

#[tokio::test]
#[ignore]
async fn test_live_list_collection_items() {
    let list_resp = client()
        .list_collections(&CollectionListParams::builder().limit(1).build())
        .await
        .unwrap();
    if let Some(coll) = list_resp.items.first() {
        let items = client()
            .list_collection_items(&coll.key, &ItemListParams::builder().limit(1).build())
            .await
            .unwrap();
        assert!(items.total_results.is_some());
    }
}

// ── Live tag tests ───────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_list_tags() {
    let params = TagListParams::builder().limit(5).build();
    let resp = client().list_tags(&params).await.unwrap();
    // Note: Total-Results may be 0 for tags even when items are returned
    assert!(resp.total_results.is_some());
    assert!(!resp.items.is_empty());
}

#[tokio::test]
#[ignore]
async fn test_live_list_items_tags() {
    let resp = client()
        .list_items_tags(&TagListParams::builder().limit(5).build())
        .await
        .unwrap();
    assert!(!resp.items.is_empty());
}

// ── Live search tests ────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_list_searches() {
    let resp = client().list_searches().await.unwrap();
    // May be empty if no saved searches
    assert!(resp.total_results.is_some());
}

// ── Live group tests ─────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_list_groups() {
    let resp = client().list_groups().await.unwrap();
    // May be empty if user has no groups
    assert!(resp.total_results.is_some());
}

// ── Live key info test ───────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn test_live_key_info() {
    let info = client().get_key_info().await.unwrap();
    assert!(info.get("userID").is_some() || info.get("key").is_some());
}
