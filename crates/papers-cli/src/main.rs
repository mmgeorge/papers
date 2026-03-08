mod cli;
mod format;

use clap::Parser;
use cli::{
    AdvancedMode, AuthorCommand, AuthorFilterArgs, Cli, ConfigCommand, ConfigSetCommand,
    DomainCommand, DomainFilterArgs, EntityCommand, FieldCommand, FieldFilterArgs, FunderCommand,
    FunderFilterArgs, InstitutionCommand, InstitutionFilterArgs, McpCommand, PublisherCommand,
    PublisherFilterArgs, DbChapterCommand, DbChunkCommand, DbCommand, DbEmbedCommand,
    DbExhibitCommand, DbSectionCommand, DbTagCommand, DbWorkCommand, SelectionCommand,
    SelectionCollectionCommand, SelectionDbCommand,
    SourceCommand,
    SourceFilterArgs, SubfieldCommand, SubfieldFilterArgs, TopicCommand, TopicFilterArgs,
    WorkCommand, WorkFilterArgs, ZoteroAnnotationCommand, ZoteroAttachmentCommand,
    ZoteroCollectionCommand, ZoteroCommand, ZoteroDeletedCommand,
    ZoteroGroupCommand, ZoteroNoteCommand, ZoteroPermissionCommand, ZoteroSearchCommand,
    ZoteroSettingCommand, ZoteroTagCommand, ZoteroWorkCommand,
};
use papers_core::zotero::{resolve_collection_key, resolve_item_key, resolve_search_key};
use papers_core::{
    AuthorListParams, DiskCache, DomainListParams, FieldListParams, FindWorksParams,
    FunderListParams, GetParams, InstitutionListParams, OpenAlexClient, PublisherListParams,
    SourceListParams, SubfieldListParams, TopicListParams, WorkListParams, filter::FilterError,
};
use papers_zotero::{
    CollectionListParams, DeletedParams, Item, ItemListParams, TagListParams, ZoteroClient,
};
use std::time::Duration;

async fn zotero_client() -> Result<ZoteroClient, papers_zotero::ZoteroError> {
    ZoteroClient::from_env_prefer_local().await
}

/// Returns the Zotero client when available, `Ok(None)` when Zotero is simply
/// not configured (env vars absent), or `Err` when Zotero is installed but not
/// running (so the caller can surface the error).
async fn optional_zotero() -> Result<Option<ZoteroClient>, papers_zotero::ZoteroError> {
    match ZoteroClient::from_env_prefer_local().await {
        Ok(z) => Ok(Some(z)),
        Err(e @ papers_zotero::ZoteroError::NotRunning { .. }) => Err(e),
        Err(_) => Ok(None),
    }
}

fn work_list_params(args: &cli::ListArgs, wf: &WorkFilterArgs) -> WorkListParams {
    WorkListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        author: wf.author.clone(),
        topic: wf.topic.clone(),
        domain: wf.domain.clone(),
        field: wf.field.clone(),
        subfield: wf.subfield.clone(),
        publisher: wf.publisher.clone(),
        source: wf.source.clone(),
        institution: wf.institution.clone(),
        year: wf.year.clone(),
        citations: wf.citations.clone(),
        country: wf.country.clone(),
        continent: wf.continent.clone(),
        r#type: wf.entity_type.clone(),
        open: if wf.open { Some(true) } else { None },
    }
}

fn author_list_params(args: &cli::ListArgs, af: &AuthorFilterArgs) -> AuthorListParams {
    AuthorListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        institution: af.institution.clone(),
        country: af.country.clone(),
        continent: af.continent.clone(),
        citations: af.citations.clone(),
        works: af.works.clone(),
        h_index: af.h_index.clone(),
    }
}

fn source_list_params(args: &cli::ListArgs, sf: &SourceFilterArgs) -> SourceListParams {
    SourceListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        publisher: sf.publisher.clone(),
        country: sf.country.clone(),
        continent: sf.continent.clone(),
        r#type: sf.entity_type.clone(),
        open: if sf.open { Some(true) } else { None },
        citations: sf.citations.clone(),
        works: sf.works.clone(),
    }
}

fn institution_list_params(
    args: &cli::ListArgs,
    inf: &InstitutionFilterArgs,
) -> InstitutionListParams {
    InstitutionListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        country: inf.country.clone(),
        continent: inf.continent.clone(),
        r#type: inf.entity_type.clone(),
        citations: inf.citations.clone(),
        works: inf.works.clone(),
    }
}

fn topic_list_params(args: &cli::ListArgs, tf: &TopicFilterArgs) -> TopicListParams {
    TopicListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        domain: tf.domain.clone(),
        field: tf.field.clone(),
        subfield: tf.subfield.clone(),
        citations: tf.citations.clone(),
        works: tf.works.clone(),
    }
}

fn publisher_list_params(args: &cli::ListArgs, pf: &PublisherFilterArgs) -> PublisherListParams {
    PublisherListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        country: pf.country.clone(),
        continent: pf.continent.clone(),
        citations: pf.citations.clone(),
        works: pf.works.clone(),
    }
}

fn funder_list_params(args: &cli::ListArgs, ff: &FunderFilterArgs) -> FunderListParams {
    FunderListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        country: ff.country.clone(),
        continent: ff.continent.clone(),
        citations: ff.citations.clone(),
        works: ff.works.clone(),
    }
}

fn domain_list_params(args: &cli::ListArgs, df: &DomainFilterArgs) -> DomainListParams {
    DomainListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        works: df.works.clone(),
    }
}

fn field_list_params(args: &cli::ListArgs, ff: &FieldFilterArgs) -> FieldListParams {
    FieldListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        domain: ff.domain.clone(),
        works: ff.works.clone(),
    }
}

fn subfield_list_params(args: &cli::ListArgs, sf: &SubfieldFilterArgs) -> SubfieldListParams {
    SubfieldListParams {
        search: None,
        filter: args.filter.clone(),
        sort: args.sort.clone(),
        per_page: Some(args.per_page),
        page: args.page,
        cursor: args.cursor.clone(),
        sample: args.sample,
        seed: args.seed,
        select: None,
        group_by: None,
        domain: sf.domain.clone(),
        field: sf.field.clone(),
        works: sf.works.clone(),
    }
}

fn print_json<T: serde::Serialize>(val: &T) {
    println!(
        "{}",
        serde_json::to_string_pretty(val).expect("JSON serialization failed")
    );
}

/// Returns true if this attachment can have annotation children (PDF, EPUB, or HTML snapshot).
fn is_annotatable_attachment(att: &Item) -> bool {
    matches!(
        att.data.content_type.as_deref(),
        Some("application/pdf") | Some("application/epub+zip") | Some("text/html")
    )
}

fn exit_err(msg: &str) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1);
}

/// Find the first PDF attachment key for a given item key.
async fn find_pdf_attachment_key(zotero: &ZoteroClient, item_key: &str) -> Result<String, String> {
    Ok(find_pdf_attachment(zotero, item_key).await?.key)
}

async fn find_pdf_attachment(zotero: &ZoteroClient, item_key: &str) -> Result<Item, String> {
    let att_params = ItemListParams {
        item_type: Some("attachment".into()),
        ..Default::default()
    };
    let children = zotero
        .list_item_children(item_key, &att_params)
        .await
        .map_err(|e| e.to_string())?;
    children
        .items
        .into_iter()
        .find(|a| a.data.content_type.as_deref() == Some("application/pdf"))
        .ok_or_else(|| format!("No PDF attachment found for item {item_key}"))
}

async fn run_extraction_for_key(
    zotero: &ZoteroClient,
    key: &str,
    mode: AdvancedMode,
) -> Result<(), String> {
    let att = find_pdf_attachment(zotero, key).await?;
    let filename = att
        .data
        .filename
        .ok_or_else(|| "attachment has no filename".to_string())?;
    let local_path = dirs::home_dir()
        .ok_or_else(|| "cannot determine home dir".to_string())?
        .join("Zotero")
        .join("storage")
        .join(&att.key)
        .join(&filename);
    let pdf_bytes = std::fs::read(&local_path)
        .map_err(|e| format!("failed to read {}: {e}", local_path.display()))?;
    let dl = papers_datalab::DatalabClient::from_env().map_err(|e| e.to_string())?;
    let processing_mode = match mode {
        AdvancedMode::Fast => papers_core::text::ProcessingMode::Fast,
        AdvancedMode::Balanced => papers_core::text::ProcessingMode::Balanced,
        AdvancedMode::Accurate => papers_core::text::ProcessingMode::Accurate,
    };
    let mut source = papers_core::text::PdfSource::ZoteroLocal {
        path: local_path.to_string_lossy().into_owned(),
    };
    papers_core::text::do_extract(pdf_bytes, key, Some(zotero), Some((&dl, processing_mode)), &mut source)
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

fn looks_like_doi(s: &str) -> bool {
    let s = s
        .strip_prefix("https://doi.org/")
        .or_else(|| s.strip_prefix("http://doi.org/"))
        .or_else(|| s.strip_prefix("doi:"))
        .unwrap_or(s);
    s.starts_with("10.") && s.contains('/')
}

/// Parse the title from the first `# ` heading in a locally-cached markdown file.
/// Falls back to the item key if no heading is found.

async fn smart_resolve_item_key(zotero: &ZoteroClient, input: &str) -> Result<String, String> {
    if papers_core::zotero::looks_like_zotero_key(input) {
        return Ok(input.to_string());
    }
    let params = if looks_like_doi(input) {
        ItemListParams {
            q: Some(input.to_string()),
            qmode: Some("everything".into()),
            limit: Some(1),
            ..Default::default()
        }
    } else {
        ItemListParams::builder().q(input).limit(1).build()
    };
    let resp = zotero
        .list_top_items(&params)
        .await
        .map_err(|e| e.to_string())?;
    resp.items
        .into_iter()
        .next()
        .map(|i| i.key)
        .ok_or_else(|| format!("No item found matching {:?}", input))
}

/// The entry point spawns the Tokio runtime on a thread with an explicit
/// 8 MB stack.  Windows' default main-thread stack is only 1 MB, which is
/// not enough for the large async state machine generated by the main
/// dispatch function (multiple `HashSet`/`Vec`/`HashMap` locals across many
/// await points inside a single giant `match`).
fn main() {
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .name("papers-main".into())
        .spawn(|| {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build Tokio runtime")
                .block_on(papers_main())
        })
        .expect("failed to spawn main thread")
        .join()
        .expect("main thread panicked");
}

async fn papers_main() {
    let cli = Cli::parse();
    let mut client = OpenAlexClient::new();
    if let Ok(cache) = DiskCache::default_location(Duration::from_secs(600)) {
        client = client.with_cache(cache);
    }

    match cli.entity {
        EntityCommand::Work { cmd } => match cmd {
            WorkCommand::List { args, work_filters } => {
                let params = work_list_params(&args, &work_filters);
                match papers_core::api::work_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_work_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            WorkCommand::Search { query, args, work_filters } => {
                let mut params = work_list_params(&args, &work_filters);
                params.search = Some(query);
                match papers_core::api::work_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_work_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            WorkCommand::Get { id, json } => {
                let zotero = optional_zotero()
                    .await
                    .unwrap_or_else(|e| exit_err(&e.to_string()));
                let zotero_configured = zotero.is_some();
                match papers_core::api::work_get_response(
                    &client,
                    zotero.as_ref(),
                    &id,
                    &GetParams::default(),
                )
                .await
                {
                    Ok(response) => {
                        if json {
                            print_json(&response);
                        } else {
                            print!(
                                "{}",
                                format::format_work_get_response(&response, zotero_configured)
                            );
                        }
                    }
                    Err(FilterError::Suggestions { query, suggestions }) if json => {
                        let candidates: Vec<_> = suggestions
                            .into_iter()
                            .map(|(name, citations)| serde_json::json!({"name": name, "citations": citations}))
                            .collect();
                        print_json(&serde_json::json!({
                            "message": "no_exact_match",
                            "query": query,
                            "candidates": candidates,
                        }));
                        std::process::exit(1);
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            WorkCommand::Autocomplete { query, json } => {
                match papers_core::api::work_autocomplete(&client, &query).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_autocomplete(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            WorkCommand::Find {
                query,
                count,
                filter,
                json,
            } => {
                if std::env::var("OPENALEX_KEY").is_err() {
                    exit_err("work find requires an API key. Set OPENALEX_KEY=<your-key>.");
                }
                let params = FindWorksParams {
                    query,
                    count,
                    filter,
                };
                match papers_core::api::work_find(&client, &params).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_find_works(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Author { cmd } => match cmd {
            AuthorCommand::List { args, filters } => {
                let params = author_list_params(&args, &filters);
                match papers_core::api::author_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_author_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            AuthorCommand::Search { query, args, filters } => {
                let mut params = author_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::author_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_author_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            AuthorCommand::Get { id, json } => {
                match papers_core::api::author_get(&client, &id, &GetParams::default()).await {
                    Ok(author) => {
                        if json {
                            print_json(&author);
                        } else {
                            print!("{}", format::format_author_get(&author));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            AuthorCommand::Autocomplete { query, json } => {
                match papers_core::api::author_autocomplete(&client, &query).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_autocomplete(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Source { cmd } => match cmd {
            SourceCommand::List { args, filters } => {
                let params = source_list_params(&args, &filters);
                match papers_core::api::source_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_source_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            SourceCommand::Search { query, args, filters } => {
                let mut params = source_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::source_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_source_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            SourceCommand::Get { id, json } => {
                match papers_core::api::source_get(&client, &id, &GetParams::default()).await {
                    Ok(source) => {
                        if json {
                            print_json(&source);
                        } else {
                            print!("{}", format::format_source_get(&source));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            SourceCommand::Autocomplete { query, json } => {
                match papers_core::api::source_autocomplete(&client, &query).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_autocomplete(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Institution { cmd } => match cmd {
            InstitutionCommand::List { args, filters } => {
                let params = institution_list_params(&args, &filters);
                match papers_core::api::institution_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_institution_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            InstitutionCommand::Search { query, args, filters } => {
                let mut params = institution_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::institution_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_institution_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            InstitutionCommand::Get { id, json } => {
                match papers_core::api::institution_get(&client, &id, &GetParams::default()).await {
                    Ok(inst) => {
                        if json {
                            print_json(&inst);
                        } else {
                            print!("{}", format::format_institution_get(&inst));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            InstitutionCommand::Autocomplete { query, json } => {
                match papers_core::api::institution_autocomplete(&client, &query).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_autocomplete(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Topic { cmd } => match cmd {
            TopicCommand::List { args, filters } => {
                let params = topic_list_params(&args, &filters);
                match papers_core::api::topic_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_topic_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            TopicCommand::Search { query, args, filters } => {
                let mut params = topic_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::topic_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_topic_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            TopicCommand::Get { id, json } => {
                match papers_core::api::topic_get(&client, &id, &GetParams::default()).await {
                    Ok(topic) => {
                        if json {
                            print_json(&topic);
                        } else {
                            print!("{}", format::format_topic_get(&topic));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Publisher { cmd } => match cmd {
            PublisherCommand::List { args, filters } => {
                let params = publisher_list_params(&args, &filters);
                match papers_core::api::publisher_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_publisher_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            PublisherCommand::Search { query, args, filters } => {
                let mut params = publisher_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::publisher_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_publisher_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            PublisherCommand::Get { id, json } => {
                match papers_core::api::publisher_get(&client, &id, &GetParams::default()).await {
                    Ok(pub_) => {
                        if json {
                            print_json(&pub_);
                        } else {
                            print!("{}", format::format_publisher_get(&pub_));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            PublisherCommand::Autocomplete { query, json } => {
                match papers_core::api::publisher_autocomplete(&client, &query).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_autocomplete(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Funder { cmd } => match cmd {
            FunderCommand::List { args, filters } => {
                let params = funder_list_params(&args, &filters);
                match papers_core::api::funder_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_funder_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            FunderCommand::Search { query, args, filters } => {
                let mut params = funder_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::funder_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_funder_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            FunderCommand::Get { id, json } => {
                match papers_core::api::funder_get(&client, &id, &GetParams::default()).await {
                    Ok(funder) => {
                        if json {
                            print_json(&funder);
                        } else {
                            print!("{}", format::format_funder_get(&funder));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            FunderCommand::Autocomplete { query, json } => {
                match papers_core::api::funder_autocomplete(&client, &query).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_autocomplete(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Domain { cmd } => match cmd {
            DomainCommand::List { args, filters } => {
                let params = domain_list_params(&args, &filters);
                match papers_core::api::domain_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_domain_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            DomainCommand::Search { query, args, filters } => {
                let mut params = domain_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::domain_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_domain_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            DomainCommand::Get { id, json } => {
                match papers_core::api::domain_get(&client, &id, &GetParams::default()).await {
                    Ok(domain) => {
                        if json {
                            print_json(&domain);
                        } else {
                            print!("{}", format::format_domain_get(&domain));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Field { cmd } => match cmd {
            FieldCommand::List { args, filters } => {
                let params = field_list_params(&args, &filters);
                match papers_core::api::field_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_field_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            FieldCommand::Search { query, args, filters } => {
                let mut params = field_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::field_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_field_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            FieldCommand::Get { id, json } => {
                match papers_core::api::field_get(&client, &id, &GetParams::default()).await {
                    Ok(field) => {
                        if json {
                            print_json(&field);
                        } else {
                            print!("{}", format::format_field_get(&field));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Subfield { cmd } => match cmd {
            SubfieldCommand::List { args, filters } => {
                let params = subfield_list_params(&args, &filters);
                match papers_core::api::subfield_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_subfield_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            SubfieldCommand::Search { query, args, filters } => {
                let mut params = subfield_list_params(&args, &filters);
                params.search = Some(query);
                match papers_core::api::subfield_list(&client, &params).await {
                    Ok(resp) => {
                        if args.json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_subfield_list(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            SubfieldCommand::Get { id, json } => {
                match papers_core::api::subfield_get(&client, &id, &GetParams::default()).await {
                    Ok(subfield) => {
                        if json {
                            print_json(&subfield);
                        } else {
                            print!("{}", format::format_subfield_get(&subfield));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
            SubfieldCommand::Autocomplete { query, json } => {
                match papers_core::api::subfield_autocomplete(&client, &query).await {
                    Ok(resp) => {
                        if json {
                            print_json(&resp);
                        } else {
                            print!("{}", format::format_autocomplete(&resp));
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        EntityCommand::Zotero { cmd } => {
            let zotero = zotero_client().await.unwrap_or_else(|e| match e {
                papers_zotero::ZoteroError::NotRunning { path } => exit_err(&format!(
                    "Zotero is installed ({path}) but the local API is not enabled.\n\
                     Fix: Zotero → Settings → Advanced → check \"Enable Local API\".\n\
                     Or set ZOTERO_CHECK_LAUNCHED=0 to skip this check and use the remote web API."
                )),
                _ => exit_err("Zotero not configured. Set ZOTERO_USER_ID and ZOTERO_API_KEY."),
            });
            match cmd {
                ZoteroCommand::Work { cmd } => match cmd {
                    ZoteroWorkCommand::List {
                        tag,
                        type_,
                        sort,
                        direction,
                        limit,
                        start,
                        since,
                        json,
                    } => {
                        let params = ItemListParams {
                            item_type: type_,
                            tag,
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            since,
                            ..Default::default()
                        };
                        match zotero.list_top_items(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_work_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Search {
                        query,
                        everything,
                        tag,
                        type_,
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let params = ItemListParams {
                            item_type: type_,
                            q: Some(query),
                            qmode: everything.then(|| "everything".to_string()),
                            tag,
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_top_items(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_work_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json {
                                    print_json(&item);
                                } else {
                                    print!("{}", format::format_zotero_item_get(&item));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Collections { key, json } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let item = zotero
                            .get_item(&key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let col_keys = item.data.collections.clone();
                        let mut collections = Vec::new();
                        for ck in &col_keys {
                            match zotero.get_collection(ck).await {
                                Ok(c) => collections.push(c),
                                Err(e) => exit_err(&e.to_string()),
                            }
                        }
                        if json {
                            print_json(&collections);
                        } else {
                            print!(
                                "{}",
                                format::format_zotero_collection_list_vec(&collections)
                            );
                        }
                    }
                    ZoteroWorkCommand::Notes {
                        key,
                        limit,
                        start,
                        json,
                    } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams {
                            item_type: Some("note".into()),
                            limit,
                            start,
                            ..Default::default()
                        };
                        match zotero.list_item_children(&key, &params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_note_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Attachments {
                        key,
                        limit,
                        start,
                        json,
                    } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams {
                            item_type: Some("attachment".into()),
                            limit,
                            start,
                            ..Default::default()
                        };
                        match zotero.list_item_children(&key, &params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_attachment_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Annotations { key, json } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_params = ItemListParams {
                            item_type: Some("attachment".into()),
                            ..Default::default()
                        };
                        let attachments = zotero
                            .list_item_children(&key, &att_params)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let ann_params = ItemListParams {
                            item_type: Some("annotation".into()),
                            ..Default::default()
                        };
                        let mut all_annotations = Vec::new();
                        for att in &attachments.items {
                            if !is_annotatable_attachment(att) {
                                continue;
                            }
                            match zotero.list_item_children(&att.key, &ann_params).await {
                                Ok(r) => all_annotations.extend(r.items),
                                Err(_) => {}
                            }
                        }
                        if json {
                            print_json(&all_annotations);
                        } else {
                            print!(
                                "{}",
                                format::format_zotero_annotation_list_vec(&all_annotations)
                            );
                        }
                    }
                    ZoteroWorkCommand::Tags {
                        key,
                        search,
                        limit,
                        start,
                        json,
                    } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = TagListParams {
                            q: search,
                            qmode: Some("contains".to_string()),
                            limit,
                            start,
                            ..Default::default()
                        };
                        match zotero.list_item_tags(&key, &params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_tag_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Text { key, json } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_key = find_pdf_attachment_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e));
                        match zotero.get_item_fulltext(&att_key).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_work_fulltext(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::ViewUrl { key } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_key = find_pdf_attachment_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e));
                        match zotero.get_item_file_view_url(&att_key).await {
                            Ok(url) => println!("{url}"),
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::View { key, output } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_key = find_pdf_attachment_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e));
                        match zotero.get_item_file_view(&att_key).await {
                            Ok(bytes) => {
                                if output == "-" {
                                    use std::io::Write;
                                    std::io::stdout()
                                        .write_all(&bytes)
                                        .unwrap_or_else(|e| exit_err(&e.to_string()));
                                } else {
                                    std::fs::write(&output, &bytes)
                                        .unwrap_or_else(|e| exit_err(&e.to_string()));
                                    eprintln!("Saved {} bytes to {output}", bytes.len());
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Attachment { cmd } => match cmd {
                    ZoteroAttachmentCommand::List {
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let params = ItemListParams {
                            item_type: Some("attachment".into()),
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_attachment_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAttachmentCommand::Search {
                        query,
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let params = ItemListParams {
                            item_type: Some("attachment".into()),
                            q: Some(query),
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_attachment_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAttachmentCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json {
                                    print_json(&item);
                                } else {
                                    print!("{}", format::format_zotero_item_get(&item));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAttachmentCommand::File { key, output } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.download_item_file(&key).await {
                            Ok(bytes) => {
                                if output == "-" {
                                    use std::io::Write;
                                    std::io::stdout()
                                        .write_all(&bytes)
                                        .unwrap_or_else(|e| exit_err(&e.to_string()));
                                } else {
                                    std::fs::write(&output, &bytes)
                                        .unwrap_or_else(|e| exit_err(&e.to_string()));
                                    eprintln!("Saved {} bytes to {output}", bytes.len());
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAttachmentCommand::Url { key } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item_file_view_url(&key).await {
                            Ok(url) => println!("{url}"),
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Annotation { cmd } => match cmd {
                    ZoteroAnnotationCommand::List { limit, start, json } => {
                        let params = ItemListParams {
                            item_type: Some("annotation".into()),
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_annotation_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAnnotationCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json {
                                    print_json(&item);
                                } else {
                                    print!("{}", format::format_zotero_item_get(&item));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Note { cmd } => match cmd {
                    ZoteroNoteCommand::List {
                        limit,
                        start,
                        json,
                    } => {
                        let params = ItemListParams {
                            item_type: Some("note".into()),
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_note_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroNoteCommand::Search {
                        query,
                        limit,
                        start,
                        json,
                    } => {
                        let params = ItemListParams {
                            item_type: Some("note".into()),
                            q: Some(query),
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_note_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroNoteCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json {
                                    print_json(&item);
                                } else {
                                    print!("{}", format::format_zotero_item_get(&item));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Collection { cmd } => match cmd {
                    ZoteroCollectionCommand::List {
                        sort,
                        direction,
                        limit,
                        start,
                        top,
                        json,
                    } => {
                        let params = CollectionListParams {
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                        };
                        let result = if top {
                            zotero.list_top_collections(&params).await
                        } else {
                            zotero.list_collections(&params).await
                        };
                        match result {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_collection_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Get { key, json } => {
                        let key = resolve_collection_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_collection(&key).await {
                            Ok(coll) => {
                                if json {
                                    print_json(&coll);
                                } else {
                                    print!("{}", format::format_zotero_collection_get(&coll));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Works {
                        key,
                        search,
                        everything,
                        tag,
                        type_,
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let key = resolve_collection_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams {
                            item_type: type_,
                            q: search,
                            qmode: everything.then(|| "everything".to_string()),
                            tag,
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_collection_top_items(&key, &params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_work_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Attachments {
                        key,
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let key = resolve_collection_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams {
                            item_type: Some("attachment".into()),
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_collection_items(&key, &params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_attachment_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Notes {
                        key,
                        search,
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let key = resolve_collection_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams {
                            item_type: Some("note".into()),
                            q: search,
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        match zotero.list_collection_items(&key, &params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_note_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Annotations { key, json } => {
                        let key = resolve_collection_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_params = ItemListParams {
                            item_type: Some("attachment".into()),
                            ..Default::default()
                        };
                        let attachments = zotero
                            .list_collection_items(&key, &att_params)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let ann_params = ItemListParams {
                            item_type: Some("annotation".into()),
                            ..Default::default()
                        };
                        let mut all_annotations = Vec::new();
                        for att in &attachments.items {
                            if !is_annotatable_attachment(att) {
                                continue;
                            }
                            match zotero.list_item_children(&att.key, &ann_params).await {
                                Ok(r) => all_annotations.extend(r.items),
                                Err(_) => {}
                            }
                        }
                        if json {
                            print_json(&all_annotations);
                        } else {
                            print!(
                                "{}",
                                format::format_zotero_annotation_list_vec(&all_annotations)
                            );
                        }
                    }
                    ZoteroCollectionCommand::Subcollections {
                        key,
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let key = resolve_collection_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = CollectionListParams {
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                        };
                        match zotero.list_subcollections(&key, &params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_collection_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Tags {
                        key,
                        search,
                        limit,
                        start,
                        top,
                        json,
                    } => {
                        let key = resolve_collection_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = TagListParams {
                            q: search,
                            qmode: Some("contains".to_string()),
                            limit,
                            start,
                            ..Default::default()
                        };
                        let result = if top {
                            zotero.list_collection_top_items_tags(&key, &params).await
                        } else {
                            zotero.list_collection_items_tags(&key, &params).await
                        };
                        match result {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_tag_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Tag { cmd } => match cmd {
                    ZoteroTagCommand::List {
                        sort,
                        direction,
                        limit,
                        start,
                        top,
                        trash,
                        json,
                    } => {
                        let params = TagListParams {
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                            ..Default::default()
                        };
                        let result = if trash {
                            zotero.list_trash_tags(&params).await
                        } else if top {
                            zotero.list_top_items_tags(&params).await
                        } else {
                            zotero.list_tags(&params).await
                        };
                        match result {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_tag_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroTagCommand::Search {
                        query,
                        sort,
                        direction,
                        limit,
                        start,
                        json,
                    } => {
                        let params = TagListParams {
                            q: Some(query),
                            qmode: Some("contains".to_string()),
                            sort,
                            direction,
                            limit: Some(limit),
                            start,
                        };
                        match zotero.list_tags(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_tag_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroTagCommand::Get { name, json } => match zotero.get_tag(&name).await {
                        Ok(resp) => {
                            if json {
                                print_json(&resp);
                            } else {
                                print!("{}", format::format_zotero_tag_list(&resp));
                            }
                        }
                        Err(e) => exit_err(&e.to_string()),
                    },
                },

                ZoteroCommand::Search { cmd } => match cmd {
                    ZoteroSearchCommand::List { json } => match zotero.list_searches().await {
                        Ok(resp) => {
                            if json {
                                print_json(&resp);
                            } else {
                                print!("{}", format::format_zotero_search_list(&resp));
                            }
                        }
                        Err(e) => exit_err(&e.to_string()),
                    },
                    ZoteroSearchCommand::Get { key, json } => {
                        let key = resolve_search_key(&zotero, &key)
                            .await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_search(&key).await {
                            Ok(search) => {
                                if json {
                                    print_json(&search);
                                } else {
                                    print!("{}", format::format_zotero_search_get(&search));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Group { cmd } => match cmd {
                    ZoteroGroupCommand::List { json } => match zotero.list_groups().await {
                        Ok(resp) => {
                            if json {
                                print_json(&resp);
                            } else {
                                print!("{}", format::format_zotero_group_list(&resp));
                            }
                        }
                        Err(e) => exit_err(&e.to_string()),
                    },
                },

                ZoteroCommand::Setting { cmd } => match cmd {
                    ZoteroSettingCommand::List { json } => match zotero.get_settings().await {
                        Ok(resp) => {
                            if json {
                                print_json(&resp);
                            } else {
                                print!("{}", format::format_zotero_setting_list(&resp));
                            }
                        }
                        Err(e) => exit_err(&e.to_string()),
                    },
                    ZoteroSettingCommand::Get { key, json } => {
                        match zotero.get_setting(&key).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_setting_get(&key, &resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Deleted { cmd } => match cmd {
                    ZoteroDeletedCommand::List { since, json } => {
                        let params = DeletedParams { since };
                        match zotero.get_deleted(&params).await {
                            Ok(resp) => {
                                if json {
                                    print_json(&resp);
                                } else {
                                    print!("{}", format::format_zotero_deleted_list(&resp));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Permission { cmd } => match cmd {
                    ZoteroPermissionCommand::List { json } => {
                        match zotero.get_current_key_info().await {
                            Ok(info) => {
                                if json {
                                    print_json(&info);
                                } else {
                                    print!("{}", format::format_zotero_permission_list(&info));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },
            }
        }

        EntityCommand::Selection { cmd } => {
            handle_selection_command(cmd, &client).await;
        }
        EntityCommand::Db { cmd } => {
            handle_db_command(cmd).await;
        }
        EntityCommand::Extract {
            pdf,
            output,
            page,
            skip_images,
            formula,
            write_layout,
        } => {
            let output_dir = output.unwrap_or_else(|| {
                pdf.parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("."))
            });

            let options = papers_extract::ExtractOptions {
                extract_images: !skip_images,
                page,
                formula: match formula {
                    cli::FormulaModelArg::PpFormulanet => papers_extract::FormulaModel::PpFormulanet,
                    cli::FormulaModelArg::GlmOcr => papers_extract::FormulaModel::GlmOcr,
                },
                debug: match write_layout {
                    Some(cli::LayoutDebugArg::Images) => papers_extract::DebugMode::Images,
                    Some(cli::LayoutDebugArg::Pdf) => papers_extract::DebugMode::Pdf,
                    None => papers_extract::DebugMode::Off,
                },
                ..papers_extract::ExtractOptions::default()
            };

            eprintln!("Extracting {} → {}", pdf.display(), output_dir.display());

            let result = tokio::task::spawn_blocking(move || {
                papers_extract::extract(&pdf, &output_dir, &options)
            })
            .await
            .unwrap();

            match result {
                Ok(result) => {
                    eprintln!(
                        "Done: {} pages, {} regions, {}ms",
                        result.metadata.page_count,
                        result.pages.iter().map(|p| p.regions.len()).sum::<usize>(),
                        result.metadata.extraction_time_ms,
                    );
                }
                Err(e) => exit_err(&format!("{e}")),
            }
        }
        EntityCommand::Config { cmd } => {
            handle_config_command(cmd);
        }
        EntityCommand::Mcp { cmd } => {
            handle_mcp_command(cmd).await;
        }
    }
}

async fn open_db_store() -> papers_db::DbStore {
    let path = papers_db::DbStore::default_path();
    match papers_db::DbStore::open(&path).await {
        Ok(store) => store,
        Err(e) => exit_err(&format!("Failed to open RAG database: {e}")),
    }
}

async fn handle_db_command(cmd: DbCommand) {
    match cmd {
        DbCommand::Chunk { cmd } => match cmd {
            DbChunkCommand::Search {
                query, selection, work, chapter_idx, section_idx,
                year_min, year_max, venue, tag, depth, limit, json,
            } => {
                let rag = open_db_store().await;
            let paper_ids = match selection.as_deref() {
                Some(sel) => match papers_core::selection::load_selection(sel) {
                    Ok(s) => Some(s.entries.iter().flat_map(|e| {
                        e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
                    }).collect()),
                    Err(e) => exit_err(&e.to_string()),
                },
                None => match work {
                    Some(id) => {
                        let resolved = match papers_db::resolve_paper_id(&rag, &id).await {
                            Ok(r) => r,
                            Err(e) => exit_err(&e.to_string()),
                        };
                        Some(vec![resolved])
                    }
                    None => None,
                },
            };
                let params = papers_db::SearchParams {
                    query, paper_ids, chapter_idx, section_idx,
                    filter_year_min: year_min, filter_year_max: year_max,
                    filter_venue: venue, filter_tags: tag, filter_depth: depth, limit,
                };
                match papers_db::query::search(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_search(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbChunkCommand::Get { chunk_id, json } => {
                let rag = open_db_store().await;
                match papers_db::query::get_chunk(&rag, &chunk_id).await {
                    Ok(result) => { if json { print_json(&result); } else { format_db_chunk_result(&result); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbChunkCommand::List { work, chapter_idx, section_idx, limit, json } => {
                let rag = open_db_store().await;
                let paper_id = match work {
                    Some(ref id) => {
                        let resolved = match papers_db::resolve_paper_id(&rag, id).await {
                            Ok(r) => r,
                            Err(e) => exit_err(&e.to_string()),
                        };
                        Some(resolved)
                    }
                    None => None,
                };
                let params = papers_db::ListChunksParams { paper_id, chapter_idx, section_idx, limit };
                match papers_db::query::list_chunks(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_chunk_list(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        DbCommand::Exhibit { cmd } => match cmd {
            DbExhibitCommand::Search { query, selection, work, exhibit_type, limit, json } => {
                let rag = open_db_store().await;
            let paper_ids = match selection.as_deref() {
                Some(sel) => match papers_core::selection::load_selection(sel) {
                    Ok(s) => Some(s.entries.iter().flat_map(|e| {
                        e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
                    }).collect()),
                    Err(e) => exit_err(&e.to_string()),
                },
                None => match work {
                    Some(id) => {
                        let resolved = match papers_db::resolve_paper_id(&rag, &id).await {
                            Ok(r) => r,
                            Err(e) => exit_err(&e.to_string()),
                        };
                        Some(vec![resolved])
                    }
                    None => None,
                },
            };
                let params = papers_db::SearchExhibitsParams {
                    query, paper_ids, filter_exhibit_type: exhibit_type, limit,
                };
                match papers_db::query::search_exhibits(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_exhibits(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbExhibitCommand::Get { exhibit_id, json } => {
                let rag = open_db_store().await;
                match papers_db::query::get_exhibit(&rag, &exhibit_id).await {
                    Ok(result) => { if json { print_json(&result); } else { format_db_exhibit(&result); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        DbCommand::Work { cmd } => match cmd {
            DbWorkCommand::List {
                selection, year_min, year_max, venue, tag, author, sort, limit, json,
            } => {
                let rag = open_db_store().await;
            let paper_ids = match selection.as_deref() {
                Some(sel) => match papers_core::selection::load_selection(sel) {
                    Ok(s) => Some(s.entries.iter().flat_map(|e| {
                        e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
                    }).collect()),
                    Err(e) => exit_err(&e.to_string()),
                },
                None => None,
            };
                let params = papers_db::ListPapersParams {
                    paper_ids, filter_year_min: year_min, filter_year_max: year_max,
                    filter_venue: venue, filter_tags: tag, filter_authors: author,
                    sort_by: sort, limit,
                };
                match papers_db::query::list_papers(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_papers(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbWorkCommand::Get { paper_id, json } => {
                let rag = open_db_store().await;
                let paper_id = match papers_db::resolve_paper_id(&rag, &paper_id).await {
                    Ok(r) => r,
                    Err(e) => exit_err(&e.to_string()),
                };
                match papers_db::query::get_work(&rag, &paper_id).await {
                    Ok(result) => { if json { print_json(&result); } else { format_db_work_metadata(&result); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbWorkCommand::Search { query, selection, year_min, year_max, venue, tag, limit, json } => {
                let rag = open_db_store().await;
            let paper_ids = match selection.as_deref() {
                Some(sel) => match papers_core::selection::load_selection(sel) {
                    Ok(s) => Some(s.entries.iter().flat_map(|e| {
                        e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
                    }).collect()),
                    Err(e) => exit_err(&e.to_string()),
                },
                None => None,
            };
                let params = papers_db::SearchWorksParams {
                    query, paper_ids, filter_year_min: year_min, filter_year_max: year_max,
                    filter_venue: venue, filter_tags: tag, limit,
                };
                match papers_db::query::search_works(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_work_search(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbWorkCommand::Add { work: item_key, all, tag, force, json, mode, force_extract } => {
                let rag = open_db_store().await;
                if all {
                    let keys = papers_db::list_cached_item_keys();
                    if keys.is_empty() {
                        if json {
                            print_json(&serde_json::json!({ "ingested": 0, "message": "no cached papers found" }));
                        } else {
                            println!("No cached papers found in DataLab cache.");
                        }
                        return;
                    }
                    let mut total_chunks = 0usize;
                    let mut total_exhibits = 0usize;
                    let mut ingested = 0usize;
                    let mut failed = 0usize;
                    if force_extract && !keys.is_empty() {
                        let zotero = zotero_client().await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        for key in &keys {
                            if !json { print!("  [re-extract] {key}... "); }
                            match run_extraction_for_key(&zotero, key, mode.clone()).await {
                                Ok(()) => { if !json { println!("done"); } }
                                Err(e) => { eprintln!("  [extract-fail] {key}: {e}"); failed += 1; }
                            }
                        }
                    }
                    for key in &keys {
                        let mut params = match papers_db::ingest_params_from_cache(key) {
                            Ok(p) => p,
                            Err(e) => { eprintln!("  [skip] {key}: {e}"); failed += 1; continue; }
                        };
                        params.force = force;
                        if !force && papers_db::is_ingested(&rag, &params.paper_id).await {
                            if !json { println!("  [skip] {key}: already indexed"); }
                            continue;
                        }
                        if !json { print!("  [ingest] {key}... "); }
                        match papers_db::ingest_paper(&rag, params).await {
                            Ok(stats) => {
                                total_chunks += stats.chunks_added;
                                total_exhibits += stats.exhibits_added;
                                ingested += 1;
                                if !json { println!("{} chunks, {} exhibits", stats.chunks_added, stats.exhibits_added); }
                            }
                            Err(e) => {
                                failed += 1;
                                if !json { println!("FAILED: {e}"); } else { eprintln!("  [fail] {key}: {e}"); }
                            }
                        }
                    }
                    if json {
                        print_json(&serde_json::json!({
                            "ingested": ingested, "failed": failed,
                            "total_chunks": total_chunks, "total_exhibits": total_exhibits,
                        }));
                    } else {
                        println!("Ingested {} papers: {} chunks, {} exhibits ({} failed)",
                            ingested, total_chunks, total_exhibits, failed);
                    }
                } else {
                    let input = match item_key {
                        Some(k) => k,
                        None => exit_err("work is required unless --all is set"),
                    };

                    // Fast path: exact key with existing cache — no Zotero needed
                    let key = if papers_core::zotero::looks_like_zotero_key(&input)
                        && !force_extract
                        && papers_core::text::datalab_cached_markdown(&input).is_some()
                    {
                        input
                    } else {
                        // Need to resolve: acquire Zotero client
                        let zotero = zotero_client().await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        // Resolve DOI / title / bare key to concrete Zotero item key
                        let key = smart_resolve_item_key(&zotero, &input)
                            .await
                            .unwrap_or_else(|e| exit_err(&format!(
                                "Could not resolve {:?} to a Zotero item. \
                                 Provide an exact item key (e.g. LF4MJWZK), DOI, or title. Error: {e}",
                                input
                            )));

                        // Extract if cache is missing or force_extract requested
                        let needs_extract = force_extract
                            || papers_core::text::datalab_cached_markdown(&key).is_none();
                        if needs_extract {
                            if !json { print!("  [extract] {key}... "); }
                            if let Err(e) = run_extraction_for_key(&zotero, &key, mode).await {
                                exit_err(&format!("Extraction failed for {key}: {e}"));
                            }
                            if !json { println!("done"); }
                        }
                        key
                    };

                    let mut params = match papers_db::ingest_params_from_cache(&key) {
                        Ok(p) => p,
                        Err(e) => exit_err(&format!("Failed to read cache for {key}: {e}")),
                    };
                    if let Some(tags) = tag { params.tags.extend(tags); }
                    params.force = force;
                    if !force && papers_db::is_ingested(&rag, &params.paper_id).await {
                        if json {
                            print_json(&serde_json::json!({
                                "skipped": true, "paper_id": params.paper_id,
                                "message": "already indexed; use --force to re-index"
                            }));
                        } else {
                            println!("Already indexed: {} (use --force to re-index)", params.paper_id);
                        }
                        return;
                    }
                    match papers_db::ingest_paper(&rag, params).await {
                        Ok(stats) => {
                            if json {
                                print_json(&serde_json::json!({
                                    "chunks_added": stats.chunks_added,
                                    "exhibits_added": stats.exhibits_added,
                                    "item_key": key,
                                }));
                            } else {
                                println!("Ingested {} chunks and {} exhibits for {}",
                                    stats.chunks_added, stats.exhibits_added, key);
                            }
                        }
                        Err(e) => exit_err(&e.to_string()),
                    }
                }
            }

            DbWorkCommand::Remove { paper_id, json } => {
                let rag = open_db_store().await;
                let paper_id = match papers_db::resolve_paper_id(&rag, &paper_id).await {
                    Ok(r) => r,
                    Err(e) => exit_err(&e.to_string()),
                };
                match papers_db::query::remove_work(&rag, &paper_id).await {
                    Ok(()) => {
                        if json {
                            print_json(&serde_json::json!({ "removed": true, "paper_id": paper_id }));
                        } else {
                            println!("Removed: {paper_id}");
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbWorkCommand::Outline { paper_id, json } => {
                let rag = open_db_store().await;
                let paper_id = match papers_db::resolve_paper_id(&rag, &paper_id).await {
                    Ok(r) => r,
                    Err(e) => exit_err(&e.to_string()),
                };
                match papers_db::query::get_paper_outline(&rag, &paper_id).await {
                    Ok(result) => { if json { print_json(&result); } else { format_db_outline(&result); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbWorkCommand::Extract { work, json } => {
                let zotero = zotero_client().await.unwrap_or_else(|e| match e {
                    papers_zotero::ZoteroError::NotRunning { path } => exit_err(&format!(
                        "Zotero is installed ({path}) but the local API is not enabled.\n\
                         Fix: Zotero \u{2192} Settings \u{2192} Advanced \u{2192} check \"Enable Local API\".\n\
                         Or set ZOTERO_CHECK_LAUNCHED=0 to skip this check."
                    )),
                    _ => exit_err("Zotero not configured. Set ZOTERO_USER_ID and ZOTERO_API_KEY."),
                });
                let key = smart_resolve_item_key(&zotero, &work).await
                    .unwrap_or_else(|e| exit_err(&e));
                if json {
                    match papers_core::text::datalab_cached_json(&key) {
                        Some(json_str) => print!("{json_str}"),
                        None => exit_err(&format!(
                            "No cached extraction for {key}. Run: papers db work add {key}"
                        )),
                    }
                } else {
                    match papers_core::text::datalab_cached_markdown(&key) {
                        Some(md) => print!("{md}"),
                        None => exit_err(&format!(
                            "No cached extraction for {key}. Run: papers db work add {key}"
                        )),
                    }
                }
            }
        },

        DbCommand::Section { cmd } => match cmd {
            DbSectionCommand::Search {
                query, selection, work, chapter_idx, year_min, year_max, venue, tag, limit, json,
            } => {
                let rag = open_db_store().await;
            let paper_ids = match selection.as_deref() {
                Some(sel) => match papers_core::selection::load_selection(sel) {
                    Ok(s) => Some(s.entries.iter().flat_map(|e| {
                        e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
                    }).collect()),
                    Err(e) => exit_err(&e.to_string()),
                },
                None => match work {
                    Some(id) => {
                        let resolved = match papers_db::resolve_paper_id(&rag, &id).await {
                            Ok(r) => r,
                            Err(e) => exit_err(&e.to_string()),
                        };
                        Some(vec![resolved])
                    }
                    None => None,
                },
            };
                let params = papers_db::SearchSectionsParams {
                    query, paper_ids, chapter_idx, filter_year_min: year_min, filter_year_max: year_max,
                    filter_venue: venue, filter_tags: tag, limit,
                };
                match papers_db::query::search_sections(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_section_search(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbSectionCommand::List { work, json } => {
                let rag = open_db_store().await;
                let paper_id = match work {
                    Some(ref id) => {
                        let resolved = match papers_db::resolve_paper_id(&rag, id).await {
                            Ok(r) => r,
                            Err(e) => exit_err(&e.to_string()),
                        };
                        Some(resolved)
                    }
                    None => None,
                };
                match papers_db::query::list_sections(&rag, papers_db::ListSectionsParams { paper_id }).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_section_list(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbSectionCommand::Get { paper_id, chapter_idx, section_idx, json } => {
                let rag = open_db_store().await;
                let paper_id = match papers_db::resolve_paper_id(&rag, &paper_id).await {
                    Ok(r) => r,
                    Err(e) => exit_err(&e.to_string()),
                };
                match papers_db::query::get_section(&rag, &paper_id, chapter_idx, section_idx).await {
                    Ok(result) => { if json { print_json(&result); } else { format_db_section(&result); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        DbCommand::Chapter { cmd } => match cmd {
            DbChapterCommand::Search {
                query, selection, work, year_min, year_max, venue, tag, limit, json,
            } => {
                let rag = open_db_store().await;
            let paper_ids = match selection.as_deref() {
                Some(sel) => match papers_core::selection::load_selection(sel) {
                    Ok(s) => Some(s.entries.iter().flat_map(|e| {
                        e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
                    }).collect()),
                    Err(e) => exit_err(&e.to_string()),
                },
                None => match work {
                    Some(id) => {
                        let resolved = match papers_db::resolve_paper_id(&rag, &id).await {
                            Ok(r) => r,
                            Err(e) => exit_err(&e.to_string()),
                        };
                        Some(vec![resolved])
                    }
                    None => None,
                },
            };
                let params = papers_db::SearchChaptersParams {
                    query, paper_ids, filter_year_min: year_min, filter_year_max: year_max,
                    filter_venue: venue, filter_tags: tag, limit,
                };
                match papers_db::query::search_chapters(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_chapter_search(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbChapterCommand::List { work, json } => {
                let rag = open_db_store().await;
                let paper_id = match work {
                    Some(ref id) => {
                        let resolved = match papers_db::resolve_paper_id(&rag, id).await {
                            Ok(r) => r,
                            Err(e) => exit_err(&e.to_string()),
                        };
                        Some(resolved)
                    }
                    None => None,
                };
                match papers_db::query::list_chapters(&rag, papers_db::ListChaptersParams { paper_id }).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_chapter_list(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }

            DbChapterCommand::Get { paper_id, chapter_idx, json } => {
                let rag = open_db_store().await;
                let paper_id = match papers_db::resolve_paper_id(&rag, &paper_id).await {
                    Ok(r) => r,
                    Err(e) => exit_err(&e.to_string()),
                };
                match papers_db::query::get_chapter(&rag, &paper_id, chapter_idx).await {
                    Ok(result) => { if json { print_json(&result); } else { format_db_chapter(&result); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        DbCommand::Tag { cmd } => match cmd {
            DbTagCommand::List { selection, json } => {
                let rag = open_db_store().await;
            let paper_ids = match selection.as_deref() {
                Some(sel) => match papers_core::selection::load_selection(sel) {
                    Ok(s) => Some(s.entries.iter().flat_map(|e| {
                        e.doi.iter().chain(e.openalex_id.iter()).chain(e.zotero_key.iter()).cloned()
                    }).collect()),
                    Err(e) => exit_err(&e.to_string()),
                },
                None => None,
            };
                let params = papers_db::ListTagsParams { paper_ids };
                match papers_db::query::list_tags(&rag, params).await {
                    Ok(results) => { if json { print_json(&results); } else { format_db_tags(&results); } }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        },

        DbCommand::Embed { cmd } => {
            handle_db_embed_command(cmd).await;
        }
    }
}


fn format_db_work_metadata(w: &papers_db::WorkMetadata) {
    println!("{}", w.paper_id);
    println!("  title: {}", w.title);
    if !w.authors.is_empty() { println!("  authors: {}", w.authors.join(", ")); }
    if let Some(y) = w.year { println!("  year: {y}"); }
    if let Some(v) = &w.venue { println!("  venue: {v}"); }
    if !w.tags.is_empty() { println!("  tags: {}", w.tags.join(", ")); }
    println!("  chunks: {}  exhibits: {}", w.chunk_count, w.exhibit_count);
}

fn format_db_work_search(results: &[papers_db::WorkSearchResult]) {
    if results.is_empty() { println!("No matching papers found."); return; }
    for r in results {
        let year = r.year.map(|y| y.to_string()).unwrap_or_else(|| "?".into());
        let venue = r.venue.as_deref().unwrap_or("");
        println!("[{:.3}] {} ({}, {})", r.score, r.title, year, venue);
        println!("       {} chunks  |  {}", r.chunk_count, r.paper_id);
        println!("       {}", r.top_chunk.chars().take(100).collect::<String>());
        println!();
    }
}

fn format_db_chunk_list(chunks: &[papers_db::ChunkListItem]) {
    if chunks.is_empty() { println!("No chunks found."); return; }
    for c in chunks {
        println!("[{}.{}.{}] {} / {} [{}]",
            c.chapter_idx, c.section_idx, c.chunk_idx,
            c.chapter_title, c.section_title, c.block_type);
        println!("    {}", c.text_preview.chars().take(100).collect::<String>());
    }
}

fn format_db_section_search(results: &[papers_db::SectionSearchResult]) {
    if results.is_empty() { println!("No matching sections found."); return; }
    for r in results {
        println!("[{:.3}] {}.{} {} \u{2014} {}",
            r.score, r.chapter_idx, r.section_idx, r.section_title, r.paper_title);
        println!("       {} chunks  |  {}", r.chunk_count, r.paper_id);
        println!("       {}", r.top_chunk.chars().take(100).collect::<String>());
        println!();
    }
}

fn format_db_section_list(sections: &[papers_db::SectionListItem]) {
    if sections.is_empty() { println!("No sections found."); return; }
    for s in sections {
        println!("  {}.{} {}  [{} chunks]",
            s.chapter_idx, s.section_idx, s.section_title, s.chunk_count);
    }
}

fn format_db_chapter_search(results: &[papers_db::ChapterSearchResult]) {
    if results.is_empty() { println!("No matching chapters found."); return; }
    for r in results {
        println!("[{:.3}] Ch.{} {} \u{2014} {}", r.score, r.chapter_idx, r.chapter_title, r.paper_title);
        println!("       {} sections, {} chunks  |  {}", r.section_count, r.chunk_count, r.paper_id);
        println!("       {}", r.top_chunk.chars().take(100).collect::<String>());
        println!();
    }
}

fn format_db_chapter_list(chapters: &[papers_db::ChapterListItem]) {
    if chapters.is_empty() { println!("No chapters found."); return; }
    for c in chapters {
        println!("  Ch.{} {}  [{} sections, {} chunks]",
            c.chapter_idx, c.chapter_title, c.section_count, c.chunk_count);
    }
}


fn handle_config_command(cmd: ConfigCommand) {
    match cmd {
        ConfigCommand::Set {
            cmd: ConfigSetCommand::Model { name },
        } => {
            if let Err(e) = papers_core::config::PapersConfig::validate_model(&name) {
                exit_err(&e.to_string());
            }
            let mut cfg = match papers_core::config::PapersConfig::load() {
                Ok(c) => c,
                Err(e) => exit_err(&format!("Failed to load config: {e}")),
            };
            cfg.embedding_model = name;
            match cfg.save() {
                Ok(()) => println!(
                    "Config saved: {}",
                    papers_core::config::PapersConfig::config_path().display()
                ),
                Err(e) => exit_err(&e.to_string()),
            }
        }
    }
}

async fn handle_mcp_command(cmd: McpCommand) {
    match cmd {
        McpCommand::Start { stdio: _ } => {
            if let Err(e) = papers_mcp::start_stdio().await {
                exit_err(&format!("MCP server error: {e}"));
            }
        }
    }
}

async fn handle_db_embed_command(cmd: DbEmbedCommand) {
    let cache = papers_db::default_embed_cache();
    let default_model = || {
        papers_core::config::PapersConfig::load()
            .map(|c| c.embedding_model)
            .unwrap_or_else(|_| "embedding-gemma-300m".to_string())
    };

    match cmd {
        DbEmbedCommand::List { work, json } => {
            if let Some(key) = work {
                match cache.list_models(&key) {
                    Ok(models) => {
                        if json {
                            print_json(&serde_json::json!({ "item_key": key, "models": models }));
                        } else if models.is_empty() {
                            println!("No cached embeddings for {key}.");
                        } else {
                            for model in &models {
                                let chunk_count = cache
                                    .load_manifest(model, &key)
                                    .ok()
                                    .flatten()
                                    .map(|m| m.chunks.len());
                                if let Some(n) = chunk_count {
                                    println!("{key}  {model}  ({n} chunks)");
                                } else {
                                    println!("{key}  {model}");
                                }
                            }
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            } else {
                match cache.list_all() {
                    Ok(pairs) => {
                        if json {
                            let out: Vec<_> = pairs
                                .iter()
                                .map(|(k, m)| serde_json::json!({ "item_key": k, "model": m }))
                                .collect();
                            print_json(&out);
                        } else if pairs.is_empty() {
                            println!("No cached embeddings.");
                        } else {
                            for (key, model) in &pairs {
                                let chunk_count = cache
                                    .load_manifest(model, key)
                                    .ok()
                                    .flatten()
                                    .map(|m| m.chunks.len());
                                if let Some(n) = chunk_count {
                                    println!("{key}  {model}  ({n} chunks)");
                                } else {
                                    println!("{key}  {model}");
                                }
                            }
                        }
                    }
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        }

        DbEmbedCommand::Add { work, model, force } => {
            let model_name = model.unwrap_or_else(default_model);
            if let Err(e) = papers_core::config::PapersConfig::validate_model(&model_name) {
                exit_err(&e.to_string());
            }

            let keys = if let Some(key) = work {
                vec![key]
            } else {
                papers_db::list_cached_item_keys()
            };

            if keys.is_empty() {
                println!("No cached papers found.");
                return;
            }

            let rag = open_db_store().await;
            for key in &keys {
                let params = match papers_db::ingest_params_from_cache(key) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("  [skip] {key}: {e}");
                        continue;
                    }
                };
                match papers_db::cache_paper_embeddings(&rag, &params, &model_name, force).await {
                    Ok(n) => println!("Cached {n} chunks for {key} [{model_name}]"),
                    Err(e) => eprintln!("  [fail] {key}: {e}"),
                }
            }
        }

        DbEmbedCommand::Delete { work, model } => {
            let model_name = model.unwrap_or_else(default_model);

            let keys = if let Some(key) = work {
                vec![key]
            } else {
                papers_db::list_cached_item_keys()
            };

            for key in &keys {
                match cache.delete(&model_name, key) {
                    Ok(()) => println!("Deleted embeddings for {key} [{model_name}]"),
                    Err(e) => exit_err(&e.to_string()),
                }
            }
        }
    }
}

// ── RAG human-readable formatters ────────────────────────────────────────────

fn format_db_search(results: &[papers_db::SearchResult]) {
    if results.is_empty() {
        println!("No results found.");
        return;
    }
    for r in results {
        let c = &r.chunk;
        println!(
            "{:.2}  {}  |  {} › {}",
            r.score, c.paper_id, c.chapter_title, c.section_title,
        );
        let preview: String = c.text.chars().take(200).collect();
        println!("      {}", preview);
        let prev = r
            .prev
            .as_ref()
            .map(|p| p.chunk_id.as_str())
            .unwrap_or("(none)");
        let next = r
            .next
            .as_ref()
            .map(|n| n.chunk_id.as_str())
            .unwrap_or("(none)");
        println!("      ← {}  /  {} →", prev, next);
        println!();
    }
}

fn format_db_exhibits(results: &[papers_db::ExhibitSearchResult]) {
    if results.is_empty() {
        println!("No exhibits found.");
        return;
    }
    for f in results {
        println!(
            "{:.2}  {}  {}  [{}]  (page {})",
            f.score,
            f.exhibit_id,
            f.paper_id,
            f.exhibit_type,
            f.page.map(|p| p.to_string()).unwrap_or_else(|| "?".into())
        );
        println!("  Caption: {}", f.caption);
        if let Some(img) = &f.image_path {
            println!("  Image: {img}");
        }
        println!();
    }
}

fn format_db_chunk_result(r: &papers_db::ChunkResult) {
    let c = &r.chunk;
    println!(
        "[{}]  {} — Ch.{} {} / Sec.{} {}",
        c.chunk_id, c.paper_id, c.chapter_idx, c.chapter_title, c.section_idx, c.section_title
    );
    println!();
    println!("{}", c.text);
    println!();
    let prev = r
        .prev
        .as_ref()
        .map(|p| p.chunk_id.as_str())
        .unwrap_or("(none)");
    let next = r
        .next
        .as_ref()
        .map(|n| n.chunk_id.as_str())
        .unwrap_or("(none)");
    println!("← {}  /  {} →", prev, next);
}

fn format_db_section(r: &papers_db::SectionResult) {
    println!(
        "[{}] Ch.{} {} / Sec.{} {} ({} chunks)",
        r.paper_id, 0, r.chapter_title, 0, r.section_title, r.total_chunks
    );
    println!();
    for chunk in &r.chunks {
        println!("{}", chunk.text);
        println!("---");
    }
}

fn format_db_chapter(r: &papers_db::ChapterResult) {
    println!(
        "[{}] Ch.{} {} ({} chunks)",
        r.paper_id, r.chapter_idx, r.chapter_title, r.total_chunks
    );
    println!();
    for sec in &r.sections {
        println!("  § {} {}", sec.section_idx, sec.section_title);
        for chunk in &sec.chunks {
            let preview: String = chunk.text.chars().take(120).collect();
            println!("    {}", preview);
        }
        println!();
    }
}

fn format_db_exhibit(f: &papers_db::ExhibitResult) {
    println!("{} [{}]", f.exhibit_id, f.exhibit_type);
    println!("  Paper: {}", f.paper_id);
    println!("  Caption: {}", f.caption);
    if let Some(img) = &f.image_path {
        println!("  Image: {img}");
    }
    if let Some(p) = f.page {
        println!("  Page: {p}");
    }
    if let Some(content) = &f.content {
        println!("  Content:");
        for line in content.lines() {
            println!("    {line}");
        }
    }
}

fn format_db_outline(r: &papers_db::PaperOutline) {
    println!(
        "{}  {:?}  ({}{})",
        r.paper_id,
        r.title,
        r.year.map(|y| y.to_string()).unwrap_or_else(|| "?".into()),
        r.venue
            .as_ref()
            .map(|v| format!(", {v}"))
            .unwrap_or_default()
    );
    for ch in &r.chapters {
        let ch_chunk_count: usize = ch.sections.iter().map(|s| s.chunk_count).sum();
        println!(
            "  {}. {}  [{} chunks]",
            ch.chapter_idx, ch.chapter_title, ch_chunk_count
        );
        for sec in &ch.sections {
            println!(
                "     {}.{} {}  [{} chunks]",
                ch.chapter_idx, sec.section_idx, sec.section_title, sec.chunk_count
            );
        }
    }
    println!(
        "Total: {} chunks, {} exhibits",
        r.total_chunks, r.total_exhibits
    );
}

fn format_db_papers(papers: &[papers_db::PaperSummary]) {
    if papers.is_empty() {
        println!("No indexed papers found.");
        return;
    }
    println!(
        "{:<6}  {:<12}  {:<6}  {}",
        "YEAR", "VENUE", "CHUNKS", "TITLE"
    );
    for p in papers {
        println!(
            "{:<6}  {:<12}  {:<6}  {}",
            p.year.map(|y| y.to_string()).unwrap_or_else(|| "?".into()),
            p.venue.as_deref().unwrap_or(""),
            p.chunk_count,
            p.title,
        );
    }
}

fn format_db_tags(tags: &[papers_db::TagSummary]) {
    if tags.is_empty() {
        println!("No tags found.");
        return;
    }
    for t in tags {
        println!("{:<30}  ({} papers)", t.tag, t.paper_count);
    }
}

/// Resolve a selection name from an Option<String>, using active selection as fallback.
fn resolve_sel_name(
    sel: Option<String>,
    active_sel: &dyn Fn() -> Option<String>,
) -> String {
    use papers_core::selection::resolve_selection;
    match sel {
        Some(s) => match resolve_selection(&s) {
            Ok(n) => n,
            Err(e) => exit_err(&e.to_string()),
        },
        None => match active_sel() {
            Some(n) => n,
            None => exit_err("no active selection; use --selection or activate one first"),
        },
    }
}

/// Check whether an entry has a PDF in the Zotero library.
/// Returns None if Zotero is unavailable; Some(bool) if it can be determined.
async fn entry_has_zotero_pdf(
    zotero: Option<&papers_zotero::ZoteroClient>,
    zotero_key: &str,
) -> Option<bool> {
    let zc = zotero?;
    let children = zc
        .list_item_children(zotero_key, &papers_zotero::ItemListParams::default())
        .await
        .ok()?;
    Some(children.items.iter().any(|child| {
        child.data.content_type.as_deref() == Some("application/pdf")
            && matches!(
                child.data.link_mode.as_deref(),
                Some("imported_file" | "imported_url")
            )
    }))
}

async fn handle_selection_command(cmd: SelectionCommand, client: &OpenAlexClient) {
    use papers_core::selection::{
        Selection, active_selection_name, delete_selection, entry_matches_remove_input,
        fill_from_zotero_item, list_selection_names, load_selection, load_state, resolve_paper,
        resolve_selection, save_selection, save_state, validate_name,
    };

    match cmd {
        SelectionCommand::List { json } => {
            let names = list_selection_names();
            let state = load_state();
            let active = state.active.as_deref();
            let items: Vec<format::SelectionListItem> = names
                .iter()
                .map(|name| {
                    let count = load_selection(name).map(|s| s.entries.len()).unwrap_or(0);
                    format::SelectionListItem {
                        name: name.clone(),
                        item_count: count,
                        is_active: Some(name.as_str()) == active,
                    }
                })
                .collect();
            if json {
                let v: Vec<_> = items
                    .iter()
                    .map(|i| {
                        serde_json::json!({
                            "name": i.name,
                            "item_count": i.item_count,
                            "is_active": i.is_active,
                        })
                    })
                    .collect();
                print_json(&v);
            } else {
                print!("{}", format::format_selection_list(&items));
            }
        }

        SelectionCommand::Set { name, json } => {
            let sel_name = match name {
                Some(n) => match resolve_selection(&n) {
                    Ok(n) => n,
                    Err(e) => exit_err(&e.to_string()),
                },
                None => match active_selection_name() {
                    Some(n) => n,
                    None => exit_err("no active selection; run: papers selection list"),
                },
            };
            let sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            // Activate it
            let mut state = load_state();
            state.active = Some(sel_name.clone());
            let _ = save_state(&state);

            let total = sel.entries.len();
            // Count PDFs (fast, local only)
            let has_pdf = sel.entries.iter().filter(|e| {
                e.zotero_key.as_deref().map(|k| papers_core::text::datalab_cached_markdown(k).is_some()).unwrap_or(false)
                    || e.doi.as_deref().map(papers_core::text::doi_pdf_cached).unwrap_or(false)
            }).count();
            // Count in DB (best-effort)
            let in_db = if let Ok(store) = papers_db::DbStore::open(&papers_db::DbStore::default_path()).await {
                let mut count = 0usize;
                for e in &sel.entries {
                    let ids: Vec<&str> = [
                        e.zotero_key.as_deref(),
                        e.doi.as_deref(),
                        e.openalex_id.as_deref(),
                    ].into_iter().flatten().collect();
                    let mut found = false;
                    for id in ids {
                        if papers_db::is_ingested(&store, id).await {
                            found = true;
                            break;
                        }
                    }
                    if found {
                        count += 1;
                    }
                }
                count
            } else {
                0
            };

            if json {
                print_json(&serde_json::json!({
                    "name": sel.name,
                    "total": total,
                    "in_db": in_db,
                    "has_pdf": has_pdf,
                }));
            } else {
                print!("{}", format::format_selection_set(&sel_name, total, in_db, has_pdf));
            }
        }

        SelectionCommand::Create { name, json } => {
            if let Err(e) = validate_name(&name) {
                exit_err(&e.to_string());
            }
            // Check if already exists
            if load_selection(&name).is_ok() {
                exit_err(&format!("selection {name:?} already exists"));
            }
            let sel = Selection {
                name: name.clone(),
                entries: Vec::new(),
            };
            if let Err(e) = save_selection(&sel) {
                exit_err(&e.to_string());
            }
            let mut state = load_state();
            state.active = Some(name.clone());
            if let Err(e) = save_state(&state) {
                exit_err(&e.to_string());
            }
            if json {
                print_json(&serde_json::json!({ "name": name, "is_active": true, "entries": [] }));
            } else {
                print!("{}", format::format_selection_create(&name));
            }
        }

        SelectionCommand::Delete { name, json } => {
            let sel_name = match resolve_selection(&name) {
                Ok(n) => n,
                Err(e) => exit_err(&e.to_string()),
            };
            let mut state = load_state();
            let was_active = state.active.as_deref() == Some(&sel_name);
            if let Err(e) = delete_selection(&sel_name) {
                exit_err(&e.to_string());
            }
            if was_active {
                state.active = None;
                let _ = save_state(&state);
            }
            if json {
                print_json(&serde_json::json!({ "name": sel_name, "was_active": was_active }));
            } else {
                print!("{}", format::format_selection_delete(&sel_name, was_active));
            }
        }

        SelectionCommand::Add {
            paper,
            selection,
            json,
        } => {
            let sel_name = resolve_sel_name(selection, &active_selection_name);
            // For selection add, Zotero is optional — treat all errors as "not available"
            let zotero = optional_zotero().await.unwrap_or(None);
            let entry = match resolve_paper(&paper, client, zotero.as_ref()).await {
                Ok(e) => e,
                Err(e) => exit_err(&e.to_string()),
            };
            let mut sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            // Deduplication check
            let is_dup = sel.entries.iter().any(|e| {
                entry
                    .zotero_key
                    .as_deref()
                    .map(|k| papers_core::selection::entry_matches_key(e, k))
                    .unwrap_or(false)
                    || entry
                        .openalex_id
                        .as_deref()
                        .map(|id| papers_core::selection::entry_matches_openalex(e, id))
                        .unwrap_or(false)
                    || entry
                        .doi
                        .as_deref()
                        .map(|d| papers_core::selection::entry_matches_doi(e, d))
                        .unwrap_or(false)
            });
            if !is_dup {
                sel.entries.push(entry.clone());
                if let Err(e) = save_selection(&sel) {
                    exit_err(&e.to_string());
                }
            }
            if json {
                print_json(&serde_json::json!({ "entry": entry, "added": !is_dup }));
            } else if is_dup {
                let title = entry.title.as_deref().unwrap_or("(unknown)");
                println!("Already in selection: {title:?}");
            } else {
                print!("{}", format::format_selection_add(&entry, &sel_name));
            }
        }

        SelectionCommand::Remove {
            paper,
            selection,
            json,
        } => {
            let sel_name = resolve_sel_name(selection, &active_selection_name);
            let mut sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            let before = sel.entries.len();

            // Try 1-based index first
            let removed_entry = if let Ok(idx) = paper.parse::<usize>() {
                if idx == 0 || idx > sel.entries.len() {
                    exit_err(&format!("index {idx} out of range (selection has {} entr{})",
                        sel.entries.len(),
                        if sel.entries.len() == 1 { "y" } else { "ies" }));
                }
                let entry = sel.entries.remove(idx - 1);
                Some(entry)
            } else {
                let entry = sel
                    .entries
                    .iter()
                    .find(|e| entry_matches_remove_input(e, &paper))
                    .cloned();
                sel.entries
                    .retain(|e| !entry_matches_remove_input(e, &paper));
                entry
            };

            if sel.entries.len() == before && removed_entry.is_none() {
                exit_err("item not found in selection");
            }
            if let Err(e) = save_selection(&sel) {
                exit_err(&e.to_string());
            }
            let title = removed_entry
                .as_ref()
                .and_then(|e| e.title.as_deref())
                .unwrap_or(&paper)
                .to_string();
            if json {
                print_json(&serde_json::json!({ "removed": title, "selection": sel_name }));
            } else {
                print!("{}", format::format_selection_remove(&title, &sel_name));
            }
        }

        SelectionCommand::Status { selection, json } => {
            let sel_name = resolve_sel_name(selection, &active_selection_name);
            let sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            let zotero = optional_zotero().await.unwrap_or(None);
            let rag = papers_db::DbStore::open(&papers_db::DbStore::default_path()).await.ok();

            let mut status_entries = Vec::new();
            for (i, entry) in sel.entries.iter().enumerate() {
                let has_zotero = entry.zotero_key.is_some();
                let pdf = if let Some(key) = &entry.zotero_key {
                    entry_has_zotero_pdf(zotero.as_ref(), key).await
                } else {
                    // Check local DOI PDF cache
                    entry.doi.as_deref().map(papers_core::text::doi_pdf_cached)
                };
                let extracted = entry.zotero_key.as_deref()
                    .map(|k| papers_core::text::datalab_cached_markdown(k).is_some())
                    .unwrap_or(false);
                let in_db = if let Some(store) = &rag {
                    let ids: Vec<&str> = [
                        entry.zotero_key.as_deref(),
                        entry.doi.as_deref(),
                        entry.openalex_id.as_deref(),
                    ].into_iter().flatten().collect();
                    let mut found = false;
                    for id in ids {
                        if papers_db::is_ingested(store, id).await {
                            found = true;
                            break;
                        }
                    }
                    found
                } else {
                    false
                };
                status_entries.push(format::SelectionStatusEntry {
                    index: i + 1,
                    title: entry.title.clone().unwrap_or_else(|| "(untitled)".to_string()),
                    year: entry.year,
                    authors: entry.authors.clone().unwrap_or_default(),
                    doi: entry.doi.clone(),
                    zotero: has_zotero,
                    pdf,
                    extracted,
                    in_db,
                });
            }

            if json {
                let v: Vec<_> = status_entries.iter().map(|e| serde_json::json!({
                    "index": e.index,
                    "title": e.title,
                    "year": e.year,
                    "zotero": e.zotero,
                    "pdf": e.pdf,
                    "extracted": e.extracted,
                    "in_db": e.in_db,
                })).collect();
                print_json(&v);
            } else {
                print!("{}", format::format_selection_status(&sel_name, &status_entries));
            }
        }

        SelectionCommand::Find { selection, open, json } => {
            let sel_name = resolve_sel_name(selection, &active_selection_name);
            let sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            let zotero = optional_zotero().await.unwrap_or(None);
            let http = reqwest::Client::new();

            let mut downloaded: Vec<(String, String)> = Vec::new();
            let mut not_found: Vec<(String, String)> = Vec::new();
            let mut skipped = 0usize;

            for entry in &sel.entries {
                let doi = match &entry.doi {
                    Some(d) => d.clone(),
                    None => { skipped += 1; continue; }
                };
                // Check if already has PDF in Zotero or local cache
                let already_has = if let Some(key) = &entry.zotero_key {
                    entry_has_zotero_pdf(zotero.as_ref(), key).await.unwrap_or(false)
                } else {
                    false
                } || papers_core::text::doi_pdf_cached(&doi);
                if already_has {
                    skipped += 1;
                    continue;
                }

                // Fetch work from OpenAlex
                let oa_id = format!("doi:{doi}");
                let work = match client.get_work(&oa_id, &papers_core::GetParams::default()).await {
                    Ok(w) => w,
                    Err(_) => {
                        not_found.push((doi.clone(), entry.title.clone().unwrap_or_else(|| doi.clone())));
                        continue;
                    }
                };

                match papers_core::text::try_download_open_access_pdf(&http, &work).await {
                    Ok(Some((bytes, _source))) => {
                        // Save to DOI cache
                        if let Some(cache_dir) = papers_core::text::doi_pdf_cache_dir(&doi) {
                            let _ = std::fs::create_dir_all(&cache_dir);
                            let safe_doi = doi.replace('/', "_");
                            let pdf_path = cache_dir.join(format!("{safe_doi}.pdf"));
                            if std::fs::write(&pdf_path, &bytes).is_ok() {
                                let title = entry.title.clone().unwrap_or_else(|| doi.clone());
                                downloaded.push((doi.clone(), title));
                            } else {
                                not_found.push((doi.clone(), entry.title.clone().unwrap_or_else(|| doi.clone())));
                            }
                        } else {
                            not_found.push((doi.clone(), entry.title.clone().unwrap_or_else(|| doi.clone())));
                        }
                    }
                    Ok(None) => {
                        not_found.push((doi.clone(), entry.title.clone().unwrap_or_else(|| doi.clone())));
                    }
                    Err(_) => {
                        not_found.push((doi.clone(), entry.title.clone().unwrap_or_else(|| doi.clone())));
                    }
                }
            }

            if json {
                print_json(&serde_json::json!({
                    "downloaded": downloaded.iter().map(|(d, t)| serde_json::json!({"doi": d, "title": t})).collect::<Vec<_>>(),
                    "not_found": not_found.iter().map(|(d, t)| serde_json::json!({"doi": d, "title": t})).collect::<Vec<_>>(),
                    "skipped": skipped,
                }));
            } else {
                print!("{}", format::format_selection_find(&sel_name, &downloaded, &not_found, skipped));
                if open && !not_found.is_empty() {
                    println!("\nOpen {} DOI link{} in browser? [y/N] ",
                        not_found.len(),
                        if not_found.len() == 1 { "" } else { "s" });
                    let mut input = String::new();
                    if std::io::stdin().read_line(&mut input).is_ok() && input.trim().eq_ignore_ascii_case("y") {
                        for (doi, _) in &not_found {
                            let url = format!("https://doi.org/{doi}");
                            let _ = open::that(&url);
                        }
                    }
                }
            }
        }

        SelectionCommand::Sync { selection, yes, json } => {
            let sel_name = resolve_sel_name(selection, &active_selection_name);
            let mut sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            let zotero = match zotero_client().await {
                Ok(z) => z,
                Err(e) => exit_err(&format!("Zotero unavailable: {e}")),
            };

            // Find or note existing collection
            let coll_params = papers_zotero::CollectionListParams::default();
            let collections = zotero.list_collections(&coll_params).await.unwrap_or_else(|e| {
                exit_err(&format!("Failed to list Zotero collections: {e}"))
            });
            let existing_coll = collections.items.iter()
                .find(|c| c.data.name.eq_ignore_ascii_case(&sel_name))
                .cloned();
            let collection_warning = if existing_coll.is_some() {
                Some(format!(
                    "Collection {:?} already exists — items will be added to it",
                    sel_name
                ))
            } else {
                None
            };

            // Build sync plan
            let mut actions: Vec<format::SyncAction> = Vec::new();
            // Track which entries need Zotero items created
            let mut needs_item: Vec<usize> = Vec::new(); // indices into sel.entries

            for (idx, entry) in sel.entries.iter().enumerate() {
                if entry.zotero_key.is_none() {
                    let doi_or_id = entry.doi.clone()
                        .or_else(|| entry.openalex_id.clone())
                        .unwrap_or_else(|| "(unknown)".to_string());
                    let title = entry.title.clone().unwrap_or_else(|| "(untitled)".to_string());
                    let item_type = entry.work_type.as_deref()
                        .map(papers_core::selection::openalex_type_to_zotero)
                        .unwrap_or("document")
                        .to_string();
                    actions.push(format::SyncAction::CreateItem { doi_or_id, title, item_type });
                    needs_item.push(idx);
                }
                if let Some(key) = &entry.zotero_key {
                    let doi = entry.doi.as_deref().unwrap_or("");
                    let has_cached_pdf = papers_core::text::doi_pdf_cached(doi);
                    if has_cached_pdf {
                        let has_pdf = entry_has_zotero_pdf(Some(&zotero), key).await.unwrap_or(false);
                        if !has_pdf {
                            actions.push(format::SyncAction::UploadPdf {
                                zotero_key: key.clone(),
                                title: entry.title.clone().unwrap_or_else(|| key.clone()),
                            });
                        }
                    }
                    if papers_core::text::datalab_cached_markdown(key).is_some() {
                        let has_extract = papers_core::text::find_papers_zip_key(&zotero, key)
                            .await
                            .unwrap_or(None)
                            .is_some();
                        if !has_extract {
                            actions.push(format::SyncAction::UploadExtract {
                                zotero_key: key.clone(),
                                title: entry.title.clone().unwrap_or_else(|| key.clone()),
                            });
                        }
                    }
                }
            }

            // Add-to-collection action for all entries with zotero_key
            let coll_keys: Vec<String> = sel.entries.iter()
                .filter_map(|e| e.zotero_key.clone())
                .collect();
            if !coll_keys.is_empty() || !needs_item.is_empty() {
                actions.push(format::SyncAction::AddToCollection {
                    keys: coll_keys.clone(),
                    coll_name: sel_name.clone(),
                });
            }

            if json {
                // In JSON mode, just output the plan
                let plan: Vec<serde_json::Value> = actions.iter().map(|a| match a {
                    format::SyncAction::CreateItem { doi_or_id, title, item_type } => serde_json::json!({"action": "create-item", "id": doi_or_id, "title": title, "item_type": item_type}),
                    format::SyncAction::UploadPdf { zotero_key, title } => serde_json::json!({"action": "upload-pdf", "key": zotero_key, "title": title}),
                    format::SyncAction::UploadExtract { zotero_key, title } => serde_json::json!({"action": "upload-extract", "key": zotero_key, "title": title}),
                    format::SyncAction::AddToCollection { keys, coll_name } => serde_json::json!({"action": "add-to-collection", "keys": keys, "collection": coll_name}),
                }).collect();
                print_json(&plan);
                return;
            }

            let plan_text = format::format_selection_sync_plan(
                &sel_name,
                collection_warning.as_deref(),
                &actions,
            );
            print!("{}", plan_text);

            if actions.is_empty() {
                return;
            }

            // Confirm
            if !yes {
                let mut input = String::new();
                let _ = std::io::stdin().read_line(&mut input);
                if !input.trim().eq_ignore_ascii_case("y") {
                    println!("Aborted.");
                    return;
                }
            }

            // Execute plan
            // 1. Create/find collection
            let coll_key = if let Some(c) = existing_coll {
                c.key.clone()
            } else {
                let resp = zotero.create_collections(vec![
                    serde_json::json!({"name": sel_name})
                ]).await.unwrap_or_else(|e| exit_err(&format!("Failed to create collection: {e}")));
                resp.successful.values().next()
                    .and_then(|v| v.get("key").and_then(|k| k.as_str()).map(String::from))
                    .unwrap_or_else(|| exit_err("Failed to get new collection key"))
            };

            // 2. Create items for entries without zotero_key
            for idx in needs_item {
                let entry = &sel.entries[idx];
                let item_type = entry.work_type.as_deref()
                    .map(papers_core::selection::openalex_type_to_zotero)
                    .unwrap_or("document");
                let mut item_data = serde_json::json!({
                    "itemType": item_type,
                    "title": entry.title.as_deref().unwrap_or(""),
                    "collections": [coll_key],
                });
                if let Some(doi) = &entry.doi {
                    item_data["DOI"] = serde_json::Value::String(doi.clone());
                }
                if let Some(year) = entry.year {
                    item_data["date"] = serde_json::Value::String(year.to_string());
                }
                if let Some(authors) = &entry.authors {
                    let creators: Vec<serde_json::Value> = authors.iter().map(|a| {
                        let parts: Vec<&str> = a.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            serde_json::json!({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]})
                        } else {
                            serde_json::json!({"creatorType": "author", "name": a})
                        }
                    }).collect();
                    item_data["creators"] = serde_json::Value::Array(creators);
                }
                match zotero.create_items(vec![item_data]).await {
                    Ok(resp) => {
                        if let Some(new_key) = resp.successful.values().next()
                            .and_then(|v| v.get("key").and_then(|k| k.as_str()).map(String::from))
                        {
                            sel.entries[idx].zotero_key = Some(new_key.clone());
                            println!("  Created Zotero item {new_key}");
                        }
                    }
                    Err(e) => eprintln!("  Failed to create item: {e}"),
                }
            }

            // 3. Upload PDFs and extractions
            for entry in &sel.entries {
                let key = match &entry.zotero_key {
                    Some(k) => k.clone(),
                    None => continue,
                };
                // Upload PDF if cached
                if let Some(doi) = &entry.doi {
                    if papers_core::text::doi_pdf_cached(doi) {
                        if let Some(cache_dir) = papers_core::text::doi_pdf_cache_dir(doi) {
                            let safe_doi = doi.replace('/', "_");
                            let pdf_path = cache_dir.join(format!("{safe_doi}.pdf"));
                            if pdf_path.exists() {
                                if let Ok(bytes) = std::fs::read(&pdf_path) {
                                    let filename = format!("{safe_doi}.pdf");
                                    match zotero.create_imported_attachment(&key, &filename, "application/pdf").await {
                                        Ok(att_key) => {
                                            match zotero.upload_attachment_file(&att_key, &filename, bytes).await {
                                                Ok(_) => println!("  Uploaded PDF for {key}"),
                                                Err(e) => eprintln!("  Failed to upload PDF for {key}: {e}"),
                                            }
                                        }
                                        Err(e) => eprintln!("  Failed to create PDF attachment for {key}: {e}"),
                                    }
                                }
                            }
                        }
                    }
                }
                // Upload extraction
                if papers_core::text::datalab_cached_markdown(&key).is_some() {
                    if let Err(e) = papers_core::text::upload_extraction_to_zotero(&zotero, &key).await {
                        eprintln!("  Failed to upload extraction for {key}: {e}");
                    } else {
                        println!("  Uploaded extraction for {key}");
                    }
                }
            }

            // 4. Add all items to collection
            for entry in &sel.entries {
                if let Some(key) = &entry.zotero_key {
                    // Get current version
                    if let Ok(item) = zotero.get_item(key).await {
                        let mut colls: Vec<String> = item.data.collections.clone();
                        if !colls.contains(&coll_key) {
                            colls.push(coll_key.clone());
                            let patch = serde_json::json!({"collections": colls});
                            if let Err(e) = zotero.patch_item(key, item.version, patch).await {
                                eprintln!("  Failed to add {key} to collection: {e}");
                            }
                        }
                    }
                }
            }

            // Save updated selection (new zotero_keys)
            let _ = save_selection(&sel);
            println!("\nSync complete.");
        }

        SelectionCommand::Db { cmd } => match cmd {
            SelectionDbCommand::Add { allow_skip, selection, json } => {
                let sel_name = resolve_sel_name(selection, &active_selection_name);
                let sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
                let store = open_db_store().await;
                let mut added = 0usize;
                let mut skipped = 0usize;
                let mut errors = 0usize;

                for entry in &sel.entries {
                    let key = match &entry.zotero_key {
                        Some(k) => k.clone(),
                        None => {
                            if allow_skip {
                                skipped += 1;
                                continue;
                            }
                            eprintln!("  Error: entry {:?} has no Zotero key; run `selection find` + `selection sync` first",
                                entry.title.as_deref().unwrap_or("(untitled)"));
                            errors += 1;
                            continue;
                        }
                    };

                    match papers_db::ingest_params_from_cache(&key) {
                        Ok(params) => {
                            match papers_db::ingest_paper(&store, params).await {
                                Ok(_) => { added += 1; }
                                Err(e) => {
                                    eprintln!("  Error ingesting {key}: {e}");
                                    errors += 1;
                                }
                            }
                        }
                        Err(_) => {
                            if allow_skip {
                                skipped += 1;
                                if !json {
                                    eprintln!("  Skipping {key}: no extraction cache (run `selection find` + `selection sync` first)");
                                }
                            } else {
                                eprintln!("  Error: no extraction cache for {key}; run `selection find` + `selection sync` first");
                                errors += 1;
                            }
                        }
                    }
                }

                if json {
                    print_json(&serde_json::json!({"added": added, "skipped": skipped, "errors": errors}));
                } else {
                    print!("{}", format::format_selection_db_add(&sel_name, added, skipped, errors));
                }
            }

            SelectionDbCommand::Remove { selection, json } => {
                let sel_name = resolve_sel_name(selection, &active_selection_name);
                let sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
                let store = open_db_store().await;
                let mut removed = 0usize;
                let mut not_found = 0usize;

                for entry in &sel.entries {
                    let ids: Vec<String> = [
                        entry.zotero_key.as_deref(),
                        entry.doi.as_deref(),
                        entry.openalex_id.as_deref(),
                    ].into_iter().flatten().map(String::from).collect();

                    let mut found_id: Option<String> = None;
                    for id in &ids {
                        match papers_db::resolve_paper_id(&store, id).await {
                            Ok(paper_id) => { found_id = Some(paper_id); break; }
                            Err(_) => continue,
                        }
                    }

                    if let Some(paper_id) = found_id {
                        match papers_db::query::remove_work(&store, &paper_id).await {
                            Ok(_) => { removed += 1; }
                            Err(e) => eprintln!("  Error removing {paper_id}: {e}"),
                        }
                    } else {
                        not_found += 1;
                    }
                }

                if json {
                    print_json(&serde_json::json!({"removed": removed, "not_found": not_found}));
                } else {
                    print!("{}", format::format_selection_db_remove(&sel_name, removed, not_found));
                }
            }
        }

        SelectionCommand::Collection { cmd } => match cmd {
            SelectionCollectionCommand::Add { collection, selection, json } => {
                let sel_name = resolve_sel_name(selection, &active_selection_name);
                let mut sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
                let zotero = match zotero_client().await {
                    Ok(z) => z,
                    Err(e) => exit_err(&format!("Zotero unavailable: {e}")),
                };

                // Resolve collection: try as 8-char key first, then search by name
                let coll_key = if collection.len() == 8 && collection.chars().all(|c| c.is_ascii_alphanumeric()) {
                    collection.to_uppercase()
                } else {
                    // Fetch all collections and find by name
                    let mut found_key: Option<String> = None;
                    let mut start = 0u32;
                    loop {
                        let params = papers_zotero::CollectionListParams {
                            limit: Some(100),
                            start: Some(start),
                            ..Default::default()
                        };
                        let resp = zotero.list_collections(&params).await
                            .unwrap_or_else(|e| exit_err(&format!("Failed to list collections: {e}")));
                        for c in &resp.items {
                            if c.data.name.eq_ignore_ascii_case(&collection) {
                                found_key = Some(c.key.clone());
                                break;
                            }
                        }
                        if found_key.is_some() || resp.items.len() < 100 {
                            break;
                        }
                        start += 100;
                    }
                    found_key.unwrap_or_else(|| exit_err(&format!("Collection {collection:?} not found in Zotero")))
                };

                // Resolve actual collection name from Zotero
                let coll_name = if let Ok(c) = zotero.get_collection(&coll_key).await {
                    c.data.name.clone()
                } else {
                    collection.clone()
                };

                // Paginate all items from the collection
                let mut new_entries = Vec::new();
                let mut start = 0u32;
                loop {
                    let params = papers_zotero::ItemListParams {
                        limit: Some(100),
                        start: Some(start),
                        ..Default::default()
                    };
                    let resp = zotero.list_collection_top_items(&coll_key, &params).await
                        .unwrap_or_else(|e| exit_err(&format!("Failed to list collection items: {e}")));
                    if resp.items.is_empty() {
                        break;
                    }
                    for item in &resp.items {
                        let mut entry = papers_core::selection::SelectionEntry {
                            zotero_key: Some(item.key.clone()),
                            openalex_id: None,
                            doi: None,
                            title: None,
                            authors: None,
                            year: None,
                            issn: None,
                            isbn: None,
                            work_type: None,
                        };
                        fill_from_zotero_item(&mut entry, item);
                        new_entries.push(entry);
                    }
                    if resp.items.len() < 100 {
                        break;
                    }
                    start += 100;
                }

                // Deduplicate
                let mut added = 0usize;
                let mut dupes = 0usize;
                for entry in new_entries {
                    let is_dup = sel.entries.iter().any(|e| {
                        entry.zotero_key.as_deref()
                            .map(|k| papers_core::selection::entry_matches_key(e, k))
                            .unwrap_or(false)
                            || entry.doi.as_deref()
                                .map(|d| papers_core::selection::entry_matches_doi(e, d))
                                .unwrap_or(false)
                    });
                    if !is_dup {
                        sel.entries.push(entry);
                        added += 1;
                    } else {
                        dupes += 1;
                    }
                }
                if let Err(e) = save_selection(&sel) {
                    exit_err(&e.to_string());
                }

                if json {
                    print_json(&serde_json::json!({"added": added, "duplicates": dupes}));
                } else {
                    print!("{}", format::format_selection_collection_add(&sel_name, &coll_name, added, dupes));
                }
            }
        }

        SelectionCommand::Merge { source, selection, json } => {
            let target_name = resolve_sel_name(selection, &active_selection_name);
            let source_name = match resolve_selection(&source) {
                Ok(n) => n,
                Err(e) => exit_err(&e.to_string()),
            };
            if source_name == target_name {
                exit_err("source and target selection are the same");
            }
            let source_sel = load_selection(&source_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            let mut target_sel = load_selection(&target_name).unwrap_or_else(|e| exit_err(&e.to_string()));

            let mut added = 0usize;
            let mut dupes = 0usize;
            for entry in &source_sel.entries {
                let is_dup = target_sel.entries.iter().any(|e| {
                    entry.zotero_key.as_deref()
                        .map(|k| papers_core::selection::entry_matches_key(e, k))
                        .unwrap_or(false)
                        || entry.openalex_id.as_deref()
                            .map(|id| papers_core::selection::entry_matches_openalex(e, id))
                            .unwrap_or(false)
                        || entry.doi.as_deref()
                            .map(|d| papers_core::selection::entry_matches_doi(e, d))
                            .unwrap_or(false)
                });
                if !is_dup {
                    target_sel.entries.push(entry.clone());
                    added += 1;
                } else {
                    dupes += 1;
                }
            }
            if let Err(e) = save_selection(&target_sel) {
                exit_err(&e.to_string());
            }

            if json {
                print_json(&serde_json::json!({"added": added, "duplicates": dupes, "source": source_name, "target": target_name}));
            } else {
                print!("{}", format::format_selection_merge(&target_name, &source_name, added, dupes));
            }
        }

        SelectionCommand::Rename { new_name, selection, json } => {
            let old_name = resolve_sel_name(selection, &active_selection_name);
            if let Err(e) = validate_name(&new_name) {
                exit_err(&e.to_string());
            }
            if load_selection(&new_name).is_ok() {
                exit_err(&format!("selection {new_name:?} already exists"));
            }
            let mut sel = load_selection(&old_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            sel.name = new_name.clone();
            if let Err(e) = save_selection(&sel) {
                exit_err(&e.to_string());
            }
            if let Err(e) = delete_selection(&old_name) {
                exit_err(&e.to_string());
            }
            // Update state if this was the active selection
            let mut state = load_state();
            if state.active.as_deref() == Some(&old_name) {
                state.active = Some(new_name.clone());
                let _ = save_state(&state);
            }

            if json {
                print_json(&serde_json::json!({"old_name": old_name, "new_name": new_name}));
            } else {
                print!("{}", format::format_selection_rename(&old_name, &new_name));
            }
        }
    }
}
