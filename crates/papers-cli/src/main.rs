mod cli;
mod format;

use clap::Parser;
use cli::{
    AdvancedMode, AuthorCommand, AuthorFilterArgs, Cli, DomainCommand, DomainFilterArgs,
    EntityCommand, FieldCommand, FieldFilterArgs, FunderCommand, FunderFilterArgs,
    InstitutionCommand, InstitutionFilterArgs, PublisherCommand, PublisherFilterArgs,
    SelectionCommand, SourceCommand, SourceFilterArgs, SubfieldCommand, SubfieldFilterArgs,
    TopicCommand, TopicFilterArgs, WorkCommand, WorkFilterArgs, ZoteroAnnotationCommand,
    ZoteroAttachmentCommand, ZoteroCollectionCommand, ZoteroCommand, ZoteroDeletedCommand,
    ZoteroExtractCommand, ZoteroGroupCommand, ZoteroNoteCommand, ZoteroPermissionCommand,
    ZoteroSearchCommand, ZoteroSettingCommand, ZoteroTagCommand, ZoteroWorkCommand,
};
use papers_core::{
    filter::FilterError,
    AuthorListParams, DiskCache, DomainListParams, FieldListParams, FindWorksParams,
    FunderListParams, GetParams, InstitutionListParams, OpenAlexClient, PublisherListParams,
    SourceListParams, SubfieldListParams, TopicListParams, WorkListParams,
};
use papers_core::zotero::{resolve_collection_key, resolve_item_key, resolve_search_key};
use papers_zotero::{CollectionListParams, DeletedParams, Item, ItemListParams, TagListParams, ZoteroClient};
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
        search: args.search.clone(),
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
        search: args.search.clone(),
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
        search: args.search.clone(),
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

fn institution_list_params(args: &cli::ListArgs, inf: &InstitutionFilterArgs) -> InstitutionListParams {
    InstitutionListParams {
        search: args.search.clone(),
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
        search: args.search.clone(),
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
        search: args.search.clone(),
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
        search: args.search.clone(),
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
        search: args.search.clone(),
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
        search: args.search.clone(),
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
        search: args.search.clone(),
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
    println!("{}", serde_json::to_string_pretty(val).expect("JSON serialization failed"));
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
    let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
    let children = zotero.list_item_children(item_key, &att_params).await
        .map_err(|e| e.to_string())?;
    children.items.into_iter()
        .find(|a| a.data.content_type.as_deref() == Some("application/pdf"))
        .ok_or_else(|| format!("No PDF attachment found for item {item_key}"))
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
    let resp = zotero.list_top_items(&params).await.map_err(|e| e.to_string())?;
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
            WorkCommand::Get { id, json } => {
                let zotero = optional_zotero().await.unwrap_or_else(|e| exit_err(&e.to_string()));
                let zotero_configured = zotero.is_some();
                match papers_core::api::work_get_response(&client, zotero.as_ref(), &id, &GetParams::default()).await {
                    Ok(response) => {
                        if json {
                            print_json(&response);
                        } else {
                            print!("{}", format::format_work_get_response(&response, zotero_configured));
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
            WorkCommand::Text { id, json, no_prompt, advanced } => {
                let zotero = optional_zotero().await.unwrap_or_else(|e| exit_err(&e.to_string()));
                let dl_client = if advanced.is_some() {
                    papers_datalab::DatalabClient::from_env().ok()
                } else {
                    None
                };
                let datalab = dl_client.as_ref().map(|dl| {
                    let mode = match advanced.as_ref().unwrap() {
                        AdvancedMode::Fast     => papers_core::text::ProcessingMode::Fast,
                        AdvancedMode::Balanced => papers_core::text::ProcessingMode::Balanced,
                        AdvancedMode::Accurate => papers_core::text::ProcessingMode::Accurate,
                    };
                    (dl, mode)
                });
                match papers_core::text::work_text(&client, zotero.as_ref(), datalab, &id).await {
                    Ok(result) => {
                        if json {
                            print_json(&result);
                        } else {
                            print!("{}", format::format_work_text(&result));
                        }
                    }
                    Err(papers_core::text::WorkTextError::NoPdfFound { ref work_id, ref title, ref doi }) => {
                        let display_title = title.as_deref().unwrap_or(work_id);

                        if no_prompt || doi.is_none() || zotero.is_none() {
                            exit_err(&format!("No PDF found for {display_title}"));
                        }

                        let doi = doi.as_ref().unwrap();
                        let bare = doi.strip_prefix("https://doi.org/").unwrap_or(doi);
                        eprintln!("No PDF found for \"{display_title}\".");
                        eprintln!("Open https://doi.org/{bare} to save this paper to Zotero? [Y/n] ");

                        let mut input = String::new();
                        std::io::stdin().read_line(&mut input).unwrap_or(0);
                        let input = input.trim().to_lowercase();

                        if !input.is_empty() && input != "y" && input != "yes" {
                            exit_err(&format!("No PDF found for {display_title}"));
                        }

                        // Open DOI URL in browser
                        let url = format!("https://doi.org/{bare}");
                        #[cfg(target_os = "windows")]
                        let _ = std::process::Command::new("cmd").args(["/c", "start", &url]).spawn();
                        #[cfg(target_os = "macos")]
                        let _ = std::process::Command::new("open").arg(&url).spawn();
                        #[cfg(target_os = "linux")]
                        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();

                        eprintln!("Waiting for paper to appear in Zotero...");
                        match papers_core::text::poll_zotero_for_work(
                            zotero.as_ref().unwrap(),
                            work_id,
                            title.as_deref(),
                            bare,
                        ).await {
                            Ok(result) => {
                                if json {
                                    print_json(&result);
                                } else {
                                    print!("{}", format::format_work_text(&result));
                                }
                            }
                            Err(e) => exit_err(&e.to_string()),
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
                        search, everything, tag, type_, sort, direction, limit, start, since, json,
                    } => {
                        let params = ItemListParams {
                            item_type: type_,
                            q: search,
                            qmode: everything.then(|| "everything".to_string()),
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
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_work_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json { print_json(&item); } else { print!("{}", format::format_zotero_item_get(&item)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Collections { key, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let item = zotero.get_item(&key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let col_keys = item.data.collections.clone();
                        let mut collections = Vec::new();
                        for ck in &col_keys {
                            match zotero.get_collection(ck).await {
                                Ok(c) => collections.push(c),
                                Err(e) => exit_err(&e.to_string()),
                            }
                        }
                        if json { print_json(&collections); } else { print!("{}", format::format_zotero_collection_list_vec(&collections)); }
                    }
                    ZoteroWorkCommand::Notes { key, limit, start, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams { item_type: Some("note".into()), limit, start, ..Default::default() };
                        match zotero.list_item_children(&key, &params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_note_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Attachments { key, limit, start, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams { item_type: Some("attachment".into()), limit, start, ..Default::default() };
                        match zotero.list_item_children(&key, &params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_attachment_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Annotations { key, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
                        let attachments = zotero.list_item_children(&key, &att_params).await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let ann_params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
                        let mut all_annotations = Vec::new();
                        for att in &attachments.items {
                            if !is_annotatable_attachment(att) { continue; }
                            match zotero.list_item_children(&att.key, &ann_params).await {
                                Ok(r) => all_annotations.extend(r.items),
                                Err(_) => {},
                            }
                        }
                        if json { print_json(&all_annotations); } else { print!("{}", format::format_zotero_annotation_list_vec(&all_annotations)); }
                    }
                    ZoteroWorkCommand::Tags { key, search, limit, start, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = TagListParams { q: search, qmode: Some("contains".to_string()), limit, start, ..Default::default() };
                        match zotero.list_item_tags(&key, &params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_tag_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::Extract { key: input, mode } => {
                        let is_exact_key = papers_core::zotero::looks_like_zotero_key(&input);

                        // Resolve to a concrete item key.
                        let key = resolve_item_key(&zotero, &input).await.unwrap_or_else(|e| exit_err(&e.to_string()));

                        // Cache hit: return immediately regardless of how the key was specified.
                        if let Some(markdown) = papers_core::text::datalab_cached_markdown(&key) {
                            print!("{markdown}");
                        } else {
                            // Cache miss — if the user gave a search string (not an exact key),
                            // error out with suggestions to avoid spending DataLab credits
                            // on the wrong paper.
                            if !is_exact_key {
                                let matched = zotero
                                    .list_top_items(&ItemListParams::builder().q(&input).limit(5).build())
                                    .await
                                    .unwrap_or_else(|e| exit_err(&e.to_string()));
                                let mut msg = format!(
                                    "Ambiguous search {:?} — use an exact item key. Did you mean:\n",
                                    input
                                );
                                for item in &matched.items {
                                    let title = item.data.title.as_deref().unwrap_or("(no title)");
                                    msg.push_str(&format!("  papers zotero work extract {}: {}\n", item.key, title));
                                }
                                exit_err(&msg);
                            }

                            // Cache miss: read the PDF from local Zotero storage (no HTTP download).
                            let att = find_pdf_attachment(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e));
                            let filename = att.data.filename.unwrap_or_else(|| exit_err("attachment has no filename"));
                            let local_path = dirs::home_dir()
                                .unwrap_or_else(|| exit_err("cannot determine home dir"))
                                .join("Zotero").join("storage").join(&att.key).join(&filename);
                            let pdf_bytes = std::fs::read(&local_path)
                                .unwrap_or_else(|e| exit_err(&format!("failed to read {}: {e}", local_path.display())));

                            let dl = papers_datalab::DatalabClient::from_env().unwrap_or_else(|e| exit_err(&e.to_string()));
                            let processing_mode = match mode {
                                AdvancedMode::Fast     => papers_core::text::ProcessingMode::Fast,
                                AdvancedMode::Balanced => papers_core::text::ProcessingMode::Balanced,
                                AdvancedMode::Accurate => papers_core::text::ProcessingMode::Accurate,
                            };
                            let mut source = papers_core::text::PdfSource::ZoteroLocal { path: local_path.to_string_lossy().into_owned() };
                            match papers_core::text::do_extract(pdf_bytes, &key, Some(&zotero), Some((&dl, processing_mode)), &mut source).await {
                                Ok(markdown) => print!("{markdown}"),
                                Err(e) => exit_err(&e.to_string()),
                            }
                        }
                    }
                    ZoteroWorkCommand::Text { key, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_key = find_pdf_attachment_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e));
                        match zotero.get_item_fulltext(&att_key).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_work_fulltext(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::ViewUrl { key } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_key = find_pdf_attachment_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e));
                        match zotero.get_item_file_view_url(&att_key).await {
                            Ok(url) => println!("{url}"),
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroWorkCommand::View { key, output } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_key = find_pdf_attachment_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e));
                        match zotero.get_item_file_view(&att_key).await {
                            Ok(bytes) => {
                                if output == "-" {
                                    use std::io::Write;
                                    std::io::stdout().write_all(&bytes)
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

                ZoteroCommand::Extract { cmd } => match cmd {
                    ZoteroExtractCommand::List { search, limit, json } => {
                        use std::collections::{HashMap, HashSet};

                        // 1. Local cache keys (filesystem scan)
                        let local_keys: HashSet<String> =
                            papers_core::text::datalab_cached_item_keys().into_iter().collect();

                        // 2. Zotero-backed keys: find all `papers_extract_*.zip` attachments
                        //    with a single targeted query — the embedded key in the filename
                        //    lets us skip parent_item lookups entirely.
                        let mut backed_up_keys: HashSet<String> = HashSet::new();
                        let att_params = ItemListParams {
                            item_type: Some("attachment".into()),
                            q: Some("papers_extract".into()),
                            limit: Some(100),
                            ..Default::default()
                        };
                        if let Ok(att_resp) = zotero.list_items(&att_params).await {
                            for item in &att_resp.items {
                                if let Some(filename) = &item.data.filename {
                                    if let Some(key) = filename
                                        .strip_prefix("papers_extract_")
                                        .and_then(|s| s.strip_suffix(".zip"))
                                    {
                                        backed_up_keys.insert(key.to_string());
                                    }
                                }
                            }
                        }

                        // 3. Union of both sources, sorted for deterministic output
                        let mut all_keys: Vec<String> =
                            local_keys.union(&backed_up_keys).cloned().collect();
                        all_keys.sort();

                        // 4. Apply optional search: intersect union with Zotero search results
                        let filtered_keys: Vec<String> = if let Some(ref q) = search {
                            let search_params = ItemListParams {
                                q: Some(q.clone()),
                                limit: Some(limit),
                                ..Default::default()
                            };
                            let search_resp = zotero.list_top_items(&search_params).await
                                .unwrap_or_else(|e| exit_err(&e.to_string()));
                            let search_set: HashSet<String> =
                                search_resp.items.into_iter().map(|i| i.key).collect();
                            all_keys.into_iter().filter(|k| search_set.contains(k)).collect()
                        } else {
                            all_keys.into_iter().take(limit as usize).collect()
                        };

                        // 5. Batch-fetch titles — also tells us which items still exist in Zotero.
                        //    Use list_items (not list_top_items) so items aren't filtered out by
                        //    the /items/top "top-level only" constraint when itemKey is specified.
                        //    Set limit == chunk size so we never silently truncate to the default 25.
                        let mut title_map: HashMap<String, String> = HashMap::new();
                        for chunk in filtered_keys.chunks(50) {
                            let keys_str = chunk.join(",");
                            let batch_params = ItemListParams {
                                item_key: Some(keys_str),
                                limit: Some(chunk.len() as u32),
                                ..Default::default()
                            };
                            if let Ok(resp) = zotero.list_items(&batch_params).await {
                                for item in resp.items {
                                    // Always insert — even empty title — so title_map.contains_key()
                                    // reliably indicates "item exists in Zotero".
                                    title_map.insert(item.key, item.data.title.unwrap_or_default());
                                }
                            }
                        }

                        // 6. Output: [local] [remote]
                        //    local  = extraction exists in the local cache
                        //    remote = papers_extract_*.zip backup exists in Zotero
                        //             annotated with "*no item*" when the parent item
                        //             is no longer in the Zotero library
                        // Helper: resolve title for a key using meta.json first,
                        // then the Zotero batch fetch, then a fallback based on presence.
                        let resolve_title = |key: &str| -> String {
                            // 1. meta.json title (fastest, no network needed)
                            if let Some(meta) = papers_core::text::read_extraction_meta(key) {
                                if let Some(t) = meta.title {
                                    if !t.is_empty() {
                                        return t;
                                    }
                                }
                            }
                            // 2. Zotero batch-fetched title
                            if let Some(t) = title_map.get(key) {
                                return if t.is_empty() {
                                    "(no title)".to_string()
                                } else {
                                    t.clone()
                                };
                            }
                            // 3. Fallback depends on whether item or backup exists
                            if backed_up_keys.contains(key) {
                                "(title unknown)".to_string()
                            } else {
                                "(not in Zotero)".to_string()
                            }
                        };

                        if json {
                            let items: Vec<_> = filtered_keys.iter().map(|k| {
                                let remote_status = if backed_up_keys.contains(k) {
                                    "ok"
                                } else if title_map.contains_key(k) {
                                    "no_backup"
                                } else {
                                    "no_item"
                                };
                                serde_json::json!({
                                    "key": k,
                                    "title": resolve_title(k),
                                    "local":  local_keys.contains(k),
                                    "remote": backed_up_keys.contains(k),
                                    "remote_status": remote_status,
                                })
                            }).collect();
                            print_json(&items);
                        } else {
                            for key in &filtered_keys {
                                let lmark = if local_keys.contains(key) { "✓" } else { "✗" };
                                let remote_col = if backed_up_keys.contains(key) {
                                    "[✓ remote]"
                                } else if title_map.contains_key(key) {
                                    "[✗ remote]"
                                } else {
                                    "[✗ remote *no item*]"
                                };
                                println!("{key}  [{lmark} local]  {remote_col}  {}", resolve_title(key));
                            }
                        }
                    }
                    ZoteroExtractCommand::Upload { dry_run } => {
                        use std::collections::HashSet;
                        let local_keys: HashSet<String> =
                            papers_core::text::datalab_cached_item_keys().into_iter().collect();

                        // Keys already backed up in Zotero
                        let backed_up_keys: HashSet<String> = {
                            let p = ItemListParams {
                                item_type: Some("attachment".into()),
                                q: Some("papers_extract".into()),
                                limit: Some(100),
                                ..Default::default()
                            };
                            match zotero.list_items(&p).await {
                                Ok(r) => r.items.iter().filter_map(|i| {
                                    let f = i.data.filename.as_deref()?;
                                    let k = f.strip_prefix("papers_extract_")?.strip_suffix(".zip")?;
                                    Some(k.to_string())
                                }).collect(),
                                Err(e) => exit_err(&e.to_string()),
                            }
                        };

                        let mut to_upload: Vec<String> = local_keys
                            .difference(&backed_up_keys)
                            .cloned()
                            .collect();
                        to_upload.sort();

                        if to_upload.is_empty() {
                            println!("All local extractions already backed up to Zotero.");
                        } else {
                            // Batch-check which keys still exist as items in Zotero.
                            // Zotero item keys are scoped to a library — if the item was
                            // deleted the key cannot be reused, so we simply skip those.
                            let mut item_exists: HashSet<String> = HashSet::new();
                            for chunk in to_upload.chunks(50) {
                                let keys_str = chunk.join(",");
                                let p = ItemListParams { item_key: Some(keys_str), ..Default::default() };
                                if let Ok(resp) = zotero.list_top_items(&p).await {
                                    for item in resp.items { item_exists.insert(item.key); }
                                }
                            }

                            let mut uploaded = 0usize;
                            for key in &to_upload {
                                if !item_exists.contains(key) {
                                    if dry_run {
                                        println!("skipping: {key}  (item not in Zotero)");
                                    } else {
                                        eprintln!("skipping: {key}  (item not in Zotero)");
                                    }
                                    continue;
                                }
                                if dry_run {
                                    println!("would upload: {key}");
                                } else {
                                    eprint!("Uploading backup {key}... ");
                                    match papers_core::text::upload_extraction_to_zotero(&zotero, key).await {
                                        Ok(()) => { eprintln!("ok"); uploaded += 1; }
                                        Err(e) => eprintln!("error: {e}"),
                                    }
                                }
                            }
                            if !dry_run {
                                eprintln!("Done. Uploaded {uploaded} extraction(s).");
                            }
                        }
                    }

                    ZoteroExtractCommand::Download { dry_run } => {
                        use std::collections::HashSet;
                        let local_keys: HashSet<String> =
                            papers_core::text::datalab_cached_item_keys().into_iter().collect();

                        // Collect (att_key, item_key) for all Zotero-backed extractions
                        let p = ItemListParams {
                            item_type: Some("attachment".into()),
                            q: Some("papers_extract".into()),
                            limit: Some(100),
                            ..Default::default()
                        };
                        let att_resp = zotero.list_items(&p).await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));

                        let mut to_download: Vec<(String, String)> = att_resp.items.iter()
                            .filter_map(|i| {
                                let f = i.data.filename.as_deref()?;
                                let item_key = f.strip_prefix("papers_extract_")?.strip_suffix(".zip")?;
                                if local_keys.contains(item_key) { return None; }
                                Some((i.key.clone(), item_key.to_string()))
                            })
                            .collect();
                        to_download.sort_by(|a, b| a.1.cmp(&b.1));

                        if to_download.is_empty() {
                            println!("All Zotero extractions already present locally.");
                        } else {
                            for (att_key, item_key) in &to_download {
                                if dry_run {
                                    println!("would download: {item_key}");
                                } else {
                                    eprint!("Downloading {item_key}... ");
                                    match papers_core::text::download_extraction_from_zotero(&zotero, att_key, item_key).await {
                                        Ok(()) => eprintln!("ok"),
                                        Err(e) => eprintln!("error: {e}"),
                                    }
                                }
                            }
                            if !dry_run {
                                eprintln!("Done. Downloaded {} extraction(s).", to_download.len());
                            }
                        }
                    }

                    other => {
                        #[derive(PartialEq)]
                        enum OutputKind { Text, Json, Get }
                        let (query, output_kind) = match other {
                            ZoteroExtractCommand::Text { query } => (query, OutputKind::Text),
                            ZoteroExtractCommand::Json { query } => (query, OutputKind::Json),
                            ZoteroExtractCommand::Get  { query } => (query, OutputKind::Get),
                            ZoteroExtractCommand::List { .. } => unreachable!(),
                            ZoteroExtractCommand::Upload { .. } => unreachable!(),
                            ZoteroExtractCommand::Download { .. } => unreachable!(),
                        };

                        let key = smart_resolve_item_key(&zotero, &query)
                            .await
                            .unwrap_or_else(|e| exit_err(&e));

                        match output_kind {
                            OutputKind::Text => match papers_core::text::datalab_cached_markdown(&key) {
                                Some(md) => print!("{md}"),
                                None => exit_err(&format!("No cached extraction for {key}. Run: papers zotero work extract {key}")),
                            },
                            OutputKind::Json => match papers_core::text::datalab_cached_json(&key) {
                                Some(json_str) => print!("{json_str}"),
                                None => exit_err(&format!("No cached extraction for {key}. Run: papers zotero work extract {key}")),
                            },
                            OutputKind::Get => {
                                let local_dir = papers_core::text::datalab_cache_dir_path(&key);
                                let local_ok = local_dir.as_ref()
                                    .map(|d| d.join(format!("{key}.md")).exists())
                                    .unwrap_or(false);
                                let item_exists = zotero.get_item(&key).await.is_ok();
                                // Only check for backup ZIP when the item exists
                                let remote_col = if item_exists {
                                    let expected_zip = format!("papers_extract_{key}.zip");
                                    let backup_ok = match zotero.list_item_children(
                                        &key,
                                        &ItemListParams { item_type: Some("attachment".into()), ..Default::default() },
                                    ).await {
                                        Ok(children) => children.items.iter().any(|c| {
                                            c.data.filename.as_deref() == Some(&expected_zip)
                                                && c.data.link_mode.as_deref() == Some("imported_file")
                                        }),
                                        Err(_) => false,
                                    };
                                    if backup_ok { "✓" } else { "✗" }
                                } else {
                                    "✗  *no item*"
                                };
                                println!("{key}");
                                match &local_dir {
                                    Some(dir) if local_ok => println!("  local:✓  {}", dir.display()),
                                    _ => println!("  local:✗"),
                                }
                                println!("  remote:{remote_col}");
                                if let Some(meta) = papers_core::text::read_extraction_meta(&key) {
                                    if let Some(t) = meta.title { println!("  title: {t}"); }
                                    if let Some(authors) = meta.authors { println!("  authors: {}", authors.join(", ")); }
                                    if let Some(et) = meta.extracted_at { println!("  extracted: {et}"); }
                                    if let Some(mode) = meta.processing_mode { println!("  mode: {mode}"); }
                                }
                                if !local_ok && remote_col == "✗" {
                                    eprintln!("No extraction found. Run: papers zotero work extract {key}");
                                }
                            }
                        }
                    }
                },

                ZoteroCommand::Attachment { cmd } => match cmd {
                    ZoteroAttachmentCommand::List { search, sort, direction, limit, start, json } => {
                        let params = ItemListParams { item_type: Some("attachment".into()), q: search, sort, direction, limit: Some(limit), start, ..Default::default() };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_attachment_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAttachmentCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json { print_json(&item); } else { print!("{}", format::format_zotero_item_get(&item)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAttachmentCommand::File { key, output } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.download_item_file(&key).await {
                            Ok(bytes) => {
                                if output == "-" {
                                    use std::io::Write;
                                    std::io::stdout().write_all(&bytes)
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
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item_file_view_url(&key).await {
                            Ok(url) => println!("{url}"),
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Annotation { cmd } => match cmd {
                    ZoteroAnnotationCommand::List { limit, start, json } => {
                        let params = ItemListParams { item_type: Some("annotation".into()), limit: Some(limit), start, ..Default::default() };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_annotation_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroAnnotationCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json { print_json(&item); } else { print!("{}", format::format_zotero_item_get(&item)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Note { cmd } => match cmd {
                    ZoteroNoteCommand::List { search, limit, start, json } => {
                        let params = ItemListParams { item_type: Some("note".into()), q: search, limit: Some(limit), start, ..Default::default() };
                        match zotero.list_items(&params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_note_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroNoteCommand::Get { key, json } => {
                        let key = resolve_item_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_item(&key).await {
                            Ok(item) => {
                                if json { print_json(&item); } else { print!("{}", format::format_zotero_item_get(&item)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Collection { cmd } => match cmd {
                    ZoteroCollectionCommand::List { sort, direction, limit, start, top, json } => {
                        let params = CollectionListParams { sort, direction, limit: Some(limit), start };
                        let result = if top {
                            zotero.list_top_collections(&params).await
                        } else {
                            zotero.list_collections(&params).await
                        };
                        match result {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_collection_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Get { key, json } => {
                        let key = resolve_collection_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_collection(&key).await {
                            Ok(coll) => {
                                if json { print_json(&coll); } else { print!("{}", format::format_zotero_collection_get(&coll)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Works { key, search, everything, tag, type_, sort, direction, limit, start, json } => {
                        let key = resolve_collection_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
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
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_work_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Attachments { key, sort, direction, limit, start, json } => {
                        let key = resolve_collection_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams { item_type: Some("attachment".into()), sort, direction, limit: Some(limit), start, ..Default::default() };
                        match zotero.list_collection_items(&key, &params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_attachment_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Notes { key, search, sort, direction, limit, start, json } => {
                        let key = resolve_collection_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = ItemListParams { item_type: Some("note".into()), q: search, sort, direction, limit: Some(limit), start, ..Default::default() };
                        match zotero.list_collection_items(&key, &params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_note_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Annotations { key, json } => {
                        let key = resolve_collection_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let att_params = ItemListParams { item_type: Some("attachment".into()), ..Default::default() };
                        let attachments = zotero.list_collection_items(&key, &att_params).await
                            .unwrap_or_else(|e| exit_err(&e.to_string()));
                        let ann_params = ItemListParams { item_type: Some("annotation".into()), ..Default::default() };
                        let mut all_annotations = Vec::new();
                        for att in &attachments.items {
                            if !is_annotatable_attachment(att) { continue; }
                            match zotero.list_item_children(&att.key, &ann_params).await {
                                Ok(r) => all_annotations.extend(r.items),
                                Err(_) => {},
                            }
                        }
                        if json { print_json(&all_annotations); } else { print!("{}", format::format_zotero_annotation_list_vec(&all_annotations)); }
                    }
                    ZoteroCollectionCommand::Subcollections { key, sort, direction, limit, start, json } => {
                        let key = resolve_collection_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = CollectionListParams { sort, direction, limit: Some(limit), start };
                        match zotero.list_subcollections(&key, &params).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_collection_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroCollectionCommand::Tags { key, search, limit, start, top, json } => {
                        let key = resolve_collection_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        let params = TagListParams { q: search, qmode: Some("contains".to_string()), limit, start, ..Default::default() };
                        let result = if top {
                            zotero.list_collection_top_items_tags(&key, &params).await
                        } else {
                            zotero.list_collection_items_tags(&key, &params).await
                        };
                        match result {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_tag_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Tag { cmd } => match cmd {
                    ZoteroTagCommand::List { search, sort, direction, limit, start, top, trash, json } => {
                        let params = TagListParams { q: search, qmode: Some("contains".to_string()), sort, direction, limit: Some(limit), start };
                        let result = if trash {
                            zotero.list_trash_tags(&params).await
                        } else if top {
                            zotero.list_top_items_tags(&params).await
                        } else {
                            zotero.list_tags(&params).await
                        };
                        match result {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_tag_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroTagCommand::Get { name, json } => {
                        match zotero.get_tag(&name).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_tag_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Search { cmd } => match cmd {
                    ZoteroSearchCommand::List { json } => {
                        match zotero.list_searches().await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_search_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroSearchCommand::Get { key, json } => {
                        let key = resolve_search_key(&zotero, &key).await.unwrap_or_else(|e| exit_err(&e.to_string()));
                        match zotero.get_search(&key).await {
                            Ok(search) => {
                                if json { print_json(&search); } else { print!("{}", format::format_zotero_search_get(&search)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Group { cmd } => match cmd {
                    ZoteroGroupCommand::List { json } => {
                        match zotero.list_groups().await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_group_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Setting { cmd } => match cmd {
                    ZoteroSettingCommand::List { json } => {
                        match zotero.get_settings().await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_setting_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                    ZoteroSettingCommand::Get { key, json } => {
                        match zotero.get_setting(&key).await {
                            Ok(resp) => {
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_setting_get(&key, &resp)); }
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
                                if json { print_json(&resp); } else { print!("{}", format::format_zotero_deleted_list(&resp)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },

                ZoteroCommand::Permission { cmd } => match cmd {
                    ZoteroPermissionCommand::List { json } => {
                        match zotero.get_current_key_info().await {
                            Ok(info) => {
                                if json { print_json(&info); } else { print!("{}", format::format_zotero_permission_list(&info)); }
                            }
                            Err(e) => exit_err(&e.to_string()),
                        }
                    }
                },
            }
        },

        EntityCommand::Selection { cmd } => {
            handle_selection_command(cmd, &client).await;
        }
    }
}

async fn handle_selection_command(cmd: SelectionCommand, client: &OpenAlexClient) {
    use papers_core::selection::{
        active_selection_name, delete_selection, entry_matches_remove_input,
        list_selection_names, load_selection, load_state, resolve_paper,
        resolve_selection, save_selection, save_state, validate_name, Selection,
    };

    match cmd {
        SelectionCommand::List { json } => {
            let names = list_selection_names();
            let state = load_state();
            let active = state.active.as_deref();
            let items: Vec<format::SelectionListItem> = names
                .iter()
                .map(|name| {
                    let count = load_selection(name)
                        .map(|s| s.entries.len())
                        .unwrap_or(0);
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

        SelectionCommand::Get { name, json } => {
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
            let is_active = true;
            if json {
                print_json(&serde_json::json!({
                    "name": sel.name,
                    "is_active": is_active,
                    "entries": sel.entries,
                }));
            } else {
                print!("{}", format::format_selection_get(&sel, is_active));
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

        SelectionCommand::Add { paper, selection, json } => {
            let sel_name = match selection {
                Some(s) => match resolve_selection(&s) {
                    Ok(n) => n,
                    Err(e) => exit_err(&e.to_string()),
                },
                None => match active_selection_name() {
                    Some(n) => n,
                    None => exit_err("no active selection; use --selection or create one first"),
                },
            };
            // For selection add, Zotero is optional — treat all errors as "not available"
            let zotero = optional_zotero().await.unwrap_or(None);
            let entry = match resolve_paper(&paper, client, zotero.as_ref()).await {
                Ok(e) => e,
                Err(e) => exit_err(&e.to_string()),
            };
            let mut sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            // Deduplication check
            let is_dup = sel.entries.iter().any(|e| {
                entry.zotero_key.as_deref().map(|k| papers_core::selection::entry_matches_key(e, k)).unwrap_or(false)
                    || entry.openalex_id.as_deref().map(|id| papers_core::selection::entry_matches_openalex(e, id)).unwrap_or(false)
                    || entry.doi.as_deref().map(|d| papers_core::selection::entry_matches_doi(e, d)).unwrap_or(false)
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

        SelectionCommand::Remove { paper, selection, json } => {
            let sel_name = match selection {
                Some(s) => match resolve_selection(&s) {
                    Ok(n) => n,
                    Err(e) => exit_err(&e.to_string()),
                },
                None => match active_selection_name() {
                    Some(n) => n,
                    None => exit_err("no active selection; use --selection or create one first"),
                },
            };
            let mut sel = load_selection(&sel_name).unwrap_or_else(|e| exit_err(&e.to_string()));
            let before = sel.entries.len();
            let removed_entry = sel.entries.iter().find(|e| entry_matches_remove_input(e, &paper)).cloned();
            sel.entries.retain(|e| !entry_matches_remove_input(e, &paper));
            if sel.entries.len() == before {
                exit_err("item not found in selection");
            }
            if let Err(e) = save_selection(&sel) {
                exit_err(&e.to_string());
            }
            let title = removed_entry
                .and_then(|e| e.title)
                .unwrap_or_else(|| paper.clone());
            if json {
                print_json(&serde_json::json!({ "removed": title, "selection": sel_name }));
            } else {
                print!("{}", format::format_selection_remove(&title, &sel_name));
            }
        }
    }
}
