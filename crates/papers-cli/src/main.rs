mod cli;
mod format;

use clap::Parser;
use cli::{
    AdvancedMode, AuthorCommand, AuthorFilterArgs, Cli, DomainCommand, DomainFilterArgs,
    EntityCommand, FieldCommand, FieldFilterArgs, FunderCommand, FunderFilterArgs,
    InstitutionCommand, InstitutionFilterArgs, PublisherCommand, PublisherFilterArgs,
    SourceCommand, SourceFilterArgs, SubfieldCommand, SubfieldFilterArgs, TopicCommand,
    TopicFilterArgs, WorkCommand, WorkFilterArgs, ZoteroAnnotationCommand, ZoteroAttachmentCommand,
    ZoteroCollectionCommand, ZoteroCommand, ZoteroGroupCommand, ZoteroNoteCommand,
    ZoteroSearchCommand, ZoteroTagCommand, ZoteroWorkCommand,
};
use papers_core::{
    AuthorListParams, DiskCache, DomainListParams, FieldListParams, FindWorksParams,
    FunderListParams, GetParams, InstitutionListParams, OpenAlexClient, PublisherListParams,
    SourceListParams, SubfieldListParams, TopicListParams, WorkListParams,
};
use papers_core::zotero::{resolve_collection_key, resolve_item_key, resolve_search_key};
use papers_zotero::{CollectionListParams, Item, ItemListParams, TagListParams, ZoteroClient};
use std::time::Duration;

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

#[tokio::main]
async fn main() {
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
                match papers_core::api::work_get(&client, &id, &GetParams::default()).await {
                    Ok(work) => {
                        if json {
                            print_json(&work);
                        } else {
                            print!("{}", format::format_work_get(&work));
                        }
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
                let zotero = ZoteroClient::from_env().ok();
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
            let zotero = ZoteroClient::from_env().unwrap_or_else(|_| {
                exit_err("Zotero not configured. Set ZOTERO_USER_ID and ZOTERO_API_KEY.")
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
            }
        },

    }
}
