mod cli;
mod format;

use clap::Parser;
use cli::{
    AuthorCommand, AuthorFilterArgs, Cli, DomainCommand, DomainFilterArgs,
    EntityCommand, FieldCommand, FieldFilterArgs, FunderCommand, FunderFilterArgs,
    InstitutionCommand, InstitutionFilterArgs, PublisherCommand, PublisherFilterArgs,
    SourceCommand, SourceFilterArgs, SubfieldCommand, SubfieldFilterArgs, TopicCommand,
    TopicFilterArgs, WorkCommand, WorkFilterArgs,
};
use papers_core::{
    AuthorListParams, DiskCache, DomainListParams, FieldListParams, FindWorksParams,
    FunderListParams, GetParams, InstitutionListParams, OpenAlexClient, PublisherListParams,
    SourceListParams, SubfieldListParams, TopicListParams, WorkListParams,
};
use papers_zotero::ZoteroClient;
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
            WorkCommand::Text { id, json, no_prompt } => {
                let zotero = ZoteroClient::from_env().ok();
                match papers_core::text::work_text(&client, zotero.as_ref(), &id).await {
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

    }
}
