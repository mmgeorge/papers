mod cli;
mod format;

use clap::Parser;
use cli::{
    AuthorCommand, Cli, ConceptCommand, EntityCommand, FunderCommand, InstitutionCommand,
    PublisherCommand, SourceCommand, TopicCommand, WorkCommand,
};
use papers::{FindWorksParams, GetParams, ListParams, OpenAlexClient};

fn list_params_from_args(args: &cli::ListArgs) -> ListParams {
    ListParams {
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
    let client = OpenAlexClient::new();

    match cli.entity {
        EntityCommand::Work { cmd } => match cmd {
            WorkCommand::List { args } => {
                let params = list_params_from_args(&args);
                match papers::api::work_list(&client, &params).await {
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
                match papers::api::work_get(&client, &id, &GetParams::default()).await {
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
                match papers::api::work_autocomplete(&client, &query).await {
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
                match papers::api::work_find(&client, &params).await {
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
            AuthorCommand::List { args } => {
                let params = list_params_from_args(&args);
                match papers::api::author_list(&client, &params).await {
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
                match papers::api::author_get(&client, &id, &GetParams::default()).await {
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
                match papers::api::author_autocomplete(&client, &query).await {
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
            SourceCommand::List { args } => {
                let params = list_params_from_args(&args);
                match papers::api::source_list(&client, &params).await {
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
                match papers::api::source_get(&client, &id, &GetParams::default()).await {
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
                match papers::api::source_autocomplete(&client, &query).await {
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
            InstitutionCommand::List { args } => {
                let params = list_params_from_args(&args);
                match papers::api::institution_list(&client, &params).await {
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
                match papers::api::institution_get(&client, &id, &GetParams::default()).await {
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
                match papers::api::institution_autocomplete(&client, &query).await {
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
            TopicCommand::List { args } => {
                let params = list_params_from_args(&args);
                match papers::api::topic_list(&client, &params).await {
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
                match papers::api::topic_get(&client, &id, &GetParams::default()).await {
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
            PublisherCommand::List { args } => {
                let params = list_params_from_args(&args);
                match papers::api::publisher_list(&client, &params).await {
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
                match papers::api::publisher_get(&client, &id, &GetParams::default()).await {
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
                match papers::api::publisher_autocomplete(&client, &query).await {
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
            FunderCommand::List { args } => {
                let params = list_params_from_args(&args);
                match papers::api::funder_list(&client, &params).await {
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
                match papers::api::funder_get(&client, &id, &GetParams::default()).await {
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
                match papers::api::funder_autocomplete(&client, &query).await {
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

        EntityCommand::Concept { cmd } => match cmd {
            ConceptCommand::Autocomplete { query, json } => {
                match papers::api::concept_autocomplete(&client, &query).await {
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
