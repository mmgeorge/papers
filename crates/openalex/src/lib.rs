//! Async Rust client for the [OpenAlex REST API](https://docs.openalex.org).
//!
//! OpenAlex is a free, open catalog of the world's scholarly research system,
//! containing 240M+ works, 110M+ authors, and related metadata across sources,
//! institutions, topics, publishers, and funders.
//!
//! # Quick start
//!
//! ```no_run
//! # async fn example() -> openalex::Result<()> {
//! use openalex::{OpenAlexClient, ListParams};
//!
//! let client = OpenAlexClient::new();
//!
//! // Search for works about machine learning
//! let params = ListParams::builder()
//!     .search("machine learning")
//!     .per_page(5)
//!     .build();
//! let response = client.list_works(&params).await?;
//! println!("Found {} works", response.meta.count);
//! for work in &response.results {
//!     println!("  - {}", work.display_name.as_deref().unwrap_or("untitled"));
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Authentication
//!
//! An API key is optional for most endpoints but recommended for higher rate
//! limits. Set the `OPENALEX_KEY` environment variable, or pass it explicitly:
//!
//! ```no_run
//! use openalex::OpenAlexClient;
//!
//! // Reads OPENALEX_KEY from environment
//! let client = OpenAlexClient::new();
//!
//! // Or pass explicitly
//! let client = OpenAlexClient::with_api_key("your-key-here");
//! ```
//!
//! The `/find/works` semantic search endpoint **requires** an API key and costs
//! 1,000 credits per request.
//!
//! # Endpoints
//!
//! The client provides 23 methods covering all OpenAlex API endpoints:
//!
//! - **7 list endpoints** — paginated entity lists with filtering, searching,
//!   sorting, sampling, and grouping
//! - **7 get endpoints** — single entity retrieval by ID (OpenAlex, DOI, ORCID,
//!   ROR, ISSN, PMID, etc.)
//! - **7 autocomplete endpoints** — fast type-ahead search (~200ms, up to 10
//!   results)
//! - **2 semantic search endpoints** — AI-powered similarity search via GET or
//!   POST

pub mod client;
pub mod error;
pub mod params;
pub mod response;
pub mod types;

pub use client::OpenAlexClient;
pub use error::{OpenAlexError, Result};
pub use params::{FindWorksParams, GetParams, ListParams};
pub use response::{
    AutocompleteResponse, AutocompleteResult, FindWorksResponse, FindWorksResult, GroupByResult,
    ListMeta, ListResponse,
};
pub use types::*;
