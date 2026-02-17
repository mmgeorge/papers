pub mod api;
pub mod summary;

pub use papers_openalex::{
    Author, Domain, Field, Funder, HierarchyEntity, HierarchyIds, Institution, Publisher, Source,
    Subfield, Topic, Work,
    OpenAlexClient, OpenAlexError, Result,
    ListParams, GetParams, FindWorksParams,
    ListMeta, ListResponse,
    AutocompleteResponse, AutocompleteResult,
    FindWorksResponse, FindWorksResult,
    GroupByResult,
};
pub use summary::SlimListResponse;
