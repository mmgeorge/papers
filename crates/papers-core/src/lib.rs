pub mod api;
pub mod filter;
pub mod selection;
pub mod summary;
pub mod text;
pub mod zotero;

pub use api::WorkGetResponse;
pub use selection::{
    Selection, SelectionEntry, SelectionError, SelectionState,
    active_selection_name, delete_selection, entry_matches_doi, entry_matches_key,
    entry_matches_openalex, entry_matches_remove_input, list_selection_names,
    load_selection, load_state, looks_like_doi as selection_looks_like_doi,
    looks_like_openalex_work_id, resolve_paper, resolve_selection, save_selection,
    save_state, selections_dir, strip_doi_prefix, validate_name,
};
pub use filter::{
    AuthorListParams, DomainListParams, FieldListParams, FilterError, FunderListParams,
    InstitutionListParams, PublisherListParams, SourceListParams, SubfieldListParams,
    TopicListParams, WorkListParams,
};
pub use text::ZoteroItemInfo;
pub use papers_openalex::{
    Author, Domain, Field, Funder, HierarchyEntity, HierarchyIds, Institution, Publisher, Source,
    Subfield, Topic, Work,
    DiskCache,
    OpenAlexClient, OpenAlexError, Result,
    ListParams, GetParams, FindWorksParams,
    ListMeta, ListResponse,
    AutocompleteResponse, AutocompleteResult,
    FindWorksResponse, FindWorksResult,
    GroupByResult,
};
pub use summary::SlimListResponse;
