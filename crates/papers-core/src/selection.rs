use std::path::PathBuf;

use papers_openalex::{GetParams, ListParams};
use papers_zotero::ItemListParams;
use serde::{Deserialize, Serialize};

// ── Error ──────────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum SelectionError {
    #[error("no data directory available")]
    NoDataDir,
    #[error("selection {0:?} not found")]
    NotFound(String),
    #[error("selection {0:?} already exists")]
    AlreadyExists(String),
    #[error("no active selection; run: papers selection list")]
    NoActiveSelection,
    #[error("invalid selection name {0:?}: use only alphanumeric, - and _")]
    InvalidName(String),
    #[error("item not found in selection")]
    ItemNotFound,
    #[error("could not resolve paper: {0}")]
    CannotResolve(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

// ── Data model ─────────────────────────────────────────────────────────────

/// A single paper in a selection. Stores as much metadata as resolved.
/// If `zotero_key` is None, the paper has not been matched to the local Zotero
/// library; metadata can be used later to prompt the user to download it.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SelectionEntry {
    pub zotero_key: Option<String>,
    pub openalex_id: Option<String>,
    pub doi: Option<String>,
    pub title: Option<String>,
    pub authors: Option<Vec<String>>,
    pub year: Option<u32>,
    pub issn: Option<Vec<String>>,
    pub isbn: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Selection {
    pub name: String,
    pub entries: Vec<SelectionEntry>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct SelectionState {
    pub active: Option<String>,
}

// ── Storage paths ──────────────────────────────────────────────────────────

/// Returns the selections directory, allowing `PAPERS_DATA_DIR` env var override
/// (used by tests). Falls back to `dirs::data_dir()/papers/selections`.
pub fn selections_dir() -> Option<PathBuf> {
    if let Ok(override_dir) = std::env::var("PAPERS_DATA_DIR") {
        let mut p = PathBuf::from(override_dir);
        p.push("papers");
        p.push("selections");
        return Some(p);
    }
    dirs::data_dir().map(|mut p| {
        p.push("papers");
        p.push("selections");
        p
    })
}

fn state_path() -> Option<PathBuf> {
    selections_dir().map(|mut p| {
        p.push("state.json");
        p
    })
}

fn selection_path(name: &str) -> Option<PathBuf> {
    selections_dir().map(|mut p| {
        p.push(format!("{name}.json"));
        p
    })
}

// ── Validation ─────────────────────────────────────────────────────────────

pub fn validate_name(name: &str) -> Result<(), SelectionError> {
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        Err(SelectionError::InvalidName(name.to_string()))
    } else {
        Ok(())
    }
}

// ── List & resolve ─────────────────────────────────────────────────────────

/// List all selection names (sorted alphabetically).
pub fn list_selection_names() -> Vec<String> {
    let dir = match selections_dir() {
        Some(d) => d,
        None => return Vec::new(),
    };
    if !dir.exists() {
        return Vec::new();
    }
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name();
            let s = name.to_string_lossy().into_owned();
            if s == "state.json" {
                return None;
            }
            s.strip_suffix(".json").map(|n| n.to_string())
        })
        .collect();
    names.sort();
    names
}

/// Resolve a name-or-index string to a selection name.
/// Accepts 1-based index or case-insensitive exact name match.
pub fn resolve_selection(input: &str) -> Result<String, SelectionError> {
    if let Ok(idx) = input.parse::<usize>() {
        let names = list_selection_names();
        if idx == 0 || idx > names.len() {
            return Err(SelectionError::NotFound(input.to_string()));
        }
        return Ok(names[idx - 1].clone());
    }
    let input_lower = input.to_lowercase();
    list_selection_names()
        .into_iter()
        .find(|n| n.to_lowercase() == input_lower)
        .ok_or_else(|| SelectionError::NotFound(input.to_string()))
}

// ── State ──────────────────────────────────────────────────────────────────

pub fn load_state() -> SelectionState {
    let path = match state_path() {
        Some(p) => p,
        None => return SelectionState::default(),
    };
    if !path.exists() {
        return SelectionState::default();
    }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

pub fn save_state(s: &SelectionState) -> Result<(), SelectionError> {
    let path = state_path().ok_or(SelectionError::NoDataDir)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(s)?;
    atomic_write(&path, &json)?;
    Ok(())
}

pub fn active_selection_name() -> Option<String> {
    load_state().active
}

// ── CRUD ───────────────────────────────────────────────────────────────────

pub fn load_selection(name: &str) -> Result<Selection, SelectionError> {
    let path = selection_path(name).ok_or(SelectionError::NoDataDir)?;
    if !path.exists() {
        return Err(SelectionError::NotFound(name.to_string()));
    }
    let s = std::fs::read_to_string(&path)?;
    Ok(serde_json::from_str(&s)?)
}

pub fn save_selection(sel: &Selection) -> Result<(), SelectionError> {
    let path = selection_path(&sel.name).ok_or(SelectionError::NoDataDir)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(sel)?;
    atomic_write(&path, &json)?;
    Ok(())
}

pub fn delete_selection(name: &str) -> Result<(), SelectionError> {
    let path = selection_path(name).ok_or(SelectionError::NoDataDir)?;
    if !path.exists() {
        return Err(SelectionError::NotFound(name.to_string()));
    }
    std::fs::remove_file(&path)?;
    Ok(())
}

fn atomic_write(path: &PathBuf, content: &str) -> Result<(), std::io::Error> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content.as_bytes())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

// ── Deduplication helpers ──────────────────────────────────────────────────

pub fn entry_matches_key(entry: &SelectionEntry, zotero_key: &str) -> bool {
    entry.zotero_key.as_deref() == Some(zotero_key)
}

pub fn entry_matches_openalex(entry: &SelectionEntry, oa_id: &str) -> bool {
    entry.openalex_id.as_deref() == Some(oa_id)
}

pub fn entry_matches_doi(entry: &SelectionEntry, doi: &str) -> bool {
    let normalized = normalize_doi(doi);
    entry
        .doi
        .as_deref()
        .map(normalize_doi)
        .as_deref()
        .map(|d| d == normalized.as_str())
        .unwrap_or(false)
}

/// Match a removal input against an entry (by key, OA ID, DOI, or title substring).
pub fn entry_matches_remove_input(entry: &SelectionEntry, input: &str) -> bool {
    if crate::zotero::looks_like_zotero_key(input) && entry_matches_key(entry, input) {
        return true;
    }
    let id = input
        .strip_prefix("https://openalex.org/")
        .unwrap_or(input);
    if looks_like_openalex_work_id(id) && entry_matches_openalex(entry, id) {
        return true;
    }
    if looks_like_doi(input) && entry_matches_doi(entry, input) {
        return true;
    }
    if let Some(title) = &entry.title {
        if title.to_lowercase().contains(&input.to_lowercase()) {
            return true;
        }
    }
    false
}

// ── Input type detection ───────────────────────────────────────────────────

pub fn looks_like_doi(input: &str) -> bool {
    let s = input
        .strip_prefix("https://doi.org/")
        .or_else(|| input.strip_prefix("http://doi.org/"))
        .or_else(|| input.strip_prefix("doi:"))
        .unwrap_or(input);
    s.starts_with("10.") && s.contains('/')
}

pub fn looks_like_openalex_work_id(input: &str) -> bool {
    let id = input
        .strip_prefix("https://openalex.org/")
        .unwrap_or(input);
    id.starts_with('W') && id.len() > 1 && id[1..].chars().all(|c| c.is_ascii_digit())
}

pub fn strip_doi_prefix(doi: &str) -> &str {
    doi.strip_prefix("https://doi.org/")
        .or_else(|| doi.strip_prefix("http://doi.org/"))
        .or_else(|| doi.strip_prefix("doi:"))
        .unwrap_or(doi)
}

fn normalize_doi(doi: &str) -> String {
    strip_doi_prefix(doi).to_lowercase()
}

// ── Smart add resolution ───────────────────────────────────────────────────

/// Resolve a paper input string to a SelectionEntry.
/// Tries Zotero first (if available), then OpenAlex. Merges metadata from both.
pub async fn resolve_paper(
    input: &str,
    client: &papers_openalex::OpenAlexClient,
    zotero: Option<&papers_zotero::ZoteroClient>,
) -> Result<SelectionEntry, SelectionError> {
    let input = input.trim();
    let mut entry = SelectionEntry {
        zotero_key: None,
        openalex_id: None,
        doi: None,
        title: None,
        authors: None,
        year: None,
        issn: None,
        isbn: None,
    };

    let is_zotero_key = crate::zotero::looks_like_zotero_key(input);
    let is_doi = looks_like_doi(input);
    let is_oa_id = looks_like_openalex_work_id(input);

    // Step 2: Attempt Zotero resolution
    if let Some(z) = zotero {
        if is_zotero_key {
            if let Ok(item) = z.get_item(input).await {
                entry.zotero_key = Some(item.key.clone());
                fill_from_zotero_item(&mut entry, &item);
            }
        } else if is_doi {
            let bare = strip_doi_prefix(input);
            let params = ItemListParams {
                q: Some(bare.to_string()),
                qmode: Some("everything".into()),
                limit: Some(1),
                ..Default::default()
            };
            if let Ok(resp) = z.list_top_items(&params).await {
                if let Some(item) = resp.items.into_iter().next() {
                    entry.zotero_key = Some(item.key.clone());
                    fill_from_zotero_item(&mut entry, &item);
                }
            }
        } else if !is_oa_id {
            // Free-text / title search in Zotero
            let params = ItemListParams::builder().q(input).limit(1).build();
            if let Ok(resp) = z.list_top_items(&params).await {
                // Only auto-pick if there is exactly 1 result
                if resp.items.len() == 1 {
                    let item = resp.items.into_iter().next().unwrap();
                    entry.zotero_key = Some(item.key.clone());
                    fill_from_zotero_item(&mut entry, &item);
                }
            }
        }
    }

    // Step 3: Attempt OpenAlex resolution
    let oa_work = resolve_via_openalex(input, client, is_doi, is_oa_id).await;
    if let Some(work) = oa_work {
        fill_from_oa_work(&mut entry, &work);

        // Step 4: Retry Zotero with DOI if step 2 failed but OA found a DOI
        if entry.zotero_key.is_none() {
            if let (Some(z), Some(doi)) = (zotero, &entry.doi.clone()) {
                let bare = strip_doi_prefix(doi);
                let params = ItemListParams {
                    q: Some(bare.to_string()),
                    qmode: Some("everything".into()),
                    limit: Some(1),
                    ..Default::default()
                };
                if let Ok(resp) = z.list_top_items(&params).await {
                    if let Some(item) = resp.items.into_iter().next() {
                        entry.zotero_key = Some(item.key.clone());
                        // Only fill Zotero fields not already set by OA
                        if entry.isbn.is_none() {
                            if let Some(isbn) = &item.data.isbn {
                                if !isbn.is_empty() {
                                    entry.isbn = Some(vec![isbn.clone()]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Step 5: Fail if nothing at all was resolved
    if entry.zotero_key.is_none()
        && entry.openalex_id.is_none()
        && entry.doi.is_none()
        && entry.title.is_none()
    {
        return Err(SelectionError::CannotResolve(input.to_string()));
    }

    Ok(entry)
}

async fn resolve_via_openalex(
    input: &str,
    client: &papers_openalex::OpenAlexClient,
    is_doi: bool,
    is_oa_id: bool,
) -> Option<papers_openalex::Work> {
    if is_doi {
        let bare = strip_doi_prefix(input);
        let oa_id = format!("doi:{bare}");
        client
            .get_work(&oa_id, &GetParams::default())
            .await
            .ok()
    } else if is_oa_id {
        let id = input
            .strip_prefix("https://openalex.org/")
            .unwrap_or(input);
        client.get_work(id, &GetParams::default()).await.ok()
    } else {
        let params = ListParams {
            search: Some(input.to_string()),
            per_page: Some(1),
            ..Default::default()
        };
        client
            .list_works(&params)
            .await
            .ok()
            .and_then(|resp| resp.results.into_iter().next())
    }
}

fn fill_from_zotero_item(entry: &mut SelectionEntry, item: &papers_zotero::Item) {
    if entry.title.is_none() {
        entry.title = item.data.title.clone();
    }
    if entry.authors.is_none() {
        let authors: Vec<String> = item
            .data
            .creators
            .iter()
            .filter_map(|c| {
                if let (Some(first), Some(last)) = (&c.first_name, &c.last_name) {
                    let name = format!("{first} {last}").trim().to_string();
                    if !name.is_empty() {
                        return Some(name);
                    }
                }
                c.name.clone().filter(|n| !n.is_empty())
            })
            .collect();
        if !authors.is_empty() {
            entry.authors = Some(authors);
        }
    }
    if entry.year.is_none() {
        // Try parsed_date first, then raw date field
        let date_str = item
            .meta
            .parsed_date
            .as_deref()
            .or_else(|| item.data.date.as_deref());
        entry.year = date_str
            .and_then(|d| d.split('-').next())
            .and_then(|y| y.parse().ok());
    }
    if entry.doi.is_none() {
        entry.doi = item.data.doi.as_deref().map(|d| {
            strip_doi_prefix(d).to_string()
        });
    }
    if entry.issn.is_none() {
        if let Some(issn) = &item.data.issn {
            if !issn.is_empty() {
                entry.issn = Some(vec![issn.clone()]);
            }
        }
    }
    if entry.isbn.is_none() {
        if let Some(isbn) = &item.data.isbn {
            if !isbn.is_empty() {
                entry.isbn = Some(vec![isbn.clone()]);
            }
        }
    }
}

fn fill_from_oa_work(entry: &mut SelectionEntry, work: &papers_openalex::Work) {
    if entry.openalex_id.is_none() {
        let id = work
            .id
            .strip_prefix("https://openalex.org/")
            .unwrap_or(&work.id);
        entry.openalex_id = Some(id.to_string());
    }
    if entry.doi.is_none() {
        entry.doi = work.doi.as_deref().map(|d| {
            d.strip_prefix("https://doi.org/")
                .or_else(|| d.strip_prefix("http://doi.org/"))
                .unwrap_or(d)
                .to_string()
        });
    }
    if entry.title.is_none() {
        entry.title = work.display_name.clone().or_else(|| work.title.clone());
    }
    if entry.authors.is_none() {
        if let Some(authorships) = &work.authorships {
            let names: Vec<String> = authorships
                .iter()
                .filter_map(|a| a.author.as_ref()?.display_name.clone())
                .collect();
            if !names.is_empty() {
                entry.authors = Some(names);
            }
        }
    }
    if entry.year.is_none() {
        entry.year = work.publication_year.map(|y| y as u32);
    }
    if entry.issn.is_none() {
        entry.issn = work
            .primary_location
            .as_ref()
            .and_then(|l| l.source.as_ref())
            .and_then(|s| s.issn.clone());
    }
}
