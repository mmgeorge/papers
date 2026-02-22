use papers_core::api::WorkGetResponse;
use papers_core::summary::{
    AuthorSummary, DomainSummary, FieldSummary, FunderSummary, InstitutionSummary,
    PublisherSummary, SlimListResponse, SourceSummary, SubfieldSummary, TopicSummary, WorkSummary,
};
use papers_core::text::WorkTextResult;
use papers_core::{
    Author, AutocompleteResponse, Domain, Field, FindWorksResponse, Funder, Institution, ListMeta,
    Publisher, Source, Subfield, Topic, Work,
};
use std::collections::HashMap;
use papers_zotero::{Collection, Creator, DeletedObjects, Group, Item, ItemFulltext, PagedResponse, SavedSearch, SettingEntry, Tag, VersionedResponse};

// ── Meta line ─────────────────────────────────────────────────────────────

fn meta_line(meta: &ListMeta) -> String {
    let page = meta.page.unwrap_or(1);
    let per_page = meta.per_page.unwrap_or(10);
    format!(
        "Found {:} results · page {} (showing {})",
        meta.count, page, per_page
    )
}

// ── Work ──────────────────────────────────────────────────────────────────

pub fn format_work_list(resp: &SlimListResponse<WorkSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, w) in resp.results.iter().enumerate() {
        let title = w.title.as_deref().unwrap_or("(untitled)");
        let year = w
            .publication_year
            .map_or(String::new(), |y| format!(" ({y})"));
        out.push_str(&format!("\n {:>2}  {}{}\n", i + 1, title, year));

        if !w.authors.is_empty() {
            out.push_str(&format!("     {}\n", w.authors.join(" · ")));
        }

        let mut meta_parts = Vec::new();
        if let Some(j) = &w.journal {
            meta_parts.push(j.clone());
        }
        if let Some(t) = &w.r#type {
            meta_parts.push(t.clone());
        }
        if let Some(c) = w.cited_by_count {
            meta_parts.push(format!("{c} citations"));
        }
        let oa = match w.is_oa {
            Some(true) => "OA: Yes",
            Some(false) => "OA: No",
            None => "",
        };
        if !oa.is_empty() {
            meta_parts.push(oa.to_string());
        }
        if !meta_parts.is_empty() {
            out.push_str(&format!("     {}\n", meta_parts.join(" · ")));
        }

        if let Some(topic) = &w.primary_topic {
            out.push_str(&format!("     Topic: {topic}\n"));
        }
        if let Some(doi) = &w.doi {
            out.push_str(&format!("     DOI: {doi}\n"));
        }
        if let Some(abs) = &w.abstract_text {
            let snippet = if abs.len() > 200 {
                format!("{}…", abs.chars().take(200).collect::<String>())
            } else {
                abs.clone()
            };
            out.push_str(&format!("\n     {snippet}\n"));
        }
    }
    out
}

pub fn format_work_get(w: &Work) -> String {
    let mut out = String::new();
    let title = w.display_name.as_deref().unwrap_or("(untitled)");
    out.push_str(&format!("Work: {title}\n"));
    out.push_str(&format!("ID:   {}\n", w.id));
    if let Some(doi) = &w.doi {
        out.push_str(&format!("DOI:  {doi}\n"));
    }

    let mut year_parts = Vec::new();
    if let Some(y) = w.publication_year {
        year_parts.push(format!("Year: {y}"));
    }
    if let Some(t) = &w.r#type {
        year_parts.push(format!("Type: {t}"));
    }
    if !year_parts.is_empty() {
        out.push_str(&format!("{}\n", year_parts.join(" · ")));
    }

    if let Some(oa) = &w.open_access {
        let is_oa = oa.is_oa.map_or("unknown", |b| if b { "Yes" } else { "No" });
        let mut oa_str = format!("OA:   {is_oa}");
        if let Some(status) = &oa.oa_status {
            oa_str.push_str(&format!(" ({status})"));
        }
        if let Some(url) = &oa.oa_url {
            oa_str.push_str(&format!(" · {url}"));
        }
        out.push_str(&format!("{oa_str}\n"));
    }

    if let Some(c) = w.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }

    if let Some(topic) = &w.primary_topic {
        if let Some(name) = &topic.display_name {
            let mut topic_str = name.clone();
            let parts: Vec<_> = [
                topic.subfield.as_ref().and_then(|s| s.display_name.as_deref()),
                topic.field.as_ref().and_then(|f| f.display_name.as_deref()),
                topic.domain.as_ref().and_then(|d| d.display_name.as_deref()),
            ]
            .into_iter()
            .flatten()
            .collect();
            if !parts.is_empty() {
                topic_str.push_str(&format!(" ({})", parts.join(" → ")));
            }
            out.push_str(&format!("Topic: {topic_str}\n"));
        }
    }

    let authorships = w.authorships.as_deref().unwrap_or_default();
    if !authorships.is_empty() {
        out.push_str("\nAuthors:\n");
        for (i, a) in authorships.iter().enumerate() {
            let name = a
                .author
                .as_ref()
                .and_then(|au| au.display_name.as_deref())
                .unwrap_or("?");
            let pos = a.author_position.as_deref().unwrap_or("");
            let inst = a
                .institutions
                .as_deref()
                .unwrap_or_default()
                .first()
                .and_then(|i| i.display_name.as_deref())
                .unwrap_or("");
            out.push_str(&format!("  {:>2}. {name} ({pos})  {inst}\n", i + 1));
        }
    }

    if let Some(abs) = &w.abstract_text {
        out.push_str(&format!("\nAbstract:\n  {abs}\n"));
    }

    out
}

pub fn format_work_get_response(response: &WorkGetResponse, zotero_configured: bool) -> String {
    let mut out = format_work_get(&response.work);
    if zotero_configured {
        out.push('\n');
        if let Some(z) = &response.zotero {
            out.push_str("Zotero:\n");
            out.push_str(&format!("  Key:   {}\n", z.key));
            out.push_str(&format!("  Open:  {}\n", z.uri));
            out.push_str(&format!("  Type:  {}\n", z.item_type));
            out.push_str(&format!("  PDF:   {}\n", if z.has_pdf { "Yes" } else { "No" }));
            if !z.tags.is_empty() {
                out.push_str(&format!("  Tags:  {}\n", z.tags.join(", ")));
            }
            if let Some(date) = &z.date_added {
                out.push_str(&format!("  Added: {}\n", &date[..10.min(date.len())]));
            }
        } else {
            out.push_str("Zotero: not in library\n");
        }
    }
    out
}

// ── Author ────────────────────────────────────────────────────────────────

pub fn format_author_list(resp: &SlimListResponse<AuthorSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, a) in resp.results.iter().enumerate() {
        let name = a.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        let mut parts = Vec::new();
        if let Some(h) = a.h_index {
            parts.push(format!("h-index: {h}"));
        }
        if let Some(c) = a.cited_by_count {
            parts.push(format!("{c} citations"));
        }
        if let Some(w) = a.works_count {
            parts.push(format!("{w} works"));
        }
        if !parts.is_empty() {
            out.push_str(&format!("     {}\n", parts.join(" · ")));
        }

        if !a.last_known_institutions.is_empty() {
            out.push_str(&format!(
                "     {}\n",
                a.last_known_institutions.join(", ")
            ));
        }
        if !a.top_topics.is_empty() {
            out.push_str(&format!("     Topics: {}\n", a.top_topics.join(", ")));
        }
        if let Some(orcid) = &a.orcid {
            out.push_str(&format!("     ORCID: {orcid}\n"));
        }
    }
    out
}

pub fn format_author_get(a: &Author) -> String {
    let mut out = String::new();
    let name = a.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Author: {name}\n"));
    out.push_str(&format!("ID:     {}\n", a.id));

    if let Some(orcid) = &a.orcid {
        out.push_str(&format!("ORCID:  {orcid}\n"));
    }
    let mut stats = Vec::new();
    if let Some(w) = a.works_count {
        stats.push(format!("{w} works"));
    }
    if let Some(c) = a.cited_by_count {
        stats.push(format!("{c} citations"));
    }
    if let Some(ss) = &a.summary_stats {
        if let Some(h) = ss.h_index {
            stats.push(format!("h-index: {h}"));
        }
    }
    if !stats.is_empty() {
        out.push_str(&format!("{}\n", stats.join(" · ")));
    }

    let insts = a.last_known_institutions.as_deref().unwrap_or_default();
    if !insts.is_empty() {
        out.push_str("Institutions:\n");
        for inst in insts {
            if let Some(name) = &inst.display_name {
                out.push_str(&format!("  {name}\n"));
            }
        }
    }

    out
}

// ── Source ────────────────────────────────────────────────────────────────

pub fn format_source_list(resp: &SlimListResponse<SourceSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, s) in resp.results.iter().enumerate() {
        let name = s.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        let mut parts = Vec::new();
        if let Some(issn) = &s.issn_l {
            parts.push(format!("ISSN: {issn}"));
        }
        if let Some(t) = &s.r#type {
            parts.push(t.clone());
        }
        let oa = match s.is_oa {
            Some(true) => Some("OA: Yes"),
            Some(false) => Some("OA: No"),
            None => None,
        };
        if let Some(o) = oa {
            parts.push(o.to_string());
        }
        if let Some(h) = s.h_index {
            parts.push(format!("h-index: {h}"));
        }
        if !parts.is_empty() {
            out.push_str(&format!("     {}\n", parts.join(" · ")));
        }

        if let Some(org) = &s.host_organization_name {
            out.push_str(&format!("     Publisher: {org}\n"));
        }
    }
    out
}

pub fn format_source_get(s: &Source) -> String {
    let mut out = String::new();
    let name = s.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Source: {name}\n"));
    out.push_str(&format!("ID:     {}\n", s.id));
    if let Some(issn) = &s.issn_l {
        out.push_str(&format!("ISSN-L: {issn}\n"));
    }
    if let Some(t) = &s.r#type {
        out.push_str(&format!("Type:   {t}\n"));
    }
    let oa_str = match s.is_oa {
        Some(true) => "Yes",
        Some(false) => "No",
        None => "?",
    };
    out.push_str(&format!("OA:     {oa_str}\n"));
    if let Some(org) = &s.host_organization_name {
        out.push_str(&format!("Publisher: {org}\n"));
    }
    if let Some(c) = s.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }
    out
}

// ── Institution ───────────────────────────────────────────────────────────

pub fn format_institution_list(resp: &SlimListResponse<InstitutionSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, inst) in resp.results.iter().enumerate() {
        let name = inst.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        let mut parts = Vec::new();
        if let Some(city) = &inst.city {
            parts.push(city.clone());
        }
        if let Some(cc) = &inst.country_code {
            parts.push(cc.clone());
        }
        if let Some(t) = &inst.r#type {
            parts.push(t.clone());
        }
        if !parts.is_empty() {
            out.push_str(&format!("     {}\n", parts.join(", ")));
        }

        let mut stats = Vec::new();
        if let Some(h) = inst.h_index {
            stats.push(format!("h-index: {h}"));
        }
        if let Some(c) = inst.cited_by_count {
            stats.push(format!("{c} citations"));
        }
        if !stats.is_empty() {
            out.push_str(&format!("     {}\n", stats.join(" · ")));
        }
    }
    out
}

pub fn format_institution_get(inst: &Institution) -> String {
    let mut out = String::new();
    let name = inst.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Institution: {name}\n"));
    out.push_str(&format!("ID:          {}\n", inst.id));
    if let Some(ror) = &inst.ror {
        out.push_str(&format!("ROR:         {ror}\n"));
    }
    if let Some(t) = &inst.r#type {
        out.push_str(&format!("Type:        {t}\n"));
    }
    if let Some(geo) = &inst.geo {
        let mut loc = Vec::new();
        if let Some(city) = &geo.city {
            loc.push(city.as_str());
        }
        if let Some(country) = &geo.country {
            loc.push(country.as_str());
        }
        if !loc.is_empty() {
            out.push_str(&format!("Location:    {}\n", loc.join(", ")));
        }
    }
    if let Some(c) = inst.cited_by_count {
        out.push_str(&format!("Citations:   {c}\n"));
    }
    out
}

// ── Topic ─────────────────────────────────────────────────────────────────

pub fn format_topic_list(resp: &SlimListResponse<TopicSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, t) in resp.results.iter().enumerate() {
        let name = t.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        let hierarchy: Vec<_> = [t.subfield.as_deref(), t.field.as_deref(), t.domain.as_deref()]
            .into_iter()
            .flatten()
            .collect();
        if !hierarchy.is_empty() {
            out.push_str(&format!("     {}\n", hierarchy.join(" → ")));
        }
        if let Some(desc) = &t.description {
            let snippet = if desc.len() > 150 {
                format!("{}…", desc.chars().take(150).collect::<String>())
            } else {
                desc.clone()
            };
            out.push_str(&format!("     {snippet}\n"));
        }
        if let Some(c) = t.cited_by_count {
            out.push_str(&format!("     {c} citations\n"));
        }
    }
    out
}

pub fn format_topic_get(t: &Topic) -> String {
    let mut out = String::new();
    let name = t.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Topic: {name}\n"));
    out.push_str(&format!("ID:    {}\n", t.id));

    let hierarchy: Vec<_> = [
        t.subfield.as_ref().and_then(|s| s.display_name.as_deref()),
        t.field.as_ref().and_then(|f| f.display_name.as_deref()),
        t.domain.as_ref().and_then(|d| d.display_name.as_deref()),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !hierarchy.is_empty() {
        out.push_str(&format!("Hierarchy: {}\n", hierarchy.join(" → ")));
    }
    if let Some(desc) = &t.description {
        out.push_str(&format!("\nDescription:\n  {desc}\n"));
    }
    if let Some(c) = t.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }
    out
}

// ── Publisher ─────────────────────────────────────────────────────────────

pub fn format_publisher_list(resp: &SlimListResponse<PublisherSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, p) in resp.results.iter().enumerate() {
        let name = p.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        let mut parts = Vec::new();
        if let Some(level) = p.hierarchy_level {
            parts.push(format!("level {level}"));
        }
        if let Some(codes) = &p.country_codes {
            parts.push(codes.join(", "));
        }
        if let Some(c) = p.cited_by_count {
            parts.push(format!("{c} citations"));
        }
        if !parts.is_empty() {
            out.push_str(&format!("     {}\n", parts.join(" · ")));
        }
    }
    out
}

pub fn format_publisher_get(p: &Publisher) -> String {
    let mut out = String::new();
    let name = p.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Publisher: {name}\n"));
    out.push_str(&format!("ID:        {}\n", p.id));
    if let Some(level) = p.hierarchy_level {
        out.push_str(&format!("Level:     {level}\n"));
    }
    if let Some(codes) = &p.country_codes {
        out.push_str(&format!("Countries: {}\n", codes.join(", ")));
    }
    if let Some(c) = p.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }
    out
}

// ── Funder ────────────────────────────────────────────────────────────────

pub fn format_funder_list(resp: &SlimListResponse<FunderSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, f) in resp.results.iter().enumerate() {
        let name = f.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        let mut parts = Vec::new();
        if let Some(cc) = &f.country_code {
            parts.push(cc.clone());
        }
        if let Some(a) = f.awards_count {
            parts.push(format!("{a} awards"));
        }
        if let Some(c) = f.cited_by_count {
            parts.push(format!("{c} citations"));
        }
        if !parts.is_empty() {
            out.push_str(&format!("     {}\n", parts.join(" · ")));
        }
        if let Some(desc) = &f.description {
            out.push_str(&format!("     {desc}\n"));
        }
    }
    out
}

pub fn format_funder_get(f: &Funder) -> String {
    let mut out = String::new();
    let name = f.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Funder: {name}\n"));
    out.push_str(&format!("ID:     {}\n", f.id));
    if let Some(cc) = &f.country_code {
        out.push_str(&format!("Country: {cc}\n"));
    }
    if let Some(desc) = &f.description {
        out.push_str(&format!("Description: {desc}\n"));
    }
    if let Some(a) = f.awards_count {
        out.push_str(&format!("Awards:  {a}\n"));
    }
    if let Some(c) = f.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }
    out
}

// ── Domain ────────────────────────────────────────────────────────────────

pub fn format_domain_list(resp: &SlimListResponse<DomainSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, d) in resp.results.iter().enumerate() {
        let name = d.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        if let Some(desc) = &d.description {
            out.push_str(&format!("     {desc}\n"));
        }
        if !d.fields.is_empty() {
            out.push_str(&format!("     Fields: {}\n", d.fields.join(", ")));
        }
        let mut stats = Vec::new();
        if let Some(w) = d.works_count {
            stats.push(format!("{w} works"));
        }
        if let Some(c) = d.cited_by_count {
            stats.push(format!("{c} citations"));
        }
        if !stats.is_empty() {
            out.push_str(&format!("     {}\n", stats.join(" · ")));
        }
    }
    out
}

pub fn format_domain_get(d: &Domain) -> String {
    let mut out = String::new();
    let name = d.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Domain: {name}\n"));
    out.push_str(&format!("ID:     {}\n", d.id));
    if let Some(desc) = &d.description {
        out.push_str(&format!("Description: {desc}\n"));
    }
    if let Some(fields) = &d.fields {
        if !fields.is_empty() {
            out.push_str("Fields:\n");
            for f in fields {
                if let Some(name) = &f.display_name {
                    out.push_str(&format!("  {name}\n"));
                }
            }
        }
    }
    if let Some(w) = d.works_count {
        out.push_str(&format!("Works: {w}\n"));
    }
    if let Some(c) = d.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }
    out
}

// ── Field ─────────────────────────────────────────────────────────────────

pub fn format_field_list(resp: &SlimListResponse<FieldSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, f) in resp.results.iter().enumerate() {
        let name = f.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        if let Some(domain) = &f.domain {
            out.push_str(&format!("     Domain: {domain}\n"));
        }
        let mut parts = Vec::new();
        parts.push(format!("{} subfields", f.subfield_count));
        if let Some(w) = f.works_count {
            parts.push(format!("{w} works"));
        }
        if let Some(c) = f.cited_by_count {
            parts.push(format!("{c} citations"));
        }
        out.push_str(&format!("     {}\n", parts.join(" · ")));

        if let Some(desc) = &f.description {
            let snippet = if desc.len() > 150 {
                format!("{}…", desc.chars().take(150).collect::<String>())
            } else {
                desc.clone()
            };
            out.push_str(&format!("     {snippet}\n"));
        }
    }
    out
}

pub fn format_field_get(f: &Field) -> String {
    let mut out = String::new();
    let name = f.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Field: {name}\n"));
    out.push_str(&format!("ID:    {}\n", f.id));
    if let Some(domain) = &f.domain {
        if let Some(dn) = &domain.display_name {
            out.push_str(&format!("Domain: {dn}\n"));
        }
    }
    if let Some(desc) = &f.description {
        out.push_str(&format!("Description: {desc}\n"));
    }
    if let Some(subfields) = &f.subfields {
        if !subfields.is_empty() {
            out.push_str("Subfields:\n");
            for sf in subfields {
                if let Some(name) = &sf.display_name {
                    out.push_str(&format!("  {name}\n"));
                }
            }
        }
    }
    if let Some(w) = f.works_count {
        out.push_str(&format!("Works: {w}\n"));
    }
    if let Some(c) = f.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }
    out
}

// ── Subfield ──────────────────────────────────────────────────────────────

pub fn format_subfield_list(resp: &SlimListResponse<SubfieldSummary>) -> String {
    let mut out = format!("{}\n", meta_line(&resp.meta));
    for (i, s) in resp.results.iter().enumerate() {
        let name = s.display_name.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  {name}\n", i + 1));

        let hierarchy: Vec<_> = [s.field.as_deref(), s.domain.as_deref()]
            .into_iter()
            .flatten()
            .collect();
        if !hierarchy.is_empty() {
            out.push_str(&format!("     {}\n", hierarchy.join(" → ")));
        }
        if let Some(desc) = &s.description {
            let snippet = if desc.len() > 150 {
                format!("{}…", desc.chars().take(150).collect::<String>())
            } else {
                desc.clone()
            };
            out.push_str(&format!("     {snippet}\n"));
        }
        if let Some(c) = s.cited_by_count {
            out.push_str(&format!("     {c} citations\n"));
        }
    }
    out
}

pub fn format_subfield_get(s: &Subfield) -> String {
    let mut out = String::new();
    let name = s.display_name.as_deref().unwrap_or("?");
    out.push_str(&format!("Subfield: {name}\n"));
    out.push_str(&format!("ID:       {}\n", s.id));

    let hierarchy: Vec<_> = [
        s.field.as_ref().and_then(|f| f.display_name.as_deref()),
        s.domain.as_ref().and_then(|d| d.display_name.as_deref()),
    ]
    .into_iter()
    .flatten()
    .collect();
    if !hierarchy.is_empty() {
        out.push_str(&format!("Hierarchy: {}\n", hierarchy.join(" → ")));
    }
    if let Some(desc) = &s.description {
        out.push_str(&format!("\nDescription:\n  {desc}\n"));
    }
    if let Some(w) = s.works_count {
        out.push_str(&format!("Works: {w}\n"));
    }
    if let Some(c) = s.cited_by_count {
        out.push_str(&format!("Citations: {c}\n"));
    }
    out
}

// ── Autocomplete ──────────────────────────────────────────────────────────

pub fn format_autocomplete(resp: &AutocompleteResponse) -> String {
    let mut out = String::new();
    for (i, r) in resp.results.iter().enumerate() {
        out.push_str(&format!("{:>2}  {} [{}]\n", i + 1, r.display_name, r.short_id.as_deref().unwrap_or("")));
        if let Some(hint) = &r.hint {
            if !hint.is_empty() {
                out.push_str(&format!("    {hint}\n"));
            }
        }
        if let Some(c) = r.cited_by_count {
            out.push_str(&format!("    {c} citations\n"));
        }
    }
    if out.is_empty() {
        out.push_str("No results.\n");
    }
    out
}

// ── Find works ────────────────────────────────────────────────────────────

pub fn format_find_works(resp: &FindWorksResponse) -> String {
    let mut out = String::new();
    if resp.results.is_empty() {
        return "No results.\n".to_string();
    }
    for (i, r) in resp.results.iter().enumerate() {
        let name = r.work["display_name"].as_str().unwrap_or("(untitled)");
        out.push_str(&format!("{:>2}  {name}\n", i + 1));
        if let Some(id) = r.work["id"].as_str() {
            out.push_str(&format!("    ID: {id}\n"));
        }
        out.push_str(&format!("    Score: {:.3}\n", r.score));
    }
    out
}

// ── Zotero ────────────────────────────────────────────────────────────────

fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn creator_display(c: &Creator) -> String {
    if let Some(name) = &c.name {
        name.clone()
    } else {
        match (&c.first_name, &c.last_name) {
            (Some(f), Some(l)) => format!("{l}, {f}"),
            (None, Some(l)) => l.clone(),
            (Some(f), None) => f.clone(),
            _ => "?".to_string(),
        }
    }
}

pub fn format_zotero_work_list(resp: &PagedResponse<Item>) -> String {
    let header = match resp.total_results {
        Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
        _ => format!("{} item(s)\n", resp.items.len()),
    };
    let mut out = header;
    for (i, item) in resp.items.iter().enumerate() {
        let title = item.data.title.as_deref().unwrap_or("(untitled)");
        let year = item
            .data
            .date
            .as_deref()
            .and_then(|d| {
                let y = d.split(['-', '/']).next()?;
                if y.len() == 4 && y.chars().all(|c| c.is_ascii_digit()) {
                    Some(format!(" ({y})"))
                } else {
                    None
                }
            })
            .unwrap_or_default();
        out.push_str(&format!("\n {:>2}  [{}] {}{}\n", i + 1, item.key, title, year));
        if !item.data.creators.is_empty() {
            let authors: Vec<String> = item.data.creators.iter().take(3).map(creator_display).collect();
            let suffix = if item.data.creators.len() > 3 { " et al." } else { "" };
            out.push_str(&format!("     {}{suffix}\n", authors.join("; ")));
        }
        let mut meta_parts = Vec::new();
        if let Some(j) = &item.data.publication_title {
            if !j.is_empty() {
                meta_parts.push(j.clone());
            }
        }
        meta_parts.push(item.data.item_type.clone());
        if let Some(doi) = &item.data.doi {
            if !doi.is_empty() {
                meta_parts.push(format!("DOI: {doi}"));
            }
        }
        if !meta_parts.is_empty() {
            out.push_str(&format!("     {}\n", meta_parts.join(" · ")));
        }
        if !item.data.tags.is_empty() {
            let tag_names: Vec<&str> = item.data.tags.iter().map(|t| t.tag.as_str()).collect();
            out.push_str(&format!("     Tags: {}\n", tag_names.join(", ")));
        }
    }
    out
}

pub fn format_zotero_item_get(item: &Item) -> String {
    let mut out = String::new();
    let title = item.data.title.as_deref().unwrap_or("(untitled)");
    out.push_str(&format!("{}: {}\n", item.data.item_type, title));
    out.push_str(&format!("Key:  {}\n", item.key));
    out.push_str(&format!("Open: zotero://select/library/items/{}\n", item.key));
    if let Some(doi) = &item.data.doi {
        if !doi.is_empty() { out.push_str(&format!("DOI:  {doi}\n")); }
    }
    if let Some(date) = &item.data.date {
        if !date.is_empty() { out.push_str(&format!("Date: {date}\n")); }
    }
    if let Some(j) = &item.data.publication_title {
        if !j.is_empty() { out.push_str(&format!("Publication: {j}\n")); }
    }
    if let Some(p) = &item.data.publisher {
        if !p.is_empty() { out.push_str(&format!("Publisher: {p}\n")); }
    }
    if let Some(url) = &item.data.url {
        if !url.is_empty() { out.push_str(&format!("URL: {url}\n")); }
    }
    if !item.data.creators.is_empty() {
        out.push_str("\nCreators:\n");
        for (i, c) in item.data.creators.iter().enumerate() {
            let name = creator_display(c);
            out.push_str(&format!("  {:>2}. {} ({})\n", i + 1, name, c.creator_type));
        }
    }
    if let Some(abs) = &item.data.abstract_note {
        if !abs.is_empty() { out.push_str(&format!("\nAbstract:\n  {abs}\n")); }
    }
    if !item.data.tags.is_empty() {
        let tag_names: Vec<&str> = item.data.tags.iter().map(|t| t.tag.as_str()).collect();
        out.push_str(&format!("\nTags: {}\n", tag_names.join(", ")));
    }
    // Attachment-specific
    if let Some(link_mode) = &item.data.link_mode {
        out.push_str(&format!("Link mode: {link_mode}\n"));
    }
    if let Some(filename) = &item.data.filename {
        out.push_str(&format!("Filename: {filename}\n"));
    }
    if let Some(ct) = &item.data.content_type {
        out.push_str(&format!("Content type: {ct}\n"));
    }
    if let Some(parent) = &item.data.parent_item {
        out.push_str(&format!("Parent: {parent}\n"));
    }
    // Note content
    if let Some(note) = &item.data.note {
        let stripped = strip_html(note);
        let trimmed = stripped.trim();
        if !trimmed.is_empty() {
            out.push_str(&format!("\nNote:\n  {trimmed}\n"));
        }
    }
    // Annotation fields
    if let Some(ann_type) = item.data.extra_fields.get("annotationType").and_then(|v| v.as_str()) {
        out.push_str(&format!("Annotation type: {ann_type}\n"));
    }
    if let Some(text) = item.data.extra_fields.get("annotationText").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            let snippet = if text.chars().count() > 500 {
                format!("{}…", text.chars().take(500).collect::<String>())
            } else {
                text.to_string()
            };
            out.push_str(&format!("Text: {snippet}\n"));
        }
    }
    if let Some(comment) = item.data.extra_fields.get("annotationComment").and_then(|v| v.as_str()) {
        if !comment.is_empty() {
            out.push_str(&format!("Comment: {comment}\n"));
        }
    }
    if let Some(page) = item.data.extra_fields.get("annotationPageLabel").and_then(|v| v.as_str()) {
        out.push_str(&format!("Page: {page}\n"));
    }
    if let Some(color) = item.data.extra_fields.get("annotationColor").and_then(|v| v.as_str()) {
        out.push_str(&format!("Color: {color}\n"));
    }
    out
}

pub fn format_zotero_attachment_list(resp: &PagedResponse<Item>) -> String {
    let header = match resp.total_results {
        Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
        _ => format!("{} attachment(s)\n", resp.items.len()),
    };
    let mut out = header;
    for (i, item) in resp.items.iter().enumerate() {
        let display = item
            .data
            .filename
            .as_deref()
            .or_else(|| item.data.url.as_deref())
            .unwrap_or("(no file)");
        let link_mode = item.data.link_mode.as_deref().unwrap_or("?");
        out.push_str(&format!("\n {:>2}  [{}] {}\n", i + 1, item.key, display));
        let mut parts = vec![link_mode.to_string()];
        if let Some(parent) = &item.data.parent_item {
            parts.push(format!("parent: {parent}"));
        }
        if let Some(ct) = &item.data.content_type {
            parts.push(ct.clone());
        }
        out.push_str(&format!("     {}\n", parts.join(" · ")));
    }
    out
}

pub fn format_zotero_annotation_list(resp: &PagedResponse<Item>) -> String {
    if resp.items.is_empty() {
        return "No annotations.\n".to_string();
    }
    let header = match resp.total_results.filter(|&n| n > 0) {
        Some(n) => format!("Found {} annotations · showing {}\n", n, resp.items.len()),
        None => format!("{} annotation(s)\n", resp.items.len()),
    };
    let mut out = header;
    for (i, item) in resp.items.iter().enumerate() {
        push_annotation_entry(&mut out, i, item);
    }
    out
}

pub fn format_zotero_annotation_list_vec(items: &[Item]) -> String {
    if items.is_empty() {
        return "No annotations.\n".to_string();
    }
    let mut out = format!("{} annotation(s)\n", items.len());
    for (i, item) in items.iter().enumerate() {
        push_annotation_entry(&mut out, i, item);
    }
    out
}

fn push_annotation_entry(out: &mut String, i: usize, item: &Item) {
    let ann_type = item
        .data
        .extra_fields
        .get("annotationType")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let page = item
        .data
        .extra_fields
        .get("annotationPageLabel")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let color = item
        .data
        .extra_fields
        .get("annotationColor")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let parent = item.data.parent_item.as_deref().unwrap_or("");
    out.push_str(&format!("\n {:>2}  [{}] {}", i + 1, item.key, ann_type));
    if !page.is_empty() {
        out.push_str(&format!(" (p. {page})"));
    }
    if !color.is_empty() {
        out.push_str(&format!(" {color}"));
    }
    out.push('\n');
    if !parent.is_empty() {
        out.push_str(&format!("     Parent: {parent}\n"));
    }
    if let Some(text) = item.data.extra_fields.get("annotationText").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            let snippet = if text.chars().count() > 120 {
                format!("{}…", text.chars().take(120).collect::<String>())
            } else {
                text.to_string()
            };
            out.push_str(&format!("     \"{snippet}\"\n"));
        }
    }
    if let Some(comment) = item.data.extra_fields.get("annotationComment").and_then(|v| v.as_str()) {
        if !comment.is_empty() {
            out.push_str(&format!("     Note: {comment}\n"));
        }
    }
}

pub fn format_zotero_note_list(resp: &PagedResponse<Item>) -> String {
    let header = match resp.total_results {
        Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
        _ => format!("{} note(s)\n", resp.items.len()),
    };
    let mut out = header;
    for (i, item) in resp.items.iter().enumerate() {
        let parent = item.data.parent_item.as_deref().unwrap_or("(standalone)");
        out.push_str(&format!("\n {:>2}  [{}] parent: {parent}\n", i + 1, item.key));
        if let Some(note) = &item.data.note {
            let stripped = strip_html(note);
            let trimmed = stripped.trim().to_string();
            if !trimmed.is_empty() {
                let preview = if trimmed.chars().count() > 80 {
                    format!("{}…", trimmed.chars().take(80).collect::<String>())
                } else {
                    trimmed
                };
                out.push_str(&format!("     {preview}\n"));
            }
        }
    }
    out
}

pub fn format_zotero_collection_list(resp: &PagedResponse<Collection>) -> String {
    let header = match resp.total_results {
        Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
        _ => format!("{} collection(s)\n", resp.items.len()),
    };
    format_zotero_collection_list_inner(&resp.items, header)
}

pub fn format_zotero_collection_list_vec(collections: &[Collection]) -> String {
    let header = format!("{} collection(s)\n", collections.len());
    format_zotero_collection_list_inner(collections, header)
}

fn format_zotero_collection_list_inner(collections: &[Collection], header: String) -> String {
    let mut out = header;
    for (i, coll) in collections.iter().enumerate() {
        let parent = coll.data.parent_key().map(|k| format!(" (sub of {k})")).unwrap_or_default();
        let items = coll.meta.num_items.map(|n| format!(", {n} items")).unwrap_or_default();
        out.push_str(&format!("\n {:>2}  [{}] {}{parent}{items}\n", i + 1, coll.key, coll.data.name));
    }
    out
}

pub fn format_zotero_collection_get(coll: &Collection) -> String {
    let mut out = String::new();
    out.push_str(&format!("Collection: {}\n", coll.data.name));
    out.push_str(&format!("Key:   {}\n", coll.key));
    if let Some(parent_key) = coll.data.parent_key() {
        out.push_str(&format!("Parent: {parent_key}\n"));
    } else {
        out.push_str("Parent: (top-level)\n");
    }
    if let Some(n) = coll.meta.num_items {
        out.push_str(&format!("Items: {n}\n"));
    }
    if let Some(n) = coll.meta.num_collections {
        out.push_str(&format!("Sub-collections: {n}\n"));
    }
    out
}

pub fn format_zotero_tag_list(resp: &PagedResponse<Tag>) -> String {
    let header = match resp.total_results {
        Some(n) if n > 0 => format!("Found {} tags · showing {}\n", n, resp.items.len()),
        _ => format!("{} tag(s)\n", resp.items.len()),
    };
    let mut out = header;
    for (i, tag) in resp.items.iter().enumerate() {
        let count = tag.meta.num_items.map(|n| format!(" ({n} items)")).unwrap_or_default();
        out.push_str(&format!("\n {:>2}  {}{count}\n", i + 1, tag.tag));
    }
    out
}

pub fn format_zotero_search_list(resp: &PagedResponse<SavedSearch>) -> String {
    let header = match resp.total_results {
        Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
        _ => format!("{} saved search(es)\n", resp.items.len()),
    };
    let mut out = header;
    for (i, search) in resp.items.iter().enumerate() {
        let n = search.data.conditions.len();
        out.push_str(&format!("\n {:>2}  [{}] {}\n", i + 1, search.key, search.data.name));
        out.push_str(&format!("     {n} condition(s)\n"));
    }
    out
}

pub fn format_zotero_search_get(search: &SavedSearch) -> String {
    let mut out = String::new();
    out.push_str(&format!("Search: {}\n", search.data.name));
    out.push_str(&format!("Key:    {}\n", search.key));
    if !search.data.conditions.is_empty() {
        out.push_str("\nConditions:\n");
        for cond in &search.data.conditions {
            out.push_str(&format!("  {} {} {}\n", cond.condition, cond.operator, cond.value));
        }
    }
    out
}

pub fn format_zotero_group_list(resp: &PagedResponse<Group>) -> String {
    let header = match resp.total_results {
        Some(n) if n > 0 => format!("Found {} results · showing {}\n", n, resp.items.len()),
        _ => format!("{} group(s)\n", resp.items.len()),
    };
    let mut out = header;
    for (i, group) in resp.items.iter().enumerate() {
        let gtype = group.data.group_type.as_deref().unwrap_or("?");
        let items = group.meta.num_items.map(|n| format!(", {n} items")).unwrap_or_default();
        out.push_str(&format!("\n {:>2}  [{}] {} ({gtype}){items}\n", i + 1, group.id, group.data.name));
        if let Some(desc) = &group.data.description {
            if !desc.is_empty() {
                let snippet = if desc.chars().count() > 100 {
                    format!("{}…", desc.chars().take(100).collect::<String>())
                } else {
                    desc.clone()
                };
                out.push_str(&format!("     {snippet}\n"));
            }
        }
    }
    out
}

// ── Work text ─────────────────────────────────────────────────────────────

pub fn format_work_text(result: &WorkTextResult) -> String {
    let mut out = String::new();
    if let Some(title) = &result.title {
        out.push_str(&format!("Work: {title}\n"));
    }
    out.push_str(&format!("ID:   {}\n", result.work_id));
    if let Some(doi) = &result.doi {
        out.push_str(&format!("DOI:  {doi}\n"));
    }
    let source_desc = match &result.source {
        papers_core::text::PdfSource::ZoteroLocal { path } => format!("Zotero (local: {path})"),
        papers_core::text::PdfSource::ZoteroRemote { item_key } => {
            format!("Zotero (remote: {item_key})")
        }
        papers_core::text::PdfSource::DirectUrl { url } => format!("Direct URL: {url}"),
        papers_core::text::PdfSource::OpenAlexContent => "OpenAlex Content API".to_string(),
        papers_core::text::PdfSource::DataLab => "DataLab Marker API".to_string(),
    };
    out.push_str(&format!("Source: {source_desc}\n"));
    out.push_str(&format!(
        "Length: {} characters\n\n",
        result.text.len()
    ));
    out.push_str(&result.text);
    if !result.text.ends_with('\n') {
        out.push('\n');
    }
    out
}

// ── Zotero fulltext ───────────────────────────────────────────────────────

pub fn format_zotero_work_fulltext(resp: &VersionedResponse<ItemFulltext>) -> String {
    let ft = &resp.data;
    let mut out = String::new();
    if let Some(v) = resp.last_modified_version {
        out.push_str(&format!("Version: {v}\n"));
    }
    let mut stats = Vec::new();
    if let Some(p) = ft.indexed_pages {
        stats.push(format!("{p} indexed pages"));
    }
    if let Some(p) = ft.total_pages {
        stats.push(format!("{p} total pages"));
    }
    if let Some(c) = ft.indexed_chars {
        stats.push(format!("{c} indexed chars"));
    }
    if !stats.is_empty() {
        out.push_str(&format!("{}\n", stats.join(" · ")));
    }
    out.push('\n');
    out.push_str(&ft.content);
    if !ft.content.ends_with('\n') {
        out.push('\n');
    }
    out
}

// ── Zotero settings ───────────────────────────────────────────────────────

pub fn format_zotero_setting_list(resp: &VersionedResponse<HashMap<String, SettingEntry>>) -> String {
    let mut out = String::new();
    if let Some(v) = resp.last_modified_version {
        out.push_str(&format!("Library version: {v}\n"));
    }
    let mut keys: Vec<_> = resp.data.keys().collect();
    keys.sort();
    for key in keys {
        let entry = &resp.data[key];
        let val_str = serde_json::to_string(&entry.value).unwrap_or_else(|_| "?".to_string());
        out.push_str(&format!("  {key} (v{}): {val_str}\n", entry.version));
    }
    out
}

pub fn format_zotero_setting_get(key: &str, resp: &VersionedResponse<SettingEntry>) -> String {
    let mut out = String::new();
    if let Some(v) = resp.last_modified_version {
        out.push_str(&format!("Version: {v}\n"));
    }
    let val_str = serde_json::to_string_pretty(&resp.data.value).unwrap_or_else(|_| "?".to_string());
    out.push_str(&format!("Setting: {key} (v{})\n", resp.data.version));
    out.push_str(&format!("Value:\n  {val_str}\n"));
    out
}

// ── Zotero deleted ────────────────────────────────────────────────────────

pub fn format_zotero_deleted_list(resp: &VersionedResponse<DeletedObjects>) -> String {
    let d = &resp.data;
    let mut out = String::new();
    if let Some(v) = resp.last_modified_version {
        out.push_str(&format!("Library version: {v}\n"));
    }
    let print_section = |out: &mut String, label: &str, keys: &[String]| {
        if !keys.is_empty() {
            out.push_str(&format!("\n{label} ({}):\n", keys.len()));
            for k in keys {
                out.push_str(&format!("  {k}\n"));
            }
        }
    };
    print_section(&mut out, "Deleted items", &d.items);
    print_section(&mut out, "Deleted collections", &d.collections);
    print_section(&mut out, "Deleted searches", &d.searches);
    print_section(&mut out, "Deleted tags", &d.tags);
    print_section(&mut out, "Deleted settings", &d.settings);
    if d.items.is_empty() && d.collections.is_empty() && d.searches.is_empty()
        && d.tags.is_empty() && d.settings.is_empty()
    {
        out.push_str("No deletions since requested version.\n");
    }
    out
}

// ── Zotero permission ─────────────────────────────────────────────────────

pub fn format_zotero_permission_list(info: &serde_json::Value) -> String {
    let mut out = String::new();
    if let Some(uid) = info.get("userID").and_then(|v| v.as_u64()) {
        out.push_str(&format!("User ID:  {uid}\n"));
    }
    if let Some(username) = info.get("username").and_then(|v| v.as_str()) {
        out.push_str(&format!("Username: {username}\n"));
    }
    if let Some(key) = info.get("key").and_then(|v| v.as_str()) {
        out.push_str(&format!("Key:      {key}\n"));
    }
    if let Some(access) = info.get("access") {
        let access_str = serde_json::to_string_pretty(access).unwrap_or_else(|_| "?".to_string());
        out.push_str(&format!("\nAccess:\n"));
        for line in access_str.lines() {
            out.push_str(&format!("  {line}\n"));
        }
    }
    out
}

// ── Selection ──────────────────────────────────────────────────────────────

pub struct SelectionListItem {
    pub name: String,
    pub item_count: usize,
    pub is_active: bool,
}

pub fn format_selection_list(items: &[SelectionListItem]) -> String {
    if items.is_empty() {
        return "No selections. Create one with: papers selection create <name>\n".to_string();
    }
    let mut out = String::new();
    for (i, item) in items.iter().enumerate() {
        let active = if item.is_active { "* " } else { "  " };
        out.push_str(&format!(
            "{}{:>2}  {} ({} item{})\n",
            active,
            i + 1,
            item.name,
            item.item_count,
            if item.item_count == 1 { "" } else { "s" }
        ));
    }
    out
}

pub fn format_selection_get(sel: &papers_core::selection::Selection, is_active: bool) -> String {
    let active = if is_active { " (active)" } else { "" };
    let mut out = format!("Selection: {}{}\n", sel.name, active);
    out.push_str(&format!("Items: {}\n", sel.entries.len()));
    if sel.entries.is_empty() {
        out.push_str("\n(empty — add papers with: papers selection add <paper>)\n");
        return out;
    }
    out.push('\n');
    for (i, entry) in sel.entries.iter().enumerate() {
        let title = entry.title.as_deref().unwrap_or("(untitled)");
        let year = entry.year.map_or(String::new(), |y| format!(" ({y})"));
        out.push_str(&format!("  {:>2}  {}{}\n", i + 1, title, year));
        if let Some(authors) = &entry.authors {
            if !authors.is_empty() {
                out.push_str(&format!("       {}\n", authors.join(" · ")));
            }
        }
        let mut ids = Vec::new();
        if let Some(k) = &entry.zotero_key {
            ids.push(format!("Zotero: {k}"));
        }
        if let Some(oa) = &entry.openalex_id {
            ids.push(format!("OA: {oa}"));
        }
        if let Some(doi) = &entry.doi {
            ids.push(format!("DOI: {doi}"));
        }
        if !ids.is_empty() {
            out.push_str(&format!("       {}\n", ids.join("  ")));
        }
    }
    out
}

pub fn format_selection_create(name: &str) -> String {
    format!("Created selection {name:?} (now active)\n")
}

pub fn format_selection_delete(name: &str, was_active: bool) -> String {
    if was_active {
        format!("Deleted selection {name:?} (was active; no selection now active)\n")
    } else {
        format!("Deleted selection {name:?}\n")
    }
}

pub fn format_selection_add(
    entry: &papers_core::selection::SelectionEntry,
    selection_name: &str,
) -> String {
    let title = entry.title.as_deref().unwrap_or("(unknown)");
    format!("Added {title:?} to selection {selection_name:?}\n")
}

pub fn format_selection_remove(title: &str, selection_name: &str) -> String {
    format!("Removed {title:?} from selection {selection_name:?}\n")
}

