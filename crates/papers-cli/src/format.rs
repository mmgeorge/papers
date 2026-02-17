use papers::summary::{
    AuthorSummary, DomainSummary, FieldSummary, FunderSummary, InstitutionSummary,
    PublisherSummary, SlimListResponse, SourceSummary, SubfieldSummary, TopicSummary, WorkSummary,
};
use papers::{
    Author, AutocompleteResponse, Domain, Field, FindWorksResponse, Funder, Institution, ListMeta,
    Publisher, Source, Subfield, Topic, Work,
};

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
                format!("{}…", &abs[..200])
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
                format!("{}…", &desc[..150])
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
                format!("{}…", &desc[..150])
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
                format!("{}…", &desc[..150])
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

