# papers

A command-line interface for the [OpenAlex](https://openalex.org) academic research database.
Query 240M+ scholarly works, authors, journals, institutions, topics, publishers, and funders.

## Install

```sh
cargo install --path crates/papers-cli
```

## Usage

```
papers <COMMAND>

Commands:
  work         Scholarly works: articles, preprints, datasets, and more
  author       Disambiguated researcher profiles
  source       Publishing venues: journals, repositories, conferences
  institution  Research organizations: universities, hospitals, companies
  topic        Research topic hierarchy (domain → field → subfield → topic)
  publisher    Publishing organizations (e.g. Elsevier, Springer Nature)
  funder       Grant-making organizations (e.g. NIH, NSF, ERC)
  concept      Deprecated concept taxonomy (autocomplete only)

Options:
  -h, --help  Print help
```

Every command accepts `--json` to output raw JSON instead of formatted text.

### Listing

```
papers work list [OPTIONS]

Options:
  -s, --search <SEARCH>      Full-text search query
  -f, --filter <FILTER>      Filter expression (e.g. "publication_year:2024,is_oa:true")
      --sort <SORT>          Sort field with optional :desc (e.g. "cited_by_count:desc")
  -n, --per-page <PER_PAGE>  Results per page [default: 10]
      --page <PAGE>          Page number for offset pagination
      --cursor <CURSOR>      Cursor for cursor-based pagination (use "*" to start)
      --sample <SAMPLE>      Random sample of N results
      --seed <SEED>          Seed for reproducible sampling
      --json                 Output raw JSON
```

`author`, `source`, `institution`, `topic`, `publisher`, and `funder` all share the same list options.

## Examples

### Search for works

```
$ papers work list -s "attention is all you need" -n 3
```

```
Found 1556581 results · page 1 (showing 3)

  1  Attention Is All You Need (2025)
     Ashish Vaswani · Noam Shazeer · Niki Parmar · Jakob Uszkoreit · Llion Jones · Aidan N. Gomez · Łukasz Kaiser · Illia Polosukhin
     preprint · 6488 citations · OA: Yes
     Topic: Natural Language Processing Techniques
     DOI: https://doi.org/10.65215/2q58a426

     The dominant sequence transduction models are based on complex recurrent or convolutional neural
     networks in an encoder-decoder configuration...

  2  Attention Is All You Need In Speech Separation (2021)
     Cem Subakan · Mirco Ravanelli · Samuele Cornell · Mirko Bronzi · Jianyuan Zhong
     article · 574 citations · OA: No
     Topic: Speech and Audio Processing
     DOI: https://doi.org/10.1109/icassp39728.2021.9413901
     ...
```

### Filter and sort

```
$ papers work list -f "publication_year:2024,is_oa:true" --sort cited_by_count:desc -n 3
```

```
Found 6211989 results · page 1 (showing 3)

  1  Global cancer statistics 2022: GLOBOCAN estimates... (2024)
     Freddie Bray · Mathieu Laversanne · ...
     CA A Cancer Journal for Clinicians · article · 19449 citations · OA: Yes
     Topic: Global Cancer Incidence and Screening
     DOI: https://doi.org/10.3322/caac.21834
     ...
```

### Get a single work

```
$ papers work get W2741809807
```

```
Work: The state of OA: a large-scale analysis of the prevalence and impact of Open Access articles
ID:   https://openalex.org/W2741809807
DOI:  https://doi.org/10.7717/peerj.4375
Year: 2018 · Type: book-chapter
OA:   Yes (gold) · https://doi.org/10.7717/peerj.4375
Citations: 1149
Topic: scientometrics and bibliometrics research (Statistics, Probability and Uncertainty → Decision Sciences → Social Sciences)

Authors:
   1. Heather Piwowar (first)  Impact Technology Development (United States)
   2. Jason Priem (middle)  Impact Technology Development (United States)
   3. Vincent Larivière (middle)  Université de Montréal
   ...

Abstract:
  Despite growing interest in Open Access (OA) to scholarly literature, there is an unmet need for
  large-scale, up-to-date, and reproducible studies assessing the prevalence and characteristics of OA...
```

You can also look up works by DOI: `papers work get https://doi.org/10.7717/peerj.4375`

### Author autocomplete

```
$ papers author autocomplete "yoshua bengio"
```

```
 1  Yoshua Bengio [authors/A5028826050]
    Mila - Quebec Artificial Intelligence Institute, Canada
    1270 citations
 2  Yoshua Bengio [authors/A5125732408]
    Mila - Quebec Artificial Intelligence Institute, Canada
    0 citations
...
```

### List journals

```
$ papers source list -s "nature" -n 3
```

```
Found 278 results · page 1 (showing 3)

  1  Nature
     ISSN: 0028-0836 · journal · OA: No · h-index: 1822
     Publisher: Springer Nature

  2  Nature Communications
     ISSN: 2041-1723 · journal · OA: Yes · h-index: 719
     Publisher: Springer Nature

  3  Nature Genetics
     ISSN: 1061-4036 · journal · OA: No · h-index: 776
     Publisher: Springer Nature
```

### JSON output

Append `--json` to any command to get machine-readable output:

```
$ papers work get W2741809807 --json
```

```json
{
  "id": "https://openalex.org/W2741809807",
  "doi": "https://doi.org/10.7717/peerj.4375",
  "title": "The state of OA: a large-scale analysis...",
  "publication_year": 2018,
  "type": "book-chapter",
  "cited_by_count": 1149,
  "referenced_works": [...],
  ...
}
```

List responses in `--json` mode return a slim subset of fields (no `referenced_works`,
`counts_by_year`, etc.) for conciseness. Use `work get --json` to retrieve the full record.

### Semantic search (requires API key)

```
$ OPENALEX_KEY=<your-key> papers work find "transformer attention mechanism self-supervised learning" -n 5
```

Requires a [polite pool API key](https://docs.openalex.org/how-to-use-the-api/rate-limits-and-authentication)
with semantic search credits enabled.

## Filter syntax

Filters follow the [OpenAlex filter syntax](https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists):

| Example | Meaning |
|---------|---------|
| `publication_year:2024` | Published in 2024 |
| `is_oa:true` | Open access only |
| `publication_year:2020\|2021\|2022` | Published 2020, 2021, or 2022 |
| `authorships.author.id:A5028826050` | Works by a specific author |
| `primary_location.source.id:S137773608` | Works in a specific journal |
| `cited_by_count:>100` | More than 100 citations |

Combine filters with commas (AND): `-f "publication_year:2024,is_oa:true"`
