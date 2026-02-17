# papers

Monorepo for `papers` — a CLI and MCP server for searching academic research via OpenAlex.

[OpenAlex](https://openalex.org) is a fully open catalog of the global research system: over
240 million scholarly works, 90 million author profiles, and comprehensive metadata on journals,
institutions, research topics, publishers, and funders.

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
  domain       Research domains (broadest level of topic hierarchy, 4 total)
  field        Academic fields (second level of topic hierarchy, 26 total)
  subfield     Research subfields (third level of topic hierarchy, ~252 total)
  help         Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

See [`papers-cli`](crates/papers-cli/)  for more usage examples.


## Crates

| Crate | Description |
|-------|-------------|
| [`papers`](crates/papers/) | Shared library consumed by both the MCP server and CLI. Some deviations from `papers-openalex` which it wraps. |
| [`papers-cli`](crates/papers-cli/) | CLI with human-readable or JSON output. |
| [`papers-mcp`](crates/papers-mcp/) | MCP server. |
| [`papers-openalex`](crates/papers-openalex/) | Typed client for the OpenAlex REST API.|


## Build and test

```sh
cargo build --workspace
cargo test --workspace
```
