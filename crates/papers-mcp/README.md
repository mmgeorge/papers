# papers-mcp

An [MCP](https://modelcontextprotocol.io) server that exposes the [OpenAlex](https://openalex.org)
academic research database to AI assistants. 240M+ scholarly works, authors, journals, institutions,
topics, publishers, and funders — all queryable from Claude.

## Tools

| Tool | Description |
|------|-------------|
| `work_list` | Search and filter scholarly works |
| `work_get` | Get a single work by ID, DOI, PMID, or PMCID |
| `work_autocomplete` | Type-ahead search for works by title |
| `work_find` | AI semantic search (requires API key) |
| `author_list` | Search and filter researcher profiles |
| `author_get` | Get a single author |
| `author_autocomplete` | Type-ahead search for authors |
| `source_list` | Search journals, repositories, conferences |
| `source_get` | Get a single source |
| `source_autocomplete` | Type-ahead search for sources |
| `institution_list` | Search universities, hospitals, companies |
| `institution_get` | Get a single institution |
| `institution_autocomplete` | Type-ahead search for institutions |
| `topic_list` | Search the topic hierarchy |
| `topic_get` | Get a single topic |
| `publisher_list` | Search publishing organizations |
| `publisher_get` | Get a single publisher |
| `publisher_autocomplete` | Type-ahead search for publishers |
| `funder_list` | Search grant-making organizations |
| `funder_get` | Get a single funder |
| `funder_autocomplete` | Type-ahead search for funders |

## Setup

### Build

```sh
cargo build --release --bin papers-mcp
```

The binary is at `target/release/papers-mcp` (or `papers-mcp.exe` on Windows).

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "papers": {
      "command": "/absolute/path/to/papers-mcp"
    }
  }
}
```

For semantic search (`work_find`), add your OpenAlex API key:

```json
{
  "mcpServers": {
    "papers": {
      "command": "/absolute/path/to/papers-mcp",
      "env": {
        "OPENALEX_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Claude Code

Add to your project's `.mcp.json` or run:

```sh
claude mcp add papers /absolute/path/to/papers-mcp
```

Or configure in `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "papers": {
      "command": "/absolute/path/to/papers-mcp",
      "env": {
        "OPENALEX_KEY": "your-api-key-here"
      }
    }
  }
}
```

## Usage

Once connected, ask Claude naturally:

- "Find recent open-access papers on transformer attention mechanisms"
- "Who is Yoshua Bengio and what are his most cited works?"
- "What journals publish the most machine learning research?"
- "Show me papers by authors at MIT published in 2023"
- "Get the full details for DOI 10.7717/peerj.4375"

## List response format

List tools return a slim subset of fields to keep responses concise — large arrays like
`referenced_works` and `counts_by_year` are omitted. Use the corresponding `_get` tool
to retrieve the full entity record.

See [`../papers/CHANGES.md`](../papers/CHANGES.md) for the complete list of differences
from the raw OpenAlex API response.

## API key

An API key is optional for most tools but enables higher rate limits and is required for
`work_find` (semantic search). Get one at [openalex.org](https://openalex.org).
