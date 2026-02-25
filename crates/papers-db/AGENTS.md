# papers-db — Internal Architecture

Local vector database built on LanceDB + Embedding Gemma 300M (768-dim). Ingests
DataLab Marker JSON, embeds text chunks, caches embeddings on disk, and serves
semantic search over indexed papers.

---

## Module tree

```
src/
  lib.rs          — pub mod declarations, re-exports, default_embed_cache()
  config.rs       — chunking constants (MIN/TARGET/MAX tokens, overlap, patterns)
  embed.rs        — Embedder wrapper (EmbeddingGemma300M, fake for tests)
  embed_cache.rs  — EmbedCache: persistent f32 binary cache per (model, item_key)
  error.rs        — DbError enum (LanceDb, Embed, Arrow, Cache, Io, Json, …)
  ingest.rs       — parse_paper_blocks, ingest_paper, cache_paper_embeddings
  query.rs        — search, search_exhibits, get_chunk, get_section, list_papers, …
  schema.rs       — Arrow schemas for chunks + exhibits tables; EMBED_DIM = 768
  store.rs        — DbStore: LanceDB connection + Arc<Mutex<Embedder>>
  types.rs        — IngestStats, SearchParams, SearchResult, ExhibitResult, …
  filter.rs       — LanceDB filter string builders
  tests.rs        — integration tests (tokio, open_for_test)
```

---

## Data flow

```
DataLab JSON (cache_dir/<key>/<key>.json)
    │
    ▼ parse_paper_blocks()
Vec<ChunkRecord> + Vec<ExhibitRecord>
    │  ├── ChunkBuffer accumulates Text/Equation/ListGroup blocks
    │  ├── Section boundaries (h2/h3/h4) flush buffer
    │  ├── Algorithm detection on h5/h6 headers → ExhibitRecord
    │  └── Cross-linking: regex matches exhibit refs in chunk text
    │
    ▼ check EmbedCache (embed_cache_base() / model / item_key)
    ├── hit  ──→ load Vec<Vec<f32>> from embeddings.bin  (no GPU)
    └── miss ──→ store.embed_documents() via Embedder   (GPU)
                      │
                      ▼ EmbedCache::save()
                 manifest.json + embeddings.bin (written once, overwrite=false)
    │
    ▼ build_chunks_batch() / build_exhibits_batch()
Arrow RecordBatch
    │
    ▼ LanceDB papers_chunks / papers_exhibits tables
```

Exhibit captions (including algorithm content) are embedded fresh each ingest
and are **not** persisted in the embedding cache — only text chunks are cached.

---

## Chunking strategy

Chunks are produced by buffer-based accumulation of whole blocks, never
splitting a single paragraph/equation/list mid-text.

### Constants (config.rs)

| Constant | Default | Purpose |
|----------|---------|---------|
| `MIN_CHUNK_TOKENS` | 200 | Fragments below this at section boundaries get merged into previous chunk |
| `TARGET_CHUNK_TOKENS` | 400 | Buffer flushes when next block would exceed this |
| `MAX_CHUNK_TOKENS` | 600 | Smart merge blocked if combined chunk would exceed this |
| `OVERLAP_SENTENCES` | 2 | Trailing sentences carried on token-limit flush |
| `TOKEN_ESTIMATE_MULTIPLIER` | 1.3 | Words × multiplier = estimated tokens |

### Token estimation

`estimate_tokens(text)` = `ceil(word_count × 1.3)`. A rough heuristic; accurate
enough for English academic text with embedding-gemma-300m.

### Accumulation rules

1. **Text/Equation/ListGroup** → push whole block into buffer
2. Before pushing, check `would_overflow`: if next block would push past TARGET:
   - Buffer non-empty → flush with overlap, start fresh buffer
   - Buffer empty → push anyway (oversized single block, never split)
3. **Section boundaries** (h2/h3/h4) → smart merge flush (no overlap)
4. **End-of-doc** → smart merge flush

### Smart merge at section boundaries

When flushing at a section boundary or end-of-doc, if the flushed content is
below MIN_CHUNK_TOKENS:
- If a previous chunk exists **in the same chapter and section**, and combined
  size ≤ MAX_CHUNK_TOKENS → merge into previous
- Otherwise → emit as standalone chunk

This prevents orphan fragments that embed poorly, while never merging across
chapter or section boundaries.

### Overlap

On token-limit flush (not section boundaries): the last 2 sentences of the
flushed chunk are carried into the new buffer. Sentence boundaries are detected
by `. ` followed by an uppercase letter.

### References section skip

When an h2 title matches "References" or "Bibliography" (case-insensitive
substring), all subsequent text blocks are skipped.

### Embedding text prefix

For embedding (not stored in the `text` field), each chunk gets a prefix:
`"Paper Title — Chapter Title — Section Title\n\n"` prepended to improve
retrieval relevance.

---

## Algorithm detection

Algorithms are detected from h5/h6 SectionHeader blocks and stored as
ExhibitRecords with `exhibit_type = "algorithm"`.

### Detection pattern

Regex: `(?i)^(?:Algorithm|Procedure|Pseudocode|Listing|Code)\s+\d+`

Requires a number after the keyword to avoid false positives on generic headers
like "Code Availability" or "Algorithm Overview".

### Accumulation

1. When h5/h6 header matches the pattern:
   - Flush current text buffer
   - Enter algorithm accumulation mode
2. Following Text blocks are accumulated into algorithm body
3. Algorithm mode exits on:
   - Any SectionHeader (any level)
   - Figure/Table/Picture block
   - End of document
4. On exit: emit ExhibitRecord with `exhibit_type = "algorithm"`,
   `caption` = header text, `content` = accumulated pseudocode

---

## Cross-linking patterns

After parsing, a post-processing pass scans chunk text for exhibit references
and populates `exhibit_ids` on chunks and `ref_count`/`first_ref_chunk_id` on
exhibits.

### Caption → exhibit_id map

Built from exhibit captions using:
`(?i)(Figure|Fig|Table|Tab|Algorithm|Alg|Procedure|Pseudocode|Listing|Code)\s*\.?\s+(\d+)`

`normalize_exhibit_kind()` maps variants to canonical types:
- `Figure`, `Fig` → `"figure"`
- `Table`, `Tab` → `"table"`
- `Algorithm`, `Alg`, `Procedure`, `Pseudocode`, `Listing`, `Code` → `"algorithm"`

### Text reference regexes

| Type | Pattern | Matches |
|------|---------|---------|
| Figure | `(?i)(?:Figures?\|Figs?)\.?\s+(\d+)` | Fig. 1, Figure 1, Figs. 1 |
| Table | `(?i)(?:Tables?\|Tabs?)\.?\s+(\d+)` | Table 1, Tab. 1 |
| Algorithm | `(?i)(?:Algorithms?\|Algs?\|Procedures?\|...)\.?\s+(\d+)` | Algorithm 1, Alg. 1, etc. |

### Exhibit reference tracking

During the chunk scan, a `HashMap<String, (Option<String>, u16)>` maps
`exhibit_id → (first_ref_chunk_id, ref_count)`. After scanning all chunks,
these values are written back onto the ExhibitRecords.

---

## Embedding binary format

`embeddings.bin` is a flat little-endian `f32` array, no header:

```
bytes [i * dim * 4 .. (i+1) * dim * 4)  →  embedding for chunks[i]
```

Total size: `N * dim * 4` bytes. `N` and `dim` come from `manifest.json`
(`chunks.len()` and top-level `dim`). Rows are in the same order as
`manifest.chunks`.

---

## Config integration

`ingest_paper` and `cache_paper_embeddings` call `default_embed_model()` which
reads `papers_core::config::PapersConfig::load()`. If the config file is missing
or unreadable, the fallback is `"embedding-gemma-300m"`.

`embed_cache_base()` checks `PAPERS_EMBED_CACHE_DIR` first, then falls back to
`{cache_dir}/papers`.

---

## Key types

### `IngestParams`

| Field | Type | Description |
|-------|------|-------------|
| `item_key` | `String` | Zotero key / DataLab directory name |
| `paper_id` | `String` | DOI or item_key (used as LanceDB row key) |
| `cache_dir` | `PathBuf` | Path to `{datalab_cache}/{item_key}/` |
| `force` | `bool` | Bypass embed cache and LanceDB skip-check |

### `ExhibitRecord`

| Field | Type | Description |
|-------|------|-------------|
| `exhibit_id` | `String` | `{paper_id}/fig{n}` |
| `exhibit_type` | `String` | `"figure"`, `"table"`, or `"algorithm"` |
| `caption` | `String` | Caption text or algorithm title |
| `description` | `Option<String>` | Alt text from img tag |
| `content` | `Option<String>` | Markdown table or algorithm pseudocode |
| `first_ref_chunk_id` | `Option<String>` | First text chunk referencing this exhibit |
| `ref_count` | `u16` | Total number of text chunks referencing this exhibit |

### Search result types

`search()` returns slim `SearchResult` with `SearchChunkResult` (no
`authors`/`year`/`venue`/`position`/`referenced_exhibits` — use `get_chunk` for
full details).  `search_exhibits()` returns `ExhibitSearchResult` with a `score`
field.

Neighbor previews (`ChunkSummary`) use sentence-aware truncation (120–300
chars) and look through equation chunks to provide context from the text chunk
beyond the formula.

### `EmbedCache`

Located at `{base_dir}/embeddings/{model}/{item_key}/`:

| File | Contents |
|------|----------|
| `manifest.json` | `EmbedManifest`: model, dim, created_at, Vec<ChunkRecord> |
| `embeddings.bin` | flat little-endian f32 array, N × dim × 4 bytes |

### `EmbedManifest.chunks[i]` ↔ `embeddings.bin` row `i`

The manifest chunk list and binary rows are always in sync (written atomically).
`load_embedding_at(i, dim)` seeks directly to `i * dim * 4`.

---

## `DbStore` and embedding

The embedding model (`EmbeddingGemma300M`) is loaded eagerly at MCP server
startup via `store.warm_up()`, which triggers the `OnceCell` init. This avoids
a 30-60s delay on the first search call. The `warm_up()` call is in
`PapersMcp::open_db_store()` — failure is logged but doesn't block the server.

In tests, `DbStore::open_for_test` uses `Embedder::fake()` which returns zero
vectors without loading any model weights.

`store.embed_documents(texts)` and `store.embed_query(query)` both delegate to
`spawn_blocking` to avoid blocking the async runtime.

---

## LanceDB tables

### `papers_chunks`

| Column | Type | Notes |
|--------|------|-------|
| `chunk_id` | Utf8 | `{paper_id}/ch{c}/s{s}/p{p}` |
| `paper_id` | Utf8 | DOI or item_key |
| `vector` | FixedSizeList<Float32>[768] | embedding |
| `chapter_title` | Utf8 | |
| `chapter_idx` | UInt16 | |
| `section_title` | Utf8 | |
| `section_idx` | UInt16 | |
| `chunk_idx` | UInt16 | within-section index |
| `depth` | Utf8 | always "paragraph" |
| `block_type` | Utf8 | always "text" (merged chunks) |
| `text` | Utf8 | |
| `page_start` | UInt16 | nullable, first page of merged blocks |
| `page_end` | UInt16 | nullable, last page of merged blocks |
| `title` | Utf8 | paper title |
| `authors` | List<Utf8> | |
| `year` | UInt16 | nullable |
| `venue` | Utf8 | nullable |
| `tags` | List<Utf8> | |
| `exhibit_ids` | List<Utf8> | referenced exhibits (figures, tables, algorithms) |

### `papers_exhibits`

| Column | Type | Notes |
|--------|------|-------|
| `exhibit_id` | Utf8 | `{paper_id}/fig{n}` |
| `paper_id` | Utf8 | |
| `vector` | FixedSizeList<Float32>[768] | embedding of caption (+content for algorithms) |
| `exhibit_type` | Utf8 | `"figure"`, `"table"`, or `"algorithm"` |
| `caption` | Utf8 | |
| `description` | Utf8 | nullable |
| `image_path` | Utf8 | nullable |
| `content` | Utf8 | nullable (markdown table or algorithm pseudocode) |
| `page` | UInt16 | nullable |
| `chapter_idx` | UInt16 | |
| `section_idx` | UInt16 | |
| `first_ref_chunk_id` | Utf8 | nullable, chunk_id of first referencing text chunk |
| `ref_count` | UInt16 | total text chunks referencing this exhibit |
| (paper metadata) | … | same as chunks |

---

## Vector indexes

LanceDB does not auto-create vector indexes. Without an index, `nearest_to()`
does a brute-force scan over all rows.

- `DbStore::ensure_indexes()` runs on startup (end of `open()`) and creates
  `Index::Auto` indexes on the `vector` column of both tables.
- After each ingestion (`ingest_paper`), the index is rebuilt for the affected
  table via `create_index`. This replaces any existing index.
- `Index::Auto` selects IVF-PQ for vector columns.
- Tables with fewer than ~256 rows may fail index creation — this is expected
  and logged. At that scale brute-force is already fast.

---

## Database path

Default LanceDB path: `{dirs::data_dir()}/papers/db`

| Platform | Default path |
|----------|-------------|
| Linux | `~/.local/share/papers/db` |
| macOS | `~/Library/Application Support/papers/db` |
| Windows | `C:\Users\<user>\AppData\Roaming\papers\rag` |

Override with `PAPERS_DB_PATH` env var, or set `PAPERS_DATA_DIR` to relocate
the entire `papers/` data root.

---

## Test infrastructure

- `DbStore::open_for_test(path)` — no GPU; uses `Embedder::fake()` (zero vectors)
- `PAPERS_DATALAB_CACHE_DIR` — redirect DataLab cache in tests
- `PAPERS_EMBED_CACHE_DIR` — redirect embed cache in tests
- `tempfile::TempDir` — all test state is isolated

---

## Benchmarks

Run with: `cargo bench -p papers-db --features bench`

Uses `criterion` with `async_tokio` support. Benchmarks measure the LanceDB
query path only — no real embedding model is loaded (uses `open_for_test` with
`Embedder::fake()`). Pre-computed random vectors bypass `embed_query`.

| Benchmark | What it measures |
|-----------|-----------------|
| `search/chunks/{N}` | Vector search + batched neighbor fetching, N rows |
| `search_exhibits/exhibits/{N}` | Vector search on exhibits table, N rows |

Parameterized over table sizes: 100, 500, 1000 rows.

---

## Schema migrations

LanceDB tables on disk may have been created with an older schema. The
`migrate_chunks_table()` function in `store.rs` runs automatically on every
`DbStore::open()` call and applies any pending migrations.

### How it works

Schema version is stored in Arrow schema metadata under the key
`papers_schema_version` using `NativeTable::replace_schema_metadata()`.
On startup the version is read via `table.schema().metadata`, and only
migrations with a version number greater than the stored version are applied.
After all pending migrations run, the version is bumped.

`CHUNK_MIGRATIONS` and `EXHIBIT_MIGRATIONS` are `&[(u32, &str, &str)]` arrays of
`(version, column_name, default_sql_expression)`. Each migration adds a column
via `table.add_columns()`. If the column already exists (e.g. the table was
created with the latest schema but the version metadata was missing), the
`add_columns` call fails silently.

**Important:** When adding a nullable column, always use an explicitly typed
default expression like `CAST(NULL AS string)`, never bare `NULL`. LanceDB's
`add_columns` infers Arrow `Null` type from an untyped `NULL`, which is not
the same as a nullable `Utf8` column and will cause runtime panics when
reading rows.

Each table tracks its own version independently via Arrow schema metadata:
- `CURRENT_CHUNKS_VERSION` must equal the highest version in `CHUNK_MIGRATIONS`
- `CURRENT_EXHIBITS_VERSION` must equal the highest version in `EXHIBIT_MIGRATIONS`

`migrate_chunks_table()` and `migrate_exhibits_table()` run automatically on
every `DbStore::open()` call.

### How to add a new column

1. Bump `CURRENT_CHUNKS_VERSION` or `CURRENT_EXHIBITS_VERSION` in `store.rs`
2. Add a `(new_version, column_name, default_expr)` entry to `CHUNK_MIGRATIONS`
   or `EXHIBIT_MIGRATIONS` — use `CAST(NULL AS string)` for nullable strings,
   never bare `NULL`
3. Add the column to `schema.rs` (`chunks_schema()` or `exhibits_schema()`)
4. Add the column to the ingest path (`ChunkRecord`, `build_chunks_batch()`, etc.)
5. Update `ChunkData` / `chunk_from_row()` in `query.rs` to read the new column
6. Update `AGENTS.md` table schemas
7. Existing databases will auto-migrate on next open — no manual steps needed

---

## How to add a new query function

1. Add the function to `query.rs` (takes `&DbStore`, returns `Result<T, DbError>`)
2. Add the return type to `types.rs` if new
3. Export from `lib.rs`
4. Add a CLI arm in `papers-cli/src/cli.rs` (new variant in `DbCommand`)
5. Add handler in `papers-cli/src/main.rs` (`handle_db_command`)
6. Add tests in `tests.rs`

## How to add a new embedding model

1. Add the model name to `papers_core::config::VALID_MODELS`
2. Add the fastembed feature flag and model constant to `embed.rs`
3. Update `Embedder::new` to select the model based on a `model: &str` parameter
4. Update `DbStore::open` to accept a model name
5. Update tests to cover the new model path
