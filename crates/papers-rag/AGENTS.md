# papers-rag — Internal Architecture

Local vector RAG system built on LanceDB + Embedding Gemma 300M (768-dim). Ingests
DataLab Marker JSON, embeds text chunks, caches embeddings on disk, and serves
semantic search over indexed papers.

---

## Module tree

```
src/
  lib.rs          — pub mod declarations, re-exports, default_embed_cache()
  embed.rs        — Embedder wrapper (EmbeddingGemma300M, fake for tests)
  embed_cache.rs  — EmbedCache: persistent f32 binary cache per (model, item_key)
  error.rs        — RagError enum (LanceDb, Embed, Arrow, Cache, Io, Json, …)
  ingest.rs       — parse_paper_blocks, ingest_paper, cache_paper_embeddings
  query.rs        — search, search_figures, get_chunk, get_section, list_papers, …
  schema.rs       — Arrow schemas for chunks + figures tables; EMBED_DIM = 768
  store.rs        — RagStore: LanceDB connection + Arc<Mutex<Embedder>>
  types.rs        — IngestStats, SearchParams, SearchResult, FigureResult, …
  filter.rs       — LanceDB filter string builders
  tests.rs        — integration tests (tokio, open_for_test)
```

---

## Data flow

```
DataLab JSON (cache_dir/<key>/<key>.json)
    │
    ▼ parse_paper_blocks()
Vec<ChunkRecord> + Vec<FigureRecord>
    │
    ▼ check EmbedCache (embed_cache_base() / model / item_key)
    ├── hit  ──→ load Vec<Vec<f32>> from embeddings.bin  (no GPU)
    └── miss ──→ store.embed_documents() via Embedder   (GPU)
                      │
                      ▼ EmbedCache::save()
                 manifest.json + embeddings.bin (written once, overwrite=false)
    │
    ▼ build_chunks_batch() / build_figures_batch()
Arrow RecordBatch
    │
    ▼ LanceDB papers_chunks / papers_figures tables
```

Figure captions are embedded fresh each ingest and are **not** persisted in the
embedding cache — only text chunks are cached.

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

## `RagStore` and embedding

`RagStore::open` loads the GPU model via `Embedder::new` (blocking, runs on
`spawn_blocking`). In tests, `RagStore::open_for_test` uses `Embedder::fake()`
which returns zero vectors without loading any model weights.

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
| `text` | Utf8 | |
| `page_start` | UInt16 | nullable |
| `page_end` | UInt16 | nullable |
| `title` | Utf8 | paper title |
| `authors` | List<Utf8> | |
| `year` | UInt16 | nullable |
| `venue` | Utf8 | nullable |
| `tags` | List<Utf8> | |
| `figure_ids` | List<Utf8> | associated figures |

### `papers_figures`

| Column | Type |
|--------|------|
| `figure_id` | Utf8 (`{paper_id}/fig{n}`) |
| `paper_id` | Utf8 |
| `vector` | FixedSizeList<Float32>[768] |
| `figure_type` | Utf8 ("figure" or "table") |
| `caption` | Utf8 |
| `description` | Utf8 |
| `image_path` | Utf8 nullable |
| `page` | UInt16 nullable |
| `chapter_idx` | UInt16 |
| `section_idx` | UInt16 |
| (paper metadata) | … same as chunks |

---

## Test infrastructure

- `RagStore::open_for_test(path)` — no GPU; uses `Embedder::fake()` (zero vectors)
- `PAPERS_DATALAB_CACHE_DIR` — redirect DataLab cache in tests
- `PAPERS_EMBED_CACHE_DIR` — redirect embed cache in tests
- `tempfile::TempDir` — all test state is isolated

---

## How to add a new query function

1. Add the function to `query.rs` (takes `&RagStore`, returns `Result<T, RagError>`)
2. Add the return type to `types.rs` if new
3. Export from `lib.rs`
4. Add a CLI arm in `papers-cli/src/cli.rs` (new variant in `RagCommand`)
5. Add handler in `papers-cli/src/main.rs` (`handle_rag_command`)
6. Add tests in `tests.rs`

## How to add a new embedding model

1. Add the model name to `papers_core::config::VALID_MODELS`
2. Add the fastembed feature flag and model constant to `embed.rs`
3. Update `Embedder::new` to select the model based on a `model: &str` parameter
4. Update `RagStore::open` to accept a model name
5. Update tests to cover the new model path
