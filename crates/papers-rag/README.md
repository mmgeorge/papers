# papers-rag

Local vector RAG (Retrieval-Augmented Generation) index for academic papers.
Ingests PDF extractions produced by DataLab Marker, embeds text chunks with
[Nomic Embed v2 MoE](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe),
stores vectors in [LanceDB](https://lancedb.github.io/lancedb/), and exposes
semantic search over indexed papers.

Used by `papers-cli` (via `papers rag` subcommands) and `papers-mcp`.

---

## Usage

```toml
[dependencies]
papers-rag = { path = "../papers-rag" }
```

### Open the store

```rust
use papers_rag::RagStore;

let store = RagStore::open(&RagStore::default_path()).await?;
```

`RagStore::open` loads the Nomic Embed model (downloads weights on first run)
and creates the LanceDB tables if they don't exist.

### Ingest a paper

```rust
use papers_rag::{IngestParams, ingest_paper};
use std::path::PathBuf;

let params = IngestParams {
    item_key: "YFACFA8C".to_string(),
    paper_id: "10.1145/example".to_string(),
    title: "My Paper".to_string(),
    authors: vec!["Alice".to_string()],
    year: Some(2024),
    venue: Some("SIGGRAPH".to_string()),
    tags: vec!["graphics".to_string()],
    cache_dir: PathBuf::from("/path/to/datalab/cache/YFACFA8C"),
    force: false,
};

let stats = ingest_paper(&store, params).await?;
println!("Indexed {} chunks, {} figures", stats.chunks_added, stats.figures_added);
```

`ingest_paper` automatically checks the embedding cache before running GPU
inference. On cache hit, no embedding model calls are made.

### Build IngestParams from the DataLab cache

```rust
use papers_rag::ingest_params_from_cache;

let params = ingest_params_from_cache("YFACFA8C")?;
```

### Semantic search

```rust
use papers_rag::{SearchParams, query};

let params = SearchParams {
    query: "neural radiance fields".to_string(),
    paper_ids: None,
    chapter_idx: None,
    section_idx: None,
    filter_year_min: None,
    filter_year_max: None,
    filter_venue: None,
    filter_tags: None,
    filter_depth: None,
    limit: 5,
};

let results = query::search(&store, params).await?;
for r in &results {
    println!("{:.2}  {}  {}", r.score, r.chunk.paper_id, &r.chunk.text[..80]);
}
```

### Embedding cache

The `EmbedCache` persists computed embeddings on disk so that LanceDB can be
rebuilt without re-running GPU inference.

```rust
use papers_rag::{default_embed_cache, cache_paper_embeddings};

// Pre-warm the cache for a paper
let n = cache_paper_embeddings(&store, &params, "nomic-embed-text-v2-moe", false).await?;
println!("Cached {n} chunks");

// Inspect what's cached
let cache = default_embed_cache();
let models = cache.list_models("YFACFA8C")?;
println!("Cached models: {:?}", models);

// Load embeddings back
if let Some(manifest) = cache.load_manifest("nomic-embed-text-v2-moe", "YFACFA8C")? {
    let embeddings = cache.load_embeddings("nomic-embed-text-v2-moe", "YFACFA8C", &manifest)?;
    println!("{} chunks × {} dims", embeddings.len(), manifest.dim);
}
```

Cache location: `$PAPERS_EMBED_CACHE_DIR` if set, otherwise
`{platform_cache_dir}/papers/embeddings/{model}/{item_key}/`.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PAPERS_RAG_DB` | `{data_dir}/papers/rag` | LanceDB path |
| `PAPERS_DATA_DIR` | platform data dir | Base for LanceDB |
| `PAPERS_DATALAB_CACHE_DIR` | `{cache_dir}/papers/datalab` | DataLab JSON cache |
| `PAPERS_EMBED_CACHE_DIR` | `{cache_dir}/papers` | Embedding cache base |
