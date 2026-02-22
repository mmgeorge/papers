use arrow_array::RecordBatchIterator;
use arrow_schema::Schema;
use lancedb::{Connection, Table};
use std::sync::{Arc, Mutex};

use crate::embed::Embedder;
use crate::error::RagError;
use crate::schema::{chunks_schema, figures_schema};

pub struct RagStore {
    pub(crate) db: Connection,
    pub(crate) embedder: Arc<Mutex<Embedder>>,
}

impl RagStore {
    /// Open (or create) the RAG database at the given path.
    /// Creates both tables with correct schemas if they don't exist yet.
    pub async fn open(path: &str) -> Result<Self, RagError> {
        let db = lancedb::connect(path).execute().await?;

        // Create tables if they don't exist
        ensure_table(&db, "papers_chunks", chunks_schema()).await?;
        ensure_table(&db, "papers_figures", figures_schema()).await?;

        eprintln!("  loading embedding model (downloads on first run)...");
        let embedder = tokio::task::spawn_blocking(Embedder::new)
            .await
            .map_err(|e| RagError::Embed(format!("spawn_blocking join error: {e}")))?
            .map_err(|e| RagError::Embed(e.to_string()))?;
        eprintln!("  embedding model ready");

        Ok(Self {
            db,
            embedder: Arc::new(Mutex::new(embedder)),
        })
    }

    /// Default path: $PAPERS_RAG_DB or {PAPERS_DATA_DIR}/rag or platform data dir.
    pub fn default_path() -> String {
        if let Ok(p) = std::env::var("PAPERS_RAG_DB") {
            return p;
        }
        let base = std::env::var("PAPERS_DATA_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| dirs::data_dir().map(|d| d.join("papers")))
            .unwrap_or_else(|| std::path::PathBuf::from(".papers"));
        base.join("rag").to_string_lossy().into_owned()
    }

    /// Test-only: open (or create) the RAG database without loading the embedding model.
    /// All embed calls return zero vectors via `Embedder::fake()`.
    #[cfg(test)]
    pub(crate) async fn open_for_test(path: &str) -> Result<Self, RagError> {
        let db = lancedb::connect(path).execute().await?;
        ensure_table(&db, "papers_chunks", chunks_schema()).await?;
        ensure_table(&db, "papers_figures", figures_schema()).await?;
        Ok(Self {
            db,
            embedder: Arc::new(Mutex::new(Embedder::fake())),
        })
    }

    pub async fn chunks_table(&self) -> Result<Table, RagError> {
        self.db
            .open_table("papers_chunks")
            .execute()
            .await
            .map_err(Into::into)
    }

    pub async fn figures_table(&self) -> Result<Table, RagError> {
        self.db
            .open_table("papers_figures")
            .execute()
            .await
            .map_err(Into::into)
    }

    /// Embed a query string asynchronously.
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>, RagError> {
        let embedder = self.embedder.clone();
        let query = query.to_string();
        tokio::task::spawn_blocking(move || {
            embedder
                .lock()
                .map_err(|e| RagError::Embed(format!("mutex poisoned: {e}")))?
                .embed_query(&query)
        })
        .await
        .map_err(|e| RagError::Embed(format!("spawn_blocking join error: {e}")))?
    }

    /// Embed document texts asynchronously.
    pub async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, RagError> {
        let embedder = self.embedder.clone();
        tokio::task::spawn_blocking(move || {
            embedder
                .lock()
                .map_err(|e| RagError::Embed(format!("mutex poisoned: {e}")))?
                .embed_documents(&texts)
        })
        .await
        .map_err(|e| RagError::Embed(format!("spawn_blocking join error: {e}")))?
    }
}

/// Open a table if it exists, or create it with the given schema.
async fn ensure_table(
    db: &Connection,
    name: &str,
    schema: Arc<Schema>,
) -> Result<Table, RagError> {
    match db.open_table(name).execute().await {
        Ok(table) => Ok(table),
        Err(_) => {
            // Create empty table with schema
            let reader = RecordBatchIterator::new(
                std::iter::empty::<Result<arrow_array::RecordBatch, arrow_schema::ArrowError>>(),
                schema,
            );
            db.create_table(name, Box::new(reader))
                .execute()
                .await
                .map_err(Into::into)
        }
    }
}
