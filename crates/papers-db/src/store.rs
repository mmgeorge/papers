use arrow_array::RecordBatchIterator;
use arrow_schema::Schema;
use lancedb::index::Index;
use lancedb::{Connection, Table};
use std::sync::{Arc, Mutex};
use tokio::sync::OnceCell;

use crate::embed::Embedder;
use crate::error::DbError;
use crate::schema::{chunks_schema, exhibits_schema};

pub struct DbStore {
    pub(crate) db: Connection,
    pub(crate) embedder: OnceCell<Arc<Mutex<Embedder>>>,
}

impl DbStore {
    /// Open (or create) the RAG database at the given path.
    /// Creates both tables with correct schemas if they don't exist yet.
    /// The embedding model is loaded lazily on first use.
    pub async fn open(path: &str) -> Result<Self, DbError> {
        let db = lancedb::connect(path).execute().await?;

        // Create tables if they don't exist; migrate schemas if needed
        let chunks = ensure_table(&db, "papers_chunks", chunks_schema()).await?;
        migrate_chunks_table(&chunks).await?;
        let exhibits = ensure_table(&db, "papers_exhibits", exhibits_schema()).await?;
        migrate_exhibits_table(&exhibits).await?;

        let store = Self {
            db,
            embedder: OnceCell::new(),
        };
        Ok(store)
    }

    /// Get or initialize the embedder (lazy loading).
    async fn embedder(&self) -> Result<Arc<Mutex<Embedder>>, DbError> {
        self.embedder
            .get_or_try_init(|| async {
                eprintln!(
                    "    loading {} [{}] (downloads on first run)...",
                    crate::embed::MODEL_NAME,
                    crate::embed::ep_name()
                );
                let t = std::time::Instant::now();
                let embedder = tokio::task::spawn_blocking(Embedder::new)
                    .await
                    .map_err(|e| DbError::Embed(format!("spawn_blocking join error: {e}")))?
                    .map_err(|e| DbError::Embed(e.to_string()))?;
                eprintln!("    embedding model ready ({:.1}s)", t.elapsed().as_secs_f64());
                Ok(Arc::new(Mutex::new(embedder)))
            })
            .await
            .cloned()
    }

    /// Default path: $PAPERS_DB_PATH or {PAPERS_DATA_DIR}/rag or platform data dir.
    pub fn default_path() -> String {
        if let Ok(p) = std::env::var("PAPERS_DB_PATH") {
            return p;
        }
        let base = std::env::var("PAPERS_DATA_DIR")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| dirs::data_dir().map(|d| d.join("papers")))
            .unwrap_or_else(|| std::path::PathBuf::from(".papers"));
        base.join("rag").to_string_lossy().into_owned()
    }

    /// Test/bench-only: open (or create) the RAG database without loading the embedding model.
    /// All embed calls return zero vectors via `Embedder::fake()`.
    #[cfg(any(test, feature = "bench"))]
    pub async fn open_for_test(path: &str) -> Result<Self, DbError> {
        let db = lancedb::connect(path).execute().await?;
        let chunks = ensure_table(&db, "papers_chunks", chunks_schema()).await?;
        migrate_chunks_table(&chunks).await?;
        let exhibits = ensure_table(&db, "papers_exhibits", exhibits_schema()).await?;
        migrate_exhibits_table(&exhibits).await?;
        let embedder = OnceCell::new();
        embedder
            .set(Arc::new(Mutex::new(Embedder::fake())))
            .unwrap();
        Ok(Self { db, embedder })
    }

    pub async fn chunks_table(&self) -> Result<Table, DbError> {
        self.db
            .open_table("papers_chunks")
            .execute()
            .await
            .map_err(Into::into)
    }

    pub async fn exhibits_table(&self) -> Result<Table, DbError> {
        self.db
            .open_table("papers_exhibits")
            .execute()
            .await
            .map_err(Into::into)
    }

    /// Create vector indexes on both tables if they don't exist.
    /// Uses `Index::Auto` which selects IVF-PQ for vector columns.
    /// Logs and continues on failure (e.g. empty tables or < 256 rows).
    pub async fn ensure_indexes(&self) {
        for table_name in &["papers_chunks", "papers_exhibits"] {
            let table = match self.db.open_table(*table_name).execute().await {
                Ok(t) => t,
                Err(_) => continue,
            };
            match table
                .create_index(&["vector"], Index::Auto)
                .execute()
                .await
            {
                Ok(_) => {
                    eprintln!("  vector index created for {table_name}");
                }
                Err(_) => {
                    // Expected for empty tables or tables with few rows
                    eprintln!("  vector index skipped for {table_name}: too few rows");
                }
            }
        }
    }

    /// Eagerly initialize the embedding model so the first search call is fast.
    /// Safe to call multiple times — subsequent calls are no-ops.
    pub async fn warm_up(&self) -> Result<(), DbError> {
        self.embedder().await?;
        Ok(())
    }

    /// Embed a query string asynchronously.
    pub async fn embed_query(&self, query: &str) -> Result<Vec<f32>, DbError> {
        let embedder = self.embedder().await?;
        let query = query.to_string();
        tokio::task::spawn_blocking(move || {
            embedder
                .lock()
                .map_err(|e| DbError::Embed(format!("mutex poisoned: {e}")))?
                .embed_query(&query)
        })
        .await
        .map_err(|e| DbError::Embed(format!("spawn_blocking join error: {e}")))?
    }

    /// Embed document texts asynchronously.
    pub async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, DbError> {
        let embedder = self.embedder().await?;
        tokio::task::spawn_blocking(move || {
            embedder
                .lock()
                .map_err(|e| DbError::Embed(format!("mutex poisoned: {e}")))?
                .embed_documents(&texts)
        })
        .await
        .map_err(|e| DbError::Embed(format!("spawn_blocking join error: {e}")))?
    }
}

/// Open a table if it exists, or create it with the given schema.
async fn ensure_table(
    db: &Connection,
    name: &str,
    schema: Arc<Schema>,
) -> Result<Table, DbError> {
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

const SCHEMA_VERSION_KEY: &str = "papers_schema_version";

/// Current schema version for the chunks table.
const CURRENT_CHUNKS_VERSION: u32 = 1;

/// Versioned schema migrations for the chunks table.
/// Each entry: (version, column_name, default_sql_expression).
/// Entries must be ordered by version. A migration runs only once, when the
/// on-disk version is below the entry's version number.
const CHUNK_MIGRATIONS: &[(u32, &str, &str)] = &[
    (1, "block_type", "'text'"),
];

/// Read the schema version stored in Arrow schema metadata, defaulting to 0.
async fn read_schema_version(table: &Table) -> Result<u32, DbError> {
    let schema = table.schema().await?;
    Ok(schema
        .metadata
        .get(SCHEMA_VERSION_KEY)
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(0))
}

/// Write the schema version into Arrow schema metadata via NativeTable.
async fn write_schema_version(table: &Table, version: u32) -> Result<(), DbError> {
    let native = table
        .as_native()
        .ok_or_else(|| DbError::Scope("table is not a NativeTable".into()))?;
    native
        .replace_schema_metadata(vec![(
            SCHEMA_VERSION_KEY.to_string(),
            version.to_string(),
        )])
        .await?;
    Ok(())
}

/// Apply pending schema migrations to the chunks table. Only runs migrations
/// whose version exceeds the stored schema version, then bumps the version.
async fn migrate_chunks_table(table: &Table) -> Result<(), DbError> {
    use lancedb::table::NewColumnTransform;

    let current = read_schema_version(table).await?;
    if current >= CURRENT_CHUNKS_VERSION {
        return Ok(());
    }

    let mut applied = 0u32;
    for &(ver, col, default_expr) in CHUNK_MIGRATIONS {
        if ver <= current {
            continue;
        }
        match table
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(col.into(), default_expr.into())]),
                None,
            )
            .await
        {
            Ok(_) => {
                eprintln!("  migrated v{ver}: added column '{col}'");
                applied += 1;
            }
            Err(_) => {
                // Column already exists (e.g. table was created with the
                // latest schema but version metadata was missing).
            }
        }
    }

    write_schema_version(table, CURRENT_CHUNKS_VERSION).await?;
    if applied > 0 {
        eprintln!(
            "  schema version: {current} → {CURRENT_CHUNKS_VERSION} ({applied} migration(s))"
        );
    }
    Ok(())
}

/// Current schema version for the exhibits table.
const CURRENT_EXHIBITS_VERSION: u32 = 1;

/// Versioned schema migrations for the exhibits table.
const EXHIBIT_MIGRATIONS: &[(u32, &str, &str)] = &[
    (1, "content", "CAST(NULL AS string)"),
];

/// Apply pending schema migrations to the exhibits table.
async fn migrate_exhibits_table(table: &Table) -> Result<(), DbError> {
    use lancedb::table::NewColumnTransform;

    let current = read_schema_version(table).await?;
    if current >= CURRENT_EXHIBITS_VERSION {
        return Ok(());
    }

    let mut applied = 0u32;
    for &(ver, col, default_expr) in EXHIBIT_MIGRATIONS {
        if ver <= current {
            continue;
        }
        match table
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(col.into(), default_expr.into())]),
                None,
            )
            .await
        {
            Ok(_) => {
                eprintln!("  migrated exhibits v{ver}: added column '{col}'");
                applied += 1;
            }
            Err(_) => {
                // Column already exists
            }
        }
    }

    write_schema_version(table, CURRENT_EXHIBITS_VERSION).await?;
    if applied > 0 {
        eprintln!(
            "  exhibits schema version: {current} → {CURRENT_EXHIBITS_VERSION} ({applied} migration(s))"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        Array, FixedSizeListArray, Float32Array, RecordBatch, StringArray, UInt16Array,
        builder::{ListBuilder, StringBuilder},
    };
    use arrow_schema::{DataType, Field};
    use serial_test::serial;

    /// Build the v0 chunks schema (before block_type was added).
    fn chunks_schema_v0() -> Arc<Schema> {
        let fields: Vec<Field> = chunks_schema()
            .fields()
            .iter()
            .filter(|f| f.name() != "block_type")
            .cloned()
            .map(|f| f.as_ref().clone())
            .collect();
        Arc::new(Schema::new(fields))
    }

    /// Build an empty string list array with 1 row (empty list).
    fn empty_string_list_array() -> arrow_array::ListArray {
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.append(true);
        builder.finish()
    }

    /// Insert a single dummy row into the v0 table (no block_type column).
    async fn insert_v0_row(table: &Table) {
        let schema = chunks_schema_v0();
        let dim = crate::schema::EMBED_DIM;
        let zeros: Vec<f32> = vec![0.0; dim as usize];
        let vector_field = Field::new("item", DataType::Float32, true);
        let vectors = FixedSizeListArray::try_new(
            Arc::new(vector_field),
            dim,
            Arc::new(Float32Array::from(zeros)),
            None,
        )
        .unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["test/ch0/s0/p0"])),  // chunk_id
                Arc::new(StringArray::from(vec!["test"])),             // paper_id
                Arc::new(vectors) as Arc<dyn Array>,                   // vector
                Arc::new(StringArray::from(vec![""])),                 // chapter_title
                Arc::new(UInt16Array::from(vec![0u16])),               // chapter_idx
                Arc::new(StringArray::from(vec![""])),                 // section_title
                Arc::new(UInt16Array::from(vec![0u16])),               // section_idx
                Arc::new(UInt16Array::from(vec![0u16])),               // chunk_idx
                Arc::new(StringArray::from(vec!["paragraph"])),        // depth
                Arc::new(StringArray::from(vec!["hello world"])),      // text
                Arc::new(UInt16Array::from(vec![None::<u16>])),        // page_start
                Arc::new(UInt16Array::from(vec![None::<u16>])),        // page_end
                Arc::new(StringArray::from(vec!["Test Paper"])),       // title
                Arc::new(empty_string_list_array()) as Arc<dyn Array>, // authors
                Arc::new(UInt16Array::from(vec![None::<u16>])),        // year
                Arc::new(StringArray::from(vec![None::<&str>])),       // venue
                Arc::new(empty_string_list_array()) as Arc<dyn Array>, // tags
                Arc::new(empty_string_list_array()) as Arc<dyn Array>, // exhibit_ids
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table
            .add(Box::new(reader))
            .execute()
            .await
            .unwrap();
    }

    #[tokio::test]
    #[serial]
    async fn test_migrate_v0_to_v1_adds_block_type() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let db = lancedb::connect(&db_path).execute().await.unwrap();

        // Create table with v0 schema (no block_type)
        let table = ensure_table(&db, "papers_chunks", chunks_schema_v0())
            .await
            .unwrap();

        // No version metadata yet
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, 0);

        // Insert a row before migration
        insert_v0_row(&table).await;

        // Run migration
        migrate_chunks_table(&table).await.unwrap();

        // Version should now be CURRENT_CHUNKS_VERSION
        // Need to re-open to see updated schema metadata
        let table = db.open_table("papers_chunks").execute().await.unwrap();
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, CURRENT_CHUNKS_VERSION);

        // Table should now have block_type column
        let schema = table.schema().await.unwrap();
        assert!(
            schema.field_with_name("block_type").is_ok(),
            "block_type column should exist after migration"
        );

        // Existing row should have default value 'text'
        use futures::TryStreamExt;
        use lancedb::query::{ExecutableQuery, QueryBase};
        let batches: Vec<RecordBatch> = table
            .query()
            .select(lancedb::query::Select::columns(&["chunk_id", "block_type"]))
            .execute()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        let bt_col = batches[0]
            .column_by_name("block_type")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(bt_col.value(0), "text");
    }

    #[tokio::test]
    #[serial]
    async fn test_migrate_idempotent_second_open_is_noop() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();

        // First open: creates table + migrates
        let db = lancedb::connect(&db_path).execute().await.unwrap();
        let table = ensure_table(&db, "papers_chunks", chunks_schema()).await.unwrap();
        migrate_chunks_table(&table).await.unwrap();

        let table = db.open_table("papers_chunks").execute().await.unwrap();
        let v1 = read_schema_version(&table).await.unwrap();
        assert_eq!(v1, CURRENT_CHUNKS_VERSION);

        // Second open: should skip migration entirely
        drop(table);
        let table = db.open_table("papers_chunks").execute().await.unwrap();
        migrate_chunks_table(&table).await.unwrap();
        let v2 = read_schema_version(&table).await.unwrap();
        assert_eq!(v2, CURRENT_CHUNKS_VERSION);
    }

    #[tokio::test]
    #[serial]
    async fn test_fresh_db_gets_version_stamped() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let db = lancedb::connect(&db_path).execute().await.unwrap();

        // Create with latest schema (already has block_type)
        let table = ensure_table(&db, "papers_chunks", chunks_schema()).await.unwrap();

        // Before migration: no version
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, 0);

        // Migration should stamp version even though column already exists
        migrate_chunks_table(&table).await.unwrap();

        let table = db.open_table("papers_chunks").execute().await.unwrap();
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, CURRENT_CHUNKS_VERSION);
    }

    #[tokio::test]
    #[serial]
    async fn test_read_write_schema_version_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let db = lancedb::connect(&db_path).execute().await.unwrap();
        let table = ensure_table(&db, "papers_chunks", chunks_schema()).await.unwrap();

        assert_eq!(read_schema_version(&table).await.unwrap(), 0);

        write_schema_version(&table, 42).await.unwrap();
        // Re-open to see metadata update
        let table = db.open_table("papers_chunks").execute().await.unwrap();
        assert_eq!(read_schema_version(&table).await.unwrap(), 42);

        // Overwrite with different version
        write_schema_version(&table, 99).await.unwrap();
        let table = db.open_table("papers_chunks").execute().await.unwrap();
        assert_eq!(read_schema_version(&table).await.unwrap(), 99);
    }

    #[tokio::test]
    #[serial]
    async fn test_migrate_skips_when_already_at_current_version() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let db = lancedb::connect(&db_path).execute().await.unwrap();

        // Create v0 table and manually stamp it at current version
        let table = ensure_table(&db, "papers_chunks", chunks_schema_v0()).await.unwrap();
        write_schema_version(&table, CURRENT_CHUNKS_VERSION).await.unwrap();

        // Re-open and migrate — should be a no-op (won't try to add block_type)
        let table = db.open_table("papers_chunks").execute().await.unwrap();
        migrate_chunks_table(&table).await.unwrap();

        // block_type should NOT exist since migration was skipped
        let schema = table.schema().await.unwrap();
        assert!(
            schema.field_with_name("block_type").is_err(),
            "block_type should not exist — migration was skipped due to version"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_open_for_test_stamps_version() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let store = DbStore::open_for_test(&db_path).await.unwrap();
        let table = store.chunks_table().await.unwrap();
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, CURRENT_CHUNKS_VERSION);
    }

    // ── Exhibits migration tests ──────────────────────────────────────────

    /// Build the v0 exhibits schema (before content was added).
    fn exhibits_schema_v0() -> Arc<Schema> {
        let fields: Vec<Field> = exhibits_schema()
            .fields()
            .iter()
            .filter(|f| f.name() != "content")
            .cloned()
            .map(|f| f.as_ref().clone())
            .collect();
        Arc::new(Schema::new(fields))
    }

    #[tokio::test]
    #[serial]
    async fn test_migrate_exhibits_v0_to_v1_adds_content() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let db = lancedb::connect(&db_path).execute().await.unwrap();

        // Create exhibits table with v0 schema (no content)
        let table = ensure_table(&db, "papers_exhibits", exhibits_schema_v0())
            .await
            .unwrap();

        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, 0);

        // Run migration
        migrate_exhibits_table(&table).await.unwrap();

        // Re-open to see updated schema
        let table = db.open_table("papers_exhibits").execute().await.unwrap();
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, CURRENT_EXHIBITS_VERSION);

        let schema = table.schema().await.unwrap();
        assert!(
            schema.field_with_name("content").is_ok(),
            "content column should exist after migration"
        );
    }

    #[tokio::test]
    #[serial]
    async fn test_exhibits_migration_idempotent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let db = lancedb::connect(&db_path).execute().await.unwrap();

        let table = ensure_table(&db, "papers_exhibits", exhibits_schema()).await.unwrap();
        migrate_exhibits_table(&table).await.unwrap();

        let table = db.open_table("papers_exhibits").execute().await.unwrap();
        let v1 = read_schema_version(&table).await.unwrap();
        assert_eq!(v1, CURRENT_EXHIBITS_VERSION);

        // Second migration should be a no-op
        migrate_exhibits_table(&table).await.unwrap();
        let v2 = read_schema_version(&table).await.unwrap();
        assert_eq!(v2, CURRENT_EXHIBITS_VERSION);
    }

    #[tokio::test]
    #[serial]
    async fn test_exhibits_version_stamped_on_fresh_db() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let db = lancedb::connect(&db_path).execute().await.unwrap();

        let table = ensure_table(&db, "papers_exhibits", exhibits_schema()).await.unwrap();
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, 0);

        migrate_exhibits_table(&table).await.unwrap();

        let table = db.open_table("papers_exhibits").execute().await.unwrap();
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, CURRENT_EXHIBITS_VERSION);
    }

    #[tokio::test]
    #[serial]
    async fn test_chunks_version_unchanged_after_exhibits_migration() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let store = DbStore::open_for_test(&db_path).await.unwrap();

        let chunks_table = store.chunks_table().await.unwrap();
        let chunks_v = read_schema_version(&chunks_table).await.unwrap();
        assert_eq!(chunks_v, CURRENT_CHUNKS_VERSION);

        let exhibits_table = store.exhibits_table().await.unwrap();
        let exhibits_v = read_schema_version(&exhibits_table).await.unwrap();
        assert_eq!(exhibits_v, CURRENT_EXHIBITS_VERSION);

        // Versions are independent
        assert_eq!(chunks_v, CURRENT_CHUNKS_VERSION);
    }

    #[tokio::test]
    #[serial]
    async fn test_open_for_test_stamps_exhibits_version() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("test.lance").to_string_lossy().into_owned();
        let store = DbStore::open_for_test(&db_path).await.unwrap();
        let table = store.exhibits_table().await.unwrap();
        let v = read_schema_version(&table).await.unwrap();
        assert_eq!(v, CURRENT_EXHIBITS_VERSION);
    }
}
