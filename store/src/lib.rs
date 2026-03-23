use chrono::{DateTime, Utc};
use duckdb::{params, Connection};
use ingestion::{verify_integrity, Evidence};
use std::{
    path::Path,
    sync::{Arc, Mutex},
};
use thiserror::Error;
use uuid::Uuid;

/// All errors that can occur during evidence store operations.
#[derive(Debug, Error)]
pub enum StoreError {
    #[error("DuckDB error: {0}")]
    Db(#[from] duckdb::Error),
    #[error("evidence integrity check failed: {0}")]
    Integrity(#[from] ingestion::IngestionError),
    #[error("duplicate evidence UUID: {0}")]
    Duplicate(Uuid),
    #[error("UUID parse error: {0}")]
    UuidParse(uuid::Error),
    #[error("timestamp parse error: {0}")]
    DateParse(chrono::format::ParseError),
    #[error("store lock poisoned")]
    LockPoisoned,
    #[error("async task join error: {0}")]
    Join(#[from] tokio::task::JoinError),
}

/// Convenience alias for `Result<T, StoreError>`.
pub type Result<T> = std::result::Result<T, StoreError>;

/// An evidence record as retrieved from the DuckDB evidence store.
///
/// Structurally identical to `ingestion::Evidence` but owned independently so
/// the `store` crate does not re-export ingestion types.
#[derive(Debug, Clone)]
pub struct StoredEvidence {
    pub uuid: Uuid,
    pub filename: String,
    pub sha256: String,
    pub source: String,
    pub operator_id: String,
    pub ingested_at: DateTime<Utc>,
    /// Raw evidence bytes as stored at ingest time. Never modified.
    pub raw_bytes: Vec<u8>,
}

/// Append-only DuckDB evidence store.
///
/// Wraps a [`duckdb::Connection`] behind an `Arc<Mutex<_>>` so it can be
/// cloned and shared across async tasks via [`tokio::task::spawn_blocking`].
/// No UPDATE or DELETE operations are ever issued against the underlying table.
#[derive(Clone)]
pub struct EvidenceStore {
    inner: Arc<Mutex<Connection>>,
}

const CREATE_TABLE: &str = "
    CREATE TABLE IF NOT EXISTS evidence (
        uuid        TEXT PRIMARY KEY,
        filename    TEXT NOT NULL,
        sha256      TEXT NOT NULL,
        source      TEXT NOT NULL,
        operator_id TEXT NOT NULL,
        ingested_at TEXT NOT NULL,
        raw_bytes   BLOB NOT NULL
    );
";

impl EvidenceStore {
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(CREATE_TABLE)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(CREATE_TABLE)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(conn)),
        })
    }

    /// Verifies evidence integrity then appends to the store.
    /// Returns `Err(StoreError::Duplicate)` if the UUID already exists.
    /// Returns `Err(StoreError::Integrity)` if the SHA-256 does not match.
    pub async fn insert(&self, evidence: &Evidence) -> Result<()> {
        verify_integrity(evidence)?;

        let store = self.clone();
        let uuid = evidence.uuid;
        let uuid_str = evidence.uuid.to_string();
        let filename = evidence.filename.clone();
        let sha256 = evidence.sha256.clone();
        let source = evidence.source.clone();
        let operator_id = evidence.operator_id.clone();
        let ingested_at_str = evidence.ingested_at.to_rfc3339();
        let raw_bytes = evidence.raw_bytes.clone();

        tokio::task::spawn_blocking(move || {
            let conn = store.inner.lock().map_err(|_| StoreError::LockPoisoned)?;

            // Duplicate guard
            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM evidence WHERE uuid = ?",
                params![&uuid_str],
                |row| row.get(0),
            )?;
            if count > 0 {
                return Err(StoreError::Duplicate(uuid));
            }

            conn.execute(
                "INSERT INTO evidence \
                 (uuid, filename, sha256, source, operator_id, ingested_at, raw_bytes) \
                 VALUES (?, ?, ?, ?, ?, ?, ?)",
                params![
                    &uuid_str,
                    &filename,
                    &sha256,
                    &source,
                    &operator_id,
                    &ingested_at_str,
                    raw_bytes.as_slice()
                ],
            )?;

            Ok(())
        })
        .await?
    }

    pub async fn get_by_uuid(&self, uuid: Uuid) -> Result<Option<StoredEvidence>> {
        let store = self.clone();
        let uuid_str = uuid.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = store.inner.lock().map_err(|_| StoreError::LockPoisoned)?;

            let mut stmt = conn.prepare(
                "SELECT uuid, filename, sha256, source, operator_id, ingested_at, raw_bytes \
                 FROM evidence WHERE uuid = ? LIMIT 1",
            )?;

            let mut rows = stmt.query(params![&uuid_str])?;

            match rows.next()? {
                None => Ok(None),
                Some(row) => parse_row(row).map(Some),
            }
        })
        .await?
    }

    pub async fn list_evidence(&self) -> Result<Vec<StoredEvidence>> {
        let store = self.clone();

        tokio::task::spawn_blocking(move || {
            let conn = store.inner.lock().map_err(|_| StoreError::LockPoisoned)?;
            let mut stmt = conn.prepare(
                "SELECT uuid, filename, sha256, source, operator_id, ingested_at, raw_bytes \
                 FROM evidence",
            )?;

            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, Vec<u8>>(6)?,
                ))
            })?;

            let mut results = Vec::new();
            for row in rows {
                let (uuid_s, filename, sha256, source, operator_id, ingested_at_s, raw_bytes) =
                    row?;
                let uuid = Uuid::parse_str(&uuid_s).map_err(StoreError::UuidParse)?;
                let ingested_at = DateTime::parse_from_rfc3339(&ingested_at_s)
                    .map(|dt| dt.with_timezone(&Utc))
                    .map_err(StoreError::DateParse)?;
                results.push(StoredEvidence {
                    uuid,
                    filename,
                    sha256,
                    source,
                    operator_id,
                    ingested_at,
                    raw_bytes,
                });
            }

            Ok(results)
        })
        .await?
    }
}

fn parse_row(row: &duckdb::Row<'_>) -> Result<StoredEvidence> {
    let uuid_s: String = row.get(0)?;
    let filename: String = row.get(1)?;
    let sha256: String = row.get(2)?;
    let source: String = row.get(3)?;
    let operator_id: String = row.get(4)?;
    let ingested_at_s: String = row.get(5)?;
    let raw_bytes: Vec<u8> = row.get(6)?;

    let uuid = Uuid::parse_str(&uuid_s).map_err(StoreError::UuidParse)?;
    let ingested_at = DateTime::parse_from_rfc3339(&ingested_at_s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(StoreError::DateParse)?;

    Ok(StoredEvidence {
        uuid,
        filename,
        sha256,
        source,
        operator_id,
        ingested_at,
        raw_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ingestion::ingest_file;
    use std::io::Write as _;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_insert_and_get_by_uuid() {
        let store = EvidenceStore::open_in_memory().unwrap();

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"phase 2 evidence").unwrap();
        let evidence = ingest_file(tmp.path(), "test-source", "op-1")
            .await
            .unwrap();
        let uuid = evidence.uuid;

        store.insert(&evidence).await.unwrap();

        let retrieved = store.get_by_uuid(uuid).await.unwrap().unwrap();
        assert_eq!(retrieved.sha256, evidence.sha256);
        assert_eq!(retrieved.source, "test-source");
        assert_eq!(retrieved.raw_bytes, b"phase 2 evidence");
    }

    #[tokio::test]
    async fn test_get_nonexistent_returns_none() {
        let store = EvidenceStore::open_in_memory().unwrap();
        let result = store.get_by_uuid(Uuid::new_v4()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_duplicate_uuid_rejected() {
        let store = EvidenceStore::open_in_memory().unwrap();

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"data").unwrap();
        let evidence = ingest_file(tmp.path(), "src", "op").await.unwrap();

        store.insert(&evidence).await.unwrap();
        let err = store.insert(&evidence).await.unwrap_err();
        assert!(matches!(err, StoreError::Duplicate(_)));
    }

    #[tokio::test]
    async fn test_tampered_bytes_rejected() {
        let store = EvidenceStore::open_in_memory().unwrap();

        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"original content").unwrap();
        let mut evidence = ingest_file(tmp.path(), "src", "op").await.unwrap();
        evidence.raw_bytes = b"tampered".to_vec();

        let err = store.insert(&evidence).await.unwrap_err();
        assert!(matches!(err, StoreError::Integrity(_)));
    }

    #[tokio::test]
    async fn test_list_evidence_append_only() {
        let store = EvidenceStore::open_in_memory().unwrap();

        for i in 0u8..4 {
            let mut tmp = NamedTempFile::new().unwrap();
            tmp.write_all(&[i; 16]).unwrap();
            let evidence = ingest_file(tmp.path(), "batch", "op").await.unwrap();
            store.insert(&evidence).await.unwrap();
        }

        let all = store.list_evidence().await.unwrap();
        assert_eq!(all.len(), 4);
    }

    #[tokio::test]
    async fn test_file_backed_store_persists() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("evidence.duckdb");

        {
            let store = EvidenceStore::open(&db_path).unwrap();
            let mut tmp = NamedTempFile::new().unwrap();
            tmp.write_all(b"persistent").unwrap();
            let evidence = ingest_file(tmp.path(), "disk", "op-2").await.unwrap();
            store.insert(&evidence).await.unwrap();
        }

        // Re-open and verify the row survives
        let store2 = EvidenceStore::open(&db_path).unwrap();
        let all = store2.list_evidence().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].raw_bytes, b"persistent");
    }
}
