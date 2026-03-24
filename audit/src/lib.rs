use chrono::{DateTime, Utc};
use sqlx::SqlitePool;
use std::path::Path;
use thiserror::Error;
use uuid::Uuid;

/// All errors that can occur during audit log initialisation or writing.
#[derive(Debug, Error)]
pub enum AuditError {
    #[error("database error: {0}")]
    Db(#[from] sqlx::Error),
    #[error("migration error: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
}

/// Convenience alias for `Result<T, AuditError>`.
pub type Result<T> = std::result::Result<T, AuditError>;

/// Borrowed view of evidence fields required for an audit entry.
///
/// Mirrors the fields of `ingestion::Evidence` without creating a circular
/// dependency between the `audit` and `ingestion` crates.
pub struct EvidenceRecord<'a> {
    pub uuid: &'a Uuid,
    pub filename: &'a str,
    pub sha256: &'a str,
    pub source: &'a str,
    pub operator_id: &'a str,
    pub ingested_at: &'a DateTime<Utc>,
}

/// Opens (or creates) the SQLite audit database at `path` and ensures the
/// `audit_log` and `event_log` tables exist.
///
/// Both tables are append-only by convention — no UPDATE or DELETE statements
/// are ever issued against them.
///
/// # Errors
/// Returns [`AuditError::Db`] if the connection or table creation fails.
pub async fn init_audit_log(path: &Path) -> Result<SqlitePool> {
    let url = format!("sqlite://{}?mode=rwc", path.display());
    let pool = SqlitePool::connect(&url).await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY,
            uuid        TEXT NOT NULL,
            filename    TEXT NOT NULL,
            sha256      TEXT NOT NULL,
            source      TEXT NOT NULL,
            operator_id TEXT NOT NULL,
            ingested_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS event_log (
            id          INTEGER PRIMARY KEY,
            event_type  TEXT NOT NULL,
            payload     TEXT NOT NULL,
            occurred_at TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;

    Ok(pool)
}

/// Appends one row to the event log for any named system event.
///
/// `event_type` is a short identifier (e.g. `"report_generated"`).
/// `payload` is a JSON string carrying event-specific fields.
///
/// # Errors
/// Returns [`AuditError::Db`] if the INSERT fails.
pub async fn log_event(pool: &SqlitePool, event_type: &str, payload: &str) -> Result<()> {
    let occurred_at = Utc::now().to_rfc3339();
    sqlx::query(
        "INSERT INTO event_log (event_type, payload, occurred_at) VALUES (?, ?, ?)",
    )
    .bind(event_type)
    .bind(payload)
    .bind(&occurred_at)
    .execute(pool)
    .await?;
    Ok(())
}

/// Appends one row to the audit log for an evidence ingest event.
///
/// This is the only write path for the audit table. No update or delete
/// operations exist anywhere in the codebase.
///
/// # Errors
/// Returns [`AuditError::Db`] if the INSERT fails.
pub async fn log_ingest(pool: &SqlitePool, record: &EvidenceRecord<'_>) -> Result<()> {
    let uuid_str = record.uuid.to_string();
    let ingested_at_str = record.ingested_at.to_rfc3339();

    sqlx::query(
        "INSERT INTO audit_log (uuid, filename, sha256, source, operator_id, ingested_at)
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(&uuid_str)
    .bind(record.filename)
    .bind(record.sha256)
    .bind(record.source)
    .bind(record.operator_id)
    .bind(&ingested_at_str)
    .execute(pool)
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::NamedTempFile;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_init_and_log() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = init_audit_log(tmp.path()).await.unwrap();

        let id = Uuid::new_v4();
        let now = Utc::now();
        let record = EvidenceRecord {
            uuid: &id,
            filename: "evidence.bin",
            sha256: "deadbeef",
            source: "test",
            operator_id: "op-42",
            ingested_at: &now,
        };

        log_ingest(&pool, &record).await.unwrap();

        let row: (String,) = sqlx::query_as("SELECT sha256 FROM audit_log WHERE uuid = ?")
            .bind(id.to_string())
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(row.0, "deadbeef");
    }

    #[tokio::test]
    async fn test_append_only_multiple_rows() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = init_audit_log(tmp.path()).await.unwrap();

        for i in 0..3u32 {
            let id = Uuid::new_v4();
            let now = Utc::now();
            let filename = format!("file-{i}.bin");
            let sha = format!("hash-{i}");
            let record = EvidenceRecord {
                uuid: &id,
                filename: &filename,
                sha256: &sha,
                source: "batch",
                operator_id: "op",
                ingested_at: &now,
            };
            log_ingest(&pool, &record).await.unwrap();
        }

        let (count,): (i64,) = sqlx::query_as("SELECT COUNT(*) FROM audit_log")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(count, 3);
    }
}
