use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};
use std::path::Path;
use thiserror::Error;
use uuid::Uuid;

/// All errors that can occur during evidence ingestion or integrity verification.
#[derive(Debug, Error)]
pub enum IngestionError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("integrity check failed: stored hash does not match recomputed hash")]
    IntegrityMismatch,
}

/// Convenience alias for `Result<T, IngestionError>`.
pub type Result<T> = std::result::Result<T, IngestionError>;

/// An immutable record of a single piece of ingested evidence.
///
/// Created by [`ingest_file`]. The `raw_bytes` and `sha256` fields are coupled:
/// [`verify_integrity`] must pass before any downstream processing.
#[derive(Debug, Clone)]
pub struct Evidence {
    /// Unique identifier assigned at ingest time (UUIDv4).
    pub uuid: Uuid,
    /// Original filename extracted from the path at ingest time.
    pub filename: String,
    /// Hex-encoded SHA-256 digest of `raw_bytes` computed at ingest time.
    pub sha256: String,
    /// Identifier of the system or operator that submitted the evidence.
    pub source: String,
    /// ID of the intake operator who performed the ingest.
    pub operator_id: String,
    /// UTC timestamp recorded immediately after the file was read.
    pub ingested_at: DateTime<Utc>,
    /// Raw file contents. Never modified after construction.
    pub raw_bytes: Vec<u8>,
}

/// Reads a file from `path`, computes its SHA-256 hash, assigns a UUIDv4, and
/// records a UTC timestamp. Returns an [`Evidence`] value ready for storage and
/// audit logging.
///
/// # Errors
/// Returns [`IngestionError::Io`] if the file cannot be read.
pub async fn ingest_file(path: &Path, source: &str, operator_id: &str) -> Result<Evidence> {
    let raw_bytes = tokio::fs::read(path).await?;

    let mut hasher = Sha256::new();
    hasher.update(&raw_bytes);
    let sha256 = hex::encode(hasher.finalize());

    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    Ok(Evidence {
        uuid: Uuid::new_v4(),
        filename,
        sha256,
        source: source.to_owned(),
        operator_id: operator_id.to_owned(),
        ingested_at: Utc::now(),
        raw_bytes,
    })
}

/// Recomputes the SHA-256 hash of `evidence.raw_bytes` and compares it against
/// `evidence.sha256`.
///
/// Returns `Ok(true)` when the hashes match.
///
/// # Errors
/// Returns [`IngestionError::IntegrityMismatch`] if the hashes differ, indicating
/// the evidence bytes were modified after ingest.
pub fn verify_integrity(evidence: &Evidence) -> Result<bool> {
    let mut hasher = Sha256::new();
    hasher.update(&evidence.raw_bytes);
    let recomputed = hex::encode(hasher.finalize());

    if recomputed == evidence.sha256 {
        Ok(true)
    } else {
        Err(IngestionError::IntegrityMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_ingest_file_produces_correct_hash() {
        let mut tmp = NamedTempFile::new().unwrap();
        let content = b"forensiq test data";
        tmp.write_all(content).unwrap();

        let evidence = ingest_file(tmp.path(), "test-source", "op-001")
            .await
            .unwrap();

        // sha256("forensiq test data")
        let mut hasher = Sha256::new();
        hasher.update(content);
        let expected = hex::encode(hasher.finalize());

        assert_eq!(evidence.sha256, expected);
        assert_eq!(evidence.source, "test-source");
        assert_eq!(evidence.operator_id, "op-001");
        assert_eq!(evidence.raw_bytes, content);
    }

    #[tokio::test]
    async fn test_verify_integrity_ok() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"hello world").unwrap();
        let evidence = ingest_file(tmp.path(), "src", "op").await.unwrap();
        assert!(verify_integrity(&evidence).unwrap());
    }

    #[tokio::test]
    async fn test_verify_integrity_tampered() {
        let mut tmp = NamedTempFile::new().unwrap();
        tmp.write_all(b"original").unwrap();
        let mut evidence = ingest_file(tmp.path(), "src", "op").await.unwrap();

        // Tamper with raw bytes
        evidence.raw_bytes = b"tampered".to_vec();

        assert!(matches!(
            verify_integrity(&evidence),
            Err(IngestionError::IntegrityMismatch)
        ));
    }
}
