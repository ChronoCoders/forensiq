use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::SqlitePool;
use thiserror::Error;
use uuid::Uuid;

/// A single evidence item that has been scored and ranked by the reasoning
/// engine.  `feature_contributions` uses `BTreeMap` so that JSON
/// serialisation is deterministic (required for stable hashing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredEvidence {
    /// Accepts both `evidence_uuid` (Python pipeline) and `uuid` (UI shorthand).
    #[serde(alias = "uuid")]
    pub evidence_uuid: String,
    pub filename: String,
    #[serde(default)]
    pub document_type: String,
    pub rank: u32,
    /// Posterior reliability in (0, 1).
    pub score: f64,
    /// Confidence in the score in (0, 1).
    #[serde(default)]
    pub confidence: f64,
    pub explanation: String,
    /// Signal name → likelihood ratio.
    #[serde(default)]
    pub feature_contributions: BTreeMap<String, f64>,
}

/// A contradiction that was detected between two evidence items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub evidence_uuid_a: String,
    pub evidence_uuid_b: String,
    /// `"CONTRADICTION"` | `"UNCERTAIN"` | `"NEUTRAL"`
    pub label: String,
    pub confidence: f64,
    pub context: String,
    pub explanation: String,
}

/// All inputs required to generate a report for one case.
pub struct ReportRequest {
    pub case_id: String,
    pub generated_by: String,
    pub ranked_evidence: Vec<ScoredEvidence>,
    pub contradictions: Vec<Contradiction>,
}

/// Intermediate representation used **only** to compute the report hash.
/// Contains every `Report` field except `report_sha256` so that the hash
/// covers exactly the content it protects.
#[derive(Serialize)]
struct ReportBody<'a> {
    report_id: &'a Uuid,
    case_id: &'a str,
    generated_at: &'a DateTime<Utc>,
    generated_by: &'a str,
    evidence_count: usize,
    contradiction_count: usize,
    ranked_evidence: &'a [ScoredEvidence],
    contradictions: &'a [Contradiction],
}

/// A fully assembled, tamper-evident case report.
///
/// `report_sha256` is the SHA-256 hex digest of the canonical JSON
/// representation of all other fields (see [`verify_integrity`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub report_id: Uuid,
    pub case_id: String,
    pub generated_at: DateTime<Utc>,
    pub generated_by: String,
    pub evidence_count: usize,
    pub contradiction_count: usize,
    pub ranked_evidence: Vec<ScoredEvidence>,
    pub contradictions: Vec<Contradiction>,
    pub report_sha256: String,
}

/// All errors that can occur during report generation or integrity verification.
#[derive(Debug, Error)]
pub enum ReportError {
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("audit error: {0}")]
    Audit(#[from] audit::AuditError),
}

/// Convenience alias for `Result<T, ReportError>`.
pub type Result<T> = std::result::Result<T, ReportError>;

/// Generates a [`Report`] from `req`, appends a `report_generated` event to
/// the audit chain, and returns the report.
///
/// The report's `report_sha256` field is the SHA-256 hex digest of the
/// canonical JSON of every other field.  Use [`verify_integrity`] to confirm
/// the report has not been altered after generation.
///
/// # Errors
/// Returns [`ReportError::Serialization`] if the canonical JSON cannot be
/// produced, or [`ReportError::Audit`] if the audit INSERT fails.
pub async fn generate(req: ReportRequest, audit_pool: &SqlitePool) -> Result<Report> {
    let report_id = Uuid::new_v4();
    let generated_at = Utc::now();
    let evidence_count = req.ranked_evidence.len();
    let contradiction_count = req.contradictions.len();

    let body = ReportBody {
        report_id: &report_id,
        case_id: &req.case_id,
        generated_at: &generated_at,
        generated_by: &req.generated_by,
        evidence_count,
        contradiction_count,
        ranked_evidence: &req.ranked_evidence,
        contradictions: &req.contradictions,
    };

    let canonical = serde_json::to_string(&body)?;
    let report_sha256 = hex::encode(Sha256::digest(canonical.as_bytes()));

    let event_payload = serde_json::json!({
        "report_id":          report_id.to_string(),
        "case_id":            req.case_id,
        "generated_by":       req.generated_by,
        "evidence_count":     evidence_count,
        "contradiction_count": contradiction_count,
        "report_sha256":      report_sha256,
    })
    .to_string();

    audit::log_event(audit_pool, "report_generated", &event_payload).await?;

    Ok(Report {
        report_id,
        case_id: req.case_id,
        generated_at,
        generated_by: req.generated_by,
        evidence_count,
        contradiction_count,
        ranked_evidence: req.ranked_evidence,
        contradictions: req.contradictions,
        report_sha256,
    })
}

/// Re-computes the report's hash and compares it against the stored
/// `report_sha256`.  Returns `true` if the report is intact.
///
/// # Errors
/// Returns [`ReportError::Serialization`] if re-serialisation fails.
pub fn verify_integrity(report: &Report) -> Result<bool> {
    let body = ReportBody {
        report_id: &report.report_id,
        case_id: &report.case_id,
        generated_at: &report.generated_at,
        generated_by: &report.generated_by,
        evidence_count: report.evidence_count,
        contradiction_count: report.contradiction_count,
        ranked_evidence: &report.ranked_evidence,
        contradictions: &report.contradictions,
    };
    let canonical = serde_json::to_string(&body)?;
    let expected = hex::encode(Sha256::digest(canonical.as_bytes()));
    Ok(expected == report.report_sha256)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    fn sample_evidence() -> ScoredEvidence {
        ScoredEvidence {
            evidence_uuid: Uuid::new_v4().to_string(),
            filename: "witness_statement.pdf".into(),
            document_type: "WITNESS_STATEMENT".into(),
            rank: 1,
            score: 0.72,
            confidence: 0.88,
            explanation: "High entity diversity; one uncertain contradiction.".into(),
            feature_contributions: BTreeMap::from([
                ("timeline_coherence(2_events)".into(), 1.5625),
                ("classification_confidence(0.89)".into(), 1.195),
                ("uncertain_contradiction_1".into(), 0.75),
            ]),
        }
    }

    fn sample_contradiction(uuid_a: &str, uuid_b: &str) -> Contradiction {
        Contradiction {
            evidence_uuid_a: uuid_a.into(),
            evidence_uuid_b: uuid_b.into(),
            label: "UNCERTAIN".into(),
            confidence: 0.71,
            context: "timeline".into(),
            explanation: "Statement A places the event at 09:00; statement B at 11:00.".into(),
        }
    }

    #[tokio::test]
    async fn test_generate_produces_report_with_hash() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = audit::init_audit_log(tmp.path()).await.unwrap();

        let ev = sample_evidence();
        let ev_uuid = ev.evidence_uuid.clone();
        let ev2 = ScoredEvidence {
            evidence_uuid: Uuid::new_v4().to_string(),
            filename: "forensic_report.pdf".into(),
            document_type: "FORENSIC_REPORT".into(),
            rank: 2,
            score: 0.85,
            confidence: 0.94,
            explanation: "Forensic report; strong classification confidence.".into(),
            feature_contributions: BTreeMap::from([
                ("classification_confidence(0.95)".into(), 1.225),
            ]),
        };
        let ev2_uuid = ev2.evidence_uuid.clone();
        let contradiction = sample_contradiction(&ev_uuid, &ev2_uuid);

        let req = ReportRequest {
            case_id: "case-001".into(),
            generated_by: "analyst-42".into(),
            ranked_evidence: vec![ev, ev2],
            contradictions: vec![contradiction],
        };

        let report = generate(req, &pool).await.unwrap();

        assert_eq!(report.case_id, "case-001");
        assert_eq!(report.generated_by, "analyst-42");
        assert_eq!(report.evidence_count, 2);
        assert_eq!(report.contradiction_count, 1);
        assert!(!report.report_sha256.is_empty());
    }

    #[tokio::test]
    async fn test_verify_integrity_passes_for_unmodified_report() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = audit::init_audit_log(tmp.path()).await.unwrap();

        let req = ReportRequest {
            case_id: "case-002".into(),
            generated_by: "analyst-7".into(),
            ranked_evidence: vec![sample_evidence()],
            contradictions: vec![],
        };

        let report = generate(req, &pool).await.unwrap();
        assert!(verify_integrity(&report).unwrap());
    }

    #[tokio::test]
    async fn test_verify_integrity_fails_after_tampering() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = audit::init_audit_log(tmp.path()).await.unwrap();

        let req = ReportRequest {
            case_id: "case-003".into(),
            generated_by: "analyst-99".into(),
            ranked_evidence: vec![sample_evidence()],
            contradictions: vec![],
        };

        let mut report = generate(req, &pool).await.unwrap();
        // Tamper with a score after generation.
        report.ranked_evidence[0].score = 0.99;
        assert!(!verify_integrity(&report).unwrap());
    }

    #[tokio::test]
    async fn test_audit_event_logged() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = audit::init_audit_log(tmp.path()).await.unwrap();

        let req = ReportRequest {
            case_id: "case-audit".into(),
            generated_by: "analyst-1".into(),
            ranked_evidence: vec![sample_evidence()],
            contradictions: vec![],
        };

        generate(req, &pool).await.unwrap();

        let (count,): (i64,) =
            sqlx::query_as("SELECT COUNT(*) FROM event_log WHERE event_type = 'report_generated'")
                .fetch_one(&pool)
                .await
                .unwrap();

        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_empty_case_generates_valid_report() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = audit::init_audit_log(tmp.path()).await.unwrap();

        let req = ReportRequest {
            case_id: "case-empty".into(),
            generated_by: "analyst-0".into(),
            ranked_evidence: vec![],
            contradictions: vec![],
        };

        let report = generate(req, &pool).await.unwrap();
        assert_eq!(report.evidence_count, 0);
        assert!(verify_integrity(&report).unwrap());
    }

    #[tokio::test]
    async fn test_report_serialises_to_valid_json() {
        let tmp = NamedTempFile::new().unwrap();
        let pool = audit::init_audit_log(tmp.path()).await.unwrap();

        let req = ReportRequest {
            case_id: "case-json".into(),
            generated_by: "analyst-5".into(),
            ranked_evidence: vec![sample_evidence()],
            contradictions: vec![],
        };

        let report = generate(req, &pool).await.unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let round_tripped: Report = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.report_id, report.report_id);
        assert_eq!(round_tripped.report_sha256, report.report_sha256);
    }
}
