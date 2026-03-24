# Phase 6 — Report Generator

**Component:** `report` (Rust crate)
**Implementing agent:** rust-implementer
**Status:** Implemented

---

## Purpose

Assemble a tamper-evident, structured JSON report from the outputs of the reasoning engine (Phase 5) and contradiction detector (Phase 4). Log a `report_generated` audit event. Provide an integrity-verification function so any downstream consumer can confirm the report has not been altered after generation.

---

## Dependencies

```toml
audit      = { path = "../audit" }
chrono     = { version = "0.4", features = ["serde"] }
hex        = "0.4"
serde      = { version = "1", features = ["derive"] }
serde_json = "1"
sha2       = "0.10"
sqlx       = { version = "0.7", features = ["sqlite", "runtime-tokio"] }
thiserror  = "1"
tokio      = { version = "1", features = ["full"] }
uuid       = { version = "1", features = ["v4", "serde"] }
```

---

## Type Definitions

### `ScoredEvidence`

One evidence item as ranked by the reasoning engine.
`feature_contributions` **must** use `BTreeMap` — deterministic key ordering is required for a stable SHA-256 hash.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredEvidence {
    pub evidence_uuid: String,
    pub filename: String,
    pub document_type: String,
    pub rank: u32,
    pub score: f64,           // posterior reliability in (0, 1)
    pub confidence: f64,      // score confidence in (0, 1)
    pub explanation: String,
    pub feature_contributions: BTreeMap<String, f64>,  // signal → likelihood ratio
}
```

### `Contradiction`

One contradiction as produced by the contradiction detector.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contradiction {
    pub evidence_uuid_a: String,
    pub evidence_uuid_b: String,
    pub label: String,        // "CONTRADICTION" | "UNCERTAIN" | "NEUTRAL"
    pub confidence: f64,
    pub context: String,
    pub explanation: String,
}
```

### `ReportRequest`

Caller-supplied inputs for one case.

```rust
pub struct ReportRequest {
    pub case_id: String,
    pub generated_by: String,
    pub ranked_evidence: Vec<ScoredEvidence>,
    pub contradictions: Vec<Contradiction>,
}
```

### `ReportBody<'a>` (internal)

Borrows all `Report` fields except `report_sha256`. Used only to produce the canonical JSON that is hashed. Never exposed publicly.

```rust
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
```

### `Report`

The final output. `report_sha256` is the SHA-256 hex digest of the canonical JSON of all other fields.

```rust
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
```

### `ReportError`

```rust
#[derive(Debug, Error)]
pub enum ReportError {
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("audit error: {0}")]
    Audit(#[from] audit::AuditError),
}

pub type Result<T> = std::result::Result<T, ReportError>;
```

---

## Function Signatures

### `generate`

```rust
pub async fn generate(req: ReportRequest, audit_pool: &SqlitePool) -> Result<Report>
```

Steps:
1. Assign `report_id = Uuid::new_v4()`, `generated_at = Utc::now()`
2. Construct `ReportBody` borrowing all fields
3. Serialize `ReportBody` to canonical JSON with `serde_json::to_string`
4. Compute `report_sha256 = hex::encode(Sha256::digest(canonical.as_bytes()))`
5. Append `report_generated` event to audit chain via `audit::log_event`; payload is JSON with fields: `report_id`, `case_id`, `generated_by`, `evidence_count`, `contradiction_count`, `report_sha256`
6. Return assembled `Report`

### `verify_integrity`

```rust
pub fn verify_integrity(report: &Report) -> Result<bool>
```

Re-serializes `ReportBody` from the given report, recomputes the hash, and compares it to `report.report_sha256`. Returns `true` if equal, `false` if tampered.

---

## Audit Crate Extension

`init_audit_log` must also create an `event_log` table:

```sql
CREATE TABLE IF NOT EXISTS event_log (
    id          INTEGER PRIMARY KEY,
    event_type  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    occurred_at TEXT NOT NULL
)
```

New function:

```rust
pub async fn log_event(pool: &SqlitePool, event_type: &str, payload: &str) -> Result<()>
```

Inserts one row with `occurred_at = Utc::now().to_rfc3339()`.

---

## Error Handling Strategy

- All errors use `thiserror` and propagate with `?`
- No `.unwrap()` or `.expect()` in production code
- Two failure modes: serialization failure (returns immediately, no audit write attempted) and audit write failure (report data is valid but not logged — caller must treat this as an error and not return the report)

---

## Test Scenarios

1. **Happy path** — `generate()` with two evidence items and one contradiction produces a `Report` with correct `evidence_count`, `contradiction_count`, and a non-empty `report_sha256`
2. **Integrity pass** — `verify_integrity()` returns `true` for an unmodified report
3. **Integrity fail** — mutating any field (e.g. `score`) after generation causes `verify_integrity()` to return `false`
4. **Audit event logged** — after `generate()`, the `event_log` table contains exactly one row with `event_type = 'report_generated'`
5. **Empty case** — zero evidence items and zero contradictions produces a valid report with a stable hash
6. **Round-trip serialization** — a `Report` serialized to JSON and deserialized back has identical `report_id` and `report_sha256`
