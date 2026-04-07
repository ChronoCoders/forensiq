use chrono::{DateTime, Utc};
use report::Report;
use sqlx::SqlitePool;
use std::{collections::HashMap, path::PathBuf, sync::Arc};
use store::EvidenceStore;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct Session {
    pub username: String,
    pub role: String,
    pub expires_at: DateTime<Utc>,
}

#[derive(Clone)]
pub struct AppState {
    pub store: EvidenceStore,
    /// Filesystem path to evidence.duckdb — passed to the pipeline subprocess.
    pub audit: SqlitePool,
    /// Filesystem path to audit.db — passed to the pipeline subprocess.
    pub audit_path: PathBuf,
    /// SQLite pool for pre-computed analysis results written by the Python pipeline.
    pub results: SqlitePool,
    /// Filesystem path to results.db — passed to the pipeline subprocess.
    pub results_path: PathBuf,
    /// In-memory session store: token → Session.
    pub sessions: Arc<Mutex<HashMap<String, Session>>>,
    /// In-memory report store (reports are appended, never deleted).
    pub reports: Arc<Mutex<Vec<Report>>>,
}
