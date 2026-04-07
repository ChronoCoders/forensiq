use axum::{
    routing::{get, post},
    Router,
};
use sqlx::SqlitePool;
use std::{collections::HashMap, path::Path, path::PathBuf, sync::Arc};
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

mod auth;
mod handlers;
mod state;

/// Create the results DB SQLite pool and ensure all pipeline output tables exist.
async fn init_results_db(path: &Path) -> sqlx::Result<SqlitePool> {
    let url = format!("sqlite://{}?mode=rwc", path.display());
    let pool = SqlitePool::connect(&url).await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS evidence_text (
            uuid          TEXT PRIMARY KEY,
            filename      TEXT NOT NULL,
            extracted_text TEXT NOT NULL,
            analyzed_at   TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS analysis_results (
            uuid             TEXT PRIMARY KEY,
            document_type    TEXT NOT NULL,
            doc_confidence   REAL NOT NULL,
            entities_json    TEXT NOT NULL,
            timeline_json    TEXT NOT NULL,
            entity_count     INTEGER NOT NULL,
            label_diversity  INTEGER NOT NULL,
            timeline_count   INTEGER NOT NULL,
            timeline_parsed  INTEGER NOT NULL,
            analyzed_at      TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS contradiction_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            evidence_uuid_a  TEXT NOT NULL,
            evidence_uuid_b  TEXT NOT NULL,
            label            TEXT NOT NULL,
            confidence       REAL NOT NULL,
            context          TEXT NOT NULL,
            explanation      TEXT NOT NULL,
            detected_at      TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS evidence_scores (
            uuid                       TEXT PRIMARY KEY,
            rank                       INTEGER NOT NULL,
            score                      REAL NOT NULL,
            confidence                 REAL NOT NULL,
            explanation                TEXT NOT NULL,
            feature_contributions_json TEXT NOT NULL,
            scored_at                  TEXT NOT NULL
        )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS graph_nodes (
            id      TEXT PRIMARY KEY,
            label   TEXT NOT NULL,
            type    TEXT NOT NULL,
            color   TEXT NOT NULL,
            x       REAL NOT NULL DEFAULT 0.5,
            y       REAL NOT NULL DEFAULT 0.5,
            sources INTEGER NOT NULL DEFAULT 1,
            conn    INTEGER NOT NULL DEFAULT 0,
            notes   TEXT NOT NULL DEFAULT ''
        )",
    )
    .execute(&pool)
    .await?;

    sqlx::query(
        "CREATE TABLE IF NOT EXISTS graph_edges (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            from_node TEXT NOT NULL,
            to_node   TEXT NOT NULL,
            type      TEXT NOT NULL DEFAULT 'co-occurrence',
            source    TEXT NOT NULL DEFAULT '',
            status    TEXT NOT NULL DEFAULT 'confirmed'
        )",
    )
    .execute(&pool)
    .await?;

    Ok(pool)
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "forensiq_api=info,tower_http=info".into()),
        )
        .init();

    // Ensure data directory exists.
    let data_dir = PathBuf::from("data");
    std::fs::create_dir_all(&data_dir).expect("cannot create data/ directory");

    let audit_path   = data_dir.join("audit.db");
    let results_path = data_dir.join("results.db");

    // Open evidence store (DuckDB, persisted to disk).
    let store = store::EvidenceStore::open(&data_dir.join("evidence.duckdb"))
        .expect("cannot open evidence store");

    // Open audit log (SQLite).
    let audit_pool = audit::init_audit_log(&audit_path)
        .await
        .expect("cannot open audit log");

    // Create cases table and seed initial case if it doesn't exist.
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS cases (
            id         TEXT PRIMARY KEY,
            title      TEXT NOT NULL DEFAULT '',
            charge     TEXT NOT NULL DEFAULT '',
            analyst    TEXT NOT NULL DEFAULT '',
            status     TEXT NOT NULL DEFAULT 'active',
            outcome    TEXT NOT NULL DEFAULT '',
            opened_at  TEXT NOT NULL,
            closed_at  TEXT
        )",
    )
    .execute(&audit_pool)
    .await
    .expect("cannot create cases table");

    sqlx::query(
        "INSERT OR IGNORE INTO cases (id, title, charge, analyst, status, outcome, opened_at)
         VALUES ('2024-0481', 'Case #2024-0481', 'Under Investigation', 'system', 'active', '', '2024-01-01T00:00:00+00:00')",
    )
    .execute(&audit_pool)
    .await
    .expect("cannot seed initial case");

    // Open / create the pipeline results DB (SQLite).
    let results_pool = init_results_db(&results_path)
        .await
        .expect("cannot open results DB");

    let app_state = state::AppState {
        store,
        audit: audit_pool,
        audit_path,
        results: results_pool,
        results_path,
        sessions: Arc::new(Mutex::new(HashMap::new())),
        reports: Arc::new(Mutex::new(Vec::new())),
    };

    // CORS: allow all origins so the static HTML UI can call the API
    // from any local file server. Tighten this for production.
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let api_routes = Router::new()
        // Evidence
        .route("/evidence",                   get(handlers::evidence::list_evidence))
        .route("/evidence/ingest",            post(handlers::evidence::ingest_evidence))
        .route("/evidence/:uuid",             get(handlers::evidence::get_evidence))
        .route("/evidence/:uuid/raw",         get(handlers::evidence::raw_evidence))
        .route("/evidence/:uuid/verify",      post(handlers::evidence::verify_evidence))
        .route("/evidence/:uuid/reanalyze",   post(handlers::evidence::reanalyze_evidence))
        // Pipeline results (written by Python after each ingest)
        .route("/analysis",                   get(handlers::analysis::list_analysis))
        .route("/analysis/:uuid",             get(handlers::analysis::get_analysis))
        .route("/contradictions",             get(handlers::analysis::list_contradictions))
        .route("/contradictions/rerun",       post(handlers::analysis::rerun_contradictions))
        .route("/scores",                     get(handlers::analysis::list_scores))
        // Relationship graph
        .route("/graph",                      get(handlers::analysis::get_graph))
        // Audit log
        .route("/audit",                      get(handlers::audit::get_audit_log))
        // Reports
        .route("/reports",                    get(handlers::reports::list_reports))
        .route("/reports/generate",           post(handlers::reports::generate_report))
        .route("/reports/:id/verify",         post(handlers::reports::verify_report))
        // Cases
        .route("/cases",                      get(handlers::cases::list_cases)
                                             .post(handlers::cases::create_case))
        .route("/cases/:id/signoff",          post(handlers::cases::signoff));

    let app = Router::new()
        .route("/auth/login",  post(auth::login))
        .route("/auth/logout", post(auth::logout))
        .nest("/api", api_routes)
        .layer(cors)
        .with_state(app_state);

    let addr = "0.0.0.0:3000";
    info!("Forensiq API  →  http://{addr}");
    info!("Run from the repo root so 'python -m pipeline.runner' is resolvable.");
    info!("Credentials  →  d.kowalski / forensiq123  |  j.park / forensiq123");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
