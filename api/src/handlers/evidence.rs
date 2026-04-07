use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::process::Stdio;
use uuid::Uuid;

use crate::{auth::AuthUser, state::AppState};

type Rejection = (StatusCode, Json<serde_json::Value>);

fn err(status: StatusCode, msg: impl std::fmt::Display) -> Rejection {
    (status, Json(serde_json::json!({ "error": msg.to_string() })))
}

// ── DTO ───────────────────────────────────────────────────────────────────────

/// Evidence representation sent over the wire — omits raw bytes.
#[derive(Serialize)]
pub struct EvidenceDto {
    pub uuid: String,
    pub filename: String,
    pub sha256: String,
    pub source: String,
    pub operator_id: String,
    pub ingested_at: String,
    pub size_bytes: usize,
}

impl From<&store::StoredEvidence> for EvidenceDto {
    fn from(e: &store::StoredEvidence) -> Self {
        EvidenceDto {
            uuid: e.uuid.to_string(),
            filename: e.filename.clone(),
            sha256: e.sha256.clone(),
            source: e.source.clone(),
            operator_id: e.operator_id.clone(),
            ingested_at: e.ingested_at.to_rfc3339(),
            size_bytes: e.raw_bytes.len(),
        }
    }
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `GET /api/evidence` — list all evidence items.
pub async fn list_evidence(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<EvidenceDto>>, Rejection> {
    let all = state
        .store
        .list_evidence()
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(all.iter().map(EvidenceDto::from).collect()))
}

/// `GET /api/evidence/:uuid` — fetch a single evidence item by UUID.
pub async fn get_evidence(
    _user: AuthUser,
    State(state): State<AppState>,
    Path(uuid_str): Path<String>,
) -> Result<Json<EvidenceDto>, Rejection> {
    let uuid =
        Uuid::parse_str(&uuid_str).map_err(|_| err(StatusCode::BAD_REQUEST, "invalid UUID"))?;

    match state.store.get_by_uuid(uuid).await {
        Ok(Some(e)) => Ok(Json(EvidenceDto::from(&e))),
        Ok(None) => Err(err(StatusCode::NOT_FOUND, "evidence not found")),
        Err(e) => Err(err(StatusCode::INTERNAL_SERVER_ERROR, e)),
    }
}

/// `POST /api/evidence/ingest` — ingest a new file via multipart upload.
///
/// Expected fields:
/// - `file`   — the binary file
/// - `source` — originating system / chain-of-custody note
pub async fn ingest_evidence(
    user: AuthUser,
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<EvidenceDto>, Rejection> {
    let mut file_data: Option<(Vec<u8>, String)> = None;
    let mut source = String::from("unknown");

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| err(StatusCode::BAD_REQUEST, e))?
    {
        match field.name().unwrap_or("") {
            "file" => {
                let filename = field.file_name().unwrap_or("upload").to_string();
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
                file_data = Some((bytes.to_vec(), filename));
            }
            "source" => {
                source = field
                    .text()
                    .await
                    .map_err(|e| err(StatusCode::BAD_REQUEST, e))?;
            }
            _ => {}
        }
    }

    let (bytes, filename) = file_data
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "missing 'file' field"))?;

    // Write to a named temp file so ingestion::ingest_file can read it.
    let tmp_dir =
        tempfile::tempdir().map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    let tmp_path = tmp_dir.path().join(&filename);
    tokio::fs::write(&tmp_path, &bytes)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let evidence = ingestion::ingest_file(&tmp_path, &source, &user.username)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
    // tmp_dir drops here, cleaning up the temp file.

    // Append to audit chain.
    let audit_record = audit::EvidenceRecord {
        uuid: &evidence.uuid,
        filename: &evidence.filename,
        sha256: &evidence.sha256,
        source: &evidence.source,
        operator_id: &evidence.operator_id,
        ingested_at: &evidence.ingested_at,
    };
    audit::log_ingest(&state.audit, &audit_record)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let dto = EvidenceDto::from(&store::StoredEvidence {
        uuid: evidence.uuid,
        filename: evidence.filename.clone(),
        sha256: evidence.sha256.clone(),
        source: evidence.source.clone(),
        operator_id: evidence.operator_id.clone(),
        ingested_at: evidence.ingested_at,
        raw_bytes: evidence.raw_bytes.clone(),
    });

    // Append to evidence store (integrity check happens inside insert).
    state
        .store
        .insert(&evidence)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    // Spawn the Python analysis pipeline in the background.
    // Pipeline runs: NLP analysis → contradiction detection → Bayesian scoring,
    // writing results to results.db. Ingest returns immediately.
    let text = String::from_utf8_lossy(&evidence.raw_bytes).into_owned();
    let pipeline_payload = serde_json::json!({
        "uuid":     evidence.uuid.to_string(),
        "filename": evidence.filename,
        "text":     text,
    })
    .to_string();

    let results_path = state.results_path.clone();
    let audit_path   = state.audit_path.clone();

    tokio::spawn(async move {
        spawn_pipeline(pipeline_payload, &results_path, &audit_path).await;
    });

    Ok(Json(dto))
}

/// `POST /api/evidence/:uuid/verify` — re-verify stored SHA-256 integrity.
pub async fn verify_evidence(
    _user: AuthUser,
    State(state): State<AppState>,
    Path(uuid_str): Path<String>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let uuid =
        Uuid::parse_str(&uuid_str).map_err(|_| err(StatusCode::BAD_REQUEST, "invalid UUID"))?;

    let e = state
        .store
        .get_by_uuid(uuid)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "evidence not found"))?;

    let mut hasher = Sha256::new();
    hasher.update(&e.raw_bytes);
    let recomputed = hex::encode(hasher.finalize());
    let valid = recomputed == e.sha256;

    Ok(Json(serde_json::json!({
        "uuid": e.uuid.to_string(),
        "sha256_stored":   e.sha256,
        "sha256_computed": recomputed,
        "valid": valid,
    })))
}

/// `GET /api/evidence/:uuid/raw` — return raw file bytes with appropriate Content-Type.
pub async fn raw_evidence(
    _user: AuthUser,
    State(state): State<AppState>,
    Path(uuid_str): Path<String>,
) -> Result<impl axum::response::IntoResponse, Rejection> {
    use axum::http::header;

    let uuid =
        Uuid::parse_str(&uuid_str).map_err(|_| err(StatusCode::BAD_REQUEST, "invalid UUID"))?;

    let e = state
        .store
        .get_by_uuid(uuid)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "evidence not found"))?;

    let content_type = guess_content_type(&e.filename).to_string();
    let disposition  = format!("inline; filename=\"{}\"", e.filename);

    Ok((
        [
            (header::CONTENT_TYPE,        content_type),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        e.raw_bytes,
    ))
}

fn guess_content_type(filename: &str) -> &'static str {
    match filename.rsplit('.').next().unwrap_or("").to_lowercase().as_str() {
        "pdf"         => "application/pdf",
        "txt"         => "text/plain; charset=utf-8",
        "csv"         => "text/csv",
        "jpg" | "jpeg"=> "image/jpeg",
        "png"         => "image/png",
        "gif"         => "image/gif",
        "mp4"         => "video/mp4",
        "mov"         => "video/quicktime",
        "zip"         => "application/zip",
        _             => "application/octet-stream",
    }
}

/// `POST /api/evidence/:uuid/reanalyze` — re-run the analysis pipeline on stored evidence.
pub async fn reanalyze_evidence(
    _user: AuthUser,
    State(state): State<AppState>,
    Path(uuid_str): Path<String>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let uuid =
        Uuid::parse_str(&uuid_str).map_err(|_| err(StatusCode::BAD_REQUEST, "invalid UUID"))?;

    let e = state
        .store
        .get_by_uuid(uuid)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "evidence not found"))?;

    let text = String::from_utf8_lossy(&e.raw_bytes).into_owned();
    let payload = serde_json::json!({
        "uuid":     e.uuid.to_string(),
        "filename": e.filename,
        "text":     text,
    })
    .to_string();

    let results_path = state.results_path.clone();
    let audit_path   = state.audit_path.clone();

    tokio::spawn(async move {
        spawn_pipeline(payload, &results_path, &audit_path).await;
    });

    Ok(Json(serde_json::json!({
        "uuid":    uuid.to_string(),
        "status":  "queued",
        "message": "Re-analysis pipeline queued",
    })))
}

// ── Pipeline subprocess ───────────────────────────────────────────────────────

/// Spawns `python -m pipeline.rerun` to re-detect contradictions across all
/// stored evidence pairs, re-score, and rebuild the relationship graph.
/// Must be called inside `tokio::spawn`.
pub async fn spawn_pipeline_rerun(
    results_path: &std::path::Path,
    audit_path: &std::path::Path,
) {
    let mut child = match tokio::process::Command::new("python")
        .args([
            "-m",
            "pipeline.rerun",
            "--results",
            results_path.to_str().unwrap_or("data/results.db"),
            "--audit",
            audit_path.to_str().unwrap_or("data/audit.db"),
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to spawn rerun subprocess: {e}");
            return;
        }
    };

    match child.wait().await {
        Ok(s) if s.success() => tracing::info!("rerun completed successfully"),
        Ok(s) => tracing::warn!("rerun exited with status {s}"),
        Err(e) => tracing::error!("rerun wait error: {e}"),
    }
}

/// Spawns `python -m pipeline.runner` as a subprocess, passes the evidence
/// JSON payload via stdin, and waits for completion.  Must be called inside
/// a `tokio::spawn` so it runs in the background.
pub async fn spawn_pipeline(
    payload: String,
    results_path: &std::path::Path,
    audit_path: &std::path::Path,
) {
    use tokio::io::AsyncWriteExt as _;

    let mut child = match tokio::process::Command::new("python")
        .args([
            "-m",
            "pipeline.runner",
            "--results",
            results_path.to_str().unwrap_or("data/results.db"),
            "--audit",
            audit_path.to_str().unwrap_or("data/audit.db"),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit()) // surface Python tracebacks in server logs
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("failed to spawn pipeline subprocess: {e}");
            return;
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        if let Err(e) = stdin.write_all(payload.as_bytes()).await {
            tracing::error!("failed to write to pipeline stdin: {e}");
        }
    }

    match child.wait().await {
        Ok(s) if s.success() => tracing::info!("pipeline completed successfully"),
        Ok(s) => tracing::warn!("pipeline exited with status {s}"),
        Err(e) => tracing::error!("pipeline wait error: {e}"),
    }
}
