use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;

use crate::{auth::AuthUser, state::AppState};

type Rejection = (StatusCode, Json<serde_json::Value>);

#[derive(Serialize)]
pub struct AuditIngestEvent {
    pub id: i64,
    pub uuid: String,
    pub filename: String,
    pub sha256: String,
    pub source: String,
    pub operator_id: String,
    pub ingested_at: String,
}

#[derive(Serialize)]
pub struct AuditSystemEvent {
    pub id: i64,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub occurred_at: String,
}

#[derive(Serialize)]
pub struct AuditLog {
    pub ingest_events: Vec<AuditIngestEvent>,
    pub system_events: Vec<AuditSystemEvent>,
}

/// `GET /api/audit` — return the full audit chain (ingest log + event log).
pub async fn get_audit_log(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<AuditLog>, Rejection> {
    let ingest_rows = sqlx::query_as::<
        _,
        (i64, String, String, String, String, String, String),
    >(
        "SELECT id, uuid, filename, sha256, source, operator_id, ingested_at \
         FROM audit_log ORDER BY id DESC",
    )
    .fetch_all(&state.audit)
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    })?;

    let event_rows =
        sqlx::query_as::<_, (i64, String, String, String)>(
            "SELECT id, event_type, payload, occurred_at \
             FROM event_log ORDER BY id DESC LIMIT 500",
        )
        .fetch_all(&state.audit)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;

    let ingest_events = ingest_rows
        .into_iter()
        .map(|(id, uuid, filename, sha256, source, operator_id, ingested_at)| {
            AuditIngestEvent {
                id,
                uuid,
                filename,
                sha256,
                source,
                operator_id,
                ingested_at,
            }
        })
        .collect();

    let system_events = event_rows
        .into_iter()
        .map(|(id, event_type, payload, occurred_at)| {
            let payload_val: serde_json::Value =
                serde_json::from_str(&payload).unwrap_or(serde_json::Value::String(payload));
            AuditSystemEvent {
                id,
                event_type,
                payload: payload_val,
                occurred_at,
            }
        })
        .collect();

    Ok(Json(AuditLog {
        ingest_events,
        system_events,
    }))
}
