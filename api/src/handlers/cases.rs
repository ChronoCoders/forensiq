use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;

use crate::{auth::AuthUser, state::AppState};

type Rejection = (StatusCode, Json<serde_json::Value>);

fn case_err(msg: impl std::fmt::Display) -> Rejection {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": msg.to_string() })))
}

#[derive(Deserialize)]
pub struct SignoffBody {
    pub notes: String,
}

#[derive(Deserialize)]
pub struct CasesQuery {
    pub status: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateCaseBody {
    pub title:  String,
    pub charge: Option<String>,
}

/// `GET /api/cases` — list all cases, optionally filtered by `?status=active|archived`.
pub async fn list_cases(
    _user: AuthUser,
    State(state): State<AppState>,
    Query(q): Query<CasesQuery>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let rows = if let Some(ref status) = q.status {
        sqlx::query(
            "SELECT id, title, charge, analyst, status, outcome, opened_at, closed_at
             FROM cases WHERE status = ? ORDER BY opened_at DESC",
        )
        .bind(status)
        .fetch_all(&state.audit)
        .await
    } else {
        sqlx::query(
            "SELECT id, title, charge, analyst, status, outcome, opened_at, closed_at
             FROM cases ORDER BY opened_at DESC",
        )
        .fetch_all(&state.audit)
        .await
    }
    .map_err(|e| case_err(e))?;

    let cases: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            serde_json::json!({
                "id":        r.get::<String, _>("id"),
                "title":     r.get::<String, _>("title"),
                "charge":    r.get::<String, _>("charge"),
                "analyst":   r.get::<String, _>("analyst"),
                "status":    r.get::<String, _>("status"),
                "outcome":   r.get::<String, _>("outcome"),
                "opened_at": r.get::<String, _>("opened_at"),
                "closed_at": r.get::<Option<String>, _>("closed_at"),
            })
        })
        .collect();

    Ok(Json(serde_json::json!(cases)))
}

/// `POST /api/cases` — create a new case.
pub async fn create_case(
    user: AuthUser,
    State(state): State<AppState>,
    Json(body): Json<CreateCaseBody>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let title = body.title.trim().to_string();
    if title.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "title is required" })),
        ));
    }
    let charge  = body.charge.unwrap_or_default();
    let now     = chrono::Utc::now();
    let case_id = format!("{}-{:04}", now.format("%Y"), case_id_suffix());
    let now_str = now.to_rfc3339();

    sqlx::query(
        "INSERT INTO cases (id, title, charge, analyst, status, outcome, opened_at)
         VALUES (?, ?, ?, ?, 'active', '', ?)",
    )
    .bind(&case_id)
    .bind(&title)
    .bind(&charge)
    .bind(&user.username)
    .bind(&now_str)
    .execute(&state.audit)
    .await
    .map_err(|e| case_err(e))?;

    Ok(Json(serde_json::json!({
        "id":        case_id,
        "title":     title,
        "charge":    charge,
        "analyst":   user.username,
        "status":    "active",
        "opened_at": now_str,
    })))
}

fn case_id_suffix() -> u16 {
    use std::time::{SystemTime, UNIX_EPOCH};
    (SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_micros()
        % 9000
        + 1000) as u16
}

/// `POST /api/cases/:id/signoff` — record an analyst sign-off event.
///
/// Validates that notes are non-empty, then appends an `analyst_signoff` event
/// to the tamper-evident audit chain.
pub async fn signoff(
    user: AuthUser,
    State(state): State<AppState>,
    Path(case_id): Path<String>,
    Json(body): Json<SignoffBody>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let notes = body.notes.trim().to_string();
    if notes.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "notes are required for sign-off" })),
        ));
    }

    let payload = serde_json::json!({
        "case_id": case_id,
        "analyst": user.username,
        "role":    user.role,
        "notes":   notes,
    })
    .to_string();

    audit::log_event(&state.audit, "analyst_signoff", &payload)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;

    Ok(Json(serde_json::json!({
        "ok":      true,
        "case_id": case_id,
        "analyst": user.username,
    })))
}
