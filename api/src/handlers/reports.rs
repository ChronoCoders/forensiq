use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use report::{Contradiction, ReportRequest, ScoredEvidence};
use serde::Deserialize;
use uuid::Uuid;

use crate::{auth::AuthUser, state::AppState};

type Rejection = (StatusCode, Json<serde_json::Value>);

#[derive(Deserialize)]
pub struct GenerateReportBody {
    pub case_id: String,
    pub ranked_evidence: Vec<ScoredEvidence>,
    pub contradictions: Vec<Contradiction>,
}

/// `GET /api/reports` — list all generated reports (in-memory store).
pub async fn list_reports(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Json<Vec<report::Report>> {
    Json(state.reports.lock().await.clone())
}

/// `POST /api/reports/generate` — generate a new tamper-evident report.
///
/// The caller supplies pre-scored evidence and contradictions (from the Python
/// analysis layer). The Rust report crate hashes the payload and logs an audit
/// event.
pub async fn generate_report(
    user: AuthUser,
    State(state): State<AppState>,
    Json(body): Json<GenerateReportBody>,
) -> Result<Json<report::Report>, Rejection> {
    let req = ReportRequest {
        case_id: body.case_id,
        generated_by: user.username,
        ranked_evidence: body.ranked_evidence,
        contradictions: body.contradictions,
    };

    let generated = report::generate(req, &state.audit)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
        })?;

    state.reports.lock().await.push(generated.clone());
    Ok(Json(generated))
}

/// `POST /api/reports/:id/verify` — verify a report's SHA-256 integrity.
pub async fn verify_report(
    _user: AuthUser,
    State(state): State<AppState>,
    Path(id_str): Path<String>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let id = Uuid::parse_str(&id_str).map_err(|_| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "invalid UUID" })),
        )
    })?;

    let reports = state.reports.lock().await;
    let rpt = reports
        .iter()
        .find(|r| r.report_id == id)
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "report not found" })),
            )
        })?;

    let valid = report::verify_integrity(rpt).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
    })?;

    Ok(Json(serde_json::json!({
        "report_id":    id_str,
        "valid":        valid,
        "report_sha256": rpt.report_sha256,
    })))
}
