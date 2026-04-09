use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use report::{Contradiction, Report, ReportRequest, ScoredEvidence};
use serde::Deserialize;
use uuid::Uuid;

use crate::{auth::AuthUser, state::AppState};

type Rejection = (StatusCode, Json<serde_json::Value>);

fn err(status: StatusCode, msg: impl std::fmt::Display) -> Rejection {
    (status, Json(serde_json::json!({ "error": msg.to_string() })))
}

#[derive(Deserialize)]
pub struct GenerateReportBody {
    pub case_id: String,
    pub ranked_evidence: Vec<ScoredEvidence>,
    pub contradictions: Vec<Contradiction>,
}

/// `GET /api/reports` — list all generated reports from SQLite.
pub async fn list_reports(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<Report>>, Rejection> {
    let rows = sqlx::query(
        "SELECT payload_json FROM reports ORDER BY generated_at DESC",
    )
    .fetch_all(&state.results)
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let reports: Vec<Report> = rows
        .iter()
        .filter_map(|r| {
            use sqlx::Row;
            let json: String = r.get("payload_json");
            serde_json::from_str(&json).ok()
        })
        .collect();

    Ok(Json(reports))
}

/// `POST /api/reports/generate` — generate a new tamper-evident report and persist it.
pub async fn generate_report(
    user: AuthUser,
    State(state): State<AppState>,
    Json(body): Json<GenerateReportBody>,
) -> Result<Json<Report>, Rejection> {
    let req = ReportRequest {
        case_id: body.case_id,
        generated_by: user.username,
        ranked_evidence: body.ranked_evidence,
        contradictions: body.contradictions,
    };

    let generated = report::generate(req, &state.audit)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let payload_json = serde_json::to_string(&generated)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    sqlx::query(
        "INSERT OR REPLACE INTO reports
            (report_id, case_id, generated_at, generated_by,
             evidence_count, contradiction_count, report_sha256, payload_json)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(generated.report_id.to_string())
    .bind(&generated.case_id)
    .bind(generated.generated_at.to_rfc3339())
    .bind(&generated.generated_by)
    .bind(generated.evidence_count as i64)
    .bind(generated.contradiction_count as i64)
    .bind(&generated.report_sha256)
    .bind(&payload_json)
    .execute(&state.results)
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(generated))
}

/// `POST /api/reports/:id/verify` — verify a report's SHA-256 integrity.
pub async fn verify_report(
    _user: AuthUser,
    State(state): State<AppState>,
    Path(id_str): Path<String>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let id = Uuid::parse_str(&id_str)
        .map_err(|_| err(StatusCode::BAD_REQUEST, "invalid UUID"))?;

    let row = sqlx::query("SELECT payload_json FROM reports WHERE report_id = ?")
        .bind(id.to_string())
        .fetch_optional(&state.results)
        .await
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "report not found"))?;

    use sqlx::Row;
    let json: String = row.get("payload_json");
    let rpt: Report = serde_json::from_str(&json)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let valid = report::verify_integrity(&rpt)
        .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "report_id":      id_str,
        "valid":          valid,
        "report_sha256":  rpt.report_sha256,
    })))
}
