use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};

use crate::{auth::AuthUser, state::AppState};

type Rejection = (StatusCode, Json<serde_json::Value>);

fn db_err(e: sqlx::Error) -> Rejection {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(serde_json::json!({ "error": e.to_string() })),
    )
}

// ── Analysis results ──────────────────────────────────────────────────────────

/// `GET /api/analysis` — all NLP analysis results (entities, timeline,
/// document classification) written by the pipeline after each ingest.
pub async fn list_analysis(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, Rejection> {
    let rows = sqlx::query(
        "SELECT uuid, document_type, doc_confidence, entities_json, timeline_json,
                entity_count, label_diversity, timeline_count, timeline_parsed, analyzed_at
         FROM analysis_results
         ORDER BY analyzed_at DESC",
    )
    .fetch_all(&state.results)
    .await
    .map_err(db_err)?;

    let items: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            let entities_json: String = r.get("entities_json");
            let timeline_json: String = r.get("timeline_json");
            serde_json::json!({
                "uuid":            r.get::<String, _>("uuid"),
                "document_type":   r.get::<String, _>("document_type"),
                "doc_confidence":  r.get::<f64, _>("doc_confidence"),
                "entities":        serde_json::from_str::<serde_json::Value>(&entities_json)
                                       .unwrap_or(serde_json::json!([])),
                "timeline":        serde_json::from_str::<serde_json::Value>(&timeline_json)
                                       .unwrap_or(serde_json::json!([])),
                "entity_count":    r.get::<i64, _>("entity_count"),
                "label_diversity": r.get::<i64, _>("label_diversity"),
                "timeline_count":  r.get::<i64, _>("timeline_count"),
                "timeline_parsed": r.get::<i64, _>("timeline_parsed"),
                "analyzed_at":     r.get::<String, _>("analyzed_at"),
            })
        })
        .collect();

    Ok(Json(items))
}

/// `GET /api/analysis/:uuid` — analysis result for a specific evidence item.
pub async fn get_analysis(
    _user: AuthUser,
    State(state): State<AppState>,
    Path(uuid): Path<String>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let row = sqlx::query(
        "SELECT uuid, document_type, doc_confidence, entities_json, timeline_json,
                entity_count, label_diversity, timeline_count, timeline_parsed, analyzed_at
         FROM analysis_results WHERE uuid = ?",
    )
    .bind(&uuid)
    .fetch_optional(&state.results)
    .await
    .map_err(db_err)?;

    match row {
        None => Err((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "analysis not found — pipeline may still be running" })),
        )),
        Some(r) => {
            use sqlx::Row;
            let entities_json: String = r.get("entities_json");
            let timeline_json: String = r.get("timeline_json");
            Ok(Json(serde_json::json!({
                "uuid":            r.get::<String, _>("uuid"),
                "document_type":   r.get::<String, _>("document_type"),
                "doc_confidence":  r.get::<f64, _>("doc_confidence"),
                "entities":        serde_json::from_str::<serde_json::Value>(&entities_json)
                                       .unwrap_or(serde_json::json!([])),
                "timeline":        serde_json::from_str::<serde_json::Value>(&timeline_json)
                                       .unwrap_or(serde_json::json!([])),
                "entity_count":    r.get::<i64, _>("entity_count"),
                "label_diversity": r.get::<i64, _>("label_diversity"),
                "timeline_count":  r.get::<i64, _>("timeline_count"),
                "timeline_parsed": r.get::<i64, _>("timeline_parsed"),
                "analyzed_at":     r.get::<String, _>("analyzed_at"),
            })))
        }
    }
}

// ── Contradictions ────────────────────────────────────────────────────────────

/// `GET /api/contradictions` — all contradiction pairs detected by the pipeline.
pub async fn list_contradictions(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, Rejection> {
    let rows = sqlx::query(
        "SELECT id, evidence_uuid_a, evidence_uuid_b, label, confidence,
                context, explanation, detected_at
         FROM contradiction_results
         ORDER BY confidence DESC",
    )
    .fetch_all(&state.results)
    .await
    .map_err(db_err)?;

    let items: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            serde_json::json!({
                "id":               r.get::<i64, _>("id"),
                "evidence_uuid_a":  r.get::<String, _>("evidence_uuid_a"),
                "evidence_uuid_b":  r.get::<String, _>("evidence_uuid_b"),
                "label":            r.get::<String, _>("label"),
                "confidence":       r.get::<f64, _>("confidence"),
                "context":          r.get::<String, _>("context"),
                "explanation":      r.get::<String, _>("explanation"),
                "detected_at":      r.get::<String, _>("detected_at"),
            })
        })
        .collect();

    Ok(Json(items))
}

// ── Graph ─────────────────────────────────────────────────────────────────────

/// `GET /api/graph` — relationship graph nodes and edges built from entity co-occurrence.
pub async fn get_graph(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let node_rows = sqlx::query(
        "SELECT id, label, type, color, x, y, sources, conn, notes FROM graph_nodes",
    )
    .fetch_all(&state.results)
    .await
    .map_err(db_err)?;

    let edge_rows = sqlx::query(
        "SELECT id, from_node, to_node, type, source, status FROM graph_edges",
    )
    .fetch_all(&state.results)
    .await
    .map_err(db_err)?;

    let nodes: Vec<serde_json::Value> = node_rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            serde_json::json!({
                "id":      r.get::<String, _>("id"),
                "label":   r.get::<String, _>("label"),
                "type":    r.get::<String, _>("type"),
                "color":   r.get::<String, _>("color"),
                "x":       r.get::<f64, _>("x"),
                "y":       r.get::<f64, _>("y"),
                "sources": r.get::<i64, _>("sources"),
                "conn":    r.get::<i64, _>("conn"),
                "notes":   r.get::<String, _>("notes"),
            })
        })
        .collect();

    let edges: Vec<serde_json::Value> = edge_rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            serde_json::json!({
                "id":     r.get::<i64, _>("id"),
                "from":   r.get::<String, _>("from_node"),
                "to":     r.get::<String, _>("to_node"),
                "type":   r.get::<String, _>("type"),
                "source": r.get::<String, _>("source"),
                "status": r.get::<String, _>("status"),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({ "nodes": nodes, "edges": edges })))
}

/// `POST /api/contradictions/rerun` — re-detect contradictions across all evidence pairs.
pub async fn rerun_contradictions(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, Rejection> {
    let results_path = state.results_path.clone();
    let audit_path   = state.audit_path.clone();

    tokio::spawn(async move {
        super::evidence::spawn_pipeline_rerun(&results_path, &audit_path).await;
    });

    Ok(Json(serde_json::json!({
        "status":  "queued",
        "message": "Full contradiction re-detection queued — results will update after completion",
    })))
}

// ── Scores ────────────────────────────────────────────────────────────────────

/// `GET /api/scores` — Bayesian reliability scores for all evidence, ranked.
pub async fn list_scores(
    _user: AuthUser,
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, Rejection> {
    let rows = sqlx::query(
        "SELECT uuid, rank, score, confidence, explanation,
                feature_contributions_json, scored_at
         FROM evidence_scores
         ORDER BY rank ASC",
    )
    .fetch_all(&state.results)
    .await
    .map_err(db_err)?;

    let items: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            use sqlx::Row;
            let fc_json: String = r.get("feature_contributions_json");
            serde_json::json!({
                "uuid":                  r.get::<String, _>("uuid"),
                "rank":                  r.get::<i64, _>("rank"),
                "score":                 r.get::<f64, _>("score"),
                "confidence":            r.get::<f64, _>("confidence"),
                "explanation":           r.get::<String, _>("explanation"),
                "feature_contributions": serde_json::from_str::<serde_json::Value>(&fc_json)
                                             .unwrap_or(serde_json::json!({})),
                "scored_at":             r.get::<String, _>("scored_at"),
            })
        })
        .collect();

    Ok(Json(items))
}
