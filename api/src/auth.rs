use async_trait::async_trait;
use axum::{
    extract::FromRequestParts,
    http::{request::Parts, HeaderMap, StatusCode},
    extract::State,
    Json,
    response::IntoResponse,
};
use chrono::{Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::state::{AppState, Session};

// ── Credentials ───────────────────────────────────────────────────────────────

/// Hardcoded users for the demo environment.
/// Replace with a proper credential store before production.
fn check_credentials(username: &str, password: &str) -> Option<&'static str> {
    match (username, password) {
        ("d.kowalski", "forensiq123") => Some("senior_analyst"),
        ("j.park",     "forensiq123") => Some("analyst"),
        ("m.torres",   "forensiq123") => Some("analyst"),
        _ => None,
    }
}

// ── Login / logout ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub user: UserInfo,
    pub expires_at: String,
}

#[derive(Serialize, Clone)]
pub struct UserInfo {
    pub username: String,
    pub role: String,
}

pub async fn login(
    State(state): State<AppState>,
    Json(body): Json<LoginRequest>,
) -> impl IntoResponse {
    let Some(role) = check_credentials(&body.username, &body.password) else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "invalid credentials" })),
        )
            .into_response();
    };

    let token = Uuid::new_v4().to_string();
    let expires_at = Utc::now() + Duration::hours(8);

    state.sessions.lock().await.insert(
        token.clone(),
        Session {
            username: body.username.clone(),
            role: role.to_string(),
            expires_at,
        },
    );

    Json(LoginResponse {
        token,
        user: UserInfo {
            username: body.username,
            role: role.to_string(),
        },
        expires_at: expires_at.to_rfc3339(),
    })
    .into_response()
}

pub async fn logout(State(state): State<AppState>, headers: HeaderMap) -> StatusCode {
    if let Some(token) = extract_bearer(&headers) {
        state.sessions.lock().await.remove(token);
    }
    StatusCode::OK
}

// ── AuthUser extractor ────────────────────────────────────────────────────────

/// Verified caller identity, extracted from the Bearer token on every
/// protected route.
pub struct AuthUser {
    pub username: String,
    pub role: String,
}

type AuthRejection = (StatusCode, Json<serde_json::Value>);

#[async_trait]
impl FromRequestParts<AppState> for AuthUser {
    type Rejection = AuthRejection;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let token = extract_bearer(&parts.headers).ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "missing Authorization header" })),
            )
        })?;

        let sessions = state.sessions.lock().await;
        let session = sessions.get(token).ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "invalid or expired token" })),
            )
        })?;

        if session.expires_at < Utc::now() {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "token expired" })),
            ));
        }

        Ok(AuthUser {
            username: session.username.clone(),
            role: session.role.clone(),
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
}
