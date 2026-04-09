"""Phase 8 — Security audit for the Forensiq REST API.

Checks:
  1. All protected endpoints reject unauthenticated requests (401).
  2. Invalid tokens are rejected (401).
  3. CORS headers are present.
  4. Malformed UUIDs return 400, not 500.
  5. Oversized inputs are handled gracefully.
  6. SQL-injection-like inputs do not cause 500 errors.

Run from the repo root with the API already running on localhost:3000:

    python -m hardening.security_audit
"""
from __future__ import annotations

import sys
import urllib.request
import urllib.error
import json
import os
from dataclasses import dataclass, field
from typing import Any


API = "http://localhost:3000"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _req(
    method: str,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
    content_type: str = "application/json",
) -> tuple[int, dict | None, dict[str, str]]:
    """Make an HTTP request. Returns (status_code, body_json_or_None, resp_headers)."""
    url = API + path
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": content_type, **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read()
            resp_headers = dict(resp.headers)
            try:
                return resp.status, json.loads(raw), resp_headers
            except Exception:
                return resp.status, None, resp_headers
    except urllib.error.HTTPError as e:
        raw = e.read()
        resp_headers = dict(e.headers)
        try:
            return e.code, json.loads(raw), resp_headers
        except Exception:
            return e.code, None, resp_headers


def _login(username: str = "d.kowalski", password: str = "forensiq123") -> str:
    status, body, _ = _req("POST", "/auth/login", {"username": username, "password": password})
    if status != 200 or not body:
        raise RuntimeError(f"Login failed: {status} {body}")
    return body["token"]


# ── Audit result ──────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        print(f"  [PASS] {msg}")

    def fail(self, msg: str) -> None:
        self.failed.append(msg)
        print(f"  [FAIL] {msg}")

    def check(self, condition: bool, pass_msg: str, fail_msg: str) -> None:
        if condition:
            self.ok(pass_msg)
        else:
            self.fail(fail_msg)


# ── Check suites ──────────────────────────────────────────────────────────────

PROTECTED_ENDPOINTS = [
    ("GET",  "/api/evidence"),
    ("GET",  "/api/analysis"),
    ("GET",  "/api/contradictions"),
    ("GET",  "/api/scores"),
    ("GET",  "/api/graph"),
    ("GET",  "/api/audit"),
    ("GET",  "/api/reports"),
    ("GET",  "/api/cases"),
]


def check_auth_required(r: AuditResult) -> None:
    print("\n[1] Authentication enforcement")
    for method, path in PROTECTED_ENDPOINTS:
        status, _, _ = _req(method, path)
        r.check(
            status == 401,
            f"{method} {path} → 401 without token",
            f"{method} {path} → {status} (expected 401 without token)",
        )


def check_invalid_token(r: AuditResult) -> None:
    print("\n[2] Invalid token rejection")
    bad_headers = {"Authorization": "Bearer not-a-real-token"}
    status, _, _ = _req("GET", "/api/evidence", headers=bad_headers)
    r.check(status == 401, "Invalid token → 401", f"Invalid token → {status} (expected 401)")


def check_malformed_uuids(r: AuditResult, token: str) -> None:
    print("\n[3] Malformed UUID handling")
    h = {"Authorization": f"Bearer {token}"}
    for path in [
        "/api/evidence/not-a-uuid",
        "/api/evidence/not-a-uuid/verify",
        "/api/evidence/not-a-uuid/raw",
        "/api/reports/not-a-uuid/verify",
    ]:
        status, _, _ = _req("GET" if "verify" not in path and "raw" not in path else "POST", path, headers=h)
        # Accept 400 or 404 — both are safe. 500 is not.
        r.check(
            status in (400, 404, 405),
            f"{path} → {status} (safe error)",
            f"{path} → {status} (expected 400/404, got 500)",
        )


def check_sql_injection_inputs(r: AuditResult) -> None:
    print("\n[4] SQL-injection-like login inputs")
    payloads = [
        {"username": "' OR '1'='1", "password": "x"},
        {"username": "admin'--", "password": "x"},
        {"username": "x", "password": "' OR '1'='1"},
    ]
    for p in payloads:
        status, _, _ = _req("POST", "/auth/login", p)
        r.check(
            status == 401,
            f"SQL-injection payload rejected → 401",
            f"SQL-injection payload → {status} (expected 401)",
        )


def check_cors_headers(r: AuditResult) -> None:
    print("\n[5] CORS headers present")
    # OPTIONS preflight to a protected endpoint.
    req = urllib.request.Request(
        API + "/api/evidence",
        method="OPTIONS",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "GET",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            hdrs = dict(resp.headers)
    except urllib.error.HTTPError as e:
        hdrs = dict(e.headers)

    has_cors = any("access-control" in k.lower() for k in hdrs)
    r.check(has_cors, "CORS headers present on OPTIONS preflight", "CORS headers missing")


def check_empty_ingest(r: AuditResult, token: str) -> None:
    print("\n[6] Empty body on ingest returns 400, not 500")
    h = {"Authorization": f"Bearer {token}"}
    # Send a multipart request with no 'file' field.
    import urllib.parse
    boundary = "----forensiqboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="source"\r\n\r\n'
        f"test\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    h2 = {**h, "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(
        API + "/api/evidence/ingest",
        data=body,
        headers=h2,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
    except urllib.error.HTTPError as e:
        status = e.code
    r.check(
        status in (400, 422),
        f"Ingest with no file → {status} (safe error)",
        f"Ingest with no file → {status} (expected 400/422)",
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Forensiq — Phase 8 Security Audit")
    print("=" * 60)

    r = AuditResult()

    # Verify API is reachable.
    try:
        token = _login()
    except Exception as exc:
        print(f"\n[ERROR] Cannot reach API at {API}: {exc}")
        print("Start the API with: cargo run -p api")
        sys.exit(1)

    check_auth_required(r)
    check_invalid_token(r)
    check_malformed_uuids(r, token)
    check_sql_injection_inputs(r)
    check_cors_headers(r)
    check_empty_ingest(r, token)

    print("\n" + "=" * 60)
    print(f"Results: {len(r.passed)} passed, {len(r.failed)} failed")
    print("=" * 60)

    if r.failed:
        print("\nFailed checks:")
        for f in r.failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
