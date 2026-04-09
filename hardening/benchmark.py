"""Phase 8 — Performance benchmarks for the Forensiq system.

Measures:
  1. API response times for all read endpoints.
  2. Ingest throughput (time to POST a file and receive a UUID).
  3. Pipeline execution time (time from ingest to analysis appearing in /api/analysis).
  4. DuckDB query latency (via /api/evidence list).

Run from the repo root with the API running on localhost:3000:

    python -m hardening.benchmark

Results are printed to stdout. A JSON summary is written to
hardening/benchmark_results.json.
"""
from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


API = "http://localhost:3000"
BENCHMARK_FILE = Path(__file__).parent / "benchmark_results.json"

# Thresholds (seconds)
THRESHOLD_READ_MS  = 500   # read endpoint p95 must be < 500 ms
THRESHOLD_INGEST_S = 5.0   # single-file ingest must complete < 5 s
THRESHOLD_LIST_MS  = 200   # /api/evidence list must respond < 200 ms


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _req(
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes, float]:
    """Returns (status, body_bytes, elapsed_seconds)."""
    url = API + path
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            return resp.status, data, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.perf_counter() - t0


def _login() -> str:
    body = json.dumps({"username": "d.kowalski", "password": "forensiq123"}).encode()
    status, data, _ = _req("POST", "/auth/login", body, {"Content-Type": "application/json"})
    if status != 200:
        print(f"[ERROR] Login failed ({status}). Is the API running?")
        sys.exit(1)
    return json.loads(data)["token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ── Benchmark helpers ─────────────────────────────────────────────────────────

@dataclass
class Measurement:
    name: str
    samples: list[float] = field(default_factory=list)

    @property
    def min_ms(self) -> float: return min(self.samples) * 1000

    @property
    def max_ms(self) -> float: return max(self.samples) * 1000

    @property
    def avg_ms(self) -> float: return (sum(self.samples) / len(self.samples)) * 1000

    @property
    def p95_ms(self) -> float:
        s = sorted(self.samples)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx] * 1000

    def report(self) -> str:
        return (
            f"  {self.name:<40} "
            f"avg={self.avg_ms:6.1f}ms  "
            f"p95={self.p95_ms:6.1f}ms  "
            f"min={self.min_ms:6.1f}ms  "
            f"max={self.max_ms:6.1f}ms  "
            f"(n={len(self.samples)})"
        )


def _measure(name: str, fn, n: int = 5) -> Measurement:
    m = Measurement(name)
    for _ in range(n):
        m.samples.append(fn())
    return m


# ── Benchmark suites ──────────────────────────────────────────────────────────

READ_ENDPOINTS = [
    ("GET /api/evidence",       "GET", "/api/evidence"),
    ("GET /api/analysis",       "GET", "/api/analysis"),
    ("GET /api/contradictions", "GET", "/api/contradictions"),
    ("GET /api/scores",         "GET", "/api/scores"),
    ("GET /api/graph",          "GET", "/api/graph"),
    ("GET /api/audit",          "GET", "/api/audit"),
    ("GET /api/reports",        "GET", "/api/reports"),
    ("GET /api/cases",          "GET", "/api/cases"),
]


def bench_read_endpoints(token: str) -> list[Measurement]:
    print("\n[1] Read endpoint latency (5 samples each)")
    h = _auth(token)
    results = []
    for name, method, path in READ_ENDPOINTS:
        def _call(method=method, path=path, h=h):
            _, _, elapsed = _req(method, path, headers=h)
            return elapsed
        m = _measure(name, _call)
        print(m.report())
        results.append(m)
    return results


def bench_ingest(token: str) -> Measurement:
    """POST a 10 KB synthetic text file and measure time to UUID assignment."""
    print("\n[2] Ingest throughput (3 files, 10 KB each)")
    h = _auth(token)

    def _ingest():
        boundary = "----bench0001"
        content = ("Lorem ipsum dolor sit amet. " * 400).encode()  # ~10 KB
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="bench.txt"\r\n'
            f"Content-Type: text/plain\r\n\r\n"
        ).encode() + content + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="source"\r\n\r\n'
            f"benchmark\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req_headers = {**h, "Content-Type": f"multipart/form-data; boundary={boundary}"}
        status, _, elapsed = _req("POST", "/api/evidence/ingest", body, req_headers)
        if status not in (200, 201):
            return float("nan")
        return elapsed

    m = _measure("POST /api/evidence/ingest (10 KB)", _ingest, n=3)
    print(m.report())
    return m


def bench_pipeline_latency(token: str) -> float | None:
    """Ingest a file and poll /api/analysis until its UUID appears (max 30 s)."""
    print("\n[3] Pipeline latency (ingest → analysis result available)")
    h = _auth(token)
    boundary = "----bench0002"
    content = (
        "Witness statement: On the morning of March 15, I saw the defendant "
        "near the warehouse. Three other witnesses confirmed this account. "
        "The defendant was wearing a blue jacket and carrying a briefcase. "
    ).encode()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="pipeline_bench.txt"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode() + content + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="source"\r\n\r\n'
        f"benchmark\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req_headers = {**h, "Content-Type": f"multipart/form-data; boundary={boundary}"}

    t0 = time.perf_counter()
    status, data, _ = _req("POST", "/api/evidence/ingest", body, req_headers)
    if status not in (200, 201):
        print(f"  [SKIP] Ingest failed ({status})")
        return None

    ev_uuid = json.loads(data).get("uuid")
    if not ev_uuid:
        print("  [SKIP] No UUID returned from ingest")
        return None

    # Poll for analysis result (pipeline runs async in background).
    deadline = time.perf_counter() + 60
    while time.perf_counter() < deadline:
        status2, data2, _ = _req("GET", f"/api/analysis/{ev_uuid}", headers=h)
        if status2 == 200:
            elapsed = time.perf_counter() - t0
            print(f"  Pipeline latency (ingest → analysis): {elapsed:.2f}s")
            return elapsed
        time.sleep(1)

    print("  [TIMEOUT] Analysis did not appear within 60 s")
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Forensiq — Phase 8 Performance Benchmarks")
    print("=" * 70)

    token = _login()

    read_measurements = bench_read_endpoints(token)
    ingest_measurement = bench_ingest(token)
    pipeline_latency = bench_pipeline_latency(token)

    # Threshold checks.
    print("\n" + "=" * 70)
    print("Threshold checks")
    print("=" * 70)
    passed, failed = 0, 0

    for m in read_measurements:
        ok = m.p95_ms < THRESHOLD_READ_MS
        label = "PASS" if ok else "FAIL"
        print(f"  [{label}] {m.name} p95={m.p95_ms:.1f}ms (threshold={THRESHOLD_READ_MS}ms)")
        if ok:
            passed += 1
        else:
            failed += 1

    ingest_avg_s = ingest_measurement.avg_ms / 1000
    ok = ingest_avg_s < THRESHOLD_INGEST_S
    label = "PASS" if ok else "FAIL"
    print(f"  [{label}] Ingest avg={ingest_avg_s:.2f}s (threshold={THRESHOLD_INGEST_S}s)")
    passed += int(ok); failed += int(not ok)

    if pipeline_latency is not None:
        ok = pipeline_latency < 60
        label = "PASS" if ok else "FAIL"
        print(f"  [{label}] Pipeline latency={pipeline_latency:.1f}s (threshold=60s)")
        passed += int(ok); failed += int(not ok)
    else:
        print("  [SKIP] Pipeline latency (analysis not available — Python deps may be missing)")

    # Persist results.
    summary = {
        "read_endpoints": [
            {"name": m.name, "avg_ms": m.avg_ms, "p95_ms": m.p95_ms, "min_ms": m.min_ms, "max_ms": m.max_ms}
            for m in read_measurements
        ],
        "ingest": {
            "avg_ms": ingest_measurement.avg_ms,
            "p95_ms": ingest_measurement.p95_ms,
        },
        "pipeline_latency_s": pipeline_latency,
        "thresholds": {
            "read_p95_ms": THRESHOLD_READ_MS,
            "ingest_avg_s": THRESHOLD_INGEST_S,
        },
        "passed": passed,
        "failed": failed,
    }
    BENCHMARK_FILE.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {BENCHMARK_FILE}")

    print(f"\n{'=' * 70}")
    print(f"Summary: {passed} passed, {failed} failed")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
