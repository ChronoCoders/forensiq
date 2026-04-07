"""SQLite results database — schema, init, and read/write helpers.

All tables are append-only in spirit:
  - evidence_text / analysis_results / evidence_scores use UUID as PRIMARY KEY
    (upsert on re-analysis).
  - contradiction_results are deleted and rewritten whenever a new evidence
    item arrives, because contradiction counts affect all scores.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_text (
    uuid          TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    extracted_text TEXT NOT NULL,
    analyzed_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_results (
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
);

CREATE TABLE IF NOT EXISTS contradiction_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    evidence_uuid_a  TEXT NOT NULL,
    evidence_uuid_b  TEXT NOT NULL,
    label            TEXT NOT NULL,
    confidence       REAL NOT NULL,
    context          TEXT NOT NULL,
    explanation      TEXT NOT NULL,
    detected_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_scores (
    uuid                      TEXT PRIMARY KEY,
    rank                      INTEGER NOT NULL,
    score                     REAL NOT NULL,
    confidence                REAL NOT NULL,
    explanation               TEXT NOT NULL,
    feature_contributions_json TEXT NOT NULL,
    scored_at                 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id      TEXT PRIMARY KEY,
    label   TEXT NOT NULL,
    type    TEXT NOT NULL,
    color   TEXT NOT NULL,
    x       REAL NOT NULL DEFAULT 0.5,
    y       REAL NOT NULL DEFAULT 0.5,
    sources INTEGER NOT NULL DEFAULT 1,
    conn    INTEGER NOT NULL DEFAULT 0,
    notes   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS graph_edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    from_node TEXT NOT NULL,
    to_node   TEXT NOT NULL,
    type      TEXT NOT NULL DEFAULT 'co-occurrence',
    source    TEXT NOT NULL DEFAULT '',
    status    TEXT NOT NULL DEFAULT 'confirmed'
);
"""


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Writes ────────────────────────────────────────────────────────────────────

def upsert_evidence_text(conn: sqlite3.Connection, uuid: str, filename: str, text: str) -> None:
    conn.execute(
        """INSERT INTO evidence_text (uuid, filename, extracted_text, analyzed_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(uuid) DO UPDATE SET
               extracted_text = excluded.extracted_text,
               analyzed_at = excluded.analyzed_at""",
        (uuid, filename, text, now_iso()),
    )
    conn.commit()


def upsert_analysis(
    conn: sqlite3.Connection,
    *,
    uuid: str,
    document_type: str,
    doc_confidence: float,
    entities: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    entity_count: int,
    label_diversity: int,
    timeline_count: int,
    timeline_parsed: int,
) -> None:
    conn.execute(
        """INSERT INTO analysis_results
               (uuid, document_type, doc_confidence, entities_json, timeline_json,
                entity_count, label_diversity, timeline_count, timeline_parsed, analyzed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(uuid) DO UPDATE SET
               document_type   = excluded.document_type,
               doc_confidence  = excluded.doc_confidence,
               entities_json   = excluded.entities_json,
               timeline_json   = excluded.timeline_json,
               entity_count    = excluded.entity_count,
               label_diversity = excluded.label_diversity,
               timeline_count  = excluded.timeline_count,
               timeline_parsed = excluded.timeline_parsed,
               analyzed_at     = excluded.analyzed_at""",
        (
            uuid,
            document_type,
            doc_confidence,
            json.dumps(entities),
            json.dumps(timeline),
            entity_count,
            label_diversity,
            timeline_count,
            timeline_parsed,
            now_iso(),
        ),
    )
    conn.commit()


def replace_contradictions(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
) -> None:
    """Delete all existing contradiction rows and insert fresh results.

    Called after every ingest so that the full contradiction matrix stays
    consistent — new evidence can create new pairs with every prior item.
    """
    conn.execute("DELETE FROM contradiction_results")
    ts = now_iso()
    conn.executemany(
        """INSERT INTO contradiction_results
               (evidence_uuid_a, evidence_uuid_b, label, confidence, context, explanation, detected_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                r["evidence_uuid_a"],
                r["evidence_uuid_b"],
                r["label"],
                r["confidence"],
                r["context"],
                r["explanation"],
                ts,
            )
            for r in rows
        ],
    )
    conn.commit()


def replace_scores(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    """Upsert evidence scores (re-rank after every ingest)."""
    ts = now_iso()
    for r in rows:
        conn.execute(
            """INSERT INTO evidence_scores
                   (uuid, rank, score, confidence, explanation, feature_contributions_json, scored_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(uuid) DO UPDATE SET
                   rank = excluded.rank,
                   score = excluded.score,
                   confidence = excluded.confidence,
                   explanation = excluded.explanation,
                   feature_contributions_json = excluded.feature_contributions_json,
                   scored_at = excluded.scored_at""",
            (
                r["uuid"],
                r["rank"],
                r["score"],
                r["confidence"],
                r["explanation"],
                json.dumps(r["feature_contributions"]),
                ts,
            ),
        )
    conn.commit()


# ── Reads ─────────────────────────────────────────────────────────────────────

def all_evidence_text(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT uuid, filename, extracted_text FROM evidence_text"
    ).fetchall()]


def all_analysis(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT uuid, document_type, doc_confidence, entities_json, timeline_json,
                  entity_count, label_diversity, timeline_count, timeline_parsed, analyzed_at
           FROM analysis_results"""
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["entities"] = json.loads(d.pop("entities_json"))
        d["timeline"] = json.loads(d.pop("timeline_json"))
        result.append(d)
    return result


def all_contradictions_raw(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        """SELECT evidence_uuid_a, evidence_uuid_b, label, confidence, context, explanation
           FROM contradiction_results"""
    ).fetchall()]


def replace_graph(
    conn: sqlite3.Connection,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> None:
    conn.execute("DELETE FROM graph_nodes")
    conn.execute("DELETE FROM graph_edges")
    conn.executemany(
        """INSERT INTO graph_nodes (id, label, type, color, x, y, sources, conn, notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (n["id"], n["label"], n["type"], n["color"],
             n["x"], n["y"], n["sources"], n["conn"], n["notes"])
            for n in nodes
        ],
    )
    conn.executemany(
        """INSERT INTO graph_edges (from_node, to_node, type, source, status)
           VALUES (?, ?, ?, ?, ?)""",
        [(e["from"], e["to"], e["type"], e["source"], e["status"]) for e in edges],
    )
    conn.commit()


def get_graph(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    nodes = [dict(r) for r in conn.execute(
        "SELECT id, label, type, color, x, y, sources, conn, notes FROM graph_nodes"
    ).fetchall()]
    edges = [
        {**dict(r), "from": r["from_node"], "to": r["to_node"]}
        for r in conn.execute(
            "SELECT id, from_node, to_node, type, source, status FROM graph_edges"
        ).fetchall()
    ]
    return {"nodes": nodes, "edges": edges}


def contradiction_counts_per_uuid(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    """Return {uuid: (contradiction_count, uncertain_count)} for all evidence."""
    rows = conn.execute(
        """SELECT evidence_uuid_a AS uuid, label FROM contradiction_results
           UNION ALL
           SELECT evidence_uuid_b AS uuid, label FROM contradiction_results"""
    ).fetchall()
    counts: dict[str, tuple[int, int]] = {}
    for r in rows:
        uuid, label = r["uuid"], r["label"]
        c, u = counts.get(uuid, (0, 0))
        if label == "CONTRADICTION":
            c += 1
        elif label == "UNCERTAIN":
            u += 1
        counts[uuid] = (c, u)
    return counts
