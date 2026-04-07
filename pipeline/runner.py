"""Forensiq analysis pipeline runner.

Invoked by the Rust API after every evidence ingest:

    python -m pipeline.runner \
        --results <path/to/results.db> \
        --audit   <path/to/audit.db>

New evidence is read as a single JSON object from stdin:

    {"uuid": "...", "filename": "...", "text": "..."}

Pipeline steps
--------------
1. Extract text (received from Rust via stdin).
2. Classify document type + extract entities + extract timeline events.
3. Fetch all previously analyzed evidence from results.db.
4. Run contradiction detection: new evidence vs. every prior item.
5. Rebuild contradiction matrix in results.db.
6. Re-score all evidence with updated contradiction counts.
7. Log events to the audit chain.

Text extraction note
--------------------
Rust passes a best-effort UTF-8 decode of the raw bytes. Binary formats
(images, video) will arrive as empty or garbled strings — analysis will
produce minimal but valid results. A production deployment should run
format-specific extractors (PyPDF2, python-docx, ffprobe) before calling
this runner.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure the repo root is importable regardless of cwd.
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pipeline.results_db as rdb


# ── Text helpers ──────────────────────────────────────────────────────────────

_MAX_CHARS = 50_000  # cap for model input


def _truncate(text: str) -> str:
    return text[:_MAX_CHARS] if len(text) > _MAX_CHARS else text


# ── Analysis ──────────────────────────────────────────────────────────────────

def _run_analysis(text: str) -> dict[str, Any]:
    """Run NLP analysis. Returns a dict with classification, entities, timeline."""
    text = _truncate(text)

    try:
        from analysis.classification import classify_document
        classification = classify_document(text)
        document_type = classification.document_type.value
        doc_confidence = classification.confidence
    except Exception:
        document_type = "OTHER"
        doc_confidence = 0.5

    try:
        from analysis.entities import extract_entities
        raw_entities = extract_entities(text)
        entities = [
            {
                "text": e.text,
                "label": e.label,
                "confidence": e.confidence,
                "explanation": e.explanation,
            }
            for e in raw_entities
        ]
    except Exception:
        entities = []

    try:
        from analysis.timeline import extract_timeline
        raw_timeline = extract_timeline(text)
        timeline = [
            {
                "sentence": t.sentence,
                "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                "timestamp_text": t.timestamp_text,
                "entities_involved": t.entities_involved,
                "confidence": t.confidence,
                "explanation": t.explanation,
            }
            for t in raw_timeline
        ]
    except Exception:
        timeline = []

    label_diversity = len({e["label"] for e in entities})
    timeline_parsed = sum(1 for t in timeline if t["timestamp"] is not None)

    return {
        "document_type": document_type,
        "doc_confidence": doc_confidence,
        "entities": entities,
        "timeline": timeline,
        "entity_count": len(entities),
        "label_diversity": label_diversity,
        "timeline_count": len(timeline),
        "timeline_parsed": timeline_parsed,
    }


# ── Contradiction detection ───────────────────────────────────────────────────

def _run_contradictions(
    new_uuid: str,
    new_text: str,
    prior: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Detect contradictions between new evidence and all prior items.

    Returns only pairs involving the new evidence — existing prior×prior
    contradictions are preserved from the previous run in results.db.
    """
    if not prior or not new_text.strip():
        return []

    try:
        from contradiction.detector import ContradictionDetector, StatementPair
        detector = ContradictionDetector()

        pairs = [
            StatementPair(
                evidence_uuid_a=new_uuid,
                evidence_uuid_b=p["uuid"],
                text_a=_truncate(new_text),
                text_b=_truncate(p["extracted_text"]),
                context="evidence",
            )
            for p in prior
            if p["extracted_text"].strip()
        ]
        if not pairs:
            return []

        results = detector.detect_batch(pairs)
        return [
            {
                "evidence_uuid_a": r.evidence_uuid_a,
                "evidence_uuid_b": r.evidence_uuid_b,
                "label": r.label,
                "confidence": r.confidence,
                "context": r.context,
                "explanation": r.explanation,
            }
            for r in results
        ]
    except Exception:
        return []


# ── Scoring ───────────────────────────────────────────────────────────────────

def _run_scoring(
    all_analysis: list[dict[str, Any]],
    contradiction_counts: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    """Score all evidence using Bayesian reliability ranking."""
    try:
        from reasoning.scoring import EvidenceFeatures, EvidenceScorer

        features_list = [
            EvidenceFeatures(
                evidence_uuid=a["uuid"],
                document_type=a["document_type"],
                classification_confidence=a["doc_confidence"],
                entity_count=a["entity_count"],
                entity_label_diversity=a["label_diversity"],
                timeline_event_count=a["timeline_count"],
                timeline_parsed_count=a["timeline_parsed"],
                graph_edge_count=a["entity_count"],  # proxy: entities can form graph edges
                contradiction_count=contradiction_counts.get(a["uuid"], (0, 0))[0],
                uncertain_contradiction_count=contradiction_counts.get(a["uuid"], (0, 0))[1],
            )
            for a in all_analysis
        ]

        scorer = EvidenceScorer()
        scores = scorer.score(features_list)
        return [
            {
                "uuid": s.evidence_uuid,
                "rank": s.rank,
                "score": s.score,
                "confidence": s.confidence,
                "explanation": s.explanation,
                "feature_contributions": s.feature_contributions,
            }
            for s in scores
        ]
    except Exception:
        # Fallback: return uniform scores in original order.
        return [
            {
                "uuid": a["uuid"],
                "rank": i + 1,
                "score": 0.5,
                "confidence": 0.5,
                "explanation": "scoring unavailable",
                "feature_contributions": {},
            }
            for i, a in enumerate(all_analysis)
        ]


# ── Relationship graph ────────────────────────────────────────────────────────

_ENTITY_COLORS: dict[str, str] = {
    "PERSON":       "rgba(34,197,94",
    "PER":          "rgba(34,197,94",
    "LOC":          "rgba(20,184,166",
    "GPE":          "rgba(20,184,166",
    "LOCATION":     "rgba(20,184,166",
    "ORG":          "rgba(168,85,247",
    "ORGANIZATION": "rgba(168,85,247",
    "SUBJECT":      "rgba(59,127,255",
    "DATE":         "rgba(245,158,11",
    "TIME":         "rgba(245,158,11",
    "EVENT":        "rgba(245,158,11",
}

_TYPE_NAMES: dict[str, str] = {
    "PER": "Person",       "PERSON": "Person",
    "LOC": "Location",     "GPE": "Location",    "LOCATION": "Location",
    "ORG": "Witness",      "ORGANIZATION": "Witness",
    "DATE": "Evidence",    "TIME": "Evidence",    "EVENT": "Evidence",
    "SUBJECT": "Subject",
}


def _build_graph(
    all_analysis: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a relationship graph from extracted entities and contradiction pairs."""
    import math

    # Set of doc-pair UUIDs that are contradictions.
    conflict_pairs: set[tuple[str, str]] = set()
    for c in contradictions:
        if c["label"] == "CONTRADICTION":
            a, b = c["evidence_uuid_a"], c["evidence_uuid_b"]
            conflict_pairs.add((a, b))
            conflict_pairs.add((b, a))

    nodes_map: dict[str, dict[str, Any]] = {}
    entity_doc_map: dict[str, set[str]] = {}  # entity_id → set of doc uuids
    edges_map: dict[tuple[str, str], dict[str, Any]] = {}  # (a,b) sorted → edge

    for doc in all_analysis:
        doc_uuid = doc["uuid"]
        seen_in_doc: list[str] = []

        for ent in doc.get("entities", [])[:20]:  # cap per document
            raw = ent.get("text", "").strip()
            if not raw:
                continue
            eid = raw.lower().replace(" ", "_")[:50]
            lbl = ent.get("label", "OTHER").upper()

            if eid not in nodes_map:
                nodes_map[eid] = {
                    "id":      eid,
                    "label":   raw[:30],
                    "type":    _TYPE_NAMES.get(lbl, lbl.capitalize()),
                    "color":   _ENTITY_COLORS.get(lbl, "rgba(245,158,11"),
                    "x":       0.5,
                    "y":       0.5,
                    "sources": 0,
                    "conn":    0,
                    "notes":   "",
                }
            nodes_map[eid]["sources"] += 1
            entity_doc_map.setdefault(eid, set()).add(doc_uuid)

            if eid not in seen_in_doc:
                seen_in_doc.append(eid)

        # Co-occurrence edges within this document.
        for i in range(len(seen_in_doc)):
            for j in range(i + 1, len(seen_in_doc)):
                a, b = seen_in_doc[i], seen_in_doc[j]
                key = (min(a, b), max(a, b))
                if key not in edges_map:
                    edges_map[key] = {
                        "from":   key[0],
                        "to":     key[1],
                        "type":   "co-occurrence",
                        "source": doc_uuid[:8],
                        "status": "confirmed",
                    }

    # Upgrade edges to conflict where the documents they span contradict each other.
    for key, edge in edges_map.items():
        docs_a = entity_doc_map.get(key[0], set())
        docs_b = entity_doc_map.get(key[1], set())
        for da in docs_a:
            for db in docs_b:
                if (da, db) in conflict_pairs:
                    edge["status"] = "conflict"
                    break

    # Update connection counts.
    for key in edges_map:
        for eid in key:
            if eid in nodes_map:
                nodes_map[eid]["conn"] += 1

    # Circular layout: spread nodes evenly around the canvas.
    nodes = list(nodes_map.values())
    n = len(nodes)
    for i, node in enumerate(nodes):
        if n == 1:
            node["x"] = node["y"] = 0.5
        else:
            angle = 2 * math.pi * i / n
            node["x"] = round(0.5 + 0.35 * math.cos(angle), 4)
            node["y"] = round(0.5 + 0.30 * math.sin(angle), 4)

    return nodes, list(edges_map.values())


# ── Audit logging ─────────────────────────────────────────────────────────────

def _log_audit(audit_path: str, event_type: str, payload: dict[str, Any]) -> None:
    try:
        conn = sqlite3.connect(audit_path)
        conn.execute(
            "INSERT INTO event_log (event_type, payload, occurred_at) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # audit failure must never crash the pipeline


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Forensiq analysis pipeline runner")
    parser.add_argument("--results", required=True, help="Path to results.db SQLite file")
    parser.add_argument("--audit",   required=True, help="Path to audit.db SQLite file")
    args = parser.parse_args()

    # Read evidence from stdin.
    try:
        evidence = json.load(sys.stdin)
        uuid: str     = evidence["uuid"]
        filename: str = evidence["filename"]
        text: str     = evidence.get("text", "")
    except Exception as exc:
        print(f"pipeline: failed to read stdin: {exc}", file=sys.stderr)
        sys.exit(1)

    results_path = Path(args.results)
    conn = rdb.connect(results_path)

    try:
        # 1. Run NLP analysis on new evidence.
        analysis = _run_analysis(text)

        # 2. Store text + analysis result.
        rdb.upsert_evidence_text(conn, uuid, filename, text)
        rdb.upsert_analysis(
            conn,
            uuid=uuid,
            document_type=analysis["document_type"],
            doc_confidence=analysis["doc_confidence"],
            entities=analysis["entities"],
            timeline=analysis["timeline"],
            entity_count=analysis["entity_count"],
            label_diversity=analysis["label_diversity"],
            timeline_count=analysis["timeline_count"],
            timeline_parsed=analysis["timeline_parsed"],
        )
        _log_audit(args.audit, "analysis_complete", {
            "uuid": uuid,
            "document_type": analysis["document_type"],
            "entity_count": analysis["entity_count"],
            "timeline_count": analysis["timeline_count"],
        })

        # 3. Load all prior evidence text (excluding the item just ingested).
        prior_texts = [
            p for p in rdb.all_evidence_text(conn) if p["uuid"] != uuid
        ]

        # 4. Detect contradictions between new evidence and all prior items.
        new_contradictions = _run_contradictions(uuid, text, prior_texts)

        # 5. Load existing contradiction pairs not involving the new UUID,
        #    merge with new pairs, and write back atomically.
        existing_rows = conn.execute(
            """SELECT evidence_uuid_a, evidence_uuid_b, label, confidence, context, explanation
               FROM contradiction_results
               WHERE evidence_uuid_a != ? AND evidence_uuid_b != ?""",
            (uuid, uuid),
        ).fetchall()

        all_contradictions = [
            {
                "evidence_uuid_a": r["evidence_uuid_a"],
                "evidence_uuid_b": r["evidence_uuid_b"],
                "label": r["label"],
                "confidence": r["confidence"],
                "context": r["context"],
                "explanation": r["explanation"],
            }
            for r in existing_rows
        ] + new_contradictions

        rdb.replace_contradictions(conn, all_contradictions)

        if new_contradictions:
            _log_audit(args.audit, "contradictions_detected", {
                "uuid": uuid,
                "new_pairs": len(new_contradictions),
                "flagged": sum(1 for c in new_contradictions if c["label"] == "CONTRADICTION"),
            })

        # 6. Re-score all evidence with updated contradiction counts.
        all_analysis_rows = rdb.all_analysis(conn)
        contradiction_counts = rdb.contradiction_counts_per_uuid(conn)
        scores = _run_scoring(all_analysis_rows, contradiction_counts)
        rdb.replace_scores(conn, scores)

        _log_audit(args.audit, "scoring_complete", {
            "uuid": uuid,
            "total_scored": len(scores),
        })

        # 7. Rebuild relationship graph from all entity data.
        all_analysis_for_graph = rdb.all_analysis(conn)
        all_contrs_raw = rdb.all_contradictions_raw(conn)
        graph_nodes, graph_edges = _build_graph(all_analysis_for_graph, all_contrs_raw)
        rdb.replace_graph(conn, graph_nodes, graph_edges)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
