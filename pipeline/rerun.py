"""Forensiq full contradiction re-run.

Re-detects contradictions across ALL stored evidence pairs (not just new vs.
prior), then re-scores and rebuilds the relationship graph.

Invoked by the Rust API as:
    python -m pipeline.rerun --results <path> --audit <path>
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from itertools import combinations
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pipeline.results_db as rdb
from pipeline.runner import _run_scoring, _build_graph, _log_audit

_MAX_CHARS = 50_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensiq full contradiction rerun")
    parser.add_argument("--results", required=True, help="Path to results.db")
    parser.add_argument("--audit",   required=True, help="Path to audit.db")
    args = parser.parse_args()

    conn = rdb.connect(Path(args.results))

    try:
        all_texts = rdb.all_evidence_text(conn)
        if len(all_texts) < 2:
            print("pipeline.rerun: fewer than 2 evidence items — nothing to compare",
                  file=sys.stderr)
            return

        # Detect contradictions across all unique pairs.
        all_contrs: list[dict[str, Any]] = []
        try:
            from contradiction.detector import ContradictionDetector, StatementPair

            pairs = [
                StatementPair(
                    evidence_uuid_a=a["uuid"],
                    evidence_uuid_b=b["uuid"],
                    text_a=a["extracted_text"][:_MAX_CHARS],
                    text_b=b["extracted_text"][:_MAX_CHARS],
                    context="evidence",
                )
                for a, b in combinations(all_texts, 2)
                if a["extracted_text"].strip() and b["extracted_text"].strip()
            ]
            if pairs:
                detector = ContradictionDetector()
                results  = detector.detect_batch(pairs)
                all_contrs = [
                    {
                        "evidence_uuid_a": r.evidence_uuid_a,
                        "evidence_uuid_b": r.evidence_uuid_b,
                        "label":           r.label,
                        "confidence":      r.confidence,
                        "context":         r.context,
                        "explanation":     r.explanation,
                    }
                    for r in results
                ]
        except Exception:
            traceback.print_exc(file=sys.stderr)

        rdb.replace_contradictions(conn, all_contrs)

        # Re-score all evidence with updated contradiction counts.
        all_analysis   = rdb.all_analysis(conn)
        contr_counts   = rdb.contradiction_counts_per_uuid(conn)
        scores         = _run_scoring(all_analysis, contr_counts)
        rdb.replace_scores(conn, scores)

        # Rebuild relationship graph.
        graph_nodes, graph_edges = _build_graph(all_analysis, all_contrs)
        rdb.replace_graph(conn, graph_nodes, graph_edges)

        _log_audit(args.audit, "contradictions_rerun", {
            "total_pairs": len(all_contrs),
            "flagged": sum(1 for c in all_contrs if c["label"] == "CONTRADICTION"),
        })

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
