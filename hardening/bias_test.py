"""Phase 8 — Bias testing for the Forensiq analysis engine.

Tests that:
  1. Document classification is deterministic (same input → same output).
  2. Scoring does not systematically rank evidence by document type alone —
     scoring should vary with content, not just label.
  3. Entities extracted from neutral text do not skew by protected categories
     (names, locations) in a way that affects reliability scores.

Run from the repo root:

    python -m hardening.bias_test
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Test corpus ───────────────────────────────────────────────────────────────

# Pairs of semantically equivalent statements that differ only in named entities.
# The analysis engine should assign similar reliability scores to both items
# in each pair — any systematic gap signals entity-driven bias.
ENTITY_SWAP_PAIRS = [
    (
        "On January 5, John Smith was seen leaving the building at 10:00 AM.",
        "On January 5, Maria Garcia was seen leaving the building at 10:00 AM.",
    ),
    (
        "The report from City Lab confirms the substance was cocaine.",
        "The report from County Lab confirms the substance was cocaine.",
    ),
    (
        "Witness A stated that the defendant arrived at noon.",
        "Witness B stated that the defendant arrived at noon.",
    ),
]

DOCUMENT_TYPE_SAMPLES = {
    "WITNESS_STATEMENT": (
        "I, Jane Doe, hereby state that on the night of March 3rd I witnessed "
        "two individuals arguing outside 42 Main Street at approximately 11:30 PM."
    ),
    "FORENSIC_REPORT": (
        "Laboratory analysis of sample #F-2024-0031 confirms the presence of "
        "methamphetamine hydrochloride. Chain of custody maintained. Analyst: Dr. Patel."
    ),
    "POLICE_REPORT": (
        "Officers responded to a disturbance call at 0230 hours. Upon arrival, "
        "suspect was found in possession of a concealed firearm. Miranda rights read."
    ),
}

DETERMINISM_SAMPLE = (
    "The accused was positively identified by three independent witnesses "
    "at the scene on April 12th at 9:15 PM."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class BiasResult:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        print(f"  [PASS] {msg}")

    def fail(self, msg: str) -> None:
        self.failed.append(msg)
        print(f"  [FAIL] {msg}")

    def check(self, condition: bool, pass_msg: str, fail_msg: str) -> None:
        (self.ok if condition else self.fail)(pass_msg if condition else fail_msg)


def _classify(text: str) -> tuple[str, float]:
    from analysis.classification import classify_document
    result = classify_document(text)
    return result.document_type.value, result.confidence


def _score_features(text: str, uuid: str = "test-uuid") -> float:
    """Return a rough reliability score for a single text document."""
    from analysis.entities import extract_entities
    from analysis.timeline import extract_timeline
    from reasoning.scoring import EvidenceFeatures, EvidenceScorer

    entities = extract_entities(text)
    timeline = extract_timeline(text)
    doc_type, confidence = _classify(text)

    features = EvidenceFeatures(
        evidence_uuid=uuid,
        document_type=doc_type,
        classification_confidence=confidence,
        entity_count=len(entities),
        entity_label_diversity=len({e.label for e in entities}),
        timeline_event_count=len(timeline),
        timeline_parsed_count=sum(1 for t in timeline if t.timestamp),
        graph_edge_count=len(entities),
        contradiction_count=0,
        uncertain_contradiction_count=0,
    )

    scorer = EvidenceScorer()
    results = scorer.score([features])
    return results[0].score if results else 0.5


# ── Test suites ───────────────────────────────────────────────────────────────

def test_classification_determinism(r: BiasResult) -> None:
    print("\n[1] Classification determinism")
    try:
        doc_type1, conf1 = _classify(DETERMINISM_SAMPLE)
        doc_type2, conf2 = _classify(DETERMINISM_SAMPLE)
        r.check(
            doc_type1 == doc_type2,
            f"Same input → same document type ({doc_type1})",
            f"Non-deterministic classification: {doc_type1} vs {doc_type2}",
        )
        r.check(
            abs(conf1 - conf2) < 0.01,
            f"Same input → stable confidence ({conf1:.3f})",
            f"Unstable confidence: {conf1:.3f} vs {conf2:.3f}",
        )
    except Exception as exc:
        r.fail(f"Classification failed: {exc}")


def test_entity_swap_score_parity(r: BiasResult) -> None:
    """Scores for semantically equivalent statements should be close."""
    print("\n[2] Entity-swap score parity")
    MAX_DELTA = 0.15  # allow up to 15% difference between paired statements
    try:
        for i, (text_a, text_b) in enumerate(ENTITY_SWAP_PAIRS, 1):
            score_a = _score_features(text_a, f"pair-{i}-a")
            score_b = _score_features(text_b, f"pair-{i}-b")
            delta = abs(score_a - score_b)
            r.check(
                delta <= MAX_DELTA,
                f"Pair {i}: score delta {delta:.3f} ≤ {MAX_DELTA} (a={score_a:.3f}, b={score_b:.3f})",
                f"Pair {i}: score delta {delta:.3f} > {MAX_DELTA} — possible entity bias "
                f"(a={score_a:.3f}, b={score_b:.3f})",
            )
    except Exception as exc:
        r.fail(f"Entity-swap test failed: {exc}")


def test_document_type_score_variance(r: BiasResult) -> None:
    """Scores should not be identical across all document types — content must matter."""
    print("\n[3] Document-type score variance")
    try:
        scores = {}
        for doc_type, text in DOCUMENT_TYPE_SAMPLES.items():
            scores[doc_type] = _score_features(text, f"dtype-{doc_type}")
            print(f"    {doc_type}: {scores[doc_type]:.3f}")

        unique_scores = len({round(s, 2) for s in scores.values()})
        r.check(
            unique_scores > 1,
            f"Scores vary across document types ({unique_scores} distinct values)",
            "All document types receive the same score — content not influencing scoring",
        )
    except Exception as exc:
        r.fail(f"Document-type variance test failed: {exc}")


def test_empty_document_handling(r: BiasResult) -> None:
    """Empty or near-empty documents should not crash the pipeline."""
    print("\n[4] Empty document handling")
    for label, text in [("empty string", ""), ("whitespace only", "   \n  "), ("single word", "evidence")]:
        try:
            doc_type, conf = _classify(text)
            score = _score_features(text, f"empty-{label}")
            r.ok(f"{label!r}: classified={doc_type}, score={score:.3f} (no crash)")
        except Exception as exc:
            r.fail(f"{label!r}: raised {type(exc).__name__}: {exc}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Forensiq — Phase 8 Bias Testing")
    print("=" * 60)
    print("(requires analysis + reasoning Python packages)")

    r = BiasResult()

    test_classification_determinism(r)
    test_entity_swap_score_parity(r)
    test_document_type_score_variance(r)
    test_empty_document_handling(r)

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
