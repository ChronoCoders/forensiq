"""Tests for contradiction.detector.

Mock strategy
-------------
The CrossEncoder model is replaced with a MagicMock that returns pre-set
``logits`` tensors, so no model weights are downloaded during tests.

Label mapping used by mock models (standard NLI order):
  0 → CONTRADICTION
  1 → ENTAILMENT
  2 → NEUTRAL
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest
import torch

from contradiction.detector import (
    ContradictionDetector,
    ContradictionResult,
    StatementPair,
    get_detector,
    reset_detector,
    set_detector,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ID2LABEL = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}


def make_contradiction_detector(logits: List[float], threshold: float = 0.85) -> ContradictionDetector:
    """Return a :class:`ContradictionDetector` backed by a mock cross-encoder.

    Parameters
    ----------
    logits:
        Raw logit values ``[contradiction, entailment, neutral]`` returned by
        the mock model.  Softmax is applied by the real detector code.
    threshold:
        Confidence threshold forwarded to the detector (default 0.85).
    """
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }

    outputs = MagicMock()
    outputs.logits = torch.tensor([logits], dtype=torch.float)

    model = MagicMock()
    model.return_value = outputs
    model.config = MagicMock()
    model.config.id2label = _ID2LABEL

    return ContradictionDetector(model=model, tokenizer=tokenizer, threshold=threshold)


def _pair(
    text_a: str,
    text_b: str,
    context: str = "location",
    uuid_a: str = "uuid-a",
    uuid_b: str = "uuid-b",
) -> StatementPair:
    return StatementPair(
        evidence_uuid_a=uuid_a,
        evidence_uuid_b=uuid_b,
        text_a=text_a,
        text_b=text_b,
        context=context,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton() -> None:  # type: ignore[return]
    """Ensure the module-level detector singleton is cleared between tests."""
    reset_detector()
    yield
    reset_detector()


# ---------------------------------------------------------------------------
# Scenario 1: CONTRADICTION — same person, different locations, same time
# ---------------------------------------------------------------------------

class TestContradictionScenario:
    """Scenario 1: statements placing same person in different locations → CONTRADICTION."""

    def test_returns_contradiction_label(self) -> None:
        # High logit on index 0 (CONTRADICTION) → softmax well above 0.85
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = _pair(
            "Witness A states John Smith was at the warehouse at 9 PM.",
            "Witness B states John Smith was at the airport at 9 PM.",
            context="location",
        )
        result = detector.detect(pair)
        assert result.label == "CONTRADICTION"

    def test_confidence_at_or_above_threshold(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = _pair(
            "Witness A states John Smith was at the warehouse at 9 PM.",
            "Witness B states John Smith was at the airport at 9 PM.",
            context="location",
        )
        result = detector.detect(pair)
        assert result.confidence >= 0.85

    def test_explanation_names_conflicting_claims(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = _pair(
            "Witness A states John Smith was at the warehouse at 9 PM.",
            "Witness B states John Smith was at the airport at 9 PM.",
            context="location",
        )
        result = detector.detect(pair)
        assert result.explanation, "explanation must be non-empty"
        assert "CONTRADICTION" in result.explanation
        assert "location" in result.explanation

    def test_uuids_passed_through(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = _pair(
            "Smith was downtown.",
            "Smith was uptown.",
            uuid_a="ev-001",
            uuid_b="ev-002",
        )
        result = detector.detect(pair)
        assert result.evidence_uuid_a == "ev-001"
        assert result.evidence_uuid_b == "ev-002"

    def test_context_passed_through(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = _pair("Smith was downtown.", "Smith was uptown.", context="timeline")
        result = detector.detect(pair)
        assert result.context == "timeline"


# ---------------------------------------------------------------------------
# Scenario 2: NEUTRAL — statements consistent with each other
# ---------------------------------------------------------------------------

class TestNeutralScenario:
    """Scenario 2: consistent statements → NEUTRAL."""

    def test_entailment_normalised_to_neutral(self) -> None:
        # High logit on index 1 (ENTAILMENT) → must be reported as NEUTRAL
        detector = make_contradiction_detector([-5.0, 10.0, -5.0])
        pair = _pair(
            "The victim was found at Central Park on Monday morning.",
            "The body was discovered in Central Park early Monday.",
            context="location",
        )
        result = detector.detect(pair)
        assert result.label == "NEUTRAL"

    def test_neutral_raw_label_reported_as_neutral(self) -> None:
        # High logit on index 2 (NEUTRAL)
        detector = make_contradiction_detector([-5.0, -5.0, 10.0])
        pair = _pair(
            "Officer Jones filed the report on Tuesday.",
            "The report was submitted by Officer Jones on Tuesday.",
            context="event",
        )
        result = detector.detect(pair)
        assert result.label == "NEUTRAL"

    def test_neutral_explanation_non_empty(self) -> None:
        detector = make_contradiction_detector([-5.0, -5.0, 10.0])
        pair = _pair("Jones arrived at noon.", "Jones was present at midday.", context="timeline")
        result = detector.detect(pair)
        assert result.explanation, "explanation must be non-empty"

    def test_neutral_explanation_mentions_context(self) -> None:
        detector = make_contradiction_detector([-5.0, -5.0, 10.0])
        pair = _pair("Jones arrived at noon.", "Jones was present at midday.", context="timeline")
        result = detector.detect(pair)
        assert "timeline" in result.explanation

    def test_neutral_confidence_populated(self) -> None:
        detector = make_contradiction_detector([-5.0, -5.0, 10.0])
        pair = _pair("A.", "B.", context="event")
        result = detector.detect(pair)
        assert 0.0 < result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Scenario 3: UNCERTAIN — ambiguous, below confidence threshold
# ---------------------------------------------------------------------------

class TestUncertainScenario:
    """Scenario 3: potential conflict below confidence threshold → UNCERTAIN."""

    def test_low_confidence_contradiction_becomes_uncertain(self) -> None:
        # logits that produce softmax ≈ 0.57 for CONTRADICTION — below 0.85
        detector = make_contradiction_detector([2.0, 0.5, 0.5], threshold=0.85)
        pair = _pair(
            "The suspect may have been near the station around midnight.",
            "The suspect was possibly seen elsewhere that evening.",
            context="location",
        )
        result = detector.detect(pair)
        assert result.label == "UNCERTAIN"

    def test_uncertain_confidence_below_threshold(self) -> None:
        detector = make_contradiction_detector([2.0, 0.5, 0.5], threshold=0.85)
        pair = _pair(
            "The suspect may have been near the station around midnight.",
            "The suspect was possibly seen elsewhere that evening.",
            context="location",
        )
        result = detector.detect(pair)
        assert result.confidence < 0.85

    def test_uncertain_explanation_flags_manual_review(self) -> None:
        detector = make_contradiction_detector([2.0, 0.5, 0.5])
        pair = _pair(
            "The suspect may have been near the station around midnight.",
            "The suspect was possibly seen elsewhere that evening.",
            context="location",
        )
        result = detector.detect(pair)
        assert result.explanation, "explanation must be non-empty"
        assert "UNCERTAIN" in result.explanation

    def test_uncertain_explanation_includes_raw_label(self) -> None:
        detector = make_contradiction_detector([2.0, 0.5, 0.5])
        pair = _pair("Maybe here.", "Maybe there.", context="location")
        result = detector.detect(pair)
        # Raw model predicted CONTRADICTION — that must appear in the explanation
        assert "CONTRADICTION" in result.explanation

    def test_custom_threshold_respected(self) -> None:
        # With threshold=0.50 and confidence ≈ 0.57, should now be CONTRADICTION
        detector = make_contradiction_detector([2.0, 0.5, 0.5], threshold=0.50)
        pair = _pair("Smith was here.", "Smith was there.", context="location")
        result = detector.detect(pair)
        assert result.label == "CONTRADICTION"


# ---------------------------------------------------------------------------
# Scenario 4: Empty / malformed input → ValueError
# ---------------------------------------------------------------------------

class TestMalformedInput:
    """Scenario 4: empty or malformed input raises ValueError with clear message."""

    def test_empty_text_a_raises(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = StatementPair(
            evidence_uuid_a="uuid-a",
            evidence_uuid_b="uuid-b",
            text_a="",
            text_b="Some valid statement.",
            context="event",
        )
        with pytest.raises(ValueError, match="text_a"):
            detector.detect(pair)

    def test_whitespace_text_a_raises(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = StatementPair(
            evidence_uuid_a="uuid-a",
            evidence_uuid_b="uuid-b",
            text_a="   ",
            text_b="Some valid statement.",
            context="event",
        )
        with pytest.raises(ValueError, match="text_a"):
            detector.detect(pair)

    def test_empty_text_b_raises(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = StatementPair(
            evidence_uuid_a="uuid-a",
            evidence_uuid_b="uuid-b",
            text_a="Some valid statement.",
            text_b="",
            context="event",
        )
        with pytest.raises(ValueError, match="text_b"):
            detector.detect(pair)

    def test_whitespace_text_b_raises(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = StatementPair(
            evidence_uuid_a="uuid-a",
            evidence_uuid_b="uuid-b",
            text_a="Some valid statement.",
            text_b="\t\n",
            context="event",
        )
        with pytest.raises(ValueError, match="text_b"):
            detector.detect(pair)

    def test_empty_uuid_a_raises(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = StatementPair(
            evidence_uuid_a="",
            evidence_uuid_b="uuid-b",
            text_a="Valid A.",
            text_b="Valid B.",
            context="event",
        )
        with pytest.raises(ValueError, match="uuid_a"):
            detector.detect(pair)

    def test_empty_uuid_b_raises(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pair = StatementPair(
            evidence_uuid_a="uuid-a",
            evidence_uuid_b="",
            text_a="Valid A.",
            text_b="Valid B.",
            context="event",
        )
        with pytest.raises(ValueError, match="uuid_b"):
            detector.detect(pair)

    def test_empty_batch_raises(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        with pytest.raises(ValueError, match="empty"):
            detector.detect_batch([])


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------

class TestSingletonHelpers:
    """set_detector / get_detector / reset_detector behave correctly."""

    def test_set_and_get_detector(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        set_detector(detector)
        assert get_detector() is detector

    def test_reset_clears_singleton(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        set_detector(detector)
        reset_detector()
        # get_detector() would try to load the real model — just verify the
        # internal state was cleared by calling set_detector again.
        new_detector = make_contradiction_detector([-5.0, -5.0, 10.0])
        set_detector(new_detector)
        assert get_detector() is new_detector


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

class TestBatchDetection:
    """detect_batch returns one result per input pair in order."""

    def test_batch_length_matches_input(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pairs = [
            _pair("A1.", "B1.", uuid_a="ev-1", uuid_b="ev-2"),
            _pair("A2.", "B2.", uuid_a="ev-3", uuid_b="ev-4"),
            _pair("A3.", "B3.", uuid_a="ev-5", uuid_b="ev-6"),
        ]
        results = detector.detect_batch(pairs)
        assert len(results) == 3

    def test_batch_order_preserved(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pairs = [
            _pair("A.", "B.", uuid_a="first-a", uuid_b="first-b"),
            _pair("C.", "D.", uuid_a="second-a", uuid_b="second-b"),
        ]
        results = detector.detect_batch(pairs)
        assert results[0].evidence_uuid_a == "first-a"
        assert results[1].evidence_uuid_a == "second-a"

    def test_batch_returns_contradiction_results(self) -> None:
        detector = make_contradiction_detector([10.0, -5.0, -5.0])
        pairs = [_pair("A.", "B."), _pair("C.", "D.")]
        results = detector.detect_batch(pairs)
        assert all(isinstance(r, ContradictionResult) for r in results)
