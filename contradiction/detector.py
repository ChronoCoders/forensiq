"""Contradiction detection between evidence statement pairs.

Uses ``cross-encoder/nli-deberta-v3-base`` to classify each pair as
CONTRADICTION, NEUTRAL, or UNCERTAIN.

Rules
-----
- Confidence threshold: 0.85.  A raw CONTRADICTION prediction below the
  threshold is reported as UNCERTAIN rather than suppressed.
- ENTAILMENT predictions are normalised to NEUTRAL (consistent statements).
- Every result carries a non-empty explanation naming the specific claims
  that conflict (or why the result is uncertain/neutral).
- Batch-only model loading: the model is loaded once via :func:`get_detector`
  and reused for all subsequent calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-base"
_THRESHOLD: float = 0.85

_detector: Optional["ContradictionDetector"] = None


@dataclass
class StatementPair:
    """A pair of evidence statements to compare for contradictions.

    Attributes
    ----------
    evidence_uuid_a:
        UUID of the first evidence item.
    evidence_uuid_b:
        UUID of the second evidence item.
    text_a:
        Claim or statement extracted from evidence A.
    text_b:
        Claim or statement extracted from evidence B.
    context:
        Aspect being compared: ``timeline``, ``location``, ``person``, ``event``.
    """

    evidence_uuid_a: str
    evidence_uuid_b: str
    text_a: str
    text_b: str
    context: str


@dataclass
class ContradictionResult:
    """Result of comparing one :class:`StatementPair`.

    Attributes
    ----------
    evidence_uuid_a:
        UUID of the first evidence item (passed through from input).
    evidence_uuid_b:
        UUID of the second evidence item (passed through from input).
    label:
        ``CONTRADICTION`` — confident conflict (confidence ≥ threshold).
        ``UNCERTAIN``     — potential conflict below confidence threshold.
        ``NEUTRAL``       — statements are consistent or unrelated.
    confidence:
        Softmax probability of the predicted raw NLI class (0.0–1.0).
    context:
        Aspect that was compared (passed through from input).
    explanation:
        Human-readable description of the finding, naming the specific
        claims that conflict or explaining why the result is uncertain.
    """

    evidence_uuid_a: str
    evidence_uuid_b: str
    label: str
    confidence: float
    context: str
    explanation: str


class ContradictionDetector:
    """Cross-encoder NLI detector for evidence statement pairs.

    Parameters
    ----------
    model:
        HuggingFace ``AutoModelForSequenceClassification`` instance.
        If ``None``, loads ``cross-encoder/nli-deberta-v3-base`` at init time.
    tokenizer:
        Matching tokenizer.  If ``None``, loaded alongside the model.
    threshold:
        Minimum confidence for a CONTRADICTION prediction to be reported as
        CONTRADICTION rather than UNCERTAIN.  Defaults to 0.85.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        threshold: float = _THRESHOLD,
    ) -> None:
        if model is None or tokenizer is None:
            from transformers import (  # noqa: PLC0415
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                _MODEL_NAME
            )
            self._model.eval()
        else:
            self._tokenizer = tokenizer
            self._model = model

        # Read label mapping from model config; fall back to standard NLI order.
        id2label: dict[int, str] = getattr(
            getattr(self._model, "config", None), "id2label", {}
        ) or {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}
        self._id2label: dict[int, str] = {
            int(k): str(v).upper() for k, v in id2label.items()
        }
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, pair: StatementPair) -> ContradictionResult:
        """Classify one :class:`StatementPair`.

        Parameters
        ----------
        pair:
            The two statements to compare.

        Returns
        -------
        ContradictionResult

        Raises
        ------
        ValueError
            If either ``text_a`` or ``text_b`` is empty or whitespace-only,
            or if ``evidence_uuid_a`` / ``evidence_uuid_b`` are empty.
        """
        _validate_pair(pair)

        inputs = self._tokenizer(
            pair.text_a,
            pair.text_b,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs: torch.Tensor = F.softmax(outputs.logits, dim=-1)[0]
        predicted_idx = int(probs.argmax().item())
        raw_label = self._id2label.get(predicted_idx, "NEUTRAL")
        confidence = float(probs[predicted_idx].item())

        label = _apply_threshold(raw_label, confidence, self.threshold)
        explanation = _build_explanation(pair, label, raw_label, confidence, self.threshold)

        return ContradictionResult(
            evidence_uuid_a=pair.evidence_uuid_a,
            evidence_uuid_b=pair.evidence_uuid_b,
            label=label,
            confidence=confidence,
            context=pair.context,
            explanation=explanation,
        )

    def detect_batch(self, pairs: List[StatementPair]) -> List[ContradictionResult]:
        """Classify a batch of statement pairs.

        Parameters
        ----------
        pairs:
            Non-empty list of :class:`StatementPair` instances.

        Returns
        -------
        List[ContradictionResult]
            One result per input pair, in the same order.

        Raises
        ------
        ValueError
            If *pairs* is empty, or if any pair contains empty text fields.
        """
        if not pairs:
            raise ValueError("pairs must not be empty")
        return [self.detect(pair) for pair in pairs]


# ------------------------------------------------------------------
# Module-level singleton helpers
# ------------------------------------------------------------------

def get_detector() -> ContradictionDetector:
    """Return the shared :class:`ContradictionDetector`, loading on first call."""
    global _detector
    if _detector is None:
        _detector = ContradictionDetector()
    return _detector


def set_detector(detector: ContradictionDetector) -> None:
    """Inject a custom detector.  Used in tests to avoid model downloads."""
    global _detector
    _detector = detector


def reset_detector() -> None:
    """Clear the cached detector so the next :func:`get_detector` call reloads."""
    global _detector
    _detector = None


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _validate_pair(pair: StatementPair) -> None:
    if not pair.text_a or not pair.text_a.strip():
        raise ValueError("StatementPair.text_a must be non-empty")
    if not pair.text_b or not pair.text_b.strip():
        raise ValueError("StatementPair.text_b must be non-empty")
    if not pair.evidence_uuid_a or not pair.evidence_uuid_a.strip():
        raise ValueError("StatementPair.evidence_uuid_a must be non-empty")
    if not pair.evidence_uuid_b or not pair.evidence_uuid_b.strip():
        raise ValueError("StatementPair.evidence_uuid_b must be non-empty")


def _apply_threshold(raw_label: str, confidence: float, threshold: float) -> str:
    """Map raw NLI label + confidence to final CONTRADICTION/UNCERTAIN/NEUTRAL."""
    if raw_label == "CONTRADICTION":
        return "CONTRADICTION" if confidence >= threshold else "UNCERTAIN"
    # ENTAILMENT means the statements support each other — report as NEUTRAL.
    return "NEUTRAL"


def _build_explanation(
    pair: StatementPair,
    label: str,
    raw_label: str,
    confidence: float,
    threshold: float,
) -> str:
    snippet_a = pair.text_a[:120].rstrip() + ("…" if len(pair.text_a) > 120 else "")
    snippet_b = pair.text_b[:120].rstrip() + ("…" if len(pair.text_b) > 120 else "")

    if label == "CONTRADICTION":
        return (
            f"CONTRADICTION detected in context '{pair.context}' "
            f"(confidence {confidence:.3f} ≥ threshold {threshold}). "
            f"Claim A: \"{snippet_a}\" conflicts with "
            f"Claim B: \"{snippet_b}\"."
        )
    if label == "UNCERTAIN":
        return (
            f"Potential conflict in context '{pair.context}' flagged as UNCERTAIN "
            f"(confidence {confidence:.3f} < threshold {threshold}). "
            f"Raw model prediction: {raw_label}. Requires manual review. "
            f"Claim A: \"{snippet_a}\" — Claim B: \"{snippet_b}\"."
        )
    return (
        f"Statements are consistent in context '{pair.context}' "
        f"(confidence {confidence:.3f}). "
        f"No actionable conflict detected. "
        f"Raw model prediction: {raw_label}."
    )
