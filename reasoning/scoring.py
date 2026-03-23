"""Evidence prioritization scoring for Forensiq.

:class:`EvidenceScorer` accepts a list of :class:`EvidenceFeatures` (one per
evidence item), runs a Bayesian reliability update for each, and returns a
ranked :class:`EvidenceScore` list ordered from most to least reliable.

Design constraints
------------------
- No verdict probabilities — scores represent *reliability priority* only.
- Every score ships with an explanation vector so analysts can see exactly
  which signals drove the ranking.
- Scores are deterministic for the same inputs; no randomness.
- Module-level singleton pattern mirrors analysis and contradiction packages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from reasoning.bayesian import (
    BayesianUpdate,
    build_updates,
    compute_posterior,
    prior_for_document_type,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvidenceFeatures:
    """All analysis signals for one evidence item.

    Attributes
    ----------
    evidence_uuid:
        UUID of the evidence item (must be non-empty).
    document_type:
        Classified document type string (e.g. ``"FORENSIC_REPORT"``).
    classification_confidence:
        Confidence in the document type classification (0–1).
    entity_count:
        Total number of named entities extracted.
    entity_label_diversity:
        Number of distinct NER label types (PERSON, ORG, GPE, …).
    timeline_event_count:
        Total timeline events found (parsed + unparsed).
    timeline_parsed_count:
        Subset of timeline events with a successfully parsed timestamp.
    graph_edge_count:
        Number of relationship edges in the co-occurrence graph.
    contradiction_count:
        Confirmed CONTRADICTION findings involving this evidence item.
    uncertain_contradiction_count:
        UNCERTAIN contradiction findings involving this evidence item.
    """

    evidence_uuid: str
    document_type: str
    classification_confidence: float
    entity_count: int
    entity_label_diversity: int
    timeline_event_count: int
    timeline_parsed_count: int
    graph_edge_count: int
    contradiction_count: int
    uncertain_contradiction_count: int


@dataclass
class EvidenceScore:
    """Prioritization result for one evidence item.

    Attributes
    ----------
    evidence_uuid:
        Passed through from :class:`EvidenceFeatures`.
    rank:
        Position in the priority list (1 = highest priority).
    score:
        Posterior reliability probability in (0, 1).  Higher is more reliable.
    confidence:
        Confidence in the score itself, derived from data completeness (0–1).
    explanation:
        Human-readable narrative of the key signals that drove the score.
    feature_contributions:
        Mapping of signal name → likelihood ratio for full auditability.
    """

    evidence_uuid: str
    rank: int
    score: float
    confidence: float
    explanation: str
    feature_contributions: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class EvidenceScorer:
    """Rank evidence items by Bayesian reliability score.

    Parameters
    ----------
    custom_priors:
        Optional override mapping of document type → prior probability.
        Merged with the default :data:`~reasoning.bayesian.DOCUMENT_TYPE_PRIORS`.
    """

    def __init__(self, custom_priors: Optional[Dict[str, float]] = None) -> None:
        self._custom_priors: Dict[str, float] = custom_priors or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, features_list: List[EvidenceFeatures]) -> List[EvidenceScore]:
        """Score and rank a list of evidence items.

        Parameters
        ----------
        features_list:
            One :class:`EvidenceFeatures` per evidence item.  Order does not
            matter — the returned list is sorted by descending score.

        Returns
        -------
        List[EvidenceScore]
            Ranked list, ``rank=1`` being the highest-priority item.

        Raises
        ------
        ValueError
            If *features_list* is empty, or if any ``evidence_uuid`` is blank.
        """
        if not features_list:
            raise ValueError("features_list must not be empty")
        for f in features_list:
            if not f.evidence_uuid or not f.evidence_uuid.strip():
                raise ValueError("evidence_uuid must be non-empty")

        scored = [self._score_one(f) for f in features_list]
        scored.sort(key=lambda s: s.score, reverse=True)
        for rank, s in enumerate(scored, start=1):
            # dataclass fields are mutable — safe to assign rank after sort
            object.__setattr__(s, "rank", rank)  # works for regular dataclass too
            s.rank = rank
        return scored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prior_for(self, document_type: str) -> float:
        key = document_type.upper()
        if key in self._custom_priors:
            return self._custom_priors[key]
        return prior_for_document_type(key)

    def _score_one(self, f: EvidenceFeatures) -> EvidenceScore:
        prior = self._prior_for(f.document_type)
        updates: List[BayesianUpdate] = build_updates(
            timeline_parsed_count=f.timeline_parsed_count,
            entity_label_diversity=f.entity_label_diversity,
            graph_edge_count=f.graph_edge_count,
            classification_confidence=f.classification_confidence,
            contradiction_count=f.contradiction_count,
            uncertain_contradiction_count=f.uncertain_contradiction_count,
        )

        posterior = compute_posterior(prior, updates)
        confidence = _data_completeness(f)
        explanation = _build_explanation(f, prior, posterior, updates)
        contributions = {u.name: u.likelihood_ratio for u in updates}

        return EvidenceScore(
            evidence_uuid=f.evidence_uuid,
            rank=0,  # assigned after sort
            score=posterior,
            confidence=confidence,
            explanation=explanation,
            feature_contributions=contributions,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scorer: Optional[EvidenceScorer] = None


def get_scorer() -> EvidenceScorer:
    """Return the shared :class:`EvidenceScorer`, creating it on first call."""
    global _scorer
    if _scorer is None:
        _scorer = EvidenceScorer()
    return _scorer


def set_scorer(scorer: EvidenceScorer) -> None:
    """Inject a custom scorer.  Used in tests."""
    global _scorer
    _scorer = scorer


def reset_scorer() -> None:
    """Clear the cached scorer."""
    global _scorer
    _scorer = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _data_completeness(f: EvidenceFeatures) -> float:
    """Estimate confidence in the score based on how many signals are present.

    A score derived from a rich feature set (entities, timeline, graph,
    classification) is more trustworthy than one computed from sparse data.
    Returns a value in (0, 1].
    """
    points = 0.0
    total = 5.0

    if f.entity_count > 0:
        points += 1.0
    if f.entity_label_diversity > 1:
        points += 0.5
    if f.timeline_event_count > 0:
        points += 1.0
    if f.timeline_parsed_count > 0:
        points += 1.0
    if f.graph_edge_count > 0:
        points += 1.0
    if f.classification_confidence >= 0.5:
        points += 0.5

    return max(0.10, min(1.0, points / total))


def _build_explanation(
    f: EvidenceFeatures,
    prior: float,
    posterior: float,
    updates: List[BayesianUpdate],
) -> str:
    """Produce a human-readable explanation of the score."""
    lines: List[str] = [
        f"Evidence {f.evidence_uuid} — document type: {f.document_type} "
        f"(prior reliability {prior:.2f}).",
        f"Posterior reliability score: {posterior:.3f}.",
    ]

    positive = [u for u in updates if u.likelihood_ratio >= 1.0]
    negative = [u for u in updates if u.likelihood_ratio < 1.0]

    if positive:
        pos_names = ", ".join(u.name for u in positive)
        lines.append(f"Positive signals: {pos_names}.")
    if negative:
        neg_names = ", ".join(u.name for u in negative)
        lines.append(f"Downward signals: {neg_names}.")
    if not positive and not negative:
        lines.append("No signals observed; score equals prior.")

    if f.contradiction_count > 0:
        lines.append(
            f"WARNING: {f.contradiction_count} confirmed contradiction(s) detected — "
            "manual review required before relying on this evidence."
        )
    if f.uncertain_contradiction_count > 0:
        lines.append(
            f"NOTE: {f.uncertain_contradiction_count} uncertain contradiction(s) flagged."
        )

    return " ".join(lines)
