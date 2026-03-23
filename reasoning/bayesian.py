"""Bayesian reliability updating for evidence items.

Each piece of evidence starts with a prior reliability probability derived
from its document type.  Observed signals (timeline coherence, entity
richness, contradiction flags, graph connectivity, classification confidence)
are expressed as likelihood ratios and applied sequentially via Bayes' rule
in log-odds form.

No verdict probabilities are computed.  The output is a reliability score
in [0, 1] that informs evidence prioritization only.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple


# ---------------------------------------------------------------------------
# Document-type priors
# ---------------------------------------------------------------------------

#: Prior reliability probability per document type (OntoNotes / legal domain
#: heuristics). Higher values mean the category is typically more reliable as
#: a starting point before any signal is applied.
DOCUMENT_TYPE_PRIORS: Dict[str, float] = {
    "FORENSIC_REPORT": 0.80,
    "COURT_DOCUMENT": 0.75,
    "EVIDENCE_LOG": 0.72,
    "POLICE_REPORT": 0.70,
    "WITNESS_STATEMENT": 0.60,
    "OTHER": 0.50,
}

_DEFAULT_PRIOR: float = 0.50


# ---------------------------------------------------------------------------
# Core Bayesian update
# ---------------------------------------------------------------------------

class BayesianUpdate(NamedTuple):
    """One named signal applied during a Bayesian update chain.

    Attributes
    ----------
    name:
        Human-readable signal name used in explanations.
    likelihood_ratio:
        P(signal | reliable) / P(signal | unreliable).
        Values > 1 increase the posterior; < 1 decrease it.
    """

    name: str
    likelihood_ratio: float


def prior_for_document_type(document_type: str) -> float:
    """Return the prior reliability probability for *document_type*.

    Parameters
    ----------
    document_type:
        One of the ``DocumentType`` enum names (e.g. ``"FORENSIC_REPORT"``).
        Unknown types receive the default prior of 0.50.

    Returns
    -------
    float
        Prior in (0, 1).
    """
    return DOCUMENT_TYPE_PRIORS.get(document_type.upper(), _DEFAULT_PRIOR)


def compute_posterior(prior: float, updates: List[BayesianUpdate]) -> float:
    """Apply a chain of likelihood ratios to *prior* and return the posterior.

    Uses log-odds form to avoid floating-point underflow:

    .. code-block:: text

        log_odds(posterior) = log_odds(prior) + Σ log(LR_i)

    Parameters
    ----------
    prior:
        Starting reliability probability in (0, 1).
    updates:
        Ordered list of :class:`BayesianUpdate` instances.  An empty list
        returns the prior unchanged.

    Returns
    -------
    float
        Posterior in (0, 1), clamped away from exact 0 and 1 to keep the
        log-odds numerically stable.

    Raises
    ------
    ValueError
        If *prior* is not in (0, 1) or any likelihood ratio is ≤ 0.
    """
    if not (0.0 < prior < 1.0):
        raise ValueError(f"prior must be in (0, 1), got {prior!r}")
    for u in updates:
        if u.likelihood_ratio <= 0.0:
            raise ValueError(
                f"likelihood_ratio must be > 0, got {u.likelihood_ratio!r} "
                f"for signal '{u.name}'"
            )

    import math

    log_odds = math.log(prior / (1.0 - prior))
    for u in updates:
        log_odds += math.log(u.likelihood_ratio)

    # Clamp to avoid overflow in exp()
    log_odds = max(-20.0, min(20.0, log_odds))
    posterior = 1.0 / (1.0 + math.exp(-log_odds))
    # Keep away from exact endpoints for numerical stability
    return max(1e-6, min(1.0 - 1e-6, posterior))


def build_updates(
    *,
    timeline_parsed_count: int,
    entity_label_diversity: int,
    graph_edge_count: int,
    classification_confidence: float,
    contradiction_count: int,
    uncertain_contradiction_count: int,
) -> List[BayesianUpdate]:
    """Construct the standard :class:`BayesianUpdate` chain for one evidence item.

    Each signal is mapped to a likelihood ratio:

    - **Timeline coherence**: +LR per parsed timestamp (capped at 3 events).
    - **Entity richness**: +LR per unique NER label (capped at 5 labels).
    - **Graph connectivity**: +LR per evidence relationship edge (capped at 10).
    - **Classification confidence**: LR proportional to model confidence.
    - **Contradiction flags**: −LR per confirmed contradiction.
    - **Uncertain contradictions**: mild −LR per uncertain flag.

    Parameters
    ----------
    timeline_parsed_count:
        Number of timeline events with a successfully parsed timestamp.
    entity_label_diversity:
        Number of distinct NER label types found in the document.
    graph_edge_count:
        Number of relationship edges in the co-occurrence graph.
    classification_confidence:
        Model confidence for the document type classification (0–1).
    contradiction_count:
        Number of confirmed (CONTRADICTION-label) findings involving this item.
    uncertain_contradiction_count:
        Number of UNCERTAIN-label findings involving this item.

    Returns
    -------
    List[BayesianUpdate]
    """
    updates: List[BayesianUpdate] = []

    # Timeline coherence: each parsed timestamp is evidence of a structured,
    # reliable document.
    parsed = min(timeline_parsed_count, 3)
    if parsed > 0:
        updates.append(
            BayesianUpdate(
                name=f"timeline_coherence({parsed}_events)",
                likelihood_ratio=1.25 ** parsed,
            )
        )

    # Entity richness: diverse entity labels (PERSON, ORG, GPE…) indicate
    # a well-evidenced, factually dense document.
    diversity = min(entity_label_diversity, 5)
    if diversity > 0:
        updates.append(
            BayesianUpdate(
                name=f"entity_diversity({diversity}_labels)",
                likelihood_ratio=1.0 + 0.06 * diversity,
            )
        )

    # Graph connectivity: documents that share many named entities with other
    # evidence items are more corroborated.
    edges = min(graph_edge_count, 10)
    if edges > 0:
        updates.append(
            BayesianUpdate(
                name=f"graph_connectivity({edges}_edges)",
                likelihood_ratio=1.0 + 0.025 * edges,
            )
        )

    # Classification confidence: high-confidence document classification
    # supports the document being what it claims to be.
    lr_classification = 0.75 + 0.50 * max(0.0, min(1.0, classification_confidence))
    updates.append(
        BayesianUpdate(
            name=f"classification_confidence({classification_confidence:.3f})",
            likelihood_ratio=lr_classification,
        )
    )

    # Confirmed contradictions: each confirmed conflict substantially lowers
    # our confidence in the evidence reliability.
    for i in range(contradiction_count):
        updates.append(
            BayesianUpdate(
                name=f"contradiction_{i + 1}",
                likelihood_ratio=0.45,
            )
        )

    # Uncertain contradictions: mild downward pressure.
    for i in range(uncertain_contradiction_count):
        updates.append(
            BayesianUpdate(
                name=f"uncertain_contradiction_{i + 1}",
                likelihood_ratio=0.75,
            )
        )

    return updates
