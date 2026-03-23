"""Tests for reasoning.bayesian and reasoning.scoring."""
from __future__ import annotations

import math
from typing import List

import pytest

from reasoning.bayesian import (
    BayesianUpdate,
    build_updates,
    compute_posterior,
    prior_for_document_type,
)
from reasoning.scoring import (
    EvidenceFeatures,
    EvidenceScore,
    EvidenceScorer,
    get_scorer,
    reset_scorer,
    set_scorer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_features(
    *,
    uuid: str = "ev-001",
    document_type: str = "FORENSIC_REPORT",
    classification_confidence: float = 0.90,
    entity_count: int = 5,
    entity_label_diversity: int = 3,
    timeline_event_count: int = 2,
    timeline_parsed_count: int = 2,
    graph_edge_count: int = 4,
    contradiction_count: int = 0,
    uncertain_contradiction_count: int = 0,
) -> EvidenceFeatures:
    return EvidenceFeatures(
        evidence_uuid=uuid,
        document_type=document_type,
        classification_confidence=classification_confidence,
        entity_count=entity_count,
        entity_label_diversity=entity_label_diversity,
        timeline_event_count=timeline_event_count,
        timeline_parsed_count=timeline_parsed_count,
        graph_edge_count=graph_edge_count,
        contradiction_count=contradiction_count,
        uncertain_contradiction_count=uncertain_contradiction_count,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton() -> None:  # type: ignore[return]
    reset_scorer()
    yield
    reset_scorer()


# ---------------------------------------------------------------------------
# Tests: prior_for_document_type
# ---------------------------------------------------------------------------


class TestPriorForDocumentType:
    def test_known_types_return_expected_priors(self) -> None:
        assert prior_for_document_type("FORENSIC_REPORT") == pytest.approx(0.80)
        assert prior_for_document_type("COURT_DOCUMENT") == pytest.approx(0.75)
        assert prior_for_document_type("EVIDENCE_LOG") == pytest.approx(0.72)
        assert prior_for_document_type("POLICE_REPORT") == pytest.approx(0.70)
        assert prior_for_document_type("WITNESS_STATEMENT") == pytest.approx(0.60)
        assert prior_for_document_type("OTHER") == pytest.approx(0.50)

    def test_unknown_type_returns_default(self) -> None:
        assert prior_for_document_type("UNKNOWN_CATEGORY") == pytest.approx(0.50)

    def test_case_insensitive(self) -> None:
        assert prior_for_document_type("forensic_report") == pytest.approx(0.80)
        assert prior_for_document_type("Witness_Statement") == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Tests: compute_posterior
# ---------------------------------------------------------------------------


class TestComputePosterior:
    def test_no_updates_returns_prior(self) -> None:
        posterior = compute_posterior(0.70, [])
        assert posterior == pytest.approx(0.70, abs=1e-4)

    def test_lr_above_one_increases_posterior(self) -> None:
        posterior = compute_posterior(0.60, [BayesianUpdate("signal", 2.0)])
        assert posterior > 0.60

    def test_lr_below_one_decreases_posterior(self) -> None:
        posterior = compute_posterior(0.60, [BayesianUpdate("signal", 0.5)])
        assert posterior < 0.60

    def test_multiple_updates_applied_in_sequence(self) -> None:
        updates = [
            BayesianUpdate("a", 2.0),
            BayesianUpdate("b", 0.5),
        ]
        # Net LR = 1.0 → posterior ≈ prior
        posterior = compute_posterior(0.70, updates)
        assert posterior == pytest.approx(0.70, abs=1e-4)

    def test_result_stays_in_unit_interval(self) -> None:
        # Extreme positive evidence
        strong_positive = [BayesianUpdate(f"s{i}", 100.0) for i in range(10)]
        p_high = compute_posterior(0.50, strong_positive)
        assert 0.0 < p_high < 1.0

        # Extreme negative evidence
        strong_negative = [BayesianUpdate(f"s{i}", 0.001) for i in range(10)]
        p_low = compute_posterior(0.50, strong_negative)
        assert 0.0 < p_low < 1.0

    def test_invalid_prior_raises(self) -> None:
        with pytest.raises(ValueError, match="prior"):
            compute_posterior(0.0, [])
        with pytest.raises(ValueError, match="prior"):
            compute_posterior(1.0, [])
        with pytest.raises(ValueError, match="prior"):
            compute_posterior(-0.1, [])

    def test_zero_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="likelihood_ratio"):
            compute_posterior(0.50, [BayesianUpdate("bad", 0.0)])

    def test_negative_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="likelihood_ratio"):
            compute_posterior(0.50, [BayesianUpdate("bad", -1.0)])


# ---------------------------------------------------------------------------
# Tests: build_updates
# ---------------------------------------------------------------------------


class TestBuildUpdates:
    def test_no_signals_returns_only_classification(self) -> None:
        updates = build_updates(
            timeline_parsed_count=0,
            entity_label_diversity=0,
            graph_edge_count=0,
            classification_confidence=0.80,
            contradiction_count=0,
            uncertain_contradiction_count=0,
        )
        # Only the classification_confidence update should be present
        assert len(updates) == 1
        assert "classification_confidence" in updates[0].name

    def test_timeline_events_produce_positive_lr(self) -> None:
        updates = build_updates(
            timeline_parsed_count=2,
            entity_label_diversity=0,
            graph_edge_count=0,
            classification_confidence=0.80,
            contradiction_count=0,
            uncertain_contradiction_count=0,
        )
        timeline_updates = [u for u in updates if "timeline" in u.name]
        assert len(timeline_updates) == 1
        assert timeline_updates[0].likelihood_ratio > 1.0

    def test_contradiction_produces_negative_lr(self) -> None:
        updates = build_updates(
            timeline_parsed_count=0,
            entity_label_diversity=0,
            graph_edge_count=0,
            classification_confidence=0.80,
            contradiction_count=2,
            uncertain_contradiction_count=0,
        )
        contradiction_updates = [u for u in updates if "contradiction_" in u.name and "uncertain" not in u.name]
        assert len(contradiction_updates) == 2
        assert all(u.likelihood_ratio < 1.0 for u in contradiction_updates)

    def test_uncertain_contradiction_produces_mild_negative_lr(self) -> None:
        updates = build_updates(
            timeline_parsed_count=0,
            entity_label_diversity=0,
            graph_edge_count=0,
            classification_confidence=0.80,
            contradiction_count=0,
            uncertain_contradiction_count=1,
        )
        uncertain_updates = [u for u in updates if "uncertain_contradiction" in u.name]
        assert len(uncertain_updates) == 1
        # Uncertain is less severe than confirmed — LR should be closer to 1
        assert 0.5 < uncertain_updates[0].likelihood_ratio < 1.0

    def test_timeline_capped_at_three(self) -> None:
        # 3 and 10 parsed events should produce the same LR
        u3 = build_updates(
            timeline_parsed_count=3,
            entity_label_diversity=0,
            graph_edge_count=0,
            classification_confidence=0.80,
            contradiction_count=0,
            uncertain_contradiction_count=0,
        )
        u10 = build_updates(
            timeline_parsed_count=10,
            entity_label_diversity=0,
            graph_edge_count=0,
            classification_confidence=0.80,
            contradiction_count=0,
            uncertain_contradiction_count=0,
        )
        tl3 = next(u for u in u3 if "timeline" in u.name)
        tl10 = next(u for u in u10 if "timeline" in u.name)
        assert tl3.likelihood_ratio == pytest.approx(tl10.likelihood_ratio)


# ---------------------------------------------------------------------------
# Tests: EvidenceScorer
# ---------------------------------------------------------------------------


class TestEvidenceScorer:
    def test_single_item_rank_is_one(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="ev-001")])
        assert len(results) == 1
        assert results[0].rank == 1

    def test_empty_list_raises(self) -> None:
        scorer = EvidenceScorer()
        with pytest.raises(ValueError, match="empty"):
            scorer.score([])

    def test_blank_uuid_raises(self) -> None:
        scorer = EvidenceScorer()
        with pytest.raises(ValueError, match="uuid"):
            scorer.score([make_features(uuid="")])

    def test_whitespace_uuid_raises(self) -> None:
        scorer = EvidenceScorer()
        with pytest.raises(ValueError, match="uuid"):
            scorer.score([make_features(uuid="   ")])

    def test_result_count_matches_input(self) -> None:
        scorer = EvidenceScorer()
        features = [make_features(uuid=f"ev-{i:03d}") for i in range(5)]
        results = scorer.score(features)
        assert len(results) == 5

    def test_ranks_are_contiguous_from_one(self) -> None:
        scorer = EvidenceScorer()
        features = [make_features(uuid=f"ev-{i:03d}") for i in range(4)]
        results = scorer.score(features)
        ranks = sorted(r.rank for r in results)
        assert ranks == [1, 2, 3, 4]

    def test_sorted_by_descending_score(self) -> None:
        scorer = EvidenceScorer()
        features = [make_features(uuid=f"ev-{i:03d}") for i in range(4)]
        results = scorer.score(features)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_forensic_report_outranks_witness_statement_all_else_equal(self) -> None:
        scorer = EvidenceScorer()
        forensic = make_features(uuid="forensic", document_type="FORENSIC_REPORT")
        witness = make_features(uuid="witness", document_type="WITNESS_STATEMENT")
        results = scorer.score([forensic, witness])
        rank_forensic = next(r.rank for r in results if r.evidence_uuid == "forensic")
        rank_witness = next(r.rank for r in results if r.evidence_uuid == "witness")
        assert rank_forensic < rank_witness  # lower rank number = higher priority

    def test_contradiction_lowers_score(self) -> None:
        scorer = EvidenceScorer()
        clean = make_features(uuid="clean", contradiction_count=0)
        contradicted = make_features(uuid="contradicted", contradiction_count=3)
        results = scorer.score([clean, contradicted])
        score_clean = next(r.score for r in results if r.evidence_uuid == "clean")
        score_contradicted = next(r.score for r in results if r.evidence_uuid == "contradicted")
        assert score_clean > score_contradicted

    def test_score_in_unit_interval(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="ev-001")])
        assert 0.0 < results[0].score < 1.0

    def test_confidence_in_unit_interval(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="ev-001")])
        assert 0.0 < results[0].confidence <= 1.0

    def test_explanation_is_non_empty(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="ev-001")])
        assert results[0].explanation

    def test_explanation_contains_uuid(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="evidence-xyz")])
        assert "evidence-xyz" in results[0].explanation

    def test_contradiction_warning_in_explanation(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="ev", contradiction_count=2)])
        assert "WARNING" in results[0].explanation
        assert "contradiction" in results[0].explanation.lower()

    def test_feature_contributions_populated(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="ev-001")])
        assert results[0].feature_contributions

    def test_feature_contributions_all_positive(self) -> None:
        scorer = EvidenceScorer()
        results = scorer.score([make_features(uuid="ev-001")])
        for name, lr in results[0].feature_contributions.items():
            assert lr > 0.0, f"LR for '{name}' must be positive"

    def test_custom_prior_overrides_default(self) -> None:
        # Give WITNESS_STATEMENT a very high prior
        scorer_custom = EvidenceScorer(custom_priors={"WITNESS_STATEMENT": 0.95})
        scorer_default = EvidenceScorer()
        f = make_features(uuid="ev", document_type="WITNESS_STATEMENT")
        score_custom = scorer_custom.score([f])[0].score
        score_default = scorer_default.score([f])[0].score
        assert score_custom > score_default

    def test_sparse_features_produce_lower_confidence(self) -> None:
        scorer = EvidenceScorer()
        sparse = make_features(
            uuid="sparse",
            entity_count=0,
            entity_label_diversity=0,
            timeline_event_count=0,
            timeline_parsed_count=0,
            graph_edge_count=0,
            classification_confidence=0.30,
        )
        rich = make_features(uuid="rich")
        results = scorer.score([sparse, rich])
        conf_sparse = next(r.confidence for r in results if r.evidence_uuid == "sparse")
        conf_rich = next(r.confidence for r in results if r.evidence_uuid == "rich")
        assert conf_sparse < conf_rich

    def test_high_timeline_and_entities_produce_high_score(self) -> None:
        scorer = EvidenceScorer()
        rich = make_features(
            uuid="rich",
            document_type="FORENSIC_REPORT",
            classification_confidence=0.98,
            entity_count=20,
            entity_label_diversity=5,
            timeline_parsed_count=3,
            graph_edge_count=10,
            contradiction_count=0,
        )
        results = scorer.score([rich])
        assert results[0].score > 0.85

    def test_many_contradictions_produce_low_score(self) -> None:
        scorer = EvidenceScorer()
        bad = make_features(
            uuid="bad",
            document_type="WITNESS_STATEMENT",
            classification_confidence=0.50,
            entity_count=1,
            entity_label_diversity=1,
            timeline_parsed_count=0,
            graph_edge_count=0,
            contradiction_count=5,
        )
        results = scorer.score([bad])
        assert results[0].score < 0.40


# ---------------------------------------------------------------------------
# Tests: singleton helpers
# ---------------------------------------------------------------------------


class TestSingletonHelpers:
    def test_set_and_get_scorer(self) -> None:
        scorer = EvidenceScorer()
        set_scorer(scorer)
        assert get_scorer() is scorer

    def test_reset_clears_singleton(self) -> None:
        scorer = EvidenceScorer()
        set_scorer(scorer)
        reset_scorer()
        new_scorer = EvidenceScorer()
        set_scorer(new_scorer)
        assert get_scorer() is new_scorer

    def test_get_scorer_creates_default_when_none(self) -> None:
        reset_scorer()
        scorer = get_scorer()
        assert isinstance(scorer, EvidenceScorer)
