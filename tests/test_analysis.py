"""Tests for the Forensiq analysis engine (Phase 3).

Each module is covered:
- entities.py  : extract_entities
- timeline.py  : extract_timeline
- graph.py     : build_relationship_graph
- classification.py : classify_document

All tests run against the mock ModelRegistry injected by conftest.py.
"""
from __future__ import annotations

import pytest

from analysis.classification import DocumentType, classify_document
from analysis.entities import ExtractedEntity, extract_entities
from analysis.graph import build_relationship_graph
from analysis.models import ModelRegistry, set_models
from analysis.timeline import extract_timeline
from tests.conftest import make_entity_nlp, make_mock_bert

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _inject_nlp(*specs: tuple[str, str]) -> None:
    """Replace the mock NLP with one that recognises the given entity specs."""
    tokenizer, bert = make_mock_bert()
    registry = ModelRegistry(
        nlp=make_entity_nlp(list(specs)),  # type: ignore[arg-type]
        tokenizer=tokenizer,
        bert=bert,
    )
    set_models(registry)


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------

class TestExtractEntities:
    def test_returns_entities_with_confidence_and_explanation(self) -> None:
        _inject_nlp(("John Smith", "PERSON"), ("New York", "GPE"))
        text = "John Smith was last seen in New York."
        entities = extract_entities(text)

        assert len(entities) == 2
        john = next(e for e in entities if e.label == "PERSON")
        assert john.text == "John Smith"
        assert 0.0 < john.confidence <= 1.0
        assert john.explanation != ""
        assert "PERSON" in john.explanation

    def test_entities_sorted_by_start_char(self) -> None:
        _inject_nlp(("Paris", "GPE"), ("Alice", "PERSON"))
        # Alice appears first in text
        entities = extract_entities("Alice traveled to Paris on Monday.")
        labels = [e.label for e in entities]
        assert labels.index("PERSON") < labels.index("GPE")

    def test_confidence_uses_label_prior(self) -> None:
        _inject_nlp(("January 5, 2023", "DATE"))
        entities = extract_entities("The incident occurred on January 5, 2023.")
        assert len(entities) == 1
        # DATE label prior is 0.92
        assert entities[0].confidence == pytest.approx(0.92)

    def test_empty_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            extract_entities("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            extract_entities("   ")

    def test_no_entities_returns_empty_list(self) -> None:
        # Default mock NLP has no entity specs
        entities = extract_entities("The quick brown fox jumps over the lazy dog.")
        assert entities == []

    def test_explanation_references_context_tokens(self) -> None:
        _inject_nlp(("Acme Corp", "ORG"))
        entities = extract_entities("Acme Corp filed the report.")
        assert len(entities) == 1
        assert "Acme Corp" in entities[0].explanation


# ---------------------------------------------------------------------------
# extract_timeline
# ---------------------------------------------------------------------------

class TestExtractTimeline:
    def test_parseable_date_produces_high_confidence(self) -> None:
        _inject_nlp(("January 15, 2023", "DATE"), ("John", "PERSON"))
        text = "John was seen on January 15, 2023. Nothing else happened."
        events = extract_timeline(text)

        assert len(events) >= 1
        dated = [e for e in events if e.timestamp is not None]
        assert len(dated) >= 1
        assert dated[0].confidence == pytest.approx(0.90)

    def test_unparseable_date_produces_low_confidence(self) -> None:
        _inject_nlp(("last Tuesday", "DATE"))
        events = extract_timeline("He arrived last Tuesday.")
        # dateutil may or may not parse "last Tuesday"; confidence reflects result
        for event in events:
            if event.timestamp is None:
                assert event.confidence == pytest.approx(0.55)

    def test_events_sorted_chronologically(self) -> None:
        _inject_nlp(("March 3, 2023", "DATE"), ("January 1, 2023", "DATE"))
        text = "The arrest was on March 3, 2023. The crime occurred on January 1, 2023."
        events = extract_timeline(text)
        parseable = [e for e in events if e.timestamp is not None]
        if len(parseable) >= 2:
            assert parseable[0].timestamp <= parseable[1].timestamp  # type: ignore[operator]

    def test_entities_involved_excludes_temporal_expressions(self) -> None:
        _inject_nlp(("Alice", "PERSON"), ("June 10, 2022", "DATE"))
        text = "Alice was present on June 10, 2022."
        events = extract_timeline(text)
        dated = [e for e in events if e.timestamp_text == "June 10, 2022"]
        assert len(dated) >= 1
        assert "Alice" in dated[0].entities_involved
        assert "June 10, 2022" not in dated[0].entities_involved

    def test_empty_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            extract_timeline("")

    def test_text_with_no_dates_returns_empty_list(self) -> None:
        events = extract_timeline("The suspect fled the scene quickly.")
        assert events == []

    def test_every_event_has_explanation(self) -> None:
        _inject_nlp(("February 2, 2024", "DATE"))
        events = extract_timeline("The report was filed on February 2, 2024.")
        for event in events:
            assert event.explanation != ""


# ---------------------------------------------------------------------------
# build_relationship_graph
# ---------------------------------------------------------------------------

class TestBuildRelationshipGraph:
    def test_nodes_created_for_each_entity(self) -> None:
        _inject_nlp(("Alice", "PERSON"), ("Bob", "PERSON"), ("London", "GPE"))
        text = "Alice met Bob in London. Alice and Bob discussed the case."
        result = build_relationship_graph(text)
        assert "Alice" in result.graph.nodes
        assert "Bob" in result.graph.nodes
        assert "London" in result.graph.nodes

    def test_co_occurring_entities_connected_by_edge(self) -> None:
        _inject_nlp(("Alice", "PERSON"), ("Bob", "PERSON"))
        result = build_relationship_graph("Alice and Bob were at the scene.")
        assert result.graph.has_edge("Alice", "Bob")

    def test_edge_weight_reflects_co_occurrence_count(self) -> None:
        _inject_nlp(("Alice", "PERSON"), ("Bob", "PERSON"))
        text = "Alice saw Bob. Alice spoke to Bob. Alice left with Bob."
        result = build_relationship_graph(text)
        assert result.graph.has_edge("Alice", "Bob")
        assert result.graph["Alice"]["Bob"]["weight"] >= 2

    def test_node_attributes_include_label_and_confidence(self) -> None:
        _inject_nlp(("New York", "GPE"))
        result = build_relationship_graph("The meeting was held in New York.")
        node = result.graph.nodes["New York"]
        assert node["label"] == "GPE"
        assert 0.0 < node["confidence"] <= 1.0

    def test_empty_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_relationship_graph("")

    def test_no_entities_produces_empty_graph(self) -> None:
        result = build_relationship_graph("Nothing happened here.")
        assert result.node_count == 0
        assert result.edge_count == 0
        assert result.confidence == pytest.approx(0.0)

    def test_result_has_explanation(self) -> None:
        _inject_nlp(("Alice", "PERSON"))
        result = build_relationship_graph("Alice filed the report.")
        assert result.explanation != ""
        assert "node" in result.explanation.lower()


# ---------------------------------------------------------------------------
# classify_document
# ---------------------------------------------------------------------------

class TestClassifyDocument:
    def test_returns_a_known_document_type(self) -> None:
        result = classify_document("Officers responded to the incident.")
        assert result.document_type in DocumentType.__members__.values()

    def test_confidence_is_in_valid_range(self) -> None:
        result = classify_document("I witnessed the events described below.")
        # Cosine similarity is in [-1, 1]; for any reasonable text it should be > -1
        assert -1.0 <= result.confidence <= 1.0

    def test_all_scores_covers_all_document_types(self) -> None:
        result = classify_document("The court hereby orders the defendant.")
        score_names = {name for name, _ in result.all_scores}
        expected = {dt.value for dt in DocumentType}
        assert score_names == expected

    def test_all_scores_sorted_descending(self) -> None:
        result = classify_document("Forensic analysis of the DNA sample.")
        scores = [s for _, s in result.all_scores]
        assert scores == sorted(scores, reverse=True)

    def test_explanation_mentions_method(self) -> None:
        result = classify_document("Chain of custody evidence log item 42.")
        assert "LegalBERT" in result.explanation
        assert "cosine" in result.explanation.lower()

    def test_empty_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            classify_document("")

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            classify_document("   ")
