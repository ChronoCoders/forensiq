"""Entity co-occurrence relationship graph using NetworkX.

Nodes represent unique named entities; edges connect entities that appear
in the same sentence, weighted by co-occurrence frequency.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import spacy

from .entities import extract_entities
from .models import get_models


@dataclass
class RelationshipGraph:
    """A sentence-level co-occurrence graph of entities in a document.

    Attributes
    ----------
    graph:
        :class:`networkx.Graph` where each node is an entity surface form
        (with ``label`` and ``confidence`` node attributes) and each edge
        carries a ``weight`` equal to the number of sentences in which the
        two endpoints co-occur.
    node_count:
        Total number of unique entity nodes.
    edge_count:
        Total number of co-occurrence edges.
    confidence:
        Mean entity confidence across all nodes.  Zero if no entities found.
    explanation:
        Human-readable summary of graph construction, including node/edge
        counts and the mean confidence.
    """

    graph: nx.Graph
    node_count: int
    edge_count: int
    confidence: float
    explanation: str


def build_relationship_graph(text: str) -> RelationshipGraph:
    """Build a co-occurrence graph of named entities from *text*.

    Construction algorithm
    ----------------------
    1. Run NER to collect all entities and their per-label confidence scores.
    2. Add one node per unique entity surface form.
    3. For each sentence, add edges between every pair of entities that
       co-occur in that sentence, incrementing ``weight`` on repeated pairs.

    Parameters
    ----------
    text:
        Raw document text.  Must be non-empty.

    Returns
    -------
    RelationshipGraph
        Populated graph with node/edge counts, mean confidence, and
        a plain-text explanation.

    Raises
    ------
    ValueError
        If *text* is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    # Entity extraction (uses registered nlp internally)
    entities = extract_entities(text)

    graph: nx.Graph = nx.Graph()
    entity_confidence: dict[str, float] = {}

    for ent in entities:
        if ent.text not in graph:
            graph.add_node(ent.text, label=ent.label, confidence=ent.confidence)
            entity_confidence[ent.text] = ent.confidence

    # Sentence-level co-occurrence edges
    nlp = get_models().nlp
    doc: spacy.tokens.Doc = nlp(text)
    for sent in doc.sents:
        sent_ent_texts = list({e.text for e in sent.ents})
        for i, a in enumerate(sent_ent_texts):
            for b in sent_ent_texts[i + 1 :]:
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    mean_confidence = (
        sum(entity_confidence.values()) / len(entity_confidence)
        if entity_confidence
        else 0.0
    )

    explanation = (
        f"Co-occurrence graph: {node_count} entity nodes, {edge_count} edges. "
        f"Edges connect entities appearing in the same sentence (weight = "
        f"co-occurrence frequency). Mean entity confidence: {mean_confidence:.2f}."
    )

    return RelationshipGraph(
        graph=graph,
        node_count=node_count,
        edge_count=edge_count,
        confidence=mean_confidence,
        explanation=explanation,
    )
