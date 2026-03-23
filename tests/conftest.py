"""Shared pytest fixtures for the Forensiq analysis engine tests.

All fixtures inject a lightweight mock ModelRegistry so no large model files
need to be downloaded during the test suite.

Mock NLP strategy
-----------------
``spacy.blank("en")`` with a ``sentencizer`` handles tokenisation and sentence
boundary detection.  A thin wrapper scans the input text for pre-registered
entity strings and injects them as ``doc.ents`` using ``char_span``.  This
produces real :class:`spacy.tokens.Span` objects (so all attribute accesses
work correctly) without requiring any downloaded model weights.

Mock BERT strategy
------------------
Tokenizer and BERT model are :class:`unittest.mock.MagicMock` objects.
The tokenizer returns a dict of zero tensors; the BERT model returns a mock
with ``.last_hidden_state`` set to a random ``(1, 10, 768)`` tensor.  This
exercises the full classification pipeline without network calls.
"""
from __future__ import annotations

from typing import Callable, List, Tuple
from unittest.mock import MagicMock

import pytest
import spacy
import torch

from analysis.models import ModelRegistry, reset_models, set_models


def make_entity_nlp(
    entity_specs: List[Tuple[str, str]] | None = None,
) -> Callable[[str], spacy.tokens.Doc]:
    """Return a callable that behaves like ``spacy.Language`` for tests.

    Parameters
    ----------
    entity_specs:
        List of ``(entity_surface_form, NER_label)`` pairs.  Every time the
        returned callable processes text that contains a surface form, it
        injects a corresponding entity span with the given label.
        Pass ``None`` (or ``[]``) for an NLP that produces no entities.
    """
    blank = spacy.blank("en")
    blank.add_pipe("sentencizer")
    specs = entity_specs or []

    def process(text: str) -> spacy.tokens.Doc:
        doc = blank(text)
        spans = []
        for surface, label in specs:
            start = text.find(surface)
            while start >= 0:
                span = doc.char_span(start, start + len(surface), label=label)
                if span is not None:
                    spans.append(span)
                start = text.find(surface, start + 1)
        # Filter overlapping spans — keep leftmost / longest
        try:
            doc.ents = spacy.util.filter_spans(spans)  # type: ignore[assignment]
        except ValueError:
            doc.ents = tuple(spans[:1])  # type: ignore[assignment]
        return doc

    return process


def make_mock_bert() -> Tuple[MagicMock, MagicMock]:
    """Return ``(tokenizer_mock, bert_mock)`` for classification tests."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }

    bert_outputs = MagicMock()
    bert_outputs.last_hidden_state = torch.randn(1, 10, 768)

    bert = MagicMock()
    bert.return_value = bert_outputs

    return tokenizer, bert


@pytest.fixture(autouse=True)
def mock_models() -> None:  # type: ignore[return]
    """Inject a default mock ModelRegistry before every test.

    Tests that need specific entities override the registry by calling
    ``set_models(...)`` themselves — that call takes precedence because this
    fixture runs first (autouse, function scope).
    """
    tokenizer, bert = make_mock_bert()
    registry = ModelRegistry(
        nlp=make_entity_nlp(),  # type: ignore[arg-type]
        tokenizer=tokenizer,
        bert=bert,
    )
    set_models(registry)
    yield
    reset_models()
