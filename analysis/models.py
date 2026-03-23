"""Singleton model registry for the Forensiq analysis engine.

All heavyweight models (spaCy, LegalBERT) are loaded once at startup via
:func:`get_models`. Tests inject a lightweight fake via :func:`set_models`
so no model files need to be downloaded during the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import spacy
from spacy.language import Language as SpacyLanguage

# Defer heavy imports to _load_models so the module can be imported in
# environments where transformers / torch are not yet on the path.

_SPACY_MODEL: str = "en_core_web_lg"
_LEGAL_BERT_MODEL: str = "nlpaueb/legal-bert-base-uncased"

_registry: Optional["ModelRegistry"] = None


@dataclass
class ModelRegistry:
    """Container for all NLP models used by the analysis engine.

    Fields
    ------
    nlp:
        spaCy pipeline (must include NER and sentencizer/parser).
    tokenizer:
        HuggingFace tokenizer for LegalBERT.
    bert:
        LegalBERT model (``AutoModel``).  Only forward-pass inference is used.
    """

    nlp: SpacyLanguage
    tokenizer: Any  # AutoTokenizer — typed as Any for mypy compat with transformers
    bert: Any       # AutoModel    — typed as Any for mypy compat with transformers


def get_models() -> ModelRegistry:
    """Return the shared :class:`ModelRegistry`, loading models on first call."""
    global _registry
    if _registry is None:
        _registry = _load_models()
    return _registry


def set_models(registry: ModelRegistry) -> None:
    """Inject a custom registry.  Used in tests to avoid large model downloads."""
    global _registry
    _registry = registry


def reset_models() -> None:
    """Clear the cached registry so the next :func:`get_models` call reloads."""
    global _registry
    _registry = None


def _load_models() -> ModelRegistry:
    from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

    nlp = spacy.load(_SPACY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(_LEGAL_BERT_MODEL)
    bert = AutoModel.from_pretrained(_LEGAL_BERT_MODEL)
    bert.eval()
    return ModelRegistry(nlp=nlp, tokenizer=tokenizer, bert=bert)
