"""Legal document classification using LegalBERT embeddings.

Uses zero-shot cosine-similarity classification against prototype phrases for
each document type.  Every result includes a confidence score (cosine
similarity to the winning prototype) and a full explanation.

Note: For production accuracy, fine-tune LegalBERT on labelled examples from
the relevant jurisdiction.  The zero-shot approach used here provides a
reasonable baseline without any training data.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .models import get_models

# Document types recognised by the classifier.
class DocumentType(str, Enum):
    WITNESS_STATEMENT = "WITNESS_STATEMENT"
    FORENSIC_REPORT = "FORENSIC_REPORT"
    POLICE_REPORT = "POLICE_REPORT"
    COURT_DOCUMENT = "COURT_DOCUMENT"
    EVIDENCE_LOG = "EVIDENCE_LOG"
    OTHER = "OTHER"


# Representative prototype phrases — the document whose CLS embedding is
# closest (by cosine similarity) to the input is the predicted type.
_PROTOTYPES: dict[DocumentType, str] = {
    DocumentType.WITNESS_STATEMENT: (
        "I witnessed the following events. The defendant was seen at the location "
        "on that date. I hereby declare this statement to be true."
    ),
    DocumentType.FORENSIC_REPORT: (
        "Forensic laboratory analysis of the submitted samples indicates. "
        "DNA profiling results. Ballistic examination findings. Chain of custody maintained."
    ),
    DocumentType.POLICE_REPORT: (
        "Officers responded to the incident at the scene. The suspect was "
        "apprehended and processed. Case number assigned. Report filed by officer."
    ),
    DocumentType.COURT_DOCUMENT: (
        "IN THE COURT OF. Plaintiff vs Defendant. The court hereby orders. "
        "Whereas the motion states. Judgment entered on the following grounds."
    ),
    DocumentType.EVIDENCE_LOG: (
        "Evidence item number. Collected at crime scene by officer. "
        "Chain of custody log. Storage location. Date and time of collection."
    ),
    DocumentType.OTHER: (
        "Document of unspecified legal or administrative type."
    ),
}


@dataclass
class DocumentClassification:
    """Classification result for a legal document.

    Attributes
    ----------
    document_type:
        Predicted :class:`DocumentType`.
    confidence:
        Cosine similarity to the winning prototype (``[−1, 1]``; higher is better).
    all_scores:
        ``(document_type_name, cosine_similarity)`` for every class, sorted
        descending.  Enables the caller to inspect the full score distribution.
    explanation:
        Human-readable description of the classification method, the winning
        prototype similarity, and the next-best alternative.
    """

    document_type: DocumentType
    confidence: float
    all_scores: List[Tuple[str, float]]
    explanation: str


def classify_document(text: str) -> DocumentClassification:
    """Classify the type of legal document in *text* using LegalBERT.

    Encodes *text* and each prototype phrase with LegalBERT [CLS] embeddings,
    then picks the document type whose prototype has the highest cosine
    similarity to the input embedding.

    Parameters
    ----------
    text:
        Raw document text.  Truncated to 512 tokens if longer.  Must be non-empty.

    Returns
    -------
    DocumentClassification
        Predicted type, confidence, full score distribution, and explanation.

    Raises
    ------
    ValueError
        If *text* is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    registry = get_models()
    doc_embedding = _encode(text, registry.tokenizer, registry.bert)

    scores: List[Tuple[str, float]] = []
    for doc_type, prototype in _PROTOTYPES.items():
        proto_embedding = _encode(prototype, registry.tokenizer, registry.bert)
        similarity = float(
            F.cosine_similarity(doc_embedding, proto_embedding, dim=1).item()
        )
        scores.append((doc_type.value, similarity))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = scores[0]
    best_type = DocumentType(best_name)

    explanation = (
        f"Classified as {best_type.value} via LegalBERT [CLS] cosine similarity "
        f"(zero-shot, no fine-tuning). "
        f"Winning score: {best_score:.3f}. "
        f"Runner-up: {scores[1][0]} ({scores[1][1]:.3f}). "
        f"For higher accuracy, fine-tune on labelled jurisdictional examples."
    )

    return DocumentClassification(
        document_type=best_type,
        confidence=best_score,
        all_scores=scores,
        explanation=explanation,
    )


def _encode(text: str, tokenizer: object, bert: object) -> torch.Tensor:
    """Encode *text* to a (1, hidden_size) CLS embedding tensor."""
    inputs = tokenizer(  # type: ignore[operator]
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = bert(**inputs)  # type: ignore[operator]
    cls_embedding: torch.Tensor = outputs.last_hidden_state[:, 0, :]
    return cls_embedding
