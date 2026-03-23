"""Timeline reconstruction from DATE and TIME entities.

Each event is the sentence containing a temporal expression, annotated with a
parsed :class:`~datetime.datetime`, the other entities in the same sentence,
a confidence score, and an explanation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import spacy
from dateutil import parser as dateutil_parser
from dateutil.parser import ParserError

from .models import get_models


@dataclass
class TimelineEvent:
    """A single event on the reconstructed case timeline.

    Attributes
    ----------
    sentence:
        Full sentence from which the event was extracted (stripped).
    timestamp:
        Parsed UTC-naive :class:`~datetime.datetime`, or ``None`` if the
        temporal expression could not be resolved to a calendar date/time.
    timestamp_text:
        Original temporal expression string (e.g. ``"January 15, 2023"``).
    entities_involved:
        Surface forms of non-temporal entities co-occurring in the sentence
        (persons, locations, organisations, etc.).
    confidence:
        ``0.90`` when *timestamp* is successfully parsed; ``0.55`` otherwise,
        reflecting reduced certainty about event ordering.
    explanation:
        Human-readable rationale including the source sentence and parse result.
    """

    sentence: str
    timestamp: Optional[datetime]
    timestamp_text: str
    entities_involved: List[str] = field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""


def extract_timeline(text: str) -> List[TimelineEvent]:
    """Extract a chronological event timeline from *text*.

    Iterates over sentences, finds DATE/TIME entities in each, attempts to
    parse them with :mod:`dateutil`, and returns events sorted by parsed
    timestamp (events without a parseable timestamp sort last).

    Parameters
    ----------
    text:
        Raw document text to analyse.  Must be non-empty.

    Returns
    -------
    List[TimelineEvent]
        Events sorted chronologically; unparseable timestamps sort after all
        parseable ones.

    Raises
    ------
    ValueError
        If *text* is empty or whitespace-only.
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    nlp = get_models().nlp
    doc: spacy.tokens.Doc = nlp(text)

    events: List[TimelineEvent] = []
    for sent in doc.sents:
        temporal_ents = [e for e in sent.ents if e.label_ in ("DATE", "TIME")]
        if not temporal_ents:
            continue

        other_ent_texts = [e.text for e in sent.ents if e.label_ not in ("DATE", "TIME")]

        for dte in temporal_ents:
            timestamp = _parse_datetime(dte.text)
            confidence = 0.90 if timestamp is not None else 0.55
            parsed_note = (
                f"Parsed to {timestamp.isoformat()}."
                if timestamp is not None
                else "Could not resolve to a structured datetime."
            )
            explanation = (
                f"Temporal expression '{dte.text}' ({dte.label_}) found in: "
                f"'{sent.text.strip()}'. {parsed_note} "
                f"Confidence reflects parse success (0.90) or failure (0.55)."
            )
            events.append(
                TimelineEvent(
                    sentence=sent.text.strip(),
                    timestamp=timestamp,
                    timestamp_text=dte.text,
                    entities_involved=other_ent_texts,
                    confidence=confidence,
                    explanation=explanation,
                )
            )

    return sorted(
        events,
        key=lambda e: (e.timestamp is None, e.timestamp or datetime.min),
    )


def _parse_datetime(text: str) -> Optional[datetime]:
    """Attempt to parse *text* as a datetime.  Returns ``None`` on failure."""
    try:
        return dateutil_parser.parse(text, fuzzy=True)
    except (ParserError, OverflowError, ValueError):
        return None
