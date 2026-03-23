# Forensiq

Evidence analysis and prioritization system for criminal case investigation.

Forensiq is a decision support tool for prosecutors, defense attorneys, and investigators. It ingests raw evidence, analyzes it using NLP and graph models, detects contradictions, scores reliability, and produces a prioritized evidence summary for human review.

**The system does not render verdicts. Every output is advisory. A human analyst reviews all findings before any action is taken.**

---

## Architecture

Forensiq is a pipeline of discrete, auditable stages. Each stage produces a signed output that feeds the next. No stage has write access to another stage's data store.

| Component | Stack | Responsibility |
|---|---|---|
| Ingestion | Rust | Evidence intake, SHA-256 hashing, UUID assignment, chain-of-custody metadata |
| Audit Chain | Rust + SQLite | Append-only cryptographic log of every operation |
| Evidence Store | Rust + DuckDB | Immutable evidence repository with integrity verification on every read |
| Analysis Engine | Python + spaCy + LegalBERT | Entity extraction, timeline reconstruction, relationship graphs, document classification |
| Contradiction Detection | Python + CrossEncoder | Cross-reference claims, flag conflicts with confidence scores |
| Reasoning Engine | Python + Rust | Bayesian evidence scoring, reliability ranking |
| Report Generator | Rust | Structured, traceable analysis reports |
| Review Interface | HTML/CSS/JS | Analyst dashboard with evidence drill-down and sign-off flow |

---

## Build Phases

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Ingestion pipeline — hashing, UUID, audit logging | ✅ Complete |
| 2 | Evidence store — DuckDB append-only interface | ✅ Complete |
| 3 | Analysis engine — NLP, timeline, graph, classification | ✅ Complete |
| 4 | Contradiction detection — CrossEncoder NLI | 🔲 Pending |
| 5 | Reasoning engine — Bayesian scoring | 🔲 Pending |
| 6 | Report generator | 🔲 Pending |
| 7 | Review interface | 🔲 Pending |
| 8 | Hardening — security audit, bias testing, benchmarks | 🔲 Pending |

---

## Security Model

- All evidence hashed with SHA-256 on ingest; hash verified before every analysis operation
- Evidence bytes are immutable after ingest — no modification, no deletion
- Audit chain is append-only; tampering invalidates all subsequent entries
- Every score and classification includes an explanation vector — no black-box outputs
- No verdict probability is ever computed

---

## Stack

- **Rust** — ingestion, audit chain, evidence store, report generation, API layer
- **Python 3.10** — NLP analysis, contradiction detection, Bayesian reasoning
- **SQLite** — append-only audit log
- **DuckDB** — columnar evidence store
- **spaCy** `en_core_web_lg` — named entity recognition, timeline extraction
- **LegalBERT** `nlpaueb/legal-bert-base-uncased` — document classification
- **NetworkX** — entity relationship graphs

---

## Getting Started

### Rust components

```bash
cargo build --workspace
cargo test --workspace
```

### Python components

```bash
pip install spacy transformers torch networkx python-dateutil
python -m pytest
python -m mypy analysis/
```

> First build of the `store` crate compiles DuckDB from source and takes several minutes.

---

## Non-Goals

- Autonomous verdicts — the system never determines guilt or innocence
- Sentencing recommendations
- Real-time surveillance integration
- Predictive policing
- Public-facing interfaces
