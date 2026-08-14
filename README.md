# Active Research Assistant

Agentic hybrid-retrieval RAG pipeline for academic literature discovery, ingestion, and citation-grounded synthesis.

See [AGENTS.md](./AGENTS.md) for the full technical specification.

## Status

**Phase 1 (foundation)** — implemented:

- Project scaffold and environment configuration
- Core domain models and provenance format
- SQLite FTS5 sparse index (BM25)
- ChromaDB dense index wrapper
- Document metadata / ingestion state store
- Weighted RRF fusion
- Evidence sufficiency gate (FlashRank scores only)
- Citation validation utilities
- Download path security helpers
- Transactional dual-index commit with rollback

**Phase 2 (hybrid retrieval)** — implemented:

- Gemini embedding service with retry policy
- FlashRank cross-encoder reranker wrapper
- HybridRetriever (dense + sparse → RRF → FlashRank → sufficiency)
- Integration tests for full retrieval stack

**Upcoming phases:**

- Secure PDF downloader
- Docling parser + section-aware chunking
- Ingestion worker
- arXiv discovery
- Groq orchestrator + grounded synthesis
- End-to-end pipeline CLI

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
copy .env.example .env        # add API keys
```

Verify configuration:

```bash
research-assistant --check-config
```

Run tests:

```bash
pytest
```

## Architecture

```
User Query → Orchestrator → Hybrid Retrieval (ChromaDB + FTS5)
         → RRF → FlashRank → Sufficiency Gate
         → [Sufficient] Citation Validation → Grounded Synthesis
         → [Insufficient] arXiv Discovery → Ingestion → Re-retrieve
```

## Configuration

All settings are loaded from environment variables. See `.env.example` for the full list aligned with AGENTS.md section 29.

API keys must never be committed to source control.
