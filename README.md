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

**Phase 3 (ingestion)** — implemented:

- Secure PDF downloader (HTTPS, domain policy, redirect validation, size limits, PDF signature check)
- Docling parser with section/content-type preservation
- Section-aware token chunker
- Transactional ingestion worker with deduplication and status tracking

**Phase 4 (active discovery)** — implemented:

- arXiv search service (`search_arxiv`)
- Paper deduplication against ingestion metadata
- Relevance-based paper selection for ingestion
- Active literature loop with bounded discovery rounds and re-retrieval

**Phase 5 (orchestrator)** — implemented:

- Groq Llama 3.3 70B orchestration (query normalization, classification, decomposition)
- Citation-grounded synthesis with validation and regeneration
- End-to-end `ResearchOrchestrator` wired through active discovery pipeline
- CLI: `research-assistant "your question"`

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

Ask a research question:

```bash
research-assistant "How does transformer attention work?"
research-assistant --verbose "Compare RAG and GraphRAG latency"
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
