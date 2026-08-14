# AGENTS.md

# Active Research Assistant — Agent Architecture & Engineering Specification

## 1. System Overview

This repository implements an **agentic, hybrid-retrieval RAG pipeline** for academic literature discovery, layout-aware document ingestion, persistent indexing, cross-encoder reranking, evidence sufficiency evaluation, and citation-grounded synthesis.

The system extends conventional static RAG by allowing the assistant to:

1. Search its persistent local knowledge base.
2. Evaluate whether the retrieved evidence is sufficiently relevant.
3. Detect context gaps.
4. Discover new academic literature from arXiv.
5. Download and ingest previously unseen papers.
6. Re-run retrieval over the expanded knowledge base.
7. Produce a citation-grounded answer with deterministic provenance.
8. Explicitly report insufficient evidence instead of fabricating unsupported claims.

The architecture is best described as:

> **An agentic RAG system with an orchestration agent, a specialized ingestion worker, hybrid retrieval, cross-encoder reranking, and active literature discovery.**

It is not considered a collection of independent autonomous agents. Retrieval, embedding, reranking, parsing, and storage components are deterministic infrastructure services/tools.

---

# 2. System Architecture & Topology

```text
                              [ USER QUERY ]
                                    │
                                    ▼
                     ┌─────────────────────────┐
                     │   ORCHESTRATOR AGENT    │
                     │     Groq Llama 3.3      │
                     │        70B             │
                     └────────────┬────────────┘
                                  │
                     Query normalization /
                     classification /
                     decomposition
                                  │
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
        [ EXISTING KNOWLEDGE ]             [ COMPLEX QUERY ]
                 │                                 │
                 │                         Query decomposition
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │   HYBRID RETRIEVAL  │
                       └──────────┬──────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             ┌────────────┐              ┌────────────┐
             │ ChromaDB   │              │ SQLite     │
             │ Dense      │              │ FTS5/BM25  │
             │ Retrieval  │              │ Sparse     │
             └──────┬─────┘              └──────┬─────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                         Reciprocal Rank Fusion
                                  │
                                  ▼
                         Top-N Candidate Pool
                              N = 15
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ FlashRank Cross-Encoder  │
                    │ ms-marco-MiniLM-L-12-v2 │
                    │       Local CPU          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                         Ranked Top-K Passages
                               K = 3
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │  EVIDENCE SUFFICIENCY    │
                    │         GATE              │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
              SUFFICIENT                 INSUFFICIENT
                    │                         │
                    │                         ▼
                    │                  ┌──────────────┐
                    │                  │  arXiv API   │
                    │                  │   Discovery  │
                    │                  └──────┬───────┘
                    │                         │
                    │                         ▼
                    │                  Deduplication
                    │                         │
                    │                         ▼
                    │                  Ingestion Worker
                    │                         │
                    │                         ▼
                    │                     Docling
                    │                         │
                    │                         ▼
                    │                   Chunk + Metadata
                    │                         │
                    │                         ▼
                    │                  Gemini Embeddings
                    │                         │
                    │                         ▼
                    │              ┌──────────┴──────────┐
                    │              ▼                     ▼
                    │          ChromaDB              SQLite FTS5
                    │              │                     │
                    │              └──────────┬──────────┘
                    │                         ▼
                    │                  Re-run Retrieval
                    │                         │
                    └─────────────────────────┘
                                              │
                                              ▼
                                  Citation Validation
                                              │
                                              ▼
                                  Grounded Synthesis
                                              │
                                              ▼
                                    [ VERIFIED OUTPUT ]
```

---

# 3. Agent and Component Catalog

## 3.1 Orchestration & Synthesis Agent

**Runtime:** Groq Cloud

**Model:**

```text
llama-3.3-70b-versatile
```

**Responsibilities:**

* Query classification
* Intent analysis
* Query normalization
* Query decomposition
* Workflow routing
* Tool selection
* Evidence sufficiency interpretation
* Citation-grounded synthesis
* Final response generation

**Configuration:**

```text
Temperature: 0.0
Max output tokens: 2048
```

The orchestrator must not fabricate evidence when the retrieval layer reports insufficient evidence.

The orchestrator may use parametric model knowledge for conversational or non-evidence-dependent content, but **technical research claims requiring source support must be grounded in retrieved evidence**.

---

## 3.2 Ingestion Worker

**Runtime:** Async Python worker process

**Responsibilities:**

* Validate source metadata
* Download academic PDFs
* Enforce download security constraints
* Parse documents using Docling
* Preserve document layout
* Extract structured content
* Generate chunks
* Generate embeddings
* Write dense and sparse indexes
* Persist document metadata
* Maintain ingestion status
* Prevent duplicate ingestion

The worker is a specialized processing service, not an autonomous LLM agent.

---

## 3.3 Document Parser

**Library:** Docling

The parser must preserve, where available:

* Two-column document structure
* Headings
* Paragraphs
* Tables
* Equations
* Figure captions
* Lists
* Page boundaries
* Section hierarchy

The parser should produce a structured intermediate representation before chunking.

---

## 3.4 Dense Vector Engine

**Embedding model:**

```text
models/text-embedding-004
```

**Provider:**

```text
Google Gemini
```

**Configured embedding dimension:**

```text
768
```

The deployed ChromaDB collection must use one consistent dimensionality.

Do not mix 768-dimensional and 1536-dimensional vectors inside the same collection.

If a different embedding model or dimensionality is introduced, it must use a separate collection/index version.

**Storage:**

```text
./data/chroma_db
```

**Recommended distance metric:**

```text
cosine
```

The implementation must explicitly convert the returned vector distance into a documented similarity value if a similarity threshold is required.

---

# 4. Sparse Retrieval Engine

**Implementation:**

```text
SQLite FTS5
```

**Retrieval model:**

```text
BM25
```

**Purpose:**

Sparse retrieval is responsible for exact and lexical matching, especially for:

* arXiv identifiers
* API names
* variable identifiers
* hardware registers
* technical terminology
* exact model names
* numeric identifiers
* uncommon scientific terms

BM25 scores are **not used directly as a fixed `0.70` sufficiency threshold** because BM25 scores are query- and corpus-dependent and are not naturally calibrated to a fixed probability-like range.

---

# 5. Reciprocal Rank Fusion

Dense and sparse retrieval results are combined using weighted Reciprocal Rank Fusion.

The configured formula is:

```text
RRF(d) =
    w_dense  / (k + rank_dense(d))
  + w_sparse / (k + rank_sparse(d))
```

Configuration:

```text
w_dense  = 0.6
w_sparse = 0.4
k        = 60
```

Therefore:

```text
RRF(d) =
    0.6 / (60 + rank_dense(d))
  + 0.4 / (60 + rank_sparse(d))
```

If a document is absent from one retrieval list, that retrieval channel contributes zero to its RRF score.

RRF scores are used for **candidate pooling and ranking**, not as semantic confidence values.

---

# 6. Candidate Pooling

The hybrid retrieval stage produces:

```text
Top 15 candidates
```

These candidates are passed to FlashRank.

Configuration:

```text
RRF_CANDIDATE_K=15
```

The candidate pool must preserve provenance and metadata for every passage.

---

# 7. Local Cross-Encoder Reranker

**Library:**

```text
FlashRank
```

**Model:**

```text
ms-marco-MiniLM-L-12-v2
```

**Execution:**

```text
Local CPU
```

**Input:**

```text
Top 15 RRF candidates
```

**Output:**

```text
Ranked passages
```

**Final retrieval context:**

```text
Top 3 passages
```

FlashRank evaluates the query and passage jointly using cross-encoder attention.

Its output is treated as a:

> **cross-encoder relevance/reranking score**

It must **not** be described as a universally calibrated probability of relevance unless calibration has been explicitly performed and validated for this deployment.

---

# 8. Evidence Sufficiency Evaluation

The sufficiency gate determines whether the system has enough relevant local evidence to proceed to synthesis or whether it must actively discover additional literature.

The trigger is:

```text
candidate_count >= MIN_CANDIDATES
AND
top_score >= MIN_RERANK_SCORE
```

Where:

```text
candidate_count
```

is the number of candidates returned by the retrieval pipeline, and:

```text
top_score
```

is the highest FlashRank relevance score among the reranked candidates.

Recommended configuration:

```text
MIN_CANDIDATES=1
MIN_RERANK_SCORE=0.70
```

The value `0.70` is an **empirical calibration parameter**, not a universal standard.

It must be calibrated using an evaluation dataset containing:

* relevant queries
* irrelevant queries
* borderline queries
* domain-specific technical queries
* known answerable questions
* known unanswerable questions

The threshold may be increased or decreased based on measured precision/recall and unsupported-answer rates.

---

# 9. Sufficiency Is Not the Same as Correctness

A high FlashRank score indicates that a passage is highly relevant to the query.

It does not guarantee that:

* the passage contains all required information
* the source is authoritative
* the answer is factually correct
* multiple requested aspects are covered
* contradictory evidence does not exist

Therefore the production system should treat the FlashRank threshold as a:

> **retrieval sufficiency signal**

and not as a guarantee of factual correctness.

For complex questions, query coverage must also be considered.

Example:

```text
Question:
Compare RAG and GraphRAG in terms of:
1. Accuracy
2. Latency
3. Hallucination rate
```

A single passage with:

```text
FlashRank = 0.91
```

may only discuss accuracy.

Therefore complex queries should be decomposed into evidence requirements.

---

# 10. Query Decomposition

For multi-part research questions, the orchestrator should identify independent evidence requirements.

Example:

```text
Original:
Compare RAG, GraphRAG and Agentic RAG across
accuracy, latency and hallucination.

Subqueries:

Q1: RAG accuracy
Q2: GraphRAG accuracy
Q3: Agentic RAG accuracy
Q4: RAG latency
Q5: GraphRAG latency
Q6: Agentic RAG latency
Q7: hallucination comparison
```

Each sub-query should be independently retrieved and evaluated.

The final synthesis may proceed only when sufficient evidence exists for the required aspects, or the system must explicitly identify unsupported aspects.

---

# 11. Active Literature Discovery

If the sufficiency gate fails:

```text
candidate_count < MIN_CANDIDATES
OR
top_score < MIN_RERANK_SCORE
```

the system must enter the active discovery workflow.

```text
Local retrieval
      ↓
Insufficient evidence
      ↓
search_arxiv()
      ↓
Candidate papers
      ↓
Deduplication
      ↓
Select relevant papers
      ↓
ingest_pdf_document()
      ↓
Re-run hybrid retrieval
      ↓
FlashRank reranking
      ↓
Sufficiency evaluation
```

The system must not automatically ingest unlimited papers.

Recommended configuration:

```text
DISCOVERY_MAX_RESULTS=5
MAX_NEW_DOCUMENTS_PER_QUERY=3
MAX_DISCOVERY_ROUNDS=2
```

These values should be configurable.

---

# 12. arXiv Discovery Tool

## Tool: `search_arxiv`

Purpose:

Search arXiv metadata for academic papers relevant to the user's query.

Schema:

```json
{
  "name": "search_arxiv",
  "description": "Searches arXiv for academic publications matching a normalized research query.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Clean technical research query without conversational filler."
      },
      "max_results": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 5
      }
    },
    "required": ["query"]
  }
}
```

The search result should contain, where available:

```text
arxiv_id
title
authors
abstract
published_date
updated_date
pdf_url
categories
```

---

# 13. PDF Ingestion Tool

## Tool: `ingest_pdf_document`

Purpose:

Download, validate, parse, chunk, embed, and persist an academic paper.

Schema:

```json
{
  "name": "ingest_pdf_document",
  "description": "Downloads, validates, parses, chunks, embeds, and indexes an academic PDF.",
  "parameters": {
    "type": "object",
    "properties": {
      "arxiv_id": {
        "type": "string",
        "description": "Unique arXiv identifier such as 2407.08608."
      },
      "pdf_url": {
        "type": "string",
        "description": "Validated HTTPS PDF URL."
      },
      "title": {
        "type": "string",
        "description": "Official paper title."
      }
    },
    "required": [
      "arxiv_id",
      "pdf_url",
      "title"
    ]
  }
}
```

The ingestion worker must verify that the URL and resolved download target comply with the configured source policy before downloading.

---

# 14. Document Deduplication

Every paper must have a unique document identifier.

For arXiv documents:

```text
document_id = normalized_arxiv_id
```

Before downloading:

```text
IF document already exists:
    skip download
    reuse existing index
ELSE:
    ingest document
```

The system should additionally support content hashing where practical:

```text
content_hash = SHA-256(document bytes)
```

This prevents duplicate indexing when the same document is encountered through different URLs.

---

# 15. Document Metadata

Every chunk must contain structured metadata.

Recommended schema:

```json
{
  "document_id": "2407.08608",
  "arxiv_id": "2407.08608",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "published_date": "2024-07-11",
  "section": "Methodology",
  "subsection": "Data Collection",
  "page": 7,
  "chunk_index": 12,
  "content_type": "paragraph",
  "source": "arxiv",
  "embedding_model": "models/text-embedding-004",
  "embedding_dimension": 768
}
```

Supported `content_type` values should include:

```text
paragraph
heading
table
equation
figure_caption
list
```

This metadata enables future filtering and citation precision.

---

# 16. Chunking Strategy

Chunking must be **section-aware and size-aware**.

Do not store an entire long academic section as one chunk.

Recommended strategy:

```text
Document
   ↓
Section
   ↓
Subsection
   ↓
Paragraph grouping
   ↓
Token/character size limit
   ↓
Small overlap
```

Recommended initial configuration:

```text
CHUNK_TARGET_TOKENS=700
CHUNK_MAX_TOKENS=1000
CHUNK_OVERLAP_TOKENS=100
MIN_CHUNK_CHARACTERS=80
```

These parameters must be benchmarked against the target corpus.

Every chunk must preserve:

```text
document_id
section
page
chunk_index
```

---

# 17. Tables, Equations and Figures

The ingestion system must not treat every document element as plain paragraph text.

Tables should retain:

* table title
* column headers
* row values
* page number
* surrounding section

Equations should retain:

* equation text
* equation number, if available
* surrounding explanation

Figures should retain:

* figure caption
* page
* surrounding section

This is particularly important for research questions involving quantitative results.

---

# 18. Persistent Storage

## Dense Store

```text
ChromaDB
./data/chroma_db
```

## Sparse Store

```text
SQLite FTS5
./data/sparse_index.db
```

## Download Cache

```text
./data/downloads
```

## Recommended additional metadata store

The system should maintain document-level ingestion state.

Example:

```text
documents
---------
document_id
arxiv_id
title
content_hash
source_url
status
created_at
updated_at
```

Possible status values:

```text
DISCOVERED
DOWNLOADING
DOWNLOADED
PARSING
PARSED
EMBEDDING
INDEXING
INGESTED
FAILED
```

---

# 19. Transactional Ingestion

A document must not be considered successfully ingested until all required indexing operations succeed.

Recommended sequence:

```text
Download
   ↓
Validate
   ↓
Parse
   ↓
Chunk
   ↓
Generate embeddings
   ↓
Prepare Chroma records
   ↓
Prepare SQLite records
   ↓
Commit indexes
   ↓
Mark document INGESTED
```

If a required operation fails:

```text
Mark document FAILED
```

and prevent the document from being considered valid retrieval evidence.

The implementation should provide rollback or cleanup mechanisms to prevent ChromaDB and SQLite from becoming inconsistent.

---

# 20. Failure Handling

The system must explicitly handle:

```text
ARXIV_SEARCH_FAILED
PDF_DOWNLOAD_FAILED
PDF_VALIDATION_FAILED
PDF_PARSE_FAILED
CHUNKING_FAILED
EMBEDDING_FAILED
CHROMA_WRITE_FAILED
SQLITE_WRITE_FAILED
INDEX_TRANSACTION_FAILED
RETRIEVAL_FAILED
RERANKING_FAILED
SYNTHESIS_FAILED
INSUFFICIENT_EVIDENCE
```

Failures must be logged with:

```text
timestamp
document_id
operation
error_type
error_message
retry_count
```

Transient external API failures should use bounded retries with exponential backoff.

---

# 21. Security Guardrails

## 21.1 Path Traversal

All downloaded files must remain inside:

```text
./data/downloads/
```

Filename generation must use strict sanitization.

Example:

```python
safe_filename = re.sub(
    r"[^a-zA-Z0-9_\-.]",
    "",
    f"{arxiv_id}.pdf"
)
```

Additionally, the resolved filesystem path must be checked to ensure it remains inside the configured download directory.

Filename sanitization alone is not considered sufficient path-traversal protection.

---

## 21.2 Download Validation

The ingestion worker should enforce:

* HTTPS-only sources
* Allowed-domain policy
* Redirect validation
* Download timeout
* Maximum file size
* PDF MIME/signature validation
* Safe temporary-file handling
* Cleanup of failed downloads

Recommended configuration:

```text
MAX_PDF_SIZE_MB=50
DOWNLOAD_TIMEOUT_SECONDS=30
```

---

# 22. Citation and Provenance Architecture

Every retrieved chunk must carry immutable provenance.

Internal provenance format:

```text
[arXiv:<ARXIV_ID> | Chunk <CHUNK_INDEX>]
```

Example:

```text
[arXiv:2407.08608 | Chunk 12]
```

The internal provenance identifier must map to:

```text
document
title
authors
page
section
chunk
source URL
```

For user-facing answers, human-readable citation identifiers are recommended:

```text
[1]
[2]
[3]
```

The citation resolver maps:

```text
[1]
 ↓
arXiv ID
 ↓
paper metadata
 ↓
page
 ↓
section
 ↓
chunk
```

This allows readable citations while preserving exact machine-verifiable provenance.

---

# 23. Citation Enforcement

Technical claims must be supported by retrieved evidence.

The synthesis layer must attach provenance to evidence-dependent claims.

The internal validation pattern is:

```regex
\[arXiv:[0-9]{4}\.[0-9]{4,5}\s*\|\s*Chunk\s*\d+\]
```

The final output must be rejected or regenerated if required evidence citations are missing.

Citation validation must verify that referenced:

```text
arXiv ID
chunk index
```

actually exist in the retrieval context.

The system must not allow the LLM to invent citation identifiers.

---

# 24. Grounding Protocol

## No Unsupported Technical Claims

The synthesis agent must not invent:

* numerical metrics
* benchmark results
* architectural specifications
* experimental findings
* algorithmic details
* hardware specifications
* equations
* implementation details

when these are expected to be supported by retrieved research evidence.

If evidence is insufficient, output:

```text
INSUFFICIENT_EVIDENCE: <specific missing evidence>
```

Example:

```text
INSUFFICIENT_EVIDENCE: No retrieved source provides a
verified latency comparison between the two architectures.
```

---

# 25. Extraction Over Abstractive Rewriting

For:

* equations
* mathematical notation
* hardware registers
* variable names
* API identifiers
* source-code fragments
* numerical measurements

the system should preserve exact tokens whenever accuracy requires it.

The synthesis agent may explain such content, but must not silently alter technical identifiers.

---

# 26. Contradictory Evidence

If multiple sources disagree, the system must not silently select one claim.

Instead:

```text
Source A reports X.
Source B reports Y.
```

The answer should identify the disagreement and provide the relevant citations.

Where appropriate, source quality and publication date may be used to contextualize the disagreement, but must not be fabricated.

---

# 27. Tool: Hybrid Retrieval

## Tool: `hybrid_retrieve`

Purpose:

Execute dense and sparse retrieval, perform RRF fusion, and apply FlashRank reranking.

Schema:

```json
{
  "name": "hybrid_retrieve",
  "description": "Retrieves candidate passages using dense and sparse search, fuses results with weighted RRF, and reranks candidates using FlashRank.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The normalized user query or decomposed research sub-query."
      },
      "top_k": {
        "type": "integer",
        "minimum": 1,
        "maximum": 10,
        "default": 3
      }
    },
    "required": ["query"]
  }
}
```

The tool should return:

```text
passage
document_id
arxiv_id
title
section
page
chunk_index
dense_rank
sparse_rank
rrf_score
rerank_score
```

---

# 28. Operational Workflow

```text
[START]
   │
   ▼
1. Query Normalization
   │
   ├─ Remove conversational filler
   ├─ Extract technical terminology
   ├─ Identify constraints
   └─ Identify temporal/source requirements
   │
   ▼
2. Query Classification
   │
   ├─ Simple question
   └─ Complex research question
   │
   ▼
3. Query Decomposition
   │
   └─ Generate subqueries where necessary
   │
   ▼
4. Hybrid Retrieval
   │
   ├─ Dense ChromaDB search
   ├─ Sparse SQLite FTS5 search
   └─ Weighted RRF
   │
   ▼
5. Candidate Pool
   │
   └─ Top 15
   │
   ▼
6. FlashRank Reranking
   │
   └─ Rank candidates
   │
   ▼
7. Evidence Sufficiency
   │
   ├─ candidate_count >= MIN_CANDIDATES
   ├─ top_score >= MIN_RERANK_SCORE
   └─ required query aspects covered
   │
   ├─────────────── TRUE ───────────────┐
   │                                    │
   ▼                                    │
8. Citation Validation                  │
   │                                    │
   ▼                                    │
9. Grounded Synthesis                   │
   │                                    │
   └────────────────────────────────────┤
                                        │
                        FALSE           │
                          │             │
                          ▼             │
                 10. arXiv Discovery    │
                          │             │
                          ▼             │
                 11. Deduplicate Papers │
                          │             │
                          ▼             │
                 12. Select Papers      │
                          │             │
                          ▼             │
                 13. Ingestion Worker   │
                          │             │
                          ▼             │
                 14. Parse + Chunk      │
                          │             │
                          ▼             │
                 15. Embed + Index      │
                          │             │
                          ▼             │
                 16. Re-run Retrieval  │
                          │             │
                          └─────────────┘
```

Maximum discovery rounds must be enforced.

If evidence remains insufficient after the configured discovery limit:

```text
INSUFFICIENT_EVIDENCE:
<specific missing information>
```

---

# 29. Environment Configuration

The application expects:

```env
# =========================
# Groq Cloud
# =========================

GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# =========================
# Google Gemini
# =========================

GOOGLE_API_KEY=your_google_api_key_here
GEMINI_EMBEDDING_MODEL=models/text-embedding-004
EMBEDDING_DIMENSION=768

# =========================
# Storage
# =========================

PERSIST_DIRECTORY=./data/chroma_db
DOWNLOAD_CACHE_DIR=./data/downloads
SQLITE_SPARSE_DB=./data/sparse_index.db

# =========================
# Retrieval
# =========================

RRF_DENSE_WEIGHT=0.6
RRF_SPARSE_WEIGHT=0.4
RRF_K_CONSTANT=60

RRF_CANDIDATE_K=15
FINAL_TOP_K=3

# =========================
# Reranking
# =========================

RERANKER_MODEL=ms-marco-MiniLM-L-12-v2

# =========================
# Sufficiency
# =========================

MIN_CANDIDATES=1
MIN_RERANK_SCORE=0.70

# =========================
# Chunking
# =========================

CHUNK_TARGET_TOKENS=700
CHUNK_MAX_TOKENS=1000
CHUNK_OVERLAP_TOKENS=100
MIN_CHUNK_CHARACTERS=80

# =========================
# Active Discovery
# =========================

DISCOVERY_MAX_RESULTS=5
MAX_NEW_DOCUMENTS_PER_QUERY=3
MAX_DISCOVERY_ROUNDS=2

# =========================
# Download Security
# =========================

MAX_PDF_SIZE_MB=50
DOWNLOAD_TIMEOUT_SECONDS=30
```

API keys must never be committed to source control.

---

# 30. Performance Targets

Performance figures are **engineering targets**, not guaranteed SLAs.

They must be benchmarked on the actual deployment environment.

| Pipeline Stage        |               Target |               Cost |
| --------------------- | -------------------: | -----------------: |
| Query routing         |              <250 ms |         LLM tokens |
| ChromaDB retrieval    |               <40 ms |      No LLM tokens |
| SQLite FTS5 retrieval |               <10 ms |      No LLM tokens |
| RRF fusion            |                <5 ms |      No LLM tokens |
| FlashRank reranking   |               <20 ms |      No LLM tokens |
| PDF ingestion         | <4.5 s typical paper | Embedding/API cost |
| Final synthesis TTFT  |       <800 ms target |         LLM tokens |

Production benchmarking should record:

```text
p50
p95
p99
error rate
throughput
```

for each pipeline stage.

Latency must be measured separately for:

* cached documents
* new document ingestion
* simple queries
* complex multi-query requests

---

# 31. Evaluation Framework

Production readiness requires a retrieval evaluation dataset.

The system should measure:

### Retrieval

```text
Recall@K
Precision@K
MRR
nDCG
```

### Reranking

```text
Precision@3
MRR@3
nDCG@3
```

### Grounding

```text
Citation accuracy
Citation completeness
Unsupported claim rate
Evidence coverage
```

### Active Discovery

```text
Discovery success rate
New-document usefulness
Duplicate ingestion rate
Ingestion failure rate
```

### Generation

```text
Answer correctness
Faithfulness
Unsupported claim rate
```

The `MIN_RERANK_SCORE` threshold should be calibrated against these evaluation metrics.

---

# 32. Observability

Every request should have a unique:

```text
request_id
```

Logs should record:

```text
request_id
timestamp
query
normalized_query
subqueries
retrieval_count
dense_results
sparse_results
RRF_candidates
rerank_scores
top_score
sufficiency_decision
papers_discovered
papers_ingested
citation_ids
final_status
latency
errors
```

Sensitive API keys and credentials must never be logged.

---

# 33. Retry Policy

Transient failures should use bounded exponential backoff.

Retryable examples:

```text
HTTP 429
HTTP 500
HTTP 502
HTTP 503
HTTP 504
network timeout
temporary API unavailable
```

Non-retryable examples:

```text
invalid API key
invalid PDF
unsupported document
invalid arXiv ID
path validation failure
malformed tool parameters
```

Recommended:

```text
MAX_RETRIES=3
```

with exponential backoff and jitter.

---

# 34. Rate Limiting and Resource Controls

The system must enforce limits on:

```text
Maximum papers discovered per request
Maximum papers ingested per request
Maximum discovery rounds
Maximum PDF size
Maximum chunk count per document
Maximum concurrent ingestion workers
Maximum LLM output tokens
```

These controls prevent runaway API usage and accidental resource exhaustion.

---

# 35. Versioning

Indexes must be versioned when changing:

* embedding model
* embedding dimension
* chunking strategy
* tokenizer
* metadata schema
* reranker model

Recommended:

```text
INDEX_VERSION=v1
EMBEDDING_VERSION=text-embedding-004-768
RERANKER_VERSION=ms-marco-MiniLM-L-12-v2
```

Changing embedding dimensions requires rebuilding the affected vector index.

---

# 36. Data Consistency

The dense and sparse indexes must represent the same logical document/chunk set.

Each chunk must have a stable:

```text
chunk_id
```

Recommended:

```text
chunk_id =
<document_id>:<chunk_index>
```

Example:

```text
2407.08608:12
```

The same `chunk_id` must be stored in:

```text
ChromaDB
SQLite FTS5
metadata store
citation resolver
```

This provides deterministic cross-index provenance.

---

# 37. Final Synthesis Rules

The synthesis model receives:

```text
User query
+
Retrieved passages
+
Provenance metadata
+
Evidence sufficiency result
```

The model must:

1. Answer only from supported evidence for research claims.
2. Cite evidence-dependent claims.
3. Preserve important technical tokens.
4. Identify contradictions.
5. Clearly distinguish evidence from interpretation.
6. State when evidence is insufficient.
7. Never invent citation identifiers.
8. Never claim that a source was consulted unless it was actually retrieved.
9. Never fabricate paper titles, authors, metrics, dates, or experimental results.

---

# 38. Final System Invariant

The system must enforce the following invariant:

```text
NO VERIFIED EVIDENCE
        ↓
NO UNSUPPORTED TECHNICAL CLAIM
```

The intended behavior is:

```text
Local evidence sufficient
        ↓
Grounded synthesis

Local evidence insufficient
        ↓
Active literature discovery
        ↓
Document ingestion
        ↓
Re-retrieval
        ↓
Grounded synthesis

Still insufficient
        ↓
INSUFFICIENT_EVIDENCE
```

The system must prefer:

> **"I do not have sufficient evidence to verify this claim."**

over an unsupported answer.

---

# 39. Core Architectural Principle

The system is designed around the following loop:

```text
RETRIEVE
   ↓
RERANK
   ↓
EVALUATE EVIDENCE
   ↓
 ┌───────────────┐
 │               │
SUFFICIENT    INSUFFICIENT
 │               │
 ▼               ▼
ANSWER        DISCOVER
                 ↓
               INGEST
                 ↓
              RE-INDEX
                 ↓
              RETRIEVE
                 ↓
              RERANK
                 ↓
              EVALUATE
                 ↓
               ANSWER
```

This loop transforms the system from a static document-question-answering application into an **active academic research assistant capable of discovering and incorporating new literature during an ongoing research session**.

# End of AGENTS.md
