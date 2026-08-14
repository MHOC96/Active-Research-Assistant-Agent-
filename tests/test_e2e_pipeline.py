"""End-to-end pipeline test with mocked external services."""

from unittest.mock import MagicMock

from research_assistant.config import Settings
from research_assistant.models import (
    ActiveResearchResult,
    HybridRetrieveResult,
    RetrievalHit,
    SufficiencyResult,
)
from research_assistant.orchestrator.agent import ResearchOrchestrator
from research_assistant.orchestrator.query_processor import QueryAnalysis


def test_e2e_pipeline_with_mocked_services(tmp_path):
    settings = Settings(
        PERSIST_DIRECTORY=str(tmp_path / "chroma"),
        SQLITE_SPARSE_DB=str(tmp_path / "sparse.db"),
        METADATA_DB=str(tmp_path / "metadata.db"),
        DOWNLOAD_CACHE_DIR=str(tmp_path / "downloads"),
        EMBEDDING_DIMENSION=768,
        FINAL_TOP_K=2,
    )

    hit = RetrievalHit(
        chunk_id="2407.08608:0",
        passage="Transformers use scaled dot-product attention.",
        document_id="2407.08608",
        arxiv_id="2407.08608",
        title="Attention Is All You Need",
        chunk_index=0,
        rerank_score=0.92,
    )
    active_result = ActiveResearchResult(
        query="transformer attention mechanism",
        request_id="pipeline-1",
        retrieval=HybridRetrieveResult(
            query="transformer attention mechanism",
            candidates=[hit],
            sufficiency=SufficiencyResult(
                sufficient=True,
                candidate_count=1,
                top_score=0.92,
            ),
        ),
    )

    pipeline = MagicMock()
    pipeline.settings = settings
    pipeline.run.return_value = active_result

    llm = MagicMock()
    query_processor = MagicMock()
    query_processor.analyze.return_value = QueryAnalysis(
        original_query="What is transformer attention?",
        normalized_query="transformer attention mechanism",
        query_type="simple",
        subqueries=["transformer attention mechanism"],
    )

    synthesizer = MagicMock()
    synthesizer.synthesize.return_value = (
        "Transformer attention uses scaled dot-product attention "
        "[arXiv:2407.08608 | Chunk 0].",
        True,
        [],
    )

    orchestrator = ResearchOrchestrator(
        pipeline=pipeline,
        llm=llm,
        query_processor=query_processor,
        synthesizer=synthesizer,
    )

    response = orchestrator.answer("What is transformer attention?")

    assert response.sufficient is True
    assert response.citations_valid is True
    assert "2407.08608" in response.answer
    pipeline.run.assert_called_once_with("transformer attention mechanism")
    synthesizer.synthesize.assert_called_once()
    synthesis_hits = synthesizer.synthesize.call_args.args[1]
    assert len(synthesis_hits) == 1
