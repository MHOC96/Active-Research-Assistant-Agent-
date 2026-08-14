"""FlashRank cross-encoder reranker."""

from __future__ import annotations

from flashrank import Ranker, RerankRequest

from research_assistant.config import Settings, get_settings
from research_assistant.models import RetrievalHit


class FlashRankReranker:
    """Local CPU reranker using ms-marco-MiniLM-L-12-v2."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        try:
            self._ranker = Ranker(model_name=self.settings.reranker_model)
        except Exception as exc:
            raise RuntimeError(f"RERANKING_FAILED: {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalHit],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        if not candidates:
            return []

        top_k = top_k or self.settings.final_top_k
        passages = [
            {"id": hit.chunk_id, "text": hit.passage, "meta": {"chunk_id": hit.chunk_id}}
            for hit in candidates
        ]

        try:
            request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(request)
        except Exception as exc:
            raise RuntimeError(f"RERANKING_FAILED: {exc}") from exc

        score_by_id = {str(item["id"]): float(item["score"]) for item in results}
        reranked: list[RetrievalHit] = []
        for item in results:
            chunk_id = str(item["id"])
            original = next((h for h in candidates if h.chunk_id == chunk_id), None)
            if original is None:
                continue
            reranked.append(
                original.model_copy(update={"rerank_score": score_by_id.get(chunk_id)})
            )

        # Preserve FlashRank ordering; fill any missing candidates with zero score.
        seen = {h.chunk_id for h in reranked}
        for hit in candidates:
            if hit.chunk_id not in seen:
                reranked.append(hit.model_copy(update={"rerank_score": 0.0}))

        reranked.sort(key=lambda h: h.rerank_score or 0.0, reverse=True)
        return reranked[:top_k]
