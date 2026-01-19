"""Optional cross-encoder reranking for search quality boost.

Cross-encoder reranking provides a second-stage ranking that evaluates
query-document pairs more accurately than first-stage retrieval.

Research Findings:
- Improves accuracy by 20-35% over retrieval alone
- Adds 200-500ms latency per batch of candidates
- Most beneficial for ambiguous or multi-intent queries
- Recommended to rerank top 20-50 candidates, not all results

Common Models:
- cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality (default)
- cross-encoder/ms-marco-TinyBERT-L-2: Faster, lower quality
- cross-encoder/ms-marco-MiniLM-L-12-v2: Slower, higher quality

Design Decision: Optional Feature
- Disabled by default (set SEARCH_RERANK_ENABLED=true to enable)
- Requires sentence-transformers library (optional dependency)
- Only run on top candidates (not all results) for efficiency
"""

from article_mind_service.config import settings


class Reranker:
    """Cross-encoder reranker for improving search result quality.

    Uses sentence-transformers cross-encoder models to score
    query-document pairs for more accurate ranking.

    Usage:
        reranker = Reranker()
        scores = await reranker.rerank(query, documents)

    Note: This is a placeholder implementation. To enable:
    1. Install: pip install sentence-transformers
    2. Set SEARCH_RERANK_ENABLED=true in .env
    3. Optionally configure SEARCH_RERANK_MODEL
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize reranker with model.

        Args:
            model_name: Cross-encoder model name (default from settings)
        """
        self.model_name = model_name or settings.search_rerank_model
        self._model = None

    def _load_model(self) -> object:
        """Lazy load the cross-encoder model.

        Returns:
            CrossEncoder model instance

        Raises:
            RuntimeError: If sentence-transformers is not installed
        """
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(self.model_name)
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of document texts to rank

        Returns:
            List of relevance scores (same order as input documents)

        Performance:
        - MiniLM-L-6: ~200-300ms for 20 documents
        - Latency grows linearly with number of documents
        - Run on GPU for faster inference in production

        Note: This is a placeholder. Actual implementation requires
        sentence-transformers library.
        """
        if not documents:
            return []

        # Placeholder: Return dummy scores if not enabled
        if not settings.search_rerank_enabled:
            return [0.5] * len(documents)

        # Load model on first use
        model = self._load_model()

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]

        # Get scores from cross-encoder
        scores = model.predict(pairs)  # type: ignore

        return [float(s) for s in scores]

    async def rerank_with_ids(
        self,
        query: str,
        documents: list[tuple[str, str]],  # [(id, content), ...]
    ) -> list[tuple[str, float]]:
        """Rerank documents and return with IDs.

        Args:
            query: Search query
            documents: List of (id, content) tuples

        Returns:
            List of (id, score) tuples sorted by score descending

        Usage:
            docs = [("chunk_1", "content 1"), ("chunk_2", "content 2")]
            reranked = await reranker.rerank_with_ids(query, docs)
        """
        if not documents:
            return []

        ids = [d[0] for d in documents]
        contents = [d[1] for d in documents]

        scores = await self.rerank(query, contents)

        # Combine and sort
        results = list(zip(ids, scores, strict=True))
        results.sort(key=lambda x: x[1], reverse=True)

        return results
