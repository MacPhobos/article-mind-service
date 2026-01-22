"""RAG pipeline orchestration for chat Q&A."""

import re
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.chat.prompts import (
    build_system_prompt,
    format_context_with_metadata,
)
from article_mind_service.chat.providers import ProviderName, get_llm_provider
from article_mind_service.config import settings
from article_mind_service.embeddings import get_embedding_provider
from article_mind_service.logging_config import get_logger
from article_mind_service.schemas.search import SearchMode, SearchRequest
from article_mind_service.search.hybrid_search import HybridSearch

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Response from RAG pipeline.

    Attributes:
        content: Generated answer text
        sources: List of source citation metadata
        llm_provider: Provider used for generation
        llm_model: Model used for generation
        tokens_used: Total tokens consumed
        chunks_retrieved: Number of chunks retrieved from search
        retrieval_metadata: Search and retrieval metadata (P2 enhancement)
        context_chunks: Full chunks used for context with scores (P2 enhancement)
    """

    content: str
    sources: list[dict[str, Any]]
    llm_provider: str
    llm_model: str
    tokens_used: int
    chunks_retrieved: int
    retrieval_metadata: dict[str, Any] | None = None
    context_chunks: list[dict[str, Any]] | None = None


class RAGPipeline:
    """Orchestrates RAG (Retrieve-Augment-Generate) pipeline.

    Flow:
    1. Retrieve relevant chunks via R6 search API
    2. Format context with citation metadata
    3. Generate answer using LLM provider
    4. Extract and map citations to sources

    Usage:
        pipeline = RAGPipeline()
        response = await pipeline.query(
            session_id=123,
            question="What is...",
            db=db_session,
        )
    """

    def __init__(
        self,
        provider: ProviderName | None = None,
        max_context_chunks: int | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            provider: LLM provider to use (explicit override)
            max_context_chunks: Max chunks to include (default from settings)
        """
        self.provider_override = provider
        self.max_context_chunks = max_context_chunks or settings.rag_context_chunks
        self._llm_provider: Any = None
        self._db_session: AsyncSession | None = None

    async def _get_llm_provider(self, db: AsyncSession | None = None) -> Any:
        """Get LLM provider using database settings when available.

        Design Decision: Lazy initialization with database support
        - Rationale: Allows db session to be passed at query time
        - Provider can use database settings or fall back to .env
        - Cached after first call to avoid repeated database queries

        Args:
            db: Database session for reading provider settings

        Returns:
            Configured LLM provider instance
        """
        if self._llm_provider is None:
            self._llm_provider = await get_llm_provider(
                db=db,
                provider_override=self.provider_override,
            )
        return self._llm_provider

    async def query(
        self,
        session_id: int,
        question: str,
        db: AsyncSession,
        search_client: Any | None = None,
    ) -> RAGResponse:
        """Execute RAG pipeline for a question.

        Args:
            session_id: Session ID to search within
            question: User's question
            db: Database session
            search_client: Optional search client (for testing)

        Returns:
            RAGResponse with answer and sources
        """
        start_time = time.time()

        logger.info(
            "rag.query.start",
            session_id=session_id,
            question=question[:100],  # Truncate for readability
            max_chunks=self.max_context_chunks,
        )

        # Step 1: Retrieve relevant chunks
        chunks = await self._retrieve_chunks(
            session_id=session_id,
            query=question,
            limit=self.max_context_chunks,
            search_client=search_client,
        )

        logger.info(
            "rag.query.chunks_retrieved",
            session_id=session_id,
            chunks_count=len(chunks),
            chunk_ids=[c.get("chunk_id") for c in chunks][:5],  # First 5 IDs only
        )

        # Debug: Log detailed chunk content preview
        if chunks:
            total_chars = sum(len(c.get("content", "")) for c in chunks)
            logger.debug(
                "rag.query.context_preview",
                session_id=session_id,
                total_context_chars=total_chars,
            )
            for idx, chunk in enumerate(chunks, start=1):
                content = chunk.get("content", "")
                chunk_id = chunk.get("chunk_id", "unknown")
                title = chunk.get("title", "Unknown Article")
                # Truncate long content to 100 chars with ellipsis
                preview = (
                    content[:100] + "..."
                    if len(content) > 100
                    else content
                )
                logger.debug(
                    "rag.query.chunk_detail",
                    session_id=session_id,
                    chunk_number=idx,
                    chunk_id=chunk_id,
                    article_title=title,
                    content_preview=preview,
                    content_chars=len(content),
                )

        # Step 2: Format context with metadata
        context_str, source_metadata = format_context_with_metadata(chunks)
        has_context = len(chunks) > 0

        logger.debug(
            "rag.query.context_formatted",
            session_id=session_id,
            context_length=len(context_str),
            context_preview=context_str[:500],  # First 500 chars for debugging
            has_context=has_context,
        )

        # Step 3: Generate answer
        system_prompt = build_system_prompt(has_context=has_context)

        # Get LLM provider with database settings support
        llm_provider = await self._get_llm_provider(db=db)

        llm_response = await llm_provider.generate(
            system_prompt=system_prompt,
            user_message=question,
            context_chunks=[context_str] if has_context else [],
            max_tokens=settings.llm_max_tokens,
            temperature=0.3,
        )

        logger.info(
            "rag.query.llm_response",
            session_id=session_id,
            provider=llm_response.provider,
            model=llm_response.model,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            total_tokens=llm_response.total_tokens,
        )

        # Step 4: Extract cited sources only
        cited_sources = self._extract_cited_sources(
            content=llm_response.content,
            source_metadata=source_metadata,
        )

        logger.info(
            "rag.query.citations_extracted",
            session_id=session_id,
            cited_count=len(cited_sources),
            total_retrieved=len(chunks),
        )

        # Step 5: Build retrieval metadata (P2 enhancement)
        search_timing_ms = int((time.time() - start_time) * 1000)

        retrieval_metadata = {
            "search_mode": "hybrid",
            "chunks_retrieved": len(chunks),
            "chunks_cited": len(cited_sources),
            "search_timing_ms": search_timing_ms,
            "max_context_chunks": self.max_context_chunks,
        }

        # Step 6: Build context chunks audit trail (P2 enhancement)
        cited_chunk_ids = {s.get("chunk_id") for s in cited_sources}
        context_chunks = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "article_id": chunk.get("article_id"),
                "content": chunk.get("content", ""),
                "score": chunk.get("score"),
                "dense_rank": chunk.get("dense_rank"),
                "sparse_rank": chunk.get("sparse_rank"),
                "cited": chunk.get("chunk_id") in cited_chunk_ids,
            }
            for chunk in chunks
        ]

        return RAGResponse(
            content=llm_response.content,
            sources=cited_sources,
            llm_provider=llm_response.provider,
            llm_model=llm_response.model,
            tokens_used=llm_response.total_tokens,
            chunks_retrieved=len(chunks),
            retrieval_metadata=retrieval_metadata,
            context_chunks=context_chunks,
        )

    async def _retrieve_chunks(
        self,
        session_id: int,
        query: str,
        limit: int,
        search_client: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks using HybridSearch.

        Args:
            session_id: Session to search within
            query: Search query
            limit: Max results
            search_client: Optional client for testing

        Returns:
            List of chunk dictionaries with content and metadata

        Error Handling:
        - Returns empty list if search fails (logs error)
        - Returns empty list if no indexed content exists
        - Gracefully degrades to prevent chat from crashing
        """
        # Use injected client for testing
        if search_client:
            results = await search_client.search(
                session_id=session_id,
                query=query,
                limit=limit,
            )
            return [
                {
                    "content": r.get("content", ""),
                    "article_id": r.get("article_id"),
                    "chunk_id": r.get("chunk_id"),
                    "title": r.get("article", {}).get("title"),
                    "url": r.get("article", {}).get("url"),
                }
                for r in results.get("results", [])
            ]

        try:
            # Initialize HybridSearch if needed
            search = HybridSearch()

            # Generate query embedding
            logger.debug(
                "rag.retrieve_chunks.embedding_start",
                session_id=session_id,
                query=query[:100],
                limit=limit,
            )

            try:
                provider = await get_embedding_provider()
                embeddings = await provider.embed([query])
                query_embedding = embeddings[0]
            except Exception as e:
                logger.error(
                    "rag.retrieve_chunks.embedding_failed",
                    session_id=session_id,
                    error=str(e),
                    exc_info=True,
                )
                return []

            # Create search request
            request = SearchRequest(
                query=query,
                top_k=limit,
                include_content=True,
                search_mode=SearchMode.HYBRID,
            )

            # Execute search
            response = await search.search(
                session_id=session_id,
                request=request,
                query_embedding=query_embedding,
            )

            logger.info(
                "rag.retrieve_chunks.search_complete",
                session_id=session_id,
                results_count=len(response.results),
                total_chunks_searched=response.total_chunks_searched,
                search_mode=response.search_mode.value,
                timing_ms=response.timing_ms,
            )

            # No results - return empty
            if not response.results:
                logger.warning(
                    "rag.retrieve_chunks.no_results",
                    session_id=session_id,
                )
                return []

            # Get database session to fetch article metadata
            # Note: We don't have direct DB access here, so we'll use the metadata
            # from the search results which includes article_id, source_url, source_title

            # Transform SearchResult to expected chunk format
            chunks = []
            for result in response.results:
                chunks.append(
                    {
                        "content": result.content or "",
                        "article_id": result.article_id,
                        "chunk_id": result.chunk_id,
                        "title": result.source_title,
                        "url": result.source_url,
                        "score": result.score,
                        "dense_rank": result.dense_rank,
                        "sparse_rank": result.sparse_rank,
                    }
                )

            return chunks

        except Exception as e:
            # Log error but don't crash chat
            logger.error(
                "rag.retrieve_chunks.search_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            return []

    def _extract_cited_sources(
        self,
        content: str,
        source_metadata: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Extract only the sources that were actually cited in the response.

        Args:
            content: LLM response content
            source_metadata: All available source metadata

        Returns:
            List of source metadata for cited sources only
        """
        # Find all citation numbers in the response
        citation_pattern = r"\[(\d+)\]"
        cited_numbers = {int(m) for m in re.findall(citation_pattern, content)}

        # Filter to only cited sources
        cited_sources = [
            source
            for source in source_metadata
            if source.get("citation_index") in cited_numbers
        ]

        return cited_sources
