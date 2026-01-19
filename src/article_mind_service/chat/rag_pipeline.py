"""RAG pipeline orchestration for chat Q&A."""

import re
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.chat.prompts import (
    build_system_prompt,
    format_context_with_metadata,
)
from article_mind_service.chat.providers import ProviderName, get_llm_provider
from article_mind_service.config import settings


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
    """

    content: str
    sources: list[dict[str, Any]]
    llm_provider: str
    llm_model: str
    tokens_used: int
    chunks_retrieved: int


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
            provider: LLM provider to use (default from settings)
            max_context_chunks: Max chunks to include (default from settings)
        """
        self.provider_name = provider or settings.llm_provider
        self.max_context_chunks = max_context_chunks or settings.rag_context_chunks
        self._llm_provider: Any = None

    @property
    def llm_provider(self) -> Any:
        """Lazy-initialize LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider(self.provider_name)
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
        # Step 1: Retrieve relevant chunks
        chunks = await self._retrieve_chunks(
            session_id=session_id,
            query=question,
            limit=self.max_context_chunks,
            search_client=search_client,
        )

        # Step 2: Format context with metadata
        context_str, source_metadata = format_context_with_metadata(chunks)
        has_context = len(chunks) > 0

        # Step 3: Generate answer
        system_prompt = build_system_prompt(has_context=has_context)

        llm_response = await self.llm_provider.generate(
            system_prompt=system_prompt,
            user_message=question,
            context_chunks=[context_str] if has_context else [],
            max_tokens=settings.llm_max_tokens,
            temperature=0.3,
        )

        # Step 4: Extract cited sources only
        cited_sources = self._extract_cited_sources(
            content=llm_response.content,
            source_metadata=source_metadata,
        )

        return RAGResponse(
            content=llm_response.content,
            sources=cited_sources,
            llm_provider=llm_response.provider,
            llm_model=llm_response.model,
            tokens_used=llm_response.total_tokens,
            chunks_retrieved=len(chunks),
        )

    async def _retrieve_chunks(
        self,
        session_id: int,
        query: str,
        limit: int,
        search_client: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks from R6 search API.

        Args:
            session_id: Session to search within
            query: Search query
            limit: Max results
            search_client: Optional client for testing

        Returns:
            List of chunk dictionaries with content and metadata
        """
        # TODO: Integrate with R6 search API when available
        # For now, return empty to demonstrate flow
        #
        # Expected R6 API call:
        # GET /api/v1/sessions/{session_id}/search?q={query}&limit={limit}
        #
        # Expected response:
        # {
        #     "results": [
        #         {
        #             "chunk_id": "...",
        #             "article_id": 123,
        #             "content": "...",
        #             "score": 0.85,
        #             "article": {
        #                 "title": "...",
        #                 "url": "..."
        #             }
        #         }
        #     ]
        # }

        if search_client:
            # Use injected client for testing
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

        # Placeholder until R6 is implemented
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
