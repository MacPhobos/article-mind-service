"""System prompts for RAG Q&A pipeline."""

from typing import Any

# Main research assistant prompt
RESEARCH_ASSISTANT_PROMPT = """You are a research assistant helping users understand their saved articles and content. Your task is to answer questions based ONLY on the provided context.

## Instructions

1. **Answer from context only**: Base your answers solely on the numbered context passages provided. Do not use external knowledge.

2. **Cite your sources**: Use inline citations in the format [1], [2], etc. to reference the context passages that support your statements.

3. **Be honest about limitations**: If the context doesn't contain enough information to fully answer the question, clearly state what information is missing.

4. **Be concise but complete**: Provide thorough answers without unnecessary verbosity.

5. **Handle contradictions**: If sources contradict each other, acknowledge this and present both perspectives with their citations.

## Citation Format

- Use [1], [2], etc. corresponding to the numbered context passages
- Place citations at the end of the relevant sentence or claim
- Multiple sources supporting the same point can be cited together: [1][2]

## Response Guidelines

- Start with a direct answer when possible
- Support claims with citations
- If you cannot answer from the context, say: "Based on the available sources, I don't have enough information to answer this question."
- Never make up information or cite non-existent sources"""


# Fallback prompt when no context is found
NO_CONTEXT_PROMPT = """You are a research assistant. The user asked a question, but no relevant content was found in their saved articles.

Respond helpfully by:
1. Acknowledging that no relevant sources were found
2. Suggesting what kind of content they might need to add
3. Offering to help refine their question if it might be too specific

Be friendly and constructive."""


def build_system_prompt(has_context: bool = True) -> str:
    """Get appropriate system prompt based on context availability.

    Args:
        has_context: Whether context chunks were retrieved

    Returns:
        Appropriate system prompt string
    """
    return RESEARCH_ASSISTANT_PROMPT if has_context else NO_CONTEXT_PROMPT


def format_context_with_metadata(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Format context chunks with metadata for prompt injection.

    Args:
        chunks: List of chunk dictionaries with content and metadata

    Returns:
        Tuple of (formatted_context_string, source_metadata_list)

    Each chunk dict expected to have:
        - content: str (chunk text)
        - article_id: int
        - chunk_id: str
        - title: str | None
        - url: str | None
        - score: float | None (RRF relevance score)
        - dense_rank: int | None (semantic search rank)
        - sparse_rank: int | None (keyword search rank)

    Design Decision: Full content in source_metadata
    - Changed from 200-char excerpt to full content
    - Enables users to verify what content was actually used
    - Allows assessment of citation relevance
    - Trade-off: Larger response size vs. transparency
    """
    if not chunks:
        return "No relevant context found.", []

    formatted_lines = []
    source_metadata = []

    for i, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "")
        title = chunk.get("title", "Unknown Source")
        url = chunk.get("url", "")

        # Format context line
        source_ref = f"(Source: {title}"
        if url:
            source_ref += f", URL: {url}"
        source_ref += ")"

        formatted_lines.append(f"[{i}] {content}\n{source_ref}")

        # Track source metadata for response with full content and search metadata
        source_metadata.append(
            {
                "citation_index": i,
                "article_id": chunk.get("article_id"),
                "chunk_id": chunk.get("chunk_id"),
                "title": title,
                "url": url,
                # CHANGED: Full content instead of 200-char excerpt
                "content": content,
                # NEW: Search metadata for transparency
                "relevance_score": chunk.get("score"),
                "search_method": "hybrid",  # From search mode
                "dense_rank": chunk.get("dense_rank"),
                "sparse_rank": chunk.get("sparse_rank"),
            }
        )

    return "\n\n".join(formatted_lines), source_metadata
