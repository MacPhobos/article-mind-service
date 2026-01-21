"""Chat API endpoints for Q&A functionality."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.chat.rag_pipeline import RAGPipeline
from article_mind_service.database import get_db
from article_mind_service.models.chat import ChatMessage
from article_mind_service.schemas.chat import (
    ChatHistoryResponse,
    ChatMessageResponse,
    ChatRequest,
    ChatResponse,
    ChatSource,
    RetrievalMetadata,
)

router = APIRouter(
    prefix="/api/v1/sessions/{session_id}/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send chat message and get response",
    description="""
Send a question about the session's content and receive an LLM-generated
answer with source citations.

The RAG pipeline:
1. Retrieves relevant content chunks from the session's articles
2. Augments the prompt with context
3. Generates an answer using the configured LLM
4. Returns the response with inline citations [1], [2], etc.

Both the user message and assistant response are persisted to the database.
""",
)
async def send_chat_message(
    session_id: int,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Send a chat message and get an AI-generated response.

    Args:
        session_id: ID of the session to query
        request: Chat message request body
        db: Database session

    Returns:
        ChatResponse with assistant's answer and citations

    Raises:
        HTTPException 404: If session not found
        HTTPException 500: If LLM generation fails
    """
    # TODO: Verify session exists (when Session model is implemented)
    # result = await db.execute(select(Session).where(Session.id == session_id))
    # session = result.scalar_one_or_none()
    # if not session:
    #     raise HTTPException(status_code=404, detail="Session not found")

    # Save user message
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.message,
    )
    db.add(user_message)
    await db.flush()  # Get user_message.id

    # Execute RAG pipeline
    try:
        pipeline = RAGPipeline()
        rag_response = await pipeline.query(
            session_id=session_id,
            question=request.message,
            db=db,
        )
    except Exception as e:
        # Log error and return helpful message
        print(f"RAG pipeline error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}",
        ) from e

    # Save assistant response with retrieval metadata (P2 enhancement)
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=rag_response.content,
        sources=rag_response.sources,
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used,
        retrieval_metadata=rag_response.retrieval_metadata,
        context_chunks=rag_response.context_chunks,
    )
    db.add(assistant_message)
    await db.commit()

    # Build retrieval metadata from RAG response (P2 enhancement)
    retrieval_metadata = None
    if rag_response.retrieval_metadata:
        retrieval_metadata = RetrievalMetadata(
            chunks_retrieved=rag_response.retrieval_metadata.get("chunks_retrieved", 0),
            chunks_cited=rag_response.retrieval_metadata.get("chunks_cited", 0),
            search_mode=rag_response.retrieval_metadata.get("search_mode", "hybrid"),
            search_timing_ms=rag_response.retrieval_metadata.get("search_timing_ms"),
            total_chunks_in_session=rag_response.retrieval_metadata.get(
                "total_chunks_in_session"
            ),
        )

    # Build response
    return ChatResponse(
        message_id=assistant_message.id,
        content=assistant_message.content,
        sources=[ChatSource(**s) for s in rag_response.sources],
        llm_provider=rag_response.llm_provider,
        llm_model=rag_response.llm_model,
        tokens_used=rag_response.tokens_used,
        created_at=assistant_message.created_at,
        retrieval_metadata=retrieval_metadata,
    )


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get chat history",
    description="Retrieve all chat messages for a session in chronological order.",
)
async def get_chat_history(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> ChatHistoryResponse:
    """Get chat history for a session.

    Args:
        session_id: ID of the session
        db: Database session

    Returns:
        ChatHistoryResponse with all messages
    """
    # Fetch messages
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at.asc())
    )
    messages = result.scalars().all()

    # Count total
    count_result = await db.execute(
        select(func.count())
        .select_from(ChatMessage)
        .where(ChatMessage.session_id == session_id)
    )
    total = count_result.scalar_one()

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                sources=(
                    [ChatSource(**s) for s in msg.sources]  # type: ignore[arg-type]
                    if msg.sources
                    else None
                ),
                created_at=msg.created_at,
            )
            for msg in messages
        ],
        total_messages=total,
    )


@router.delete(
    "/history",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Clear chat history",
    description="Delete all chat messages for a session.",
)
async def clear_chat_history(
    session_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Clear all chat history for a session.

    Args:
        session_id: ID of the session
        db: Database session
    """
    await db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
    await db.commit()
