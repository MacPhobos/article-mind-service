"""Settings API endpoints for provider configuration.

This router provides administrative endpoints for managing provider settings,
specifically embedding and LLM providers. These settings control which AI services
are used for generating embeddings and LLM responses.

Design Decisions:

1. Endpoint Structure:
   - GET /api/v1/settings/providers - Get current provider configuration
   - PATCH /api/v1/settings/providers/embedding - Update embedding provider
   - PATCH /api/v1/settings/providers/llm - Update LLM provider

   Rationale: RESTful design with provider-specific endpoints
   Trade-off: Multiple endpoints vs. single update endpoint with type field

2. Dimension Change Handling:
   - Returns 400 if embedding provider dimension changes without confirmation
   - Requires confirm_reindex=True to proceed with dimension change
   - Rationale: Prevents accidental data loss from dimension mismatch
   - Trade-off: Extra API call required vs. more explicit/safe

3. Background Reindex Integration:
   - Uses existing reindex infrastructure from admin router
   - Triggers reindex asynchronously using BackgroundTasks
   - Rationale: Reuses tested reindex logic, consistent behavior
   - Trade-off: Coupling with admin module vs. code duplication

4. Error Handling:
   - 400 for dimension mismatch without confirmation
   - 400 for provider not available (missing API key)
   - 422 for invalid provider values (handled by Pydantic)
   - Rationale: Clear error messages guide admin to correct action

Performance:
- GET /api/v1/settings/providers: O(1) database query (singleton)
- PATCH operations: O(1) database update
- Reindex trigger: Background task, returns immediately

Error Cases:
1. Dimension Change Without Confirmation:
   - Error: 400 Bad Request
   - Message: Includes current/new dimensions, suggests confirm_reindex=True
   - Recovery: Retry with confirm_reindex=True

2. Provider Not Available:
   - Error: 400 Bad Request
   - Message: "Provider 'X' not configured. Check API keys in .env"
   - Recovery: Configure API key in .env and restart service

3. Database Error:
   - Error: 500 Internal Server Error
   - Message: Generic database error message
   - Recovery: Check database connectivity, retry operation
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.database import AsyncSessionLocal, get_db
from article_mind_service.schemas.settings import (
    ProviderConfigResponse,
    UpdateEmbeddingProviderRequest,
    UpdateEmbeddingProviderResponse,
    UpdateLlmProviderRequest,
    UpdateLlmProviderResponse,
)
from article_mind_service.services.settings_service import (
    get_available_providers,
    get_settings,
    update_embedding_provider,
    update_llm_provider,
)
from article_mind_service.tasks import reindex_all_articles, task_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@router.get("/providers", response_model=ProviderConfigResponse)
async def get_provider_config(
    db: AsyncSession = Depends(get_db),
) -> ProviderConfigResponse:
    """Get current provider configuration and available options.

    Returns the currently selected embedding and LLM providers, along with
    lists of available providers based on configured API keys in .env.

    Available providers depend on API key configuration:
    - openai: Requires OPENAI_API_KEY
    - anthropic: Requires ANTHROPIC_API_KEY
    - ollama: Always available (local, no API key needed)

    Returns:
        ProviderConfigResponse with current and available providers

    Example:
        GET /api/v1/settings/providers

        Response:
        {
          "embedding_provider": "openai",
          "embedding_provider_available": ["openai", "ollama"],
          "llm_provider": "openai",
          "llm_provider_available": ["openai", "anthropic"]
        }
    """
    # Get current settings from database
    settings = await get_settings(db)

    # Get available providers based on API keys
    available = get_available_providers()

    return ProviderConfigResponse(
        embedding_provider=settings.embedding_provider,
        embedding_provider_available=available["embedding"],
        llm_provider=settings.llm_provider,
        llm_provider_available=available["llm"],
    )


@router.patch("/providers/embedding", response_model=UpdateEmbeddingProviderResponse)
async def update_embedding_provider_endpoint(
    request: UpdateEmbeddingProviderRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> UpdateEmbeddingProviderResponse:
    """Update embedding provider with optional reindex trigger.

    Switching embedding providers may require reindexing if embedding dimensions
    change. Different providers use different dimension sizes:
    - OpenAI text-embedding-3-small: 1536 dimensions
    - Ollama nomic-embed-text: 1024 dimensions

    If dimensions change without confirm_reindex=True, returns 400 error.
    If confirm_reindex=True, updates provider and triggers background reindex.

    Args:
        request: UpdateEmbeddingProviderRequest with provider and confirm_reindex
        background_tasks: FastAPI background task manager
        db: Database session (dependency injected)

    Returns:
        UpdateEmbeddingProviderResponse with provider, reindex status, and warning

    Raises:
        HTTPException: 400 if provider not available or dimension mismatch without confirmation

    Example (No Dimension Change):
        PATCH /api/v1/settings/providers/embedding
        {
          "provider": "openai",
          "confirm_reindex": false
        }

        Response:
        {
          "provider": "openai",
          "reindex_triggered": false,
          "warning": null
        }

    Example (Dimension Change With Confirmation):
        PATCH /api/v1/settings/providers/embedding
        {
          "provider": "ollama",
          "confirm_reindex": true
        }

        Response:
        {
          "provider": "ollama",
          "reindex_triggered": true,
          "warning": null
        }

    Example (Dimension Change Without Confirmation):
        PATCH /api/v1/settings/providers/embedding
        {
          "provider": "ollama",
          "confirm_reindex": false
        }

        Response: 400 Bad Request
        {
          "detail": "Embedding dimension change detected (1536 -> 1024). Set confirm_reindex=true to proceed with reindexing."
        }
    """
    # Check if provider is available (has API key configured)
    available = get_available_providers()
    if request.provider not in available["embedding"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{request.provider}' not configured. Check API keys in .env",
        )

    # Get current provider BEFORE updating (for error message)
    current_settings = await get_settings(db)
    old_provider = current_settings.embedding_provider

    # Update provider and check if dimension changed
    settings, dimension_changed = await update_embedding_provider(db, request.provider)

    # If dimension changed, require confirmation
    if dimension_changed and not request.confirm_reindex:
        # Get dimension values for error message (using old_provider from BEFORE update)
        old_dim = 1536 if old_provider == "openai" else 1024
        new_dim = 1024 if request.provider == "ollama" else 1536

        # Rollback the transaction (provider was already updated with flush())
        await db.rollback()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Embedding dimension change detected ({old_dim} -> {new_dim}). "
            f"Set confirm_reindex=true to proceed with reindexing.",
        )

    # If dimension changed and confirmed, trigger reindex
    reindex_triggered = False
    if dimension_changed and request.confirm_reindex:
        # Create reindex task in registry
        task_id = task_registry.create_task(
            task_type="reindex_embedding",
            total_items=0,  # Will be calculated in background task
        )

        logger.info(
            f"Embedding provider changed to {request.provider}, "
            f"triggering reindex: task_id={task_id}"
        )

        # Queue background reindex task
        # Reindex ALL sessions (session_ids=None) with force=True
        background_tasks.add_task(
            reindex_all_articles,
            task_id=task_id,
            session_ids=None,  # Reindex all sessions
            force=True,  # Reindex even if embedding_status="completed"
            task_registry=task_registry,
            db_session_factory=AsyncSessionLocal,
        )

        reindex_triggered = True

    # Commit database changes
    await db.commit()

    return UpdateEmbeddingProviderResponse(
        provider=request.provider,
        reindex_triggered=reindex_triggered,
        warning=None,
    )


@router.patch("/providers/llm", response_model=UpdateLlmProviderResponse)
async def update_llm_provider_endpoint(
    request: UpdateLlmProviderRequest,
    db: AsyncSession = Depends(get_db),
) -> UpdateLlmProviderResponse:
    """Update LLM provider (no reindex needed).

    Switching LLM providers does not require reindexing because embeddings
    are not affected. This is a simple provider switch operation.

    Args:
        request: UpdateLlmProviderRequest with provider
        db: Database session (dependency injected)

    Returns:
        UpdateLlmProviderResponse with new provider

    Raises:
        HTTPException: 400 if provider not available (missing API key)

    Example:
        PATCH /api/v1/settings/providers/llm
        {
          "provider": "anthropic"
        }

        Response:
        {
          "provider": "anthropic"
        }
    """
    # Check if provider is available (has API key configured)
    available = get_available_providers()
    if request.provider not in available["llm"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{request.provider}' not configured. Check API keys in .env",
        )

    # Update provider
    settings = await update_llm_provider(db, request.provider)

    # Commit database changes
    await db.commit()

    logger.info(f"LLM provider updated to {request.provider}")

    return UpdateLlmProviderResponse(provider=settings.llm_provider)
