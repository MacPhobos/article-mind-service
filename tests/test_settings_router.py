"""Tests for settings router endpoints.

Test Coverage:
1. GET /api/v1/settings/providers - Get current provider configuration
2. PATCH /api/v1/settings/providers/embedding - Update embedding provider
   - Successful update (same dimension)
   - Dimension change without confirmation (400 error)
   - Dimension change with confirmation (reindex triggered)
   - Provider not available (missing API key, 400 error)
   - Invalid provider value (422 validation error)
3. PATCH /api/v1/settings/providers/llm - Update LLM provider
   - Successful update
   - Provider not available (missing API key, 400 error)
   - Invalid provider value (422 validation error)

Mocking Strategy:
- Mock get_available_providers to control which providers are "available"
- Use db_session fixture for database isolation
- No ChromaDB or file system operations needed (settings are database-only)
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.models.provider_settings import ProviderSettings


@pytest.mark.asyncio
async def test_get_provider_config_initial_defaults(async_client: AsyncClient, db_session: AsyncSession):
    """Test GET /api/v1/settings/providers returns default values on first call.

    Expected behavior:
    - Settings singleton is created if it doesn't exist
    - Default providers: embedding=openai, llm=openai
    - Available providers depend on API keys in .env
    """
    # Mock available providers (simulate both OpenAI and Ollama available)
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai", "anthropic"],
        }

        response = await async_client.get("/api/v1/settings/providers")

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert data["embedding_provider"] == "openai"  # Default
    assert data["llm_provider"] == "openai"  # Default
    assert "openai" in data["embedding_provider_available"]
    assert "ollama" in data["embedding_provider_available"]
    assert "openai" in data["llm_provider_available"]
    assert "anthropic" in data["llm_provider_available"]

    # Verify settings were created in database
    stmt = select(ProviderSettings).where(ProviderSettings.id == 1)
    result = await db_session.execute(stmt)
    settings = result.scalar_one_or_none()
    assert settings is not None
    assert settings.embedding_provider == "openai"
    assert settings.llm_provider == "openai"


@pytest.mark.asyncio
async def test_get_provider_config_existing_settings(async_client: AsyncClient, db_session: AsyncSession):
    """Test GET /api/v1/settings/providers returns existing settings from database."""
    # Create settings with non-default values
    settings = ProviderSettings(
        id=1,
        embedding_provider="ollama",
        llm_provider="anthropic",
    )
    db_session.add(settings)
    await db_session.commit()

    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai", "anthropic"],
        }

        response = await async_client.get("/api/v1/settings/providers")

    assert response.status_code == 200
    data = response.json()

    # Verify it returns existing (non-default) values
    assert data["embedding_provider"] == "ollama"
    assert data["llm_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_update_llm_provider_success(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/llm successfully updates LLM provider."""
    # Create initial settings
    settings = ProviderSettings(id=1, embedding_provider="openai", llm_provider="openai")
    db_session.add(settings)
    await db_session.commit()

    # Mock provider availability (anthropic available)
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai", "anthropic"],
        }

        response = await async_client.patch(
            "/api/v1/settings/providers/llm",
            json={"provider": "anthropic"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "anthropic"

    # Verify database was updated
    await db_session.refresh(settings)
    assert settings.llm_provider == "anthropic"


@pytest.mark.asyncio
async def test_update_llm_provider_not_available(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/llm returns 400 if provider not available (missing API key)."""
    # Create initial settings
    settings = ProviderSettings(id=1, embedding_provider="openai", llm_provider="openai")
    db_session.add(settings)
    await db_session.commit()

    # Mock provider availability (anthropic NOT available)
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai"],  # anthropic missing
        }

        response = await async_client.patch(
            "/api/v1/settings/providers/llm",
            json={"provider": "anthropic"}
        )

    assert response.status_code == 400
    assert "not configured" in response.json()["detail"]
    assert "API keys" in response.json()["detail"]

    # Verify database was NOT updated
    await db_session.refresh(settings)
    assert settings.llm_provider == "openai"  # Unchanged


@pytest.mark.asyncio
async def test_update_llm_provider_invalid_value(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/llm returns 422 for invalid provider value."""
    response = await async_client.patch(
        "/api/v1/settings/providers/llm",
        json={"provider": "invalid_provider"}
    )

    assert response.status_code == 422
    # Pydantic validation error


@pytest.mark.asyncio
async def test_update_embedding_provider_same_dimension(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/embedding updates without reindex if dimension unchanged.

    OpenAI has 1536 dimensions. Switching between OpenAI models doesn't change dimension.
    In this test, we're simulating a same-dimension switch (though the current implementation
    only supports openai and ollama, which have different dimensions).
    """
    # Create initial settings (openai)
    settings = ProviderSettings(id=1, embedding_provider="openai", llm_provider="openai")
    db_session.add(settings)
    await db_session.commit()

    # Mock provider availability
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai"],
        }

        # Switch to openai again (same dimension, no reindex needed)
        response = await async_client.patch(
            "/api/v1/settings/providers/embedding",
            json={"provider": "openai", "confirm_reindex": False}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "openai"
    assert data["reindex_triggered"] is False
    assert data["warning"] is None


@pytest.mark.asyncio
async def test_update_embedding_provider_dimension_change_no_confirm(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/embedding returns 400 if dimension changes without confirmation.

    Expected behavior:
    - OpenAI (1536 dims) -> Ollama (1024 dims): dimension change detected
    - Without confirm_reindex=True: return 400 with error message
    - Error message includes old/new dimensions and suggest confirm_reindex=True
    """
    # Create initial settings (openai)
    settings = ProviderSettings(id=1, embedding_provider="openai", llm_provider="openai")
    db_session.add(settings)
    await db_session.commit()

    # Mock provider availability
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai"],
        }

        # Switch to ollama (dimension change) without confirmation
        response = await async_client.patch(
            "/api/v1/settings/providers/embedding",
            json={"provider": "ollama", "confirm_reindex": False}
        )

    assert response.status_code == 400
    detail = response.json()["detail"]

    # Verify error message contains useful information
    assert "dimension change" in detail.lower()
    assert "1536" in detail  # Old dimension
    assert "1024" in detail  # New dimension
    assert "confirm_reindex=true" in detail.lower()

    # Verify database was NOT updated (rollback on error)
    await db_session.refresh(settings)
    assert settings.embedding_provider == "openai"  # Unchanged


@pytest.mark.asyncio
async def test_update_embedding_provider_dimension_change_with_confirm(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/embedding triggers reindex when dimension changes with confirmation.

    Expected behavior:
    - OpenAI (1536 dims) -> Ollama (1024 dims) with confirm_reindex=True
    - Provider updated in database
    - Background reindex task created and triggered
    - Response indicates reindex_triggered=True
    """
    # Create initial settings (openai)
    settings = ProviderSettings(id=1, embedding_provider="openai", llm_provider="openai")
    db_session.add(settings)
    await db_session.commit()

    # Mock provider availability and background task
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available, \
         patch("article_mind_service.routers.settings.task_registry") as mock_registry, \
         patch("article_mind_service.routers.settings.reindex_all_articles") as mock_reindex:

        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai"],
        }

        # Mock task registry to return task_id
        mock_registry.create_task.return_value = "test-task-id-123"

        # Switch to ollama with confirmation
        response = await async_client.patch(
            "/api/v1/settings/providers/embedding",
            json={"provider": "ollama", "confirm_reindex": True}
        )

    assert response.status_code == 200
    data = response.json()

    # Verify response indicates reindex was triggered
    assert data["provider"] == "ollama"
    assert data["reindex_triggered"] is True
    assert data["warning"] is None

    # Verify database was updated
    await db_session.refresh(settings)
    assert settings.embedding_provider == "ollama"

    # Verify task was created
    mock_registry.create_task.assert_called_once_with(
        task_type="reindex_embedding",
        total_items=0,
    )

    # Note: We can't easily verify BackgroundTasks.add_task was called
    # because it's part of the FastAPI lifecycle, not a simple function call
    # In a real test, you'd check that reindex_all_articles was queued


@pytest.mark.asyncio
async def test_update_embedding_provider_not_available(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/embedding returns 400 if provider not available."""
    # Create initial settings
    settings = ProviderSettings(id=1, embedding_provider="openai", llm_provider="openai")
    db_session.add(settings)
    await db_session.commit()

    # Mock provider availability (ollama NOT available - though it should always be)
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai"],  # ollama missing
            "llm": ["openai"],
        }

        response = await async_client.patch(
            "/api/v1/settings/providers/embedding",
            json={"provider": "ollama", "confirm_reindex": False}
        )

    assert response.status_code == 400
    assert "not configured" in response.json()["detail"]

    # Verify database was NOT updated
    await db_session.refresh(settings)
    assert settings.embedding_provider == "openai"


@pytest.mark.asyncio
async def test_update_embedding_provider_invalid_value(async_client: AsyncClient, db_session: AsyncSession):
    """Test PATCH /api/v1/settings/providers/embedding returns 422 for invalid provider value."""
    response = await async_client.patch(
        "/api/v1/settings/providers/embedding",
        json={"provider": "invalid_provider", "confirm_reindex": False}
    )

    assert response.status_code == 422
    # Pydantic validation error (Literal type constraint)


@pytest.mark.asyncio
async def test_update_embedding_provider_reverse_dimension_change(async_client: AsyncClient, db_session: AsyncSession):
    """Test dimension change detection works in reverse direction (Ollama -> OpenAI).

    Verifies that dimension change detection works regardless of direction:
    - Ollama (1024 dims) -> OpenAI (1536 dims) should also require confirmation
    """
    # Create initial settings (ollama)
    settings = ProviderSettings(id=1, embedding_provider="ollama", llm_provider="openai")
    db_session.add(settings)
    await db_session.commit()

    # Mock provider availability
    with patch("article_mind_service.routers.settings.get_available_providers") as mock_available:
        mock_available.return_value = {
            "embedding": ["openai", "ollama"],
            "llm": ["openai"],
        }

        # Switch to openai (dimension change) without confirmation
        response = await async_client.patch(
            "/api/v1/settings/providers/embedding",
            json={"provider": "openai", "confirm_reindex": False}
        )

    assert response.status_code == 400
    detail = response.json()["detail"]

    # Verify error message shows reverse dimension change
    assert "dimension change" in detail.lower()
    assert "1024" in detail  # Old dimension (ollama)
    assert "1536" in detail  # New dimension (openai)

    # Verify database was NOT updated
    await db_session.refresh(settings)
    assert settings.embedding_provider == "ollama"  # Unchanged


@pytest.mark.asyncio
async def test_openapi_spec_includes_settings_endpoints(client):
    """Test that settings endpoints are properly documented in OpenAPI spec."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi_spec = response.json()

    # Verify settings endpoints exist in OpenAPI spec
    assert "/api/v1/settings/providers" in openapi_spec["paths"]
    assert "/api/v1/settings/providers/embedding" in openapi_spec["paths"]
    assert "/api/v1/settings/providers/llm" in openapi_spec["paths"]

    # Verify endpoint methods
    assert "get" in openapi_spec["paths"]["/api/v1/settings/providers"]
    assert "patch" in openapi_spec["paths"]["/api/v1/settings/providers/embedding"]
    assert "patch" in openapi_spec["paths"]["/api/v1/settings/providers/llm"]

    # Verify schemas exist
    assert "ProviderConfigResponse" in openapi_spec["components"]["schemas"]
    assert "UpdateEmbeddingProviderRequest" in openapi_spec["components"]["schemas"]
    assert "UpdateEmbeddingProviderResponse" in openapi_spec["components"]["schemas"]
    assert "UpdateLlmProviderRequest" in openapi_spec["components"]["schemas"]
    assert "UpdateLlmProviderResponse" in openapi_spec["components"]["schemas"]
