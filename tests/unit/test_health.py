"""Detailed tests for health check endpoint."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_ok_status(async_client: AsyncClient) -> None:
    """Test health check returns 'ok' when database is healthy."""
    response = await async_client.get("/health")
    data = response.json()

    assert response.status_code == 200
    assert data["version"] == "0.1.0"

    # If DB is available, should be ok
    if data["database"] == "connected":
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_health_check_degraded_on_db_failure(async_client: AsyncClient) -> None:
    """Test health check returns 'degraded' when database is down.

    Design Decision: Mock at session level, not engine level

    Rationale: Mocking get_db dependency is more reliable than mocking
    the database engine, as it operates at the FastAPI dependency injection
    layer rather than SQLAlchemy internals.

    Trade-offs:
    - Testability: Easier to mock AsyncSession.execute than async engine
    - Isolation: Doesn't require deep knowledge of SQLAlchemy internals
    - Maintenance: Less brittle when SQLAlchemy versions change
    """
    # Create mock session that raises on execute
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Database connection failed")

    # Mock the get_db dependency to return our failing session
    async def mock_get_db():
        yield mock_session

    # Patch the dependency in the router module
    with patch("article_mind_service.routers.health.get_db", mock_get_db):
        response = await async_client.get("/health")
        data = response.json()

        assert response.status_code == 200  # Still returns 200
        assert data["status"] == "degraded"
        assert data["database"] == "disconnected"
        assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health_check_response_schema_validation(async_client: AsyncClient) -> None:
    """Test health check response validates against Pydantic schema."""
    response = await async_client.get("/health")
    data = response.json()

    # Required fields must be present
    assert "status" in data
    assert "version" in data
    assert "database" in data

    # Type validation
    assert isinstance(data["status"], str)
    assert isinstance(data["version"], str)
    assert isinstance(data["database"], str)

    # Enum validation
    assert data["status"] in ["ok", "degraded", "error"]
    assert data["database"] in ["connected", "disconnected"]


@pytest.mark.asyncio
async def test_health_check_performance(async_client: AsyncClient) -> None:
    """Test health check responds quickly.

    Performance Requirements:
    - Time Complexity: O(1) - single SELECT 1 query
    - Expected Response Time: <1 second (includes network overhead in tests)
    - Database Query Time: <50ms in production

    This test verifies the health check doesn't have performance regressions
    like N+1 queries or blocking operations.
    """
    import time

    start = time.time()
    response = await async_client.get("/health")
    duration = time.time() - start

    assert response.status_code == 200
    assert duration < 1.0  # Should respond in less than 1 second


@pytest.mark.asyncio
async def test_health_check_version_matches_config(async_client: AsyncClient) -> None:
    """Test health check returns version from settings."""
    from article_mind_service.config import settings

    response = await async_client.get("/health")
    data = response.json()

    assert data["version"] == settings.app_version


@pytest.mark.asyncio
async def test_health_check_with_various_db_errors(async_client: AsyncClient) -> None:
    """Test health check handles different database error types gracefully.

    Error Handling Coverage:
    - Connection timeout
    - Authentication failure
    - Network errors
    - Database not found

    All should result in "degraded" status, not crashes.
    """
    error_scenarios = [
        ConnectionError("Connection timeout"),
        RuntimeError("Authentication failed"),
        Exception("Network unreachable"),
        ValueError("Database not found"),
    ]

    for error in error_scenarios:
        # Create mock session with error bound to avoid loop variable capture
        mock_session = AsyncMock()
        mock_session.execute.side_effect = error

        # Use default argument to bind loop variable (Python closure best practice)
        async def mock_get_db(session=mock_session):
            yield session

        with patch("article_mind_service.routers.health.get_db", mock_get_db):
            response = await async_client.get("/health")
            data = response.json()

            # All error types should result in degraded status
            assert response.status_code == 200
            assert data["status"] == "degraded"
            assert data["database"] == "disconnected"


@pytest.mark.asyncio
async def test_health_check_openapi_schema_examples(async_client: AsyncClient) -> None:
    """Test OpenAPI spec includes example responses."""
    response = await async_client.get("/openapi.json")
    openapi = response.json()

    health_endpoint = openapi["paths"]["/health"]["get"]

    # Verify examples exist in OpenAPI spec
    assert "responses" in health_endpoint
    assert "200" in health_endpoint["responses"]

    response_spec = health_endpoint["responses"]["200"]["content"]["application/json"]

    # Check for examples
    assert "examples" in response_spec
    examples = response_spec["examples"]

    # Should have both healthy and degraded examples
    assert "healthy" in examples or "ok" in str(examples).lower()
    assert "degraded" in examples or "degraded" in str(examples).lower()
