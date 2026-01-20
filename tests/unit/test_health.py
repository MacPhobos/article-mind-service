"""Detailed tests for health check endpoint."""

from collections.abc import AsyncGenerator
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
async def test_health_check_degraded_on_db_failure() -> None:
    """Test health check returns 'degraded' when database is down.

    Design Decision: Override get_db dependency directly

    Rationale: app.dependency_overrides takes precedence over patches,
    so we must override the dependency at the app level for the test
    to properly inject the mock.

    Trade-offs:
    - Testability: Direct override is more explicit than patching
    - Isolation: Requires manual cleanup to avoid affecting other tests
    - Correctness: Guaranteed to work with FastAPI's dependency system
    """
    from httpx import ASGITransport, AsyncClient

    from article_mind_service.database import get_db
    from article_mind_service.main import app

    # Create mock session that raises on execute
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Database connection failed")

    # Mock the get_db dependency to return our failing session
    async def mock_get_db() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    # Override the dependency
    app.dependency_overrides[get_db] = mock_get_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
            data = response.json()

            assert response.status_code == 200  # Still returns 200
            assert data["status"] == "degraded"
            assert data["database"] == "disconnected"
            assert data["version"] == "0.1.0"
    finally:
        # Cleanup: remove the override
        app.dependency_overrides.pop(get_db, None)


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
async def test_health_check_with_various_db_errors() -> None:
    """Test health check handles different database error types gracefully.

    Error Handling Coverage:
    - Connection timeout
    - Authentication failure
    - Network errors
    - Database not found

    All should result in "degraded" status, not crashes.
    """
    from httpx import ASGITransport, AsyncClient

    from article_mind_service.database import get_db
    from article_mind_service.main import app

    error_scenarios = [
        ConnectionError("Connection timeout"),
        RuntimeError("Authentication failed"),
        Exception("Network unreachable"),
        ValueError("Database not found"),
    ]

    for error in error_scenarios:
        # Create mock session with error bound to avoid loop variable capture
        # We need to create a new function for each iteration to capture the error

        def make_mock_get_db(err: Exception):
            """Factory to create mock_get_db with bound error."""

            async def mock_get_db() -> AsyncGenerator[AsyncMock, None]:
                mock_session = AsyncMock()
                mock_session.execute.side_effect = err
                yield mock_session

            return mock_get_db

        # Override the dependency
        app.dependency_overrides[get_db] = make_mock_get_db(error)

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/health")
                data = response.json()

                # All error types should result in degraded status
                assert response.status_code == 200
                assert data["status"] == "degraded"
                assert data["database"] == "disconnected"
        finally:
            # Cleanup: remove the override
            app.dependency_overrides.pop(get_db, None)


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
