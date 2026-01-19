"""Tests for main application."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


def test_root(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Article Mind Service API"}


@pytest.mark.asyncio
async def test_root_async(async_client: AsyncClient) -> None:
    """Test root endpoint with async client."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Article Mind Service API"}


def test_openapi_docs(client: TestClient) -> None:
    """Test OpenAPI documentation is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()
    assert openapi["info"]["title"] == "Article Mind Service"


def test_health_check_endpoint_exists(client: TestClient) -> None:
    """Test health check endpoint is accessible."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_check_response_structure(client: TestClient) -> None:
    """Test health check response follows schema."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "database" in data
    assert data["status"] in ["ok", "degraded", "error"]
    assert isinstance(data["version"], str)
    assert data["database"] in ["connected", "disconnected"]


@pytest.mark.asyncio
async def test_health_check_with_database(async_client: AsyncClient) -> None:
    """Test health check includes database status."""
    response = await async_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "database" in data
    # Database status depends on if DB is running
    # Don't assert specific value, just that it exists and is valid
    assert data["database"] in ["connected", "disconnected"]


@pytest.mark.asyncio
async def test_health_check_database_connected(async_client: AsyncClient) -> None:
    """Test health check reports database as connected when DB is healthy."""
    response = await async_client.get("/health")
    data = response.json()

    # If database is connected, overall status should be ok
    if data["database"] == "connected":
        assert data["status"] == "ok"
    # If database is disconnected, status should be degraded
    elif data["database"] == "disconnected":
        assert data["status"] == "degraded"


def test_health_check_in_openapi(client: TestClient) -> None:
    """Test health check endpoint is documented in OpenAPI spec."""
    response = client.get("/openapi.json")
    openapi = response.json()

    # Verify /health endpoint exists in OpenAPI spec
    assert "/health" in openapi["paths"]

    health_spec = openapi["paths"]["/health"]
    assert "get" in health_spec

    # Verify response schema
    get_spec = health_spec["get"]
    assert "200" in get_spec["responses"]
    assert "application/json" in get_spec["responses"]["200"]["content"]


@pytest.mark.asyncio
async def test_health_check_no_authentication_required(async_client: AsyncClient) -> None:
    """Test health check does not require authentication."""
    # No auth headers provided
    response = await async_client.get("/health")
    assert response.status_code == 200  # Should succeed without auth
