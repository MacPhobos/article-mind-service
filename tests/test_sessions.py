"""Integration tests for session CRUD API."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestCreateSession:
    """Tests for POST /api/v1/sessions."""

    async def test_create_session_success(self, async_client: AsyncClient) -> None:
        """Test successful session creation."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Test Session", "description": "A test description"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Session"
        assert data["description"] == "A test description"
        assert data["status"] == "draft"
        assert data["article_count"] == 0
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    async def test_create_session_minimal(self, async_client: AsyncClient) -> None:
        """Test session creation with only required fields."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Minimal Session"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Minimal Session"
        assert data["description"] is None

    async def test_create_session_empty_name_fails(self, async_client: AsyncClient) -> None:
        """Test that empty name fails validation."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": ""},
        )

        assert response.status_code == 422  # Validation error

    async def test_create_session_whitespace_name_trimmed(self, async_client: AsyncClient) -> None:
        """Test that whitespace is trimmed from name."""
        response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "  Trimmed Name  "},
        )

        assert response.status_code == 201
        assert response.json()["name"] == "Trimmed Name"


@pytest.mark.asyncio
class TestListSessions:
    """Tests for GET /api/v1/sessions."""

    async def test_list_sessions_empty(self, async_client: AsyncClient) -> None:
        """Test listing when no sessions exist."""
        response = await async_client.get("/api/v1/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)

    async def test_list_sessions_with_data(self, async_client: AsyncClient) -> None:
        """Test listing with existing sessions."""
        # Create a session first
        await async_client.post(
            "/api/v1/sessions",
            json={"name": "Session 1"},
        )

        response = await async_client.get("/api/v1/sessions")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert any(s["name"] == "Session 1" for s in data["sessions"])

    async def test_list_sessions_filter_by_status(self, async_client: AsyncClient) -> None:
        """Test filtering by status."""
        # Create sessions with different statuses
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Draft Session"},
        )
        session_id = create_response.json()["id"]

        # Change status to active
        await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        # Filter by active status
        response = await async_client.get("/api/v1/sessions?status=active")

        assert response.status_code == 200
        data = response.json()
        assert all(s["status"] == "active" for s in data["sessions"])


@pytest.mark.asyncio
class TestGetSession:
    """Tests for GET /api/v1/sessions/{id}."""

    async def test_get_session_success(self, async_client: AsyncClient) -> None:
        """Test getting an existing session."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Get Test", "description": "Test description"},
        )
        session_id = create_response.json()["id"]

        # Get the session
        response = await async_client.get(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["name"] == "Get Test"
        assert data["description"] == "Test description"

    async def test_get_session_not_found(self, async_client: AsyncClient) -> None:
        """Test getting a non-existent session."""
        response = await async_client.get("/api/v1/sessions/99999")

        assert response.status_code == 404


@pytest.mark.asyncio
class TestUpdateSession:
    """Tests for PATCH /api/v1/sessions/{id}."""

    async def test_update_session_name(self, async_client: AsyncClient) -> None:
        """Test updating session name."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Original Name"},
        )
        session_id = create_response.json()["id"]

        # Update the name
        response = await async_client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"name": "Updated Name"},
        )

        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"

    async def test_update_session_description(self, async_client: AsyncClient) -> None:
        """Test updating session description."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Test", "description": "Original"},
        )
        session_id = create_response.json()["id"]

        # Update the description
        response = await async_client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"description": "Updated description"},
        )

        assert response.status_code == 200
        assert response.json()["description"] == "Updated description"

    async def test_update_session_clear_description(self, async_client: AsyncClient) -> None:
        """Test clearing session description with empty string."""
        # Create a session with description
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Test", "description": "Has description"},
        )
        session_id = create_response.json()["id"]

        # Clear the description
        response = await async_client.patch(
            f"/api/v1/sessions/{session_id}",
            json={"description": ""},
        )

        assert response.status_code == 200
        assert response.json()["description"] is None

    async def test_update_session_not_found(self, async_client: AsyncClient) -> None:
        """Test updating non-existent session."""
        response = await async_client.patch(
            "/api/v1/sessions/99999",
            json={"name": "Updated"},
        )

        assert response.status_code == 404


@pytest.mark.asyncio
class TestDeleteSession:
    """Tests for DELETE /api/v1/sessions/{id}."""

    async def test_delete_session_success(self, async_client: AsyncClient) -> None:
        """Test soft deleting a session."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "To Delete"},
        )
        session_id = create_response.json()["id"]

        # Delete the session
        response = await async_client.delete(f"/api/v1/sessions/{session_id}")

        assert response.status_code == 204

        # Verify it's not accessible
        get_response = await async_client.get(f"/api/v1/sessions/{session_id}")
        assert get_response.status_code == 404

    async def test_delete_session_not_found(self, async_client: AsyncClient) -> None:
        """Test deleting non-existent session."""
        response = await async_client.delete("/api/v1/sessions/99999")

        assert response.status_code == 404


@pytest.mark.asyncio
class TestChangeSessionStatus:
    """Tests for POST /api/v1/sessions/{id}/status."""

    async def test_change_status_draft_to_active(self, async_client: AsyncClient) -> None:
        """Test transitioning from draft to active."""
        # Create a session (starts as draft)
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Status Test"},
        )
        session_id = create_response.json()["id"]
        assert create_response.json()["status"] == "draft"

        # Change to active
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "active"

    async def test_change_status_active_to_completed(self, async_client: AsyncClient) -> None:
        """Test transitioning from active to completed."""
        # Create and activate session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Status Test"},
        )
        session_id = create_response.json()["id"]

        await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        # Change to completed
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "completed"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"

    async def test_change_status_to_archived(self, async_client: AsyncClient) -> None:
        """Test transitioning to archived from any non-archived status."""
        # Create a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Archive Test"},
        )
        session_id = create_response.json()["id"]

        # Archive directly from draft
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "archived"},
        )

        assert response.status_code == 200
        assert response.json()["status"] == "archived"

    async def test_change_status_invalid_transition(self, async_client: AsyncClient) -> None:
        """Test invalid status transition is rejected."""
        # Create a session (draft)
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Invalid Test"},
        )
        session_id = create_response.json()["id"]

        # Try to go directly to completed (invalid: draft -> completed)
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "completed"},
        )

        assert response.status_code == 400
        assert "Cannot transition" in response.json()["detail"]

    async def test_change_status_from_archived_fails(self, async_client: AsyncClient) -> None:
        """Test that archived sessions cannot change status."""
        # Create and archive a session
        create_response = await async_client.post(
            "/api/v1/sessions",
            json={"name": "Archived Test"},
        )
        session_id = create_response.json()["id"]

        await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "archived"},
        )

        # Try to change from archived
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/status",
            json={"status": "active"},
        )

        assert response.status_code == 400

    async def test_change_status_not_found(self, async_client: AsyncClient) -> None:
        """Test changing status of non-existent session."""
        response = await async_client.post(
            "/api/v1/sessions/99999/status",
            json={"status": "active"},
        )

        assert response.status_code == 404


@pytest.mark.asyncio
class TestSessionOpenAPI:
    """Tests for OpenAPI documentation."""

    async def test_sessions_in_openapi(self, async_client: AsyncClient) -> None:
        """Test that session endpoints are in OpenAPI spec."""
        response = await async_client.get("/openapi.json")

        assert response.status_code == 200
        openapi = response.json()

        # Verify endpoints exist
        assert "/api/v1/sessions" in openapi["paths"]
        assert "/api/v1/sessions/{session_id}" in openapi["paths"]
        assert "/api/v1/sessions/{session_id}/status" in openapi["paths"]

    async def test_session_schemas_in_openapi(self, async_client: AsyncClient) -> None:
        """Test that session schemas are in OpenAPI spec."""
        response = await async_client.get("/openapi.json")

        assert response.status_code == 200
        openapi = response.json()

        # Verify schemas exist
        schemas = openapi["components"]["schemas"]
        assert "SessionResponse" in schemas
        assert "SessionListResponse" in schemas
        assert "CreateSessionRequest" in schemas
        assert "UpdateSessionRequest" in schemas
        assert "ChangeStatusRequest" in schemas
