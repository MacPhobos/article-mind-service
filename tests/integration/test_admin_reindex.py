"""Integration tests for admin reindex endpoints.

Test Coverage:
- POST /api/v1/admin/reindex - Start reindex task
- GET /api/v1/admin/reindex/{task_id} - Get task status
- POST /api/v1/admin/reindex/{task_id}/cancel - Cancel task
- GET /api/v1/admin/reindex/{task_id}/progress - SSE stream (basic test)

Design Decisions:

1. Test Database:
   - Uses same database as other integration tests (fixtures in conftest.py)
   - Creates real Article records for testing
   - Rationale: Tests real database queries and business logic
   - Trade-off: Slower tests vs. realistic scenarios

2. Mock vs Real Embedding Pipeline:
   - Mocks EmbeddingPipeline.process_article to avoid slow operations
   - Rationale: Tests focus on admin API, not embedding logic
   - Trade-off: Doesn't catch embedding pipeline issues vs. fast tests

3. SSE Testing:
   - Basic test for SSE endpoint (checks response type)
   - Full SSE stream testing is complex and better suited for E2E tests
   - Rationale: Integration tests focus on API contract
   - Trade-off: Limited SSE coverage vs. simpler tests

4. Background Task Testing:
   - Uses small sleep to allow background task to process
   - Polls task status to verify completion
   - Rationale: Background tasks are async, need time to complete
   - Trade-off: Slightly slower tests vs. reliable verification
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from article_mind_service.models.article import Article
from article_mind_service.tasks.registry import task_registry


@pytest.mark.asyncio
async def test_start_reindex_all_sessions(async_client: AsyncClient, sample_articles):
    """Test starting reindex for all sessions."""
    # Setup: Ensure articles have pending embeddings
    # (sample_articles fixture creates articles with pending status)

    # Act: Start reindex
    response = await async_client.post(
        "/api/v1/admin/reindex",
        json={"session_ids": None, "force": False},
    )

    # Assert: Returns 200 with task details
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["total_sessions"] >= 1
    assert data["total_articles"] >= 1
    assert "progress_url" in data
    assert f"/api/v1/admin/reindex/{data['task_id']}/progress" in data["progress_url"]


@pytest.mark.asyncio
async def test_start_reindex_specific_sessions(async_client: AsyncClient, sample_articles):
    """Test starting reindex for specific sessions."""
    # Setup: Get session ID from sample articles
    session_id = sample_articles[0].session_id

    # Act: Start reindex with specific session
    response = await async_client.post(
        "/api/v1/admin/reindex",
        json={"session_ids": [session_id], "force": False},
    )

    # Assert: Returns 200 with task details
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["total_sessions"] == 1


@pytest.mark.asyncio
async def test_start_reindex_force_mode(async_client: AsyncClient, sample_articles, db):
    """Test reindex with force=True includes completed articles."""
    # Setup: Update article to completed status
    article = sample_articles[0]
    article.embedding_status = "completed"
    await db.commit()
    await db.refresh(article)

    # Act: Start reindex with force=True
    response = await async_client.post(
        "/api/v1/admin/reindex",
        json={"session_ids": [article.session_id], "force": True},
    )

    # Assert: Returns 200 (force includes completed articles)
    assert response.status_code == 200
    data = response.json()
    assert data["total_articles"] >= 1


@pytest.mark.asyncio
async def test_start_reindex_no_articles_found(async_client: AsyncClient, db):
    """Test reindex returns 404 when no articles match criteria."""
    # Act: Try to reindex non-existent session
    response = await async_client.post(
        "/api/v1/admin/reindex",
        json={"session_ids": [99999], "force": False},
    )

    # Assert: Returns 404
    assert response.status_code == 404
    assert "No articles found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_task_status(async_client: AsyncClient, sample_articles):
    """Test getting task status via polling endpoint."""
    # Setup: Start reindex task
    response = await async_client.post(
        "/api/v1/admin/reindex",
        json={"session_ids": None, "force": False},
    )
    task_id = response.json()["task_id"]

    # Act: Get task status
    response = await async_client.get(f"/api/v1/admin/reindex/{task_id}")

    # Assert: Returns task status
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == task_id
    assert data["task_type"] == "reindex"
    assert data["status"] in ["pending", "running", "completed"]
    assert "total_items" in data
    assert "processed_items" in data
    assert "failed_items" in data
    assert "progress_percent" in data
    assert 0 <= data["progress_percent"] <= 100


@pytest.mark.asyncio
async def test_get_task_status_not_found(async_client: AsyncClient):
    """Test getting status for non-existent task returns 404."""
    # Act: Get status for fake task
    response = await async_client.get("/api/v1/admin/reindex/fake-task-id")

    # Assert: Returns 404
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_cancel_task(async_client: AsyncClient, sample_articles):
    """Test canceling a reindex task."""
    # Setup: Start reindex task
    response = await async_client.post(
        "/api/v1/admin/reindex",
        json={"session_ids": None, "force": False},
    )
    task_id = response.json()["task_id"]

    # Act: Cancel task
    response = await async_client.post(f"/api/v1/admin/reindex/{task_id}/cancel")

    # Assert: Returns success
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "cancellation_requested"
    assert data["task_id"] == task_id

    # Verify cancellation flag is set
    assert task_registry.is_cancelled(task_id)


@pytest.mark.asyncio
async def test_cancel_task_not_found(async_client: AsyncClient):
    """Test canceling non-existent task returns 404."""
    # Act: Cancel fake task
    response = await async_client.post("/api/v1/admin/reindex/fake-task-id/cancel")

    # Assert: Returns 404
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_sse_progress_endpoint_basic(async_client: AsyncClient, sample_articles):
    """Test SSE progress endpoint returns event stream (basic test).

    Note: Full SSE stream testing is complex and better suited for E2E tests.
    This test just verifies the endpoint responds with correct content type.
    """
    # Setup: Start reindex task
    response = await async_client.post(
        "/api/v1/admin/reindex",
        json={"session_ids": None, "force": False},
    )
    task_id = response.json()["task_id"]

    # Act: Connect to SSE endpoint
    # Note: We can't easily test full SSE stream with httpx AsyncClient
    # So we just verify the endpoint exists and returns correct content type
    async with async_client.stream(
        "GET", f"/api/v1/admin/reindex/{task_id}/progress"
    ) as response:
        # Assert: Returns 200 with SSE content type
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


@pytest.mark.asyncio
async def test_reindex_background_task_completion(
    async_client: AsyncClient, sample_articles, db
):
    """Test that reindex background task actually completes.

    This test mocks the embedding pipeline to make it fast, then verifies
    the task completes successfully.
    """
    # Setup: Mock embedding pipeline to avoid slow operations
    with patch(
        "article_mind_service.tasks.reindex.get_embedding_pipeline"
    ) as mock_get_pipeline:
        # Create mock pipeline that succeeds instantly
        mock_pipeline = AsyncMock()
        mock_pipeline.process_article = AsyncMock(return_value=5)  # Return chunk count
        mock_get_pipeline.return_value = mock_pipeline

        # Start reindex
        response = await async_client.post(
            "/api/v1/admin/reindex",
            json={"session_ids": None, "force": False},
        )
        task_id = response.json()["task_id"]

        # Wait for background task to complete (max 5 seconds)
        for _ in range(50):
            await asyncio.sleep(0.1)
            status_response = await async_client.get(f"/api/v1/admin/reindex/{task_id}")
            status = status_response.json()["status"]
            if status in ["completed", "failed"]:
                break

        # Assert: Task completed successfully
        final_status = await async_client.get(f"/api/v1/admin/reindex/{task_id}")
        data = final_status.json()
        assert data["status"] == "completed"
        assert data["progress_percent"] == 100
        assert data["processed_items"] == data["total_items"]


@pytest.mark.asyncio
async def test_reindex_error_handling(async_client: AsyncClient, sample_articles, db):
    """Test that reindex handles article processing errors gracefully."""
    # Setup: Mock embedding pipeline to fail on some articles
    with patch(
        "article_mind_service.tasks.reindex.get_embedding_pipeline"
    ) as mock_get_pipeline:
        # Create mock pipeline that fails intermittently
        mock_pipeline = AsyncMock()
        call_count = 0

        async def mock_process_article(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Simulated embedding failure")
            return 5

        mock_pipeline.process_article = mock_process_article
        mock_get_pipeline.return_value = mock_pipeline

        # Start reindex
        response = await async_client.post(
            "/api/v1/admin/reindex",
            json={"session_ids": None, "force": False},
        )
        task_id = response.json()["task_id"]

        # Wait for completion
        for _ in range(50):
            await asyncio.sleep(0.1)
            status_response = await async_client.get(f"/api/v1/admin/reindex/{task_id}")
            status = status_response.json()["status"]
            if status in ["completed", "failed"]:
                break

        # Assert: Task completed (not failed) but has failed items
        final_status = await async_client.get(f"/api/v1/admin/reindex/{task_id}")
        data = final_status.json()
        assert data["status"] == "completed"  # Partial success
        assert data["failed_items"] > 0  # Some failures occurred
        assert len(data["errors"]) > 0  # Errors were recorded
