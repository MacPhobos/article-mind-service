"""Integration tests for session reindex endpoint."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from article_mind_service.models.article import Article
from article_mind_service.models.session import ResearchSession


@pytest.mark.asyncio
class TestReindexSession:
    """Tests for POST /api/v1/sessions/{session_id}/reindex."""

    async def test_reindex_session_with_pending_articles(
        self, async_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Test reindexing session with articles needing embedding."""
        # Create session
        session = ResearchSession(name="Test Session", status="active")
        db_session.add(session)
        await db_session.flush()

        # Create articles with different embedding statuses
        article1 = Article(
            session_id=session.id,
            type="url",
            original_url="https://example.com/1",
            extraction_status="completed",
            embedding_status="pending",
            content_text="Test content 1",
        )
        article2 = Article(
            session_id=session.id,
            type="url",
            original_url="https://example.com/2",
            extraction_status="completed",
            embedding_status="failed",
            content_text="Test content 2",
        )
        article3 = Article(
            session_id=session.id,
            type="url",
            original_url="https://example.com/3",
            extraction_status="completed",
            embedding_status="completed",  # Already embedded
            content_text="Test content 3",
        )
        db_session.add_all([article1, article2, article3])
        await db_session.commit()

        # Call reindex endpoint
        response = await async_client.post(f"/api/v1/sessions/{session.id}/reindex")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session.id
        assert data["articles_queued"] == 2  # Only pending and failed
        assert set(data["article_ids"]) == {article1.id, article2.id}

    async def test_reindex_session_with_no_articles(
        self, async_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Test reindexing session with no articles needing reindex."""
        # Create session with no articles
        session = ResearchSession(name="Empty Session", status="active")
        db_session.add(session)
        await db_session.commit()

        response = await async_client.post(f"/api/v1/sessions/{session.id}/reindex")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session.id
        assert data["articles_queued"] == 0
        assert data["article_ids"] == []

    async def test_reindex_session_not_found(self, async_client: AsyncClient) -> None:
        """Test reindexing non-existent session returns 404."""
        response = await async_client.post("/api/v1/sessions/99999/reindex")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    async def test_reindex_session_excludes_deleted_articles(
        self, async_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Test that soft-deleted articles are not reindexed."""
        from datetime import UTC, datetime

        # Create session
        session = ResearchSession(name="Test Session", status="active")
        db_session.add(session)
        await db_session.flush()

        # Create article that is soft-deleted
        article = Article(
            session_id=session.id,
            type="url",
            original_url="https://example.com/deleted",
            extraction_status="completed",
            embedding_status="pending",
            content_text="Deleted content",
            deleted_at=datetime.now(UTC),  # Soft deleted
        )
        db_session.add(article)
        await db_session.commit()

        # Call reindex endpoint
        response = await async_client.post(f"/api/v1/sessions/{session.id}/reindex")

        assert response.status_code == 200
        data = response.json()
        assert data["articles_queued"] == 0  # Deleted article not included

    async def test_reindex_session_only_completed_extraction(
        self, async_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Test that only articles with completed extraction are reindexed."""
        # Create session
        session = ResearchSession(name="Test Session", status="active")
        db_session.add(session)
        await db_session.flush()

        # Create article with pending extraction
        article = Article(
            session_id=session.id,
            type="url",
            original_url="https://example.com/pending",
            extraction_status="pending",  # Not completed
            embedding_status="pending",
        )
        db_session.add(article)
        await db_session.commit()

        # Call reindex endpoint
        response = await async_client.post(f"/api/v1/sessions/{session.id}/reindex")

        assert response.status_code == 200
        data = response.json()
        assert data["articles_queued"] == 0  # Pending extraction not included

    async def test_reindex_session_idempotent(
        self, async_client: AsyncClient, db_session: AsyncSession
    ) -> None:
        """Test that calling reindex multiple times is safe.

        Note: After first reindex, articles may be in 'processing' or 'completed' state,
        so the second call may queue fewer articles. The test verifies both calls succeed.
        """
        # Create session
        session = ResearchSession(name="Test Session", status="active")
        db_session.add(session)
        await db_session.flush()

        # Create multiple articles to ensure at least one is still pending after first call
        article1 = Article(
            session_id=session.id,
            type="url",
            original_url="https://example.com/1",
            extraction_status="completed",
            embedding_status="pending",
            content_text="Test content 1",
        )
        article2 = Article(
            session_id=session.id,
            type="url",
            original_url="https://example.com/2",
            extraction_status="completed",
            embedding_status="pending",
            content_text="Test content 2",
        )
        db_session.add_all([article1, article2])
        await db_session.commit()

        # Call reindex twice - both should succeed without error
        response1 = await async_client.post(f"/api/v1/sessions/{session.id}/reindex")
        response2 = await async_client.post(f"/api/v1/sessions/{session.id}/reindex")

        # Both should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # First call should queue articles
        assert response1.json()["articles_queued"] >= 1

        # Second call should succeed (may queue 0 if already processed)
        # The key is that it doesn't error
        assert "articles_queued" in response2.json()
        assert "article_ids" in response2.json()
