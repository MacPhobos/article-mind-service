"""Integration tests for articles API endpoints."""

from pathlib import Path

import pytest
from httpx import AsyncClient


@pytest.fixture(scope="function")
async def session_id(async_client: AsyncClient) -> int:
    """Create a test session and return its ID."""
    response = await async_client.post(
        "/api/v1/sessions",
        json={"name": "Test Session for Articles", "description": "Integration test session"},
    )
    assert response.status_code == 201
    return response.json()["id"]


class TestAddUrlArticle:
    """Tests for POST /api/v1/sessions/{session_id}/articles/url"""

    @pytest.mark.asyncio
    async def test_add_url_article(self, async_client: AsyncClient, session_id: int) -> None:
        """Test adding an article from URL."""
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={
                "url": "https://example.com/article",
                "title": "Example Article",
            },
        )
        assert response.status_code == 201

        data = response.json()
        assert data["type"] == "url"
        assert data["original_url"] == "https://example.com/article"
        assert data["title"] == "Example Article"
        assert data["extraction_status"] == "pending"
        assert data["session_id"] == session_id
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert data["has_content"] is False

    @pytest.mark.asyncio
    async def test_add_url_without_title(self, async_client: AsyncClient, session_id: int) -> None:
        """Test adding URL article without title."""
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://arxiv.org/abs/2301.00001"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["type"] == "url"
        assert data["original_url"] == "https://arxiv.org/abs/2301.00001"
        assert data["title"] is None  # No title provided

    @pytest.mark.asyncio
    async def test_add_url_to_nonexistent_session(self, async_client: AsyncClient) -> None:
        """Test adding article to nonexistent session returns 404."""
        response = await async_client.post(
            "/api/v1/sessions/99999/articles/url",
            json={"url": "https://example.com/article"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_add_url_invalid_scheme(self, async_client: AsyncClient, session_id: int) -> None:
        """Test adding URL with invalid scheme fails."""
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "ftp://example.com/file"},
        )
        assert response.status_code == 422  # Validation error


class TestUploadArticleFile:
    """Tests for POST /api/v1/sessions/{session_id}/articles/upload"""

    @pytest.mark.asyncio
    async def test_upload_file(self, async_client: AsyncClient, session_id: int) -> None:
        """Test uploading a file."""
        # Create a small test file
        file_content = b"This is a test PDF content"
        files = {"file": ("test_article.pdf", file_content, "application/pdf")}

        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/upload",
            files=files,
        )
        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert data["filename"] == "test_article.pdf"
        assert data["size_bytes"] == len(file_content)
        assert data["extraction_status"] == "pending"
        assert "created_at" in data

        # Verify file was saved to filesystem
        article_id = data["id"]
        upload_path = Path("data/uploads") / str(session_id) / str(article_id) / "original.pdf"
        assert upload_path.exists()

        # Clean up
        upload_path.unlink(missing_ok=True)
        upload_path.parent.rmdir()

    @pytest.mark.asyncio
    async def test_upload_multiple_file_types(
        self, async_client: AsyncClient, session_id: int
    ) -> None:
        """Test uploading different file types."""
        file_types = [
            ("test.pdf", "application/pdf"),
            (
                "test.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            ("test.txt", "text/plain"),
            ("test.md", "text/markdown"),
            ("test.html", "text/html"),
        ]

        for filename, content_type in file_types:
            files = {"file": (filename, b"Test content", content_type)}
            response = await async_client.post(
                f"/api/v1/sessions/{session_id}/articles/upload",
                files=files,
            )
            assert response.status_code == 201, f"Failed to upload {filename}"

            data = response.json()
            assert data["filename"] == filename

            # Clean up
            article_id = data["id"]
            ext = Path(filename).suffix
            upload_path = (
                Path("data/uploads") / str(session_id) / str(article_id) / f"original{ext}"
            )
            upload_path.unlink(missing_ok=True)
            upload_path.parent.rmdir()

    @pytest.mark.asyncio
    async def test_upload_unsupported_file_type(
        self, async_client: AsyncClient, session_id: int
    ) -> None:
        """Test uploading unsupported file type fails."""
        files = {"file": ("test.exe", b"Test content", "application/x-msdownload")}
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/upload",
            files=files,
        )
        assert response.status_code == 415  # Unsupported Media Type

    @pytest.mark.asyncio
    async def test_upload_no_filename(self, async_client: AsyncClient, session_id: int) -> None:
        """Test uploading file without filename fails."""
        files = {"file": ("", b"Test content", "application/pdf")}
        response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/upload",
            files=files,
        )
        assert response.status_code == 400


class TestListArticles:
    """Tests for GET /api/v1/sessions/{session_id}/articles"""

    @pytest.mark.asyncio
    async def test_list_empty_articles(self, async_client: AsyncClient, session_id: int) -> None:
        """Test listing articles in empty session."""
        response = await async_client.get(f"/api/v1/sessions/{session_id}/articles")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_list_articles(self, async_client: AsyncClient, session_id: int) -> None:
        """Test listing articles in session."""
        # Add some articles
        await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article1"},
        )
        await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article2"},
        )

        response = await async_client.get(f"/api/v1/sessions/{session_id}/articles")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert data["total"] >= 2
        assert len(data["items"]) >= 2

    @pytest.mark.asyncio
    async def test_list_articles_nonexistent_session(self, async_client: AsyncClient) -> None:
        """Test listing articles for nonexistent session returns 404."""
        response = await async_client.get("/api/v1/sessions/99999/articles")
        assert response.status_code == 404


class TestGetArticle:
    """Tests for GET /api/v1/sessions/{session_id}/articles/{article_id}"""

    @pytest.mark.asyncio
    async def test_get_article(self, async_client: AsyncClient, session_id: int) -> None:
        """Test getting a specific article."""
        # Create article
        create_response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article", "title": "Test Article"},
        )
        article_id = create_response.json()["id"]

        # Get article
        response = await async_client.get(f"/api/v1/sessions/{session_id}/articles/{article_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == article_id
        assert data["session_id"] == session_id
        assert data["title"] == "Test Article"

    @pytest.mark.asyncio
    async def test_get_nonexistent_article(
        self, async_client: AsyncClient, session_id: int
    ) -> None:
        """Test getting nonexistent article returns 404."""
        response = await async_client.get(f"/api/v1/sessions/{session_id}/articles/99999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_article_from_wrong_session(self, async_client: AsyncClient) -> None:
        """Test getting article from different session returns 404."""
        # Create two sessions
        session1 = await async_client.post("/api/v1/sessions", json={"name": "Session 1"})
        session1_id = session1.json()["id"]

        session2 = await async_client.post("/api/v1/sessions", json={"name": "Session 2"})
        session2_id = session2.json()["id"]

        # Create article in session1
        article_response = await async_client.post(
            f"/api/v1/sessions/{session1_id}/articles/url",
            json={"url": "https://example.com/article"},
        )
        article_id = article_response.json()["id"]

        # Try to get from session2
        response = await async_client.get(f"/api/v1/sessions/{session2_id}/articles/{article_id}")
        assert response.status_code == 404


class TestDeleteArticle:
    """Tests for DELETE /api/v1/sessions/{session_id}/articles/{article_id}"""

    @pytest.mark.asyncio
    async def test_delete_article(self, async_client: AsyncClient, session_id: int) -> None:
        """Test soft deleting an article."""
        # Create article
        create_response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article"},
        )
        article_id = create_response.json()["id"]

        # Delete article
        response = await async_client.delete(f"/api/v1/sessions/{session_id}/articles/{article_id}")
        assert response.status_code == 204

        # Verify article is not found
        get_response = await async_client.get(
            f"/api/v1/sessions/{session_id}/articles/{article_id}"
        )
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_article(
        self, async_client: AsyncClient, session_id: int
    ) -> None:
        """Test deleting nonexistent article returns 404."""
        response = await async_client.delete(f"/api/v1/sessions/{session_id}/articles/99999")
        assert response.status_code == 404


class TestGetArticleContent:
    """Tests for GET /api/v1/sessions/{session_id}/articles/{article_id}/content"""

    @pytest.mark.asyncio
    async def test_get_content_not_ready(self, async_client: AsyncClient, session_id: int) -> None:
        """Test getting content when extraction not completed."""
        # Create article (status will be pending)
        create_response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article"},
        )
        article_id = create_response.json()["id"]

        # Try to get content
        response = await async_client.get(
            f"/api/v1/sessions/{session_id}/articles/{article_id}/content"
        )
        assert response.status_code == 400
        assert "not available" in response.json()["detail"].lower()


class TestSessionArticleCount:
    """Tests for verifying session article_count is computed correctly."""

    @pytest.mark.asyncio
    async def test_article_count_updates(self, async_client: AsyncClient, session_id: int) -> None:
        """Test that session article_count reflects actual article count."""
        # Check initial count
        session_response = await async_client.get(f"/api/v1/sessions/{session_id}")
        initial_count = session_response.json()["article_count"]

        # Add article
        await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article1"},
        )

        # Check count increased
        session_response = await async_client.get(f"/api/v1/sessions/{session_id}")
        assert session_response.json()["article_count"] == initial_count + 1

        # Add another article
        await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article2"},
        )

        # Check count increased again
        session_response = await async_client.get(f"/api/v1/sessions/{session_id}")
        assert session_response.json()["article_count"] == initial_count + 2

    @pytest.mark.asyncio
    async def test_article_count_after_delete(
        self, async_client: AsyncClient, session_id: int
    ) -> None:
        """Test that article_count decreases after deleting article."""
        # Add article
        create_response = await async_client.post(
            f"/api/v1/sessions/{session_id}/articles/url",
            json={"url": "https://example.com/article"},
        )
        article_id = create_response.json()["id"]

        # Get initial count
        session_response = await async_client.get(f"/api/v1/sessions/{session_id}")
        count_before = session_response.json()["article_count"]

        # Delete article
        await async_client.delete(f"/api/v1/sessions/{session_id}/articles/{article_id}")

        # Check count decreased
        session_response = await async_client.get(f"/api/v1/sessions/{session_id}")
        assert session_response.json()["article_count"] == count_before - 1
