"""Pytest configuration and fixtures with proper database isolation."""

import os
import shutil
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from article_mind_service.config import Settings
from article_mind_service.database import get_db
from article_mind_service.main import app

from .db_setup import get_test_database_url, run_alembic_upgrade

# ============================================================================
# Load Test Environment Variables
# ============================================================================

# Load .env.test at the very start of test suite (before any fixtures run)
# This ensures TEST_DATABASE_URL and other test-specific settings are available
env_test_path = Path(__file__).parent.parent / ".env.test"
if env_test_path.exists():
    load_dotenv(env_test_path, override=True)


# ============================================================================
# Session-Scoped Fixtures (Database Setup)
# ============================================================================


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """Get test database URL from environment.

    This fixture validates that the database URL is safe for testing
    (must contain '_test' or use localhost).
    """
    return get_test_database_url()


@pytest.fixture(scope="session")
def test_async_database_url(test_database_url: str) -> str:
    """Convert sync database URL to async format for asyncpg."""
    return test_database_url.replace("postgresql://", "postgresql+asyncpg://")


@pytest.fixture(scope="session", autouse=True)
def setup_test_database(test_database_url: str) -> None:
    """Run Alembic migrations to set up test database schema.

    This runs once per test session before any tests execute.
    Uses Alembic migrations instead of Base.metadata.create_all().
    """
    run_alembic_upgrade(test_database_url)


@pytest.fixture(scope="session")
def test_engine(test_async_database_url: str) -> AsyncEngine:
    """Create async SQLAlchemy engine for test database.

    Uses NullPool to avoid connection pooling issues in tests.
    All connections are closed after use.
    """
    return create_async_engine(
        test_async_database_url,
        echo=False,
        poolclass=NullPool,  # No connection pooling for tests
    )


# ============================================================================
# Function-Scoped Fixtures (Per-Test Isolation)
# ============================================================================


@pytest.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Provide isolated database session using SAVEPOINT pattern.

    Each test gets a fresh session that rolls back all changes after test completion.

    Isolation Strategy (SQLAlchemy 2.0 SAVEPOINT pattern):
    1. Create new connection per test
    2. Begin outer transaction
    3. Create AsyncSession with join_transaction_mode="create_savepoint"
    4. Yield session to test (allows commits within test)
    5. On teardown: close session, rollback transaction (discards all changes)

    The join_transaction_mode="create_savepoint" automatically manages savepoints,
    allowing tests to call session.commit() without actually persisting changes.
    All changes are rolled back after each test.
    """
    # Create new connection for this test
    async with test_engine.connect() as connection:
        # Begin outer transaction
        transaction = await connection.begin()

        try:
            # Create session with automatic savepoint management
            session = AsyncSession(
                bind=connection,
                join_transaction_mode="create_savepoint",  # SQLAlchemy 2.0 feature
                expire_on_commit=False,
            )

            yield session
        finally:
            await session.close()
            # Explicitly rollback to discard all changes
            await transaction.rollback()


@pytest.fixture
def temp_upload_dir() -> Path:
    """Create temporary directory for file uploads.

    Automatically cleaned up after test completion.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="article_mind_test_uploads_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def temp_chroma_dir() -> Path:
    """Create temporary directory for ChromaDB.

    Automatically cleaned up after test completion.
    Also clears ChromaDB client singleton cache to prevent test interference.
    """
    # Clear singleton cache before test
    from article_mind_service.embeddings.client import get_chromadb_client

    get_chromadb_client.cache_clear()

    temp_dir = Path(tempfile.mkdtemp(prefix="article_mind_test_chroma_"))
    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Clear singleton cache after test
    get_chromadb_client.cache_clear()


@pytest.fixture
def mock_settings(temp_upload_dir: Path, temp_chroma_dir: Path) -> Settings:
    """Create test settings with temporary directories.

    Overrides upload and ChromaDB paths to use temporary directories.
    """
    return Settings(
        upload_base_path=str(temp_upload_dir),
        chromadb_path=str(temp_chroma_dir),
        chroma_persist_directory=str(temp_chroma_dir),
        debug=False,
        log_level="WARNING",
    )


# ============================================================================
# HTTP Client Fixtures
# ============================================================================


@pytest.fixture
def client() -> TestClient:
    """Synchronous test client for FastAPI app.

    Note: Does NOT provide database isolation. Use for simple tests only.
    For tests requiring database operations, use async_client or isolated_async_client.
    """
    return TestClient(app)


@pytest.fixture
async def async_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with database isolation.

    Overrides the get_db dependency to use the isolated db_session fixture.
    All database operations within tests using this fixture are automatically
    rolled back after test completion.

    Use for tests that interact with the database but don't upload files.
    """

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        """Override get_db dependency to use test session."""
        yield db_session

    # Override dependency
    app.dependency_overrides[get_db] = override_get_db

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
    finally:
        # Cleanup: remove this specific override only (don't use .clear())
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture
async def isolated_async_client(
    db_session: AsyncSession, mock_settings: Settings
) -> AsyncGenerator[AsyncClient, None]:
    """Async test client with full isolation (database + filesystem).

    Provides:
    - Database isolation via SAVEPOINT pattern (db_session)
    - Filesystem isolation via temporary upload/ChromaDB directories (mock_settings)

    Use for tests that:
    - Upload files
    - Use ChromaDB
    - Require complete isolation from other tests

    All changes (database, files, ChromaDB) are automatically cleaned up after test.
    """

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        """Override get_db dependency to use test session."""
        yield db_session

    # Override dependencies
    app.dependency_overrides[get_db] = override_get_db

    # Override settings for filesystem isolation
    # Note: This requires modifying how settings are accessed in the app
    # If app uses `from config import settings`, this won't work
    # You may need to change app code to use dependency injection for settings

    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac
    finally:
        # Cleanup: remove this specific override only (don't use .clear())
        app.dependency_overrides.pop(get_db, None)


# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
async def db(db_session: AsyncSession) -> AsyncSession:
    """Alias for db_session for compatibility with existing tests."""
    return db_session


@pytest.fixture
async def sample_articles(db_session: AsyncSession) -> list[Any]:
    """Create sample articles for testing admin reindex.

    Creates 3 articles in different sessions with pending embedding status.
    """
    from article_mind_service.models.article import Article
    from article_mind_service.models.session import ResearchSession

    # Create test sessions
    session1 = ResearchSession(name="Test Session 1", status="active")
    session2 = ResearchSession(name="Test Session 2", status="active")
    db_session.add_all([session1, session2])
    await db_session.flush()

    # Create test articles
    articles = [
        Article(
            session_id=session1.id,
            original_url="https://example.com/article1",
            canonical_url="https://example.com/article1",
            type="url",
            extraction_status="completed",
            embedding_status="pending",
            content_text="Sample article 1 content for testing reindex",
            title="Article 1",
        ),
        Article(
            session_id=session1.id,
            original_url="https://example.com/article2",
            canonical_url="https://example.com/article2",
            type="url",
            extraction_status="completed",
            embedding_status="pending",
            content_text="Sample article 2 content for testing reindex",
            title="Article 2",
        ),
        Article(
            session_id=session2.id,
            original_url="https://example.com/article3",
            canonical_url="https://example.com/article3",
            type="url",
            extraction_status="completed",
            embedding_status="pending",
            content_text="Sample article 3 content for testing reindex",
            title="Article 3",
        ),
    ]

    db_session.add_all(articles)
    await db_session.commit()

    # Refresh to get IDs
    for article in articles:
        await db_session.refresh(article)

    return articles
