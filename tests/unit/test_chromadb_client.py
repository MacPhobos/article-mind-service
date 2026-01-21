"""Tests for ChromaDB client singleton pattern.

Verifies that:
1. get_chromadb_client() returns same instance across calls
2. Client uses consistent settings
3. Thread-safe behavior (if applicable)
4. Integration with ChromaDBStore and DenseSearch
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest

from article_mind_service.embeddings.client import get_chromadb_client
from article_mind_service.embeddings.chromadb_store import ChromaDBStore
from article_mind_service.search.dense_search import DenseSearch


class TestChromaDBClientSingleton:
    """Test singleton behavior of get_chromadb_client()."""

    def test_returns_same_instance(self, temp_chroma_dir: Path) -> None:
        """Test that get_chromadb_client returns same instance across calls."""
        # Clear cache to ensure clean state
        get_chromadb_client.cache_clear()

        # Mock settings to use temp directory
        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            # First call
            client1 = get_chromadb_client()

            # Second call
            client2 = get_chromadb_client()

            # Should be same instance
            assert client1 is client2, "get_chromadb_client should return same instance"

    def test_client_type(self, temp_chroma_dir: Path) -> None:
        """Test that client is ChromaDB PersistentClient."""
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            client = get_chromadb_client()

            # Verify client type by checking it has expected methods
            assert hasattr(client, "get_or_create_collection"), "Client should have get_or_create_collection"
            assert hasattr(client, "get_collection"), "Client should have get_collection"
            assert hasattr(client, "list_collections"), "Client should have list_collections"
            assert hasattr(client, "delete_collection"), "Client should have delete_collection"

    def test_client_settings_consistency(self, temp_chroma_dir: Path) -> None:
        """Test that client uses consistent settings."""
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            client = get_chromadb_client()

            # Verify client was created with correct settings
            # ChromaDB client doesn't expose settings directly, but we can verify
            # it works by creating collections
            collection = client.get_or_create_collection("test_collection")
            assert collection is not None
            assert collection.name == "test_collection"

    def test_cache_clear_allows_new_instance(self, temp_chroma_dir: Path) -> None:
        """Test that cache_clear allows creating new instance with different settings."""
        get_chromadb_client.cache_clear()

        # Create first instance with temp_dir1
        temp_dir1 = Path(tempfile.mkdtemp(prefix="chroma_test_1_"))
        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_dir1)
            client1 = get_chromadb_client()
            collection1 = client1.get_or_create_collection("test1")
            assert collection1.name == "test1"

        # Clear cache
        get_chromadb_client.cache_clear()

        # Create second instance with temp_dir2
        temp_dir2 = Path(tempfile.mkdtemp(prefix="chroma_test_2_"))
        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_dir2)
            client2 = get_chromadb_client()

            # Should be different instance (different path)
            # Note: ChromaDB clients aren't comparable by ID if paths differ
            # Verify by checking collections don't overlap
            collections = client2.list_collections()
            collection_names = [c.name for c in collections]
            assert "test1" not in collection_names, "New client should not see old collections"

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir1, ignore_errors=True)
        shutil.rmtree(temp_dir2, ignore_errors=True)


class TestChromaDBStoreIntegration:
    """Test that ChromaDBStore uses singleton client."""

    def test_chromadb_store_uses_singleton(self, temp_chroma_dir: Path) -> None:
        """Test that ChromaDBStore uses singleton client."""
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            # Get singleton client directly
            singleton_client = get_chromadb_client()

            # Create ChromaDBStore (should use same client)
            store = ChromaDBStore()

            # Should be same instance
            assert store.client is singleton_client, "ChromaDBStore should use singleton client"

    def test_multiple_stores_share_client(self, temp_chroma_dir: Path) -> None:
        """Test that multiple ChromaDBStore instances share same client."""
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            # Create two stores
            store1 = ChromaDBStore()
            store2 = ChromaDBStore()

            # Should share same client
            assert (
                store1.client is store2.client
            ), "Multiple stores should share singleton client"

    def test_persist_path_ignored(self, temp_chroma_dir: Path) -> None:
        """Test that persist_path parameter is ignored (uses singleton)."""
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            # Create stores with different persist_path (should be ignored)
            store1 = ChromaDBStore(persist_path="/path/to/somewhere")
            store2 = ChromaDBStore(persist_path="/different/path")

            # Both should use singleton client (same instance)
            assert (
                store1.client is store2.client
            ), "persist_path should be ignored, both use singleton"


class TestDenseSearchIntegration:
    """Test that DenseSearch uses singleton client."""

    def test_dense_search_uses_singleton(self, temp_chroma_dir: Path) -> None:
        """Test that DenseSearch uses singleton client."""
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            # Get singleton client directly
            singleton_client = get_chromadb_client()

            # Create DenseSearch (should use same client)
            dense_search = DenseSearch()

            # Should be same instance
            assert (
                dense_search.client is singleton_client
            ), "DenseSearch should use singleton client"

    def test_dense_search_and_store_share_client(self, temp_chroma_dir: Path) -> None:
        """Test that DenseSearch and ChromaDBStore share same client.

        This is the critical test: verifies fix for the original bug where
        dense_search.py and chromadb_store.py created separate clients with
        different settings for the same path.
        """
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            # Create both DenseSearch and ChromaDBStore
            store = ChromaDBStore()
            dense_search = DenseSearch()

            # CRITICAL: Should share same client instance
            assert (
                store.client is dense_search.client
            ), "DenseSearch and ChromaDBStore MUST share singleton client"

            # Verify they can see same collections
            collection = store.get_or_create_collection(session_id="test_session", dimensions=384)
            assert collection.name == "session_test_session"

            # DenseSearch should see the same collection
            retrieved = dense_search.client.get_collection("session_test_session")
            assert retrieved.name == collection.name


class TestConcurrentAccess:
    """Test thread-safety of singleton client."""

    def test_concurrent_client_access(self, temp_chroma_dir: Path) -> None:
        """Test that singleton is safe for concurrent access.

        Note: This is a smoke test. Real concurrency testing would require
        threading or multiprocessing, which is beyond scope for unit tests.
        """
        get_chromadb_client.cache_clear()

        with patch("article_mind_service.embeddings.client.settings") as mock_settings:
            mock_settings.chroma_persist_directory = str(temp_chroma_dir)

            # Simulate concurrent access by creating multiple stores
            stores = [ChromaDBStore() for _ in range(10)]

            # All should use same client
            first_client = stores[0].client
            for store in stores[1:]:
                assert (
                    store.client is first_client
                ), "All concurrent stores should use singleton client"


@pytest.fixture(autouse=True)
def cleanup_singleton_cache() -> None:
    """Clear singleton cache after each test to prevent test interference.

    This fixture runs automatically after each test in this module.
    """
    yield
    # Cleanup after test
    get_chromadb_client.cache_clear()
