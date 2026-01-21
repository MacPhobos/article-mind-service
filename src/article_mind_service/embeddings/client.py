"""ChromaDB client singleton for consistent client management.

This module provides a singleton ChromaDB client to prevent client conflicts
across different modules that access the same persistent storage path.

Design Decision: Singleton Pattern
    - PROBLEM: Multiple modules (chromadb_store.py, dense_search.py) were creating
      ChromaDB clients with different settings for the same persist path
    - SYMPTOM: Database lock conflicts, inconsistent behavior
    - SOLUTION: Single shared client instance with consistent settings

Implementation: functools.lru_cache
    - Thread-safe singleton via @lru_cache(maxsize=1)
    - Settings frozen at first call (consistent for entire process lifetime)
    - No need for complex thread locking mechanisms

Settings Consistency:
    - anonymized_telemetry=False (privacy, no external calls)
    - allow_reset=True (enables testing and cleanup)
    - Path from settings.chroma_persist_directory (configurable)

Thread Safety:
    - ChromaDB PersistentClient is thread-safe
    - @lru_cache ensures single instance across threads
    - Safe for concurrent async requests

Usage:
    from article_mind_service.embeddings.client import get_chromadb_client

    client = get_chromadb_client()
    collection = client.get_or_create_collection("my_collection")
"""

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from article_mind_service.config import settings


@lru_cache(maxsize=1)
def get_chromadb_client() -> Any:  # chromadb.PersistentClient doesn't export proper type
    """Get singleton ChromaDB client instance.

    Returns the same client instance across all calls within the process lifetime.
    Uses consistent settings to prevent client conflicts.

    Returns:
        ChromaDB PersistentClient with consistent configuration.

    Design Decisions:
        1. Singleton via @lru_cache(maxsize=1):
           - First call creates client and caches it
           - Subsequent calls return cached instance
           - Thread-safe (functools.lru_cache handles locking)

        2. Settings from Config:
           - Path: settings.chroma_persist_directory (configurable via env)
           - Telemetry: False (no external calls)
           - Reset: True (enables cleanup for testing)

        3. No Manual Reset Function:
           - Client persists for process lifetime
           - Tests use temporary directories (via mock_settings fixture)
           - Production uses single persistent path

    Performance:
        - First call: ~50-100ms (creates client, initializes DB)
        - Subsequent calls: <1ms (returns cached instance)

    Thread Safety:
        - lru_cache is thread-safe by default
        - ChromaDB PersistentClient is designed for concurrent access
        - Safe to call from multiple async handlers simultaneously

    Example:
        # Module A
        from article_mind_service.embeddings.client import get_chromadb_client

        client = get_chromadb_client()
        collection = client.get_or_create_collection("session_123")

        # Module B (gets SAME client instance)
        client = get_chromadb_client()
        collection = client.get_collection("session_123")  # Same collection

    Testing:
        Tests should use temporary directories via mock_settings fixture:

        def test_chromadb(temp_chroma_dir):
            # Override settings.chroma_persist_directory for test
            with patch('article_mind_service.embeddings.client.settings') as mock:
                mock.chroma_persist_directory = str(temp_chroma_dir)
                # Must clear lru_cache to pick up new path
                get_chromadb_client.cache_clear()

                client = get_chromadb_client()
                # Client uses temp directory
    """
    return chromadb.PersistentClient(
        path=str(settings.chroma_persist_directory),
        settings=ChromaSettings(
            anonymized_telemetry=False,  # Disable external telemetry calls
            allow_reset=True,  # Enable reset for testing/cleanup
        ),
    )
