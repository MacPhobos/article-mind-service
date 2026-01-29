"""Unit tests for BM25 index persistence with pickle."""

import tempfile
from pathlib import Path

import pytest

from article_mind_service.search.sparse_search import BM25Index, BM25IndexCache


class TestBM25IndexPersistence:
    """Tests for BM25 index pickle persistence."""

    def test_persist_and_load_empty_index(self) -> None:
        """Test persisting and loading an empty BM25 index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty index
            index = BM25Index(session_id=1, persist_dir=tmpdir)
            index.persist()

            # Load from disk
            loaded = BM25Index.load(session_id=1, persist_dir=tmpdir)

            assert loaded is not None
            assert len(loaded) == 0
            assert loaded.session_id == 1

    def test_persist_and_load_with_documents(self) -> None:
        """Test persisting and loading index with documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index with documents
            index = BM25Index(session_id=2, persist_dir=tmpdir)
            index.add_document("chunk_1", "JWT authentication tokens")
            index.add_document("chunk_2", "Database connection pooling")
            index.add_document("chunk_3", "API documentation with OpenAPI")
            index.build()
            index.persist()

            # Load from disk
            loaded = BM25Index.load(session_id=2, persist_dir=tmpdir)

            assert loaded is not None
            assert len(loaded) == 3
            assert loaded.chunk_ids == ["chunk_1", "chunk_2", "chunk_3"]
            assert loaded.get_content("chunk_1") == "JWT authentication tokens"
            assert loaded.get_content("chunk_2") == "Database connection pooling"
            assert loaded.get_content("chunk_3") == "API documentation with OpenAPI"

    def test_load_nonexistent_file_returns_none(self) -> None:
        """Test loading from nonexistent file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = BM25Index.load(session_id=999, persist_dir=tmpdir)
            assert loaded is None

    def test_search_after_load_works(self) -> None:
        """Test that search works correctly after loading from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and persist index
            index = BM25Index(session_id=3, persist_dir=tmpdir)
            index.add_document("chunk_1", "User authentication using JWT tokens")
            index.add_document("chunk_2", "Database migration with Alembic")
            index.add_document("chunk_3", "API testing with pytest")
            index.build()
            index.persist()

            # Load from disk
            loaded = BM25Index.load(session_id=3, persist_dir=tmpdir)
            assert loaded is not None

            # Search should work
            results = loaded.search("JWT authentication", top_k=3)
            assert len(results) > 0
            # chunk_1 should rank highest (has both JWT and authentication)
            assert results[0][0] == "chunk_1"

    def test_version_mismatch_returns_none(self) -> None:
        """Test that version mismatch returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index with current version
            index = BM25Index(session_id=4, persist_dir=tmpdir)
            index.add_document("chunk_1", "test content")
            index.persist()

            # Manually corrupt version in pickle file
            persist_path = Path(tmpdir)
            file_path = persist_path / f"session_4_v{BM25Index.VERSION}.pkl"

            import pickle

            with open(file_path, "rb") as f:
                data = pickle.load(f)

            # Change version
            data["version"] = 999

            with open(file_path, "wb") as f:
                pickle.dump(data, f)

            # Load should return None due to version mismatch
            loaded = BM25Index.load(session_id=4, persist_dir=tmpdir)
            assert loaded is None

    def test_corrupted_file_returns_none(self) -> None:
        """Test that corrupted pickle file returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid index
            index = BM25Index(session_id=5, persist_dir=tmpdir)
            index.add_document("chunk_1", "test")
            index.persist()

            # Corrupt the file
            persist_path = Path(tmpdir)
            file_path = persist_path / f"session_5_v{BM25Index.VERSION}.pkl"

            with open(file_path, "wb") as f:
                f.write(b"corrupted pickle data")

            # Load should return None
            loaded = BM25Index.load(session_id=5, persist_dir=tmpdir)
            assert loaded is None

    def test_invalidate_disk_removes_file(self) -> None:
        """Test that invalidate_disk removes the pickle file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and persist index
            index = BM25Index(session_id=6, persist_dir=tmpdir)
            index.add_document("chunk_1", "test content")
            index.persist()

            # Verify file exists
            persist_path = Path(tmpdir)
            file_path = persist_path / f"session_6_v{BM25Index.VERSION}.pkl"
            assert file_path.exists()

            # Invalidate disk
            index.invalidate_disk()

            # File should be removed
            assert not file_path.exists()

    def test_invalidate_disk_idempotent(self) -> None:
        """Test that invalidate_disk is idempotent (doesn't error on missing file)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = BM25Index(session_id=7, persist_dir=tmpdir)

            # Should not raise even though file doesn't exist
            index.invalidate_disk()
            index.invalidate_disk()  # Call twice


class TestBM25IndexCachePersistence:
    """Tests for BM25IndexCache with disk persistence."""

    def test_cache_get_loads_from_disk(self) -> None:
        """Test that cache.get() loads from disk on memory miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and persist index (not via cache)
            index = BM25Index(session_id=10, persist_dir=tmpdir)
            index.add_document("chunk_1", "test content")
            index.persist()

            # Clear memory cache
            BM25IndexCache.invalidate(session_id=10)

            # Get should load from disk
            loaded = BM25IndexCache.get(session_id=10, persist_dir=tmpdir)
            assert loaded is not None
            assert len(loaded) == 1
            assert loaded.get_content("chunk_1") == "test content"

    def test_cache_get_returns_memory_first(self) -> None:
        """Test that cache.get() returns memory cache before checking disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index in memory
            index = BM25Index(session_id=11, persist_dir=tmpdir)
            index.add_document("chunk_1", "memory content")
            BM25IndexCache._indexes[11] = index

            # Create different index on disk
            disk_index = BM25Index(session_id=11, persist_dir=tmpdir)
            disk_index.add_document("chunk_1", "disk content")
            disk_index.persist()

            # Get should return memory version
            cached = BM25IndexCache.get(session_id=11, persist_dir=tmpdir)
            assert cached is not None
            assert cached.get_content("chunk_1") == "memory content"

    def test_cache_set_persists_to_disk(self) -> None:
        """Test that cache.set() persists index to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create index
            index = BM25Index(session_id=12, persist_dir=tmpdir)
            index.add_document("chunk_1", "test content")
            index.build()

            # Set in cache (should persist)
            BM25IndexCache.set(session_id=12, index=index)

            # Verify file exists
            persist_path = Path(tmpdir)
            file_path = persist_path / f"session_12_v{BM25Index.VERSION}.pkl"
            assert file_path.exists()

            # Clear memory and load from disk
            BM25IndexCache._indexes.clear()
            loaded = BM25Index.load(session_id=12, persist_dir=tmpdir)
            assert loaded is not None
            assert len(loaded) == 1

    def test_cache_invalidate_removes_disk_file(self) -> None:
        """Test that cache.invalidate() removes disk file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and cache index
            index = BM25Index(session_id=13, persist_dir=tmpdir)
            index.add_document("chunk_1", "test")
            BM25IndexCache.set(session_id=13, index=index)

            persist_path = Path(tmpdir)
            file_path = persist_path / f"session_13_v{BM25Index.VERSION}.pkl"
            assert file_path.exists()

            # Invalidate should remove file
            BM25IndexCache.invalidate(session_id=13)
            assert not file_path.exists()

    def test_populate_from_chunks_persists(self) -> None:
        """Test that populate_from_chunks persists to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = [
                ("chunk_1", "First chunk content"),
                ("chunk_2", "Second chunk content"),
                ("chunk_3", "Third chunk content"),
            ]

            # Populate (should persist)
            index = BM25IndexCache.populate_from_chunks(
                session_id=14, chunks=chunks, persist_dir=tmpdir
            )

            # Verify file exists
            persist_path = Path(tmpdir)
            file_path = persist_path / f"session_14_v{BM25Index.VERSION}.pkl"
            assert file_path.exists()

            # Clear memory and load
            BM25IndexCache._indexes.clear()
            loaded = BM25Index.load(session_id=14, persist_dir=tmpdir)
            assert loaded is not None
            assert len(loaded) == 3

    def test_get_or_create_tries_disk_before_creating(self) -> None:
        """Test that get_or_create tries disk before creating new index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and persist index
            index = BM25Index(session_id=15, persist_dir=tmpdir)
            index.add_document("chunk_1", "existing content")
            index.persist()

            # Clear memory
            BM25IndexCache._indexes.clear()

            # get_or_create should load from disk, not create new
            result = BM25IndexCache.get_or_create(session_id=15, persist_dir=tmpdir)
            assert len(result) == 1
            assert result.get_content("chunk_1") == "existing content"

    def test_get_or_create_creates_if_no_disk_file(self) -> None:
        """Test that get_or_create creates new index if no disk file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # get_or_create with no existing data
            result = BM25IndexCache.get_or_create(session_id=16, persist_dir=tmpdir)
            assert len(result) == 0
            assert result.session_id == 16
