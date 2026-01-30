"""Tests for embedding cache with memory LRU and disk persistence.

Test Coverage:
- Cache miss returns None
- Cache hit returns correct embedding (memory)
- Cache hit returns correct embedding (disk, after memory eviction)
- LRU eviction works (exceed max_memory_entries)
- Disk persistence (put, clear memory, get from disk)
- Cache key determinism (same text = same key)
- Corrupted disk file handled gracefully
- Cache stats tracking (hits, misses, hit_rate)
- clear() removes both memory and disk entries
- Batch integration: mix of cached and uncached texts
"""

import json
import tempfile
from pathlib import Path

import pytest

from article_mind_service.embeddings.cache import EmbeddingCache


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def cache(temp_cache_dir):
    """Create embedding cache with temporary directory."""
    return EmbeddingCache(
        cache_dir=temp_cache_dir,
        max_memory_entries=3,  # Small limit for easier testing
        max_disk_size_mb=100,
    )


def test_cache_miss_returns_none(cache):
    """Test that cache miss returns None."""
    result = cache.get("nonexistent text")
    assert result is None
    assert cache.stats["misses"] == 1
    assert cache.stats["hits"] == 0


def test_cache_hit_memory(cache):
    """Test cache hit from memory returns correct embedding."""
    # Put embedding in cache
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    cache.put("Hello world", embedding)

    # Get from cache (should hit memory)
    result = cache.get("Hello world")

    assert result == embedding
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 0
    assert cache.stats["memory_entries"] == 1


def test_cache_hit_disk_after_eviction(cache):
    """Test cache hit from disk after memory eviction."""
    # Fill cache beyond memory limit (max_memory_entries=3)
    embeddings = {
        "text1": [0.1, 0.2, 0.3],
        "text2": [0.4, 0.5, 0.6],
        "text3": [0.7, 0.8, 0.9],
        "text4": [1.0, 1.1, 1.2],  # This will evict text1 from memory
    }

    for text, emb in embeddings.items():
        cache.put(text, emb)

    # text1 should be evicted from memory but still on disk
    assert cache.stats["memory_entries"] == 3

    # Clear memory to force disk read
    cache._memory.clear()
    assert cache.stats["memory_entries"] == 0

    # Get text1 from disk (should promote to memory)
    result = cache.get("text1")

    assert result == [0.1, 0.2, 0.3]
    assert cache.stats["memory_entries"] == 1  # Promoted back to memory
    # Verify text1's key is in memory
    key1 = cache._cache_key("text1")
    assert key1 in cache._memory


def test_lru_eviction(cache):
    """Test LRU eviction when exceeding max_memory_entries."""
    # Add 4 entries (max is 3)
    cache.put("text1", [1.0])
    cache.put("text2", [2.0])
    cache.put("text3", [3.0])

    # Verify all in memory
    assert cache.stats["memory_entries"] == 3

    # Add one more (should evict text1, the oldest)
    cache.put("text4", [4.0])

    assert cache.stats["memory_entries"] == 3

    # text1 should not be in memory
    key1 = cache._cache_key("text1")
    assert key1 not in cache._memory

    # But text2, text3, text4 should be
    key2 = cache._cache_key("text2")
    key3 = cache._cache_key("text3")
    key4 = cache._cache_key("text4")
    assert key2 in cache._memory
    assert key3 in cache._memory
    assert key4 in cache._memory


def test_disk_persistence(cache):
    """Test disk persistence survives memory clear."""
    # Put embedding
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    cache.put("Persistent text", embedding)

    # Verify file exists on disk
    key = cache._cache_key("Persistent text")
    disk_path = Path(cache.cache_dir) / f"{key}.json"
    assert disk_path.exists()

    # Clear memory (simulate restart)
    cache._memory.clear()
    assert cache.stats["memory_entries"] == 0

    # Get from cache (should read from disk)
    result = cache.get("Persistent text")

    assert result == embedding
    assert cache.stats["memory_entries"] == 1  # Promoted to memory


def test_cache_key_determinism(cache):
    """Test that same text produces same cache key."""
    text = "Deterministic text"
    key1 = cache._cache_key(text)
    key2 = cache._cache_key(text)

    assert key1 == key2
    assert len(key1) == 16  # 16 hex characters

    # Different text should produce different key
    key3 = cache._cache_key("Different text")
    assert key3 != key1


def test_corrupted_disk_file_handled_gracefully(cache):
    """Test that corrupted disk file is removed and treated as cache miss."""
    # Create corrupted cache file
    key = cache._cache_key("corrupted")
    disk_path = Path(cache.cache_dir) / f"{key}.json"

    # Write invalid JSON
    with open(disk_path, "w") as f:
        f.write("{ invalid json }")

    # Get should return None and delete corrupted file
    result = cache.get("corrupted")

    assert result is None
    assert not disk_path.exists()  # Corrupted file removed
    assert cache.stats["misses"] == 1


def test_invalid_embedding_format_handled(cache):
    """Test that invalid embedding format is removed and treated as cache miss."""
    # Create cache file with invalid embedding (not a list)
    key = cache._cache_key("invalid_format")
    disk_path = Path(cache.cache_dir) / f"{key}.json"

    # Write invalid embedding (string instead of list)
    with open(disk_path, "w") as f:
        json.dump("not_a_list", f)

    # Get should return None and delete invalid file
    result = cache.get("invalid_format")

    assert result is None
    assert not disk_path.exists()  # Invalid file removed
    assert cache.stats["misses"] == 1


def test_cache_stats_tracking(cache):
    """Test cache stats tracking (hits, misses, hit_rate)."""
    # Initial stats
    assert cache.stats == {
        "memory_entries": 0,
        "hits": 0,
        "misses": 0,
        "hit_rate": 0.0,
    }

    # Add embedding
    cache.put("text1", [1.0])

    # First get (hit)
    cache.get("text1")
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 0
    assert cache.stats["hit_rate"] == 1.0

    # Second get (hit)
    cache.get("text1")
    assert cache.stats["hits"] == 2
    assert cache.stats["misses"] == 0
    assert cache.stats["hit_rate"] == 1.0

    # Miss
    cache.get("nonexistent")
    assert cache.stats["hits"] == 2
    assert cache.stats["misses"] == 1
    assert cache.stats["hit_rate"] == 2 / 3  # 2 hits out of 3 total


def test_clear_removes_memory_and_disk(cache):
    """Test that clear() removes both memory and disk entries."""
    # Add multiple entries
    cache.put("text1", [1.0])
    cache.put("text2", [2.0])
    cache.put("text3", [3.0])

    # Verify files exist
    assert cache.stats["memory_entries"] == 3
    cache_files = list(Path(cache.cache_dir).glob("*.json"))
    assert len(cache_files) == 3

    # Clear cache
    cache.clear()

    # Verify memory cleared
    assert cache.stats["memory_entries"] == 0
    assert cache.stats["hits"] == 0
    assert cache.stats["misses"] == 0

    # Verify disk cleared
    cache_files = list(Path(cache.cache_dir).glob("*.json"))
    assert len(cache_files) == 0


def test_batch_integration_mixed_cache_hits(cache):
    """Test batch processing with mix of cached and uncached texts."""
    # Pre-populate cache with some texts
    cache.put("cached1", [0.1, 0.2])
    cache.put("cached2", [0.3, 0.4])

    # Batch of texts (mix of cached and uncached)
    texts = ["cached1", "uncached1", "cached2", "uncached2"]

    # Process batch (simulating pipeline behavior)
    embeddings = []
    uncached_texts = []

    for text in texts:
        cached_emb = cache.get(text)
        if cached_emb is not None:
            embeddings.append(cached_emb)
        else:
            # In real code, this would call provider.embed()
            new_emb = [float(len(text))]  # Dummy embedding
            cache.put(text, new_emb)
            embeddings.append(new_emb)
            uncached_texts.append(text)

    # Verify results
    assert len(embeddings) == 4
    assert embeddings[0] == [0.1, 0.2]  # cached1
    assert embeddings[2] == [0.3, 0.4]  # cached2
    assert len(uncached_texts) == 2
    assert "uncached1" in uncached_texts
    assert "uncached2" in uncached_texts

    # Verify cache stats
    assert cache.stats["hits"] == 2  # cached1 and cached2
    assert cache.stats["misses"] == 2  # uncached1 and uncached2


def test_put_creates_cache_directory(temp_cache_dir):
    """Test that cache directory is created if it doesn't exist."""
    cache_dir = Path(temp_cache_dir) / "nested" / "cache"

    # Directory shouldn't exist yet
    assert not cache_dir.exists()

    # Create cache (should create directory)
    cache = EmbeddingCache(cache_dir=str(cache_dir))

    # Directory should now exist
    assert cache_dir.exists()

    # Put should work
    cache.put("test", [1.0])
    assert cache.get("test") == [1.0]


def test_lru_ordering_with_access(cache):
    """Test that accessing entries updates LRU order."""
    # Add three entries
    cache.put("text1", [1.0])
    cache.put("text2", [2.0])
    cache.put("text3", [3.0])

    # Access text1 (moves to end of LRU)
    cache.get("text1")

    # Add text4 (should evict text2, not text1)
    cache.put("text4", [4.0])

    # text1 should still be in memory (was accessed recently)
    key1 = cache._cache_key("text1")
    assert key1 in cache._memory

    # text2 should be evicted (oldest unaccessed)
    key2 = cache._cache_key("text2")
    assert key2 not in cache._memory


def test_cache_with_empty_text(cache):
    """Test cache with empty text."""
    # Empty text should be cacheable
    cache.put("", [0.0])
    result = cache.get("")

    assert result == [0.0]
    assert cache.stats["hits"] == 1


def test_cache_with_large_embedding(cache):
    """Test cache with large embedding vector (1536 dimensions like OpenAI)."""
    # Create large embedding vector
    large_embedding = [float(i) for i in range(1536)]

    cache.put("large text", large_embedding)
    result = cache.get("large text")

    assert result == large_embedding
    assert len(result) == 1536


def test_cache_with_unicode_text(cache):
    """Test cache with Unicode text."""
    unicode_text = "Hello 世界 🌍"
    embedding = [0.1, 0.2, 0.3]

    cache.put(unicode_text, embedding)
    result = cache.get(unicode_text)

    assert result == embedding


def test_disk_write_failure_handled_gracefully(cache, monkeypatch):
    """Test that disk write failure doesn't crash, just logs warning."""
    # Mock open() to raise OSError
    def mock_open(*args, **kwargs):
        raise OSError("Disk full")

    monkeypatch.setattr("builtins.open", mock_open)

    # Put should not raise exception (just log warning)
    cache.put("test", [1.0])

    # Memory cache should still work
    result = cache.get("test")
    assert result == [1.0]


def test_hit_rate_with_no_accesses(cache):
    """Test hit_rate returns 0.0 when no cache accesses."""
    assert cache.hit_rate == 0.0
    assert cache.stats["hit_rate"] == 0.0
