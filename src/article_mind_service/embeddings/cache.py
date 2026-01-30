"""Embedding cache with memory LRU + disk persistence.

Design Decision: Two-Tier Cache Architecture
============================================

Rationale: Reduce redundant embedding API calls by caching embeddings both in memory
and on disk, with memory serving as a fast L1 cache and disk as persistent L2 cache.

Cache Key: sha256(text)[:16] (16-character hexadecimal hash)
Cache Value: list[float] (embedding vector)

Architecture:
- Tier 1 (Memory): OrderedDict-based LRU cache, max 2000 entries
  - O(1) access time
  - Evicts least recently used when full
  - Lost on process restart
- Tier 2 (Disk): JSON files in data/embedding_cache/, keyed by hash
  - Persistent across restarts
  - Automatic promotion to memory on access
  - Simple file-per-embedding design for easy debugging

Benefits:
- Cost Savings: Eliminate redundant embedding API calls (OpenAI charges per token)
- Performance: Memory cache provides sub-millisecond access
- Persistence: Disk cache survives service restarts
- Deduplication: Identical text chunks produce same cache key

Trade-offs:
- Disk I/O: Adds ~1-5ms per cache miss (vs. 100-500ms for API call)
- Storage: ~4KB per embedding (1536 floats * 2.5 bytes/float as JSON)
- Complexity: Two-tier system vs. simple in-memory only
- Cache Invalidation: No TTL (assumes embeddings never change for same text)

Performance:
- Memory hit: <1ms (O(1) dict lookup + move_to_end)
- Disk hit: ~1-5ms (file read + JSON parse + memory promotion)
- Cache miss: 100-500ms (embedding API call + cache write)

Alternatives Considered:
1. Redis: Rejected due to operational complexity and network latency
2. SQLite: Rejected due to disk I/O overhead on writes
3. In-memory only: Rejected due to loss of cache on restart
4. Disk only: Rejected due to slow access on every lookup

Extension Points:
- Can migrate to Redis if distributed caching becomes necessary
- Can add TTL if embedding models change frequently
- Can add compression if disk usage becomes issue (gzip reduces size 60-80%)
"""

import hashlib
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Two-tier embedding cache with memory LRU and disk persistence.

    Time Complexity:
    - get() from memory: O(1) average case
    - get() from disk: O(1) hash lookup + O(n) JSON parse where n = embedding size
    - put(): O(1) memory insert + O(n) disk write

    Space Complexity:
    - Memory: O(k * d) where k = max_memory_entries, d = embedding dimensions
    - Disk: O(n * d) where n = total unique embeddings

    Example:
        >>> cache = EmbeddingCache()
        >>> cache.put("Hello world", [0.1, 0.2, ..., 0.9])
        >>> embedding = cache.get("Hello world")
        >>> cache.stats
        {'memory_entries': 1, 'hits': 1, 'misses': 0, 'hit_rate': 1.0}
    """

    def __init__(
        self,
        cache_dir: str = "./data/embedding_cache",
        max_memory_entries: int = 2000,
        max_disk_size_mb: int = 500,
    ):
        """Initialize embedding cache.

        Args:
            cache_dir: Directory for disk cache files.
            max_memory_entries: Maximum entries in memory LRU cache.
            max_disk_size_mb: Maximum disk cache size (currently not enforced,
                reserved for future implementation).

        Design Decision: max_disk_size_mb not enforced
        - Current implementation: No disk size limit (rely on filesystem)
        - Future: Add background cleanup when disk usage exceeds limit
        - Rationale: Keep initial implementation simple, add cleanup when needed
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.max_disk_size_mb = max_disk_size_mb
        self._memory: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate cache key from text using SHA-256 hash.

        Args:
            text: Text to hash.

        Returns:
            First 16 characters of SHA-256 hash (hexadecimal).

        Performance:
        - Time Complexity: O(n) where n = len(text)
        - Typical speed: <1ms for 1000 characters
        - SHA-256 is cryptographically secure and well-optimized

        Collision Probability:
        - 16 hex chars = 64 bits = 2^64 combinations
        - Probability of collision: ~1 in 18 quintillion
        - Acceptable for embedding cache (false positives extremely rare)

        Example:
            >>> EmbeddingCache._cache_key("Hello world")
            '64ec88ca00b268e5'
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> Optional[list[float]]:
        """Look up embedding: memory first, then disk.

        Args:
            text: Text to look up embedding for.

        Returns:
            Embedding vector if found, None if not cached.

        Performance:
        - Memory hit: <1ms (O(1) dict lookup)
        - Disk hit: ~1-5ms (file I/O + JSON parse + memory promotion)
        - Cache miss: returns None immediately

        Side Effects:
        - On memory hit: Moves entry to end of LRU (most recently used)
        - On disk hit: Promotes embedding to memory cache
        - Increments hit/miss counters for statistics

        Example:
            >>> cache = EmbeddingCache()
            >>> cache.put("Hello", [0.1, 0.2, 0.3])
            >>> embedding = cache.get("Hello")
            >>> embedding
            [0.1, 0.2, 0.3]
        """
        key = self._cache_key(text)

        # Tier 1: Memory
        if key in self._memory:
            self._memory.move_to_end(key)  # O(1) mark as recently used
            self._hits += 1
            return self._memory[key]

        # Tier 2: Disk
        disk_path = self.cache_dir / f"{key}.json"
        if disk_path.exists():
            try:
                with open(disk_path, "r") as f:
                    embedding = json.load(f)

                # Validate embedding is a list of floats
                if not isinstance(embedding, list) or not all(
                    isinstance(x, (int, float)) for x in embedding
                ):
                    logger.warning("Invalid embedding format in cache file: %s", disk_path)
                    disk_path.unlink(missing_ok=True)
                    self._misses += 1
                    return None

                # Promote to memory cache
                self._put_memory(key, embedding)
                self._hits += 1
                return embedding

            except (json.JSONDecodeError, OSError) as e:
                # Corrupted cache file, remove it
                logger.warning("Corrupted cache file %s: %s", disk_path, e)
                disk_path.unlink(missing_ok=True)

        self._misses += 1
        return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store embedding in both memory and disk caches.

        Args:
            text: Text that was embedded.
            embedding: Embedding vector (list of floats).

        Performance:
        - Memory write: O(1) dict insert + O(1) eviction if full
        - Disk write: O(n) where n = len(embedding) for JSON serialization

        Side Effects:
        - Writes to disk (blocks for ~1-5ms on typical systems)
        - May evict least recently used entry from memory if full
        - Creates cache directory if it doesn't exist

        Example:
            >>> cache = EmbeddingCache()
            >>> cache.put("Hello world", [0.1, 0.2, 0.3])
            >>> cache.stats['memory_entries']
            1
        """
        key = self._cache_key(text)
        self._put_memory(key, embedding)
        self._put_disk(key, embedding)

    def _put_memory(self, key: str, embedding: list[float]) -> None:
        """Store embedding in memory LRU cache.

        Args:
            key: Cache key (16-char hash).
            embedding: Embedding vector.

        Design Decision: LRU Eviction with OrderedDict
        - OrderedDict maintains insertion/access order
        - move_to_end() marks as recently used (O(1))
        - popitem(last=False) evicts least recently used (O(1))
        - Alternative: Use lru_cache decorator (rejected: need custom logic)

        Performance:
        - Insert: O(1) amortized
        - Eviction: O(1) when full
        - Memory overhead: ~56 bytes per entry (dict overhead + pointer)
        """
        self._memory[key] = embedding
        self._memory.move_to_end(key)  # Mark as recently used

        # Evict least recently used if over limit
        while len(self._memory) > self.max_memory_entries:
            self._memory.popitem(last=False)  # Remove oldest entry

    def _put_disk(self, key: str, embedding: list[float]) -> None:
        """Store embedding as JSON file on disk.

        Args:
            key: Cache key (16-char hash, used as filename).
            embedding: Embedding vector.

        Design Decision: File-Per-Embedding vs. Single Database File
        - Chosen: File-per-embedding (simple, easy to debug, no locking)
        - Rejected: SQLite (overhead, write contention, complexity)
        - Rejected: Single JSON file (requires locking, slow to update)

        Performance:
        - Write time: ~1-5ms on SSD, ~10-50ms on HDD
        - File size: ~4KB for 1536-dimensional embedding (JSON overhead)

        Error Handling:
        - Logs warning on OSError (disk full, permissions, etc.)
        - Does NOT raise exception (cache write failure is non-fatal)
        - Missing cache entries are simply cache misses
        """
        try:
            disk_path = self.cache_dir / f"{key}.json"
            with open(disk_path, "w") as f:
                json.dump(embedding, f)
        except OSError as e:
            logger.warning("Failed to write embedding cache to %s: %s", disk_path, e)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as float between 0.0 and 1.0.
            Returns 0.0 if no cache accesses yet.

        Example:
            >>> cache = EmbeddingCache()
            >>> cache.put("A", [1.0])
            >>> cache.get("A")  # Hit
            >>> cache.get("B")  # Miss
            >>> cache.hit_rate
            0.5
        """
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with:
            - memory_entries: Number of entries in memory cache
            - hits: Total cache hits (memory + disk)
            - misses: Total cache misses
            - hit_rate: Ratio of hits to total accesses (0.0-1.0)

        Example:
            >>> cache = EmbeddingCache()
            >>> cache.stats
            {'memory_entries': 0, 'hits': 0, 'misses': 0, 'hit_rate': 0.0}
        """
        return {
            "memory_entries": len(self._memory),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def clear(self) -> None:
        """Clear both memory and disk caches.

        Design Decision: Clear vs. Invalidate
        - Current: clear() removes all cache entries
        - Future: Add invalidate(pattern) for selective removal
        - Rationale: Start simple, add complexity when needed

        Performance:
        - Memory clear: O(n) where n = memory entries
        - Disk clear: O(m) where m = disk files (can be slow for large caches)

        Side Effects:
        - Deletes all JSON files from cache directory
        - Resets hit/miss counters
        - Does NOT delete cache directory itself

        Example:
            >>> cache = EmbeddingCache()
            >>> cache.put("A", [1.0])
            >>> cache.clear()
            >>> cache.stats['memory_entries']
            0
        """
        # Clear memory
        self._memory.clear()
        self._hits = 0
        self._misses = 0

        # Clear disk
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink(missing_ok=True)
