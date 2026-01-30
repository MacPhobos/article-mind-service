"""Tests for embedding retry logic and progress tracking."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from article_mind_service.embeddings.exceptions import EmbeddingError
from article_mind_service.embeddings.pipeline import (
    EmbeddingProgress,
    _is_permanent_error,
    embed_with_retry,
)


class TestEmbedWithRetry:
    """Test suite for embed_with_retry function."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self) -> None:
        """Test that successful embedding on first attempt requires no retry."""
        # Arrange
        provider = AsyncMock()
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4]]
        provider.embed.return_value = expected_embeddings
        texts = ["text1", "text2"]

        # Act
        result = await embed_with_retry(provider, texts, max_retries=3)

        # Assert
        assert result == expected_embeddings
        assert provider.embed.call_count == 1
        provider.embed.assert_called_once_with(texts)

    @pytest.mark.asyncio
    async def test_succeeds_on_second_attempt_after_transient_error(self) -> None:
        """Test that transient error triggers retry and succeeds on second attempt."""
        # Arrange
        provider = AsyncMock()
        expected_embeddings = [[0.1, 0.2]]
        # First call raises transient error, second succeeds
        provider.embed.side_effect = [
            EmbeddingError("429 Rate limit exceeded"),
            expected_embeddings,
        ]
        texts = ["text1"]

        # Act
        with patch("asyncio.sleep", new_callable=AsyncMock):  # Skip actual delay
            result = await embed_with_retry(provider, texts, max_retries=3, base_delay=1.0)

        # Assert
        assert result == expected_embeddings
        assert provider.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_fails_after_max_retries_exhausted(self) -> None:
        """Test that function raises after all retries are exhausted."""
        # Arrange
        provider = AsyncMock()
        # All attempts fail with transient error
        provider.embed.side_effect = EmbeddingError("Network timeout")
        texts = ["text1"]

        # Act & Assert
        with patch("asyncio.sleep", new_callable=AsyncMock):  # Skip actual delay
            with pytest.raises(EmbeddingError, match="Network timeout"):
                await embed_with_retry(provider, texts, max_retries=3, base_delay=1.0)

        # Should attempt 3 times (max_retries=3)
        assert provider.embed.call_count == 3

    @pytest.mark.asyncio
    async def test_permanent_error_raises_immediately_no_retry(self) -> None:
        """Test that permanent errors (auth, invalid input) do not trigger retry."""
        # Arrange
        provider = AsyncMock()
        provider.embed.side_effect = EmbeddingError("401 Unauthorized: invalid API key")
        texts = ["text1"]

        # Act & Assert
        with pytest.raises(EmbeddingError, match="invalid API key"):
            await embed_with_retry(provider, texts, max_retries=3, base_delay=1.0)

        # Should only attempt once (no retries for permanent errors)
        assert provider.embed.call_count == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self) -> None:
        """Test that retry delays follow exponential backoff pattern."""
        # Arrange
        provider = AsyncMock()
        provider.embed.side_effect = [
            EmbeddingError("500 Server error"),
            EmbeddingError("500 Server error"),
            EmbeddingError("500 Server error"),
        ]
        texts = ["text1"]

        # Track sleep calls to verify backoff
        sleep_delays = []

        async def mock_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        # Act
        with patch("asyncio.sleep", side_effect=mock_sleep):
            with pytest.raises(EmbeddingError):
                await embed_with_retry(provider, texts, max_retries=3, base_delay=1.0)

        # Assert: Delays should be exponential (with jitter)
        # Expected pattern: ~1s, ~2s (before 3rd attempt)
        assert len(sleep_delays) == 2  # 2 retries = 2 delays
        # First delay: 2^0 * 1.0 = 1.0 ± 0.5 = [0.5, 1.5]
        assert 0.5 <= sleep_delays[0] <= 1.5
        # Second delay: 2^1 * 1.0 = 2.0 ± 0.5 = [1.5, 2.5]
        assert 1.5 <= sleep_delays[1] <= 2.5

    @pytest.mark.asyncio
    async def test_jitter_is_applied(self) -> None:
        """Test that jitter is applied to prevent thundering herd."""
        # Arrange
        texts = ["text1"]

        # Track sleep calls across multiple runs
        all_delays = []

        async def mock_sleep(delay: float) -> None:
            all_delays.append(delay)

        # Act: Run multiple times to verify jitter varies
        for _ in range(5):
            # Create new provider for each iteration
            provider = AsyncMock()
            provider.embed.side_effect = [
                EmbeddingError("503 Service unavailable"),
                EmbeddingError("503 Service unavailable"),
            ]

            with patch("asyncio.sleep", side_effect=mock_sleep):
                with pytest.raises(EmbeddingError):
                    await embed_with_retry(provider, texts, max_retries=2, base_delay=1.0)

        # Assert: We should see variation in delays due to jitter
        # Each iteration should have 1 delay (1 retry before 2nd failure)
        assert len(all_delays) == 5

        # All delays should be in the expected range: 2^0 * 1.0 ± 0.5 = [0.5, 1.5]
        for delay in all_delays:
            assert 0.5 <= delay <= 1.5

        # With random jitter, we expect some variation (not all identical)
        unique_delays = len(set(round(d, 1) for d in all_delays))
        # Expect at least 2 different delay values with 5 samples
        # (This is probabilistic but should pass with very high confidence)
        assert unique_delays >= 2, f"Expected variation in delays, got: {all_delays}"


class TestErrorClassification:
    """Test suite for _is_permanent_error function."""

    def test_auth_errors_are_permanent(self) -> None:
        """Test that authentication errors are classified as permanent."""
        permanent_errors = [
            EmbeddingError("invalid API key"),
            EmbeddingError("401 Unauthorized"),
            EmbeddingError("403 Forbidden"),
            EmbeddingError("Authentication failed"),
        ]
        for error in permanent_errors:
            assert _is_permanent_error(error), f"Should be permanent: {error}"

    def test_invalid_input_errors_are_permanent(self) -> None:
        """Test that invalid input errors are classified as permanent."""
        permanent_errors = [
            EmbeddingError("400 Bad Request"),
            EmbeddingError("Invalid input: text too long"),
            EmbeddingError("Model not found"),
        ]
        for error in permanent_errors:
            assert _is_permanent_error(error), f"Should be permanent: {error}"

    def test_transient_errors_are_not_permanent(self) -> None:
        """Test that transient errors are NOT classified as permanent."""
        transient_errors = [
            EmbeddingError("429 Rate limit exceeded"),
            EmbeddingError("500 Internal server error"),
            EmbeddingError("502 Bad gateway"),
            EmbeddingError("503 Service unavailable"),
            EmbeddingError("Connection timeout"),
            EmbeddingError("Network error"),
        ]
        for error in transient_errors:
            assert not _is_permanent_error(error), f"Should be transient: {error}"

    def test_case_insensitive_matching(self) -> None:
        """Test that error classification is case-insensitive."""
        assert _is_permanent_error(EmbeddingError("INVALID API KEY"))
        assert _is_permanent_error(EmbeddingError("Invalid Api Key"))
        assert _is_permanent_error(EmbeddingError("authentication failed"))


class TestEmbeddingProgress:
    """Test suite for EmbeddingProgress class."""

    def test_initialization(self) -> None:
        """Test that progress tracker initializes with correct state."""
        progress = EmbeddingProgress(article_id=123, total_chunks=500)
        assert progress.article_id == 123
        assert progress.total_chunks == 500
        assert progress.last_successful_batch == -1
        assert progress.failed_batches == []
        assert not progress.is_complete
        assert progress.resumable_from == 0

    def test_mark_batch_complete(self) -> None:
        """Test marking batches as complete updates state correctly."""
        progress = EmbeddingProgress(article_id=123, total_chunks=500)

        progress.mark_batch_complete(0)
        assert progress.last_successful_batch == 0
        assert progress.resumable_from == 1

        progress.mark_batch_complete(1)
        assert progress.last_successful_batch == 1
        assert progress.resumable_from == 2

    def test_mark_batch_failed(self) -> None:
        """Test marking batches as failed updates state correctly."""
        progress = EmbeddingProgress(article_id=123, total_chunks=500)

        progress.mark_batch_failed(2)
        assert 2 in progress.failed_batches
        assert not progress.is_complete

    def test_is_complete_all_batches_succeed(self) -> None:
        """Test is_complete returns True when all batches succeed."""
        progress = EmbeddingProgress(article_id=123, total_chunks=200)

        # Complete batches 0 and 1 (covers all 200 chunks)
        progress.mark_batch_complete(0)
        progress.mark_batch_complete(1)

        assert progress.is_complete
        assert len(progress.failed_batches) == 0

    def test_is_complete_with_failures(self) -> None:
        """Test is_complete returns False when any batch fails."""
        progress = EmbeddingProgress(article_id=123, total_chunks=300)

        progress.mark_batch_complete(0)
        progress.mark_batch_complete(1)
        progress.mark_batch_failed(2)  # Batch 2 failed

        assert not progress.is_complete

    def test_resumable_from(self) -> None:
        """Test resumable_from returns correct batch index to resume."""
        progress = EmbeddingProgress(article_id=123, total_chunks=500)

        # Initially, resume from batch 0
        assert progress.resumable_from == 0

        # After completing batch 0, resume from batch 1
        progress.mark_batch_complete(0)
        assert progress.resumable_from == 1

        # After completing batch 1, resume from batch 2
        progress.mark_batch_complete(1)
        assert progress.resumable_from == 2

    def test_summary_all_succeed(self) -> None:
        """Test summary string when all batches succeed."""
        progress = EmbeddingProgress(article_id=123, total_chunks=200)

        progress.mark_batch_complete(0)
        progress.mark_batch_complete(1)

        summary = progress.summary()
        assert "Article 123" in summary
        assert "batch 2/2 complete" in summary
        assert "failed" not in summary.lower()

    def test_summary_with_failures(self) -> None:
        """Test summary string when some batches fail."""
        progress = EmbeddingProgress(article_id=123, total_chunks=300)

        progress.mark_batch_complete(0)
        progress.mark_batch_complete(1)
        progress.mark_batch_failed(2)

        summary = progress.summary()
        assert "Article 123" in summary
        assert "batch 2/3 complete" in summary
        assert "1 failed" in summary

    def test_summary_multiple_failures(self) -> None:
        """Test summary string with multiple failed batches."""
        progress = EmbeddingProgress(article_id=456, total_chunks=500)

        progress.mark_batch_complete(0)
        progress.mark_batch_failed(1)
        progress.mark_batch_failed(2)

        summary = progress.summary()
        assert "Article 456" in summary
        assert "2 failed" in summary

    def test_partial_failure_resume(self) -> None:
        """Test that progress allows resuming after partial failure."""
        progress = EmbeddingProgress(article_id=123, total_chunks=500)

        # Simulate processing: batch 0 succeeds, batch 1 fails, batch 2 succeeds
        progress.mark_batch_complete(0)
        progress.mark_batch_failed(1)
        progress.mark_batch_complete(2)

        # Should resume from batch 3 (after last successful batch 2)
        assert progress.resumable_from == 3
        assert 1 in progress.failed_batches
        assert not progress.is_complete
