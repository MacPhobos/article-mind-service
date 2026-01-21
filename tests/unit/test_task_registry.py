"""Unit tests for TaskRegistry progress tracking system.

Test Coverage:
- Task creation and unique ID generation
- Progress updates and state management
- SSE streaming with multiple listeners
- Task cancellation (cooperative)
- Error recording and tracking
- Multiple concurrent tasks (isolation)
- Terminal state handling (completed, failed, cancelled)
"""

import asyncio
from datetime import UTC, datetime

import pytest

from article_mind_service.tasks.registry import TaskProgress, TaskRegistry, task_registry


class TestTaskCreation:
    """Test task creation and ID generation."""

    def test_create_task_returns_unique_id(self) -> None:
        """Task creation returns unique UUID string."""
        registry = TaskRegistry()

        task_id = registry.create_task("reindex", total_items=100)

        assert isinstance(task_id, str)
        assert len(task_id) == 36  # UUID format: 8-4-4-4-12

    def test_create_task_initializes_state(self) -> None:
        """Task is created with correct initial state."""
        registry = TaskRegistry()

        task_id = registry.create_task("reindex", total_items=100)
        task = registry.get_task(task_id)

        assert task is not None
        assert task.task_id == task_id
        assert task.task_type == "reindex"
        assert task.status == "pending"
        assert task.total_items == 100
        assert task.processed_items == 0
        assert task.failed_items == 0
        assert task.current_item is None
        assert task.message is None
        assert task.completed_at is None
        assert task.errors == []

    def test_multiple_tasks_have_unique_ids(self) -> None:
        """Multiple tasks get different IDs."""
        registry = TaskRegistry()

        task_id1 = registry.create_task("reindex", total_items=100)
        task_id2 = registry.create_task("export", total_items=50)

        assert task_id1 != task_id2

    def test_singleton_pattern(self) -> None:
        """TaskRegistry uses singleton pattern."""
        registry1 = TaskRegistry()
        registry2 = TaskRegistry()

        assert registry1 is registry2
        assert registry1 is task_registry


class TestProgressUpdates:
    """Test progress update functionality."""

    @pytest.mark.asyncio
    async def test_update_progress_changes_status_to_running(self) -> None:
        """First progress update changes status from pending to running."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        await registry.update_progress(task_id, processed=10)

        task = registry.get_task(task_id)
        assert task is not None
        assert task.status == "running"
        assert task.processed_items == 10

    @pytest.mark.asyncio
    async def test_update_progress_incremental(self) -> None:
        """Progress can be updated multiple times."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        await registry.update_progress(task_id, processed=25)
        await registry.update_progress(task_id, processed=50)
        await registry.update_progress(task_id, processed=75)

        task = registry.get_task(task_id)
        assert task is not None
        assert task.processed_items == 75

    @pytest.mark.asyncio
    async def test_update_progress_with_current_item(self) -> None:
        """Current item is updated correctly."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        await registry.update_progress(
            task_id, processed=10, current_item="article-123"
        )

        task = registry.get_task(task_id)
        assert task is not None
        assert task.current_item == "article-123"

    @pytest.mark.asyncio
    async def test_update_progress_with_message(self) -> None:
        """Status message is updated correctly."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        await registry.update_progress(
            task_id, processed=10, message="Processing article 10 of 100"
        )

        task = registry.get_task(task_id)
        assert task is not None
        assert task.message == "Processing article 10 of 100"

    @pytest.mark.asyncio
    async def test_update_progress_nonexistent_task_raises_error(self) -> None:
        """Updating nonexistent task raises KeyError."""
        registry = TaskRegistry()

        with pytest.raises(KeyError, match="Task .* not found"):
            await registry.update_progress("nonexistent-id", processed=10)


class TestTaskCompletion:
    """Test task completion and failure states."""

    @pytest.mark.asyncio
    async def test_mark_complete(self) -> None:
        """Task can be marked as completed."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        await registry.mark_complete(task_id, failed_count=5)

        task = registry.get_task(task_id)
        assert task is not None
        assert task.status == "completed"
        assert task.failed_items == 5
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_mark_failed(self) -> None:
        """Task can be marked as failed."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        await registry.mark_failed(task_id, error="Database connection lost")

        task = registry.get_task(task_id)
        assert task is not None
        assert task.status == "failed"
        assert task.message == "Database connection lost"
        assert task.completed_at is not None


class TestErrorRecording:
    """Test error recording functionality."""

    def test_record_error(self) -> None:
        """Errors are recorded correctly."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        registry.record_error(task_id, item_id="article-123", error="Timeout")

        task = registry.get_task(task_id)
        assert task is not None
        assert task.failed_items == 1
        assert len(task.errors) == 1
        assert task.errors[0] == {"item_id": "article-123", "error": "Timeout"}

    def test_record_multiple_errors(self) -> None:
        """Multiple errors are accumulated."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        registry.record_error(task_id, item_id="article-1", error="Error 1")
        registry.record_error(task_id, item_id="article-2", error="Error 2")

        task = registry.get_task(task_id)
        assert task is not None
        assert task.failed_items == 2
        assert len(task.errors) == 2


class TestTaskCancellation:
    """Test task cancellation functionality."""

    def test_cancel_task(self) -> None:
        """Task cancellation flag is set."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        result = registry.cancel_task(task_id)

        assert result is True
        assert registry.is_cancelled(task_id) is True

    def test_cancel_nonexistent_task(self) -> None:
        """Cancelling nonexistent task returns False."""
        registry = TaskRegistry()

        result = registry.cancel_task("nonexistent-id")

        assert result is False

    def test_is_cancelled_nonexistent_task(self) -> None:
        """Checking nonexistent task returns False."""
        registry = TaskRegistry()

        result = registry.is_cancelled("nonexistent-id")

        assert result is False


class TestSSEStreaming:
    """Test SSE progress streaming functionality."""

    @pytest.mark.asyncio
    async def test_stream_progress_yields_initial_state(self) -> None:
        """Stream yields initial task state immediately."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        # Consume first event from stream
        stream = registry.stream_progress(task_id)
        initial_state = await anext(stream)

        assert initial_state.task_id == task_id
        assert initial_state.status == "pending"
        assert initial_state.processed_items == 0

    @pytest.mark.asyncio
    async def test_stream_progress_yields_updates(self) -> None:
        """Stream yields updates when progress changes."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        # Start streaming in background
        stream = registry.stream_progress(task_id)
        initial_state = await anext(stream)

        # Update progress
        await registry.update_progress(task_id, processed=50)

        # Should receive update
        update = await anext(stream)
        assert update.processed_items == 50
        assert update.status == "running"

    @pytest.mark.asyncio
    async def test_stream_progress_stops_on_completion(self) -> None:
        """Stream stops when task reaches terminal state."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        # Start streaming
        stream = registry.stream_progress(task_id)
        await anext(stream)  # Initial state

        # Mark as complete
        await registry.mark_complete(task_id)

        # Should receive completion event
        final_state = await anext(stream)
        assert final_state.status == "completed"

        # Stream should stop (StopAsyncIteration)
        with pytest.raises(StopAsyncIteration):
            await anext(stream)

    @pytest.mark.asyncio
    async def test_multiple_listeners(self) -> None:
        """Multiple SSE listeners receive same updates."""
        registry = TaskRegistry()
        task_id = registry.create_task("reindex", total_items=100)

        # Start two streams
        stream1 = registry.stream_progress(task_id)
        stream2 = registry.stream_progress(task_id)

        # Consume initial states
        await anext(stream1)
        await anext(stream2)

        # Update progress
        await registry.update_progress(task_id, processed=50)

        # Both listeners should receive update
        update1 = await anext(stream1)
        update2 = await anext(stream2)

        assert update1.processed_items == 50
        assert update2.processed_items == 50

    @pytest.mark.asyncio
    async def test_stream_nonexistent_task_raises_error(self) -> None:
        """Streaming nonexistent task raises KeyError."""
        registry = TaskRegistry()

        with pytest.raises(KeyError, match="Task .* not found"):
            stream = registry.stream_progress("nonexistent-id")
            await anext(stream)


class TestConcurrentTasks:
    """Test multiple concurrent tasks are isolated."""

    @pytest.mark.asyncio
    async def test_multiple_tasks_isolated(self) -> None:
        """Multiple tasks maintain separate state."""
        registry = TaskRegistry()

        task_id1 = registry.create_task("reindex", total_items=100)
        task_id2 = registry.create_task("export", total_items=50)

        await registry.update_progress(task_id1, processed=25)
        await registry.update_progress(task_id2, processed=10)

        task1 = registry.get_task(task_id1)
        task2 = registry.get_task(task_id2)

        assert task1 is not None
        assert task2 is not None
        assert task1.processed_items == 25
        assert task2.processed_items == 10
        assert task1.total_items == 100
        assert task2.total_items == 50

    @pytest.mark.asyncio
    async def test_concurrent_streams_isolated(self) -> None:
        """SSE streams for different tasks are isolated."""
        registry = TaskRegistry()

        task_id1 = registry.create_task("reindex", total_items=100)
        task_id2 = registry.create_task("export", total_items=50)

        stream1 = registry.stream_progress(task_id1)
        stream2 = registry.stream_progress(task_id2)

        # Consume initial states
        state1 = await anext(stream1)
        state2 = await anext(stream2)

        assert state1.task_id == task_id1
        assert state2.task_id == task_id2

        # Update task 1
        await registry.update_progress(task_id1, processed=25)

        # Only stream 1 should receive update
        update1 = await anext(stream1)
        assert update1.processed_items == 25

        # Stream 2 should not receive update (would block if no update)
        # Instead, we verify task 2 state hasn't changed
        task2 = registry.get_task(task_id2)
        assert task2 is not None
        assert task2.processed_items == 0


class TestProgressPercent:
    """Test progress percentage calculation."""

    def test_progress_percent_zero(self) -> None:
        """Progress is 0% when no items processed."""
        task = TaskProgress(
            task_id="test",
            task_type="reindex",
            status="pending",
            total_items=100,
            processed_items=0,
            started_at=datetime.now(UTC),
        )

        assert task.progress_percent() == 0

    def test_progress_percent_half(self) -> None:
        """Progress is 50% when half items processed."""
        task = TaskProgress(
            task_id="test",
            task_type="reindex",
            status="running",
            total_items=100,
            processed_items=50,
            started_at=datetime.now(UTC),
        )

        assert task.progress_percent() == 50

    def test_progress_percent_complete(self) -> None:
        """Progress is 100% when all items processed."""
        task = TaskProgress(
            task_id="test",
            task_type="reindex",
            status="completed",
            total_items=100,
            processed_items=100,
            started_at=datetime.now(UTC),
        )

        assert task.progress_percent() == 100

    def test_progress_percent_zero_total(self) -> None:
        """Progress is 0% when total items is 0 (edge case)."""
        task = TaskProgress(
            task_id="test",
            task_type="reindex",
            status="completed",
            total_items=0,
            processed_items=0,
            started_at=datetime.now(UTC),
        )

        assert task.progress_percent() == 0
