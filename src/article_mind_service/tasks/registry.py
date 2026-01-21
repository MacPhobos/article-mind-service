"""Generic task progress tracking system with SSE streaming support.

Design Decisions:

1. In-Memory Storage:
   - TaskRegistry stores task state in memory (not database)
   - Rationale: Admin operations are infrequent, short-lived, and don't require persistence
   - Trade-off: State lost on server restart vs. simpler implementation
   - Acceptable for admin use case (operations complete within minutes)

2. Singleton Pattern:
   - Single global registry instance shared across all requests
   - Rationale: Simplifies access, ensures single source of truth for task state
   - Thread-safe due to Python's GIL and asyncio's single-threaded nature
   - Trade-off: Global state vs. dependency injection complexity

3. SSE Event Distribution:
   - Uses asyncio.Queue per task for broadcasting updates to multiple listeners
   - Rationale: Supports multiple concurrent SSE connections watching same task
   - Each listener gets its own queue to avoid blocking others
   - Listeners automatically cleaned up when connection closes

4. Task Cancellation:
   - Cooperative cancellation (worker must check is_cancelled flag)
   - Rationale: Cannot forcefully kill async tasks safely
   - Worker must periodically check and gracefully stop
   - Trade-off: Delayed cancellation vs. safe cleanup

5. Error Recording:
   - Errors stored as list of dicts (not exceptions)
   - Rationale: Serializable for SSE/JSON, preserves full error context
   - Limited to prevent memory bloat (consider max_errors limit if needed)
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskProgress(BaseModel):
    """Progress state for a background task.

    This schema represents the current state of a long-running background task
    and is used for both internal state management and SSE event payloads.

    Design Decisions:
    - Literal types for status enum (type-safe, auto-generates OpenAPI constraints)
    - Optional fields for task-specific data (current_item, message)
    - errors as list of dicts (serializable, preserves context)
    - Timestamps in UTC (avoids timezone ambiguity)
    """

    task_id: str = Field(description="Unique task identifier (UUID)")
    task_type: str = Field(
        description="Type of task (reindex, export, etc.)", examples=["reindex", "export"]
    )
    status: Literal["pending", "running", "completed", "failed", "cancelled"] = Field(
        description="Current task status"
    )
    total_items: int = Field(description="Total number of items to process", ge=0)
    processed_items: int = Field(default=0, description="Number of items processed so far", ge=0)
    failed_items: int = Field(default=0, description="Number of items that failed", ge=0)
    current_item: str | None = Field(
        default=None, description="Currently processing item (optional)"
    )
    message: str | None = Field(
        default=None, description="Human-readable status message (optional)"
    )
    started_at: datetime = Field(description="Task start time (UTC)")
    completed_at: datetime | None = Field(default=None, description="Task completion time (UTC)")
    errors: list[dict[str, str]] = Field(
        default_factory=list, description="List of errors encountered"
    )

    def progress_percent(self) -> int:
        """Calculate progress percentage (0-100)."""
        if self.total_items == 0:
            return 0
        return min(100, int((self.processed_items / self.total_items) * 100))


class TaskRegistry:
    """In-memory registry for tracking background tasks with SSE streaming.

    This singleton class manages the state of all active background tasks and
    provides SSE streaming capabilities for real-time progress updates.

    Thread Safety:
    - Safe for use in asyncio (single-threaded event loop)
    - All state modifications happen within event loop
    - No mutex needed due to Python GIL and asyncio guarantees

    Memory Management:
    - Old completed tasks should be cleaned up periodically (not implemented in MVP)
    - Consider adding max_tasks limit or TTL-based cleanup for production

    Usage:
        registry = TaskRegistry()  # Get singleton instance
        task_id = registry.create_task("reindex", total_items=100)
        await registry.update_progress(task_id, processed=50)
        async for progress in registry.stream_progress(task_id):
            # Send progress to SSE client
    """

    _instance: "TaskRegistry | None" = None
    _tasks: dict[str, TaskProgress]
    _queues: dict[str, list[asyncio.Queue[TaskProgress]]]
    _cancelled: set[str]

    def __new__(cls) -> "TaskRegistry":
        """Singleton pattern: return existing instance or create new one."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._tasks = {}
            instance._queues = {}
            instance._cancelled = set()
            cls._instance = instance
        return cls._instance

    def create_task(self, task_type: str, total_items: int) -> str:
        """Create new task and return task_id.

        Args:
            task_type: Type of task (e.g., "reindex", "export")
            total_items: Total number of items to process

        Returns:
            Unique task ID (UUID string)

        Example:
            >>> task_id = registry.create_task("reindex", total_items=100)
            >>> print(task_id)
            '550e8400-e29b-41d4-a716-446655440000'
        """
        task_id = str(uuid4())

        task = TaskProgress(
            task_id=task_id,
            task_type=task_type,
            status="pending",
            total_items=total_items,
            started_at=datetime.now(UTC),
        )

        self._tasks[task_id] = task
        self._queues[task_id] = []  # Initialize empty listener queue list

        logger.info(f"Created task {task_id} (type={task_type}, total_items={total_items})")
        return task_id

    async def update_progress(
        self,
        task_id: str,
        processed: int | None = None,
        current_item: str | None = None,
        message: str | None = None,
    ) -> None:
        """Update progress and notify SSE listeners.

        This method updates the task state and broadcasts the update to all
        active SSE listeners via their queues.

        Args:
            task_id: Task to update
            processed: New processed_items count (optional, incremental if not provided)
            current_item: Currently processing item description (optional)
            message: Human-readable status message (optional)

        Raises:
            KeyError: If task_id not found

        Example:
            >>> await registry.update_progress(
            ...     task_id,
            ...     processed=50,
            ...     current_item="article-123",
            ...     message="Processing article 50 of 100"
            ... )
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]

        # Update status to running if still pending
        if task.status == "pending":
            task.status = "running"

        # Update fields if provided
        if processed is not None:
            task.processed_items = processed
        if current_item is not None:
            task.current_item = current_item
        if message is not None:
            task.message = message

        # Broadcast update to all SSE listeners
        await self._broadcast(task_id, task)

    def record_error(self, task_id: str, item_id: str, error: str) -> None:
        """Record an error for an item.

        Errors are stored as serializable dicts to support SSE streaming.

        Args:
            task_id: Task that encountered error
            item_id: Item that failed (e.g., article ID)
            error: Error message

        Raises:
            KeyError: If task_id not found

        Example:
            >>> registry.record_error(
            ...     task_id,
            ...     item_id="article-123",
            ...     error="Embedding generation timeout"
            ... )
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task.failed_items += 1
        task.errors.append({"item_id": item_id, "error": error})

        logger.warning(f"Task {task_id} error on {item_id}: {error}")

    async def mark_complete(self, task_id: str, failed_count: int = 0) -> None:
        """Mark task as completed.

        Args:
            task_id: Task to mark complete
            failed_count: Number of items that failed (optional)

        Raises:
            KeyError: If task_id not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task.status = "completed"
        task.failed_items = failed_count
        task.completed_at = datetime.now(UTC)

        # Send final update to listeners
        await self._broadcast(task_id, task)

        logger.info(
            f"Task {task_id} completed: "
            f"{task.processed_items} processed, {task.failed_items} failed"
        )

    async def mark_failed(self, task_id: str, error: str) -> None:
        """Mark task as failed.

        Args:
            task_id: Task to mark failed
            error: Error message describing failure

        Raises:
            KeyError: If task_id not found
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        task = self._tasks[task_id]
        task.status = "failed"
        task.message = error
        task.completed_at = datetime.now(UTC)

        # Send final update to listeners
        await self._broadcast(task_id, task)

        logger.error(f"Task {task_id} failed: {error}")

    def cancel_task(self, task_id: str) -> bool:
        """Request cancellation (sets flag, worker must check).

        Cooperative cancellation: The worker must periodically check
        is_cancelled() and gracefully stop when True.

        Args:
            task_id: Task to cancel

        Returns:
            True if task exists and cancellation requested, False otherwise
        """
        if task_id not in self._tasks:
            return False

        self._cancelled.add(task_id)
        logger.info(f"Cancellation requested for task {task_id}")
        return True

    def is_cancelled(self, task_id: str) -> bool:
        """Check if task cancellation was requested.

        Workers should call this periodically and stop gracefully if True.

        Args:
            task_id: Task to check

        Returns:
            True if cancellation requested, False otherwise
        """
        return task_id in self._cancelled

    def get_task(self, task_id: str) -> TaskProgress | None:
        """Get current task state.

        Args:
            task_id: Task to retrieve

        Returns:
            TaskProgress model or None if not found
        """
        return self._tasks.get(task_id)

    async def stream_progress(self, task_id: str) -> AsyncGenerator[TaskProgress, None]:
        """Stream progress updates via async generator for SSE.

        This method creates a dedicated queue for this listener and yields
        progress updates as they arrive. The queue is automatically cleaned
        up when the generator is closed.

        Args:
            task_id: Task to stream

        Yields:
            TaskProgress updates as they occur

        Raises:
            KeyError: If task_id not found

        Example:
            >>> async for progress in registry.stream_progress(task_id):
            ...     yield progress.model_dump_json()  # Send via SSE
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        # Create dedicated queue for this listener
        queue: asyncio.Queue[TaskProgress] = asyncio.Queue()
        self._queues[task_id].append(queue)

        try:
            # Send initial state immediately
            current_state = self._tasks[task_id]
            yield current_state

            # Stream updates until task completes
            while True:
                # Wait for next update
                progress = await queue.get()
                yield progress

                # Stop streaming when task reaches terminal state
                if progress.status in ("completed", "failed", "cancelled"):
                    break

        finally:
            # Clean up listener queue
            if task_id in self._queues:
                try:
                    self._queues[task_id].remove(queue)
                except ValueError:
                    pass  # Queue already removed

    async def _broadcast(self, task_id: str, task: TaskProgress) -> None:
        """Broadcast task update to all SSE listeners.

        Internal method called by update_progress, mark_complete, mark_failed.

        Args:
            task_id: Task being updated
            task: Updated task state
        """
        if task_id not in self._queues:
            return

        # Put update in all listener queues
        for queue in self._queues[task_id]:
            try:
                # Use put_nowait to avoid blocking if queue is full
                # (listener may be slow or disconnected)
                queue.put_nowait(task.model_copy())
            except asyncio.QueueFull:
                logger.warning(f"Listener queue full for task {task_id}, dropping update")


# Singleton instance for global access
task_registry = TaskRegistry()
