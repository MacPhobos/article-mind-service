"""Unit tests for ResearchSession model."""

from datetime import UTC

import pytest

from article_mind_service.models.session import ResearchSession


class TestResearchSessionModel:
    """Tests for ResearchSession model methods."""

    def test_repr(self) -> None:
        """Test string representation."""
        session = ResearchSession(id=1, name="Test", status="draft")
        repr_str = repr(session)

        assert "ResearchSession" in repr_str
        assert "id=1" in repr_str
        assert "name='Test'" in repr_str
        assert "status='draft'" in repr_str

    def test_is_deleted_false(self) -> None:
        """Test is_deleted returns False when not deleted."""
        session = ResearchSession(id=1, name="Test", status="draft", deleted_at=None)

        assert session.is_deleted is False

    def test_is_deleted_true(self) -> None:
        """Test is_deleted returns True when deleted."""
        from datetime import datetime

        session = ResearchSession(
            id=1,
            name="Test",
            status="draft",
            deleted_at=datetime.now(UTC),
        )

        assert session.is_deleted is True


class TestStatusTransitions:
    """Tests for status transition validation."""

    @pytest.mark.parametrize(
        "current_status,new_status,expected",
        [
            # Valid transitions from draft
            ("draft", "active", True),
            ("draft", "archived", True),
            ("draft", "completed", False),  # Invalid: must go through active
            ("draft", "draft", False),  # Same status
            # Valid transitions from active
            ("active", "completed", True),
            ("active", "archived", True),
            ("active", "draft", False),  # Can't go back
            ("active", "active", False),  # Same status
            # Valid transitions from completed
            ("completed", "archived", True),
            ("completed", "draft", False),  # Can't go back
            ("completed", "active", False),  # Can't go back
            ("completed", "completed", False),  # Same status
            # No transitions from archived
            ("archived", "draft", False),
            ("archived", "active", False),
            ("archived", "completed", False),
            ("archived", "archived", False),
        ],
    )
    def test_can_transition_to(self, current_status: str, new_status: str, expected: bool) -> None:
        """Test status transition validation."""
        session = ResearchSession(id=1, name="Test", status=current_status)

        assert session.can_transition_to(new_status) == expected
