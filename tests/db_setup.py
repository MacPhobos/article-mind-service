"""Test database setup using Alembic migrations."""

import os
from pathlib import Path

from alembic import command
from alembic.config import Config

from .db_safety import validate_test_database_url


def get_test_database_url() -> str:
    """Get test database URL from environment."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        raise ValueError(
            "TEST_DATABASE_URL environment variable is required for tests. "
            "Set it in .env.test or export it directly."
        )
    return url


def run_alembic_upgrade(database_url: str) -> None:
    """Run Alembic migrations against the test database."""
    validate_test_database_url(database_url)

    # Override DATABASE_URL environment variable so alembic/env.py uses it
    original_db_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url

    try:
        project_root = Path(__file__).parent.parent
        alembic_ini = project_root / "alembic.ini"

        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", database_url)

        command.upgrade(config, "head")
    finally:
        # Restore original DATABASE_URL
        if original_db_url is not None:
            os.environ["DATABASE_URL"] = original_db_url
        else:
            os.environ.pop("DATABASE_URL", None)
