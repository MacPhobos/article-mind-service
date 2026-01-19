"""Test database connection."""

import asyncio
from sqlalchemy import text

from article_mind_service.database import engine


async def test_connection() -> None:
    """Test async database connection."""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Database connection successful!")
            print(f"PostgreSQL version: {version}")

            # Test basic query
            result = await conn.execute(text("SELECT 1 + 1 AS result"))
            value = result.scalar()
            print(f"Test query result: 1 + 1 = {value}")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(test_connection())
