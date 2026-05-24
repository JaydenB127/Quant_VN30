# -*- coding: utf-8 -*-
"""
Database session management using SQLAlchemy 2.0 Async features.
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

logger = logging.getLogger(__name__)

# Default SQLite fallback URL for development.
# In production / Docker Compose, this will be overridden by env variable (PostgreSQL URL).
DEFAULT_DB_URL = "sqlite+aiosqlite:///ets.db"


class DatabaseManager:
    """Manages connection engines and async session pools for SQLAlchemy."""

    def __init__(self, database_url: str | None = None):
        import os
        self.database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DB_URL)
        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            pool_pre_ping=True,
        )
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Dependency helper returning async session generator."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create_tables(self) -> None:
        """Create declarative tables for dev/testing."""
        from db.models import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialized successfully.")
