from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from .settings import settings
import logging

logger = logging.getLogger(__name__)

# Async engine for async operations
async_engine = create_async_engine(
    settings.async_database_url, echo=False, pool_pre_ping=True, pool_recycle=300
)

# Sync engine for migrations and sync operations
sync_engine = create_engine(
    settings.database_url, echo=False, pool_pre_ping=True, pool_recycle=300
)

# Async session factory
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Sync session factory
SyncSessionLocal = sessionmaker(sync_engine, expire_on_commit=False)


async def get_async_session() -> AsyncSession:
    """Get async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_session():
    """Get sync database session"""
    with SyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            session.rollback()
            raise
        finally:
            session.close()


async def init_database():
    """Initialize database tables"""
    from ..infrastructure.database.models.document_model import Base

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def close_database():
    """Close database connections"""
    await async_engine.dispose()
    logger.info("Database connections closed")
