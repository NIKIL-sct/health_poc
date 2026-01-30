"""
Database Configuration and Session Management
Handles both sync and async SQLAlchemy sessions
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from contextlib import asynccontextmanager, contextmanager
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Database connection settings
# Change these according to your PostgreSQL configuration
DB_USER = "postgres"
DB_PASSWORD = "vigilxdev"
DB_HOST = "192.168.1.144"
DB_PORT = 5432
DB_NAME = "healthcheck_vigilx"


# Connection URLs
SYNC_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ASYNC_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Global engine instances
_sync_engine: Optional[create_engine] = None
_async_engine: Optional[create_async_engine] = None
_sync_session_factory: Optional[sessionmaker] = None
_async_session_factory: Optional[async_sessionmaker] = None


# ============================================
# SYNC ENGINE & SESSION (for workers)
# ============================================

def get_sync_engine():
    """Get or create sync engine"""
    global _sync_engine
    
    if _sync_engine is None:
        _sync_engine = create_engine(
            SYNC_DATABASE_URL,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before use
            echo=False  # Set to True for SQL debugging
        )
        logger.info("Sync database engine created")
    
    return _sync_engine


def get_sync_session_factory():
    """Get or create sync session factory"""
    global _sync_session_factory
    
    if _sync_session_factory is None:
        engine = get_sync_engine()
        _sync_session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    
    return _sync_session_factory


@contextmanager
def get_sync_db():
    """
    Context manager for sync database sessions
    Usage:
        with get_sync_db() as db:
            db.add(camera)
            db.commit()
    """
    factory = get_sync_session_factory()
    session: Session = factory()
    
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        session.close()


# ============================================
# ASYNC ENGINE & SESSION (for FastAPI)
# ============================================

def get_async_engine():
    """Get or create async engine"""
    global _async_engine
    
    if _async_engine is None:
        _async_engine = create_async_engine(
            ASYNC_DATABASE_URL,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False
        )
        logger.info("Async database engine created")
    
    return _async_engine


def get_async_session_factory():
    """Get or create async session factory"""
    global _async_session_factory
    
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
    
    return _async_session_factory


@asynccontextmanager
async def get_async_db():
    """
    Context manager for async database sessions
    Usage:
        async with get_async_db() as db:
            db.add(camera)
            await db.commit()
    """
    factory = get_async_session_factory()
    session: AsyncSession = factory()
    
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        await session.close()


# ============================================
# DEPENDENCY INJECTION (for FastAPI)
# ============================================

async def get_db_dependency():
    """
    FastAPI dependency for database sessions
    Usage in routes:
        @app.get("/")
        async def route(db: AsyncSession = Depends(get_db_dependency)):
            ...
    """
    factory = get_async_session_factory()
    session: AsyncSession = factory()
    
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database error: {e}", exc_info=True)
        raise
    finally:
        await session.close()


# ============================================
# CLEANUP
# ============================================

async def close_async_engine():
    """Cleanup async engine on shutdown"""
    global _async_engine
    
    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        logger.info("Async database engine disposed")


def close_sync_engine():
    """Cleanup sync engine"""
    global _sync_engine
    
    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
        logger.info("Sync database engine disposed")