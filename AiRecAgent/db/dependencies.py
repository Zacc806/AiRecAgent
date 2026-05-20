from collections.abc import AsyncGenerator

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.requests import Request


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    Create and get database session.

    :param request: current request.
    :yield: database session.
    """
    session: AsyncSession = request.app.state.db_session_factory()
    logger.debug("db_session: opened for {} {}", request.method, request.url.path)

    try:
        yield session
    except Exception:
        logger.exception("db_session: exception during request, rolling back")
        await session.rollback()
        raise
    finally:
        logger.debug(
            "db_session: committing and closing for {} {}",
            request.method,
            request.url.path,
        )
        await session.commit()
        await session.close()
