import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from AiRecAgent.db.meta import meta
from AiRecAgent.db.models import load_all_models
from AiRecAgent.services import email_service
from AiRecAgent.services.matching.semantic import _load_model
from AiRecAgent.settings import settings

_IMAP_POLL_INTERVAL_SECONDS = 30


def _setup_db(app: FastAPI) -> None:  # pragma: no cover
    """
    Creates connection to the database.

    This function creates SQLAlchemy engine instance,
    session_factory for creating sessions
    and stores them in the application's state property.

    :param app: fastAPI application.
    """
    engine = create_async_engine(str(settings.db_url), echo=settings.db_echo)
    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )
    app.state.db_engine = engine
    app.state.db_session_factory = session_factory


async def _create_tables() -> None:  # pragma: no cover
    """Populates tables in the database."""
    load_all_models()
    engine = create_async_engine(str(settings.db_url))
    async with engine.begin() as connection:
        await connection.run_sync(meta.create_all)
    await engine.dispose()


async def _import_attachments(
    app: FastAPI, attachments: list[tuple[str, bytes]]
) -> int:
    """Persist a list of (filename, bytes) resume attachments. Returns import count."""
    from AiRecAgent.db.dao.candidate_dao import CandidateDAO
    from AiRecAgent.services import resume_parser

    imported = 0
    session = app.state.db_session_factory()
    try:
        candidate_dao = CandidateDAO(session=session)
        for filename, data in attachments:
            try:
                raw_text = resume_parser.extract_text(data, filename)
                logger.info(
                    "email_import: extracted {} chars from {!r}",
                    len(raw_text),
                    filename,
                )
                if not raw_text.strip():
                    logger.warning(
                        "email_import: empty text for {!r} — skipping",
                        filename,
                    )
                    continue

                parsed = resume_parser.parse_resume(
                    raw_text, api_key=settings.anthropic_api_key
                )
                logger.info(
                    "email_import: parsed {!r} → name={!r} email={!r} skills={} exp={}",
                    filename,
                    parsed.get("name"),
                    parsed.get("email"),
                    len(parsed.get("skills") or []),
                    parsed.get("experience_years"),
                )

                await candidate_dao.create(
                    raw_text=raw_text,
                    name=parsed.get("name"),
                    email=parsed.get("email"),
                    skills=parsed.get("skills") or [],
                    experience_years=parsed.get("experience_years"),
                    education=parsed.get("education"),
                    source_file=filename,
                )
                imported += 1
            except Exception as exc:
                logger.error(
                    "email_import: failed to import {!r}: {} — {}",
                    filename,
                    type(exc).__name__,
                    exc,
                )

        await session.commit()
    except Exception as exc:
        logger.error("email_import: session error: {} — {}", type(exc).__name__, exc)
        await session.rollback()
    finally:
        await session.close()

    return imported


async def _email_poll_worker(app: FastAPI) -> None:  # pragma: no cover
    """Background task: poll IMAP every 30 s and import any new resume emails."""
    logger.info(
        "email_poll_worker: started, polling every {} seconds",
        _IMAP_POLL_INTERVAL_SECONDS,
    )
    while True:
        await asyncio.sleep(_IMAP_POLL_INTERVAL_SECONDS)
        logger.info("email_poll_worker: polling for new emails")

        attachments = email_service.fetch_resume_attachments(settings)
        if not attachments:
            logger.info("email_poll_worker: no new emails")
            continue

        imported = await _import_attachments(app, attachments)
        logger.info(
            "email_poll_worker: cycle done — fetched={} imported={}",
            len(attachments),
            imported,
        )


@asynccontextmanager
async def lifespan_setup(
    app: FastAPI,
) -> AsyncGenerator[None, None]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """

    app.middleware_stack = None
    _setup_db(app)
    await _create_tables()
    app.middleware_stack = app.build_middleware_stack()

    # Pre-warm the embedding model so the first recommendations request
    # doesn't block waiting for a cold model load.
    logger.info("lifespan: loading embedding model…")
    await asyncio.get_event_loop().run_in_executor(
        None, _load_model, settings.embedding_model
    )
    logger.info("lifespan: embedding model ready")

    # Record UIDNEXT so existing inbox emails are never imported.
    email_service.initialize_watermark(settings)

    # Start background IMAP polling loop.
    poll_task = asyncio.create_task(_email_poll_worker(app))

    yield

    poll_task.cancel()
    with suppress(asyncio.CancelledError):
        await poll_task
    await app.state.db_engine.dispose()
