"""Job posting data access object."""

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from AiRecAgent.db.dependencies import get_db_session
from AiRecAgent.db.models.job_model import JobModel


class JobDAO:
    """Provides database access for job postings."""

    def __init__(self, session: AsyncSession = Depends(get_db_session)) -> None:
        self.session = session

    async def create(
        self,
        title: str,
        description: str,
        requirements: str | None = None,
    ) -> JobModel:
        """Create and persist a new job posting."""
        job = JobModel(title=title, description=description, requirements=requirements)
        self.session.add(job)
        await self.session.flush()
        await self.session.refresh(job)
        return job

    async def get_by_id(self, job_id: int) -> JobModel | None:
        """Return a job posting by primary key."""
        result = await self.session.execute(
            select(JobModel).where(JobModel.id == job_id),
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[JobModel]:
        """Return all job postings with pagination."""
        result = await self.session.execute(
            select(JobModel).limit(limit).offset(offset),
        )
        return list(result.scalars().fetchall())

    async def update_embedding(
        self,
        job: JobModel,
        embedding: list[float],
    ) -> None:
        """Persist an embedding vector onto an existing job record."""
        job.embedding = embedding  # type: ignore[assignment]
        self.session.add(job)
