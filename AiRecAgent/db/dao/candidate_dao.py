"""Candidate data access object."""

from typing import Any

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from AiRecAgent.db.dependencies import get_db_session
from AiRecAgent.db.models.candidate_model import CandidateModel


class CandidateDAO:
    """Provides database access for candidates."""

    def __init__(self, session: AsyncSession = Depends(get_db_session)) -> None:
        self.session = session

    async def create(
        self,
        raw_text: str,
        name: str | None = None,
        email: str | None = None,
        skills: list[str] | None = None,
        experience_years: float | None = None,
        education: str | None = None,
        embedding: list[float] | None = None,
        source_file: str | None = None,
    ) -> CandidateModel:
        """Create and persist a new candidate record."""
        candidate = CandidateModel(
            raw_text=raw_text,
            name=name,
            email=email,
            skills=skills,
            experience_years=experience_years,
            education=education,
            embedding=embedding,
            source_file=source_file,
        )
        self.session.add(candidate)
        await self.session.flush()
        await self.session.refresh(candidate)
        return candidate

    async def get_by_id(self, candidate_id: int) -> CandidateModel | None:
        """Return a candidate by primary key."""
        result = await self.session.execute(
            select(CandidateModel).where(CandidateModel.id == candidate_id),
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[CandidateModel]:
        """Return all candidates with pagination."""
        result = await self.session.execute(
            select(CandidateModel).limit(limit).offset(offset),
        )
        return list(result.scalars().fetchall())

    async def get_with_embeddings(self) -> list[CandidateModel]:
        """Return candidates that have a computed embedding."""
        result = await self.session.execute(
            select(CandidateModel).where(CandidateModel.embedding.isnot(None)),
        )
        return list(result.scalars().fetchall())

    async def update_embedding(
        self,
        candidate: CandidateModel,
        embedding: list[float],
    ) -> None:
        """Persist an embedding vector onto an existing candidate."""
        candidate.embedding = embedding  # type: ignore[assignment]
        self.session.add(candidate)

    async def update_parsed_fields(
        self,
        candidate: CandidateModel,
        parsed: dict[str, Any],
    ) -> None:
        """Apply LLM-extracted fields to an existing candidate record."""
        for field in ("name", "email", "skills", "experience_years", "education"):
            value = parsed.get(field)
            if value is not None:
                setattr(candidate, field, value)
        self.session.add(candidate)
