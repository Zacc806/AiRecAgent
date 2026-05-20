"""Match score data access object."""

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from AiRecAgent.db.dependencies import get_db_session
from AiRecAgent.db.models.candidate_model import CandidateModel
from AiRecAgent.db.models.match_model import MatchModel


class MatchDAO:
    """Provides database access for candidate-job match scores."""

    def __init__(self, session: AsyncSession = Depends(get_db_session)) -> None:
        self.session = session

    async def upsert(
        self,
        candidate_id: int,
        job_id: int,
        semantic_score: float | None = None,
        tfidf_score: float | None = None,
        llm_score: float | None = None,
        overall_score: float | None = None,
        explanation: str | None = None,
    ) -> MatchModel:
        """Insert or update a match record for a candidate-job pair."""
        result = await self.session.execute(
            select(MatchModel).where(
                MatchModel.candidate_id == candidate_id,
                MatchModel.job_id == job_id,
            ),
        )
        match = result.scalar_one_or_none()
        if match is None:
            match = MatchModel(candidate_id=candidate_id, job_id=job_id)

        match.semantic_score = semantic_score  # type: ignore[assignment]
        match.tfidf_score = tfidf_score  # type: ignore[assignment]
        match.llm_score = llm_score  # type: ignore[assignment]
        match.overall_score = overall_score  # type: ignore[assignment]
        match.explanation = explanation  # type: ignore[assignment]

        self.session.add(match)
        await self.session.flush()
        await self.session.refresh(match)
        return match

    async def get_by_job(self, job_id: int) -> list[MatchModel]:
        """Return all match records for a given job, sorted by overall score."""
        result = await self.session.execute(
            select(MatchModel)
            .where(MatchModel.job_id == job_id)
            .order_by(MatchModel.overall_score.desc()),
        )
        return list(result.scalars().fetchall())

    async def get_by_job_with_candidates(
        self,
        job_id: int,
    ) -> list[tuple[MatchModel, CandidateModel]]:
        """Return match records for a job joined with their candidate, sorted by score."""
        result = await self.session.execute(
            select(MatchModel, CandidateModel)
            .join(CandidateModel, MatchModel.candidate_id == CandidateModel.id)
            .where(MatchModel.job_id == job_id)
            .order_by(MatchModel.overall_score.desc()),
        )
        return list(result.tuples().fetchall())
