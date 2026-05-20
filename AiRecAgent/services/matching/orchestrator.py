"""Orchestrates all three matching approaches and returns ranked recommendations."""

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from AiRecAgent.db.models.candidate_model import CandidateModel
from AiRecAgent.db.models.job_model import JobModel
from AiRecAgent.services.matching import llm as llm_service
from AiRecAgent.services.matching import semantic, tfidf
from AiRecAgent.settings import settings

# Weights for the combined score
_W_SEMANTIC = 0.40
_W_TFIDF = 0.30
_W_LLM = 0.30


@dataclass
class CandidateScore:
    """Aggregated match result for a single candidate."""

    candidate: CandidateModel
    semantic_score: float | None
    tfidf_score: float | None
    llm_score: float | None
    overall_score: float
    explanation: str


def _weighted_score(
    sem: float | None,
    tfi: float | None,
    llm: float | None,
) -> float:
    """Compute weighted average of available scores."""
    total_weight = 0.0
    total_score = 0.0
    if sem is not None:
        total_score += _W_SEMANTIC * sem
        total_weight += _W_SEMANTIC
    if tfi is not None:
        total_score += _W_TFIDF * tfi
        total_weight += _W_TFIDF
    if llm is not None:
        total_score += _W_LLM * llm
        total_weight += _W_LLM
    if total_weight == 0:
        return 0.0
    return total_score / total_weight


def _build_job_text(job: JobModel) -> str:
    """Concatenate job fields into a single text for vectorisation."""
    parts = [job.title, job.description]
    if job.requirements:
        parts.append(job.requirements)
    return "\n".join(parts)


async def get_recommendations(
    job: JobModel,
    candidates: list[CandidateModel],
    session: AsyncSession,
    top_k: int = 5,
) -> list[CandidateScore]:
    """Score all candidates against the job and return the top-k results.

    Approach 1 — Semantic (Sentence-BERT cosine similarity)
    Approach 2 — TF-IDF cosine similarity
    Approach 3 — LLM (Claude) relevance scoring
    """
    if not candidates:
        return []

    job_text = _build_job_text(job)

    # --- Approach 1: semantic embeddings ---
    if job.embedding is None:
        logger.info("Computing job embedding for job_id={}", job.id)
        job.embedding = semantic.encode(job_text, settings.embedding_model)  # type: ignore[assignment]

    sem_scores: list[float | None] = []
    for c in candidates:
        if c.embedding is None:
            logger.debug("Candidate {} has no embedding, computing now", c.id)
            c.embedding = semantic.encode(c.raw_text, settings.embedding_model)  # type: ignore[assignment]
        sem_scores.append(
            semantic.score_candidate(
                list(c.embedding),  # type: ignore[arg-type]
                list(job.embedding),  # type: ignore[arg-type]
            )
        )

    # --- Approach 2: TF-IDF ---
    candidate_texts = [c.raw_text for c in candidates]
    tfidf_scores_raw = tfidf.score_candidates(candidate_texts, job_text)
    tfi_scores: list[float | None] = list(tfidf_scores_raw)

    # --- Approach 3: LLM ---
    llm_scores: list[float | None] = [None] * len(candidates)
    explanations: list[str] = [""] * len(candidates)

    if settings.anthropic_api_key:
        for i, c in enumerate(candidates):
            score, explanation = await llm_service.score_candidate(
                candidate_text=c.raw_text,
                job_text=job_text,
                api_key=settings.anthropic_api_key,
            )
            llm_scores[i] = score
            explanations[i] = explanation
    else:
        logger.info("No ANTHROPIC_API_KEY set, skipping LLM scoring")

    # --- Combine ---
    results: list[CandidateScore] = []
    for i, c in enumerate(candidates):
        sem = sem_scores[i] if i < len(sem_scores) else None
        tfi = tfi_scores[i] if i < len(tfi_scores) else None
        llm = llm_scores[i]
        overall = _weighted_score(sem, tfi, llm)
        results.append(
            CandidateScore(
                candidate=c,
                semantic_score=sem,
                tfidf_score=tfi,
                llm_score=llm,
                overall_score=overall,
                explanation=explanations[i],
            )
        )

    results.sort(key=lambda r: r.overall_score, reverse=True)
    return results[:top_k]
