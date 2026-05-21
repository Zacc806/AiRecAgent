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
    """Concatenate job fields into a single text for vectorisation.

    Structured fields (required_skills, tech_stack) are appended so that
    the NLP-parsed job structure is reflected in TF-IDF and semantic scores.
    """
    parts = [job.title, job.description]
    if job.requirements:
        parts.append(job.requirements)
    if job.required_skills:
        parts.append("Required skills: " + ", ".join(job.required_skills))  # type: ignore[arg-type]
    if job.tech_stack:
        parts.append("Tech stack: " + ", ".join(job.tech_stack))  # type: ignore[arg-type]
    return "\n".join(parts)


def _compute_semantic_scores(
    job: JobModel,
    job_text: str,
    candidates: list[CandidateModel],
    score_only_ids: set[int] | None,
) -> dict[int, float]:
    if job.embedding is None:
        logger.info("Computing job embedding for job_id={}", job.id)
        job.embedding = semantic.encode(job_text, settings.embedding_model)  # type: ignore[assignment]
    scores: dict[int, float] = {}
    for c in candidates:
        if score_only_ids is not None and c.id not in score_only_ids:
            continue
        if c.embedding is None:
            logger.debug("Candidate {} has no embedding, computing now", c.id)
            c.embedding = semantic.encode(c.raw_text, settings.embedding_model)  # type: ignore[assignment]
        scores[c.id] = semantic.score_candidate(
            list(c.embedding),  # type: ignore[arg-type]
            list(job.embedding),  # type: ignore[arg-type]
        )
    return scores


async def _compute_llm_scores(
    candidates: list[CandidateModel],
    job_text: str,
    score_only_ids: set[int] | None,
) -> tuple[dict[int, float], dict[int, str]]:
    scores: dict[int, float] = {}
    explanations: dict[int, str] = {}
    if not settings.anthropic_api_key:
        logger.info("No ANTHROPIC_API_KEY set, skipping LLM scoring")
        return scores, explanations
    for c in candidates:
        if score_only_ids is not None and c.id not in score_only_ids:
            continue
        score, explanation = await llm_service.score_candidate(
            candidate_text=c.raw_text,
            job_text=job_text,
            api_key=settings.anthropic_api_key,
        )
        scores[c.id] = score
        explanations[c.id] = explanation
    return scores, explanations


async def get_recommendations(
    job: JobModel,
    candidates: list[CandidateModel],
    session: AsyncSession,
    top_k: int = 5,
    score_only_ids: set[int] | None = None,
) -> list[CandidateScore]:
    """Score candidates against the job and return results.

    When score_only_ids is provided, all candidates are used for the TF-IDF
    corpus (preserving IDF quality) but LLM and semantic scoring only run for
    the specified subset. Results contain only those candidates.

    Approach 1 — Semantic (Sentence-BERT cosine similarity)
    Approach 2 — TF-IDF cosine similarity
    Approach 3 — LLM (Claude) relevance scoring
    """
    if not candidates:
        return []

    job_text = _build_job_text(job)

    sem_scores = _compute_semantic_scores(job, job_text, candidates, score_only_ids)

    candidate_texts = [c.raw_text for c in candidates]
    tfidf_scores_raw = tfidf.score_candidates(candidate_texts, job_text)
    tfi_scores: dict[int, float] = {
        c.id: float(tfidf_scores_raw[i])
        for i, c in enumerate(candidates)
        if score_only_ids is None or c.id in score_only_ids
    }

    llm_scores, explanations = await _compute_llm_scores(
        candidates, job_text, score_only_ids
    )

    results: list[CandidateScore] = []
    for c in candidates:
        if score_only_ids is not None and c.id not in score_only_ids:
            continue
        sem = sem_scores.get(c.id)
        tfi = tfi_scores.get(c.id)
        llm = llm_scores.get(c.id)
        overall = _weighted_score(sem, tfi, llm)
        results.append(
            CandidateScore(
                candidate=c,
                semantic_score=sem,
                tfidf_score=tfi,
                llm_score=llm,
                overall_score=overall,
                explanation=explanations.get(c.id, ""),
            )
        )

    results.sort(key=lambda r: r.overall_score, reverse=True)
    return results[:top_k]
