"""Recruiting API endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from AiRecAgent.db.dao.candidate_dao import CandidateDAO
from AiRecAgent.db.dao.job_dao import JobDAO
from AiRecAgent.db.dao.match_dao import MatchDAO
from AiRecAgent.db.dependencies import get_db_session
from AiRecAgent.services import email_service, resume_parser
from AiRecAgent.services.matching import orchestrator
from AiRecAgent.settings import settings
from AiRecAgent.web.api.recruiting.schema import (
    CandidateDTO,
    EmailPollResultDTO,
    JobCreateDTO,
    JobDTO,
    RecommendationDTO,
    RecommendationListDTO,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@router.post("/jobs", response_model=JobDTO, status_code=201)
async def create_job(
    body: JobCreateDTO,
    job_dao: JobDAO = Depends(),
) -> JobDTO:
    """Create a new job posting."""
    job = await job_dao.create(
        title=body.title,
        description=body.description,
        requirements=body.requirements,
    )
    return JobDTO.model_validate(job)


@router.get("/jobs", response_model=list[JobDTO])
async def list_jobs(
    limit: int = 50,
    offset: int = 0,
    job_dao: JobDAO = Depends(),
) -> list[JobDTO]:
    """Return all job postings."""
    jobs = await job_dao.get_all(limit=limit, offset=offset)
    return [JobDTO.model_validate(j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobDTO)
async def get_job(
    job_id: int,
    job_dao: JobDAO = Depends(),
) -> JobDTO:
    """Return a single job posting by ID."""
    job = await job_dao.get_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobDTO.model_validate(job)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


@router.post("/resumes/upload", response_model=CandidateDTO, status_code=201)
async def upload_resume(
    file: UploadFile = File(..., description="Resume file (PDF, DOCX, or TXT)"),
    candidate_dao: CandidateDAO = Depends(),
) -> CandidateDTO:
    """Accept a resume file upload (PDF, DOCX, or TXT) and ingest it."""
    data = await file.read()
    filename = file.filename or "resume.txt"
    raw_text = resume_parser.extract_text(data, filename)

    parsed = resume_parser.parse_resume(raw_text, api_key=settings.anthropic_api_key)
    logger.info("Parsed resume from {}: name={}", filename, parsed.get("name"))

    candidate = await candidate_dao.create(
        raw_text=raw_text,
        name=parsed.get("name"),
        email=parsed.get("email"),
        skills=parsed.get("skills") or [],
        experience_years=parsed.get("experience_years"),
        education=parsed.get("education"),
        source_file=filename,
    )
    return CandidateDTO.model_validate(candidate)


@router.get("/candidates", response_model=list[CandidateDTO])
async def list_candidates(
    limit: int = 50,
    offset: int = 0,
    candidate_dao: CandidateDAO = Depends(),
) -> list[CandidateDTO]:
    """Return all ingested candidates."""
    candidates = await candidate_dao.get_all(limit=limit, offset=offset)
    return [CandidateDTO.model_validate(c) for c in candidates]


@router.get("/candidates/{candidate_id}", response_model=CandidateDTO)
async def get_candidate(
    candidate_id: int,
    candidate_dao: CandidateDAO = Depends(),
) -> CandidateDTO:
    """Return a single candidate by ID."""
    candidate = await candidate_dao.get_by_id(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return CandidateDTO.model_validate(candidate)


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@router.get("/recommendations", response_model=RecommendationListDTO)
async def get_recommendations(
    job_id: int,
    limit: int = 5,
    session: AsyncSession = Depends(get_db_session),
    job_dao: JobDAO = Depends(),
    candidate_dao: CandidateDAO = Depends(),
    match_dao: MatchDAO = Depends(),
) -> RecommendationListDTO:
    """Return the top-N most relevant candidates for a given job.

    Scores are computed via three approaches:
    1. Semantic (Sentence-BERT cosine similarity)
    2. TF-IDF cosine similarity
    3. LLM (Claude) relevance scoring
    """
    job = await job_dao.get_by_id(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    candidates = await candidate_dao.get_all(limit=1000)
    if not candidates:
        return RecommendationListDTO(job_id=job_id, recommendations=[])

    scored = await orchestrator.get_recommendations(
        job=job,
        candidates=candidates,
        session=session,
        top_k=limit,
    )

    # Persist scores for caching
    for s in scored:
        await match_dao.upsert(
            candidate_id=s.candidate.id,
            job_id=job_id,
            semantic_score=s.semantic_score,
            tfidf_score=s.tfidf_score,
            llm_score=s.llm_score,
            overall_score=s.overall_score,
            explanation=s.explanation,
        )

    recs = [
        RecommendationDTO(
            candidate=CandidateDTO.model_validate(s.candidate),
            semantic_score=s.semantic_score,
            tfidf_score=s.tfidf_score,
            llm_score=s.llm_score,
            overall_score=s.overall_score,
            explanation=s.explanation,
        )
        for s in scored
    ]
    return RecommendationListDTO(job_id=job_id, recommendations=recs)


# ---------------------------------------------------------------------------
# Email polling
# ---------------------------------------------------------------------------


@router.post("/email/poll", response_model=EmailPollResultDTO)
async def poll_email(
    candidate_dao: CandidateDAO = Depends(),
) -> EmailPollResultDTO:
    """Trigger an IMAP poll to fetch new resume attachments from the inbox."""
    attachments = email_service.fetch_resume_attachments(settings)
    imported = 0
    for filename, data in attachments:
        try:
            raw_text = resume_parser.extract_text(data, filename)
            parsed = resume_parser.parse_resume(
                raw_text, api_key=settings.anthropic_api_key
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
            logger.error("Failed to import attachment {}: {}", filename, exc)

    return EmailPollResultDTO(fetched=len(attachments), imported=imported)
