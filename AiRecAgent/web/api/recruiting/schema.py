"""Pydantic schemas for the recruiting API."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobCreateDTO(BaseModel):
    """Request body for creating a job posting."""

    title: str
    description: str
    requirements: str | None = None


class JobDTO(BaseModel):
    """Response schema for a job posting."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    description: str
    requirements: str | None
    required_skills: list[str] | None
    experience_level: str | None
    tech_stack: list[str] | None
    nlp_keywords: list[str] | None
    created_at: datetime


class CandidateDTO(BaseModel):
    """Response schema for a candidate profile."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str | None
    name: str | None
    skills: list[str] | None
    experience_years: float | None
    education: str | None
    source_file: str | None
    created_at: datetime


class RecommendationDTO(BaseModel):
    """Single candidate recommendation with match scores."""

    candidate: CandidateDTO
    semantic_score: float | None
    tfidf_score: float | None
    llm_score: float | None
    overall_score: float
    explanation: str


class RecommendationListDTO(BaseModel):
    """Ranked list of candidate recommendations for a job."""

    job_id: int | None
    recommendations: list[RecommendationDTO]


class EmailPollResultDTO(BaseModel):
    """Result of an IMAP email polling operation."""

    fetched: int
    imported: int


class EmailStatsDTO(BaseModel):
    """Cumulative email polling statistics since server start."""

    last_poll_at: datetime | None
    emails_checked: int
    attachments_found: int
