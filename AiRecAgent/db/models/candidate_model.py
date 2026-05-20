"""Candidate database model."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from AiRecAgent.db.base import Base


class CandidateModel(Base):
    """Stores a candidate's resume and parsed profile data."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text)
    skills: Mapped[Any] = mapped_column(JSON, nullable=True)
    experience_years: Mapped[float | None] = mapped_column(Float, nullable=True)
    education: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding: Mapped[Any] = mapped_column(JSON, nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
