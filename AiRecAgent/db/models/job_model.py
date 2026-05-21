"""Job posting database model."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from AiRecAgent.db.base import Base


class JobModel(Base):
    """Stores a job posting with its requirements and embedding."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text)
    requirements: Mapped[str | None] = mapped_column(Text, nullable=True)
    required_skills: Mapped[Any] = mapped_column(JSON, nullable=True)
    experience_level: Mapped[str | None] = mapped_column(String(50), nullable=True)
    tech_stack: Mapped[Any] = mapped_column(JSON, nullable=True)
    nlp_keywords: Mapped[Any] = mapped_column(JSON, nullable=True)
    embedding: Mapped[Any] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
