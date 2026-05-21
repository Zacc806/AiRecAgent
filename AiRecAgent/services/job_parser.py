"""Job description parsing: structured extraction of skills, level, and tech stack."""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from AiRecAgent.services.nlp_pipeline import run_pipeline

# Common experience-level signals
_LEVEL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Lead", re.compile(r"\b(lead|principal|staff|head of)\b", re.IGNORECASE)),
    ("Senior", re.compile(r"\b(senior|sr\.?|старший)\b", re.IGNORECASE)),
    ("Junior", re.compile(r"\b(junior|jr\.?|младший|intern|стажёр)\b", re.IGNORECASE)),
    ("Mid", re.compile(r"\b(mid[- ]?level|middle|middle-level)\b", re.IGNORECASE)),
]

# Regex for "N+ years" / "N лет опыта" patterns
_EXP_YEARS_RE = re.compile(
    r"(\d+)\+?\s*(?:years?|лет|год(?:а)?)\s*(?:of\s+)?(?:experience|опыт)",  # noqa: RUF001
    re.IGNORECASE,
)


def _infer_experience_level(text: str) -> str | None:
    """Return the most specific experience level found in text, or None."""
    for level, pattern in _LEVEL_PATTERNS:
        if pattern.search(text):
            return level
    # Fall back to year-count heuristic
    match = _EXP_YEARS_RE.search(text)
    if match:
        years = int(match.group(1))
        if years >= 7:
            return "Senior"
        if years >= 3:
            return "Mid"
        return "Junior"
    return None


def _parse_job_regex(
    title: str,
    description: str,
    requirements: str | None,
) -> dict[str, Any]:
    """Extract structured fields via regex + NLP pipeline (no LLM required)."""
    full_text = "\n".join(filter(None, [title, description, requirements]))

    # Run the NLP pipeline for keywords and entity context
    nlp_result = run_pipeline(full_text)

    experience_level = _infer_experience_level(full_text)

    return {
        "required_skills": [],
        "experience_level": experience_level,
        "tech_stack": [],
        "nlp_keywords": nlp_result.keywords,
    }


_JOB_PARSE_PROMPT = """\
You are a technical recruiter parsing a job posting. Extract structured information.
Return ONLY a valid JSON object with these fields:
- required_skills: list of must-have technical skills / competencies (array of strings, in English)
- experience_level: one of "Junior", "Mid", "Senior", "Lead", or null
- tech_stack: list of specific technologies, frameworks, tools mentioned (array of strings)

Job title: {title}

Job description:
{description}

Requirements section:
{requirements}

JSON output:"""


def _parse_job_llm(
    title: str,
    description: str,
    requirements: str | None,
    api_key: str,
) -> dict[str, Any]:
    """Use Claude to extract structured fields from a job posting."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _JOB_PARSE_PROMPT.format(
        title=title,
        description=description[:2000],
        requirements=(requirements or "")[:1000],
    )
    logger.debug("job_parser: calling Claude for job_id structuring")
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise ValueError(f"Unexpected block type: {type(block)}")
    raw = block.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed: dict[str, Any] = json.loads(raw)
        logger.info(
            "job_parser LLM: required_skills={} level={!r} tech_stack={}",
            len(parsed.get("required_skills") or []),
            parsed.get("experience_level"),
            len(parsed.get("tech_stack") or []),
        )
        return parsed
    except json.JSONDecodeError as exc:
        logger.warning("job_parser LLM returned non-JSON ({}), using regex", exc)
        raise


def parse_job(
    title: str,
    description: str,
    requirements: str | None = None,
    api_key: str = "",
) -> dict[str, Any]:
    """Return structured job dict from posting fields.

    Always runs the NLP pipeline (for nlp_keywords).  Uses the Claude API for
    required_skills / experience_level / tech_stack when an api_key is provided,
    falling back to regex heuristics otherwise.
    """
    full_text = "\n".join(filter(None, [title, description, requirements]))
    nlp_result = run_pipeline(full_text)

    structured: dict[str, Any] = {
        "required_skills": [],
        "experience_level": None,
        "tech_stack": [],
        "nlp_keywords": nlp_result.keywords,
    }

    if api_key:
        try:
            llm_data = _parse_job_llm(title, description, requirements, api_key)
            structured["required_skills"] = llm_data.get("required_skills") or []
            structured["experience_level"] = llm_data.get("experience_level")
            structured["tech_stack"] = llm_data.get("tech_stack") or []
            return structured
        except Exception as exc:
            logger.warning(
                "job_parser: LLM failed ({}: {}), falling back to regex",
                type(exc).__name__,
                exc,
            )

    # Regex fallback: infer experience level, use NLP keywords as proxy for skills
    structured["experience_level"] = _infer_experience_level(full_text)
    structured["required_skills"] = nlp_result.keywords[:10]
    structured["tech_stack"] = []
    logger.info(
        "job_parser regex: level={!r} keywords_as_skills={}",
        structured["experience_level"],
        len(structured["required_skills"]),
    )
    return structured
