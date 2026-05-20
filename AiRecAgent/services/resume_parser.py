"""Resume file parsing and structured data extraction."""

import io
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


def _extract_text_from_pdf(data: bytes) -> str:
    """Extract plain text from a PDF byte stream."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = reader.pages
    logger.debug("PDF has {} page(s)", len(pages))
    text = "\n".join(page.extract_text() or "" for page in pages)
    logger.debug("PDF extracted {} chars", len(text))
    return text


def _extract_text_from_docx(data: bytes) -> str:
    """Extract plain text from a DOCX byte stream."""
    from docx import Document  # type: ignore[import-untyped]

    doc = Document(io.BytesIO(data))
    text = "\n".join(para.text for para in doc.paragraphs)
    logger.debug("DOCX extracted {} chars", len(text))
    return text


def extract_text(data: bytes, filename: str) -> str:
    """Dispatch file-format detection and return raw text."""
    ext = Path(filename).suffix.lower()
    logger.info(
        "extract_text: filename={!r} ext={!r} size={} bytes", filename, ext, len(data)
    )
    if ext == ".pdf":
        return _extract_text_from_pdf(data)
    if ext in (".docx", ".doc"):
        return _extract_text_from_docx(data)
    # Plain text fallback (txt, md, etc.)
    text = data.decode("utf-8", errors="replace")
    logger.debug("Plain-text decoded {} chars", len(text))
    return text


# Matches the EDUCATION / ОБРАЗОВАНИЕ section header then captures following non-empty lines.
_EDU_SECTION_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:education|образование)[ \t]*\n((?:[ \t]*[^\n]+\n?){1,8})",
    re.IGNORECASE,
)

# Fallback: match a degree keyword and the rest of that line.
_EDU_DEGREE_RE = re.compile(
    r"(?:bachelor|master|ph\.?d\.?|mba|b\.?s\.?c?\.?|m\.?s\.?c?\.?|b\.?a\.?|m\.?a\.?|"
    r"associate|specialist|бакалавр|магистр|доктор|аспирант|специалист|диплом)"
    r"[^\n]{0,150}",
    re.IGNORECASE,
)


def _extract_education(text: str) -> str | None:
    """Return education as a semicolon-separated string, or None."""
    section = _EDU_SECTION_RE.search(text)
    if section:
        lines = [ln.strip() for ln in section.group(1).splitlines() if ln.strip()]
        return "; ".join(lines[:6]) or None
    # No section header — fall back to degree-keyword scan.
    matches = _EDU_DEGREE_RE.findall(text)
    return "; ".join(m.strip() for m in matches[:3]) or None


def _regex_fallback(text: str) -> dict[str, Any]:
    """Extract basic fields with regex when no LLM key is configured."""
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    years_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:year|лет|год|года)\b",
        text,
        re.IGNORECASE,
    )
    education = _extract_education(text)
    result: dict[str, Any] = {
        "name": None,
        "email": email_match.group(0) if email_match else None,
        "skills": [],
        "experience_years": float(years_match.group(1)) if years_match else None,
        "education": education,
    }
    logger.info(
        "regex_fallback → email={!r} experience_years={} education={!r}",
        result["email"],
        result["experience_years"],
        result["education"],
    )
    return result


def _parse_with_llm(text: str, api_key: str) -> dict[str, Any]:
    """Use Claude to extract structured fields from resume text."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""Extract structured information from the following resume text.
Return ONLY a valid JSON object with these fields:
- name: full name of the candidate (string or null)
- email: email address (string or null)
- skills: list of technical skills and technologies (array of strings)
- experience_years: total years of work experience as a number (float or null)
- education: list of education entries, each with keys: degree (e.g. "Bachelor's", "Master's", "PhD"), field_of_study, institution, graduation_year. The institution name and degree may appear on separate lines within the same education section — group them together as one entry. (array of objects, or null if no education found)

Resume text:
{text[:4000]}

JSON output:"""

    logger.debug("Calling Claude for resume parse (text_length={})", len(text))
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=768,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise ValueError(f"Unexpected response block type: {type(block)}")
    raw = block.text.strip()
    logger.debug("Claude raw response: {!r}", raw[:300])
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed: dict[str, Any] = json.loads(raw)
        # Normalize education: list of objects → human-readable string
        edu = parsed.get("education")
        if isinstance(edu, list):
            parts = []
            for entry in edu:
                if isinstance(entry, dict):
                    parts.append(
                        ", ".join(
                            str(v)
                            for v in [
                                entry.get("degree"),
                                entry.get("field_of_study"),
                                entry.get("institution"),
                                entry.get("graduation_year"),
                            ]
                            if v
                        )
                    )
                elif isinstance(entry, str):
                    parts.append(entry)
            parsed["education"] = "; ".join(parts) if parts else None
        logger.info(
            "LLM parse → name={!r} email={!r} skills={} experience_years={} education={!r}",
            parsed.get("name"),
            parsed.get("email"),
            len(parsed.get("skills") or []),
            parsed.get("experience_years"),
            parsed.get("education"),
        )
        return parsed
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM returned non-JSON ({}) raw={!r} — falling back to regex",
            exc,
            raw[:200],
        )
        return _regex_fallback(text)


def parse_resume(text: str, api_key: str = "") -> dict[str, Any]:
    """Return structured profile dict from raw resume text.

    Uses the Claude API when an api_key is provided, falls back to
    regex-based extraction otherwise.
    """
    logger.info("parse_resume: text_length={} llm_enabled={}", len(text), bool(api_key))
    if api_key:
        try:
            return _parse_with_llm(text, api_key)
        except Exception as exc:
            logger.warning(
                "LLM resume parse failed ({}: {}) — using regex fallback",
                type(exc).__name__,
                exc,
            )
    return _regex_fallback(text)
