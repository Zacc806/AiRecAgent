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
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_text_from_docx(data: bytes) -> str:
    """Extract plain text from a DOCX byte stream."""
    from docx import Document  # type: ignore[import-untyped]

    doc = Document(io.BytesIO(data))
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text(data: bytes, filename: str) -> str:
    """Dispatch file-format detection and return raw text."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return _extract_text_from_pdf(data)
    if ext in (".docx", ".doc"):
        return _extract_text_from_docx(data)
    # Plain text fallback (txt, md, etc.)
    return data.decode("utf-8", errors="replace")


def _regex_fallback(text: str) -> dict[str, Any]:
    """Extract basic fields with regex when no LLM key is configured."""
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    years_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:year|лет|год|года)\b",
        text,
        re.IGNORECASE,
    )
    return {
        "name": None,
        "email": email_match.group(0) if email_match else None,
        "skills": [],
        "experience_years": float(years_match.group(1)) if years_match else None,
        "education": None,
    }


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
- education: highest education level and institution (string or null)

Resume text:
{text[:4000]}

JSON output:"""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    block = message.content[0]
    if not isinstance(block, anthropic.types.TextBlock):
        raise ValueError(f"Unexpected response block type: {type(block)}")
    raw = block.text.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON for resume parse, falling back to regex")
        return _regex_fallback(text)


def parse_resume(text: str, api_key: str = "") -> dict[str, Any]:
    """Return structured profile dict from raw resume text.

    Uses the Claude API when an api_key is provided, falls back to
    regex-based extraction otherwise.
    """
    if api_key:
        try:
            return _parse_with_llm(text, api_key)
        except Exception as exc:
            logger.warning("LLM resume parse failed ({}), using regex fallback", exc)
    return _regex_fallback(text)
