"""LLM-based candidate-job matching using the Claude API."""

import json
import re

from loguru import logger

_SCORE_PROMPT = """\
You are a professional recruiter. Score how well the candidate's resume matches the job description.

JOB DESCRIPTION:
{job_text}

CANDIDATE RESUME:
{candidate_text}

Return ONLY a valid JSON object:
{{
  "score": <float between 0.0 and 1.0>,
  "explanation": "<one or two sentences explaining the score in Russian>"
}}

JSON output:"""


async def score_candidate(
    candidate_text: str,
    job_text: str,
    api_key: str,
) -> tuple[float, str]:
    """Call Claude and return (relevance_score, explanation).

    score is in [0.0, 1.0].  Returns (0.0, "") on any failure.
    """
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    prompt = _SCORE_PROMPT.format(
        job_text=job_text[:2000],
        candidate_text=candidate_text[:2000],
    )
    try:
        message = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        block = message.content[0]
        if not isinstance(block, anthropic.types.TextBlock):
            return 0.0, ""
        raw = block.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
        explanation = str(data.get("explanation", ""))
        return max(0.0, min(1.0, score)), explanation
    except Exception as exc:
        logger.warning("LLM scoring failed: {}", exc)
        return 0.0, ""
