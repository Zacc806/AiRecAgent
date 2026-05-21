"""Unit tests for job description parsing (no DB or LLM required)."""

from AiRecAgent.services.job_parser import _infer_experience_level, parse_job

_JD_SENIOR = (
    "We are looking for a Senior Python Engineer with 7+ years of experience "
    "in backend development using FastAPI, PostgreSQL, and Docker."
)

_JD_JUNIOR = (
    "Junior Frontend Developer wanted. 1 year of experience with React and JavaScript."
)

_JD_LEAD = (
    "Lead Software Architect to head the platform team. "
    "Design distributed systems and mentor engineers."
)

_JD_AMBIGUOUS = (
    "Software Engineer role working on data pipelines with Python and Spark."
)


# ---------------------------------------------------------------------------
# _infer_experience_level
# ---------------------------------------------------------------------------


def test_infer_level_senior_keyword() -> None:
    """Senior keyword maps to Senior level."""
    assert _infer_experience_level(_JD_SENIOR) == "Senior"


def test_infer_level_junior_keyword() -> None:
    """Junior keyword maps to Junior level."""
    assert _infer_experience_level(_JD_JUNIOR) == "Junior"


def test_infer_level_lead_keyword() -> None:
    """Lead keyword maps to Lead level."""
    assert _infer_experience_level(_JD_LEAD) == "Lead"


def test_infer_level_senior_from_years() -> None:
    """8 years of experience infers Senior level."""
    text = "Requires 8 years of experience in cloud infrastructure."
    assert _infer_experience_level(text) == "Senior"


def test_infer_level_mid_from_years() -> None:
    """4 years of experience infers Mid level."""
    text = "Ideal candidate has 4 years experience with microservices."
    assert _infer_experience_level(text) == "Mid"


def test_infer_level_none_when_undetectable() -> None:
    """Returns None when no level signal is present."""
    assert _infer_experience_level("We are hiring an engineer.") is None


# ---------------------------------------------------------------------------
# parse_job — regex/NLP path (no api_key)
# ---------------------------------------------------------------------------


def test_parse_job_returns_dict() -> None:
    """parse_job returns a dict."""
    result = parse_job(
        title="Senior Python Engineer",
        description=_JD_SENIOR,
    )
    assert isinstance(result, dict)


def test_parse_job_has_required_keys() -> None:
    """parse_job result contains the expected top-level keys."""
    result = parse_job(title="Backend Dev", description=_JD_SENIOR)
    assert "required_skills" in result
    assert "experience_level" in result
    assert "tech_stack" in result
    assert "nlp_keywords" in result


def test_parse_job_experience_level_senior() -> None:
    """Senior title and description produce Senior experience level."""
    result = parse_job(title="Senior Python Engineer", description=_JD_SENIOR)
    assert result["experience_level"] == "Senior"


def test_parse_job_experience_level_junior() -> None:
    """Junior title and description produce Junior experience level."""
    result = parse_job(title="Junior Frontend Developer", description=_JD_JUNIOR)
    assert result["experience_level"] == "Junior"


def test_parse_job_nlp_keywords_populated() -> None:
    """nlp_keywords is a non-empty list for a normal description."""
    result = parse_job(title="Data Engineer", description=_JD_AMBIGUOUS)
    assert isinstance(result["nlp_keywords"], list)
    assert len(result["nlp_keywords"]) > 0


def test_parse_job_required_skills_is_list() -> None:
    """required_skills is always a list."""
    result = parse_job(title="Backend Dev", description=_JD_SENIOR)
    assert isinstance(result["required_skills"], list)


def test_parse_job_tech_stack_is_list() -> None:
    """tech_stack is always a list."""
    result = parse_job(title="Backend Dev", description=_JD_SENIOR)
    assert isinstance(result["tech_stack"], list)


def test_parse_job_with_requirements_section() -> None:
    """Requirements text is considered when inferring level and keywords."""
    result = parse_job(
        title="ML Engineer",
        description="Build and deploy machine learning models.",
        requirements="Python, PyTorch, Kubernetes, 5+ years experience",
    )
    assert result["experience_level"] is not None or result["nlp_keywords"]
