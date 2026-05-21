"""Unit tests for the three-stage NLP pipeline (no DB required)."""

from AiRecAgent.services.nlp_pipeline import (
    NLPResult,
    extract_entities,
    extract_keywords,
    run_pipeline,
    tokenize,
)

_SAMPLE = (
    "Jane Doe is a Senior Python developer at Acme Corp with 6 years of experience "
    "in FastAPI, PostgreSQL, and Docker. She holds a Master's degree from MIT."
)


# ---------------------------------------------------------------------------
# Stage 1 — tokenize
# ---------------------------------------------------------------------------


def test_tokenize_returns_non_empty() -> None:
    """Tokenize produces tokens and at least one sentence."""
    tokens, sentences = tokenize(_SAMPLE)
    assert len(tokens) > 10
    assert len(sentences) >= 1


def test_tokenize_no_whitespace_tokens() -> None:
    """Tokenize strips pure whitespace tokens from output."""
    tokens, _ = tokenize(_SAMPLE)
    assert all(t.strip() != "" for t in tokens)


def test_tokenize_sentences_are_strings() -> None:
    """Tokenize returns sentences as plain strings."""
    _, sentences = tokenize(_SAMPLE)
    assert all(isinstance(s, str) for s in sentences)


# ---------------------------------------------------------------------------
# Stage 2 — extract_entities
# ---------------------------------------------------------------------------


def test_extract_entities_finds_person() -> None:
    """PERSON entities include the name from the sample text."""
    entities = extract_entities(_SAMPLE)
    persons = entities.get("PERSON", [])
    assert any("Jane" in p or "Doe" in p for p in persons)


def test_extract_entities_finds_org() -> None:
    """ORG entities include organisations from the sample text."""
    entities = extract_entities(_SAMPLE)
    orgs = entities.get("ORG", [])
    assert any("Acme" in o or "MIT" in o for o in orgs)


def test_extract_entities_returns_dict_of_lists() -> None:
    """extract_entities returns a dict mapping str labels to str lists."""
    entities = extract_entities(_SAMPLE)
    assert isinstance(entities, dict)
    for label, mentions in entities.items():
        assert isinstance(label, str)
        assert isinstance(mentions, list)


def test_extract_entities_deduplicates() -> None:
    """Duplicate mentions in text appear only once per entity label."""
    repeated = _SAMPLE + " " + _SAMPLE
    entities = extract_entities(repeated)
    for mentions in entities.values():
        assert len(mentions) == len(set(mentions))


# ---------------------------------------------------------------------------
# Stage 3 — extract_keywords
# ---------------------------------------------------------------------------


def test_extract_keywords_returns_list() -> None:
    """extract_keywords returns a non-empty list for normal input."""
    tokens, _ = tokenize(_SAMPLE)
    keywords = extract_keywords(tokens)
    assert isinstance(keywords, list)
    assert len(keywords) > 0


def test_extract_keywords_no_stop_words() -> None:
    """Common stop words are excluded from extracted keywords."""
    tokens, _ = tokenize(_SAMPLE)
    keywords = extract_keywords(tokens)
    stop_words = {"the", "and", "with", "for", "she", "has", "from", "are", "his"}
    assert not any(kw in stop_words for kw in keywords)


def test_extract_keywords_top_n_respected() -> None:
    """extract_keywords honours the top_n limit."""
    tokens, _ = tokenize(_SAMPLE)
    keywords = extract_keywords(tokens, top_n=3)
    assert len(keywords) <= 3


def test_extract_keywords_empty_tokens() -> None:
    """Empty token list returns empty keywords list."""
    assert extract_keywords([]) == []


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def test_run_pipeline_returns_nlp_result() -> None:
    """run_pipeline returns an NLPResult dataclass instance."""
    result = run_pipeline(_SAMPLE)
    assert isinstance(result, NLPResult)


def test_run_pipeline_tokens_count_matches() -> None:
    """tokens_count field equals the length of the tokens list."""
    result = run_pipeline(_SAMPLE)
    assert result.tokens_count == len(result.tokens)


def test_run_pipeline_entities_and_keywords_populated() -> None:
    """Pipeline extracts at least one entity type and one keyword."""
    result = run_pipeline(_SAMPLE)
    assert len(result.entities) > 0
    assert len(result.keywords) > 0


def test_run_pipeline_empty_string() -> None:
    """Empty input produces empty collections in NLPResult."""
    result = run_pipeline("")
    assert result.tokens == []
    assert result.sentences == []
    assert result.entities == {}
    assert result.keywords == []
