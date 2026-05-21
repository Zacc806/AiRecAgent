"""Classic NLP pipeline: tokenization, NER, and keyword extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from loguru import logger

# Maximum characters fed into spaCy per call (avoids memory spikes on huge docs)
_MAX_CHARS = 50_000


@dataclass
class NLPResult:
    """Output of the three-stage NLP pipeline."""

    tokens: list[str]
    sentences: list[str]
    entities: dict[str, list[str]]  # label → list of surface forms
    keywords: list[str]
    tokens_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.tokens_count = len(self.tokens)


@lru_cache(maxsize=1)
def _get_nlp() -> object:
    """Load and cache the spaCy model."""
    import spacy  # type: ignore[import-untyped]

    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("nlp_pipeline: loaded en_core_web_sm")
        return nlp
    except OSError:
        logger.warning(
            "nlp_pipeline: en_core_web_sm not found, falling back to blank model"
        )
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


# ---------------------------------------------------------------------------
# Stage 1 — Tokenization
# ---------------------------------------------------------------------------


def tokenize(text: str) -> tuple[list[str], list[str]]:
    """Stage 1: Tokenize text into word tokens and sentences via spaCy.

    Returns a (tokens, sentences) tuple.  Whitespace-only tokens are excluded.
    """
    nlp = _get_nlp()
    doc = nlp(text[:_MAX_CHARS])  # type: ignore[operator]
    tokens = [t.text for t in doc if not t.is_space]
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    logger.debug(
        "nlp_pipeline.tokenize: {} tokens, {} sentences", len(tokens), len(sentences)
    )
    return tokens, sentences


# ---------------------------------------------------------------------------
# Stage 2 — Named Entity Recognition
# ---------------------------------------------------------------------------


def extract_entities(text: str) -> dict[str, list[str]]:
    """Stage 2: Named Entity Recognition via spaCy.

    Returns a dict mapping spaCy entity labels (PERSON, ORG, DATE, GPE, …)
    to deduplicated lists of surface forms found in the text.
    """
    nlp = _get_nlp()
    doc = nlp(text[:_MAX_CHARS])  # type: ignore[operator]
    entities: dict[str, list[str]] = {}
    seen: set[str] = set()
    for ent in doc.ents:
        key = f"{ent.label_}:{ent.text}"
        if key in seen:
            continue
        seen.add(key)
        entities.setdefault(ent.label_, []).append(ent.text)
    logger.debug("nlp_pipeline.extract_entities: {} entity types found", len(entities))
    return entities


# ---------------------------------------------------------------------------
# Stage 3 — Keyword extraction
# ---------------------------------------------------------------------------


def extract_keywords(tokens: list[str], top_n: int = 20) -> list[str]:
    """Stage 3: Extract keywords by term frequency over non-stop content words.

    Filters punctuation, stop words (via spaCy vocab), and short tokens, then
    ranks by descending frequency and returns the top *top_n* lemmas.
    """
    nlp = _get_nlp()
    freq: dict[str, int] = {}
    for token_text in tokens:
        if not token_text.isalpha() or len(token_text) < 3:
            continue
        lower = token_text.lower()
        # Use spaCy's built-in stop-word list when the vocab is available
        lexeme = nlp.vocab[lower]  # type: ignore[attr-defined]
        if lexeme.is_stop:
            continue
        freq[lower] = freq.get(lower, 0) + 1

    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    keywords = [word for word, _ in ranked[:top_n]]
    logger.debug("nlp_pipeline.extract_keywords: {} keywords extracted", len(keywords))
    return keywords


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(text: str) -> NLPResult:
    """Run all three NLP pipeline stages and return a combined NLPResult."""
    logger.debug("nlp_pipeline.run_pipeline: processing {} chars", len(text))

    tokens, sentences = tokenize(text)
    entities = extract_entities(text)
    keywords = extract_keywords(tokens)

    result = NLPResult(
        tokens=tokens,
        sentences=sentences,
        entities=entities,
        keywords=keywords,
    )
    logger.info(
        "nlp_pipeline: {} tokens | {} sentences | {} entity types | {} keywords",
        result.tokens_count,
        len(sentences),
        len(entities),
        len(keywords),
    )
    return result
