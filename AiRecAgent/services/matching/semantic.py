"""Semantic similarity matching using Sentence-BERT embeddings."""

import functools

import numpy as np
from loguru import logger


@functools.lru_cache(maxsize=1)
def _load_model(model_name: str) -> "SentenceTransformer":  # type: ignore[name-defined]  # noqa: F821
    """Load and cache the SentenceTransformer model (singleton per process)."""
    from sentence_transformers import (
        SentenceTransformer,  # type: ignore[import-untyped]
    )

    logger.info("Loading embedding model: {}", model_name)
    return SentenceTransformer(model_name)


def encode(text: str, model_name: str) -> list[float]:
    """Encode a single text into a unit-normalised embedding vector."""
    model = _load_model(model_name)
    vector: np.ndarray = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two pre-normalised embedding vectors."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def score_candidate(
    candidate_embedding: list[float],
    job_embedding: list[float],
) -> float:
    """Return cosine similarity score in [0, 1] between candidate and job embeddings."""
    raw = cosine_similarity(candidate_embedding, job_embedding)
    # Shift from [-1, 1] to [0, 1]
    return (raw + 1.0) / 2.0
