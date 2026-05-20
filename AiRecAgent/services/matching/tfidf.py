"""TF-IDF based candidate-job matching."""

import numpy as np
from sklearn.feature_extraction.text import (
    TfidfVectorizer,  # type: ignore[import-untyped]
)
from sklearn.metrics.pairwise import (
    cosine_similarity as sk_cosine,  # type: ignore[import-untyped]
)


def score_candidates(
    candidate_texts: list[str],
    job_text: str,
) -> list[float]:
    """Fit a TF-IDF vectorizer on all texts and return per-candidate similarity scores.

    The vectorizer is fit on the union of all candidate texts plus the job
    description so that the IDF weights reflect the full corpus.

    Returns a list of float scores in [0, 1], one per candidate.
    """
    if not candidate_texts:
        return []

    corpus = [*candidate_texts, job_text]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Last row is the job; all others are candidates
    job_vector = tfidf_matrix[-1]
    candidate_matrix = tfidf_matrix[:-1]

    similarities: np.ndarray = sk_cosine(candidate_matrix, job_vector).flatten()
    return [float(s) for s in similarities]
