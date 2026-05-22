"""TF-IDF based candidate-job matching."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine


def score_candidates(
    candidate_texts: list[str],
    job_text: str,
) -> list[float]:
    """Score candidates against a job description using TF-IDF cosine similarity and a logistic classifier."""
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
    job_vector = tfidf_matrix[-1]
    candidate_matrix = tfidf_matrix[:-1]

    cosine_scores: np.ndarray = sk_cosine(candidate_matrix, job_vector).flatten()

    # With fewer than 2 candidates there's nothing to classify.
    if len(candidate_texts) < 2:
        return [float(s) for s in cosine_scores]

    # Pseudo-labels: candidates above the median cosine score are "relevant".
    threshold = float(np.median(cosine_scores))
    pseudo_labels = (cosine_scores > threshold).astype(int)

    # Need both classes to train; fall back to cosine if all scores are equal.
    if len(np.unique(pseudo_labels)) < 2:
        return [float(s) for s in cosine_scores]

    clf = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    clf.fit(candidate_matrix, pseudo_labels)
    proba: np.ndarray = clf.predict_proba(candidate_matrix)[:, 1]

    return [float(p) for p in proba]
