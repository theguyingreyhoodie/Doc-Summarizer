"""Extractive summarisation: select the most informative sentences verbatim.

Three classic algorithms are implemented from first principles:

* ``FrequencySummarizer`` - Luhn-style word-frequency scoring.
* ``TextRankSummarizer``  - graph ranking (PageRank) over a TF-IDF cosine
  similarity graph of sentences (Mihalcea & Tarau, 2004).
* ``LSASummarizer``       - latent semantic analysis with SVD, scoring each
  sentence by its weight in the top latent topics (Steinberger & Jezek, 2004).

All of them share the same selection step: pick the top-k sentences, optionally
skipping near-duplicates (redundancy filtering), and return them in their
original document order so the summary reads coherently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import analyzer, content_words, split_sentences


@dataclass
class ExtractiveResult:
    summary: str
    selected: list[int]            # indices of chosen sentences (document order)
    sentences: list[str]           # all sentences of the document
    scores: list[float] = field(default_factory=list)


def num_sentences_for(n_total: int, ratio: float | None, num_sentences: int | None) -> int:
    """Resolve the target summary length."""
    if num_sentences is not None:
        k = num_sentences
    else:
        k = round(n_total * (ratio if ratio is not None else 0.2))
    return max(1, min(n_total, k))


class BaseExtractiveSummarizer:
    name = "base"

    def __init__(self, redundancy_threshold: float = 0.7, position_weight: float = 0.0,
                 ideal_length: int = 12):
        # Sentences whose TF-IDF cosine similarity to an already-selected one is
        # above this threshold are skipped (simple MMR-style redundancy control).
        self.redundancy_threshold = redundancy_threshold
        # Optional bonus for early sentences (news articles front-load facts).
        self.position_weight = position_weight
        # Sentences shorter than this many words are softly penalised: short
        # ones ("This brought new challenges.") rarely make good summaries.
        self.ideal_length = ideal_length

    def score(self, sentences: list[str]) -> np.ndarray:
        raise NotImplementedError

    def summarize(
        self,
        text: str,
        ratio: float | None = None,
        num_sentences: int | None = None,
    ) -> ExtractiveResult:
        sentences = split_sentences(text)
        if not sentences:
            return ExtractiveResult("", [], [])
        k = num_sentences_for(len(sentences), ratio, num_sentences)
        if len(sentences) <= k:
            return ExtractiveResult(" ".join(sentences), list(range(len(sentences))),
                                    sentences, [1.0] * len(sentences))

        scores = _normalise(self.score(sentences))
        if self.ideal_length:
            lengths = np.array([len(s.split()) for s in sentences])
            scores = scores * np.minimum(1.0, lengths / self.ideal_length) ** 0.5
        if self.position_weight:
            position = 1.0 / np.sqrt(np.arange(1, len(sentences) + 1))
            scores = (1 - self.position_weight) * scores + self.position_weight * position

        selected = self._select(sentences, scores, k)
        summary = " ".join(sentences[i] for i in selected)
        return ExtractiveResult(summary, selected, sentences, scores.tolist())

    def _select(self, sentences: list[str], scores: np.ndarray, k: int) -> list[int]:
        order = np.argsort(-scores, kind="stable")
        if self.redundancy_threshold >= 1.0:
            return sorted(order[:k].tolist())
        sim = _similarity_matrix(sentences)
        chosen: list[int] = []
        for idx in order:
            if all(sim[idx, j] < self.redundancy_threshold for j in chosen):
                chosen.append(int(idx))
            if len(chosen) == k:
                break
        # If filtering was too strict, top up with the best remaining sentences.
        for idx in order:
            if len(chosen) == k:
                break
            if idx not in chosen:
                chosen.append(int(idx))
        return sorted(chosen)


class LeadSummarizer(BaseExtractiveSummarizer):
    """Baseline: the first k sentences (Lead-k). Strong on news text."""

    name = "lead"

    def score(self, sentences):
        return np.linspace(1.0, 0.0, num=len(sentences))

    def _select(self, sentences, scores, k):
        return list(range(k))


class FrequencySummarizer(BaseExtractiveSummarizer):
    """Luhn (1958)-style scoring: sentences dense in frequent content words win."""

    name = "frequency"

    def score(self, sentences):
        tokenised = [content_words(s) for s in sentences]
        freq = Counter(w for words in tokenised for w in words)
        if not freq:
            return np.zeros(len(sentences))
        top = max(freq.values())
        weights = {w: c / top for w, c in freq.items()}
        scores = []
        for words in tokenised:
            if not words:
                scores.append(0.0)
                continue
            # Average weight, damped by sqrt(length) so long sentences are not
            # favoured purely for being long, yet short ones are not over-rewarded.
            scores.append(sum(weights[w] for w in words) / np.sqrt(len(words)))
        return np.array(scores)


class TextRankSummarizer(BaseExtractiveSummarizer):
    """PageRank over a sentence-similarity graph."""

    name = "textrank"

    def __init__(self, damping: float = 0.85, min_similarity: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.damping = damping
        self.min_similarity = min_similarity

    def score(self, sentences):
        sim = _similarity_matrix(sentences)
        np.fill_diagonal(sim, 0.0)
        sim[sim < self.min_similarity] = 0.0
        graph = nx.from_numpy_array(sim)
        if graph.number_of_edges() == 0:
            return np.zeros(len(sentences))
        ranks = nx.pagerank(graph, alpha=self.damping, weight="weight", max_iter=200)
        return np.array([ranks[i] for i in range(len(sentences))])


class LSASummarizer(BaseExtractiveSummarizer):
    """Latent Semantic Analysis: rank sentences by their strength across topics."""

    name = "lsa"

    def __init__(self, n_topics: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics

    def score(self, sentences):
        try:
            matrix = TfidfVectorizer(analyzer=analyzer).fit_transform(sentences)
        except ValueError:  # empty vocabulary
            return np.zeros(len(sentences))
        max_topics = min(matrix.shape) - 1
        if max_topics < 1:
            return np.asarray(matrix.sum(axis=1)).ravel()
        n_topics = self.n_topics or max(1, min(max_topics, len(sentences) // 5 + 1))
        n_topics = min(n_topics, max_topics)
        svd = TruncatedSVD(n_components=n_topics, random_state=0)
        sentence_topics = svd.fit_transform(matrix)        # U * Sigma
        # Steinberger & Jezek: length of the sentence vector in the
        # singular-value-weighted latent space.
        return np.sqrt((sentence_topics ** 2).sum(axis=1))


EXTRACTIVE_METHODS = {
    "lead": LeadSummarizer,
    "frequency": FrequencySummarizer,
    "textrank": TextRankSummarizer,
    "lsa": LSASummarizer,
}


def get_extractive(name: str, **kwargs) -> BaseExtractiveSummarizer:
    try:
        return EXTRACTIVE_METHODS[name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown extractive method {name!r}; choose from {list(EXTRACTIVE_METHODS)}") from None


def _similarity_matrix(sentences: list[str]) -> np.ndarray:
    try:
        tfidf = TfidfVectorizer(analyzer=analyzer).fit_transform(sentences)
    except ValueError:
        return np.zeros((len(sentences), len(sentences)))
    return cosine_similarity(tfidf)


def _normalise(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)
