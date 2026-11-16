"""Hybrid (extract-then-abstract) summarisation.

Abstractive models can only read ~1024 tokens at once. Instead of blindly
truncating a long document (losing everything after the window) or running
many map-reduce passes, the hybrid approach first uses TextRank to keep the
most central sentences that fit in the window, in their original order, and
then lets the transformer rewrite that condensed text fluently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .abstractive import AbstractiveSummarizer
from .extractive import TextRankSummarizer, _normalise
from .preprocess import split_sentences


@dataclass
class HybridResult:
    summary: str
    extracted: str          # intermediate extractive text fed to the model
    selected: list[int]
    sentences: list[str]


class HybridSummarizer:
    def __init__(self, abstractive: AbstractiveSummarizer, token_budget: int | None = None):
        self.abstractive = abstractive
        self.extractor = TextRankSummarizer(position_weight=0.1)
        self.token_budget = token_budget or abstractive.chunk_tokens

    def summarize(self, text: str, **gen_kwargs) -> HybridResult:
        sentences = split_sentences(text)
        if not sentences:
            return HybridResult("", "", [], [])

        lengths = [self.abstractive.count_tokens(s) for s in sentences]
        if sum(lengths) <= self.token_budget:
            selected = list(range(len(sentences)))   # already fits - no extraction needed
        else:
            scores = _normalise(self.extractor.score(sentences))
            position = 1.0 / np.sqrt(np.arange(1, len(sentences) + 1))
            scores = 0.9 * scores + 0.1 * position
            selected, used = [], 0
            for idx in np.argsort(-scores, kind="stable"):
                if used + lengths[idx] <= self.token_budget:
                    selected.append(int(idx))
                    used += lengths[idx]
            selected.sort()

        extracted = " ".join(sentences[i] for i in selected)
        summary = self.abstractive.summarize(extracted, **gen_kwargs)
        return HybridResult(summary, extracted, selected, sentences)
