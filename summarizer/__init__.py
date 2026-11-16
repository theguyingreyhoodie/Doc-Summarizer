"""Automatic document summarisation with extractive and abstractive NLP."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .extractive import EXTRACTIVE_METHODS, get_extractive
from .metrics import compression_ratio, novel_ngram_ratio

ABSTRACTIVE_METHODS = ("abstractive", "hybrid")
ALL_METHODS = tuple(EXTRACTIVE_METHODS) + ABSTRACTIVE_METHODS


@dataclass
class Summary:
    method: str
    text: str
    seconds: float
    compression: float          # % of source words kept
    novel_bigrams: float        # % of summary bigrams not in the source
    selected: list[int] = field(default_factory=list)   # extracted sentence indices
    sentences: list[str] = field(default_factory=list)  # source sentences
    intermediate: str = ""      # hybrid: the extractive text passed to the model


def summarize(
    text: str,
    method: str = "textrank",
    ratio: float | None = 0.2,
    num_sentences: int | None = None,
    model: str = "distilbart",
    max_length: int = 142,
    min_length: int = 30,
    num_beams: int = 4,
) -> Summary:
    """One entry point for every method.

    Extractive methods use ``ratio`` / ``num_sentences``; abstractive and hybrid
    use ``model``, ``max_length``, ``min_length`` (in tokens) and ``num_beams``.
    """
    start = time.perf_counter()
    selected: list[int] = []
    sentences: list[str] = []
    intermediate = ""

    if method in EXTRACTIVE_METHODS:
        result = get_extractive(method).summarize(text, ratio=ratio, num_sentences=num_sentences)
        output, selected, sentences = result.summary, result.selected, result.sentences
    elif method in ABSTRACTIVE_METHODS:
        from .abstractive import AbstractiveSummarizer

        abstractive = AbstractiveSummarizer(model)
        gen = dict(max_length=max_length, min_length=min_length, num_beams=num_beams)
        if method == "abstractive":
            output = abstractive.summarize(text, **gen)
        else:
            from .hybrid import HybridSummarizer

            result = HybridSummarizer(abstractive).summarize(text, **gen)
            output, selected, sentences = result.summary, result.selected, result.sentences
            intermediate = result.extracted
    else:
        raise ValueError(f"Unknown method {method!r}; choose from {list(ALL_METHODS)}")

    return Summary(
        method=method,
        text=output,
        seconds=time.perf_counter() - start,
        compression=compression_ratio(output, text),
        novel_bigrams=novel_ngram_ratio(output, text),
        selected=selected,
        sentences=sentences,
        intermediate=intermediate,
    )


__all__ = ["summarize", "Summary", "ALL_METHODS", "EXTRACTIVE_METHODS", "ABSTRACTIVE_METHODS"]
