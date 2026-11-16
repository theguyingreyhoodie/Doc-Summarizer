"""Evaluation metrics for summaries."""

from __future__ import annotations

from rouge_score import rouge_scorer

from .preprocess import split_sentences, tokenize

_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)


def rouge(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 (x100). ROUGE-L is the summary-level variant (rougeLsum)."""
    # rougeLsum expects sentences separated by newlines.
    pred = "\n".join(split_sentences(prediction, min_words=1))
    ref = "\n".join(split_sentences(reference, min_words=1))
    scores = _scorer.score(ref, pred)
    return {
        "rouge1": 100 * scores["rouge1"].fmeasure,
        "rouge2": 100 * scores["rouge2"].fmeasure,
        "rougeL": 100 * scores["rougeLsum"].fmeasure,
    }


def novel_ngram_ratio(summary: str, source: str, n: int = 2) -> float:
    """Share of summary n-grams absent from the source (abstractiveness, x100).

    Extractive summaries score ~0; abstractive ones score higher the more they
    paraphrase instead of copying.
    """
    def ngrams(text):
        # Within sentences only: joining extracted sentences must not create
        # "novel" n-grams across the boundary.
        grams = set()
        for sentence in split_sentences(text, min_words=1) or [text]:
            tokens = tokenize(sentence)
            grams.update(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return grams

    summary_ngrams = ngrams(summary)
    if not summary_ngrams:
        return 0.0
    return 100 * len(summary_ngrams - ngrams(source)) / len(summary_ngrams)


def compression_ratio(summary: str, source: str) -> float:
    """Summary length as a percentage of source length (in words)."""
    source_words = len(tokenize(source))
    return 100 * len(tokenize(summary)) / source_words if source_words else 0.0
