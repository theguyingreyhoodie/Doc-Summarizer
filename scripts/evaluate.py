"""Benchmark every summariser on the CNN/DailyMail test set with ROUGE.

    python scripts/evaluate.py                 # 100 articles, all methods
    python scripts/evaluate.py -n 500 --methods lead textrank lsa

Writes per-article scores to results/scores_per_article.csv and an aggregate
table to results/rouge_cnn_dailymail.md.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from summarizer import ALL_METHODS, summarize  # noqa: E402
from summarizer.metrics import compression_ratio, novel_ngram_ratio, rouge  # noqa: E402


TEST_SIZE = 11_490  # CNN/DailyMail 3.0.0 test split


def load_articles(n: int, spread: bool = True):
    """Yield n test articles.

    The test split stores all CNN articles first, then DailyMail ones, and the
    CNN subset is much harder (Lead-3 ROUGE-1 ~29 vs ~40 on DailyMail). With
    ``spread`` the articles are taken at an even stride across the whole split
    so the sample mirrors the full test set; otherwise the first n are used.
    """
    from datasets import load_dataset

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test", streaming=True)
    stride = max(1, TEST_SIZE // n) if spread else 1
    taken = 0
    for i, row in enumerate(ds):
        if taken >= n:
            break
        if i % stride == 0:
            taken += 1
            yield row["id"], row["article"], row["highlights"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-articles", type=int, default=100)
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    parser.add_argument("--model", default="distilbart")
    parser.add_argument("--first", action="store_true",
                        help="use the first n articles (CNN only) instead of an even spread")
    parser.add_argument("--sentences", type=int, default=3,
                        help="extractive summary length (3 matches the ~3-4 sentence highlights)")
    args = parser.parse_args()

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    articles = list(load_articles(args.num_articles, spread=not args.first))
    print(f"Loaded {len(articles)} CNN/DailyMail test articles")

    rows, timings = [], {m: 0.0 for m in args.methods}
    for idx, (doc_id, article, reference) in enumerate(articles, 1):
        for method in args.methods:
            start = time.perf_counter()
            result = summarize(article, method=method, num_sentences=args.sentences, model=args.model)
            timings[method] += time.perf_counter() - start
            rows.append({
                "id": doc_id, "method": method, **rouge(result.text, reference),
                "words": len(result.text.split()),
                "compression": compression_ratio(result.text, article),
                "novel_bigrams": novel_ngram_ratio(result.text, article),
            })
        if idx % 10 == 0 or idx == len(articles):
            print(f"  {idx}/{len(articles)} articles done")

    with open(out_dir / "scores_per_article.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    model_note = f" ({args.model})"
    lines = [
        f"# ROUGE on CNN/DailyMail test ({len(articles)} articles, "
        + ("first n - CNN only" if args.first else "evenly spread over the test split") + ")",
        "",
        f"Extractive methods select {args.sentences} sentences. ROUGE values are F1 x 100; "
        "ROUGE-L is the summary-level variant (rougeLsum).",
        "",
        "| Method | Type | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg words | Novel bigrams % | Sec/doc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in args.methods:
        subset = [r for r in rows if r["method"] == method]
        mean = lambda key: statistics.mean(r[key] for r in subset)  # noqa: E731
        kind = "abstractive" if method == "abstractive" else "hybrid" if method == "hybrid" else (
            "baseline" if method == "lead" else "extractive")
        name = method + (model_note if kind in ("abstractive", "hybrid") else "")
        lines.append(
            f"| {name} | {kind} | {mean('rouge1'):.2f} | {mean('rouge2'):.2f} | {mean('rougeL'):.2f} "
            f"| {mean('words'):.0f} | {mean('novel_bigrams'):.1f} | {timings[method] / len(articles):.2f} |"
        )
    table = "\n".join(lines) + "\n"
    (out_dir / "rouge_cnn_dailymail.md").write_text(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
