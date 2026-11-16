"""Command-line interface.

Examples
--------
    python cli.py samples/urban_heat_islands.txt
    python cli.py report.pdf --method lsa --sentences 5
    python cli.py article.docx --method hybrid --model bart-large-cnn
    python cli.py samples/history_of_the_internet.txt --method all
    python cli.py article.txt --method all --reference reference_summary.txt
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from summarizer import ALL_METHODS, summarize
from summarizer.abstractive import DEFAULT_MODEL, MODELS
from summarizer.loaders import load_document
from summarizer.metrics import rouge
from summarizer.preprocess import word_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a document (.txt, .md, .pdf, .docx).")
    parser.add_argument("path", help="document to summarise, or '-' to read stdin")
    parser.add_argument("-m", "--method", default="textrank", choices=[*ALL_METHODS, "all"])
    length = parser.add_mutually_exclusive_group()
    length.add_argument("-n", "--sentences", type=int, help="extractive: number of sentences")
    length.add_argument("-r", "--ratio", type=float, default=0.2,
                        help="extractive: fraction of sentences to keep (default 0.2)")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    parser.add_argument("--max-length", type=int, default=142, help="abstractive: max tokens")
    parser.add_argument("--min-length", type=int, default=30, help="abstractive: min tokens")
    parser.add_argument("--beams", type=int, default=4, help="abstractive: beam width")
    parser.add_argument("--reference", help="reference summary file, to report ROUGE scores")
    args = parser.parse_args(argv)

    text = sys.stdin.read() if args.path == "-" else load_document(args.path)
    reference = load_document(args.reference) if args.reference else None
    print(f"Document: {word_count(text)} words\n")

    methods = ALL_METHODS if args.method == "all" else (args.method,)
    for method in methods:
        result = summarize(
            text, method=method, ratio=args.ratio, num_sentences=args.sentences,
            model=args.model, max_length=args.max_length, min_length=args.min_length,
            num_beams=args.beams,
        )
        label = method if method not in ("abstractive", "hybrid") else f"{method} ({args.model})"
        stats = (f"{result.seconds:.2f}s | {word_count(result.text)} words "
                 f"({result.compression:.1f}% of source) | novel bigrams {result.novel_bigrams:.1f}%")
        if reference:
            scores = rouge(result.text, reference)
            stats += " | " + " ".join(f"{k.upper()} {v:.1f}" for k, v in scores.items())
        print(f"=== {label.upper()} ===\n{stats}\n")
        print(textwrap.fill(result.text, width=100), end="\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
