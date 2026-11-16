"""Text preprocessing: cleaning, sentence segmentation and word tokenisation.

Everything here is self-contained (no corpora downloads), so the extractive
summarisers work offline.
"""

from __future__ import annotations

import re
import unicodedata

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)
_stemmer = PorterStemmer()

# Abbreviations whose trailing period does not end a sentence.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc",
    "inc", "ltd", "co", "corp", "dept", "univ", "fig", "figs", "eq", "no",
    "vol", "pp", "approx", "est", "e.g", "i.e", "cf", "al", "gen", "col",
    "lt", "sgt", "capt", "gov", "sen", "rep", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "u.s", "u.k",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")
# Candidate boundary: ., ! or ? (plus closing quotes/brackets), whitespace, then
# an uppercase letter, digit or opening quote.
_BOUNDARY_RE = re.compile(r"([.!?][\"')\]]*)\s+(?=[\"'(\[]?[A-Z0-9])")


def clean_text(text: str) -> str:
    """Normalise unicode, fix hyphenated line breaks and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # hyphenation across lines
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)    # citation markers like [12]
    # Paragraph breaks become sentence breaks; single newlines become spaces.
    text = re.sub(r"\n\s*\n+", " \n\n ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text.strip()


def split_sentences(text: str, min_words: int = 3) -> list[str]:
    """Split text into sentences with a rule-based segmenter.

    Handles common abbreviations, initials ("J. R. R. Tolkien") and decimals,
    and treats blank lines (paragraph/heading breaks) as hard boundaries.
    Short paragraphs without closing punctuation (titles, headings) are dropped.
    Fragments shorter than ``min_words`` words are merged into a neighbour.
    """
    text = clean_text(text)
    sentences: list[str] = []
    for paragraph in re.split(r"\n\n", text):
        paragraph = paragraph.strip()
        if not paragraph or (_looks_like_heading(paragraph) and len(paragraph.split()) <= 15):
            continue  # empty, or a title/heading line rather than prose
        start = 0
        for match in _BOUNDARY_RE.finditer(paragraph):
            end = match.end(1)
            candidate = paragraph[start:end]
            last_token = candidate.rstrip(".!?\"')]").split()[-1:] or [""]
            token = last_token[0].lower().rstrip(".")
            if paragraph[match.start(1)] == "." and (
                token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha())
            ):
                continue  # abbreviation or initial, not a real boundary
            sentences.append(candidate.strip())
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return _merge_short(sentences, min_words)


def _merge_short(sentences: list[str], min_words: int) -> list[str]:
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(sentence.split()) < min_words and not _looks_like_heading(sentence):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    # Drop stand-alone fragments such as headings ("Introduction") from the pool.
    return [s for s in merged if len(s.split()) >= min_words]


def _looks_like_heading(sentence: str) -> bool:
    return not sentence.rstrip().endswith((".", "!", "?", '"', "'"))


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens."""
    return [t.lower() for t in _WORD_RE.findall(text)]


def content_words(text: str, stem: bool = True) -> list[str]:
    """Tokens with stop words and very short tokens removed, optionally stemmed."""
    words = [t for t in tokenize(text) if t not in STOP_WORDS and len(t) > 1]
    return [_stemmer.stem(w) for w in words] if stem else words


def analyzer(text: str) -> list[str]:
    """Analyzer for scikit-learn vectorisers (stop-word removal + stemming)."""
    return content_words(text, stem=True)


def word_count(text: str) -> int:
    return len(tokenize(text))
