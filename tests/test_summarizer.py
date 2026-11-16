import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from summarizer import summarize  # noqa: E402
from summarizer.extractive import EXTRACTIVE_METHODS, get_extractive  # noqa: E402
from summarizer.loaders import load_document  # noqa: E402
from summarizer.metrics import compression_ratio, novel_ngram_ratio, rouge  # noqa: E402
from summarizer.preprocess import clean_text, content_words, split_sentences  # noqa: E402

SAMPLE = (ROOT / "samples" / "urban_heat_islands.txt").read_text()


# --- preprocessing -------------------------------------------------------
def test_split_handles_abbreviations_initials_and_decimals():
    text = ("Dr. Smith met J. R. R. Tolkien in the U.S. last year. "
            "The rate rose 3.5 percent overall. Was it worth it? Yes, it was worth it!")
    assert split_sentences(text) == [
        "Dr. Smith met J. R. R. Tolkien in the U.S. last year.",
        "The rate rose 3.5 percent overall.",
        "Was it worth it?",
        "Yes, it was worth it!",
    ]


def test_split_drops_headings_and_respects_paragraphs():
    text = "Introduction\n\nThe first paragraph ends here\n\nA second paragraph starts here."
    # Short paragraphs without closing punctuation are treated as headings.
    assert split_sentences(text) == ["A second paragraph starts here."]


def test_clean_text_rejoins_hyphenation_and_strips_citations():
    assert clean_text("summari-\nsation works [12] well") == "summarisation works well"


def test_content_words_removes_stopwords_and_stems():
    assert content_words("The cities are running") == ["citi", "run"]


# --- extractive ----------------------------------------------------------
@pytest.mark.parametrize("method", list(EXTRACTIVE_METHODS))
def test_extractive_returns_k_source_sentences_in_order(method):
    result = get_extractive(method).summarize(SAMPLE, num_sentences=4)
    assert len(result.selected) == 4
    assert result.selected == sorted(result.selected)
    for idx in result.selected:
        assert result.sentences[idx] in result.summary
        assert result.sentences[idx] in clean_text(SAMPLE)


def test_ratio_controls_length():
    short = summarize(SAMPLE, method="textrank", ratio=0.1)
    long = summarize(SAMPLE, method="textrank", ratio=0.4)
    assert len(short.selected) < len(long.selected)


def test_redundancy_filter_skips_duplicates():
    text = ("Solar panels convert sunlight into electricity for homes. "
            "Solar panels convert sunlight into electricity for homes and offices. "
            "Wind turbines generate power from moving air near the coast. "
            "Batteries store surplus energy for use at night time.")
    result = get_extractive("frequency", redundancy_threshold=0.6).summarize(text, num_sentences=2)
    chosen = [result.sentences[i] for i in result.selected]
    assert not all(s.startswith("Solar") for s in chosen)


def test_short_and_empty_documents():
    assert summarize("", method="textrank").text == ""
    one = "Only one sentence is here in this tiny document."
    assert summarize(one, method="lsa").text == one


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        summarize(SAMPLE, method="nope")


# --- metrics -------------------------------------------------------------
def test_rouge_perfect_and_disjoint():
    assert rouge("the cat sat on the mat.", "the cat sat on the mat.")["rouge1"] == pytest.approx(100)
    assert rouge("dogs bark loudly.", "the cat sat on the mat.")["rouge2"] == 0


def test_extractive_has_no_novel_bigrams():
    result = summarize(SAMPLE, method="textrank", num_sentences=3)
    assert novel_ngram_ratio(result.text, SAMPLE) == 0
    assert 0 < compression_ratio(result.text, SAMPLE) < 100


# --- loaders -------------------------------------------------------------
def test_load_docx_roundtrip():
    from docx import Document

    doc = Document()
    doc.add_paragraph("First paragraph of the document.")
    doc.add_paragraph("Second paragraph of the document.")
    buf = io.BytesIO()
    doc.save(buf)
    text = load_document(buf.getvalue(), filename="test.docx")
    assert "First paragraph" in text and "Second paragraph" in text


def test_load_rejects_unknown_extension():
    with pytest.raises(ValueError):
        load_document(b"data", filename="file.xyz")


# --- abstractive (needs the model weights; skipped when not downloaded) ----
def _model_cached(hf_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(hf_id, "config.json"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _model_cached("google-t5/t5-small"), reason="t5-small not downloaded")
@pytest.mark.parametrize("method", ["abstractive", "hybrid"])
def test_abstractive_generates_text(method):
    result = summarize(SAMPLE, method=method, model="t5-small", max_length=60, num_beams=2)
    assert 5 < len(result.text.split()) < len(SAMPLE.split())
    assert result.text[0].isupper()
