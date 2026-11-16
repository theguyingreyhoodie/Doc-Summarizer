"""Streamlit web interface:  streamlit run app.py"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from summarizer import ABSTRACTIVE_METHODS, ALL_METHODS, summarize
from summarizer.abstractive import MODELS
from summarizer.loaders import SUPPORTED, load_document
from summarizer.metrics import rouge
from summarizer.preprocess import split_sentences, word_count

SAMPLES_DIR = Path(__file__).parent / "samples"

METHOD_INFO = {
    "lead": ("Lead-k baseline", "Takes the first k sentences. A surprisingly strong baseline for news."),
    "frequency": ("Word frequency (Luhn)", "Scores sentences by how many frequent content words they contain."),
    "textrank": ("TextRank", "PageRank over a graph of sentences linked by TF-IDF cosine similarity."),
    "lsa": ("LSA", "SVD of the TF-IDF matrix; picks sentences strongest in the main latent topics."),
    "abstractive": ("Abstractive (transformer)", "BART/T5 writes a new summary; long inputs are chunked and summarised map-reduce style."),
    "hybrid": ("Hybrid (TextRank → transformer)", "TextRank condenses the document to fit the model window, then the transformer rewrites it."),
}

st.set_page_config(page_title="Document Summarizer", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .summary-box {border-left: 4px solid #4C78A8; padding: 0.9rem 1.1rem; border-radius: 6px;
                    background: rgba(76,120,168,0.08); line-height: 1.65; font-size: 1.02rem;}
      .source-box {max-height: 420px; overflow-y: auto; line-height: 1.7; padding: 0.5rem 0.2rem;}
      mark {background: rgba(242,183,5,0.38); padding: 0 2px; border-radius: 3px; color: inherit;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -- sidebar: settings ----------------------------------------------------
with st.sidebar:
    st.header("Settings")
    compare = st.toggle("Compare all methods", value=False)
    method = None
    if not compare:
        method = st.radio("Method", ALL_METHODS, index=2, format_func=lambda m: METHOD_INFO[m][0])
        st.caption(METHOD_INFO[method][1])

    show_extractive = compare or method not in ABSTRACTIVE_METHODS
    show_abstractive = compare or method in ABSTRACTIVE_METHODS

    ratio, num_sentences = 0.2, None
    if show_extractive:
        st.subheader("Extractive length")
        mode = st.radio("Length by", ["Sentences", "Ratio"], horizontal=True)
        if mode == "Sentences":
            num_sentences = st.slider("Sentences", 1, 15, 4)
        else:
            ratio = st.slider("Ratio of document", 0.05, 0.6, 0.2, 0.05)

    model, max_length, min_length, beams = "distilbart", 142, 30, 4
    if show_abstractive:
        st.subheader("Abstractive model")
        model = st.selectbox("Model", list(MODELS), format_func=lambda k: k)
        st.caption(MODELS[model].description)
        min_length, max_length = st.slider("Summary length (tokens)", 10, 300, (30, 142))
        beams = st.slider("Beam width", 1, 8, 4, help="Higher = better search, slower.")

# -- main: input ----------------------------------------------------------
st.title("Automatic Document Summarizer")
st.write("Extractive (Lead, Frequency, TextRank, LSA) and abstractive (BART / T5) summarisation.")

tab_paste, tab_upload, tab_sample = st.tabs(["Paste text", "Upload file", "Sample document"])
text = ""
with tab_paste:
    pasted = st.text_area("Document text", height=220, placeholder="Paste an article, report or essay...")
with tab_upload:
    uploaded = st.file_uploader("Upload a document", type=[s.lstrip(".") for s in SUPPORTED])
with tab_sample:
    samples = sorted(SAMPLES_DIR.glob("*.txt"))
    sample = st.selectbox("Sample", samples, format_func=lambda p: p.stem.replace("_", " ").title())
    use_sample = st.checkbox("Use this sample", value=not pasted and uploaded is None)

if uploaded is not None:
    try:
        text = load_document(uploaded.getvalue(), filename=uploaded.name)
    except Exception as exc:  # noqa: BLE001 - show any parsing error to the user
        st.error(f"Could not read {uploaded.name}: {exc}")
elif pasted.strip():
    text = pasted
elif use_sample and sample:
    text = sample.read_text()

with st.expander("Reference summary (optional, for ROUGE scores)"):
    reference = st.text_area("Human-written reference summary", height=100, label_visibility="collapsed")

if not text.strip():
    st.info("Add a document using one of the tabs above.")
    st.stop()

sentences = split_sentences(text)
source_name = uploaded.name if uploaded else ("pasted text" if pasted.strip() else sample.name)
st.caption(f"Source: **{source_name}**")
c1, c2, c3 = st.columns(3)
c1.metric("Words", f"{word_count(text):,}")
c2.metric("Sentences", len(sentences))
c3.metric("Avg sentence length", f"{word_count(text) / max(1, len(sentences)):.1f} words")

if not st.button("Summarize", type="primary", use_container_width=True):
    st.stop()


# -- run ------------------------------------------------------------------
def run(m: str):
    return summarize(text, method=m, ratio=ratio, num_sentences=num_sentences, model=model,
                     max_length=max_length, min_length=min_length, num_beams=beams)


def highlighted_source(result) -> str:
    chosen = set(result.selected)
    parts = [f"<mark>{html.escape(s)}</mark>" if i in chosen else html.escape(s)
             for i, s in enumerate(result.sentences)]
    return f'<div class="source-box">{" ".join(parts)}</div>'


def show_result(result, heading: str | None = None):
    if heading:
        st.subheader(heading)
    st.markdown(f'<div class="summary-box">{html.escape(result.text)}</div>', unsafe_allow_html=True)
    cols = st.columns(5 if reference.strip() else 4)
    cols[0].metric("Summary words", word_count(result.text))
    cols[1].metric("Compression", f"{result.compression:.1f}%")
    cols[2].metric("Novel bigrams", f"{result.novel_bigrams:.1f}%",
                   help="Share of summary bigrams not found in the source: ~0% for extractive.")
    cols[3].metric("Time", f"{result.seconds:.2f}s")
    if reference.strip():
        scores = rouge(result.text, reference)
        cols[4].metric("ROUGE-1 / 2 / L", " / ".join(f"{v:.0f}" for v in scores.values()))
    if result.selected and result.sentences:
        label = ("Sentences TextRank passed to the model" if result.method == "hybrid"
                 else "Selected sentences in the source")
        with st.expander(label):
            st.markdown(highlighted_source(result), unsafe_allow_html=True)


def spinner_text(m: str) -> str:
    if m in ABSTRACTIVE_METHODS:
        return f"Running {METHOD_INFO[m][0]} with {model} (first run loads the model)..."
    return f"Running {METHOD_INFO[m][0]}..."


st.divider()
if compare:
    results = []
    progress = st.progress(0.0)
    for i, m in enumerate(ALL_METHODS, 1):
        with st.spinner(spinner_text(m)):
            results.append(run(m))
        progress.progress(i / len(ALL_METHODS))
    progress.empty()

    table = []
    for r in results:
        row = {"Method": METHOD_INFO[r.method][0], "Words": word_count(r.text),
               "Compression %": round(r.compression, 1), "Novel bigrams %": round(r.novel_bigrams, 1),
               "Seconds": round(r.seconds, 2)}
        if reference.strip():
            row.update({k.upper(): round(v, 1) for k, v in rouge(r.text, reference).items()})
        table.append(row)
    st.subheader("Comparison")
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    for r in results:
        show_result(r, METHOD_INFO[r.method][0])
else:
    with st.spinner(spinner_text(method)):
        result = run(method)
    show_result(result, METHOD_INFO[method][0])
    if method == "hybrid" and result.intermediate:
        st.caption(f"TextRank kept {len(result.selected)} of {len(result.sentences)} sentences "
                   f"({word_count(result.intermediate)} words) before abstractive rewriting.")
    st.download_button("Download summary", result.text, file_name="summary.txt")
