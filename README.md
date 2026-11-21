# Automatic Document Summarization System

An NLP mini project that summarises documents in two ways and compares them:

- **Extractive** methods choose the most important sentences from the document and copy them unchanged.
- **Abstractive** methods use a pretrained transformer (BART / T5) to write a new summary in its own words.

It accepts `.txt`, `.md`, `.pdf` and `.docx` files and offers a Streamlit web app, a command-line tool and a ROUGE benchmark on CNN/DailyMail.

## Methods

| Method | Type | Idea |
|---|---|---|
| **Lead-k** | baseline | First *k* sentences. A strong baseline for news, where key facts come first. |
| **Frequency (Luhn)** | extractive | Count content words (stop words removed, Porter-stemmed). A sentence scores higher when it has more frequent words, normalised by √length. |
| **TextRank** | extractive | Build a graph with sentences as nodes and TF-IDF cosine similarities as edge weights, then run PageRank (d = 0.85). Central sentences rank highest. |
| **LSA** | extractive | SVD of the sentence × term TF-IDF matrix. Each sentence is scored by the length of its vector in the singular-value-weighted latent topic space (Steinberger & Ježek). |
| **Abstractive** | abstractive | A pretrained seq2seq transformer (DistilBART / BART-large-CNN / T5-small) with beam search. Documents longer than the model's input window are handled by **map-reduce**: split them into sentence-aligned chunks, summarise each chunk, then summarise the joined partial summaries. |
| **Hybrid** | extract → abstract | TextRank keeps the most central sentences that fit the model's 1024-token window, in their original order. The transformer then rewrites that text, so nothing has to be truncated blindly. |

All extractive methods share the same post-processing:

- **Redundancy filtering:** a sentence is skipped if its cosine similarity to an already-chosen sentence is 0.7 or higher.
- **Short-sentence penalty:** very short sentences are scored lower, because lines like "This brought new challenges." make poor summaries.
- **Original order:** selected sentences are returned in document order.

### Pipeline

```
document (.txt/.pdf/.docx)
   │  loaders.py: text extraction
   ▼
preprocess.py: clean (unicode, hyphenation, citations) → sentence split (abbreviations,
               initials, decimals, headings) → tokenise, stop words, Porter stemming
   │
   ├── extractive.py ── score sentences (Lead / Frequency / TextRank / LSA)
   │                     → redundancy filter → top-k in document order
   │
   ├── abstractive.py ── chunk to model window → BART/T5 beam search → reduce pass
   │
   └── hybrid.py ─────── TextRank selects sentences within the token budget → BART/T5
   ▼
metrics.py: ROUGE-1/2/L, compression ratio, novel-bigram % (how abstractive the output is)
```

## Setup

```bash
python3.12 -m venv .venv          # PyTorch needs Python ≤ 3.12/3.13
source .venv/bin/activate
pip install -r requirements.txt
```

The first abstractive run downloads the model from Hugging Face (DistilBART ≈ 1.2 GB, T5-small ≈ 240 MB). Extractive methods run fully offline. On Apple Silicon the model runs on the GPU through MPS. Set `SUMMARIZER_DEVICE=cpu` to override.

## Usage

**Web app**

```bash
streamlit run app.py
```

Paste text, upload a file or pick a sample, then choose a method. **Compare all methods** runs all six side by side. For extractive and hybrid results, the chosen sentences are highlighted in the source. If you paste a reference summary, the app also shows ROUGE scores.

**Command line**

```bash
python cli.py samples/urban_heat_islands.txt                       # TextRank, 20% of sentences
python cli.py report.pdf --method lsa --sentences 5
python cli.py essay.docx --method hybrid --model bart-large-cnn
python cli.py samples/history_of_the_internet.txt --method all     # compare every method
python cli.py article.txt --method all --reference reference.txt   # + ROUGE scores
```

**Python**

```python
from summarizer import summarize

result = summarize(text, method="textrank", num_sentences=4)
result = summarize(text, method="abstractive", model="distilbart", max_length=120)
print(result.text, result.compression, result.novel_bigrams)
```

## Evaluation

```bash
python scripts/evaluate.py -n 100
```

This scores every method on 100 articles taken at an even stride across the CNN/DailyMail test set. The references are the human-written highlights, and extractive methods select 3 sentences.

| Method | Type | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg words | Novel bigrams % | Sec/doc |
|---|---|---:|---:|---:|---:|---:|---:|
| Lead-3 | baseline | 41.44 | 20.13 | 37.66 | 76 | 0.0 | 0.00 |
| Frequency | extractive | 34.13 | 13.89 | 30.75 | 98 | 0.0 | 0.00 |
| TextRank | extractive | 36.11 | 15.27 | 32.59 | 78 | 0.0 | 0.01 |
| LSA | extractive | 32.41 | 12.49 | 29.35 | 69 | 0.0 | 0.01 |
| Abstractive (DistilBART) | abstractive | 43.32 | 21.45 | 40.41 | 57 | 12.0 | 2.21 |
| **Hybrid (TextRank → DistilBART)** | hybrid | **44.25** | **22.10** | **41.31** | 62 | 11.4 | 1.57 |

*ROUGE is F1 × 100, and ROUGE-L is the summary-level variant (rougeLsum). Timings are from an Apple M-series GPU (MPS).*

**Observations**

- **Abstractive beats extractive.** It scores about 8 ROUGE-1 above the best unsupervised extractive method (TextRank) while writing shorter summaries (57 vs 78 words). About 12% of its bigrams do not appear in the source, which shows it paraphrases rather than only copying.
- **The hybrid is the best and the fastest model-based method.**
  - Its score: compared with plain abstractive, it gains about +0.9 ROUGE-1, +0.7 ROUGE-2 and +0.9 ROUGE-L. TextRank removes peripheral sentences before the model sees them, so the 1024-token window holds the most central content.
  - Its speed: it is about 30% faster, because the long-document map-reduce passes are no longer needed.
- **Lead-3 is hard to beat among extractive methods on news.** Journalists put the key facts first, and none of the unsupervised methods beat it here. TextRank is the best of them. LSA picks sentences that are distinct but less central.
- **The numbers match published work.** See et al. (2017) report Lead-3 at 40.3 / 17.7 / 36.6 on the full test set. DistilBART-12-6's model card reports 44.2 ROUGE-1.
- **Sampling matters.** The test split stores all CNN articles before the DailyMail ones, and CNN articles are harder. On the first 100 articles (all CNN; run with `--first`, results in `results/rouge_cnn_only_first100.md`), Lead-3 drops to 30.6 ROUGE-1 and the hybrid to 35.8. The ranking of methods stays the same.

## Tests

```bash
pytest -q
```

There are 27 tests. They cover:

- sentence splitting edge cases (abbreviations, initials, decimals, headings);
- each extractive algorithm and the redundancy filter;
- the metrics and the file loaders;
- CLI argument validation (out-of-range lengths, empty documents);
- the abstractive and hybrid paths, which run only when T5-small is downloaded.

## Project structure

```
summarizer/
  preprocess.py    cleaning, sentence segmentation, tokenisation, stemming
  extractive.py    Lead, Frequency, TextRank, LSA + shared selection logic
  abstractive.py   BART/T5 wrapper with chunked map-reduce summarisation
  hybrid.py        TextRank → transformer pipeline
  metrics.py       ROUGE, compression ratio, novel n-gram ratio
  loaders.py       .txt / .md / .pdf / .docx text extraction
app.py             Streamlit UI
cli.py             command-line interface
scripts/evaluate.py  CNN/DailyMail ROUGE benchmark
tests/             pytest suite
samples/           example documents
results/           benchmark output
```

## Limitations

- **Hallucination:** abstractive models can merge facts incorrectly. On the bundled internet-history sample, the hybrid summary says "By 1991, CERN released the World Wide Web…". The document says the browser was built by 1991 and the release came in 1993. Extractive output is always faithful to the source because it copies sentences.
- **Domain:** DistilBART and BART-large-CNN were fine-tuned on news, so they write news-style leads. On scientific or legal text they tend to copy opening sentences rather than abstract.
- **T5-small** is fast but weak. It lowercases proper nouns and is the least fluent of the models.
- **Scanned PDFs** need OCR first. Only PDFs with a text layer are supported.

## References

- Luhn, H. P. (1958). *The automatic creation of literature abstracts.* IBM Journal of R&D.
- Mihalcea, R. & Tarau, P. (2004). *TextRank: Bringing order into texts.* EMNLP.
- Steinberger, J. & Ježek, K. (2004). *Using latent semantic analysis in text summarization and summary evaluation.* ISIM.
- Lewis, M. et al. (2020). *BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension.* ACL.
- Raffel, C. et al. (2020). *Exploring the limits of transfer learning with a unified text-to-text transformer (T5).* JMLR.
- Lin, C.-Y. (2004). *ROUGE: A package for automatic evaluation of summaries.* ACL Workshop.
- See, A., Liu, P. & Manning, C. (2017). *Get to the point: Summarization with pointer-generator networks* (CNN/DailyMail). ACL.
