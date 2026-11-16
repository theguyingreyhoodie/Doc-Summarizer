"""Abstractive summarisation with pretrained encoder-decoder transformers.

The model *generates* new sentences rather than copying them. Pretrained
seq2seq models (BART, T5) have a fixed input window (1024 / 512 tokens), so
long documents are handled by map-reduce:

1. split the document into sentence-aligned chunks that fit the window,
2. summarise each chunk ("map"),
3. if the joined chunk summaries are still too long, summarise them again
   ("reduce"), repeating until the text fits one final pass.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from .preprocess import split_sentences


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    max_input_tokens: int
    prefix: str = ""
    description: str = ""


MODELS = {
    "distilbart": ModelSpec(
        "sshleifer/distilbart-cnn-12-6", 1024,
        description="DistilBART fine-tuned on CNN/DailyMail (306M params) - good quality/speed balance",
    ),
    "bart-large-cnn": ModelSpec(
        "facebook/bart-large-cnn", 1024,
        description="BART-large fine-tuned on CNN/DailyMail (406M params) - best quality, slowest",
    ),
    "t5-small": ModelSpec(
        "google-t5/t5-small", 512, prefix="summarize: ",
        description="T5-small multi-task model (60M params) - fastest, lowest quality",
    ),
}
DEFAULT_MODEL = "distilbart"


def pick_device() -> str:
    import torch

    forced = os.environ.get("SUMMARIZER_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=2)
def _load(model_key: str, device: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.utils import logging as hf_logging

    # Silence the "sequence longer than max length" notice from count_tokens();
    # long inputs are chunked before they ever reach the model.
    hf_logging.set_verbosity_error()

    spec = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(spec.hf_id).to(device).eval()
    return tokenizer, model


class AbstractiveSummarizer:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None):
        if model not in MODELS:
            raise ValueError(f"Unknown model {model!r}; choose from {list(MODELS)}")
        self.model_key = model
        self.spec = MODELS[model]
        self.device = device or pick_device()
        self.tokenizer, self.model = _load(model, self.device)
        # Leave headroom for the task prefix and special tokens.
        self.chunk_tokens = self.spec.max_input_tokens - 24

    # -- public API -----------------------------------------------------
    def summarize(
        self,
        text: str,
        max_length: int = 142,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        max_rounds: int = 3,
    ) -> str:
        sentences = split_sentences(text, min_words=1)
        if not sentences:
            return ""
        gen = dict(num_beams=num_beams, length_penalty=length_penalty,
                   no_repeat_ngram_size=no_repeat_ngram_size, early_stopping=True)

        current = sentences
        for _ in range(max_rounds):
            chunks = self.chunk(current)
            if len(chunks) == 1:
                break
            # Map step: shorter partial summaries keep the reduce input small.
            partials = self._generate(chunks, max_length=max(60, max_length // 2 + 40),
                                      min_length=min(min_length, 20), **gen)
            current = split_sentences(" ".join(partials), min_words=1)
        else:
            # Still too long after max_rounds: keep what fits in one window.
            current = split_sentences(self.chunk(current)[0], min_words=1)

        final_input = " ".join(current)
        n_tokens = self.count_tokens(final_input)
        # Never ask for a summary longer than its source.
        max_len = max(16, min(max_length, int(n_tokens * 0.9)))
        min_len = min(min_length, max_len - 1)
        return self._generate([final_input], max_length=max_len, min_length=min_len, **gen)[0]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def chunk(self, sentences: list[str]) -> list[str]:
        """Greedily pack whole sentences into chunks that fit the input window."""
        chunks, buf, buf_len = [], [], 0
        for sentence in sentences:
            n = self.count_tokens(sentence)
            if n > self.chunk_tokens:  # pathological sentence: hard-truncate it
                ids = self.tokenizer(sentence, add_special_tokens=False)["input_ids"][: self.chunk_tokens]
                sentence, n = self.tokenizer.decode(ids), self.chunk_tokens
            if buf and buf_len + n > self.chunk_tokens:
                chunks.append(" ".join(buf))
                buf, buf_len = [], 0
            buf.append(sentence)
            buf_len += n
        if buf:
            chunks.append(" ".join(buf))
        return chunks

    # -- internals ------------------------------------------------------
    def _generate(self, texts: list[str], batch_size: int = 4, **gen_kwargs) -> list[str]:
        import torch

        outputs: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = [self.spec.prefix + t for t in texts[i : i + batch_size]]
            enc = self.tokenizer(batch, max_length=self.spec.max_input_tokens, truncation=True,
                                 padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                ids = self.model.generate(**enc, **gen_kwargs)
            outputs.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return [_tidy(o) for o in outputs]


def _tidy(text: str) -> str:
    text = " ".join(text.split())
    # CNN/DailyMail-trained models emit "word ." - drop the space before punctuation.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # T5 tends to emit lower-case sentence starts; capitalise them.
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    return text[:1].upper() + text[1:]
