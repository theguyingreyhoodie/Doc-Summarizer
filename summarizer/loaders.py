"""Read plain text out of .txt, .md, .pdf and .docx documents."""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED = (".txt", ".md", ".pdf", ".docx")


def load_document(source: str | Path | bytes, filename: str | None = None) -> str:
    """Return the text of a document given a path, or raw bytes plus a filename."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        data, suffix = path.read_bytes(), path.suffix.lower()
    else:
        if not filename:
            raise ValueError("filename is required when passing raw bytes")
        data, suffix = source, Path(filename).suffix.lower()

    if suffix in (".txt", ".md"):
        return _decode(data)
    if suffix == ".pdf":
        return _read_pdf(data)
    if suffix == ".docx":
        return _read_docx(data)
    raise ValueError(f"Unsupported file type {suffix!r}; supported: {', '.join(SUPPORTED)}")


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise ValueError("No extractable text found in the PDF (it may be a scanned image).")
    return text


def _read_docx(data: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(data))
    return "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
