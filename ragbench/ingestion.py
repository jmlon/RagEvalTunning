"""Corpus ingestion: source files in inputs/ -> normalized Documents.

v1 supports PDF via plain-text extraction. The loader dispatch table is keyed
by file extension so new formats can be added without touching RAG systems.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from ragbench.models import Document

logger = logging.getLogger(__name__)


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


#: extension -> text extractor. Add new formats here.
LOADERS: dict[str, Callable[[Path], str]] = {
    ".pdf": _load_pdf,
}


def load_corpus(inputs_dir: str | Path) -> list[Document]:
    """Load every supported file under inputs_dir into a list of Documents."""
    inputs_dir = Path(inputs_dir)
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Inputs directory not found: {inputs_dir}")

    documents: list[Document] = []
    for path in sorted(inputs_dir.iterdir()):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.info("Skipping unsupported file: %s", path.name)
            continue
        text = loader(path).strip()
        if not text:
            logger.warning("No text extracted from %s; skipping.", path.name)
            continue
        documents.append(Document(id=path.stem, text=text, source=path.name))

    if not documents:
        raise RuntimeError(f"No supported documents found in {inputs_dir}")
    logger.info("Loaded %d document(s) from %s", len(documents), inputs_dir)
    return documents
