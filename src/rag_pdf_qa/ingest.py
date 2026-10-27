"""Load PDFs -> clean -> chunk -> embed -> persist a FAISS index."""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .embeddings import build_embeddings


def load_pdfs(paths: list[str | Path]) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("*.pdf")) if p.is_dir() else [p]
        for f in files:
            for d in PyPDFLoader(str(f)).load():
                d.page_content = _clean(d.page_content)
                if not d.page_content:
                    continue
                d.metadata = {
                    "source": f.name,
                    "page": int(d.metadata.get("page", 0)) + 1,  # 1-based, as a reader sees it
                }
                docs.append(d)
    if not docs:
        raise FileNotFoundError(f"No readable PDF pages found in: {paths}")
    return docs


def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split(docs: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def build_index(paths: list[str | Path], settings: Settings) -> FAISS:
    chunks = split(load_pdfs(paths), settings)
    store = FAISS.from_documents(
        chunks,
        build_embeddings(settings),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # vectors are normalized -> cosine
    )
    out = Path(settings.index_dir)
    store.save_local(str(out))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "sources": sorted({c.metadata["source"] for c in chunks}),
                "chunks": len(chunks),
                "embedding_backend": settings.embedding_backend,
                "embedding_model": settings.embedding_model,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
            indent=2,
        )
    )
    return store


def load_index(settings: Settings) -> FAISS:
    path = Path(settings.index_dir)
    if not (path / "index.faiss").exists():
        raise FileNotFoundError(f"No index at {path}/. Run `rag-pdf-qa ingest <pdf>` first.")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["embedding_backend"] != settings.embedding_backend:
        raise ValueError(
            f"Index was built with embedding_backend={manifest['embedding_backend']!r}; "
            f"current setting is {settings.embedding_backend!r}. Re-ingest or change the setting."
        )
    # The index is created locally by this tool, so loading its pickle is safe.
    return FAISS.load_local(
        str(path),
        build_embeddings(settings),
        allow_dangerous_deserialization=True,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
