"""Embedding backends.

- ``hf``: sentence-transformers model (default ``BAAI/bge-base-en-v1.5``), normalized vectors,
  with the BGE query instruction applied to queries only.
- ``hash``: deterministic hashed bag-of-words vectors. No download, no GPU. Used by the test
  suite and CI so the pipeline can be exercised end to end offline.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re

from langchain_core.embeddings import Embeddings

from .config import Settings, resolve_device


class HashingEmbeddings(Embeddings):
    """Tiny lexical embedding: token + bigram hashing into a fixed-size, L2-normalized vector."""

    _token = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        toks = self._token.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in itertools.pairwise(toks)]
        vec = [0.0] * self.dim
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class InstructedEmbeddings(Embeddings):
    """Wraps an embedding model and prefixes queries (not documents) with an instruction."""

    def __init__(self, inner: Embeddings, query_instruction: str = ""):
        self.inner = inner
        self.query_instruction = query_instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self.query_instruction + text)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_backend == "hash":
        return HashingEmbeddings()
    if settings.embedding_backend == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings

        hf = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": resolve_device(settings.device)},
            encode_kwargs={"normalize_embeddings": True},
        )
        return InstructedEmbeddings(hf, settings.query_instruction)
    raise ValueError(f"Unknown embedding_backend: {settings.embedding_backend!r}")
