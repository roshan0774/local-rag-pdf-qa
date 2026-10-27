"""Central configuration. Every field can be overridden with an env var (RAG_<FIELD>)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _env(name: str, default):
    raw = os.getenv(f"RAG_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Settings:
    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Embeddings: "hf" (sentence-transformers, default) or "hash" (offline, no download; for CI/tests)
    embedding_backend: str = "hf"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    # BGE v1.5 retrieves better when queries carry this instruction prefix
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # Retrieval
    top_k: int = 4
    search_type: str = "mmr"  # "similarity" or "mmr" (diverse results, fewer near-duplicate chunks)
    fetch_k: int = 20
    # Cosine-similarity floor. If the best chunk scores below it, return "I don't know." without
    # calling the LLM. 0 disables the gate; tune it with `rag-pdf-qa eval` on your own documents.
    min_relevance: float = 0.0

    # Generation
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 300

    # Storage
    index_dir: str = "index"

    device: str = field(default="auto")

    @classmethod
    def from_env(cls) -> Settings:
        base = cls()
        return cls(**{f.name: _env(f.name, getattr(base, f.name)) for f in fields(cls)})


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
