"""Retrieve -> (relevance gate) -> grounded generation with page citations."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import Settings
from .llm import ChatLLM

IDK = "I don't know."

SYSTEM_PROMPT = """You answer questions about the documents provided in the context.
Rules:
- Use ONLY the context below. Do not use outside knowledge.
- Cite the page for every fact, like [p. 7]. Use the page labels shown in the context.
- If the context does not contain the answer, reply exactly: I don't know.
- Be concise. For numbers, quote them exactly as written.

Context:
{context}"""


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict] = field(default_factory=list)
    top_score: float | None = None

    @property
    def pages(self) -> list[int]:
        return sorted({s["page"] for s in self.sources})


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source', 'doc')} | p. {d.metadata.get('page')}]\n{d.page_content}" for d in docs
    )


class RAGPipeline:
    def __init__(self, store: FAISS, llm: ChatLLM | None, settings: Settings):
        self.store = store
        self.llm = llm
        self.s = settings

    def retrieve(self, question: str, k: int | None = None) -> tuple[list[Document], float | None]:
        k = k or self.s.top_k
        scored = self.store.similarity_search_with_score(question, k=max(k, 1))
        top = float(scored[0][1]) if scored else None
        if self.s.search_type == "mmr":
            docs = self.store.max_marginal_relevance_search(question, k=k, fetch_k=max(self.s.fetch_k, k))
        else:
            docs = [d for d, _ in scored[:k]]
        return docs, top

    def ask(self, question: str, k: int | None = None) -> Answer:
        docs, top = self.retrieve(question, k)
        sources = [
            {"source": d.metadata.get("source"), "page": d.metadata.get("page"), "snippet": d.page_content[:200]}
            for d in docs
        ]
        if not docs or (top is not None and top < self.s.min_relevance):
            return Answer(question, IDK, [], top)
        if self.llm is None:
            raise RuntimeError("No LLM configured; use retrieve() for retrieval-only mode.")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=format_context(docs))},
            {"role": "user", "content": question},
        ]
        return Answer(question, self.llm(messages).strip(), sources, top)
