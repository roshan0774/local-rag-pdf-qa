"""Golden-set evaluation.

Each JSONL row: {"question": str, "pages": [int], "answer_contains": [str], "answerable": bool}

Retrieval metrics (no LLM needed):
  hit@k  - share of answerable questions where an expected page is in the top-k chunks
  mrr    - mean reciprocal rank of the first chunk from an expected page
Answer metrics (with LLM):
  answer_match - share of answerable questions whose answer contains every expected string
  idk_correct  - share of unanswerable questions answered "I don't know."
  citation_ok  - share of answerable questions whose answer cites at least one expected page
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .pipeline import IDK, RAGPipeline


def load_golden(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def evaluate(pipe: RAGPipeline, golden: list[dict], k: int, with_llm: bool) -> dict:
    rows, hits, rr = [], [], []
    ans_ok, idk_ok, cite_ok = [], [], []
    for g in golden:
        docs, top = pipe.retrieve(g["question"], k)
        pages = [d.metadata.get("page") for d in docs]
        row = {"question": g["question"], "retrieved_pages": pages, "top_score": round(top or 0, 3)}
        if g.get("answerable", True):
            exp = set(g["pages"])
            rank = next((i + 1 for i, p in enumerate(pages) if p in exp), None)
            hits.append(rank is not None)
            rr.append(1 / rank if rank else 0.0)
            row["hit"] = rank is not None
        if with_llm:
            a = pipe.ask(g["question"], k)
            row["answer"] = a.answer
            if g.get("answerable", True):
                ok = all(_norm(x) in _norm(a.answer) for x in g.get("answer_contains", []))
                ans_ok.append(ok)
                cited = {int(p) for p in re.findall(r"p\.\s*(\d+)", a.answer)}
                cite_ok.append(bool(cited & set(g["pages"])))
                row["answer_match"] = ok
            else:
                ok = _norm(IDK).rstrip(".") in _norm(a.answer)
                idk_ok.append(ok)
                row["idk_correct"] = ok
        rows.append(row)

    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None

    summary = {"n": len(golden), "k": k, f"hit@{k}": mean(hits), "mrr": mean(rr)}
    if with_llm:
        summary.update(answer_match=mean(ans_ok), citation_ok=mean(cite_ok), idk_correct=mean(idk_ok))
    return {"summary": summary, "rows": rows}
