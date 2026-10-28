"""Offline tests: hashing embeddings + a fake LLM, so CI needs no model downloads or GPU."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_pdf import build  # noqa: E402

from rag_pdf_qa.cli import main  # noqa: E402
from rag_pdf_qa.config import Settings  # noqa: E402
from rag_pdf_qa.evaluate import evaluate, load_golden  # noqa: E402
from rag_pdf_qa.ingest import build_index, load_index, load_pdfs, split  # noqa: E402
from rag_pdf_qa.pipeline import IDK, RAGPipeline  # noqa: E402


@pytest.fixture(scope="session")
def pdf(tmp_path_factory):
    return build(tmp_path_factory.mktemp("data") / "sample.pdf")


@pytest.fixture
def settings(tmp_path):
    return Settings(embedding_backend="hash", index_dir=str(tmp_path / "index"), search_type="similarity")


class EchoLLM:
    """Fake LLM: records the prompt and 'answers' with the first context line it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return "stub answer [p. 3]"


def test_load_pdfs_uses_one_based_pages(pdf):
    docs = load_pdfs([pdf])
    assert [d.metadata["page"] for d in docs] == list(range(1, 9))
    assert all(d.metadata["source"] == "sample.pdf" for d in docs)


def test_split_assigns_chunk_ids(pdf, settings):
    chunks = split(load_pdfs([pdf]), settings)
    assert len(chunks) >= 8
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)


def test_index_roundtrip_and_manifest(pdf, settings):
    store = build_index([pdf], settings)
    manifest = json.loads((Path(settings.index_dir) / "manifest.json").read_text())
    assert manifest["chunks"] == store.index.ntotal
    assert load_index(settings).index.ntotal == store.index.ntotal


def test_load_index_rejects_backend_mismatch(pdf, settings):
    build_index([pdf], settings)
    settings.embedding_backend = "hf"
    with pytest.raises(ValueError, match="embedding_backend"):
        load_index(settings)


def test_retrieval_finds_right_page(pdf, settings):
    pipe = RAGPipeline(build_index([pdf], settings), None, settings)
    docs, top = pipe.retrieve("How many claims were processed in FY2026?", k=3)
    assert 4 in [d.metadata["page"] for d in docs]
    assert top is not None and top > 0


def test_ask_builds_grounded_prompt_with_page_labels(pdf, settings):
    llm = EchoLLM()
    pipe = RAGPipeline(build_index([pdf], settings), llm, settings)
    ans = pipe.ask("What was net income?")
    system = llm.calls[0][0]["content"]
    assert "Use ONLY the context" in system and "p. " in system
    assert ans.sources and ans.answer == "stub answer [p. 3]"


def test_relevance_gate_skips_llm(pdf, settings):
    settings.min_relevance = 0.99
    llm = EchoLLM()
    ans = RAGPipeline(build_index([pdf], settings), llm, settings).ask("What is the capital of Peru?")
    assert ans.answer == IDK and not llm.calls


def test_mmr_search_returns_k_docs(pdf, settings):
    settings.search_type = "mmr"
    docs, _ = RAGPipeline(build_index([pdf], settings), None, settings).retrieve("combined ratio", k=3)
    assert len(docs) == 3


def test_golden_retrieval_hit_rate(pdf, settings):
    pipe = RAGPipeline(build_index([pdf], settings), None, settings)
    res = evaluate(pipe, load_golden(ROOT / "eval" / "golden_sample.jsonl"), k=4, with_llm=False)
    # Even the crude offline hashing embedder should find the right page most of the time.
    assert res["summary"]["hit@4"] >= 0.75


def test_cli_ingest_and_retrieval_only(pdf, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RAG_SEARCH_TYPE", "similarity")
    idx = str(tmp_path / "cli_index")
    assert main(["--index-dir", idx, "--embedding-backend", "hash", "ingest", str(pdf)]) == 0
    assert main(["--index-dir", idx, "--embedding-backend", "hash", "ask", "--retrieval-only", "-k", "2",
                 "Who is the chief executive officer?"]) == 0
    assert "Dana Whitfield" in " ".join(capsys.readouterr().out.split())
