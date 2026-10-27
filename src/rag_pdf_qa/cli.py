"""Command line: ingest | ask | chat | eval."""

from __future__ import annotations

import argparse
import json
import sys

from .config import Settings


def _pipeline(settings: Settings, need_llm: bool = True):
    from .ingest import load_index
    from .pipeline import RAGPipeline

    store = load_index(settings)
    llm = None
    if need_llm:
        from .llm import HFChatLLM

        print(f"Loading {settings.llm_model} ...", file=sys.stderr)
        llm = HFChatLLM(settings)
    return RAGPipeline(store, llm, settings)


def _print_answer(a) -> None:
    print(f"\nAnswer:\n{a.answer}\n")
    if a.sources:
        print("Sources:")
        for s in a.sources:
            print(f"  - {s['source']} p. {s['page']}: {s['snippet'][:90].replace(chr(10), ' ')}...")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag-pdf-qa", description="Local, private RAG question answering over PDFs.")
    p.add_argument("--index-dir", help="Where the FAISS index lives (default: ./index)")
    p.add_argument("--embedding-backend", choices=["hf", "hash"], help="hf = BGE model, hash = offline test mode")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="Index one or more PDFs (or folders of PDFs)")
    i.add_argument("paths", nargs="+")

    a = sub.add_parser("ask", help="Ask one question")
    a.add_argument("question")
    a.add_argument("-k", type=int)
    a.add_argument("--json", action="store_true", help="Print the result as JSON")
    a.add_argument("--retrieval-only", action="store_true", help="Show retrieved chunks; skip the LLM")

    sub.add_parser("chat", help="Interactive Q&A loop")

    e = sub.add_parser("eval", help="Score retrieval (and answers) against a golden JSONL set")
    e.add_argument("golden")
    e.add_argument("-k", type=int)
    e.add_argument("--with-llm", action="store_true", help="Also generate and score answers")
    e.add_argument("--out", help="Write per-question results to this JSON file")

    args = p.parse_args(argv)
    s = Settings.from_env()
    if args.index_dir:
        s.index_dir = args.index_dir
    if args.embedding_backend:
        s.embedding_backend = args.embedding_backend

    if args.cmd == "ingest":
        from .ingest import build_index

        store = build_index(args.paths, s)
        print(f"Indexed {store.index.ntotal} chunks -> {s.index_dir}/")
        return 0

    if args.cmd == "ask":
        pipe = _pipeline(s, need_llm=not args.retrieval_only)
        if args.retrieval_only:
            docs, top = pipe.retrieve(args.question, args.k)
            print(f"top cosine score: {top:.3f}")
            for d in docs:
                print(f"\n--- {d.metadata['source']} p. {d.metadata['page']} ---\n{d.page_content}")
            return 0
        ans = pipe.ask(args.question, args.k)
        if args.json:
            print(json.dumps({**ans.__dict__}, indent=2))
        else:
            _print_answer(ans)
        return 0

    if args.cmd == "chat":
        pipe = _pipeline(s)
        print("Ask about your documents. Empty line or Ctrl-D to quit.")
        while True:
            try:
                q = input("\nYou: ").strip()
            except EOFError:
                break
            if not q:
                break
            _print_answer(pipe.ask(q))
        return 0

    if args.cmd == "eval":
        from .evaluate import evaluate, load_golden

        pipe = _pipeline(s, need_llm=args.with_llm)
        res = evaluate(pipe, load_golden(args.golden), args.k or s.top_k, args.with_llm)
        print(json.dumps(res["summary"], indent=2))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(res, f, indent=2)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
