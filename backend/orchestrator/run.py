"""Run one document through OCR, segmentation and regulation retrieval from the command line.

    cd backend
    python -m orchestrator.run path/to/terms.pdf            # real retrieval (needs BGE-M3 and the Chroma store)
    python -m orchestrator.run path/to/terms.pdf --stub     # real OCR + segmentation, stand-in retrieval
    python -m orchestrator.run path/to/terms.pdf --agent    # retrieval through the Regulatory Retrieval Agent
    python -m orchestrator.run path/to/terms.pdf --agent --llm   # … with its LLM step (OPENAI_API_KEY in .env)
"""
import argparse
import json
import uuid

from . import paths  # noqa: F401
from .graph import build_graph
from .stubs import StubRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--stub", action="store_true", help="stand-in retriever")
    ap.add_argument("--agent", action="store_true", help="Regulatory Retrieval Agent (pipeline/retrieval_agent.py)")
    ap.add_argument("--llm", action="store_true", help="with --agent: add the LLM step")
    ap.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--out", default="retrieval.json")
    a = ap.parse_args()

    if a.stub:
        retriever = StubRetriever()
    else:
        from retriever import RegulationRetriever
        retriever = RegulationRetriever()
        if a.agent:
            from retrieval_agent import LLM, RetrievalAgent
            retriever = RetrievalAgent(retriever, llm=LLM() if a.llm else None)

    graph = build_graph(retriever, k=a.k, search_kwargs={"mode": a.mode})
    state = graph.invoke({"review_id": uuid.uuid4().hex[:8], "file_path": a.file, "results": []},
                         {"max_concurrency": 8})
    if state.get("output"):
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(state["output"], f, ensure_ascii=False, indent=1)
        print("clauses:", len(state["output"]["clauses"]), "->", a.out)


if __name__ == "__main__":
    main()
