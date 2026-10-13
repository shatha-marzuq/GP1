"""Run one review from the command line.

    cd backend
    python -m orchestrator.run path/to/terms.pdf            # real agents (needs BGE-M3, Chroma, MITHAQ_LLM key)
    python -m orchestrator.run path/to/terms.pdf --stub     # real OCR + segmentation, stand-in retrieval and LLM
"""
import argparse
import json
import uuid

from . import paths  # noqa: F401
from .graph import build_graph
from .stubs import StubRetriever, stub_verifier
from .verifier import LLMVerifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--stub", action="store_true", help="stand-in retriever and verifier")
    ap.add_argument("--out", default="report.json")
    a = ap.parse_args()

    if a.stub:
        retriever, verifier = StubRetriever(), stub_verifier
    else:
        from retriever import RegulationRetriever
        retriever, verifier = RegulationRetriever(), LLMVerifier.from_env()

    graph = build_graph(retriever, verifier)
    state = graph.invoke({"review_id": uuid.uuid4().hex[:8], "file_path": a.file, "findings": []},
                         {"max_concurrency": 8})
    if state.get("report"):
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(state["report"], f, ensure_ascii=False, indent=1)
        print("counts:", state["report"]["counts"], "->", a.out)


if __name__ == "__main__":
    main()
