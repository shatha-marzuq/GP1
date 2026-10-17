"""MITHAQ Orchestrator: a fixed LangGraph pipeline from a T&C file to retrieved CMA regulations.

    extract ─► segment ─► retrieve × N (parallel, one per clause) ─► collect
       └──────────┴─► failed

extract  = Document Processing Agent (docproc OCR, items with page numbers)
segment  = segmentation.segment_clauses (paragraphs -> clauses)
retrieve = Regulatory Retrieval Agent (pipeline/retriever.py over the Chroma vector DB)

Every agent is passed in, so tests and demos can swap in stand-ins.
The compliance verdict (LLM) is not part of this graph yet; it will consume the `results` this graph returns.
"""
from __future__ import annotations

import threading
from collections import Counter
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
# شكل البيانات نجيبه من ملف ال state 
from .state import ClauseTask, ReviewState

#  بعدين لما نسوي FastAPI. بنبدلها بدالة تحفظ الحالة في قاعدة البيانات، والواجهة تعرض للمستخدم "جاري الاستخراج..." مثلاً 
def print_status(review_id: str, status: str, **info):
    print(f"[{review_id}] {status}", info or "")

# ocr شذي يربطه مع الاوركسترا
def docproc_extract(path: str) -> dict:
    """Document Processing Agent: the team's docproc pipeline (PDF/Word -> items with page numbers)."""
    from . import paths  # noqa: F401
    from mithaq_docproc import build_output, process_document
    return build_output(process_document(path), path)

# نفس الشي لكود جواهر حق التقسيم 
def default_segment(items: list[dict]) -> list[dict]:
    from . import paths  # noqa: F401
    from segmentation import segment_clauses
    return segment_clauses(items)


def build_graph(retriever, *,
                extract: Callable[[str], dict] = docproc_extract,
                segment: Callable[[list[dict]], list[dict]] = default_segment,
                on_status: Callable[..., None] = print_status,
                k: int = 5, search_kwargs: dict | None = None, checkpointer=None):
    """retriever: anything with .search(query, k=…, **search_kwargs) -> hits like pipeline/retriever.py.
    search_kwargs: passed to every search, e.g. {"mode": "hybrid"} (retriever defaults: dense,
    fund_type="private", rerank=True)."""
    search_kwargs = search_kwargs or {}
    done, lock = Counter(), threading.Lock()

    def extract_node(state: ReviewState):
        on_status(state["review_id"], "extracting")
        try:
            out = extract(state["file_path"])
        except Exception as e:  # docproc raises DocProcError with an Arabic message
            return {"error": getattr(e, "message_ar", None) or str(e)}
        return {"document": out["document"], "items": out["items"]}

    def segment_node(state: ReviewState):
        on_status(state["review_id"], "segmenting")
        return {"clauses": segment(state["items"])}

    def failed_node(state: ReviewState):
        on_status(state["review_id"], "failed", reason=state.get("error") or "no text or clauses found")
        return {}

    def after_extract(state: ReviewState):
        return "failed" if state.get("error") or not state.get("items") else "segment"
# بعد السقمنت هنا نبدا نرسل البند مع  ال retrieve
    def fan_out(state: ReviewState):
        clauses = state.get("clauses") or []
        if not clauses:
            return "failed"
        on_status(state["review_id"], "retrieving", done=0, total=len(clauses))
        return [Send("retrieve", {"review_id": state["review_id"], "clause": c}) for c in clauses]
#تاخذ البند، وتبحث بنصه في قاعدة الأنظمة  كود الرتريف
    def retrieve_node(task: ClauseTask):
        clause = task["clause"]
        hits = retriever.search(clause["text"], k=k, **search_kwargs)
        with lock:
            done[task["review_id"]] += 1
            n = done[task["review_id"]]
        on_status(task["review_id"], "retrieving", done=n)
        return {"results": [_result(clause, hits)]}

    def collect_node(state: ReviewState):
        rid = state["review_id"]
        order = {c["clause_id"]: n for n, c in enumerate(state["clauses"])}
        results = sorted(state["results"], key=lambda r: order[r["clause_id"]])
        doc = state.get("document", {})
        output = {
            "review_id": rid,
            "filename": doc.get("filename"),
            "document_needs_review": bool(doc.get("needs_review")),
            "document_warnings": doc.get("warnings", []),
            "clauses": results,
        }
        on_status(rid, "done", clauses=len(results))
        return {"output": output}
# سوينا قراف جديد
    g = StateGraph(ReviewState)
    g.add_node("extract", extract_node)
    g.add_node("segment", segment_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("collect", collect_node)
    g.add_node("failed", failed_node)
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", after_extract, ["segment", "failed"])
    g.add_conditional_edges("segment", fan_out, ["retrieve", "failed"])
    g.add_edge("retrieve", "collect")  # runs once, after every branch has finished
    g.add_edge("collect", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)

# نرتب النواتج 
def _result(clause: dict, hits: list[dict]) -> dict:
    return {
        "clause_id": clause["clause_id"],
        "title": clause.get("title"),
        "page_start": clause.get("page_start"),
        "page_end": clause.get("page_end"),
        "from_ocr": clause.get("from_ocr"),
        "text": clause["text"],
        "regulations": [{"id": h["id"], "score": h["score"], "citation": h["citation"],
                         "pdf_pages": h.get("pdf_pages"), "text": h["text"]} for h in hits],
    }
