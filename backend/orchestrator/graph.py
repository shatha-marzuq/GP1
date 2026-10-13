"""MITHAQ Orchestrator: a fixed LangGraph pipeline.

    extract ─► segment ─► review_clause × N (parallel, one per clause) ─► report
       └──────────┴─► failed

The order of steps never changes. The LLM makes decisions only inside review_clause
(the verdict, and whether to search again with a better query).
Every agent is passed in, so tests and demos can swap in stand-ins.
"""
from __future__ import annotations

import threading
from collections import Counter
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .state import ClauseTask, ReviewState
from .verifier import Verdict, check_verdict, label

STATUS_ORDER = {"Violation": 0, "Needs Review": 1, "Compliant": 2}


def print_status(review_id: str, status: str, **info):
    print(f"[{review_id}] {status}", info or "")


def docproc_extract(path: str) -> dict:
    """Document Processing Agent: the team's docproc pipeline (PDF/Word -> items with page numbers)."""
    from . import paths  # noqa: F401
    from mithaq_docproc import build_output, process_document
    return build_output(process_document(path), path)


def default_segment(items: list[dict]) -> list[dict]:
    from . import paths  # noqa: F401
    from segmentation import segment_clauses
    return segment_clauses(items)


def build_graph(retriever, verifier: Callable[[dict, list[dict]], Verdict], *,
                extract: Callable[[str], dict] = docproc_extract,
                segment: Callable[[list[dict]], list[dict]] = default_segment,
                on_status: Callable[..., None] = print_status,
                k: int = 5, min_score: float | None = None, max_retries: int = 2,
                checkpointer=None):
    """retriever: anything with .search(query, k=…) -> hits like pipeline/retriever.py.
    min_score: below this top score the clause skips the LLM and becomes Needs Review.
    Leave None until it is calibrated on the BLOM test clauses (reranker scores are logits)."""
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

    def fan_out(state: ReviewState):
        clauses = state.get("clauses") or []
        if not clauses:
            return "failed"
        on_status(state["review_id"], "verifying", done=0, total=len(clauses))
        flag = bool(state.get("document", {}).get("needs_review"))
        return [Send("review_clause", {"review_id": state["review_id"], "clause": c, "doc_needs_review": flag})
                for c in clauses]

    def search(query: str) -> list[dict]:
        return retriever.search(query, k=k)

    def review_clause(task: ClauseTask):
        clause = task["clause"]
        hits = search(clause["text"])
        for attempt in range(max_retries + 1):
            if not hits or (min_score is not None and hits[0]["score"] < min_score):
                finding = _finding(clause, hits, None, ["no matching regulation found"])
                break
            verdict = verifier(clause, hits)
            if verdict.needs_more_context and verdict.suggested_query and attempt < max_retries:
                hits = _merge(search(verdict.suggested_query), hits, k)
                continue
            finding = _judge(clause, hits, verdict, task["doc_needs_review"])
            break
        with lock:
            done[task["review_id"]] += 1
            n = done[task["review_id"]]
        on_status(task["review_id"], "verifying", done=n)
        return {"findings": [finding]}

    def report_node(state: ReviewState):
        rid = state["review_id"]
        on_status(rid, "reporting")
        order = {c["clause_id"]: n for n, c in enumerate(state["clauses"])}
        findings = sorted(state["findings"], key=lambda f: (STATUS_ORDER[f["status"]], order[f["clause_id"]]))
        doc = state.get("document", {})
        report = {
            "review_id": rid,
            "filename": doc.get("filename"),
            "counts": {s: sum(f["status"] == s for f in findings) for s in STATUS_ORDER},
            "document_needs_review": bool(doc.get("needs_review")),
            "document_warnings": doc.get("warnings", []),
            "findings": findings,
        }
        on_status(rid, "done", counts=report["counts"])
        return {"report": report}

    g = StateGraph(ReviewState)
    g.add_node("extract", extract_node)
    g.add_node("segment", segment_node)
    g.add_node("review_clause", review_clause)
    g.add_node("report", report_node)
    g.add_node("failed", failed_node)
    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", after_extract, ["segment", "failed"])
    g.add_conditional_edges("segment", fan_out, ["review_clause", "failed"])
    g.add_edge("review_clause", "report")  # runs once, after every branch has finished
    g.add_edge("report", END)
    g.add_edge("failed", END)
    return g.compile(checkpointer=checkpointer)


def _merge(new: list[dict], old: list[dict], k: int) -> list[dict]:
    seen, out = set(), []
    for h in new + old:
        if h["id"] not in seen:
            seen.add(h["id"])
            out.append(h)
    return out[:k]


def _judge(clause: dict, hits: list[dict], v: Verdict, doc_needs_review: bool) -> dict:
    notes = check_verdict(v, clause, hits)
    if v.needs_more_context:
        notes.append("regulatory context still insufficient after re-searching")
    # docproc rule: a violation resting on OCR text from a flagged document must be checked by a person
    if v.status == "Violation" and clause.get("from_ocr") and doc_needs_review:
        notes.append("based on OCR text from a document flagged for review")
    return _finding(clause, hits, v, notes)


def _finding(clause: dict, hits: list[dict], v: Verdict | None, notes: list[str]) -> dict:
    labels = {label(n): h for n, h in enumerate(hits, 1)}
    status = "Needs Review" if v is None or notes else v.status
    return {
        "clause_id": clause["clause_id"],
        "title": clause.get("title"),
        "page_start": clause.get("page_start"),
        "page_end": clause.get("page_end"),
        "status": status,
        "llm_status": v.status if v else None,
        "clause_quote": v.clause_quote if v else "",
        "regulations": [{"id": labels[r]["id"], "citation": labels[r]["citation"]}
                        for r in (v.regulation_ids if v else []) if r in labels],
        "regulation_quote": v.regulation_quote if v else "",
        "reasoning_ar": v.reasoning_ar if v else "",
        "notes": notes,
        "retrieved": [h["id"] for h in hits],
    }
