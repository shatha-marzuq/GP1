"""Shared state that travels through the graph. Each node returns only the keys it changed."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ReviewState(TypedDict, total=False):
    review_id: str
    file_path: str
    document: dict       # docproc's "document" block: filename, needs_review, warnings …
    items: list[dict]    # docproc's "items": headings / paragraphs / tables with page numbers
    clauses: list[dict]  # segmentation.segment_clauses output
    findings: Annotated[list[dict], operator.add]  # parallel clause branches append here
    report: dict
    error: str


class ClauseTask(TypedDict):
    """What one parallel branch receives: a single clause plus the document-level flags it needs."""
    review_id: str
    clause: dict
    doc_needs_review: bool
