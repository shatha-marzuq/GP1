"""Graph behaviour with a stand-in retriever.  Run from backend/:  python -m pytest orchestrator/tests -q"""
from orchestrator.graph import build_graph
from orchestrator.stubs import StubRetriever

CLAUSES = [
    {"clause_id": "C001", "title": "الرسوم", "text": "يدفع المشترك رسوم اشتراك بنسبة 2% من مبلغ الاشتراك.",
     "page_start": 3, "page_end": 3, "from_ocr": False},
    {"clause_id": "C002", "title": "الاسترداد", "text": "يحق لمدير الصندوق تأجيل الاسترداد دون إشعار.",
     "page_start": 4, "page_end": 5, "from_ocr": True},
]


def fake_extract(path):
    return {"document": {"filename": "t.pdf", "needs_review": True, "warnings": ["low OCR confidence"]},
            "items": [{"id": "p1-1"}]}


def run(retriever, clauses=CLAUSES, extract=fake_extract, **kw):
    graph = build_graph(retriever, extract=extract, segment=lambda items: clauses,
                        on_status=lambda *a, **k: None, **kw)
    return graph.invoke({"review_id": "t", "file_path": "t.pdf", "results": []})


def test_every_clause_is_searched_once_and_kept_in_document_order():
    retriever = StubRetriever()
    out = run(retriever)["output"]
    assert [c["clause_id"] for c in out["clauses"]] == ["C001", "C002"]
    assert sorted(retriever.queries) == sorted(c["text"] for c in CLAUSES)


def test_result_keeps_pages_and_regulation_citation():
    c = run(StubRetriever())["output"]["clauses"][1]
    assert (c["page_start"], c["page_end"], c["from_ocr"]) == (4, 5, True)
    assert c["regulations"][0]["citation"] == "لائحة صناديق الاستثمار، الملحق 11"
    assert c["regulations"][0]["pdf_pages"] == [120]


def test_document_flags_from_docproc_are_passed_through():
    out = run(StubRetriever())["output"]
    assert out["document_needs_review"] is True and out["document_warnings"] == ["low OCR confidence"]


def test_k_and_search_options_reach_the_retriever():
    seen = []

    class Spy(StubRetriever):
        def search(self, query, k=5, **kwargs):
            seen.append((k, kwargs))
            return super().search(query, k)
    run(Spy(), clauses=CLAUSES[:1], k=3, search_kwargs={"mode": "hybrid"})
    assert seen == [(3, {"mode": "hybrid"})]


def test_no_clauses_ends_without_output():
    state = run(StubRetriever(), clauses=[])
    assert "output" not in state


def test_failed_extraction_ends_without_output():
    def broken(path):
        raise ValueError("ملف تالف")
    state = run(StubRetriever(), extract=broken)
    assert state["error"] == "ملف تالف" and "output" not in state
