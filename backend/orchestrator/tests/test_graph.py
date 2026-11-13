"""Graph behaviour with a stand-in retriever.  Run from backend/:  python -m pytest orchestrator/tests -q"""
from orchestrator.graph import SegmentationAgent, build_graph
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


class FakeSegmentationAgent(SegmentationAgent):
    """The Segmentation Agent's interface without the LLM: moves one clause and reports it."""
    def __call__(self, items, document=None):
        self.document = document
        fixed = [dict(CLAUSES[0], fixed_by_agent=True, agent_note="نُقلت بداية البند"), CLAUSES[1]]
        return fixed, {"status": "fixed", "summary": "صُحّح بند واحد", "problems_found": [],
                       "changes": [{"clause": 1, "reason": "نُقلت بداية البند"}], "steps": []}


def test_segmentation_agent_gets_the_document_and_its_report_reaches_the_output():
    agent = FakeSegmentationAgent()
    graph = build_graph(StubRetriever(), extract=fake_extract, segment=agent, on_status=lambda *a, **k: None)
    out = graph.invoke({"review_id": "t", "file_path": "t.pdf", "results": []})["output"]
    assert agent.document["filename"] == "t.pdf"
    assert out["segmentation"]["status"] == "fixed"
    assert [c["fixed_by_agent"] for c in out["clauses"]] == [True, False]
    assert out["clauses"][0]["agent_note"] == "نُقلت بداية البند"


def test_rules_only_segmentation_has_no_report():
    out = run(StubRetriever())["output"]
    assert out["segmentation"] is None
    assert not any(c["fixed_by_agent"] for c in out["clauses"])


def test_segmentation_agent_failure_falls_back_to_the_rules(monkeypatch):
    import segmentation_agent
    from orchestrator import graph as g

    def broken(*a, **k):
        raise RuntimeError("no API key")
    monkeypatch.setattr(segmentation_agent, "run_segmentation_agent", broken)
    monkeypatch.setattr(g, "default_segment", lambda items: CLAUSES)
    graph = build_graph(StubRetriever(), extract=fake_extract, segment=SegmentationAgent(),
                        on_status=lambda *a, **k: None)
    out = graph.invoke({"review_id": "t", "file_path": "t.pdf", "results": []})["output"]
    assert out["segmentation"]["status"] == "error" and "no API key" in out["segmentation"]["summary"]
    assert [c["clause_id"] for c in out["clauses"]] == ["C001", "C002"]
