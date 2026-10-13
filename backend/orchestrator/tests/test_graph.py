"""Graph behaviour with stand-in agents.  Run from backend/:  python -m pytest orchestrator/tests -q"""
from orchestrator.graph import build_graph
from orchestrator.stubs import StubRetriever, stub_verifier
from orchestrator.verifier import Verdict

REG = "يجب أن تتضمن شروط وأحكام الصندوق الخاص بيان الرسوم ومقابل الخدمات."

CLAUSES = [
    {"clause_id": "C001", "title": "الرسوم", "text": "يدفع المشترك رسوم اشتراك بنسبة 2% من مبلغ الاشتراك.",
     "page_start": 3, "page_end": 3, "from_ocr": False},
    {"clause_id": "C002", "title": "الاسترداد", "text": "يحق لمدير الصندوق تأجيل الاسترداد دون إشعار.",
     "page_start": 4, "page_end": 5, "from_ocr": True},
]


def fake_extract(needs_review=False):
    return lambda path: {"document": {"filename": "t.pdf", "needs_review": needs_review, "warnings": []},
                         "items": [{"id": "p1-1"}]}


def run(verifier, retriever=None, needs_review=False, clauses=CLAUSES, **kw):
    graph = build_graph(retriever or StubRetriever(REG), verifier, extract=fake_extract(needs_review),
                        segment=lambda items: clauses, on_status=lambda *a, **k: None, **kw)
    return graph.invoke({"review_id": "t", "file_path": "t.pdf", "findings": []})


def violation(clause, hits):
    return Verdict(status="Violation", regulation_ids=["REG-1"], clause_quote=clause["text"][:20],
                   regulation_quote="بيان الرسوم ومقابل الخدمات", reasoning_ar="سبب")


def test_every_clause_gets_one_finding_and_report_counts_them():
    report = run(stub_verifier)["report"]
    assert sorted(f["clause_id"] for f in report["findings"]) == ["C001", "C002"]
    assert report["counts"] == {"Violation": 0, "Needs Review": 2, "Compliant": 0}


def test_valid_violation_keeps_citation_and_page():
    f = run(violation)["report"]["findings"][0]
    assert f["status"] == "Violation" and f["notes"] == []
    assert f["regulations"] == [{"id": "IFR-anx11-c03", "citation": "لائحة صناديق الاستثمار، الملحق 11"}]
    assert f["page_start"] == 3


def test_made_up_quote_is_downgraded_to_needs_review():
    def bad(clause, hits):
        return violation(clause, hits).model_copy(update={"regulation_quote": "نص غير موجود في اللائحة"})
    for f in run(bad)["report"]["findings"]:
        assert f["status"] == "Needs Review" and f["llm_status"] == "Violation"
        assert "regulation_quote is not in any cited regulation" in f["notes"]


def test_ocr_violation_in_flagged_document_needs_review():
    by_id = {f["clause_id"]: f for f in run(violation, needs_review=True)["report"]["findings"]}
    assert by_id["C001"]["status"] == "Violation"        # native text
    assert by_id["C002"]["status"] == "Needs Review"     # OCR text in a flagged document


def test_insufficient_context_searches_again_then_gives_up():
    retriever = StubRetriever(REG)

    def unsure(clause, hits):
        return Verdict(status="Needs Review", reasoning_ar="غير كاف", needs_more_context=True,
                       suggested_query="أحكام تأجيل الاسترداد")
    report = run(unsure, retriever, clauses=CLAUSES[:1], max_retries=2)["report"]
    assert len(retriever.queries) == 3                  # first search + 2 re-searches
    assert report["findings"][0]["status"] == "Needs Review"


def test_low_retrieval_score_skips_the_llm():
    def never(clause, hits):
        raise AssertionError("verifier should not be called")
    f = run(never, min_score=2.0)["report"]["findings"][0]
    assert f["status"] == "Needs Review" and f["notes"] == ["no matching regulation found"]


def test_failed_extraction_ends_without_report():
    def broken(path):
        raise ValueError("ملف تالف")
    graph = build_graph(StubRetriever(REG), stub_verifier, extract=broken, on_status=lambda *a, **k: None)
    state = graph.invoke({"review_id": "t", "file_path": "x.pdf", "findings": []})
    assert state["error"] == "ملف تالف" and "report" not in state
