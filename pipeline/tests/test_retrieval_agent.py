"""Offline tests of the Regulatory Retrieval Agent: a scripted stand-in for the LLM and BM25 search, so they need
no API key, no network and no embedding model (only the Chroma store in output/).

    cd pipeline && python -m pytest tests -q
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import retrieval_agent as ra                                   # noqa: E402
from retriever import RegulationRetriever                     # noqa: E402

TRADING = ("[الشروط والأحكام › 9. تداول وحدات الصندوق] يتيح مدير الصندوق التداول الداخلي لمالكي الوحدات "
           "ويتم تحديث سجل مالكي الوحدات لدى أمين الحفظ وفقا للمادة 92")
KW = dict(mode="bm25", rerank=False)


class FakeLLM:
    """Answers like the real LLM; `select` picks keys from the candidates it is shown."""
    model = "fake"

    def __init__(self, plan=None, pick=None, fail=None):
        self.plan, self.pick, self.fail, self.calls = plan, pick, fail, []

    def ask_json(self, system, user, max_tokens=900):
        stage = "understand" if system.startswith("You help") else "select"
        self.calls.append((stage, user))
        if self.fail == stage:
            raise RuntimeError("boom")
        if stage == "understand":
            return self.plan or {"topics": [{"topic": "نقل ملكية الوحدات",
                                             "query": "نقل ملكية وحدات الصندوق الخاص القيود على نشاط السوق الثانوية"}],
                                 "articles": [], "annex11_item": 9}, False
        keys = re.findall(r'"key": "([^"]+)"', user)
        return {"selected": self.pick(keys) if self.pick else []}, False


@pytest.fixture(scope="module")
def r():
    return RegulationRetriever()


def agent(r, llm=None):
    return ra.RetrievalAgent(r, llm=llm)


def test_three_stages_selected_first_and_every_candidate_kept(r):
    pick = lambda keys: [{"key": "MADE-UP", "role": "governing", "reason": "x"},          # not in the pool: dropped
                         {"key": "IFR-art012", "role": "related", "reason": "سجل"},
                         {"key": "IFR-art098", "role": "governing", "reason": "نقل الملكية"}]
    out = agent(r, FakeLLM(pick=pick)).run(TRADING, **KW)
    steps = [s["action"] for s in out["steps"]]
    assert steps == ["analyse", "understand", "retrieve", "pool", "select"]
    pool = next(s for s in out["steps"] if s["action"] == "pool")["candidates"]
    hits = out["hits"]
    assert [h["section_key"] for h in hits[:2]] == ["IFR-art098", "IFR-art012"]      # governing before related
    assert [h["selected"] for h in hits[:2]] == [True, True] and all(h["selected"] is False for h in hits[2:])
    assert len(hits) == len(pool) and {ra.RetrievalAgent._unit(h) for h in hits} == set(pool)   # nothing dropped
    assert all(h["citation"] and h["heading"] and h["excerpt"] for h in hits)
    assert hits[0]["reason"] and hits[0]["role"] == "governing"


def test_annex_item_and_cited_article_enter_the_pool(r):
    a = agent(r, FakeLLM())
    out = a.run(TRADING, **KW)
    pool = next(s for s in out["steps"] if s["action"] == "pool")["candidates"]
    assert a.annex_items[9] in pool and "IFR-art092" in pool
    cited = next(h for h in out["hits"] if h["section_key"] == "IFR-art092")
    assert cited["cited"] and "cited" in cited["found_in"] and cited["full_text"]     # cited: full text kept


def test_full_text_only_for_selected_or_cited(r):
    pick = lambda keys: [{"key": "IFR-art098", "role": "governing", "reason": "x"}]
    hits = agent(r, FakeLLM(pick=pick)).run(TRADING, **KW)["hits"]
    for h in hits:
        assert bool(h["full_text"]) == bool(h["selected"] or h["cited"])
    art098 = hits[0]
    assert art098["full_text"] == agent(r).full_text("IFR-art098")


@pytest.mark.parametrize("fail", ["understand", "select", None])
def test_fallbacks_keep_candidates_and_mark_selection_unknown(r, fail):
    llm = FakeLLM(fail=fail) if fail else None
    out = agent(r, llm).run(TRADING, **KW)
    assert out["hits"] and all(h["selected"] is None for h in out["hits"])
    assert out["hits"][0]["section_key"] == "IFR-art092"          # rules: cited article first
    if fail:
        assert any(s["action"] == "llm_error" and s["stage"] == fail for s in out["steps"])
    if fail == "select":                                           # the pool built before the failure is kept too
        pool = next(s for s in out["steps"] if s["action"] == "pool")["candidates"]
        assert set(pool) <= {ra.RetrievalAgent._unit(h) for h in out["hits"]}


def test_scope_is_private_funds_only(r):
    # public-fund (IFR Part 4) and corporate-governance provisions never enter the pool unless the clause cites them
    plan = {"topics": [{"topic": "خطأ التقييم", "query": "التقييم أو التسعير الخاطئ تعويض مالكي الوحدات"}],
            "articles": ["IFR-art075", "IFR-art085", "IFR-art064", "NOT-A-KEY", "IFR-art092", "IFR-art093"],
            "annex11_item": 12}
    a = agent(r, FakeLLM(plan=plan))
    out = a.run("[الشروط والأحكام › 12. أصول الصندوق] في حال الخطأ في التقويم أو التسعير يعوض مدير الصندوق مالكي الوحدات", **KW)
    pool = next(s for s in out["steps"] if s["action"] == "pool")["candidates"]
    scopes = {r.metas[r.index[a.sections[u][0] if u in a.sections else u]]["fund_scope"] for u in pool}
    assert scopes <= {"general", "funds_all", "private", "private_foreign"}
    assert "IFR-art075" not in pool and "IFR-art064" not in pool       # named by the LLM but public-fund only
    named = next(s for s in out["steps"] if s["action"] == "understand")["articles"]
    kept = [u for u in pool if u in ("IFR-art085", "IFR-art092", "IFR-art093")]
    assert kept == ["IFR-art085", "IFR-art092", "IFR-art093"] and len(named) == 6   # in scope, at most MAX_NAMED
    cited = agent(r).run("وفقا للمادة 75 من لائحة صناديق الاستثمار يعوض المدير مالكي الوحدات", **KW)["hits"]
    assert cited[0]["section_key"] == "IFR-art075" and cited[0]["cited"]      # a cited one is shown, marked cited


def test_excerpt_keeps_the_related_paragraph_not_the_opening(r):
    a = agent(r)
    paras = a._paragraphs("IFR-art023")
    assert len(paras) > 3 and paras[0].startswith("أ)")
    target = next(p for p in paras if p.startswith("ن)"))            # paragraph 14 of 22: far past the opening
    assert target not in "\n".join(paras)[:ra.EXCERPT_CHARS]          # cutting from the start would miss it
    ex = a._excerpt("IFR-art023", "لا يتقاضى مدير الصندوق أتعاب إدارة بعد البدء في تصفية الصندوق", budget=350)
    assert target in ex and ex.startswith("…") and len(ex) < 500
    short = a._excerpt("IFR-art098", "نقل ملكية الوحدات")
    assert "…" not in short and short == "\n".join(a._paragraphs("IFR-art098"))      # short articles: whole


def test_select_sees_excerpts_and_marks_cited(r):
    llm = FakeLLM(pick=lambda keys: [])
    agent(r, llm).run(TRADING, **KW)
    user = next(u for stage, u in llm.calls if stage == "select")
    cards = json.loads(user.split("المواد المرشحة:\n", 1)[1])
    assert len(cards) == ra.POOL_SIZE and all(c["text"] for c in cards)
    assert next(c for c in cards if c["key"] == "IFR-art092")["cited_in_clause"]


def test_cache_key_covers_version_params_and_messages(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")                # never used: create() is replaced below
    monkeypatch.setenv("RETRIEVAL_CACHE", "1")
    monkeypatch.setattr(ra, "CACHE_DIR", str(tmp_path))
    llm = ra.LLM(model="gpt-4o", backend="openai")
    msgs = [{"role": "user", "content": "x"}]
    key = llm.cache_key(msgs, llm.params(900))
    assert key == llm.cache_key(msgs, llm.params(900))
    assert key != llm.cache_key(msgs, llm.params(700))                                 # max tokens
    assert key != llm.cache_key(msgs, {**llm.params(900), "temperature": 0.5})         # temperature
    assert key != llm.cache_key([{"role": "user", "content": "y"}], llm.params(900))  # messages
    monkeypatch.setattr(ra, "CACHE_VERSION", "test-bump")
    assert key != llm.cache_key(msgs, llm.params(900))                                 # cache version
    other = ra.LLM(model="gpt-4o-mini", backend="openai")
    assert other.cache_key(msgs, other.params(900)) != llm.cache_key(msgs, llm.params(900))   # model

    calls = []
    def create(**kw):
        calls.append(kw)
        if kw["messages"][1]["content"] == "fail":
            raise RuntimeError("server down")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": 1}'))])
    monkeypatch.setattr(llm.client.chat.completions, "create", create)
    assert llm.ask_json("s", "u") == ({"ok": 1}, False)
    assert llm.ask_json("s", "u") == ({"ok": 1}, True) and len(calls) == 1             # second answer: cache
    assert calls[0]["temperature"] == 0 and calls[0]["response_format"] == {"type": "json_object"}
    with pytest.raises(RuntimeError):
        llm.ask_json("s", "fail")
    with pytest.raises(RuntimeError):
        llm.ask_json("s", "fail")                                   # failures are never cached
    assert len(calls) == 3                                          # 1 answer + 2 failed attempts


def test_select_receives_topics_and_suggestions_and_records_topic(r):
    plan = {"topics": [{"topic": "أيام التداول", "query": "يوم التعامل الموعد النهائي لتقديم طلبات الاشتراك والاسترداد"},
                       {"topic": "سجل مالكي الوحدات", "query": "سجل مالكي الوحدات الدليل القاطع على الملكية"}],
            "articles": ["IFR-art085", "IFR-art012"], "annex11_item": 9}
    pick = lambda keys: [{"key": "IFR-art085", "role": "governing", "topic": 1, "reason": "يوم التعامل"},
                         {"key": "IFR-art012", "role": "governing", "topic": 2, "reason": "السجل"},
                         {"key": next(k for k in keys if k.startswith("IFR-anx11-")), "role": "disclosure",
                          "topic": 7, "reason": "بند التداول"}]                     # 7: no such topic → None
    llm = FakeLLM(plan=plan, pick=pick)
    hits = agent(r, llm).run(TRADING, **KW)["hits"]
    user = next(u for stage, u in llm.calls if stage == "select")
    assert "موضوعات البند:\n1) أيام التداول\n2) سجل مالكي الوحدات" in user          # step 3 sees step 1's topics
    cards = {c["key"]: c for c in json.loads(user.split("المواد المرشحة:\n", 1)[1])}
    assert cards["IFR-art085"].get("suggested") and cards["IFR-art012"].get("suggested")
    assert not cards["IFR-art098"].get("suggested") if "IFR-art098" in cards else True
    sel = [h for h in hits if h["selected"]]
    assert [(h["section_key"], h["topic"]) for h in sel] == [("IFR-art085", "أيام التداول"),
                                                              ("IFR-art012", "سجل مالكي الوحدات"),
                                                              ("IFR-anx11", None)]
