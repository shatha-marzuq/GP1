"""
Regulatory Retrieval Agent — rule-based steps on top of RegulationRetriever, plus an optional LLM step.

retriever.search() is one fixed step: clause → one search → top-k. The agent instead runs a short
decide-and-act loop per clause and records every decision with its reason:

  1. analyse   : read the clause title (from the segmentation breadcrumb or the caller) and look for
                 explicit references ("وفقًا للمادة 76", "الملحق (11)", "المادة الثالثة والتسعون")
  2. reference : fetch every referenced article/annex directly from the knowledge base (tool: get_section)
  3. search    : the normal search (tool: retriever.search, same mode / filters / rerank as before)
  4. check     : is the top result supported by an independent signal?
                   - lexical : it is also in the top 3 of a BM25 search of the same text
                   - title   : its article title matches the clause title (tool: title_match)
  5. retry     : if not supported, search again with other strategies — title + text in hybrid mode,
                 and, for a long clause, each part of it on its own — then fuse all lists (RRF)
  6. llm       : (optional, llm=LLM()) an LLM reads the candidate sections, may read a full section or search
                 again (tools: read_section, search_regulations), and returns the ranking + found/unsure/none.
                 If it fails (no key, server down) the rule ranking of steps 3-5 stands. A referenced article
                 stays first only if the LLM picks it; otherwise it follows the LLM's picks and the step
                 "reference_mismatch" is logged (the T&C may cite an old numbering of the regulation).
  7. decide    : confirmed | confirmed_by_reference | confirmed_after_retry | confirmed_by_llm | uncertain | no_match
                 ("no_match" when nothing passes min_score, or when the LLM finds no governing section)

Drop-in for the orchestrator: agent.search(text, k=…, **kw) returns the same hit dicts as
retriever.search (+ "found_by"); agent.run(text, title=…) also returns the decision and the steps.

    from retriever import RegulationRetriever
    from retrieval_agent import RetrievalAgent
    agent = RetrievalAgent(RegulationRetriever())
    out = agent.run(clause["text"], title=clause.get("title"), k=5)
    out["decision"], out["steps"], out["hits"]

CLI:  python3 retrieval_agent.py "نص البند" [--title "عنوان البند"] [--k 5] [--mode dense|hybrid|bm25] [--no-rerank] [--llm]
"""
import argparse, json, os, re, time
from collections import Counter, defaultdict

from chunk import search_normalize
from retriever import BM25, PIN_SECTION, tokenize

try:  # the API key and model come from the .env, like the segmentation agent
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SHORT_OF_DOC = {"investment_funds_regulations": "IFR", "capital_market_institutions_regulations": "CMIR",
                "corporate_governance_regulations": "CGR", "simplified_investment_funds_instructions": "SIFI"}
# اسم اللائحة اللي يجي بعد الإحالة → المستند. بدون اسم = لائحة صناديق الاستثمار (مرجع الشروط والأحكام)
DOC_HINTS = [("مؤسسات السوق", "CMIR"), ("حوكمة الشركات", "CGR"), ("المبسطة", "SIFI"), ("صناديق الاستثمار", "IFR")]
# مواد من أنظمة مو موجودة في قاعدة المعرفة (مثل نظام السوق المالية نفسه) نتجاهلها
OUTSIDE_KB = ("نظام السوق المالية", "نظام الشركات", "نظام مكافحة")

REF_NUM = re.compile(r"(?:ب|ل|و|ف)?(?:ال|ل)?(ماده|ملحق)\s*(?:رقم\s*)?\(?\s*(\d{1,3})\s*\)?")
BREADCRUMB = re.compile(r"^\s*\[([^\]]+)\]")
LEAD_NUMBERING = re.compile(r"^[\d٠-٩]+(?:\s*-\s*(?:[\d٠-٩]+|[ء-ي](?![ء-ي])))*\s*[-.)]?\s*")
PART_SUFFIX = re.compile(r"\(الجزء[^)]*\)")
SENT_SPLIT = re.compile(r"(?<=[.؛!؟\n])\s+")


LLM_PROMPT = """You are the Regulatory Retrieval Agent of Mithaq, a system that checks the Terms and Conditions
(الشروط والأحكام) of Saudi private investment funds against the Capital Market Authority (CMA) regulations.

Your job: for ONE clause of a Terms and Conditions document, choose the regulation sections that a compliance
officer must check this clause against, most relevant first. A later agent will judge compliance using only
the sections you choose, so the governing requirement must be in your list.

You get candidate sections found by search (key, citation, title, start of the text). You can:
- read_section(section_key): read the full text of a candidate when its start is not enough to decide.
- search_regulations(query): search the regulations again when no candidate governs the clause, e.g. with
  the regulatory wording of the clause's subject. New sections are added to the candidates.
- finish(ranked, status, reason): end with the section keys, most relevant first (only keys you were shown).

Rules:
- Rank first the section that sets the requirement for this clause's subject, not a section that only
  shares words with it. Definitions, scope and general sections come after the specific requirement.
- IFR-anx11 is the checklist of topics a private fund's Terms and Conditions must contain. When a specific
  article regulates the clause's subject in detail, rank that article first and IFR-anx11 after it. Rank
  IFR-anx11 first only when no candidate article regulates the subject itself.
- status: "found" when a candidate clearly governs the clause, "unsure" when you are not confident,
  "none" when no section in the knowledge base governs it.
- Never invent section keys. Write the reason in Arabic, one short sentence.
"""

LLM_TOOLS = [
    {"name": "read_section",
     "description": "Full text of one candidate regulation section (article or annex).",
     "parameters": {"type": "object", "properties": {"section_key": {"type": "string"}},
                    "required": ["section_key"]}},
    {"name": "search_regulations",
     "description": "Search the CMA regulations knowledge base again with a new query (Arabic). "
                    "Returns new candidate sections.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "finish",
     "description": "Return the chosen sections, most relevant first.",
     "parameters": {"type": "object", "properties": {
         "ranked": {"type": "array", "items": {"type": "string"}, "description": "section keys, best first"},
         "status": {"type": "string", "enum": ["found", "unsure", "none"]},
         "reason": {"type": "string"}},
         "required": ["ranked", "status", "reason"]}},
]


class LLM:
    """OpenAI-compatible chat model with tools (OPENAI_API_KEY / OPENAI_BASE_URL / RETRIEVAL_MODEL in .env)."""

    def __init__(self, model=None, tool_choice=None):
        from openai import OpenAI
        # the default waits up to 10 min per call: one hung request stalled a whole evaluation run for 30+ min.
        # On timeout the agent falls back to the rule ranking (llm_error step).
        self.client = OpenAI(timeout=float(os.getenv("RETRIEVAL_TIMEOUT", 60)), max_retries=2)
        self.model = model or os.getenv("RETRIEVAL_MODEL", "gpt-4o")
        self.tool_choice = tool_choice or os.getenv("RETRIEVAL_TOOL_CHOICE", "required")

    def __call__(self, messages, tools):
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, tool_choice=self.tool_choice,
                    tools=[{"type": "function", "function": t} for t in tools])
                return r.choices[0].message
            except Exception as e:
                # 429 is also returned when the account has no credit left: waiting does not help there
                no_credit = (getattr(e, "body", None) or {}).get("type") == "insufficient_quota"
                if getattr(e, "status_code", None) == 429 and not no_credit and attempt < 5:
                    time.sleep(20)
                else:
                    raise


def _norm(text):
    # نفس توحيد البحث + "السبعين" = "السبعون" عشان الأرقام بالكلام تطابق عناوين المواد
    t = search_normalize(text)
    t = re.sub(r"\b(?:لل|بال|وال|فال)(ماده|ملحق)", r"ال\1", t)
    return re.sub(r"ين\b", "ون", t)


def clause_title(text, title=None):
    """Title from the caller, else the last part of the segmentation breadcrumb "[الباب › العنوان (الجزء 1 من 3)]"."""
    if not title:
        m = BREADCRUMB.match(text)
        title = m.group(1).split("›")[-1] if m else ""
    title = PART_SUFFIX.sub("", title)
    return LEAD_NUMBERING.sub("", title.strip()).strip()


class RetrievalAgent:
    def __init__(self, retriever, *, check_top=3, max_retries=2, long_clause_words=120, part_words=60,
                 min_score=None, llm=None, llm_candidates=10, llm_steps=6):
        """check_top: rank a signal must put the top section in to count as support.
        min_score: optional floor on the final top score (scale depends on mode/rerank, so off by default).
        llm: a chat model with tools (LLM()); it reads the candidates and chooses the ranking (step 6).
        None = rules only."""
        self.r = retriever
        self.llm, self.llm_candidates, self.llm_steps = llm, llm_candidates, llm_steps
        self.check_top, self.max_retries = check_top, max_retries
        self.long_clause_words, self.part_words, self.min_score = long_clause_words, part_words, min_score
        self.sections = defaultdict(list)                       # section_key → chunk ids in order
        for i, m in zip(self.r.ids, self.r.metas):
            self.sections[m["section_key"]].append(i)
        first = {k: self.r.metas[self.r.index[v[0]]] for k, v in self.sections.items()}
        self.section_meta = first
        self.keys = list(first)
        self.title_index = BM25([tokenize(m.get("section_title") or "") or ["_"] for m in first.values()])
        # "المادة الثالثة والتسعون" → IFR-art093 ، أطول عنوان أول عشان "الثالثة عشرة" ما تنقص لـ"الثالثة"
        # (نفس الصياغة في كل اللوائح، فنأخذها من لائحة صناديق الاستثمار والمستند يتحدد من اسم اللائحة بعدها)
        self.labels = sorted({(_norm(m["article_label"]), m["article_number"]) for m in first.values()
                              if m.get("article_label") and m["doc_id"] == "investment_funds_regulations"},
                             key=lambda x: -len(x[0]))

    # ---------------------------------------------------------------- tools
    def get_section(self, section_key):
        """All chunks of one article / annex, as hits."""
        return [self._hit(i, None) for i in self.sections.get(section_key, [])]

    def title_match(self, title, fund_type=None, n=5):
        """Sections whose article title shares words with the clause title (BM25 over titles)."""
        q = tokenize(title)
        if not q:
            return []
        sc = self.title_index.scores(q)
        ok = [j for j in range(len(self.keys)) if sc[j] > 0
              and self.r._allowed(self.section_meta[self.keys[j]], None, None, None, fund_type)]
        return [(self.keys[j], round(sc[j], 3)) for j in sorted(ok, key=lambda j: -sc[j])[:n]]

    def find_references(self, text):
        """Explicit references to articles / annexes inside the clause → section keys."""
        t = _norm(text)
        found, taken = [], [False] * len(t)

        def doc_after(end):
            window = t[end:end + 50]
            if any(_norm(w) in window for w in OUTSIDE_KB):
                return None
            return next((d for hint, d in DOC_HINTS if _norm(hint) in window), "IFR")

        def add(kind, num, start, end):
            if any(taken[start:end]):
                return
            taken[start:end] = [True] * (end - start)
            doc = doc_after(end)
            key = None if doc is None else f"{doc}-art{int(num):03d}" if kind == "ماده" else f"{doc}-anx{int(num)}"
            ok = key in self.sections
            found.append({"ref": t[start:end], "key": key if ok else None,
                          "why": None if ok else "نظام خارج قاعدة المعرفة" if doc is None else "غير موجودة في قاعدة المعرفة"})

        for label, num in self.labels:                          # المادة الثالثة والتسعون
            for m in re.finditer(re.escape(label) + r"\b", t):
                add("ماده", num, m.start(), m.end())
        for m in REF_NUM.finditer(t):                           # المادة 76 / الملحق (11)
            add(m.group(1), m.group(2), m.start(), m.end())
        return found

    # ---------------------------------------------------------------- agent loop
    def run(self, text, title=None, k=5, mode="dense", fund_type="private", rerank=True, **search_kw):
        steps = []
        log = lambda action, **info: steps.append({"step": len(steps) + 1, "action": action, **info})
        kw = dict(mode=mode, fund_type=fund_type, rerank=rerank, **search_kw)
        pool = max(k, 10)

        # 1-2) analyse + explicit references
        title = clause_title(text, title)
        refs = self.find_references(text)
        ref_keys = list(dict.fromkeys(r["key"] for r in refs if r["key"]))
        log("analyse", title=title or None, words=len(text.split()),
            references=[{"ref": r["ref"], "section": r["key"], "note": r["why"]} for r in refs])

        # 3) normal search
        primary = self.r.search(text, k=pool, **kw)
        log("search", mode=mode, rerank=rerank, top=[h["section_key"] for h in primary[:3]])
        if not primary and not ref_keys:
            log("decide", decision="no_match", reason="البحث ما رجع أي نتيجة")
            return self._out([], "no_match", steps, k)

        # 4) check the top result against independent signals
        lexical = None
        if mode != "bm25":
            lexical = self.r.search(text, k=pool, mode="bm25", fund_type=fund_type, rerank=False,
                                    pin_annex11=False, **search_kw)
        titled = self.title_match(title, fund_type) if title else []
        support = self._support(primary, lexical, titled)
        log("check", top=primary[0]["section_key"] if primary else None, support=support or None)

        lists = {"search": primary}
        decision = "confirmed" if support else "confirmed_by_reference" if ref_keys else None
        retries = 0
        # 5) retry with other strategies until the top is supported or retries run out
        strategies = []
        if title:
            strategies.append(("title+text", f"{title}\n{text}", "hybrid" if mode != "bm25" else "bm25"))
        if len(text.split()) > self.long_clause_words:
            strategies.append(("parts", None, mode))
        while decision is None and retries < min(self.max_retries, len(strategies)):
            name, query, m = strategies[retries]
            retries += 1
            if name == "parts":
                parts = self._split(text)
                for n, p in enumerate(parts, 1):
                    lists[f"part{n}"] = self.r.search(p, k=pool, **{**kw, "mode": m})
                log("retry", strategy="parts", parts=len(parts),
                    top=[lists[f"part{n}"][0]["section_key"] for n in range(1, len(parts) + 1) if lists[f"part{n}"]])
            else:
                lists[name] = self.r.search(query, k=pool, **{**kw, "mode": m})
                log("retry", strategy=name, mode=m, top=[h["section_key"] for h in lists[name][:3]])
            fused = self._fuse(lists, titled)
            support = self._support(fused, lexical, titled, extra=[v for n, v in lists.items() if n != "search"])
            log("check", top=fused[0]["section_key"] if fused else None, support=support or None)
            if support:
                decision = "confirmed_after_retry"
        if decision is None:
            decision = "uncertain"
        # نغيّر الترتيب بس إذا إعادة البحث أكدت نتيجة؛ غير كذا نرجّع ترتيب البحث الأصلي ونعلّمه "غير مؤكد"
        hits = self._fuse(lists, titled) if decision == "confirmed_after_retry" else primary

        # 6) the LLM reads the candidates (and may read sections or search again) and chooses the ranking
        picked = None
        if self.llm is not None:
            picked = self._llm_rank(text, title, hits, lists, lexical, titled, ref_keys, kw, pool, log)
            if picked:
                ranked, status = picked
                rule_top = hits[0]["section_key"] if hits else None
                # إحالة البند ما تتقدم إلا إذا النموذج اختارها: الشروط والأحكام أحيانا تحيل لترقيم قديم للائحة
                # (ملكية: "المادة 92" للإنهاء وهي اليوم التقييم)، فالإحالة اللي ما تطابق تنزل بعد اختيار النموذج
                unmatched = [key for key in ref_keys if key not in ranked]
                hits = [self._best_chunk(key, lists, "llm") for key in ranked] + \
                       [self._best_chunk(key, lists) for key in unmatched] + \
                       [h for h in hits if h["section_key"] not in ranked and h["section_key"] not in unmatched]
                if unmatched:
                    log("reference_mismatch", references=unmatched,
                        note="البند يحيل لمادة موضوعها لا يطابق البند، يمكن ترقيم قديم للائحة")
                # تأكيد = القواعد أكدت أعلى نتيجة والنموذج اختار نفسها (47 من 47 صح بالتقييم)؛ أي اختلاف يروح للمراجعة
                agree = decision in ("confirmed", "confirmed_after_retry", "confirmed_by_reference") \
                    and status == "found" and hits[0]["section_key"] == rule_top
                decision = "no_match" if status == "none" else \
                    "confirmed_by_reference" if status == "found" and hits[0]["section_key"] in ref_keys else \
                    "confirmed_by_llm" if agree else "uncertain"

        # explicit references go first: the clause itself names them (rules only; with the LLM see step 6)
        if ref_keys and not picked:
            ref_hits = [self._best_chunk(key, lists) for key in ref_keys]
            hits = ref_hits + [h for h in hits if h["section_key"] not in ref_keys]
        hits = self._pin_annex11(hits, lists, fund_type, kw.get("pin_annex11"))
        if self.min_score is not None and hits and not ref_keys and hits[0]["score"] < self.min_score:
            decision = "no_match"
        reasons = {"confirmed": "أعلى نتيجة تؤيدها إشارة ثانية مستقلة",
                   "confirmed_by_reference": "البند يحيل للمادة صراحة" + (" والنموذج أكد موضوعها" if picked else ""),
                   "confirmed_after_retry": "بعد إعادة البحث صارت أعلى نتيجة مؤيدة",
                   "confirmed_by_llm": "القواعد والنموذج اتفقوا على نفس المادة",
                   "uncertain": "ما فيه إشارة تؤيد أعلى نتيجة حتى بعد إعادة البحث، تحتاج مراجعة",
                   "no_match": "أعلى درجة أقل من الحد الأدنى"}
        log("decide", decision=decision, reason=reasons[decision], references_first=ref_keys if ref_keys and not picked else None)
        return self._out(hits, decision, steps, k)

    def search(self, query, k=5, **kw):
        """Same contract as RegulationRetriever.search, so build_graph can use the agent unchanged."""
        return self.run(query, k=k, **kw)["hits"]

    # ---------------------------------------------------------------- helpers
    def _support(self, ranked, lexical, titled, extra=()):
        if not ranked:
            return []
        top = ranked[0]["section_key"]
        out = []
        if lexical and top in self._top_sections(lexical):
            out.append("lexical")
        if titled and top in [s for s, _ in titled[:self.check_top]]:
            out.append("title")
        if any(top in self._top_sections(lst, 1) for lst in extra if lst):
            out.append("retry_agrees")
        # a retry agreeing with a fused list it is part of is weak on its own: needs lexical/title too
        return out if ("lexical" in out or "title" in out) else []

    def _top_sections(self, hits, n=None):
        return list(dict.fromkeys(h["section_key"] for h in hits))[:n or self.check_top]

    def _fuse(self, lists, titled, rrf_k=60):
        """Reciprocal-rank fusion at section level; each section keeps its best-ranked chunk."""
        score, best = Counter(), {}
        for name, hits in lists.items():
            weight = 1.0 if name == "search" else 0.8
            for rank, key in enumerate(self._top_sections(hits, len(hits))):
                score[key] += weight / (rrf_k + rank + 1)
                best.setdefault(key, next(h for h in hits if h["section_key"] == key))
        for rank, (key, _) in enumerate(titled[:self.check_top]):
            score[key] += 0.5 / (rrf_k + rank + 1)
        out = []
        for key, s in score.most_common():
            h = best.get(key) or self._hit(self.sections[key][0], None)
            out.append({**h, "agent_score": round(s, 5), "found_by": h.get("found_by") or "search"})
        return out

    def _best_chunk(self, key, lists, found_by="reference"):
        for hits in lists.values():
            for h in hits:
                if h["section_key"] == key:
                    return {**h, "found_by": found_by}
        return {**self._hit(self.sections[key][0], None), "found_by": found_by}

    # ---------------------------------------------------------------- LLM step
    def _llm_rank(self, text, title, hits, lists, lexical, titled, ref_keys, kw, pool, log):
        """Tool loop: candidates → (read_section | search_regulations)* → finish. None = keep the rule ranking."""
        keys = list(ref_keys) + self._top_sections(hits, self.llm_candidates)
        for extra in [[s for s, _ in titled[:3]], self._top_sections(lexical or [], 3)] + \
                     [self._top_sections(v, 3) for v in lists.values()]:
            keys += [k for k in extra if k not in keys]
        keys = list(dict.fromkeys(keys))[:self.llm_candidates + 4]
        shown = set(keys)
        user = (f"عنوان البند: {title or '-'}\n\nنص البند:\n{BREADCRUMB.sub('', text).strip()}\n\n"
                f"المرشحون:\n{json.dumps(self._cards(keys, lists), ensure_ascii=False, indent=1)}")
        messages = [{"role": "system", "content": LLM_PROMPT}, {"role": "user", "content": user}]
        n_search = 0
        try:
            for _ in range(self.llm_steps):
                msg = self.llm(messages, LLM_TOOLS)
                calls = msg.tool_calls or []
                messages.append(msg.model_dump(exclude_none=True))
                if not calls:
                    messages.append({"role": "user", "content": "Use a tool. End with finish."})
                    continue
                for c in calls:
                    name, args = c.function.name, json.loads(c.function.arguments or "{}")
                    if name == "finish":
                        ranked = [k for k in dict.fromkeys(args.get("ranked") or []) if k in shown]
                        status = args.get("status") if args.get("status") in ("found", "unsure", "none") else "unsure"
                        log("llm_finish", ranked=ranked, status=status, reason=args.get("reason"))
                        return (ranked, status) if ranked or status == "none" else None
                    if name == "read_section" and args.get("section_key") in self.sections:
                        key = args["section_key"]
                        shown.add(key)
                        result = {"section_key": key, "text": " ".join(h["text"] for h in self.get_section(key))[:4000]}
                    elif name == "search_regulations" and args.get("query"):
                        n_search += 1
                        found = self.r.search(args["query"], k=pool, **kw)
                        lists[f"llm_search{n_search}"] = found
                        new = self._top_sections(found, 5)
                        shown.update(new)
                        result = self._cards(new, lists)
                    else:
                        result = {"error": "unknown tool or section key"}
                    log("llm_tool", tool=name, **{k: v for k, v in args.items() if k in ("section_key", "query")})
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": json.dumps(result, ensure_ascii=False)})
            log("llm_gave_up", steps=self.llm_steps)
        except Exception as e:  # no key, server down, bad JSON → the rule ranking stands
            log("llm_error", error=f"{type(e).__name__}: {e}"[:300])
        return None

    def _cards(self, keys, lists, chars=500):
        out = []
        for key in keys:
            h = self._best_chunk(key, lists)
            out.append({"section_key": key, "citation": h["citation"],
                        "title": self.section_meta[key].get("section_title"), "text": h["text"][:chars]})
        return out

    def _pin_annex11(self, hits, lists, fund_type, pin):
        # نفس قاعدة retriever: الملحق 11 (قائمة إفصاحات الصندوق الخاص) يكون ضمن أول قسمين إذا كان مرشح
        if (pin is False) or (pin is None and fund_type != "private") or not hits:
            return hits
        if PIN_SECTION in self._top_sections(hits, 2):
            return hits
        cand = next((h for h in hits if h["section_key"] == PIN_SECTION), None)
        cand = cand or next((h for hs in lists.values() for h in hs if h["section_key"] == PIN_SECTION), None)
        if not cand:
            return hits
        rest = [h for h in hits if h["section_key"] != PIN_SECTION]
        return rest[:1] + [{**cand, "pinned": True}] + rest[1:]

    def _split(self, text):
        body = BREADCRUMB.sub("", text).strip()
        parts, cur = [], []
        for s in SENT_SPLIT.split(body):
            cur.append(s)
            if len(" ".join(cur).split()) >= self.part_words:
                parts.append(" ".join(cur)); cur = []
        if cur:
            if parts and len(" ".join(cur).split()) < self.part_words // 3:
                parts[-1] += " " + " ".join(cur)
            else:
                parts.append(" ".join(cur))
        return parts[:6]

    def _hit(self, chunk_id, score):
        m = self.r.metas[self.r.index[chunk_id]]
        return {"id": chunk_id, "score": score, "section_key": m["section_key"], "citation": m["citation"],
                "binding_status": m["binding_status"], "pdf_pages": m["pdf_pages"],
                "text": self.r.docs[self.r.index[chunk_id]], "metadata": m}

    @staticmethod
    def _out(hits, decision, steps, k):
        hits = [{**h, "found_by": h.get("found_by", "search")} for h in hits[:k]]
        return {"decision": decision, "hits": hits, "steps": steps}


if __name__ == "__main__":
    from retriever import RegulationRetriever
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--title")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="dense", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--llm", action="store_true", help="add the LLM step (needs OPENAI_API_KEY)")
    a = ap.parse_args()
    out = RetrievalAgent(RegulationRetriever(), llm=LLM() if a.llm else None).run(a.query, a.title, a.k, a.mode,
                                                    rerank=not a.no_rerank and a.mode != "bm25")
    print("decision:", out["decision"])
    for s in out["steps"]:
        print(json.dumps(s, ensure_ascii=False))
    for n, h in enumerate(out["hits"], 1):
        print(f"#{n} {h['section_key']:12} {h['found_by']:9} {h['citation']}")
