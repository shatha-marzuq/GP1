"""
Regulatory Retrieval Agent — finds the CMA provisions that apply to each clause of a private fund's T&C.

retriever.search() is one fixed step: clause text → one search → top-k, ranked by textual similarity.
The agent coordinates the rule-based search with an LLM so the result is the *applicable* provisions:

  1. understand : the LLM reads the clause and lists its regulatory topics (a clause often has several:
                  retail cap, minimum size, offering period …). For each it writes a search query in the
                  regulation's own terminology (query expansion), may name up to 3 articles from the catalogue
                  of private-fund provisions, and names the Annex 11 item the clause's section corresponds to.
                  Explicit references ("وفقًا للمادة 76", "الملحق (11)") are found by rules.
  2. retrieve   : rules build one candidate pool, always filtered to the provisions that govern private funds
                  (fund_type="private": CMIR, the IFR parts common to all funds, IFR Part 5 and its annexes) —
                  the clause text (dense + cross-encoder rerank), BM25 on the clause, hybrid (dense + BM25) search
                  for every topic query, article-title match — fused with reciprocal-rank fusion, plus the cited
                  articles, the named articles and the Annex 11 item.
  3. select     : the LLM reads the pool once — for each candidate the paragraphs most related to the clause,
                  not just its opening — and selects the provisions a compliance officer must read, each with a
                  role (governing / disclosure / related) and a one-line reason. It drops look-alikes (rules written
                  only for public or real-estate funds, shared words only) and cited articles whose subject does not
                  match (T&Cs often cite an older numbering of the IFR).

The agent does NOT judge compliance and gives no confidence status: that is the Compliance Analysis Agent's
job. Every candidate it retrieved is returned: the selected provisions first (selected=True, with role and
reason), then the other candidates (selected=False); without the selection step (no LLM) selected=None.
Each hit carries its citation, heading, the paragraphs most related to the clause (excerpt) and where it was
found (found_in); selected and cited provisions also carry their full text.

Without an LLM (no key, server down, LLM_BACKEND unreachable) it falls back to the measured rule ranking:
cited articles first, then the reranked search with Annex 11 kept in the top two.
LLM answers are cached on disk (output/llm_cache, RETRIEVAL_CACHE=0 to disable). The cache key covers the
cache version, backend, endpoint, model, every call parameter and the full messages (prompt, clause, candidates).

    from retriever import RegulationRetriever
    from retrieval_agent import LLM, RetrievalAgent
    agent = RetrievalAgent(RegulationRetriever(), llm=LLM())
    out = agent.run(clause["text"], title=clause.get("title"))
    out["hits"], out["topics"], out["steps"]

CLI:  python3 retrieval_agent.py "نص البند" [--title "عنوان البند"] [--k 5] [--mode dense|hybrid|bm25] [--no-rerank] [--llm]
"""
import argparse, atexit, hashlib, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict

from chunk import search_normalize
from retriever import BM25, PIN_SECTION, tokenize

try:  # settings (OPENAI_API_KEY, RETRIEVAL_MODEL, or LLM_BACKEND=vllm + VLLM_MODEL …) come from the .env
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
ANNEX_ITEM = re.compile(r"› البند \((\d+)\): ([^\]]+)\]")       # Annex 11 chunks are split per numbered item

MAX_TOPICS = 4
MAX_NAMED = 3           # articles the LLM may name from the catalogue in step 1
SEARCH_DEPTH = 20       # results kept from each search; 10 left right articles out (depth note in the README)
POOL_SIZE = 18          # candidate provisions the LLM reads per clause (14/16/18/20 measured, depth note in README)
MAX_SELECTED = 8        # a clause may have up to 4 topics, each with its provision, plus disclosure
EXCERPT_CHARS = 1100    # of each candidate shown to the LLM: its paragraphs most related to the clause
CLAUSE_CHARS = 3000
CACHE_VERSION = "3"     # bump when the meaning of a cached answer changes without the request changing
CATALOGUE_EXTRA = {"CMIR-art031", "CMIR-art063"}   # CMIR provisions listed in the step-1 catalogue
LETTER_ITEM = re.compile(r"^\s*\(?[\u0621-\u064A]{1,2}\)\s")   # "أ) " "هـ) " starts a paragraph of an article

UNDERSTAND_PROMPT = """You help the Regulatory Retrieval Agent of MITHAQ find the Capital Market Authority (CMA)
regulations that apply to one clause of the Terms and Conditions (الشروط والأحكام) of a Saudi PRIVATE investment fund.

Read the clause and return JSON:
{{"topics": [{{"topic": "...", "query": "..."}}], "articles": ["IFR-art…"], "annex11_item": <number or null>}}

- topics: the 1-{max_topics} distinct regulatory subjects the clause deals with (for example: the cap on retail
  investors, the minimum fund size, extending the offering period). Return [] for text with no regulatory subject
  (addresses, contact details, a table of contents, "not applicable").
  A clause that limits or excludes the fund manager's liability, makes investors waive rights or claims, or makes
  an investor right lapse (e.g. losing the right to object by not attending a meeting) ALWAYS has regulatory
  subjects: the manager's liability, the nullity of liability exemptions, unitholders' rights. Never return [] for it.
  "topic": a short Arabic label (2-6 words).
  "query": an Arabic search query (8-20 words) in the regulations' own terminology: name the regulated concept the
  way the regulations name it (the catalogue titles below show their vocabulary) and add its synonyms; do not copy
  the fund's wording (e.g. "نقل ملكية وحدات الصندوق الخاص القيود على نشاط السوق الثانوية" for internal trading).
- articles: up to {max_named} keys from the catalogue whose subject is one of the clause's topics. [] if none clearly is.
- annex11_item: the Annex 11 item that this clause's section corresponds to; 0 for the cover page and introductory
  notices; null if none.

Annex 11 — contents required in a private fund's T&C:
{items}

Catalogue — Investment Funds Regulations provisions that apply to private funds, plus key Capital Market
Institutions Regulations provisions (key: title):
{catalogue}"""

SELECT_PROMPT = """You are the Regulatory Retrieval Agent of MITHAQ. For one clause of the Terms and Conditions of a
Saudi PRIVATE investment fund, select from the candidate regulation sections the ones a compliance officer must read
to assess this clause, most important first. Do NOT assess compliance: only identify the applicable provisions.

Return JSON: {{"selected": [{{"key": "...", "role": "governing|disclosure|related", "topic": <topic number or null>,
"reason": "..."}}]}}
- The clause's topics (from reading it earlier) are listed with numbers. Cover every topic: for each one, select the
  provision(s) that govern it if a candidate does, and give that topic's number. A topic may have no governing
  candidate; do not drop a topic because another topic has more matches. "topic" is null for disclosure items and
  provisions that concern the clause as a whole.
- governing: the section sets the requirement, limit or procedure for the clause's subject and applies to private
  funds (IFR general parts and Part 5, CMIR duties of the fund manager).
- disclosure: an Annex 11 item stating what the T&C must disclose for this part of the document.
- related: definitions or general duties that help read the clause. Use sparingly.
- Order: governing (most specific first), then disclosure, then related. At most {max_selected} entries.
- Check what each candidate actually regulates: its subject, whom it applies to (investors, the fund manager, the
  custodian, the board …) and in what situation. Sharing a term with the clause is not enough.
- Leave out rules written only for public funds, real estate funds or other fund types.
- A candidate marked "cited_in_clause" is named by the clause: select it only if its text deals with the clause's
  subject (T&Cs sometimes cite an older numbering of the regulations).
- A candidate marked "suggested" was proposed from its title when the clause was first read: check its text like any
  other candidate; select it only if it governs one of the topics.
- Each candidate's "text" holds its paragraphs most related to the clause; "…" marks text left out.
- reason: one short Arabic sentence on what the section requires that bears on this clause. No compliance verdict.
- Use only the given keys. Return {{"selected": []}} if no candidate applies."""


# Optional vLLM backend (LLM_BACKEND=vllm; all settings can be set in the .env). The default is a 4-bit 7B model: good Arabic, ~6 GB of GPU memory,
# so it fits next to BGE-M3 + the reranker on the same card.
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
VLLM_PORT = int(os.getenv("VLLM_PORT", 8001))                  # 8000 is the interface
VLLM_GPU_SHARE = os.getenv("VLLM_GPU_SHARE", "0.55")           # share of GPU memory vLLM may take
VLLM_MAX_LEN = os.getenv("VLLM_MAX_LEN", "16384")              # context length (prompt + candidates)
VLLM_START_TIMEOUT = int(os.getenv("VLLM_START_TIMEOUT", 1800))  # the first start downloads the model
VLLM_LOG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "vllm.log"))
CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "llm_cache"))


def _served_models(url):
    from openai import OpenAI
    try:
        return [m.id for m in OpenAI(base_url=url, api_key="EMPTY", timeout=3, max_retries=0).models.list().data]
    except Exception:
        return None


def start_vllm(model=VLLM_MODEL, port=VLLM_PORT):
    """Make sure a vLLM server with `model` answers on localhost:port and return its URL.
    If one is already running (started by hand or by another process) it is reused; otherwise the agent
    starts `vllm serve` itself in the background, waits until it is ready, and stops it when Python exits."""
    url = f"http://localhost:{port}/v1"
    served = _served_models(url)
    if served is not None:
        if model not in served:
            raise RuntimeError(f"another vLLM server on port {port} serves {served}, not {model}")
        return url
    try:
        import vllm  # noqa: F401
    except ImportError:
        raise RuntimeError("vLLM is not installed (pip install vllm). It runs on Linux / WSL2 with an NVIDIA GPU")
    os.makedirs(os.path.dirname(VLLM_LOG), exist_ok=True)
    log = open(VLLM_LOG, "w", encoding="utf-8")
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port),
           "--gpu-memory-utilization", VLLM_GPU_SHARE, "--max-model-len", VLLM_MAX_LEN]
    print(f"starting vLLM ({model}) on port {port}; the first start downloads the model. Log: {VLLM_LOG}", flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    atexit.register(proc.terminate)
    deadline = time.time() + VLLM_START_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:
            with open(VLLM_LOG, encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-15:])
            raise RuntimeError(f"vLLM stopped while starting (see {VLLM_LOG}):\n{tail}")
        if _served_models(url) is not None:
            print(f"vLLM ready: {model}", flush=True)
            return url
        time.sleep(3)
    proc.terminate()
    raise RuntimeError(f"vLLM did not start within {VLLM_START_TIMEOUT}s (see {VLLM_LOG})")


class LLM:
    """Chat model answering in JSON for steps 1 and 3. Default backend: the OpenAI API (OPENAI_API_KEY,
    RETRIEVAL_MODEL, optional OPENAI_BASE_URL). LLM_BACKEND=vllm in the .env runs a local model on vLLM instead,
    started by the agent itself (start_vllm)."""

    def __init__(self, model=None, backend=None):
        from openai import OpenAI
        self.backend = backend or os.getenv("LLM_BACKEND", "openai")
        # the default waits up to 10 min per call: one hung request stalled a whole evaluation run for 30+ min.
        # On timeout the agent falls back to the rule ranking (llm_error step).
        timeout = float(os.getenv("RETRIEVAL_TIMEOUT", 60))
        if self.backend == "vllm":
            self.model = model or VLLM_MODEL
            self.client = OpenAI(base_url=start_vllm(self.model), api_key="EMPTY", timeout=timeout, max_retries=2)
        else:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set (put it in the .env at the repo root)")
            self.model = model or os.getenv("RETRIEVAL_MODEL", "gpt-4o")
            self.client = OpenAI(timeout=timeout, max_retries=2)
        self.base_url = str(self.client.base_url)
        self.cache = os.getenv("RETRIEVAL_CACHE", "1") != "0"
        # reasoning models (gpt-5, o-series) accept only the default temperature
        self.deterministic = not self.model.startswith(("gpt-5", "o1", "o3", "o4"))

    def params(self, max_tokens):
        """Every parameter of the call besides the messages (all of them go into the cache key)."""
        p = {"model": self.model, "response_format": {"type": "json_object"},
             ("max_tokens" if self.backend == "vllm" else "max_completion_tokens"): max_tokens}
        if self.deterministic:
            p["temperature"] = 0
        return p

    def cache_key(self, messages, params):
        return hashlib.sha256(json.dumps({"version": CACHE_VERSION, "backend": self.backend, "base_url": self.base_url,
                                          "params": params, "messages": messages},
                                         ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def ask_json(self, system, user, max_tokens=900):
        """One chat call that must answer with a JSON object → (answer, came_from_cache).
        Only successful, parsed answers are cached; errors are raised and never stored."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        params = self.params(max_tokens)
        key = self.cache_key(messages, params)
        path = os.path.join(CACHE_DIR, key[:2], key + ".json")
        if self.cache and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f), True
        for attempt in range(6):
            try:
                content = self.client.chat.completions.create(messages=messages, **params).choices[0].message.content
                break
            except Exception as e:
                # 429 is also returned when the account has no credit left: waiting does not help there
                no_credit = (getattr(e, "body", None) or {}).get("type") == "insufficient_quota"
                if getattr(e, "status_code", None) == 429 and not no_credit and attempt < 5:
                    time.sleep(20)
                else:
                    raise
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"expected a JSON object, got {type(data).__name__}")
        if self.cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        return data, False


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


def _body(text):
    """Chunk or clause text without its leading breadcrumb."""
    return BREADCRUMB.sub("", text, count=1).strip()


class RetrievalAgent:
    def __init__(self, retriever, *, llm=None, pool_size=POOL_SIZE):
        """llm: a chat model answering in JSON (LLM()); it plans the searches and selects the provisions.
        None = rules only. It can also be attached later (agent.llm = LLM()), as the interface does."""
        self.r, self.llm, self.pool_size = retriever, llm, pool_size
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
        # Annex 11 item number → its chunk (0 = cover page); each T&C section maps to one item
        self.annex_items, self.annex_titles = {}, {}
        for cid in self.sections.get(PIN_SECTION, []):
            m = ANNEX_ITEM.search(self.r.docs[self.r.index[cid]])
            n, t = (int(m.group(1)), m.group(2).strip()) if m else (0, "صفحة الغلاف والمقدمة")
            self.annex_items.setdefault(n, cid)
            self.annex_titles.setdefault(n, t)
        self.items_text = "\n".join(f"{n}) {t}" for n, t in sorted(self.annex_titles.items()))
        # IFR provisions that govern private funds, shown to the LLM so its queries use the regulation's vocabulary
        # + a few CMIR provisions T&C clauses often need but plain search misses (e.g. art 31: liability exemptions
        #   are void; art 63: complaints), so the LLM can name them in step 1
        self.catalogue = "\n".join(f"{key}: {m.get('section_title')}" for key, m in first.items()
                                   if m.get("section_title") and (key in CATALOGUE_EXTRA or (
                                       m["doc_id"] == "investment_funds_regulations"
                                       and self.r._allowed(m, None, None, None, "private"))))
        self._paras, self._vecs = {}, {}                       # unit → paragraphs / their embeddings (memo)

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

    # ---------------------------------------------------------------- agent
    def run(self, text, title=None, k=5, mode="dense", fund_type="private", rerank=True, **search_kw):
        """k is the minimum number of hits: every candidate the agent retrieved is returned (see the module doc)."""
        steps = []
        log = lambda action, **info: steps.append({"step": len(steps) + 1, "action": action, **info})
        kw = dict(mode=mode, fund_type=fund_type, rerank=rerank, **search_kw)
        quick = dict(mode="hybrid" if mode != "bm25" else "bm25", fund_type=fund_type, rerank=False,
                     pin_annex11=False, **search_kw)            # expansion queries: no cross-encoder, the LLM ranks
        depth = max(k, SEARCH_DEPTH)
        llm = self.llm                                         # read once: the interface may attach it meanwhile
        private = fund_type == "private"

        title = clause_title(text, title)
        refs = self.find_references(text)
        cited = list(dict.fromkeys(r["key"] for r in refs if r["key"]))
        log("analyse", title=title or None, words=len(text.split()),
            references=[{"ref": r["ref"], "section": r["key"], "note": r["why"]} for r in refs])

        # 1) understand: topics, regulatory search queries, named articles, Annex 11 item
        plan = None
        if llm is not None:
            try:
                plan, cached = self._understand(llm, title, text)
                log("understand", topics=[t["topic"] for t in plan["topics"]], queries=[t["query"] for t in plan["topics"]],
                    articles=plan["articles"], annex11_item=plan["annex11_item"], cached=cached)
            except Exception as e:
                log("llm_error", stage="understand", error=f"{type(e).__name__}: {e}"[:300])

        # 2) retrieve: one candidate pool from several searches, all inside the fund-type scope
        lists = {"clause": self.r.search(text, k=depth, **{**kw, "pin_annex11": False})}
        if mode != "bm25":
            lists["lexical"] = self.r.search(text, k=depth, mode="bm25", fund_type=fund_type, rerank=False,
                                             pin_annex11=False, **search_kw)
        for n, t in enumerate((plan or {}).get("topics", []), 1):
            lists[f"topic{n}"] = self.r.search(t["query"], k=depth, **quick)
        titled = self.title_match(title, fund_type) if title else []
        log("retrieve", searches=list(lists), depth=depth, top={name: self._units(hits)[:3] for name, hits in lists.items()})
        topics = (plan or {}).get("topics", [])
        qtext = "\n".join([_body(text)[:CLAUSE_CHARS]] + [t["query"] for t in topics])   # what excerpts are scored against
        qvec = self.r.embed_queries([qtext])[0] if self.r._model is not None else None   # dense only when loaded

        named = [a for a in (plan or {}).get("articles", []) if a in self.sections and a != PIN_SECTION
                 and self.r._allowed(self.section_meta[a], None, None, None, fund_type)][:MAX_NAMED]
        annex = ([self.annex_items[plan["annex11_item"]]]
                 if plan and private and plan["annex11_item"] in self.annex_items else [])
        sources = self._sources(lists, titled, cited, named, annex)
        if plan is None:                                       # rules only: the measured fallback ranking
            hits = self._rules_ranking(lists["clause"], cited, private)
            return self._out(hits, k, steps, [], cited, sources, qtext, qvec, judged=False)

        units = self._pool(cited + named + annex, self._fuse(lists, titled))
        log("pool", candidates=units)

        # 3) select the applicable provisions
        excerpts = {u: self._excerpt(u, qtext, qvec) for u in units}
        try:
            selected, cached = self._select(llm, title, text, units, lists, cited, excerpts, topics, named)
            log("select", selected=[(s["key"], s["role"], s["topic"]) for s in selected], cached=cached)
        except Exception as e:
            log("llm_error", stage="select", error=f"{type(e).__name__}: {e}"[:300])
            hits = self._rules_ranking(lists["clause"], cited, private)
            hits += [self._best(u, lists) for u in units]       # keep the whole pool
            return self._out(hits, k, steps, topics, cited, sources, qtext, qvec, judged=False)

        chosen = {s["key"] for s in selected}
        hits = [{**self._best(s["key"], lists), "selected": True, "role": s["role"], "reason": s["reason"],
                 "topic": topics[s["topic"] - 1]["topic"] if s["topic"] else None} for s in selected]
        hits += [{**self._best(u, lists), "selected": False} for u in units if u not in chosen]
        return self._out(hits, k, steps, topics, cited, sources, qtext, qvec, excerpts=excerpts)

    def search(self, query, k=5, **kw):
        """Same contract as RegulationRetriever.search, so build_graph can use the agent unchanged."""
        return self.run(query, k=k, **kw)["hits"]

    # ---------------------------------------------------------------- LLM steps
    def _understand(self, llm, title, text):
        user = f"عنوان البند: {title or '-'}\nموقعه في الشروط والأحكام: {self._crumb(text)}\n\nنص البند:\n{_body(text)[:CLAUSE_CHARS]}"
        system = UNDERSTAND_PROMPT.format(max_topics=MAX_TOPICS, max_named=MAX_NAMED, items=self.items_text,
                                          catalogue=self.catalogue)
        data, cached = llm.ask_json(system, user, 700)
        topics = [{"topic": str(t.get("topic", "")).strip(), "query": str(t.get("query", "")).strip()}
                  for t in data.get("topics") or [] if isinstance(t, dict) and t.get("query")][:MAX_TOPICS]
        named = [a for a in data.get("articles") or [] if isinstance(a, str)]
        item = data.get("annex11_item")
        return {"topics": topics, "articles": named, "annex11_item": item if isinstance(item, int) else None}, cached

    def _select(self, llm, title, text, units, lists, cited, excerpts, topics=(), named=()):
        cards = []
        for u in units:
            h = self._best(u, lists)
            cards.append({"key": u, "citation": h["citation"], "title": self._heading(h),
                          **({"cited_in_clause": True} if h["section_key"] in cited else {}),
                          **({"suggested": True} if u in named else {}), "text": excerpts[u]})
        listed = "\n".join(f"{n}) {t['topic']}" for n, t in enumerate(topics, 1)) or "-"
        user = (f"عنوان البند: {title or '-'}\nموقعه في الشروط والأحكام: {self._crumb(text)}\n\nنص البند:\n"
                f"{_body(text)[:CLAUSE_CHARS]}\n\nموضوعات البند:\n{listed}\n\n"
                f"المواد المرشحة:\n{json.dumps(cards, ensure_ascii=False, indent=1)}")
        data, cached = llm.ask_json(SELECT_PROMPT.format(max_selected=MAX_SELECTED), user, 900)
        out, seen = [], set()
        for s in data.get("selected") or []:
            key = s.get("key") if isinstance(s, dict) else None
            if key in units and key not in seen:                # never a key that was not in the pool
                seen.add(key)
                role = s.get("role") if s.get("role") in ("governing", "disclosure", "related") else "related"
                topic = s.get("topic") if isinstance(s.get("topic"), int) and 1 <= s.get("topic") <= len(topics) else None
                out.append({"key": key, "role": role, "topic": topic, "reason": str(s.get("reason") or "").strip()})
        order = {"governing": 0, "disclosure": 1, "related": 2}
        return sorted(out, key=lambda s: order[s["role"]])[:MAX_SELECTED], cached   # stable: keeps the LLM's order

    # ---------------------------------------------------------------- pool, excerpts
    def _pool(self, priority, fused):
        """Cited, named and Annex 11 units first, then the fused ranking, up to POOL_SIZE (≤ 2 Annex 11 items)."""
        units = list(dict.fromkeys(priority))
        annexes = sum(u.startswith(PIN_SECTION + "-") for u in units)
        for u in fused:
            if len(units) >= self.pool_size:
                break
            if u in units:
                continue
            if u.startswith(PIN_SECTION + "-"):
                if annexes >= 2:
                    continue
                annexes += 1
            units.append(u)
        return units

    def _sources(self, lists, titled, cited, named, annex):
        """unit → where it was found: search names, 'title', 'cited', 'named' (by the LLM), 'annex11_item'."""
        found = defaultdict(list)
        for name, hits in lists.items():
            for u in self._units(hits):
                found[u].append(name)
        for key, _ in titled[:3]:
            found[key].append("title")
        for key, tag in [(c, "cited") for c in cited] + [(a, "named") for a in named] + [(a, "annex11_item") for a in annex]:
            found[key].append(tag)
        return found

    def _paragraphs(self, unit):
        """The unit's text split at its lettered paragraphs (أ) ب) …); numbered sub-items stay with their paragraph."""
        if unit not in self._paras:
            chunks = [unit] if unit.startswith(PIN_SECTION + "-") else self.sections[unit]
            paras, cur = [], []
            for line in "\n".join(_body(self.r.docs[self.r.index[c]]) for c in chunks).split("\n"):
                line = line.strip()
                if not line:
                    continue
                if cur and (LETTER_ITEM.match(line) or sum(map(len, cur)) > 900):
                    paras.append(" ".join(cur)); cur = []
                cur.append(line)
            if cur:
                paras.append(" ".join(cur))
            self._paras[unit] = paras
        return self._paras[unit]

    def _excerpt(self, unit, qtext, qvec=None, budget=EXCERPT_CHARS):
        """The unit's paragraphs most related to the clause (BM25 + embedding similarity), in document order;
        "…" marks paragraphs left out. Short units are returned whole."""
        paras = self._paragraphs(unit)
        if sum(map(len, paras)) + len(paras) <= budget:
            return "\n".join(paras)
        bm = BM25([tokenize(p) or ["_"] for p in paras]).scores(tokenize(qtext))
        top = max(bm) or 1.0
        score = [b / top for b in bm]
        if qvec is not None:
            if unit not in self._vecs:
                self._vecs[unit] = self.r.embed_queries(paras)
            score = [0.5 * s + 0.5 * float(c) for s, c in zip(score, self._vecs[unit] @ qvec)]
        keep, used = [], 0
        for i in sorted(range(len(paras)), key=lambda i: -score[i]):
            if keep and used + len(paras[i]) > budget:
                continue
            keep.append(i)
            used += min(len(paras[i]), budget)
        keep.sort()
        parts = ["…"] if keep[0] > 0 else []
        for n, i in enumerate(keep):
            if n and i != keep[n - 1] + 1:
                parts.append("…")
            parts.append(paras[i][:budget])
        if keep[-1] < len(paras) - 1:
            parts.append("…")
        return "\n".join(parts)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _unit(h):
        """Ranking unit: the article, or the Annex 11 item (all items share one section_key)."""
        return h["id"] if h["section_key"] == PIN_SECTION else h["section_key"]

    def _units(self, hits):
        return list(dict.fromkeys(self._unit(h) for h in hits))

    def _fuse(self, lists, titled, rrf_k=60):
        """Reciprocal-rank fusion of all candidate lists at article / Annex 11 item level."""
        weight = {"clause": 1.0, "lexical": 0.6}
        score = Counter()
        for name, hits in lists.items():
            for rank, u in enumerate(self._units(hits)):
                score[u] += weight.get(name, 0.9) / (rrf_k + rank + 1)
        for rank, (key, _) in enumerate(titled[:3]):
            if key != PIN_SECTION:
                score[key] += 0.5 / (rrf_k + rank + 1)
        return [u for u, _ in score.most_common()]

    def _best(self, unit, lists):
        """Best-ranked chunk of a unit across the lists (the clause search first), else its first chunk."""
        for hits in lists.values():
            for h in hits:
                if self._unit(h) == unit:
                    return dict(h)
        cid = unit if unit.startswith(PIN_SECTION + "-") else self.sections[unit][0]
        return self._hit(cid, None)

    def _rules_ranking(self, primary, cited, private):
        """Without the LLM: cited articles first, then the reranked search with Annex 11 in the top two."""
        hits = [self._hit(self.sections[key][0], None) for key in cited]
        hits += [h for h in primary if h["section_key"] not in cited]
        if private and PIN_SECTION not in [h["section_key"] for h in hits[:2]]:
            pin = next((h for h in hits if h["section_key"] == PIN_SECTION), None)
            if pin:
                hits.remove(pin)
                hits.insert(1, pin)
        return hits

    def _heading(self, h):
        """'المادة الثامنة والتسعون: القيود على نشاط السوق الثانوية' or 'الملحق 11 — البند (11): الرسوم …'."""
        m = h.get("metadata") or self.r.metas[self.r.index[h["id"]]]
        if h["section_key"] == PIN_SECTION:
            item = ANNEX_ITEM.search(h["text"])
            return f"الملحق 11 — البند ({item.group(1)}): {item.group(2).strip()}" if item else "الملحق 11 — صفحة الغلاف"
        label = m.get("article_label") or (f"الملحق {m.get('annex_number')}" if m.get("annex_number") else "")
        return f"{label}: {m.get('section_title')}" if label and m.get("section_title") else label or m.get("section_title") or ""

    def full_text(self, unit):
        """The whole article (all its chunks); an Annex 11 item stays one chunk."""
        chunks = [unit] if unit.startswith(PIN_SECTION + "-") else self.sections.get(unit, [])
        return "\n\n".join(_body(self.r.docs[self.r.index[c]]) for c in chunks)

    @staticmethod
    def _crumb(text):
        m = BREADCRUMB.match(text)
        return PART_SUFFIX.sub("", m.group(1)).strip() if m else "-"

    def _hit(self, chunk_id, score):
        m = self.r.metas[self.r.index[chunk_id]]
        return {"id": chunk_id, "score": score, "section_key": m["section_key"], "citation": m["citation"],
                "binding_status": m["binding_status"], "pdf_pages": m["pdf_pages"],
                "text": self.r.docs[self.r.index[chunk_id]], "metadata": m}

    def _out(self, hits, k, steps, topics, cited, sources, qtext, qvec, judged=True, excerpts=None):
        """Every candidate once, in order. judged=False: no LLM selection ran, so selected is None (unknown),
        not False (rejected). Selected and cited provisions carry their full text; every hit its excerpt."""
        out, seen, excerpts = [], set(), excerpts or {}
        for h in hits:
            u = self._unit(h)
            if u in seen:
                continue
            seen.add(u)
            selected = h.get("selected", False) if judged else None
            is_cited = h["section_key"] in cited
            out.append({**h, "selected": selected, "cited": is_cited, "heading": self._heading(h),
                        "regulation": (h.get("metadata") or {}).get("doc_title"), "found_in": sources.get(u, []),
                        "excerpt": excerpts.get(u) or self._excerpt(u, qtext, qvec),
                        "full_text": self.full_text(u) if selected or is_cited else None})
        return {"hits": out, "topics": [t["topic"] for t in topics], "steps": steps}


if __name__ == "__main__":
    from retriever import RegulationRetriever
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--title")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="dense", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--llm", action="store_true", help="add the LLM steps (OpenAI; LLM_BACKEND=vllm for a local model)")
    a = ap.parse_args()
    out = RetrievalAgent(RegulationRetriever(), llm=LLM() if a.llm else None).run(a.query, a.title, a.k, a.mode,
                                                    rerank=not a.no_rerank and a.mode != "bm25")
    print("topics:", "، ".join(out["topics"]) or "-")
    for s in out["steps"]:
        print(json.dumps(s, ensure_ascii=False))
    for n, h in enumerate(out["hits"], 1):
        mark = h.get("role") if h["selected"] else "candidate"
        print(f"#{n} {h['section_key']:12} {mark:10} {h['heading']}" + (f"\n    {h['reason']}" if h.get("reason") else ""))
