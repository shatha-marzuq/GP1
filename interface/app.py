"""MITHAQ demo interface: upload a T&C file, watch it go through the orchestrator, browse the clauses
with their closest CMA regulations.  Compliance officers sign up with email + password (confirmed by a code
sent to the email) and log in with email + password (auth.py); each sees their own reviews.

    cd interface
    python app.py                    # Regulatory Retrieval Agent + its LLM step (OpenAI, key in .env)
    python app.py --no-llm           # the agent with its rule-based steps only
    python app.py --seg-agent        # + the LLM segmentation step of the Document Processing Agent (checks/fixes the clauses; with --no-llm: rules + check)
    python app.py --retriever full   # plain search: BGE-M3 + reranker over the Chroma store, no agent
    python app.py --retriever bm25   # real regulations, lexical search only (no models to download)
    python app.py --retriever stub   # stand-in regulation (OCR + segmentation only)

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(HERE))

from orchestrator import paths  # noqa: E402,F401  (makes docproc / pipeline / backend importable)
from orchestrator.graph import SegmentationAgent, build_graph, default_segment  # noqa: E402
from orchestrator.stubs import StubRetriever  # noqa: E402

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from sqlalchemy import select  # noqa: E402

import auth  # noqa: E402
import db  # noqa: E402

UPLOADS = HERE / "uploads"      # uploaded files stay on the device that processed them; reviews go to the database
ALLOWED = {".pdf", ".docx"}
K = 5

# ─────────────────────────── retriever (built once, warmed in the background) ───────────────────────────
MODE = "agent"
USE_LLM = True
SEG_AGENT = False   # --seg-agent: the Document Processing Agent's segmentation with its LLM instead of the rules only
ENGINE = {"ready": False, "error": None, "retriever": None, "llm": None}


def search_kwargs():
    return {"mode": "bm25", "rerank": False} if MODE == "bm25" else {}


def load_engine():
    try:
        if MODE == "stub":
            ENGINE["retriever"] = StubRetriever()
        else:
            from retriever import RegulationRetriever
            r = RegulationRetriever()
            if MODE in ("agent", "full"):   # download/load BGE-M3 + reranker now, not during the first upload
                r.search("رسوم الاشتراك في الصندوق", k=1)
            if MODE == "agent":
                from retrieval_agent import RetrievalAgent   # importing it loads the project .env
                r = RetrievalAgent(r)
                if USE_LLM:   # OpenAI is instant; vLLM (LLM_BACKEND=vllm) can take minutes to start:
                    threading.Thread(target=attach_llm, args=(r,), daemon=True).start()   # rules meanwhile
            ENGINE["retriever"] = r
        ENGINE["ready"] = True
    except Exception as e:
        ENGINE["error"] = f"{type(e).__name__}: {e}"


def attach_llm(agent):
    """Give the agent its LLM step: OpenAI by default (OPENAI_API_KEY in the .env), or a local vLLM model
    with LLM_BACKEND=vllm (which can take minutes to start). Until then, or if it can't be reached, the
    agent works with its rule-based steps."""
    from retrieval_agent import LLM
    try:
        agent.llm = LLM()
    except Exception as e:
        print(f"  LLM step off, the agent uses its rule-based steps only: {e}", flush=True)
        return
    ENGINE["llm"] = agent.llm.model
    print(f"  LLM step on: {agent.llm.backend} / {agent.llm.model}", flush=True)


# ─────────────────────────────────────────── reviews ───────────────────────────────────────────
# The database keeps only the finished result of a review (db.py). While a review is processed, its progress
# (status, done/total) lives in PROGRESS, in this server's memory: it means something only during the
# processing. A review that is not in PROGRESS is done if it has an output, else it failed: the review keeps a
# short Arabic reason for the officer (db.Review.error), and the technical details go to the server log only.
# LOCK keeps the progress updates in order.
LOCK = threading.Lock()
PROGRESS: dict[str, dict] = {}      # review_id -> {"status", "done", "total"} while it is processed

ARABIC = re.compile(r"[؀-ۿ]")
NO_CLAUSES = "لم يُستخرج نص أو بنود من الملف."
UNEXPECTED = "حدث خطأ غير متوقع أثناء مراجعة المستند. حاول رفعه مرة أخرى."
ENGINE_DOWN = "تعذّر تشغيل محرك البحث في الأنظمة. حاول مرة أخرى لاحقًا."
INTERRUPTED = "توقفت المراجعة قبل اكتمالها. ارفع الملف مرة أخرى."   # e.g. the server restarted mid-review


def officer_message(reason):
    """A failure reason as the officer sees it. The Document Processing Agent's reasons are already short Arabic
    messages (e.g. the file is password-protected) and are kept; anything technical (an exception, an English
    message) becomes a general message, and its details stay in the server log."""
    if not reason or reason == "no text or clauses found":
        return NO_CLAUSES
    return reason if ARABIC.search(reason) else UNEXPECTED


def init_db():
    """Create the tables (dropping the columns no longer kept) and bring in the old JSON files."""
    db.init()


def on_status(review_id, status, **info):
    if status == "failed":
        print(f"  review {review_id} failed: {info.get('reason')}", flush=True)
        with db.session() as s:                             # the reason the officer will see
            r = s.get(db.Review, review_id)
            if r is not None and r.output is None:
                r.error = info.get("message") or officer_message(info.get("reason"))
    if status == "done":                                    # done = its output is saved (run_review)
        return
    with LOCK:
        p = PROGRESS.get(review_id)
        if p is None:
            return
        p["status"] = status
        if status == "retrieving":
            if "total" in info:                             # first update: done 0 of total
                p["total"], p["done"] = info["total"], info.get("done", 0)
            else:                                           # parallel branches can report out of order
                p["done"] = max(p["done"], info.get("done", 0))


def run_review(review_id, path):
    try:
        output = process(review_id, path)
        if output:
            with LOCK, db.session() as s:
                r = s.get(db.Review, review_id)
                if r is not None:                           # not deleted while it was running
                    r.output = enrich(output)
    finally:
        with LOCK:                                          # after the output is saved: the review is done
            PROGRESS.pop(review_id, None)


def process(review_id, path):
    """Runs the orchestrator on the file: its output, or None when the review failed."""
    while not ENGINE["ready"]:
        if ENGINE["error"]:
            on_status(review_id, "failed", reason="search engine failed to start: " + ENGINE["error"],
                      message=ENGINE_DOWN)
            return None
        time.sleep(0.5)
    try:
        segment = SegmentationAgent(use_llm=USE_LLM) if SEG_AGENT else default_segment
        graph = build_graph(ENGINE["retriever"], segment=segment, on_status=on_status, k=K,
                            search_kwargs=search_kwargs())
        state = graph.invoke({"review_id": review_id, "file_path": str(path), "results": []},
                             {"max_concurrency": 8})
    except Exception as e:
        on_status(review_id, "failed", reason=f"{type(e).__name__}: {e}", message=UNEXPECTED)
        return None
    with LOCK:
        reported = PROGRESS.get(review_id, {}).get("status") == "failed"
    if not state.get("output") and not reported:
        on_status(review_id, "failed", reason=state.get("error") or NO_CLAUSES)
    return state.get("output")


def cma_pages(pdf_pages):
    """Chroma keeps the PDF page index ("169,170"); the printed CMA page number is PDF page − 1 (the cover
    has no number), the same rule pipeline/chunk.py uses for page_start / page_end."""
    if isinstance(pdf_pages, str):
        pdf_pages = [int(p) for p in pdf_pages.split(",") if p.strip().isdigit()]
    printed = [p - 1 for p in pdf_pages or [] if p > 1]
    return (min(printed), max(printed)) if printed else (None, None)


PART = re.compile(r"\s*\(الجزء[^)]*\)")


def agent_view(c):
    """What the interface needs from the agent's trace: whether its LLM steps ran for this clause.
    Selection, role and reason of each provision are on the regulations themselves."""
    steps = c.get("retrieval_steps") or []
    c["llm_error"] = next((s.get("error") for s in steps if s["action"] == "llm_error"), None)
    c.setdefault("topics", [])


def enrich(output):
    for c in output["clauses"]:
        agent_view(c)
        text = c["text"]
        if text.startswith("[") and "]" in text:      # segmentation prefixes a breadcrumb: [الشروط والأحكام › …]
            c["breadcrumb"], text = text[1:text.index("]")], text[text.index("]") + 1:].strip()
        parts = [PART.sub("", x).strip() for x in (c.get("breadcrumb") or "").split("›")]
        c["section"] = parts[1] if len(parts) > 1 else (parts[0] or c.get("title") or "")   # the T&C section
        c["body"] = text
        for n, reg in enumerate(c["regulations"], 1):
            reg["rank"] = n
            reg["page_start"], reg["page_end"] = cma_pages(reg.get("pdf_pages"))
            reg["document"] = reg["citation"].split("،")[0].strip()
            body = reg.pop("text", None) or reg.get("excerpt") or ""      # plain search keeps the matched chunk
            body = body[body.index("]") + 1:].strip() if body.startswith("[") and "]" in body else body
            reg["heading"] = reg.get("heading") or reg["citation"]
            reg["excerpt"] = reg.get("excerpt") or body   # full_text: selected / cited only, else /api/articles
    return output


def output_stats(out):
    """Counts shown in the reviews list, computed from the result (not stored)."""
    out = out or {}
    clauses = out.get("clauses", [])
    return {
        "clauses": len(clauses),
        # distinct provisions the agent selected (all retrieved ones for plain search)
        "regulations": len({g.get("section_key") or g["id"] for c in clauses for g in c["regulations"]
                            if g.get("selected") is not False}),
        "needs_review": out.get("document_needs_review", False),
        "llm_errors": sum(bool(c.get("llm_error")) for c in clauses),   # clauses that fell back to the rules
    }


def running_now():
    """A copy of PROGRESS. Taken before reading the reviews: a review it lacks had its output saved already."""
    with LOCK:
        return {k: dict(v) for k, v in PROGRESS.items()}


def summary(r, running):
    progress = running.get(r["id"]) or {"status": "done" if r.get("output") else "failed", "done": 0, "total": 0}
    error = None
    if progress["status"] == "failed":                      # never a raw error; no reason = stopped mid-way
        error = officer_message(r["error"]) if r.get("error") else INTERRUPTED
    return ({k: r.get(k) for k in ("id", "filename", "created")} | progress | {"error": error}
            | {"mode": MODE} | output_stats(r.get("output")))       # mode: the page warns in stub mode


# ─────────────────────────────────────────── API ───────────────────────────────────────────
app = FastAPI(title="MITHAQ")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
ACCOUNTS = auth.Accounts()
COOKIE = "mithaq_session"


@app.exception_handler(auth.AuthError)
def auth_error(request, e: auth.AuthError):
    return JSONResponse({"detail": e.message}, status_code=e.status)


def current_user(request: Request):
    user = ACCOUNTS.user_for(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, "سجّل الدخول أولًا")
    return user


def mine(r, user):
    return r.get("owner") == user["id"]   # each user sees only their own reviews


class SignUp(BaseModel):
    name: str
    email: str
    password: str


class Login(BaseModel):
    email: str
    password: str


class Email(BaseModel):
    email: str


class Verify(BaseModel):
    email: str
    code: str


class Name(BaseModel):
    name: str


class Password(BaseModel):
    password: str


class NewPassword(BaseModel):
    current: str
    new: str


def start_session(response, session, user):
    response.set_cookie(COOKIE, session, httponly=True, samesite="lax", max_age=auth.SESSION_TTL)
    return user


@app.post("/api/auth/signup")
def signup(body: SignUp, response: Response):
    """With SMTP in .env: checks the details and emails a code; the account is created by /api/auth/verify.
    Without it: the account is created and logged in right away (no code)."""
    out = ACCOUNTS.signup(body.name, body.email, body.password)
    return out if isinstance(out, dict) else start_session(response, *out)


@app.post("/api/auth/resend")
def resend(body: Email):
    return ACCOUNTS.resend(body.email)


@app.post("/api/auth/verify")
def verify(body: Verify, response: Response):
    """Step 2: the code from the email. Creates the account and logs in."""
    return start_session(response, *ACCOUNTS.verify(body.email, body.code))


@app.post("/api/auth/login")
def login(body: Login, response: Response):
    return start_session(response, *ACCOUNTS.login(body.email, body.password))


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    ACCOUNTS.logout(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return ACCOUNTS.public(user)


# ─────────── my account: the officer's own details; nothing new is stored for it
@app.get("/api/account")
def account(user=Depends(current_user)):
    """The officer's name and email (the counts of their documents are on the dashboard)."""
    return ACCOUNTS.public(user)


@app.post("/api/account/name")
def rename(body: Name, user=Depends(current_user)):
    return ACCOUNTS.rename(user["id"], body.name)


@app.post("/api/account/password")
def change_password(body: NewPassword, request: Request, user=Depends(current_user)):
    """The current password, then the new one; the officer's other devices are logged out."""
    return ACCOUNTS.change_password(user["id"], body.current, body.new, request.cookies.get(COOKIE))


@app.post("/api/account/delete")
def delete_account(body: Password, response: Response, user=Depends(current_user)):
    """Deletes the account, its sessions and reviews (after the password), and the reviews' uploaded files."""
    with LOCK:                                          # no progress update writes a review meanwhile
        review_ids = ACCOUNTS.delete_account(user["id"], body.password)
    for review_id in review_ids:
        shutil.rmtree(UPLOADS / review_id, ignore_errors=True)
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/")
def index():
    # no-cache: the browser always asks for the latest page (an old copy kept the old sign-up screen)
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/info")
def info(user=Depends(current_user)):
    return {"mode": MODE, "ready": ENGINE["ready"], "error": ENGINE["error"], "k": K, "llm": ENGINE["llm"],
            "seg_agent": SEG_AGENT}


@app.get("/api/articles/{chunk_id}")
def article(chunk_id: str, user=Depends(current_user)):
    """Full text of the provision a retrieved chunk belongs to (the whole article; an Annex 11 item alone),
    for candidates whose full text is not stored in the review."""
    base = ENGINE["retriever"]
    base = getattr(base, "r", base)                       # the agent wraps the retriever
    if not hasattr(base, "index") or chunk_id not in base.index:
        raise HTTPException(404, "المادة غير موجودة")
    key = base.metas[base.index[chunk_id]]["section_key"]
    ids = [chunk_id] if key == "IFR-anx11" else [i for i, m in zip(base.ids, base.metas) if m["section_key"] == key]
    strip = lambda t: t[t.index("]") + 1:].strip() if t.startswith("[") and "]" in t else t
    return {"id": chunk_id, "section_key": key, "full_text": "\n\n".join(strip(base.docs[base.index[i]]) for i in ids)}


# ───────── regulations page: a whole CMA regulation, laid out by part, chapter and article, so a clause's
# matched provisions can be shown (and highlighted) in their place. Built once per regulation from the
# knowledge base itself (the same chunks the agent searches), so the chunk IDs in a review point into it.
REG_DOCS: dict[str, dict] = {}
REG_ORDER = {"art": 0, "anx": 1}


def knowledge_base():
    base = ENGINE["retriever"]
    base = getattr(base, "r", base)                       # the agent wraps the retriever
    if not hasattr(base, "metas"):
        raise HTTPException(503, "قاعدة اللوائح غير جاهزة بعد، حاول بعد قليل.")
    return base


def _section_order(key):
    """'IFR-art093' → (0, 93); 'IFR-anx11' → (1, 11): articles first, then annexes, in number order."""
    m = re.match(r"[A-Z]+-([a-z]+)(\d+)", key)
    return (REG_ORDER.get(m.group(1), 2), int(m.group(2))) if m else (3, 0)


def build_regulation(doc_id):
    base = knowledge_base()
    title, sections = None, {}
    for cid, text, m in zip(base.ids, base.docs, base.metas):
        if m.get("doc_id") != doc_id:
            continue
        title = title or m.get("doc_title")
        crumb = ""
        if text.startswith("[") and "]" in text:
            crumb, text = text[1:text.index("]")], text[text.index("]") + 1:].strip()
        parts = [PART.sub("", x).strip() for x in crumb.split("›") if x.strip()]
        s = sections.setdefault(m["section_key"], {"key": m["section_key"], "pdf_pages": [], "chunks": []})
        s["pdf_pages"] += [int(p) for p in str(m.get("pdf_pages") or "").split(",") if p.strip().isdigit()]
        s["chunks"].append({"id": cid, "parts": parts, "text": text})
    if not sections:
        raise HTTPException(404, "اللائحة غير موجودة")
    out = []
    for key in sorted(sections, key=_section_order):
        s = sections[key]
        s["chunks"].sort(key=lambda c: c["id"])
        # breadcrumb: regulation › part › chapter › article; the parts all chunks share give the article's
        # heading and place, the rest names a sub-item (Annex 11: "البند (3): …")
        common = s["chunks"][0]["parts"]
        for c in s["chunks"][1:]:
            n = 0
            while n < min(len(common), len(c["parts"])) and common[n] == c["parts"][n]:
                n += 1
            common = common[:n]
        s["heading"] = common[-1] if common else key
        s["path"] = common[1:-1] or (["الملاحق"] if "-anx" in key else [])
        for c in s["chunks"]:
            c["sub"] = " › ".join(c.pop("parts")[len(common):]) or None
        s["page_start"], s["page_end"] = cma_pages(s.pop("pdf_pages"))
        out.append(s)
    return {"id": doc_id, "title": title, "sections": out}


@app.get("/api/regulations")
def regulations(user=Depends(current_user)):
    """The regulations in the knowledge base: id, title and the short code their chunk IDs start with."""
    base, docs = knowledge_base(), {}
    for cid, m in zip(base.ids, base.metas):
        docs.setdefault(m["doc_id"], {"id": m["doc_id"], "title": m.get("doc_title"), "short": cid.split("-")[0]})
    return list(docs.values())


@app.get("/api/regulations/{doc_id}")
def regulation(doc_id: str, user=Depends(current_user)):
    if doc_id not in REG_DOCS:
        REG_DOCS[doc_id] = build_regulation(doc_id)
    return REG_DOCS[doc_id]


def owned_by(user):
    return db.Review.owner == user["id"]


def find_review(s, review_id, user, with_output=False):
    r = s.get(db.Review, review_id)
    if r is None or not mine(r.as_dict(), user):
        raise HTTPException(404, "المراجعة غير موجودة")
    return r.as_dict(with_output)


@app.get("/api/reviews")
def list_reviews(user=Depends(current_user)):
    running = running_now()
    with db.session() as s:
        rows = s.scalars(select(db.Review).where(owned_by(user)).order_by(db.Review.created.desc()))
        return [summary(r.as_dict(with_output=True), running) for r in rows]


@app.get("/api/reviews/{review_id}")
def get_review(review_id: str, user=Depends(current_user)):
    running = running_now()
    with db.session() as s:
        r = find_review(s, review_id, user, with_output=True)
    return summary(r, running) | {"output": for_browser(r.get("output"))}


def for_browser(output):
    """What the results page needs: the saved review keeps everything, but the page gets no agent trace and
    no excerpt except for the provisions the agent selected (others are fetched when opened)."""
    if not output:
        return output
    clauses = []
    for c in output["clauses"]:
        regs = [{k: v for k, v in g.items() if not (k == "excerpt" and g.get("selected") is not True)}
                for g in c["regulations"]]
        clauses.append({k: v for k, v in c.items() if k != "retrieval_steps"} | {"regulations": regs})
    return output | {"clauses": clauses}


# The JSON export (/api/reviews/{id}/json) was removed: compliance officers don't need it. Reviews are
# saved in the database (db.py), so they survive a restart and the team sees them on every device.


@app.delete("/api/reviews/{review_id}")
def delete(review_id: str, user=Depends(current_user)):
    with LOCK, db.session() as s:
        find_review(s, review_id, user)
        s.delete(s.get(db.Review, review_id))
    return {"ok": True}


@app.post("/api/reviews")
async def upload(file: UploadFile = File(...), user=Depends(current_user)):
    name = Path(file.filename or "document").name
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(400, "الصيغ المدعومة: PDF و Word (docx)")
    review_id = uuid.uuid4().hex[:8]
    dest = UPLOADS / review_id / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    with LOCK:
        PROGRESS[review_id] = {"status": "received", "done": 0, "total": 0}
    with db.session() as s:
        s.add(db.Review(id=review_id, owner=user["id"], filename=name, created=time.time()))
    threading.Thread(target=run_review, args=(review_id, dest), daemon=True).start()
    return {"id": review_id}


def main():
    global MODE, USE_LLM, SEG_AGENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="agent", choices=["agent", "full", "bm25", "stub"])
    ap.add_argument("--no-llm", action="store_true", help="agent without its LLM step")
    ap.add_argument("--seg-agent", action="store_true", help="segmentation with the LLM review (Document Processing Agent)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    MODE, USE_LLM, SEG_AGENT = a.retriever, not a.no_llm, a.seg_agent
    init_db()
    threading.Thread(target=load_engine, daemon=True).start()
    import uvicorn
    print(f"\n  MITHAQ  ➜  http://{a.host}:{a.port}   (retriever: {MODE}{'' if USE_LLM else ', no LLM'}{', LLM segmentation' if SEG_AGENT else ''}"
          f" | database: {db.describe()})\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
