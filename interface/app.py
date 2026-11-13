"""MITHAQ demo interface: upload a T&C file, watch it go through the orchestrator, browse the clauses
with their closest CMA regulations.  Compliance officers sign up with email + password (confirmed by a code
sent to the email) and log in with email + password (auth.py); each sees their own reviews.

    cd interface
    python app.py                    # Regulatory Retrieval Agent + its LLM step (OpenAI, key in .env)
    python app.py --no-llm           # the agent with its rule-based steps only
    python app.py --seg-agent        # + the Segmentation Agent (its LLM checks/fixes the clauses; with --no-llm: rules + check)
    python app.py --retriever full   # plain search: BGE-M3 + reranker over the Chroma store, no agent
    python app.py --retriever bm25   # real regulations, lexical search only (no models to download)
    python app.py --retriever stub   # stand-in regulation (OCR + segmentation only)

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import re
import socket
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

from sqlalchemy import or_, select  # noqa: E402

import auth  # noqa: E402
import db  # noqa: E402

UPLOADS = HERE / "uploads"      # uploaded files stay on the device that processed them; reviews go to the database
HOST = socket.gethostname()
ALLOWED = {".pdf", ".docx"}
K = 5

# ─────────────────────────── retriever (built once, warmed in the background) ───────────────────────────
MODE = "agent"
USE_LLM = True
SEG_AGENT = False   # --seg-agent: segment through the Segmentation Agent instead of the rules only
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
# Reviews live in the database (db.py). LOCK keeps this server's progress updates for one review in order.
LOCK = threading.Lock()
RUNNING = ("received", "extracting", "segmenting", "retrieving")


def init_db():
    """Create the tables, bring in the old JSON files, and close reviews this device was processing when it
    stopped (they would otherwise stay "in progress" forever)."""
    db.init()
    with db.session() as s:
        for r in s.scalars(select(db.Review).where(db.Review.host == HOST, db.Review.status.in_(RUNNING))):
            r.status, r.error, r.finished = "failed", "توقف الخادم أثناء المعالجة. ارفع الملف مرة ثانية.", time.time()
        for r in s.scalars(select(db.Review).where(db.Review.stats.is_(None), db.Review.status == "done")):
            r.stats = output_stats(r.output)                                   # reviews from the JSON files


def on_status(review_id, status, **info):
    with LOCK, db.session() as s:
        r = s.get(db.Review, review_id)
        if r is None:                                       # deleted while it was running
            return
        r.status = status
        if status == "retrieving":
            if "total" in info:                             # first update: done 0 of total
                r.total, r.done = info["total"], info.get("done", 0)
            else:                                           # parallel branches can report out of order
                r.done = max(r.done or 0, info.get("done", 0))
        if status == "failed":
            r.error = info.get("reason")


def run_review(review_id, path):
    while not ENGINE["ready"]:
        if ENGINE["error"]:
            on_status(review_id, "failed", reason="تعذر تشغيل محرك البحث في الأنظمة: " + ENGINE["error"])
            return
        time.sleep(0.5)
    try:
        segment = SegmentationAgent(use_llm=USE_LLM) if SEG_AGENT else default_segment
        graph = build_graph(ENGINE["retriever"], segment=segment, on_status=on_status, k=K,
                            search_kwargs=search_kwargs())
        state = graph.invoke({"review_id": review_id, "file_path": str(path), "results": []},
                             {"max_concurrency": 8})
    except Exception as e:
        on_status(review_id, "failed", reason=f"{type(e).__name__}: {e}")
        state = {}
    with LOCK, db.session() as s:
        r = s.get(db.Review, review_id)
        if r is None:
            return
        if state.get("output"):
            r.output = enrich(state["output"])
            r.stats = output_stats(r.output)
            r.status = "done"
        elif r.status != "failed":
            r.status, r.error = "failed", state.get("error") or "لم يُستخرج نص أو بنود من الملف."
        r.finished = time.time()


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
    """Counts shown in the reviews list, saved with the review so the list doesn't load every result."""
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


def summary(r):
    return {k: r.get(k) for k in ("id", "filename", "size", "created", "finished", "status", "done", "total",
                                  "error", "mode", "llm")} | (r.get("stats") or output_stats(None))


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
    return r.get("owner") in (None, user["id"])   # reviews saved before accounts existed stay visible


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


def owned_by(user):
    return or_(db.Review.owner == user["id"], db.Review.owner.is_(None))


def find_review(s, review_id, user, with_output=False):
    r = s.get(db.Review, review_id)
    if r is None or not mine(r.as_dict(), user):
        raise HTTPException(404, "المراجعة غير موجودة")
    return r.as_dict(with_output)


@app.get("/api/reviews")
def list_reviews(user=Depends(current_user)):
    with db.session() as s:
        rows = s.scalars(select(db.Review).where(owned_by(user)).order_by(db.Review.created.desc()))
        return [summary(r.as_dict()) for r in rows]


@app.get("/api/reviews/{review_id}")
def get_review(review_id: str, user=Depends(current_user)):
    with db.session() as s:
        r = find_review(s, review_id, user, with_output=True)
    return summary(r) | {"output": for_browser(r.get("output"))}


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
    data = await file.read()
    dest.write_bytes(data)
    with db.session() as s:
        s.add(db.Review(id=review_id, owner=user["id"], filename=name, size=len(data), file_path=str(dest),
                        host=HOST, created=time.time(), status="received", done=0, total=0, mode=MODE,
                        llm=ENGINE["llm"]))
    threading.Thread(target=run_review, args=(review_id, dest), daemon=True).start()
    return {"id": review_id}


def main():
    global MODE, USE_LLM, SEG_AGENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="agent", choices=["agent", "full", "bm25", "stub"])
    ap.add_argument("--no-llm", action="store_true", help="agent without its LLM step")
    ap.add_argument("--seg-agent", action="store_true", help="segment through the Segmentation Agent")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    MODE, USE_LLM, SEG_AGENT = a.retriever, not a.no_llm, a.seg_agent
    init_db()
    threading.Thread(target=load_engine, daemon=True).start()
    import uvicorn
    print(f"\n  MITHAQ  ➜  http://{a.host}:{a.port}   (retriever: {MODE}{'' if USE_LLM else ', no LLM'}{', segmentation agent' if SEG_AGENT else ''}"
          f" | database: {db.describe()})\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
