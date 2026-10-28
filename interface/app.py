"""MITHAQ demo interface: upload a T&C file, watch it go through the orchestrator, browse the clauses
with their closest CMA regulations.  Compliance officers sign up / log in with phone number + password
(auth.py) and each sees their own reviews.

    cd interface
    python app.py                    # Regulatory Retrieval Agent + its LLM step (OpenAI, key in .env)
    python app.py --no-llm           # the agent with its rule-based steps only
    python app.py --retriever full   # plain search: BGE-M3 + reranker over the Chroma store, no agent
    python app.py --retriever bm25   # real regulations, lexical search only (no models to download)
    python app.py --retriever stub   # stand-in regulation (OCR + segmentation only)

Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
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
from orchestrator.graph import build_graph  # noqa: E402
from orchestrator.stubs import StubRetriever  # noqa: E402

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import auth  # noqa: E402

UPLOADS = HERE / "uploads"
DATA = HERE / "data"            # finished reviews, so they survive a restart (handy before a demo)
ALLOWED = {".pdf", ".docx"}
K = 5

# ─────────────────────────── retriever (built once, warmed in the background) ───────────────────────────
MODE = "agent"
USE_LLM = True
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
REVIEWS: dict[str, dict] = {}
LOCK = threading.Lock()


def load_saved():
    DATA.mkdir(exist_ok=True)
    for f in DATA.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        REVIEWS[r["id"]] = r


def save(r):
    DATA.mkdir(exist_ok=True)
    (DATA / f"{r['id']}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")


def on_status(review_id, status, **info):
    with LOCK:
        r = REVIEWS[review_id]
        r["status"] = status
        if status == "retrieving":
            r["total"] = info.get("total", r.get("total", 0))
            r["done"] = info.get("done", 0)
        if status == "failed":
            r["error"] = info.get("reason")


def run_review(review_id, path):
    while not ENGINE["ready"]:
        if ENGINE["error"]:
            on_status(review_id, "failed", reason="تعذر تشغيل محرك البحث في الأنظمة: " + ENGINE["error"])
            return
        time.sleep(0.5)
    r = REVIEWS[review_id]
    try:
        graph = build_graph(ENGINE["retriever"], on_status=on_status, k=K, search_kwargs=search_kwargs())
        state = graph.invoke({"review_id": review_id, "file_path": str(path), "results": []},
                             {"max_concurrency": 4})
    except Exception as e:
        on_status(review_id, "failed", reason=f"{type(e).__name__}: {e}")
        state = {}
    with LOCK:
        if state.get("output"):
            r["output"] = enrich(state["output"])
            r["status"] = "done"
        elif r["status"] != "failed":
            r["status"], r["error"] = "failed", state.get("error") or "لم يُستخرج نص أو بنود من الملف."
        r["finished"] = time.time()
        save(r)


def cma_pages(pdf_pages):
    """Chroma keeps the PDF page index ("169,170"); the printed CMA page number is PDF page − 1 (the cover
    has no number), the same rule pipeline/chunk.py uses for page_start / page_end."""
    if isinstance(pdf_pages, str):
        pdf_pages = [int(p) for p in pdf_pages.split(",") if p.strip().isdigit()]
    printed = [p - 1 for p in pdf_pages or [] if p > 1]
    return (min(printed), max(printed)) if printed else (None, None)


DOC_NAMES = {"IFR": "لائحة صناديق الاستثمار", "CMIR": "لائحة مؤسسات السوق المالية",
             "CGR": "لائحة حوكمة الشركات", "SIFI": "تعليمات صناديق الاستثمار المبسطة"}


def section_label(key):
    """IFR-art092 → لائحة صناديق الاستثمار، المادة 92 (for the reference-mismatch note)."""
    doc, _, sec = key.partition("-")
    kind, num = ("المادة", sec[3:].lstrip("0")) if sec.startswith("art") else ("الملحق", sec[3:])
    return f"{DOC_NAMES.get(doc, doc)}، {kind} {num}"


def agent_view(c):
    """Pull what the interface shows out of the retrieval agent's steps (absent with plain search)."""
    steps = c.get("retrieval_steps") or []
    last = lambda action: next((s for s in reversed(steps) if s["action"] == action), {})
    c["decision"] = c.get("retrieval_decision")
    c["decision_reason"] = last("decide").get("reason")
    c["llm_reason"] = last("llm_finish").get("reason")
    c["llm_error"] = last("llm_error").get("error")
    c["reference_mismatch"] = [section_label(k) for k in last("reference_mismatch").get("references", [])]


def enrich(output):
    for c in output["clauses"]:
        agent_view(c)
        text = c["text"]
        if text.startswith("[") and "]" in text:      # segmentation prefixes a breadcrumb: [الشروط والأحكام › …]
            c["breadcrumb"], text = text[1:text.index("]")], text[text.index("]") + 1:].strip()
        c["body"] = text
        for n, reg in enumerate(c["regulations"], 1):
            reg["rank"] = n
            reg["page_start"], reg["page_end"] = cma_pages(reg.get("pdf_pages"))
            reg["document"] = reg["citation"].split("،")[0].strip()
            body = reg["text"]
            reg["body"] = body[body.index("]") + 1:].strip() if body.startswith("[") and "]" in body else body
    return output


def summary(r):
    out = r.get("output") or {}
    clauses = out.get("clauses", [])
    return {k: r.get(k) for k in ("id", "filename", "size", "created", "finished", "status", "done", "total",
                                  "error", "mode", "llm")} | {
        "clauses": len(clauses),
        "regulations": len({g["id"] for c in clauses for g in c["regulations"]}),
        "needs_review": out.get("document_needs_review", False),
        # retrieval agent: clauses to check by hand, references to an old numbering, LLM calls that failed
        "to_review": sum(c.get("decision") in ("uncertain", "no_match") for c in clauses),
        "mismatches": sum(bool(c.get("reference_mismatch")) for c in clauses),
        "llm_errors": sum(bool(c.get("llm_error")) for c in clauses),
    }


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
    phone: str
    password: str


class Login(BaseModel):
    phone: str
    password: str


def start_session(response, session, user):
    response.set_cookie(COOKIE, session, httponly=True, samesite="lax", max_age=auth.SESSION_TTL)
    return user


@app.post("/api/auth/signup")
def signup(body: SignUp, response: Response):
    return start_session(response, *ACCOUNTS.signup(body.name, body.phone, body.password))


@app.post("/api/auth/login")
def login(body: Login, response: Response):
    return start_session(response, *ACCOUNTS.login(body.phone, body.password))


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
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info(user=Depends(current_user)):
    return {"mode": MODE, "ready": ENGINE["ready"], "error": ENGINE["error"], "k": K, "llm": ENGINE["llm"]}


@app.get("/api/reviews")
def list_reviews(user=Depends(current_user)):
    with LOCK:
        return sorted((summary(r) for r in REVIEWS.values() if mine(r, user)), key=lambda s: -s["created"])


@app.get("/api/reviews/{review_id}")
def get_review(review_id: str, user=Depends(current_user)):
    with LOCK:
        if review_id not in REVIEWS or not mine(REVIEWS[review_id], user):
            raise HTTPException(404, "المراجعة غير موجودة")
        r = REVIEWS[review_id]
        return summary(r) | {"output": r.get("output")}


@app.get("/api/reviews/{review_id}/json")
def download(review_id: str, user=Depends(current_user)):
    r = REVIEWS.get(review_id)
    if not r or not r.get("output") or not mine(r, user):
        raise HTTPException(404, "لا توجد نتائج")
    return JSONResponse(r["output"], headers={
        "Content-Disposition": f'attachment; filename="mithaq-{review_id}.json"'})


@app.delete("/api/reviews/{review_id}")
def delete(review_id: str, user=Depends(current_user)):
    with LOCK:
        if review_id not in REVIEWS or not mine(REVIEWS[review_id], user):
            raise HTTPException(404, "المراجعة غير موجودة")
        REVIEWS.pop(review_id)
    (DATA / f"{review_id}.json").unlink(missing_ok=True)
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
    with LOCK:
        REVIEWS[review_id] = {"id": review_id, "filename": name, "size": len(data), "created": time.time(),
                              "status": "received", "done": 0, "total": 0, "mode": MODE, "llm": ENGINE["llm"], "owner": user["id"]}
    threading.Thread(target=run_review, args=(review_id, dest), daemon=True).start()
    return {"id": review_id}


def main():
    global MODE, USE_LLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="agent", choices=["agent", "full", "bm25", "stub"])
    ap.add_argument("--no-llm", action="store_true", help="agent without its LLM step")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    MODE, USE_LLM = a.retriever, not a.no_llm
    load_saved()
    threading.Thread(target=load_engine, daemon=True).start()
    import uvicorn
    print(f"\n  MITHAQ  ➜  http://{a.host}:{a.port}   (retriever: {MODE}{'' if USE_LLM else ', no LLM'})\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
