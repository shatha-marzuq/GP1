"""MITHAQ demo interface: upload a T&C file, watch it go through the orchestrator, browse the clauses
with their closest CMA regulations.  Only what is built today runs (OCR → segmentation → retrieval);
compliance verification and the report are shown as "coming soon".

    cd interface
    python app.py                    # real retrieval: BGE-M3 + reranker over the Chroma store
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

from orchestrator import paths  # noqa: E402,F401  (makes docproc / pipeline / backend importable)
from orchestrator.graph import build_graph  # noqa: E402
from orchestrator.stubs import StubRetriever  # noqa: E402

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

UPLOADS = HERE / "uploads"
DATA = HERE / "data"            # finished reviews, so they survive a restart (handy before a demo)
ALLOWED = {".pdf", ".docx"}
K = 5

# ─────────────────────────── retriever (built once, warmed in the background) ───────────────────────────
MODE = "full"
ENGINE = {"ready": False, "error": None, "retriever": None}


def search_kwargs():
    return {"mode": "bm25", "rerank": False} if MODE == "bm25" else {}


def load_engine():
    try:
        if MODE == "stub":
            ENGINE["retriever"] = StubRetriever()
        else:
            from retriever import RegulationRetriever
            r = RegulationRetriever()
            if MODE == "full":       # download/load BGE-M3 + reranker now, not during the first upload
                r.search("رسوم الاشتراك في الصندوق", k=1)
            ENGINE["retriever"] = r
        ENGINE["ready"] = True
    except Exception as e:
        ENGINE["error"] = f"{type(e).__name__}: {e}"


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


def enrich(output):
    for c in output["clauses"]:
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
                                  "error", "mode")} | {
        "clauses": len(clauses),
        "regulations": len({g["id"] for c in clauses for g in c["regulations"]}),
        "needs_review": out.get("document_needs_review", False),
    }


# ─────────────────────────────────────────── API ───────────────────────────────────────────
app = FastAPI(title="MITHAQ")
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    return {"mode": MODE, "ready": ENGINE["ready"], "error": ENGINE["error"], "k": K}


@app.get("/api/reviews")
def list_reviews():
    with LOCK:
        return sorted((summary(r) for r in REVIEWS.values()), key=lambda s: -s["created"])


@app.get("/api/reviews/{review_id}")
def get_review(review_id: str):
    with LOCK:
        if review_id not in REVIEWS:
            raise HTTPException(404, "المراجعة غير موجودة")
        r = REVIEWS[review_id]
        return summary(r) | {"output": r.get("output")}


@app.get("/api/reviews/{review_id}/json")
def download(review_id: str):
    r = REVIEWS.get(review_id)
    if not r or not r.get("output"):
        raise HTTPException(404, "لا توجد نتائج")
    return JSONResponse(r["output"], headers={
        "Content-Disposition": f'attachment; filename="mithaq-{review_id}.json"'})


@app.delete("/api/reviews/{review_id}")
def delete(review_id: str):
    with LOCK:
        REVIEWS.pop(review_id, None)
    (DATA / f"{review_id}.json").unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/reviews")
async def upload(file: UploadFile = File(...)):
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
                              "status": "received", "done": 0, "total": 0, "mode": MODE}
    threading.Thread(target=run_review, args=(review_id, dest), daemon=True).start()
    return {"id": review_id}


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="full", choices=["full", "bm25", "stub"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    MODE = a.retriever
    load_saved()
    threading.Thread(target=load_engine, daemon=True).start()
    import uvicorn
    print(f"\n  MITHAQ  ➜  http://{a.host}:{a.port}   (retriever: {MODE})\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
