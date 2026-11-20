"""Processing time per T&C file, from upload to results, for the project report.

Runs the same graph as the interface (build_graph) with the report's setup: Document Processing Agent (segmentation with LLM) +
Regulatory Retrieval Agent with its LLM steps, on every T&C file, once each, and records:
  total     : wall time of graph.invoke (the main number)
  per stage : from the on_status messages — extract (extracting → segmenting), segment (segmenting → first
              "retrieving"), retrieve (first → last clause done), collect (last clause → done)
  per file  : pages, OCR pages, clauses, time per clause, LLM answers, failures (failed review, segmentation
              agent error, clauses whose LLM step fell back to the rules)

Honest timing:
  - the retrieval agent's LLM cache is off (RETRIEVAL_CACHE=0, set here); the segmentation LLM has no cache
  - BGE-M3 + the reranker are loaded once before timing (like the interface at start-up); load time is recorded apart
  - a short warm-up (one clause, rules only, no API call) runs before the first file
  - max_concurrency=8, the value both run.py and interface/app.py use

Writes output/10_report_tests/raw/timing_<file>.json (timings and counts only, no T&C text: raw/ is git-ignored
anyway) and the whole review output to docproc/testset/eval/timing_outputs/ (git-ignored, like the T&Cs).
Then rebuilds the report tables (pipeline/report_tables.py).

    cd backend
    python -m orchestrator.time_files                       # every PDF/Word file in docproc/testset/
    python -m orchestrator.time_files a.pdf b.docx          # given files
    python -m orchestrator.time_files --no-llm --mode bm25  # zero-cost check of the script (no API call)
"""
import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

os.environ["RETRIEVAL_CACHE"] = "0"   # before the agent reads it: every LLM answer comes from the API

from . import paths  # noqa: F401,E402
from .graph import SegmentationAgent, build_graph, default_segment, docproc_extract  # noqa: E402

ROOT = paths.ROOT
TESTSET = ROOT / "docproc" / "testset"
OUT = ROOT / "output" / "10_report_tests"
RAW = OUT / "raw"
FULL_OUT = TESTSET / "eval" / "timing_outputs"
CONCURRENCY = 8          # run.py and interface/app.py both use 8
WARMUP_CLAUSE = "يجوز لمدير الصندوق تمديد فترة الطرح الأولي لمدة لا تزيد على عشرين يوم عمل بعد موافقة الهيئة."


def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def environment(retriever, llm, seg_model, load_s, warm_s, args):
    import torch
    from embed_and_store import MODEL as EMBED_MODEL, pick_device
    from retriever import RERANK_MODEL
    mem = _sh(["sysctl", "-n", "hw.memsize"]) if sys.platform == "darwin" else ""
    device = pick_device(getattr(retriever, "_device", None))
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "machine": _sh(["sysctl", "-n", "hw.model"]) if sys.platform == "darwin" else platform.node(),
        "os": f"{platform.system()} {platform.release()}" + (f" (macOS {platform.mac_ver()[0]})" if sys.platform == "darwin" else ""),
        "cpu": (_sh(["sysctl", "-n", "machdep.cpu.brand_string"]) if sys.platform == "darwin" else platform.processor()),
        "cpu_cores": os.cpu_count(),
        "memory_gb": round(int(mem) / 2**30, 1) if mem.isdigit() else None,
        "accelerator": ("CUDA " + torch.cuda.get_device_name(0)) if torch.cuda.is_available()
                       else "Apple MPS (GPU)" if torch.backends.mps.is_available() else "none (CPU)",
        "torch_device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "embedding_model": EMBED_MODEL,
        "reranker": RERANK_MODEL,
        "retrieval_llm": f"{llm.backend} / {llm.model}" if llm else None,
        "segmentation_llm": seg_model,
        "retrieval_mode": args.mode,
        "llm_cache": "off (RETRIEVAL_CACHE=0); the segmentation LLM has no cache",
        "max_concurrency": CONCURRENCY,
        "model_load_seconds": round(load_s, 2),
        "warmup_seconds": round(warm_s, 2),
    }


def time_file(path, graph_kwargs, retriever):
    events, captured = [], {}
    t0 = time.perf_counter()

    def on_status(review_id, status, **info):
        events.append({"t": round(time.perf_counter() - t0, 3), "status": status, **info})

    def extract(p):
        out = docproc_extract(p)
        captured.update(out)
        return out

    graph = build_graph(retriever, extract=extract, on_status=on_status, **graph_kwargs)
    t0 = time.perf_counter()
    error = None
    try:
        state = graph.invoke({"review_id": uuid.uuid4().hex[:8], "file_path": str(path), "results": []},
                             {"max_concurrency": CONCURRENCY})
    except Exception as e:
        state, error = {}, f"{type(e).__name__}: {e}"
    total = time.perf_counter() - t0

    def first(status, cond=lambda e: True):
        return next((e["t"] for e in events if e["status"] == status and cond(e)), None)

    t_ext, t_seg = first("extracting"), first("segmenting")
    t_ret0 = first("retrieving", lambda e: "total" in e)
    clause_done = [e["t"] for e in events if e["status"] == "retrieving" and "total" not in e]
    t_ret1 = max(clause_done) if clause_done else None
    t_done = first("done")
    span = lambda a, b: round(b - a, 3) if a is not None and b is not None else None

    doc, pages = captured.get("document", {}), captured.get("pages", [])
    n_pages = doc.get("page_count") or len(pages) or None
    ocr_pages = sum(p.get("method") == "ocr" for p in pages)
    failed_pages = sum(p.get("method") == "failed" for p in pages)
    kind = ("word" if (doc.get("source_type") or "").lower() in ("docx", "word") else
            "scanned" if n_pages and ocr_pages >= n_pages / 2 else "mixed" if ocr_pages else "text")
    output = state.get("output") or {}
    clauses = output.get("clauses", [])
    seg = output.get("segmentation") or {}
    llm_answers = llm_fallback = 0
    for c in clauses:
        steps = c.get("retrieval_steps") or []
        llm_answers += sum(s["action"] in ("understand", "select") and not s.get("cached") for s in steps)
        llm_fallback += any(s["action"] == "llm_error" for s in steps)
    n = len(clauses)
    retrieve = span(t_ret0, t_ret1)
    rec = {
        "file": Path(path).name,
        "pages": n_pages, "ocr_pages": ocr_pages, "failed_pages": failed_pages, "kind": kind,
        "source_type": doc.get("source_type"),
        "clauses": n,
        "total_s": round(total, 3),
        "extract_s": span(t_ext, t_seg),
        "segment_s": span(t_seg, t_ret0),
        "retrieve_s": retrieve,
        "collect_s": span(t_ret1, t_done),
        "retrieve_per_clause_s": round(retrieve / n, 3) if retrieve is not None and n else None,
        "total_per_clause_s": round(total / n, 3) if n else None,
        "segmentation_status": seg.get("status"),
        "segmentation_changes": len(seg.get("changes", [])) if seg else None,
        "llm_api_answers": llm_answers,
        "clauses_llm_fallback": llm_fallback,
        "failed": bool(error) or any(e["status"] == "failed" for e in events) or not output,
        "error": error or next((e.get("reason") for e in events if e["status"] == "failed"), None),
        "document_warnings": [w.get("code") for w in doc.get("warnings", [])],
        "events": events,
    }
    return rec, output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="T&C files (default: every PDF/Word file in docproc/testset/)")
    ap.add_argument("--no-llm", action="store_true", help="zero-cost check: rules only for both agents (no API call)")
    ap.add_argument("--mode", default="dense", choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--tag", default="", help="suffix of the raw file names (e.g. 'check' for a test run)")
    a = ap.parse_args()
    files = [Path(f) for f in a.files] or sorted(p for p in TESTSET.iterdir()
                                                 if p.suffix.lower() in (".pdf", ".docx") and not p.name.startswith("~$"))
    if not files:
        sys.exit(f"no T&C files given or found in {TESTSET}")

    from retriever import RegulationRetriever
    from retrieval_agent import LLM, RetrievalAgent
    import segmentation_agent

    t = time.perf_counter()
    r = RegulationRetriever()
    if a.mode != "bm25":
        r.search("رسوم الاشتراك في الصندوق", k=1)       # load BGE-M3 + the reranker, as the interface does at start-up
    load_s = time.perf_counter() - t
    agent = RetrievalAgent(r)
    t = time.perf_counter()
    kw = {"mode": "bm25", "rerank": False} if a.mode == "bm25" else {"mode": a.mode}
    agent.run(WARMUP_CLAUSE, k=5, **kw)                   # warm-up: one clause, rules only, no API call
    warm_s = time.perf_counter() - t
    llm = None
    if not a.no_llm:
        llm = LLM()
        assert not llm.cache, "the LLM cache must be off for timing"
        agent.llm = llm
    segment = SegmentationAgent(use_llm=not a.no_llm)
    env = environment(r, llm, None if a.no_llm else segmentation_agent.MODEL, load_s, warm_s, a)
    env["setup"] = ("Document Processing Agent (rules only) + Retrieval Agent (rules only) — zero-cost check" if a.no_llm
                    else "Document Processing Agent (segmentation with LLM) + Regulatory Retrieval Agent (LLM), as in run.py --seg-agent --agent --llm")
    print(json.dumps(env, ensure_ascii=False, indent=1), flush=True)

    RAW.mkdir(parents=True, exist_ok=True)
    FULL_OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{a.tag}" if a.tag else ""
    (RAW / f"timing_environment{suffix}.json").write_text(json.dumps(env, ensure_ascii=False, indent=1), encoding="utf-8")
    graph_kwargs = {"segment": segment, "k": 5, "search_kwargs": kw}
    for f in files:
        print(f"\n▶ {f.name}", flush=True)
        rec, output = time_file(f, graph_kwargs, agent)
        rec["setup"] = env["setup"]
        stem = re.sub(r"[^\w.-]+", "_", f.stem)
        (RAW / f"timing_{stem}{suffix}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        (FULL_OUT / f"{stem}{suffix}.json").write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
        print({k: v for k, v in rec.items() if k != "events"}, flush=True)

    if not a.tag:   # test runs (tagged) stay out of the report tables
        from report_tables import build
        build()


if __name__ == "__main__":
    main()
