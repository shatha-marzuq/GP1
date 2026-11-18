"""Build the report tables in output/10_report_tests/ from the raw runs (graduation report, Release 1).

Inputs
  accuracy : test_agent.py reports with --llm — the full agent (--tag full) and one per ablation (--ablate NAME),
             found in output/09_agent_eval/ or docproc/testset/eval/ (the newest of each) and copied to raw/
  timing   : raw/timing_*.json written by backend/orchestrator/time_files.py (untagged runs only)
Outputs
  accuracy_ablation.csv/.md, timing_per_file.csv/.md, timing_summary.md, environment.md, REPORT_TABLES.md

    python3 report_tables.py
"""
import csv, json, re, shutil, statistics
from pathlib import Path

from retrieval_agent import ABLATIONS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "10_report_tests"
RAW = OUT / "raw"
REPORT_DIRS = [ROOT / "output" / "09_agent_eval", ROOT / "docproc" / "testset" / "eval"]
RUNS = ["full"] + list(ABLATIONS)
LABEL = {"full": "Full agent", "plain": "Plain search (no agent)",
         "no_rerank_clause": "− clause rerank", "no_lexical": "− BM25 on clause", "no_topics": "− topic searches",
         "no_title": "− title match", "no_named": "− LLM-named articles", "no_annex_item": "− Annex 11 item"}
STAGES = [("extract_s", "Extraction (docproc / OCR)"), ("segment_s", "Segmentation (agent)"),
          ("retrieve_s", "Retrieval (all clauses, parallel)"), ("collect_s", "Collection"), ("total_s", "Total")]


def n2(x):
    return "–" if x is None else f"{x:.2f}"


def secs(x):
    if x is None:
        return "–"
    return f"{x:.2f} s" + (f" ({x / 60:.2f} min)" if x > 60 else "")


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# ------------------------------------------------------------------ accuracy
def find_report(run):
    pat = re.compile(r"report_dense_rerank_llm(_[a-z_]+?)?_" + re.escape(run) + r"\.json$")
    found = [p for d in REPORT_DIRS if d.exists() for p in d.glob("report_*.json") if pat.search(p.name)]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def accuracy_rows():
    rows = []
    for run in RUNS:
        src = find_report(run)
        if src is None:
            continue
        shutil.copy2(src, RAW / f"accuracy_{run}.json")
        rep = json.loads(src.read_text(encoding="utf-8"))
        for name, s in rep["sets"].items():
            m = s["all"]
            if run == "full":   # plain retriever.search at the same depth, from the same run
                b = m["baseline"]
                rows.append({"run": "plain", "set": name, "clauses": m["n"], "pool_recall": b["pool"],
                             "selection_recall": None, "selection_precision": None, "hit1": b["hit1"],
                             "hit3": b["hit3"], "mrr": b["mrr"], "seconds_per_clause": None,
                             "llm_answers": None, "llm_api_calls": None, "valid": None, "source": src.name})
            ag, sl = m["agent"], m["selection"]
            rows.append({"run": run, "set": name, "clauses": m["n"], "pool_recall": ag["pool"],
                         "selection_recall": sl["recall"], "selection_precision": sl["precision"],
                         "hit1": ag["hit1"], "hit3": ag["hit3"], "mrr": ag["mrr"],
                         "seconds_per_clause": s["seconds_per_clause"], "llm_answers": s.get("llm_answers"),
                         "llm_api_calls": s.get("llm_api_calls"), "valid": rep.get("valid"), "source": src.name})
    return rows


def accuracy_md(rows, with_delta=True):
    full = {r["set"]: r for r in rows if r["run"] == "full"}
    out = []
    for st in dict.fromkeys(r["set"] for r in rows):
        sub = [r for r in rows if r["set"] == st]
        out += [f"**{st}** ({sub[0]['clauses']} clauses/queries; one item ≈ {100 / sub[0]['clauses']:.1f} pp)", ""]
        body = []
        for r in sub:
            def cell(k):
                v = r[k]
                if v is None:
                    return "–"
                f = full.get(st, {}).get(k)
                if with_delta and r["run"] not in ("full", "plain") and f is not None:
                    return f"{v:.2f} ({v - f:+.2f})"
                return f"{v:.2f}"
            body.append([LABEL.get(r["run"], r["run"]), cell("pool_recall"), cell("selection_recall"),
                         cell("selection_precision"), cell("hit1"), cell("hit3"), cell("mrr")])
        out += [table(["Configuration", "Pool recall", "Selection recall", "Selection precision", "Hit@1", "Hit@3",
                       "MRR"], body), ""]
    return "\n".join(out)


# ------------------------------------------------------------------ timing
def timing_records():
    recs = []
    for p in sorted(RAW.glob("timing_*.json")):
        if p.name.startswith("timing_environment"):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "_check" in p.stem or rec.get("setup", "").endswith("zero-cost check"):
            continue
        recs.append(rec)
    return recs


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": statistics.mean(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def timing_rows(recs):
    return [{"file": r["file"], "pages": r["pages"], "type": r["kind"], "ocr_pages": r["ocr_pages"],
             "clauses": r["clauses"], "total_s": r["total_s"], "extract_s": r["extract_s"],
             "segment_s": r["segment_s"], "retrieve_s": r["retrieve_s"], "collect_s": r["collect_s"],
             "retrieve_per_clause_s": r["retrieve_per_clause_s"], "total_per_clause_s": r["total_per_clause_s"],
             "llm_api_answers": r["llm_api_answers"], "clauses_llm_fallback": r["clauses_llm_fallback"],
             "segmentation_status": r["segmentation_status"], "failed": r["failed"]} for r in recs]


def timing_file_md(rows):
    return table(["File", "Pages", "Type", "OCR pages", "Clauses", "Total", "Extraction", "Segmentation", "Retrieval",
                  "Collection", "Total per clause", "Failures"],
                 [[r["file"], r["pages"], r["type"], r["ocr_pages"], r["clauses"], secs(r["total_s"]),
                   secs(r["extract_s"]), secs(r["segment_s"]), secs(r["retrieve_s"]), secs(r["collect_s"]),
                   secs(r["total_per_clause_s"]),
                   "failed" if r["failed"] else (f"{r['clauses_llm_fallback']} clause(s) without LLM"
                                                 if r["clauses_llm_fallback"] else "none")] for r in rows])


def timing_summary_md(rows):
    keys = STAGES + [("retrieve_per_clause_s", "Retrieval per clause (wall time / clauses)"),
                     ("total_per_clause_s", "Total per clause")]
    body, mean_total = [], (stats([r["total_s"] for r in rows]) or {}).get("mean")
    for k, name in keys:
        s = stats([r[k] for r in rows])
        if s:
            share = f"{100 * s['mean'] / mean_total:.1f} %" if mean_total and k in dict(STAGES) else "–"
            body.append([name, secs(s["mean"]), secs(s["median"]), secs(s["min"]), secs(s["max"]), share])
    return table(["Stage", "Mean", "Median", "Min", "Max", "Share of mean total"], body)


# ------------------------------------------------------------------ environment / report
def env_md(env, acc_cfg):
    lines = ["# Test environment", ""]
    if env:
        lines += [table(["Item", "Value"], [[k.replace("_", " "), v] for k, v in env.items()]), ""]
    if acc_cfg:
        lines += ["## Retrieval accuracy runs (pipeline/test_agent.py)", "",
                  table(["Item", "Value"], [[k.replace("_", " "), v] for k, v in acc_cfg.items()]), ""]
    return "\n".join(lines)


def build():
    RAW.mkdir(parents=True, exist_ok=True)
    acc = accuracy_rows()
    write_csv(OUT / "accuracy_ablation.csv", acc)
    if acc:
        (OUT / "accuracy_ablation.md").write_text(
            "# Retrieval accuracy — full agent vs ablations (all labels, values in 0–1, Δ vs full agent)\n\n"
            + accuracy_md(acc) + "\n\n" + table(
                ["Run", "Set", "s / clause", "LLM answers", "LLM API calls (not cached)", "Valid"],
                [[LABEL.get(r["run"], r["run"]), r["set"], n2(r["seconds_per_clause"]), r["llm_answers"],
                  r["llm_api_calls"], r["valid"]] for r in acc if r["run"] != "plain"]) + "\n", encoding="utf-8")
    recs = timing_records()
    trows = timing_rows(recs)
    write_csv(OUT / "timing_per_file.csv", trows)
    if trows:
        (OUT / "timing_per_file.md").write_text("# Processing time per file\n\n" + timing_file_md(trows) + "\n",
                                                encoding="utf-8")
        (OUT / "timing_summary.md").write_text(
            f"# Processing time — summary over {len(trows)} file(s)\n\n" + timing_summary_md(trows) + "\n",
            encoding="utf-8")
    envp = RAW / "timing_environment.json"
    env = json.loads(envp.read_text(encoding="utf-8")) if envp.exists() else None
    acc_cfg = None
    if (RAW / "accuracy_full.json").exists():
        acc_cfg = json.loads((RAW / "accuracy_full.json").read_text(encoding="utf-8"))["config"]
    (OUT / "environment.md").write_text(env_md(env, acc_cfg), encoding="utf-8")
    (OUT / "REPORT_TABLES.md").write_text(report_tables(acc, trows, env, acc_cfg), encoding="utf-8")
    print("report tables ->", OUT)


def report_tables(acc, trows, env, acc_cfg):
    hw = (f"{env['machine']} ({env['cpu']}, {env['memory_gb']} GB RAM, {env['accelerator']})" if env else "–")
    L = ["# Report tables — MITHAQ (Release 1)", ""]
    if acc:
        sets = list(dict.fromkeys(r["set"] for r in acc))
        sizes = {r["set"]: r["clauses"] for r in acc if r["run"] == "full"}
        abl = any(r["run"] not in ("full", "plain") for r in acc)
        L += [f"## Table: Retrieval accuracy ({'full agent vs ablations' if abl else 'plain search vs full agent'})", "",
              accuracy_md(acc), "",
              "The table compares the full Regulatory Retrieval Agent with plain vector search (dense + rerank, top "
              "results at the same depth)" + (" and with versions of the agent in which one component is switched off "
              "(ablation); values in brackets are the change against the full agent." if abl else ".") + " Pool recall is the share of the expected provisions among the candidates the agent retrieves; "
              "selection recall and precision measure the LLM's final choice among those candidates; Hit@k and MRR rank "
              "the first expected provision.", "",
              f"*Method:* labelled sets {', '.join(f'{s} ({sizes.get(s)})' for s in sets)}; retrieval LLM "
              f"{acc_cfg.get('llm') if acc_cfg else '–'}, dense search with cross-encoder rerank, private-fund scope, "
              f"search depth {acc_cfg.get('search_depth') if acc_cfg else '–'}, pool size "
              f"{acc_cfg.get('pool_size') if acc_cfg else '–'}. Each configuration was run once at temperature 0. "
              "With about 33–37 items per set, one item is ≈ 3 percentage points, so differences below 0.03 are "
              "within noise.", ""]
    if trows:
        n = len(trows)
        types = ", ".join(f"{r['file']} ({r['pages']} pages, {r['type']})" for r in trows)
        L += ["## Table: Processing time per file", "", timing_file_md(trows), "",
              "End-to-end time of one review, from the start of the pipeline (file uploaded) to the final results: "
              "text extraction, clause segmentation by the Segmentation Agent, regulation retrieval for every clause by "
              "the Regulatory Retrieval Agent (8 clauses in parallel), and collection of the results.", "",
              f"*Method:* {n} T&C file(s) — {types}; each file processed once; LLM answer cache off; embedding and "
              f"reranker models loaded before timing (load time {secs(env['model_load_seconds']) if env else '–'}, "
              f"excluded); hardware: {hw}; retrieval LLM {env['retrieval_llm'] if env else '–'}, segmentation LLM "
              f"{env['segmentation_llm'] if env else '–'}.", "",
              "## Table: Average processing time per stage", "", timing_summary_md(trows), ""]
        s = {k: stats([r[k] for r in trows]) for k, _ in STAGES}
        slow = max((k for k, _ in STAGES if k != "total_s" and s[k]), key=lambda k: s[k]["mean"], default=None)
        if slow:
            L += [f"The {dict(STAGES)[slow].split(' (')[0].lower()} stage takes the largest share of the processing time "
                  f"({100 * s[slow]['mean'] / s['total_s']['mean']:.1f} % of the mean total of "
                  f"{secs(s['total_s']['mean'])}). Per-clause times are the stage's wall time divided by the number of "
                  "clauses (clauses are processed in parallel).", "",
                  f"*Method:* mean, median, minimum and maximum over the {n} file(s) of the previous table; same setup.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    build()
