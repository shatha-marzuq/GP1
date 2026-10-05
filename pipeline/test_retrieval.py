"""
Step 12 — Test the retrieval quality of the regulatory knowledge base.

Runs the labelled query set in eval/retrieval_queries.json (T&C-style clauses + questions, each with the
article(s) that should be retrieved) through the retrieval modes (+ cross-encoder reranking) and reports, at article level
(a split article counts as found if any of its chunks is returned):
  Hit@1 / Hit@3 / Hit@5 / Hit@10  — share of queries with at least one expected article in the top-k
  MRR@10                            — mean reciprocal rank of the first expected article
  Recall@10                         — share of the expected articles found in the top-10

Output: output/07_retrieval_eval/report.json + report.md
Usage : python3 test_retrieval.py [--min-hit5 0.8]   (exit code 1 if dense Hit@5 is below the threshold)
"""
import argparse, json, sys
from pathlib import Path

from retriever import RegulationRetriever

ROOT = Path(__file__).resolve().parent.parent
QUERIES = Path(__file__).resolve().parent / "eval" / "retrieval_queries.json"
OUT = ROOT / "output" / "07_retrieval_eval"
# name → RegulationRetriever.search kwargs (explicit, so the reported table doesn't depend on search() defaults)
_BASE = {"fund_type": None, "rerank": False}
MODES = {"dense": {**_BASE, "mode": "dense"}, "bm25": {**_BASE, "mode": "bm25"}, "hybrid": {**_BASE, "mode": "hybrid"},
         "dense+rerank": {**_BASE, "mode": "dense", "rerank": True}}
K = 10


def ranked_sections(hits):
    seen = []
    for h in hits:
        if h["section_key"] not in seen:
            seen.append(h["section_key"])
    return seen


def evaluate(r, queries):
    qvecs = r.embed_queries([q["query"] for q in queries])
    per_mode = {}
    for mode, kw in MODES.items():
        rows = []
        for q, v in zip(queries, qvecs):
            # fetch extra chunks so that K distinct articles remain after collapsing split articles
            secs = ranked_sections(r.search(q["query"], k=K * 2, qvec=v, **kw))[:K]
            exp = set(q["expected"])
            rank = next((n for n, s in enumerate(secs, 1) if s in exp), None)
            rows.append({"id": q["id"], "style": q["style"], "rank": rank, "top5": secs[:5],
                         "recall10": len(exp & set(secs)) / len(exp)})
        per_mode[mode] = rows
    return per_mode


def metrics(rows):
    n = len(rows)
    m = {f"hit@{k}": round(sum(r["rank"] is not None and r["rank"] <= k for r in rows) / n, 3) for k in (1, 3, 5, 10)}
    m["mrr@10"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)
    m["recall@10"] = round(sum(r["recall10"] for r in rows) / n, 3)
    m["n"] = n
    return m


def write_report(queries, per_mode):
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {mode: {"all": metrics(rows),
                      **{st: metrics([r for r in rows if r["style"] == st]) for st in ("clause", "question")}}
               for mode, rows in per_mode.items()}
    (OUT / "report.json").write_text(json.dumps({"summary": summary, "queries": queries, "results": per_mode},
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    cols = ["hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "recall@10"]
    md = ["# Retrieval evaluation — CMA regulatory knowledge base", "",
          f"{len(queries)} labelled queries (T&C clauses + questions), article-level matching, top-{K}.", "",
          "| mode | subset | n | " + " | ".join(cols) + " |", "|---|---|---|" + "---|" * len(cols)]
    for mode in MODES:
        for sub in ("all", "clause", "question"):
            m = summary[mode][sub]
            md.append(f"| {mode} | {sub} | {m['n']} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + " |")
    md += ["", "## Per-query results (dense — default mode)", "", "| id | rank | expected | top-3 retrieved | query |", "|---|---|---|---|---|"]
    for q, r in zip(queries, per_mode["dense"]):
        md.append(f"| {q['id']} | {r['rank'] or '✗'} | {', '.join(q['expected'])} | {', '.join(r['top5'][:3])} | {q['query']} |")
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-hit5", type=float, default=0.8)
    a = ap.parse_args()
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    r = RegulationRetriever()
    known = set(m["section_key"] for m in r.metas)
    unknown = {e for q in queries for e in q["expected"]} - known
    assert not unknown, f"expected sections not in the index: {unknown}"
    per_mode = evaluate(r, queries)
    summary = write_report(queries, per_mode)
    for mode in MODES:
        print(f"{mode:7s}", summary[mode]["all"])
    misses = [(q["id"], q["expected"], x["top5"][:3]) for q, x in zip(queries, per_mode["dense"])
              if not x["rank"] or x["rank"] > 5]
    for m in misses:
        print("  dense miss@5:", *m)
    print(f"report → {OUT.relative_to(ROOT)}/report.md")
    sys.exit(0 if summary["dense"]["all"]["hit@5"] >= a.min_hit5 else 1)
