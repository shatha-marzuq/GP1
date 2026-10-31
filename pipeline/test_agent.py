"""
Evaluate the Regulatory Retrieval Agent (retrieval_agent.py) against plain retriever.search on the labelled sets:

  blom : 33 clauses of a real private-fund T&C (output/08_tc_evaluation/blom_clauses.json), with their headings
         as clause titles, fund_type="private"
  kb   : the 36 step-12 queries (eval/retrieval_queries.json), no titles, no fund filter
  mulkia: 37 hand-labelled clauses of a second private-fund T&C, never used while tuning the agent
          (docproc/testset/mulkia_clauses.json — git-ignored like the T&C itself; its report is written there too)

Each stage is measured on its own (section level; an expected doc prefix such as "CMIR" matches any of its sections):
  pool      : share of the expected articles among the agent's candidates (retrieval, before the LLM selects),
              against plain search's top POOL_SIZE at the same depth
  selection : recall = expected articles selected / expected articles in the pool; precision = selected articles
              that are expected (a lower bound: the labels list the main provisions, not every relevant one)
  ranking   : Hit@1/3/5 and MRR (first expected article), Cov@5 (share of the expected articles in the top 5)
Labels that can never be retrieved (outside the set's fund-type scope, or not in the knowledge base) are reported,
and every metric is also given without them. They are never changed here.

A run is VALID only if every clause went through both LLM steps; otherwise the clauses that fell back to the rules
are listed and the exit code is 2 (re-run: answers already received come from the cache). Cache hits are counted.

Output: output/09_agent_eval/report_<tag>.json + .md (docproc/testset/eval/ for sets that include mulkia)
Usage : python3 test_agent.py [--mode dense|hybrid|bm25] [--no-rerank] [--llm [--model NAME] [--no-cache]]
                              [--sets blom kb mulkia] [--tag NAME]
        (bm25 without --llm needs no models and no API — use it to check the logic)
"""
import argparse, json, sys, time
from pathlib import Path

from retriever import RegulationRetriever
from retrieval_agent import LLM, POOL_SIZE, SEARCH_DEPTH, RetrievalAgent

ROOT = Path(__file__).resolve().parent.parent
SETS = {"blom": (ROOT / "output" / "08_tc_evaluation" / "blom_clauses.json", "private", True),
        "kb": (Path(__file__).resolve().parent / "eval" / "retrieval_queries.json", None, False),
        "mulkia": (ROOT / "docproc" / "testset" / "mulkia_clauses.json", "private", True)}
OUT = ROOT / "output" / "09_agent_eval"
LOCAL_OUT = ROOT / "docproc" / "testset" / "eval"   # reports of sets built from T&Cs that stay out of the repo


def match(key, expected):
    return any(key == e or ("-" not in e and key.split("-")[0] == e) for e in expected)


def sections(hits):
    return list(dict.fromkeys(h["section_key"] for h in hits))


def rank_of(secs, expected):
    return next((n for n, s in enumerate(secs, 1) if match(s, expected)), None)


def share(secs, expected):
    """Share of the expected articles found among secs."""
    return sum(any(match(s, [e]) for s in secs) for e in expected) / len(expected) if expected else None


def reachable(agent, key, fund_type):
    """Can retrieval ever return this label? (a doc prefix such as "CMIR" counts if any of its sections can)"""
    keys = [key] if "-" in key else [k for k in agent.section_meta if k.split("-")[0] == key]
    return any(k in agent.section_meta and agent.r._allowed(agent.section_meta[k], None, None, None, fund_type)
               for k in keys)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarise(rows, key):
    """Metrics over rows, with expected = row[key] (all labels, or only the reachable ones)."""
    rows = [r for r in rows if r[key]]
    out = {"n": len(rows)}
    for who in ("baseline", "agent"):
        ranks = [rank_of(r[f"{who}_secs"], r[key]) for r in rows]
        out[who] = {"hit1": mean([rk == 1 for rk in ranks]), "hit3": mean([bool(rk) and rk <= 3 for rk in ranks]),
                    "hit5": mean([bool(rk) and rk <= 5 for rk in ranks]), "mrr": mean([1 / rk if rk else 0 for rk in ranks]),
                    "cov5": mean([share(r[f"{who}_secs"][:5], r[key]) for r in rows]),
                    "pool": mean([share(r[f"{who}_pool"], r[key]) for r in rows])}
    judged = [r for r in rows if r["selected"] is not None]
    out["selection"] = {
        "recall": mean([share(r["selected"], [e for e in r[key] if any(match(s, [e]) for s in r["agent_pool"])])
                        for r in judged]),
        "precision": mean([sum(match(s, r[key]) for s in r["selected"]) / len(r["selected"]) if r["selected"] else None
                           for r in judged]),
        "avg_selected": mean([len(r["selected"]) for r in judged])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dense", choices=["dense", "hybrid", "bm25"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--llm", action="store_true", help="the agent's LLM steps (needs OPENAI_API_KEY)")
    ap.add_argument("--model", help="LLM model (default: RETRIEVAL_MODEL or gpt-4o)")
    ap.add_argument("--no-cache", action="store_true", help="ask the LLM again even when an answer is cached")
    ap.add_argument("--sets", nargs="+", default=["blom", "kb"], choices=list(SETS))
    ap.add_argument("--tag", default="", help="suffix for the report file names (to keep runs apart)")
    a = ap.parse_args()
    rerank = not a.no_rerank and a.mode != "bm25"
    r = RegulationRetriever()
    llm = LLM(a.model) if a.llm else None
    if llm and a.no_cache:
        llm.cache = False
    agent = RetrievalAgent(r, llm=llm)
    report = {"config": {"mode": a.mode, "rerank": rerank, "llm": llm.model if llm else None,
                         "search_depth": SEARCH_DEPTH, "pool_size": POOL_SIZE}, "sets": {}}
    invalid = []
    for name in a.sets:
        path, fund_type, use_title = SETS[name]
        rows, t0 = [], time.time()
        for q in json.loads(path.read_text(encoding="utf-8")):
            kw = dict(mode=a.mode, fund_type=fund_type, rerank=rerank)
            base = r.search(q["query"], k=SEARCH_DEPTH, **kw)
            out = agent.run(q["query"], title=q.get("section") if use_title else None, k=5, **kw)
            steps = {s["action"]: s for s in out["steps"]}
            hits = out["hits"]
            sel = [h["section_key"] for h in hits if h["selected"]] if any(h["selected"] is not None for h in hits) else None
            rows.append({
                "id": q["id"], "expected": q["expected"],
                "reachable": [e for e in q["expected"] if reachable(agent, e, fund_type)],
                "baseline_secs": sections(base), "baseline_pool": sections(base)[:POOL_SIZE],
                "agent_secs": sections(hits), "agent_pool": sections(hits), "selected": sel,
                "llm_ok": "understand" in steps and "select" in steps and "llm_error" not in steps,
                "cache_hits": sum(bool(s.get("cached")) for s in out["steps"]),
                "steps": out["steps"]})
        s = {"n": len(rows), "seconds_per_clause": round((time.time() - t0) / len(rows), 1),
             "all": summarise(rows, "expected"), "reachable_only": summarise(rows, "reachable"),
             "unreachable_labels": [{"id": x["id"], "labels": sorted(set(x["expected"]) - set(x["reachable"]))}
                                    for x in rows if len(x["reachable"]) < len(x["expected"])],
             "rows": rows}
        if llm:
            s["llm_failed"] = [x["id"] for x in rows if not x["llm_ok"]]
            s["cache_hits"] = sum(x["cache_hits"] for x in rows)
            invalid += [f"{name}:{i}" for i in s["llm_failed"]]
        report["sets"][name] = s
    report["valid"] = not invalid

    out = LOCAL_OUT if "mulkia" in a.sets else OUT
    tag = f"{a.mode}{'_rerank' if rerank else ''}{'_llm' if llm else ''}"
    tag += ("" if a.sets == ["blom", "kb"] else "_" + "_".join(a.sets)) + (f"_{a.tag}" if a.tag else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Retrieval agent vs plain search — mode={a.mode}, rerank={rerank}, llm={report['config']['llm']}, "
             f"depth={SEARCH_DEPTH}, pool={POOL_SIZE}", ""]
    if llm and invalid:
        lines += [f"**INVALID RUN: {len(invalid)} clause(s) fell back to the rules (LLM error):** {', '.join(invalid)}", ""]
    for name, s in report["sets"].items():
        lines += [f"## {name} ({s['n']} clauses, {s['seconds_per_clause']} s per clause"
                  + (f", {s['cache_hits']} cached LLM answers" if llm else "") + ")", ""]
        for part in ("all", "reachable_only"):
            m = s[part]
            lines += [f"**{'All labels' if part == 'all' else 'Reachable labels only'}** ({m['n']} clauses)", "",
                      "| | Pool recall | Hit@1 | Hit@3 | Hit@5 | MRR | Cov@5 |", "|---|---|---|---|---|---|---|"]
            for who in ("baseline", "agent"):
                x = m[who]
                lines.append(f"| {who} | {x['pool']} | {x['hit1']} | {x['hit3']} | {x['hit5']} | {x['mrr']} | {x['cov5']} |")
            sl = m["selection"]
            lines += ["", f"Selection: recall {sl['recall']}, precision {sl['precision']}, {sl['avg_selected']} selected per clause", ""]
        if s["unreachable_labels"]:
            lines += ["Unreachable labels: " + "; ".join(f"{u['id']} {', '.join(u['labels'])}" for u in s["unreachable_labels"]), ""]
    (out / f"report_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    sys.exit(2 if llm and invalid else 0)


if __name__ == "__main__":
    main()
