"""
Compare the Regulatory Retrieval Agent (retrieval_agent.py) with plain retriever.search on the labelled sets:

  blom : 33 clauses of a real private-fund T&C (output/08_tc_evaluation/blom_clauses.json), with their headings
         as clause titles, fund_type="private"
  kb   : the 36 step-12 queries (eval/retrieval_queries.json), no titles, no fund filter
  mulkia: 37 hand-labelled clauses of a second private-fund T&C, never used while tuning the agent
          (docproc/testset/mulkia_clauses.json — git-ignored like the T&C itself; its report is written there too)

Section-level metrics (a split article counts once; an expected doc prefix such as "CMIR" matches any of its sections):
Hit@1/3/5, MRR@5, plus how often each agent decision was taken and how accurate it was.

Output: output/09_agent_eval/report.json + report.md
Usage : python3 test_agent.py [--mode dense|hybrid|bm25] [--no-rerank] [--llm [--model NAME]] [--sets blom kb mulkia]
        (bm25 needs no models — use it to check the logic; dense + rerank is the real configuration)
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

from retriever import RegulationRetriever
from retrieval_agent import LLM, RetrievalAgent

ROOT = Path(__file__).resolve().parent.parent
SETS = {"blom": (ROOT / "output" / "08_tc_evaluation" / "blom_clauses.json", "private", True),
        "kb": (Path(__file__).resolve().parent / "eval" / "retrieval_queries.json", None, False),
        "mulkia": (ROOT / "docproc" / "testset" / "mulkia_clauses.json", "private", True)}
OUT = ROOT / "output" / "09_agent_eval"
LOCAL_OUT = ROOT / "docproc" / "testset" / "eval"   # reports of sets built from T&Cs that stay out of the repo


def match(key, expected):
    return any(key == e or ("-" not in e and key.split("-")[0] == e) for e in expected)


def rank_of(hits, expected):
    sections = list(dict.fromkeys(h["section_key"] for h in hits))
    return next((n for n, s in enumerate(sections, 1) if match(s, expected)), None)


def metrics(ranks):
    n = len(ranks)
    return {"hit1": round(sum(r == 1 for r in ranks) / n, 3), "hit3": round(sum(bool(r) and r <= 3 for r in ranks) / n, 3),
            "hit5": round(sum(bool(r) and r <= 5 for r in ranks) / n, 3),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dense", choices=["dense", "hybrid", "bm25"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--llm", action="store_true", help="add the LLM step (needs OPENAI_API_KEY)")
    ap.add_argument("--model", help="LLM model (default: RETRIEVAL_MODEL or gpt-4o)")
    ap.add_argument("--sets", nargs="+", default=["blom", "kb"], choices=list(SETS))
    a = ap.parse_args()
    rerank = not a.no_rerank and a.mode != "bm25"
    r = RegulationRetriever()
    agent = RetrievalAgent(r, llm=LLM(a.model) if a.llm else None)
    tag = f"{a.mode}{'_rerank' if rerank else ''}{'_llm' if a.llm else ''}"
    report = {"config": {"mode": a.mode, "rerank": rerank, "llm": agent.llm.model if a.llm else None}, "sets": {}}
    for name in a.sets:
        path, fund_type, use_title = SETS[name]
        rows, base_ranks, agent_ranks = [], [], []
        by_decision = defaultdict(list)
        for q in json.loads(path.read_text(encoding="utf-8")):
            kw = dict(k=5, mode=a.mode, fund_type=fund_type, rerank=rerank)
            base = r.search(q["query"], **kw)
            out = agent.run(q["query"], title=q.get("section") if use_title else None, **kw)
            rb, ra = rank_of(base, q["expected"]), rank_of(out["hits"], q["expected"])
            base_ranks.append(rb); agent_ranks.append(ra); by_decision[out["decision"]].append(ra)
            rows.append({"id": q["id"], "expected": q["expected"], "baseline_rank": rb, "agent_rank": ra,
                         "baseline_top3": [h["section_key"] for h in base[:3]],
                         "decision": out["decision"], "agent_top3": [h["section_key"] for h in out["hits"][:3]],
                         "steps": out["steps"]})
        report["sets"][name] = {
            "n": len(rows), "baseline": metrics(base_ranks), "agent": metrics(agent_ranks),
            "decisions": {d: {"count": len(v), "hit1": round(sum(x == 1 for x in v) / len(v), 3)}
                          for d, v in sorted(by_decision.items())},
            "changed": [{"id": x["id"], "baseline_rank": x["baseline_rank"], "agent_rank": x["agent_rank"],
                         "decision": x["decision"]} for x in rows if x["baseline_rank"] != x["agent_rank"]],
            "rows": rows}
    out = LOCAL_OUT if "mulkia" in a.sets else OUT
    tag += "" if a.sets == ["blom", "kb"] else "_" + "_".join(a.sets)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"report_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# Retrieval agent vs plain search — mode={a.mode}, rerank={rerank}, llm={report['config']['llm']}", ""]
    for name, s in report["sets"].items():
        lines += [f"## {name} ({s['n']} queries)", "", "| | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
        for who in ("baseline", "agent"):
            m = s[who]
            lines.append(f"| {who} | {m['hit1']} | {m['hit3']} | {m['hit5']} | {m['mrr']} |")
        lines += ["", "Decisions: " + ", ".join(f"{d} {v['count']} (Hit@1 {v['hit1']})" for d, v in s["decisions"].items()),
                  "", "Changed ranks: " + (", ".join(f"{c['id']} {c['baseline_rank']}→{c['agent_rank']}"
                                                      for c in s["changed"]) or "none"), ""]
    (out / f"report_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
