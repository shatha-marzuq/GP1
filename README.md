# MITHAQ

MITHAQ reviews the **Terms and Conditions (T&C) of Saudi private investment funds** against the regulations of the
Capital Market Authority (CMA). A compliance officer uploads a T&C file; the system extracts the text, splits it into
clauses, and finds the CMA articles that govern each clause, with a confidence decision and the reasoning behind it.

**Scope:** private (and simplified) funds only. Applicable sources are the Investment Funds Regulations (IFR) general
parts, Part 5 (private funds, arts 80–98) and Annex 11, the Capital Market Institutions Regulations (CMIR), and the
Simplified Investment Funds Instructions (SIFI) for simplified funds.

```
upload ─► Document Processing (OCR) ─► Segmentation ─► Regulatory Retrieval Agent ─► Compliance Verification ─► Report
           docproc/                     backend/        pipeline/retrieval_agent.py   (next phase)              (next phase)
                          └──────── orchestrated by LangGraph: backend/orchestrator/ ────────┘
```

## Repository layout

| Path | What it is |
|---|---|
| `docproc/mithaq_docproc/` | Document Processing Agent: PDF/Word → clean items with page numbers (PyMuPDF, Tesseract, Surya OCR). Own README. |
| `backend/segmentation.py` | Rule-based splitting of the items into clauses (each clause keeps a breadcrumb and its pages). |
| `backend/segmentation_agent.py`, `segmentation_tools.py` | LLM tool-using version of segmentation (checks and fixes the rule split). Not wired into the graph yet. |
| `backend/orchestrator/` | LangGraph pipeline: extract → segment → retrieve × N clauses (parallel) → collect. Own README. |
| `pipeline/` | Regulatory knowledge base (steps 3–12 below), `retriever.py` (search) and `retrieval_agent.py` (the agent). |
| `interface/` | FastAPI + single-page web app (Arabic, RTL): upload, live progress, clauses with their regulations. |
| `output/` | Knowledge-base artefacts (`01`–`06`), Chroma vector store (`chroma_db/`), evaluation reports (`07`–`09`). |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r docproc/mithaq_docproc/requirements.txt -r pipeline/requirements.txt \
            -r backend/orchestrator/requirements.txt -r interface/requirements.txt

# optional: the LLM step of the retrieval agent (the file is git-ignored)
echo "OPENAI_API_KEY=sk-..."   >> .env
echo "RETRIEVAL_MODEL=gpt-4o"  >> .env

# web interface → http://127.0.0.1:8000
cd interface && python app.py                 # retrieval agent (+ LLM step if a key is set)
python app.py --no-llm                        # agent, rule-based steps only
python app.py --retriever full|bm25|stub      # plain search | lexical only | stand-in (no models)

# command line: one document → backend/retrieval.json
cd backend && python -m orchestrator.run path/to/terms.pdf --agent --llm
```

The first run downloads BGE-M3 and the reranker (~4.5 GB); afterwards `HF_HUB_OFFLINE=1` works. The Chroma store is
in the repo, so the knowledge base does not need rebuilding unless the regulations change
(`cd pipeline && python run_all.py --skip-extract`).

## Regulatory knowledge base (`pipeline/`, steps 3–12)

| Source | Pages | Articles | Annexes | Chunks |
|---|---|---|---|---|
| Investment Funds Regulations (amended 2025-11-24) | 181 | 113 | 14 | 238 |
| Capital Market Institutions Regulations (amended 2026-01-07) | 111 | 99 | 11 | 142 |
| Corporate Governance Regulations (amended 2023-01-18) | 65 | 95 | 1 | 109 |
| Simplified Investment Funds Instructions (2026-03-02) | 17 | 31 | 3 | 35 |

1. **Extract** (`extract.py`): glyph-level pdfminer extraction that fixes the reversed Arabic ligatures common
   in CMA PDFs (`املالية` → `المالية`), restores numbers/Latin direction and footnote markers.
2. **Clean** (`clean.py`): Unicode normalisation, diacritics, kashida artefacts; removes covers, tables of contents,
   page numbers (log in `output/02_clean/noise_report.json`). Footnotes are kept: in the governance regulations they
   mark provisions as guidance or binding.
3. **Structure** (`structure.py`): document → part → chapter → article/annex → paragraphs; validated against each
   document's table of contents (338/338 articles, 29/29 annexes).
4. **Chunk** (`chunk.py`): never crosses an article; ≤ 300 words, split at paragraph boundaries; every chunk starts
   with its breadcrumb. Annexes 1 and 11 (T&C content checklists) are split per numbered item.
5. **Embed and store** (`embed_and_store.py`): BGE-M3 dense vectors (1024-d) in Chroma (cosine), with metadata for
   filtering — notably `fund_scope` (`private`, `public`, `funds_all`, `general`, …) used by `fund_type`.

## Retrieval

**`retriever.py`**: dense search (BGE-M3), optional BM25/hybrid, reranking with `bge-reranker-v2-m3`, and a
`fund_type="private"` filter that drops look-alike public-fund articles. Defaults match MITHAQ's scope
(`fund_type="private"`, `rerank=True`, Annex 11 pinned to rank 2).

**`retrieval_agent.py`, the Regulatory Retrieval Agent.** It finds the provisions that *apply* to each clause, not
just the textually similar ones, by coordinating the rule-based search with an LLM (two calls per clause):

1. **understand** (LLM): the clause's regulatory topics, one search query per topic in the regulations' own
   terminology, up to 3 articles named from the catalogue of private-fund IFR provisions, and the Annex 11 item the
   clause's section corresponds to. Explicit references ("وفقًا للمادة 76") are found by rules.
2. **retrieve** (rules): the clause text (dense + rerank), BM25, hybrid search per topic query and title match, each
   20 deep, fused (RRF) into a pool of 18 candidates, after the cited, named and Annex 11 provisions. Everything is
   filtered to the private-fund scope: CMIR, the IFR parts common to all funds, IFR Part 5 and its annexes. Public-fund
   (IFR Part 4) and corporate-governance provisions never enter the pool unless the clause cites them.
3. **select** (LLM): reads each candidate's paragraphs most related to the clause (BM25 + embedding similarity, not
   the chunk's opening) and selects the provisions to read, each with a role (governing / disclosure / related) and a
   one-line reason; it leaves out rules written only for public or real-estate funds and mismatched citations.

Retrieval makes **no compliance or confidence decision**: that belongs to the Compliance Analysis Agent. Every
candidate is kept: selected ones first (`selected: true`, role, reason, full text), then the others
(`selected: false`, citation, heading, related paragraphs; full text on request). Without the LLM (no key, API
error, 60 s timeout) `selected` is `null` and the rule ranking is used; the clause records `llm_error`.
LLM answers are cached in `output/llm_cache/`, keyed on cache version, backend, endpoint, model, every call parameter
and the full messages.

**Depth and pool size** were chosen on the three labelled sets, locally (rule lists only, no LLM). Share of the
expected articles that reach the candidate pool:

| Search depth (pool 14) | BlomInvest | KB queries | Mulkia |
|---|---|---|---|
| 10 (previous) | 0.889 | 0.969 | 0.824 |
| **20** | **0.914** | **0.980** | **0.869** |
| 30 | 0.864 | 0.958 | 0.784 |

At depth 30 weaker results crowd the right articles out of the fused pool. At depth 20, pools of 14 / 16 / 18 / 20
reach 0.914 / 0.914 / 0.934 / 0.949 (BlomInvest) and 0.869 / 0.892 / 0.905 / 0.914 (Mulkia): 18 keeps most of the gain
for ~10% fewer tokens and less noise than 20.
Letting the cross-encoder re-score the top 40 first-stage results instead of 20 was tried and rejected: pool
recall +0.036 on BlomInvest but −0.004 on the KB queries and −0.022 on Mulkia (held out).

## Evaluation

`pipeline/test_agent.py` measures each stage separately (section level; a clause is a hit if any expected article
appears at that rank): **pool recall** (retrieval, before the LLM, against plain search at the same depth),
**selection** recall and precision, and the **final ranking** (Hit@1/3/5, MRR, Cov@5 = share of the expected
articles in the top 5). A run is valid only if every clause went through both LLM steps; otherwise it is reported
as INVALID with the failing clauses, and exits with code 2. Labels that can never be retrieved are reported, with
every metric also given without them; labels are never changed by the script.

Previous agent (tool-calling LLM step, before this redesign), GPT-4o, all clauses valid:

| Set (clauses) | Plain search Hit@1 / MRR / Cov@5 | Previous agent Hit@1 / MRR / Cov@5 |
|---|---|---|
| Knowledge-base queries (36) | 0.889 / 0.933 / 0.889 | 0.972 / 0.986 / 0.924 |
| Real private-fund T&C, BlomInvest (33) | 0.727 / 0.833 / 0.783 | 0.909 / 0.939 / 0.864 |
| Second private-fund T&C, Mulkia (37), held out | 0.514 / 0.608 / 0.541 | 0.784 / 0.860 / 0.802 |

The redesigned agent has not been evaluated with the LLM yet: `cd pipeline && python test_agent.py --llm --sets blom
kb mulkia`. Offline checks: `cd pipeline && python -m pytest tests -q` (stand-in LLM, no API).

- Labels were made by one person; the BlomInvest set was also used to tune the agent's prompts, Mulkia was not.
- Known label issues, for review (not changed): IFR-art075 (public funds, unreachable) in B21 and C045; IFR-art086
  (real-estate funds) in C046; different conventions between the sets for Annex 11 (e.g. fees, conflict of interest)
  and for valuation errors (B21 vs C045). Details: `docproc/testset/eval/label_audit.md`.
- Reports: `output/07_retrieval_eval` (search), `output/09_agent_eval` (agent). Test T&Cs, their labels and their
  reports stay out of the repo (`docproc/testset/`, git-ignored).

## Known limitations and next steps

- **Compliance Verification Agent** (next phase): judge each clause as Compliant / Violation / Needs Review with
  quotations. It also needs a checklist of mandatory requirements (Annex 11, arts 80–98) to detect *missing* clauses,
  which clause-by-clause retrieval cannot find.
- Some PDF fonts store `ق` as `ة` (`حةوق`); docproc now repairs it inside words, but not when two words are glued
  (`…ةال…`), and similar lam ligature errors (`المخحق`) remain.
- Segmentation sometimes merges the front matter or trailing sections (Shariah controls, complaints) into one clause.
- LLM results vary by a clause or two between runs; `segmentation_agent.py` still retries on HTTP 429 even when the
  account has no credit.
