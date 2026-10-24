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

**`retrieval_agent.py`, the Regulatory Retrieval Agent.** For each clause:

1. **analyse** the title and find explicit references ("وفقًا للمادة 76", "الملحق (11)");
2. **search** as above, then **check** the top result against independent signals (BM25, article-title match);
3. **retry** with other strategies (title + text, clause parts) if unsupported;
4. **LLM step** (optional): the model reads the candidates, may read full articles (`read_section`) or search again
   (`search_regulations`), and returns a ranking. A cited article stays first only if the LLM picks it; otherwise
   a `reference_mismatch` is logged (T&Cs often cite an older numbering of the IFR);
5. **decide**, with every step recorded in `retrieval_steps`:

| Decision | Meaning | Interface |
|---|---|---|
| `confirmed_by_llm` | rules and LLM independently agree on the top article | مؤكدة |
| `confirmed_by_reference` | the clause cites the article and the LLM confirms its subject | مؤكدة |
| `uncertain` | rules and LLM disagree, or no supporting signal | تحتاج مراجعة |
| `no_match` | the LLM finds no governing article | لا توجد مادة مطابقة |

Without a key, or if the API fails (no credit, timeout of 60 s), the agent falls back to its rule-based ranking and
logs `llm_error`; the interface shows a warning for those clauses.

## Evaluation

Section-level metrics: a clause is a hit if any expected article appears at that rank.

| Set (clauses) | Plain search Hit@1 / MRR | Agent + GPT-4o Hit@1 / MRR | `confirmed_by_llm` accuracy |
|---|---|---|---|
| Knowledge-base queries (36) | 0.889 / 0.933 | **0.944 / 0.972** | 28/28 |
| Real private-fund T&C, BlomInvest (33) | 0.727 / 0.833 | **0.909 / 0.939** | 21/21 |
| Second private-fund T&C, Mulkia (37), held out | 0.514 / 0.608 | **0.784 / 0.867** ¹ | 9/9 |

¹ Measured before the reference-mismatch fix (the four remaining reference errors are what it targets); re-run with
`cd pipeline && python test_agent.py --llm --sets mulkia` once API credit is available.

- The rule-based agent alone does not beat plain search; the gain comes from the LLM step. When rules and LLM agree,
  the top article was right in every evaluated case, so `uncertain` is a useful "needs review" signal.
- Labels were made by one person; the Mulkia labels are strict (the specific governing article; Annex 11 only when no
  specific article exists). The BlomInvest set was also used to tune the agent's prompt; Mulkia was not.
- Reports: `output/07_retrieval_eval` (search), `output/09_agent_eval` (agent). Test T&Cs, their labels and their
  reports stay out of the repo (`docproc/testset/`, git-ignored).

## Known limitations and next steps

- **Compliance Verification Agent** (next phase): judge each clause as Compliant / Violation / Needs Review with
  quotations. It also needs a checklist of mandatory requirements (Annex 11, arts 80–98) to detect *missing* clauses,
  which clause-by-clause retrieval cannot find.
- Text extraction maps `ق` to `ة` in some PDF fonts (`حةوق`); this weakens lexical matching.
- Segmentation sometimes merges the front matter or trailing sections (Shariah controls, complaints) into one clause.
- LLM results vary by a clause or two between runs; `segmentation_agent.py` still retries on HTTP 429 even when the
  account has no credit.
