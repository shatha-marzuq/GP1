# Orchestrator (LangGraph)

Connects the team's agents into one fixed pipeline. Every clause runs in its own parallel branch.

```
extract (docproc) ─► segment (segmentation.py) ─► review_clause × N ─► report
        └────────────────┴──► failed
review_clause = retrieve (pipeline/retriever.py) ─► verify (LLM) ─► quote checks
                    ▲                                   │
                    └──── not enough context: new query, at most 2 retries
```

| File | What it does |
|---|---|
| `graph.py` | The graph: nodes, edges, parallel fan-out with `Send`, report |
| `state.py` | The shared state that travels through the graph |
| `verifier.py` | Verifier prompt, structured `Verdict`, and the quote checks |
| `stubs.py` | Stand-in retriever and verifier (no model, no API key) |
| `run.py` | Command-line runner |

## Run

```bash
cd backend
pip install -r orchestrator/requirements.txt
python -m pytest orchestrator/tests -q                    # graph behaviour with stand-ins
python -m orchestrator.run ../docproc/testset/test7.docx --stub   # real OCR + segmentation, stand-in retrieval/LLM
MITHAQ_LLM=azure_openai:gpt-4o python -m orchestrator.run terms.pdf   # everything real
```

The real run needs the pipeline's BGE-M3 model and Chroma store, and the provider's keys in the environment
(for Azure: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION`).

## Safety rules built in

- A verdict whose clause or regulation quote is not found word for word becomes **Needs Review**.
- A **Violation** must have both quotes and a citation, otherwise it becomes **Needs Review**.
- A Violation on OCR text in a document that docproc flagged (`needs_review`) becomes **Needs Review**.
- `min_score` (off by default) skips the LLM when retrieval finds nothing close. Calibrate it on
  `output/08_tc_evaluation/blom_clauses.json` before turning it on.

## Not done yet

- FastAPI endpoints and saving status/findings to a database (`on_status` is the hook).
- An Arabic summary paragraph in the report.
- The Annex 11 coverage check for missing clauses.
