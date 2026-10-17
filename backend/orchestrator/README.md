# Orchestrator (LangGraph)

Connects the document processing (OCR) agent and the regulatory retrieval agent (vector DB) into one fixed pipeline.
Every clause is searched in its own parallel branch.

```
extract (docproc) ─► segment (segmentation.py) ─► retrieve × N (pipeline/retriever.py) ─► collect
        └────────────────┴──► failed
```

The output is every clause (with its pages) plus the top-k CMA regulations retrieved for it.
Compliance verification and the report are later phases and are not in this graph.

| File | What it does |
|---|---|
| `graph.py` | The graph: nodes, edges, parallel fan-out with `Send`, collected output |
| `state.py` | The shared state that travels through the graph |
| `stubs.py` | Stand-in retriever (no BGE-M3 model, no Chroma store) |
| `run.py` | Command-line runner |

## Run

```bash
cd backend
pip install -r orchestrator/requirements.txt
python -m pytest orchestrator/tests -q                              # graph behaviour with a stand-in retriever
python -m orchestrator.run ../docproc/testset/test7.docx --stub     # real OCR + segmentation, stand-in retrieval
python -m orchestrator.run terms.pdf --mode hybrid -k 5             # real retrieval (BGE-M3 + Chroma store)
```

Writes `retrieval.json`.
