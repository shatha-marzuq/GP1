# Report tables — MITHAQ (Release 1)

## Table: Retrieval accuracy (plain search vs full agent)

**blom** (33 clauses/queries; one item ≈ 3.0 pp)

| Configuration | Pool recall | Selection recall | Selection precision | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|---|
| Plain search (no agent) | 0.92 | – | – | 0.73 | 0.91 | 0.84 |
| Full agent | 0.96 | 0.90 | 0.66 | 0.85 | 1.00 | 0.92 |

**kb** (36 clauses/queries; one item ≈ 2.8 pp)

| Configuration | Pool recall | Selection recall | Selection precision | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|---|
| Plain search (no agent) | 0.98 | – | – | 0.89 | 0.97 | 0.93 |
| Full agent | 0.99 | 0.80 | 0.56 | 0.94 | 0.97 | 0.96 |

**mulkia** (37 clauses/queries; one item ≈ 2.7 pp)

| Configuration | Pool recall | Selection recall | Selection precision | Hit@1 | Hit@3 | MRR |
|---|---|---|---|---|---|---|
| Plain search (no agent) | 0.91 | – | – | 0.51 | 0.78 | 0.66 |
| Full agent | 0.96 | 0.91 | 0.57 | 0.76 | 0.95 | 0.86 |


The table compares the full Regulatory Retrieval Agent with plain vector search (dense + rerank, top results at the same depth). Pool recall is the share of the expected provisions among the candidates the agent retrieves; selection recall and precision measure the LLM's final choice among those candidates; Hit@k and MRR rank the first expected provision.

*Method:* labelled sets blom (33), kb (36), mulkia (37); retrieval LLM gpt-4o, dense search with cross-encoder rerank, private-fund scope, search depth 20, pool size 18. Each configuration was run once at temperature 0. With about 33–37 items per set, one item is ≈ 3 percentage points, so differences below 0.03 are within noise.

## Table: Processing time per file

| File | Pages | Type | OCR pages | Clauses | Total | Extraction | Segmentation | Retrieval | Collection | Total per clause | Failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T&C - Blominvest IPO Fund.pdf | 22 | text | 0 | 52 | 203.08 s (3.38 min) | 1.81 s | 7.80 s | 193.47 s (3.22 min) | 0.00 s | 3.90 s | none |
| mulkia.pdf | 35 | text | 0 | 80 | 308.50 s (5.14 min) | 3.25 s | 10.51 s | 294.72 s (4.91 min) | 0.00 s | 3.86 s | none |

End-to-end time of one review, from the start of the pipeline (file uploaded) to the final results: text extraction, clause segmentation by the Segmentation Agent, regulation retrieval for every clause by the Regulatory Retrieval Agent (8 clauses in parallel), and collection of the results.

*Method:* 2 T&C file(s) — T&C - Blominvest IPO Fund.pdf (22 pages, text), mulkia.pdf (35 pages, text); each file processed once; LLM answer cache off; embedding and reranker models loaded before timing (load time 15.33 s, excluded); hardware: Mac17,2 (Apple M5, 16.0 GB RAM, Apple MPS (GPU)); retrieval LLM openai / gpt-4o, segmentation LLM gpt-5-mini.

## Table: Average processing time per stage

| Stage | Mean | Median | Min | Max | Share of mean total |
|---|---|---|---|---|---|
| Extraction (docproc / OCR) | 2.53 s | 2.53 s | 1.81 s | 3.25 s | 1.0 % |
| Segmentation (agent) | 9.16 s | 9.16 s | 7.80 s | 10.51 s | 3.6 % |
| Retrieval (all clauses, parallel) | 244.09 s (4.07 min) | 244.09 s (4.07 min) | 193.47 s (3.22 min) | 294.72 s (4.91 min) | 95.4 % |
| Collection | 0.00 s | 0.00 s | 0.00 s | 0.00 s | 0.0 % |
| Total | 255.79 s (4.26 min) | 255.79 s (4.26 min) | 203.08 s (3.38 min) | 308.50 s (5.14 min) | 100.0 % |
| Retrieval per clause (wall time / clauses) | 3.70 s | 3.70 s | 3.68 s | 3.72 s | – |
| Total per clause | 3.88 s | 3.88 s | 3.86 s | 3.90 s | – |

The retrieval stage takes the largest share of the processing time (95.4 % of the mean total of 255.79 s (4.26 min)). Per-clause times are the stage's wall time divided by the number of clauses (clauses are processed in parallel).

*Method:* mean, median, minimum and maximum over the 2 file(s) of the previous table; same setup.
