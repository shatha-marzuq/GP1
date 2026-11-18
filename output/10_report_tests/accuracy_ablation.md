# Retrieval accuracy — full agent vs ablations (all labels, values in 0–1, Δ vs full agent)

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


| Run | Set | s / clause | LLM answers | LLM API calls (not cached) | Valid |
|---|---|---|---|---|---|
| Full agent | blom | 8.50 | 66 | 66 | True |
| Full agent | kb | 8.40 | 72 | 72 | True |
| Full agent | mulkia | 12.50 | 74 | 74 | True |
