# Retrieval agent vs plain search — mode=dense, rerank=True, llm=gpt-4o

## blom (33 queries)

| | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| baseline | 0.727 | 0.909 | 0.97 | 0.833 |
| agent | 0.909 | 0.97 | 0.97 | 0.939 |

Decisions: confirmed_by_llm 21 (Hit@1 1.0), uncertain 12 (Hit@1 0.75)

Changed ranks: B02 2→1, B04 4→1, B11 2→1, B12 2→1, B21 2→1, B31 4→1

## kb (36 queries)

| | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| baseline | 0.889 | 0.972 | 1.0 | 0.933 |
| agent | 0.944 | 1.0 | 1.0 | 0.972 |

Decisions: confirmed_by_llm 28 (Hit@1 1.0), uncertain 8 (Hit@1 0.75)

Changed ranks: q05 3→1, q11 4→2, q14 2→1, q23 1→2, q24 2→1
