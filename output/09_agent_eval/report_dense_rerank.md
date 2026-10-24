# Retrieval agent vs plain search — mode=dense, rerank=True

## blom (33 queries)

| | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| baseline | 0.727 | 0.909 | 0.97 | 0.833 |
| agent | 0.727 | 0.909 | 0.97 | 0.833 |

Decisions: confirmed 27 (Hit@1 0.889), confirmed_after_retry 2 (Hit@1 0.0), uncertain 4 (Hit@1 0.0)

Changed ranks: none

## kb (36 queries)

| | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| baseline | 0.889 | 0.972 | 1.0 | 0.933 |
| agent | 0.889 | 0.972 | 1.0 | 0.933 |

Decisions: confirmed 31 (Hit@1 0.935), uncertain 5 (Hit@1 0.6)

Changed ranks: none
