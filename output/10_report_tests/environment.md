# Test environment

| Item | Value |
|---|---|
| date | 2026-11-18T20:27:58 |
| machine | Mac17,2 |
| os | Darwin 27.0.0 (macOS 27.0) |
| cpu | Apple M5 |
| cpu cores | 10 |
| memory gb | 16.0 |
| accelerator | Apple MPS (GPU) |
| torch device | mps |
| python | 3.14.7 |
| torch | 2.14.1 |
| embedding model | BAAI/bge-m3 |
| reranker | BAAI/bge-reranker-v2-m3 |
| retrieval llm | openai / gpt-4o |
| segmentation llm | gpt-5-mini |
| retrieval mode | dense |
| llm cache | off (RETRIEVAL_CACHE=0); the Segmentation Agent has no cache |
| max concurrency | 8 |
| model load seconds | 15.33 |
| warmup seconds | 4.31 |
| setup | Segmentation Agent (LLM) + Regulatory Retrieval Agent (LLM), as in run.py --seg-agent --agent --llm |

## Retrieval accuracy runs (pipeline/test_agent.py)

| Item | Value |
|---|---|
| mode | dense |
| rerank | True |
| llm | gpt-4o |
| search depth | 20 |
| pool size | 18 |
| ablate | None |
| llm cache | True |
