# Processing time — summary over 3 file(s)

| Stage | Mean | Median | Min | Max | Share of mean total |
|---|---|---|---|---|---|
| Document Processing — extraction (docproc / OCR) | 2.45 s | 2.29 s | 1.81 s | 3.25 s | 1.0 % |
| Document Processing — segmentation (LLM review) | 8.71 s | 7.83 s | 7.80 s | 10.51 s | 3.5 % |
| **Document Processing Agent (extraction + segmentation)** | 11.17 s | 10.12 s | 9.61 s | 13.77 s | 4.5 % |
| Retrieval (all clauses, parallel) | 239.56 s (3.99 min) | 230.48 s (3.84 min) | 193.47 s (3.22 min) | 294.72 s (4.91 min) | 95.5 % |
| Collection | 0.00 s | 0.00 s | 0.00 s | 0.01 s | 0.0 % |
| Total | 250.74 s (4.18 min) | 240.65 s (4.01 min) | 203.08 s (3.38 min) | 308.50 s (5.14 min) | 100.0 % |
| Retrieval per clause (wall time / clauses) | 3.61 s | 3.68 s | 3.44 s | 3.72 s | – |
| Total per clause | 3.78 s | 3.86 s | 3.59 s | 3.90 s | – |
