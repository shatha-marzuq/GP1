"""
Retrieval over the CMA regulatory knowledge base (Chroma collection built by embed_and_store.py).

  dense  : BGE-M3 cosine similarity (Chroma HNSW)
  bm25   : lexical BM25 over the normalised `text_search` field (أ/إ/آ→ا، ى→ي، ة→ه + light prefix stripping)
  hybrid : reciprocal-rank fusion of the two lists
Default is dense: on the step-12 evaluation set it scored best (Hit@1 0.92, Hit@5 1.00, MRR 0.94);
hybrid/bm25 are kept for comparison (see output/07_retrieval_eval/report.md).

Usage from code (e.g. the MITHAQ T&C analyser):
    from retriever import RegulationRetriever
    r = RegulationRetriever()
    hits = r.search("يحق لمدير الصندوق تأجيل طلبات الاسترداد …", k=5, doc_ids=["investment_funds_regulations"])
    for h in hits: print(h["citation"], h["score"])

Recommended for T&C clauses: fund_type (drops look-alike provisions of other fund regimes) + rerank.
On 33 clauses of a real private-fund T&C: dense 0.39 Hit@1 → +fund_type 0.67 → +rerank 0.73 (MRR 0.81).
    hits = r.search(clause, k=5, fund_type="private", rerank=True)

CLI:  python3 retriever.py "نص البند أو السؤال" [--k 5] [--mode dense|hybrid|bm25] [--doc DOC_ID]
                                              [--fund-type private] [--rerank]
"""
import argparse, json, math, re
from collections import Counter

from chunk import search_normalize
from embed_and_store import COLLECTION, FUND_TYPE_SCOPES, chroma_client, load_model, pick_device

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # multilingual cross-encoder, same family as BGE-M3
RERANK_CANDIDATES = 20                     # first-stage hits re-scored (40 gave no gain on the T&C test, 2× slower)
PIN_SECTION = "IFR-anx11"                  # private-fund T&C disclosure checklist, pinned to rank 2 (see search)

DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")


def tokenize(text):
    toks = []
    for t in search_normalize(DIACRITICS.sub("", text)).split():
        for p in PREFIXES:
            if t.startswith(p) and len(t) - len(p) >= 3:
                t = t[len(p):]
                break
        if len(t) > 1:
            toks.append(t)
    return toks


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / len(docs)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q):
        out = []
        for tf, L in zip(self.tfs, self.lens):
            s = 0.0
            for t in set(q):
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


class RegulationRetriever:
    def __init__(self, device=None, model=None):
        self.col = chroma_client().get_collection(COLLECTION)
        self._model, self._device, self._reranker = model, device, None
        data = self.col.get(include=["metadatas", "documents"])
        self.ids, self.docs, self.metas = data["ids"], data["documents"], data["metadatas"]
        self.index = {i: n for n, i in enumerate(self.ids)}
        self.bm25 = BM25([tokenize(m["text_search"]) for m in self.metas])

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self._device)
        return self._model

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANK_MODEL, device=pick_device(self._device), max_length=1024)
        return self._reranker

    def embed_queries(self, queries):
        return self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True)

    @staticmethod
    def _where(doc_ids, section_types, binding_status, fund_type=None):
        conds = []
        if doc_ids:
            conds.append({"doc_id": {"$in": list(doc_ids)}})
        if section_types:
            conds.append({"section_type": {"$in": list(section_types)}})
        if binding_status:
            conds.append({"binding_status": binding_status})
        if fund_type:
            conds.append({"fund_scope": {"$in": FUND_TYPE_SCOPES[fund_type]}})
        return None if not conds else conds[0] if len(conds) == 1 else {"$and": conds}

    def _allowed(self, m, doc_ids, section_types, binding_status, fund_type=None):
        return ((not doc_ids or m["doc_id"] in doc_ids) and (not section_types or m["section_type"] in section_types)
                and (not binding_status or m["binding_status"] == binding_status)
                and (not fund_type or m["fund_scope"] in FUND_TYPE_SCOPES[fund_type]))

    def dense(self, query, n, where=None, qvec=None):
        qvec = self.embed_queries([query])[0] if qvec is None else qvec
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=n, where=where, include=["distances"])
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def lexical(self, query, n, filt):
        sc = self.bm25.scores(tokenize(query))
        ranked = sorted((k for k in range(len(self.ids)) if filt(self.metas[k])), key=lambda k: -sc[k])
        return [(self.ids[k], sc[k]) for k in ranked[:n] if sc[k] > 0]

    def search(self, query, k=5, mode="dense", doc_ids=None, section_types=None, binding_status=None,
               fund_type="private", rerank=True, candidates=50, rrf_k=60, qvec=None, pin_annex11=None):
        """Return the top-k chunks (dicts with id, score, citation, text, metadata …) for a query.

        Defaults match MITHAQ's scope (private-fund T&Cs): fund_type="private", rerank=True.
        fund_type ("public" | "private" | "foreign" | "simplified") keeps only the provisions that govern
        that kind of fund, e.g. "private" drops public-fund Part 4 and corporate governance; None = no filter.
        rerank=True re-scores the top RERANK_CANDIDATES first-stage hits with the cross-encoder
        (`score` becomes the reranker logit, the first-stage score is kept in `first_stage_score`).
        pin_annex11 (default: on when fund_type="private") leaves rank 1 alone but, if Annex 11 is in the
        candidate pool and not among the top two sections, moves its best chunk to rank 2 (marked `pinned`).
        Annex 11 lists the disclosures every private-fund T&C section maps to, yet general articles tend to
        outrank it; on 201 real T&C clauses this raised Hit@3 0.886 → 0.965 with Hit@1 unchanged."""
        if pin_annex11 is None:
            pin_annex11 = fund_type == "private"
        k_final, k = k, (max(k, RERANK_CANDIDATES) if rerank or pin_annex11 else k)
        where = self._where(doc_ids, section_types, binding_status, fund_type)
        filt = lambda m: self._allowed(m, doc_ids, section_types, binding_status, fund_type)
        if mode == "dense":
            ranked = self.dense(query, k, where, qvec)
        elif mode == "bm25":
            ranked = self.lexical(query, k, filt)
        elif mode == "hybrid":
            fused = Counter()
            for lst in (self.dense(query, candidates, where, qvec), self.lexical(query, candidates, filt)):
                for rank, (i, _) in enumerate(lst):
                    fused[i] += 1 / (rrf_k + rank + 1)
            ranked = fused.most_common(k)
        else:
            raise ValueError(mode)
        out = []
        for i, s in ranked:
            m = self.metas[self.index[i]]
            out.append({"id": i, "score": round(float(s), 4), "section_key": m["section_key"],
                        "citation": m["citation"], "binding_status": m["binding_status"],
                        "pdf_pages": m["pdf_pages"], "text": self.docs[self.index[i]], "metadata": m})
        if rerank and out:
            scores = self.reranker.predict([(query, h["text"]) for h in out], batch_size=8)
            for h, s in zip(out, scores):
                h["first_stage_score"], h["score"] = h["score"], round(float(s), 4)
            out.sort(key=lambda h: -h["score"])
        if pin_annex11 and out:
            top_two = {out[0]["section_key"]} | {next((h["section_key"] for h in out
                                                       if h["section_key"] != out[0]["section_key"]), None)}
            pin = next((h for h in out if h["section_key"] == PIN_SECTION), None)
            if pin and PIN_SECTION not in top_two:
                out.remove(pin)
                out.insert(1, {**pin, "pinned": True})
        return out[:k_final]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", default="dense", choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--doc", action="append", help="restrict to doc_id (repeatable)")
    ap.add_argument("--no-annex", action="store_true", help="exclude annexes (forms / templates)")
    ap.add_argument("--fund-type", default="private", choices=sorted(FUND_TYPE_SCOPES) + ["all"],
                    help="only provisions governing this fund type (default: private; 'all' = no filter)")
    ap.add_argument("--no-rerank", action="store_true", help=f"skip re-scoring the top {RERANK_CANDIDATES} with {RERANK_MODEL}")
    ap.add_argument("--no-pin-annex11", action="store_true", help=f"don't move {PIN_SECTION} up to rank 2")
    a = ap.parse_args()
    r = RegulationRetriever()
    for n, h in enumerate(r.search(a.query, a.k, a.mode, a.doc, ["article", "preamble"] if a.no_annex else None,
                                     fund_type=None if a.fund_type == "all" else a.fund_type,
                                     rerank=not a.no_rerank, pin_annex11=False if a.no_pin_annex11 else None), 1):
        print(f"\n#{n}  {h['id']}  score={h['score']}{'  (pinned)' if h.get('pinned') else ''}  [{h['binding_status']}]  pdf p.{h['pdf_pages']}")
        print("   ", h["citation"])
        print("   ", h["text"][:300].replace("\n", " ") + " …")
