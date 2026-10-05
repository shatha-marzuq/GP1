"""
Steps 9 → 11 — Prepare the chunks for embedding, embed them with BGE-M3, store them in Chroma.

  9) prepare : validate every chunk from step 7 and turn its metadata into Chroma-compatible scalars
               (Chroma accepts only str / int / float / bool, no None, no lists or dicts)
 10) embed   : dense BGE-M3 vectors (1024-d, L2-normalised) of the chunk `text` field
               (breadcrumb + body + footnotes); cached on disk and reused while the chunks are unchanged
 11) store   : (re)build a persistent Chroma collection (cosine space) with ids, documents, metadata
               and the pre-computed embeddings, then verify it

Input : output/04_chunks/chunks.jsonl
Output: output/05_embedding_input/records.jsonl + prep_report.json
        output/06_embeddings/embeddings.npy + ids.json + embedding_info.json
        output/chroma_db/   (collection "cma_regulations")

Usage:  python3 embed_and_store.py [--batch-size 8] [--device mps|cuda|cpu] [--force-embed]
"""
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "output" / "04_chunks" / "chunks.jsonl"
PREP = ROOT / "output" / "05_embedding_input"
EMB = ROOT / "output" / "06_embeddings"
CHROMA_DIR = ROOT / "output" / "chroma_db"
COLLECTION = "cma_regulations"
MODEL = "BAAI/bge-m3"
MAX_SEQ_TOKENS = 1024   # longest chunk is ≈ 2.6k chars; BGE-M3 supports 8192, 1024 keeps batches fast

# scalar metadata copied as-is (None values are dropped: Chroma rejects them)
SCALAR = ["doc_id", "doc_title", "doc_title_en", "doc_type", "issuer", "source_file", "version_decision",
          "version_date_gregorian", "section_type", "part_number", "part_title", "chapter_number",
          "chapter_title", "article_number", "article_label", "annex_number", "section_title",
          "chunk_index", "chunk_count", "binding_status", "citation", "breadcrumb", "word_count", "char_count"]
REQUIRED = ["chunk_id", "doc_id", "doc_title", "section_type", "citation", "text", "content", "binding_status"]


# Which fund regime a chunk governs — lets the T&C analyser drop look-alike provisions of other regimes
# (e.g. public-fund meetings art 77 vs private-fund meetings art 93, which embed almost identically).
#   general     : CMIR — applies to every fund manager
#   companies   : corporate governance — listed companies, not funds
#   funds_all   : IFR provisions common to public and private funds
#   simplified  : SIFI — replaces IFR for simplified funds (SIFI art 1/b)
IFR_PART_SCOPE = {4: "public", 5: "private", 6: "foreign"}          # other parts (1-3, 7-9) → funds_all
IFR_ANNEX_SCOPE = {"1": "public", "2": "public", "4": "public", "5": "public", "12": "public", "13": "public",
                   "6": "private_foreign", "9": "private_foreign", "7": "private", "11": "private", "8": "foreign"}
DOC_SCOPE = {"capital_market_institutions_regulations": "general",
             "corporate_governance_regulations": "companies",
             "simplified_investment_funds_instructions": "simplified"}
# fund type → fund_scope values that apply to it (used by retriever.search(fund_type=…))
FUND_TYPE_SCOPES = {"public": ["general", "funds_all", "public"],
                    "private": ["general", "funds_all", "private", "private_foreign"],
                    "foreign": ["general", "funds_all", "foreign", "private_foreign"],
                    "simplified": ["general", "simplified"]}


def fund_scope(c):
    if c["doc_id"] in DOC_SCOPE:
        return DOC_SCOPE[c["doc_id"]]
    if c["section_type"] == "annex":
        return IFR_ANNEX_SCOPE.get(c["annex_number"], "funds_all")
    return IFR_PART_SCOPE.get(c.get("part_number"), "funds_all")


# ───────────────────────────── step 9: prepare ─────────────────────────────
def to_metadata(c):
    m = {k: c[k] for k in SCALAR if c.get(k) is not None}
    m["section_key"] = c["chunk_id"].rsplit("-", 1)[0]          # e.g. CMIR-art043 — groups split articles
    m["fund_scope"] = fund_scope(c)
    m["paragraphs"] = "، ".join(c["paragraphs"])                  # "أ، ب" — readable
    m["pdf_pages"] = ",".join(map(str, c["pdf_pages"]))
    m["pdf_page_start"] = c["pdf_pages"][0] if c["pdf_pages"] else 0
    m["guidance_paragraphs"] = json.dumps(c["guidance_paragraphs"], ensure_ascii=False)
    m["has_guidance_paragraphs"] = bool(c["guidance_paragraphs"])
    m["footnotes"] = json.dumps(c["footnotes"], ensure_ascii=False)
    m["has_footnotes"] = bool(c["footnotes"])
    m["text_search"] = c["text_search"]                           # lets the BM25 side rebuild from Chroma alone
    return m


def prepare():
    chunks = [json.loads(l) for l in CHUNKS.read_text(encoding="utf-8").splitlines() if l.strip()]
    problems = []
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate chunk ids")
    for c in chunks:
        missing = [k for k in REQUIRED if not c.get(k)]
        if missing:
            problems.append(f"{c.get('chunk_id')}: missing {missing}")
        if not c["text"].startswith("[") or c["breadcrumb"] not in c["text"]:
            problems.append(f"{c['chunk_id']}: text does not start with its breadcrumb")
    if problems:
        raise SystemExit("chunk validation failed:\n  " + "\n  ".join(problems[:20]))

    records = [{"id": c["chunk_id"], "document": c["text"], "metadata": to_metadata(c)} for c in chunks]
    for r in records:
        bad = {k: type(v).__name__ for k, v in r["metadata"].items() if not isinstance(v, (str, int, float, bool))}
        assert not bad, (r["id"], bad)

    PREP.mkdir(parents=True, exist_ok=True)
    with open(PREP / "records.jsonl", "w", encoding="utf-8") as o:
        for r in records:
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    digest = hashlib.sha256("".join(r["id"] + r["document"] for r in records).encode()).hexdigest()
    report = {"records": len(records), "content_sha256": digest,
              "by_doc": {d: sum(r["metadata"]["doc_id"] == d for r in records)
                         for d in sorted({r["metadata"]["doc_id"] for r in records})},
              "by_section_type": {t: sum(r["metadata"]["section_type"] == t for r in records)
                                  for t in sorted({r["metadata"]["section_type"] for r in records})},
              "chars_max": max(len(r["document"]) for r in records),
              "metadata_fields": sorted({k for r in records for k in r["metadata"]})}
    (PREP / "prep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[9] prepared {len(records)} records  {report['by_doc']}")
    return records, digest


# ───────────────────────────── step 10: embed ─────────────────────────────
def pick_device(pref=None):
    import torch
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device=None):
    from sentence_transformers import SentenceTransformer
    # the official repo ships pytorch_model.bin; skip transformers' slow lookup of a safetensors conversion PR
    model = SentenceTransformer(MODEL, device=pick_device(device), model_kwargs={"use_safetensors": False})
    model.max_seq_length = MAX_SEQ_TOKENS
    return model


def embed(records, digest, batch_size, device, force):
    EMB.mkdir(parents=True, exist_ok=True)
    info_f, vec_f, ids_f = EMB / "embedding_info.json", EMB / "embeddings.npy", EMB / "ids.json"
    if not force and info_f.exists() and vec_f.exists():
        info = json.loads(info_f.read_text())
        if info.get("content_sha256") == digest and info.get("model") == MODEL:
            print(f"[10] chunks unchanged — reusing cached embeddings {vec_f.relative_to(ROOT)}")
            return np.load(vec_f)

    model = load_model(device)
    docs = [r["document"] for r in records]
    tok_lens = [len(model.tokenizer(d, add_special_tokens=True)["input_ids"]) for d in docs]
    truncated = sum(n > MAX_SEQ_TOKENS for n in tok_lens)
    # sort by length so each batch has similar padding → much faster on GPU/MPS
    order = sorted(range(len(docs)), key=lambda i: tok_lens[i])
    t0 = time.time()
    vecs = model.encode([docs[i] for i in order], batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    out = np.empty_like(vecs)
    out[order] = vecs
    out = out.astype(np.float32)
    secs = round(time.time() - t0, 1)

    assert out.shape == (len(records), 1024) and np.isfinite(out).all()
    np.save(vec_f, out)
    ids_f.write_text(json.dumps([r["id"] for r in records]), encoding="utf-8")
    info = {"model": MODEL, "dim": int(out.shape[1]), "normalized": True, "count": len(records),
            "max_seq_tokens": MAX_SEQ_TOKENS, "device": str(model.device), "seconds": secs,
            "tokens_max": max(tok_lens), "tokens_mean": round(sum(tok_lens) / len(tok_lens)),
            "truncated_chunks": truncated, "content_sha256": digest}
    info_f.write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[10] embedded {len(records)} chunks on {model.device} in {secs}s "
          f"(max {max(tok_lens)} tokens, {truncated} truncated)")
    return out


# ───────────────────────────── step 11: store ─────────────────────────────
def chroma_client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def store(records, vecs):
    client = chroma_client()
    # rebuild from scratch so chunks deleted/renamed upstream never linger in the index
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    col = client.create_collection(COLLECTION, embedding_function=None, metadata={
        "hnsw:space": "cosine", "embedding_model": MODEL, "dim": int(vecs.shape[1]),
        "source": "CMA regulations (4 documents)"})
    B = 128
    for k in range(0, len(records), B):
        part = records[k:k + B]
        col.add(ids=[r["id"] for r in part], documents=[r["document"] for r in part],
                metadatas=[r["metadata"] for r in part], embeddings=vecs[k:k + B].tolist())

    # verify: count, round-trip of one record, and every chunk retrieves itself as top-1
    assert col.count() == len(records), (col.count(), len(records))
    got = col.get(ids=[records[0]["id"]], include=["documents", "metadatas"])
    assert got["documents"][0] == records[0]["document"]
    res = col.query(query_embeddings=vecs.tolist(), n_results=1, include=[])
    self_hits = sum(res["ids"][i][0] == r["id"] for i, r in enumerate(records))
    print(f"[11] stored {col.count()} chunks in {CHROMA_DIR.relative_to(ROOT)} (collection '{COLLECTION}'); "
          f"self-retrieval {self_hits}/{len(records)}")
    return col


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", choices=["mps", "cuda", "cpu"], default=None)
    ap.add_argument("--force-embed", action="store_true", help="recompute embeddings even if cached")
    args = ap.parse_args()
    records, digest = prepare()
    vecs = embed(records, digest, args.batch_size, args.device, args.force_embed)
    store(records, vecs)
