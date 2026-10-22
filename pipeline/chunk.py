import argparse, json, re
from pathlib import Path
from statistics import mean, median
#القطعة ما تقطع مادة أبدًا. مادة كاملة = قطعة وحدة إذا حجمها أقل من 300 كلمة.
#والـ300 كلمة عربية تقريبًا تناسب نماذج embedding اللي حدها 512 token.
ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "output" / "03_structured"
OUT = ROOT / "output" / "04_chunks"
#اختصارات المستندات
SHORT = {"investment_funds_regulations": "IFR", "capital_market_institutions_regulations": "CMIR",
         "corporate_governance_regulations": "CGR", "simplified_investment_funds_instructions": "SIFI"}
FOOT = re.compile(r"\s*\[\^(\d+)\]")
EXTRACTED = ROOT / "output" / "01_extracted"

#رقم الصفحة المكتوب في مستند الهيئة (أسفل الصفحة) لكل صفحة PDF
#الغلاف ما عليه رقم، فالصفحة المكتوبة = صفحة الـ PDF ناقص 1
def printed_page_numbers(doc_id):
    numbers = {}
    for p in json.loads((EXTRACTED / f"{doc_id}.json").read_text(encoding="utf-8"))["pages"]:
        digits = [l["text"].strip() for l in p["lines"] if l["text"].strip().isdigit()]
        if digits:
            numbers[p["page"]] = int(digits[-1])
    return numbers

#عدد الكلمات
def words(t):
    return len(t.split())

#توحيد للبحث النصي (BM25)
def search_normalize(t):
    t = re.sub(r"[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

#تجمع كل فقرة رئيسية (مستوى 1 أو أقل) مع فقراتها الفرعية في وحدة وحدة.
def units_of(section):
    """Group paragraphs: a top-level paragraph (level ≤ 1) + its deeper sub-paragraphs."""
    units = []
    for p in section["paragraphs"]:
        if not units or p["level"] <= 1 or p.get("is_subheading"):
            units.append([p])
        else:
            units[-1].append(p)
    return units

#تقسم عند . أو ؛، وتجمع الجمل لين توصل الحد.
def split_sentences(text, max_words):
    parts = re.split(r"(?<=[.؛])\s+", text)
    out, cur = [], ""
    for s in parts:
        if cur and words(cur) + words(s) > max_words:
            out.append(cur); cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    # still too long (no sentence breaks) → hard split on commas / words
    final = []
    for o in out:
        if words(o) <= max_words:
            final.append(o); continue
        w = o.split()
        for k in range(0, len(w), max_words):
            final.append(" ".join(w[k:k + max_words]))
    return final


def pieces_of(section, max_words):
    """Yield pieces: dict(paras=[...], lead=optional lead-in text). Each piece ≤ max_words when possible."""
    pieces = []
    intro = None
    for u in units_of(section):
        head = u[0]
        if head["level"] == 0 and head["text"].rstrip().endswith(":") and not pieces:
            intro = head["text"]
        txt = " ".join(p["text"] for p in u)
        if words(txt) <= max_words:
            pieces.append({"paras": u, "lead": None})
            continue
        # unit too long: split between its sub-paragraphs, repeating the lead-in
        lead = head["text"] if len(u) > 1 else None
        if lead and words(lead) <= max_words // 2:
            pieces.append({"paras": [head], "lead": None, "lead_of_next": True})
            for p in u[1:]:
                if words(p["text"]) <= max_words:
                    pieces.append({"paras": [p], "lead": lead})
                else:
                    for s in split_sentences(p["text"], max_words):
                        pieces.append({"paras": [{**p, "text": s}], "lead": lead})
        else:
            for p in u:
                for s in split_sentences(p["text"], max_words):
                    pieces.append({"paras": [{**p, "text": s}], "lead": None})
    return pieces, intro

#التجميع
def pack(pieces, max_words, intro):
    # balance chunk sizes: aim for equal parts instead of one full chunk + a small remainder
    total = sum(words(p["text"]) for pc in pieces for p in pc["paras"])
    n = max(1, -(-total // max_words))
    max_words = min(max_words, int(total / n * 1.2) + 1) if n > 1 else max_words
    chunks, cur = [], None
    for pc in pieces:
        w = sum(words(p["text"]) for p in pc["paras"])
        if cur and cur["words"] + w <= max_words and (pc["lead"] is None or pc["lead"] in cur["leads"]
                                                      or cur.get("open_lead") == pc["lead"]):
            cur["paras"] += pc["paras"]; cur["words"] += w
        else:
            cur = {"paras": list(pc["paras"]), "words": w, "context": None, "leads": set()}
            if pc["lead"]:
                cur["context"] = pc["lead"]
            elif chunks and intro and pc["paras"][0]["text"] != intro:
                cur["context"] = intro
            chunks.append(cur)
        if pc.get("lead_of_next"):
            cur["open_lead"] = pc["paras"][0]["text"]
        if pc["lead"]:
            cur["leads"].add(pc["lead"])
    return chunks

#مسار المادة كامل
def breadcrumb(doc, s):
    parts = [doc["title"]]
    if s["part"]:
        parts.append(f"{s['part']['label']}: {s['part']['title']}" if s["part"].get("title") else s["part"]["label"])
    if s["chapter"]:
        parts.append(f"{s['chapter']['label']}: {s['chapter']['title']}" if s["chapter"].get("title") else s["chapter"]["label"])
    if s["type"] != "preamble":
        parts.append(f"{s['label']}: {s['title']}" if s.get("title") else s["label"])
    return " › ".join(parts)


def para_text(p):
    return FOOT.sub(lambda m: f" [{m.group(1)}]", p["text"]).strip()

#بناء القطعة
def chunk_doc(doc, max_words):
    out = []
    code = SHORT[doc["doc_id"]]
    printed = printed_page_numbers(doc["doc_id"])
    for s in doc["sections"]:
        if not s["paragraphs"]:
            continue
        pieces, intro = pieces_of(s, max_words)
        packed = pack(pieces, max_words, intro)
        if s["type"] == "article":
            sid = f"art{s['number']:03d}"
        elif s["type"] == "annex":
            sid = "anx" + re.sub(r"[^\w]", "", s["number"]).replace("ا", "")
        else:
            sid = "pre" + str(s["pages"][0])
        crumb = breadcrumb(doc, s)
        for k, ch in enumerate(packed, 1):
            paths = [p["path"] for p in ch["paras"] if p.get("path")]
            top = []
            for pth in paths:
                m = re.match(r"^\(([^)]+)\)", pth)
                if m and m.group(1) not in top:
                    top.append(m.group(1))
            body = "\n".join(para_text(p) for p in ch["paras"])
            fns = {int(n) for p in ch["paras"] for n in FOOT.findall(p["text"])}
            fn_items = [f for f in s["footnotes"] if f["scope"] == "section" or f["n"] in fns]
            header = crumb + (f" (الجزء {k} من {len(packed)})" if len(packed) > 1 else "")
            text = f"[{header}]\n"
            if ch["context"]:
                text += f"(سياق: {FOOT.sub('', ch['context']).strip()})\n"
            text += body
            if fn_items:
                text += "\nالحواشي:\n" + "\n".join(f"[{f['n']}] {f['text']}" for f in fn_items)
            guidance_paras = [p["path"] for p in ch["paras"] if p.get("binding_status") == "guidance"]
            citation = f"{doc['title']}، {s['label'] or ''}".strip("، ")
            if len(packed) > 1 and top:
                citation += "، " + ("الفقرة" if len(top) == 1 else "الفقرات") + " " + "، ".join(f"({t})" for t in top)
            pages = sorted({pg for p in ch["paras"] for pg in p["pages"]}) or s["pages"]
            cma_pages = [printed[pg] for pg in pages if pg in printed]
            out.append({
                "chunk_id": f"{code}-{sid}-c{k:02d}",
                "doc_id": doc["doc_id"],
                "doc_title": doc["title"],
                "doc_title_en": doc["title_en"],
                "doc_type": doc["doc_type"],
                "issuer": "هيئة السوق المالية",
                "source_file": doc["source_file"],
                "version_decision": doc["last_amended_decision"] or doc["issued_by_decision"],
                "version_date_gregorian": doc["last_amended_gregorian"] or doc["issued_gregorian"],
                "section_type": s["type"],
                "part_number": (s["part"] or {}).get("number"),
                "part_title": (s["part"] or {}).get("title"),
                "chapter_number": (s["chapter"] or {}).get("number"),
                "chapter_title": (s["chapter"] or {}).get("title"),
                "article_number": s["number"] if s["type"] == "article" else None,
                "article_label": s["label"] if s["type"] == "article" else None,
                "annex_number": s["number"] if s["type"] == "annex" else None,
                "section_title": s.get("title"),
                "chunk_index": k,
                "chunk_count": len(packed),
                "paragraphs": top,
                "pdf_pages": pages,
                #أول وآخر صفحة للقطعة بأرقام صفحات مستند الهيئة نفسه (مو ترقيم الـ PDF)
                "page_start": min(cma_pages) if cma_pages else None,
                "page_end": max(cma_pages) if cma_pages else None,
                "pages": sorted(cma_pages),
                "binding_status": s["binding_status"],
                "guidance_paragraphs": guidance_paras,
                "footnotes": fn_items,
                "citation": citation,
                "breadcrumb": crumb,
                "content": body,
                "text": text,
                "text_search": search_normalize(text),
                "word_count": words(text),
                "char_count": len(text),
            })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-words", type=int, default=300,
                    help="max Arabic words of body text per chunk (≈ 300 words fits 512-token embedders)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    allc, stats = [], {}
    for f in sorted(IN.glob("*.json")):
        if f.name == "validation.json":
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        cs = chunk_doc(doc, args.max_words)
        with open(OUT / f"{doc['doc_id']}.jsonl", "w", encoding="utf-8") as o:
            for c in cs:
                o.write(json.dumps(c, ensure_ascii=False) + "\n")
        allc += cs
        wc = [c["word_count"] for c in cs]
        stats[doc["doc_id"]] = {"chunks": len(cs), "articles": sum(s["type"] == "article" for s in doc["sections"]),
                                "annexes": sum(s["type"] == "annex" for s in doc["sections"]),
                                "split_sections": len({c["chunk_id"].rsplit('-', 1)[0] for c in cs if c["chunk_count"] > 1}),
                                "words_min": min(wc), "words_median": median(wc), "words_mean": round(mean(wc)),
                                "words_max": max(wc)}
        print(doc["doc_id"], stats[doc["doc_id"]])
    with open(OUT / "chunks.jsonl", "w", encoding="utf-8") as o:
        for c in allc:
            o.write(json.dumps(c, ensure_ascii=False) + "\n")
    ids = [c["chunk_id"] for c in allc]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    stats["_total"] = {"chunks": len(allc), "max_words_setting": args.max_words}
    (OUT / "chunk_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print("total", len(allc))
