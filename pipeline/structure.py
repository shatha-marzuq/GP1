"""
Step 6 — Structure the regulatory information.

Input : output/02_clean/<doc_id>.json
Output: output/03_structured/<doc_id>.json   hierarchy: document → part (الباب) → chapter (الفصل)
                                              → article (المادة) / annex (الملحق) → paragraphs
        output/03_structured/<doc_id>.md     readable rendering for review
        output/03_structured/validation.json  structure checked against each document's own TOC
"""
import json, re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "output" / "02_clean"
OUT = ROOT / "output" / "03_structured"

# Source metadata read from each document's cover page
DOC_META = {
    "investment_funds_regulations": {
        "title": "لائحة صناديق الاستثمار", "title_en": "Investment Funds Regulations",
        "doc_type": "لائحة", "issuer": "مجلس هيئة السوق المالية",
        "issued_by_decision": "1-219-2006", "issued_hijri": "1427/12/3", "issued_gregorian": "2006-12-24",
        "last_amended_decision": "1-135-2025", "last_amended_hijri": "1447/06/03", "last_amended_gregorian": "2025-11-24",
        "legal_basis": "نظام السوق المالية الصادر بالمرسوم الملكي رقم م/30 وتاريخ 1424/6/2هـ",
        "source_file": "Investment_Funds_Regulations_11_2025_AR.pdf"},
    "capital_market_institutions_regulations": {
        "title": "لائحة مؤسسات السوق المالية", "title_en": "Capital Market Institutions Regulations",
        "doc_type": "لائحة", "issuer": "مجلس هيئة السوق المالية",
        "issued_by_decision": "1-83-2005", "issued_hijri": "1426/5/21", "issued_gregorian": "2005-06-28",
        "last_amended_decision": "2-3-2026", "last_amended_hijri": "1447/07/18", "last_amended_gregorian": "2026-01-07",
        "legal_basis": "نظام السوق المالية الصادر بالمرسوم الملكي رقم م/30 وتاريخ 1424/6/2هـ",
        "source_file": "the_Capital_Market_Institutions_Regulations-ar.pdf"},
    "corporate_governance_regulations": {
        "title": "لائحة حوكمة الشركات", "title_en": "Corporate Governance Regulations",
        "doc_type": "لائحة", "issuer": "مجلس هيئة السوق المالية",
        "issued_by_decision": "8-16-2017", "issued_hijri": "1438/5/16", "issued_gregorian": "2017-02-13",
        "last_amended_decision": "8-5-2023", "last_amended_hijri": "1444/6/25", "last_amended_gregorian": "2023-01-18",
        "legal_basis": "نظام الشركات الصادر بالمرسوم الملكي رقم م/132 وتاريخ 1443/12/1هـ",
        "source_file": "CorpGovReg.pdf"},
    "simplified_investment_funds_instructions": {
        "title": "تعليمات صناديق الاستثمار المبسطة", "title_en": "Instructions of Simplified Investment Funds",
        "doc_type": "تعليمات", "issuer": "مجلس هيئة السوق المالية",
        "issued_by_decision": "1-26-2026", "issued_hijri": "1447/09/13", "issued_gregorian": "2026-03-02",
        "last_amended_decision": None, "last_amended_hijri": None, "last_amended_gregorian": None,
        "legal_basis": "نظام السوق المالية الصادر بالمرسوم الملكي رقم م/30 وتاريخ 1424/06/02هـ",
        "source_file": "Instructions_of_Simplified_Investment_Funds_AR.pdf"},
}

# ---------------------------------------------------------------- Arabic ordinals → int
UNITS = {"الأول": 1, "الأولى": 1, "الحادي": 1, "الحادية": 1, "الثاني": 2, "الثانية": 2, "الثالث": 3,
         "الثالثة": 3, "الرابع": 4, "الرابعة": 4, "الخامس": 5, "الخامسة": 5, "السادس": 6, "السادسة": 6,
         "السابع": 7, "السابعة": 7, "الثامن": 8, "الثامنة": 8, "التاسع": 9, "التاسعة": 9,
         "العاشر": 10, "العاشرة": 10}
TENS = {"العشرون": 20, "الثلاثون": 30, "الأربعون": 40, "الخمسون": 50, "الستون": 60, "السبعون": 70,
        "الثمانون": 80, "التسعون": 90, "المئة": 100, "المائة": 100}


def ordinal(s):
    total, seen = 0, False
    for tok in s.split():
        if tok in ("بعد",):
            continue
        if tok in ("عشر", "عشرة"):
            total += 10; continue
        t = tok[1:] if tok.startswith("و") and tok[1:] in {**TENS, **UNITS} else tok
        if t in UNITS:
            total += UNITS[t]; seen = True
        elif t in TENS:
            total += TENS[t]; seen = True
        else:
            return None
    return total if seen else None


ORD = r"(?:ال[ء-ي]+)(?:\s+(?:و?ال[ء-ي]+|عشرة?|بعد))*"
RE_PART = re.compile(rf"^الباب\s+({ORD})\s*(?::\s*(.+))?$")
RE_CHAP = re.compile(rf"^الفصل\s+({ORD})\s*(?::\s*(.+))?$")
RE_ART = re.compile(rf"^(المادة\s+)?({ORD})\s*:\s*(.+)$")
RE_ANNEX = re.compile(r"^(?:الملحق|ملحق)\s*\(?\s*(\d+(?:\s*-\s*\d+)?)\s*\)?(?:\s*\(([ء-ي])\))?\s*(?::\s*)?(.*)$")

LETTER = r"[ء-ي]ـ?"
MARKER = re.compile(
    rf"^(?:\(?({LETTER})\)\s*|\((\d{{1,3}})\)\s*|(\d{{1,3}})\s*[).\-]\s*|([•▪o])\s+|"
    r"(أولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\s*:\s*)")


def marker_of(text):
    m = MARKER.match(text)
    if not m:
        return None
    letter, pnum, num, bullet, ordw = m.groups()
    if letter:
        return {"marker": letter.rstrip("ـ"), "kind": "letter", "raw": m.group(0).strip()}
    if pnum or num:
        return {"marker": pnum or num, "kind": "number", "raw": m.group(0).strip()}
    if bullet:
        return {"marker": "•", "kind": "bullet", "raw": m.group(0).strip()}
    return {"marker": ordw, "kind": "ordinal_word", "raw": m.group(0).strip()}


FOOT = re.compile(r"\[\^(\d+)\]")


class Builder:
    def __init__(self, clean):
        self.c = clean
        self.body = clean["body"]
        bs = clean["body_font_size"]
        xs = sorted(b["x0"] for b in self.body if b["size"] == bs)
        x1s = Counter(round(b["x1"]) for b in self.body if b["size"] == bs)
        self.left = xs[len(xs) // 10] if xs else 70
        self.right = x1s.most_common(1)[0][0] if x1s else 530
        self.bs = bs

    def centered(self, b):
        mid = (b["x0"] + b["x1"]) / 2
        block_mid = (self.left + self.right) / 2
        return abs(mid - block_mid) < 40 and (b["x1"] - b["x0"]) < 0.8 * (self.right - self.left)

    def headingish(self, b):
        return b["size"] > self.bs + 0.5 or (self.centered(b) and not b["text"].endswith((".", "،")))

    def short(self, b):
        return b["x0"] > self.left + 0.12 * (self.right - self.left)


def build(clean):
    B = Builder(clean)
    meta = DOC_META[clean["doc_id"]]
    doc = {"doc_id": clean["doc_id"], **meta, "body_font_size": clean["body_font_size"],
           "sections": []}
    part = chapter = None
    sec = None
    expected_art = 1
    body = B.body
    i = 0

    def new_section(kind, **kw):
        nonlocal sec
        sec = {"type": kind, "part": part, "chapter": chapter, "paragraphs": [], "pages": [],
               "heading_footnotes": [], **kw}
        doc["sections"].append(sec)
        return sec

    def take_title(j):
        """Title on the following centered line (for 'الباب الأول' style headings)."""
        def ok(x):
            return (B.centered(x) or x["size"] > B.bs + 0.5) and not RE_ART.match(x["text"]) \
                and not RE_CHAP.match(x["text"]) and not RE_ANNEX.match(x["text"]) \
                and not RE_PART.match(x["text"]) and not marker_of(x["text"])
        if j < len(body) and ok(body[j]):
            title, size = body[j]["text"], body[j]["size"]
            j += 1
            # long titles wrap onto a second line in the same (large) font
            while j < len(body) and body[j]["size"] == size and size > B.bs + 0.5 and ok(body[j]):
                title += " " + body[j]["text"]; j += 1
            return title, j
        return None, j

    prev = None
    while i < len(body):
        b = body[i]
        t = b["text"]
        tf = FOOT.sub("", t).strip()
        foots = FOOT.findall(t)
        m_part, m_chap, m_art, m_anx = RE_PART.match(tf), RE_CHAP.match(tf), RE_ART.match(tf), RE_ANNEX.match(tf)

        if m_part and B.headingish(b) and ordinal(m_part.group(1)):
            title = m_part.group(2)
            j = i + 1
            if not title:
                title, j = take_title(j)
            part = {"number": ordinal(m_part.group(1)), "label": f"الباب {m_part.group(1)}", "title": title}
            chapter = None; sec = None; prev = None; i = j; continue

        if m_chap and B.headingish(b) and ordinal(m_chap.group(1)):
            title = m_chap.group(2)
            j = i + 1
            if not title:
                title, j = take_title(j)
            chapter = {"number": ordinal(m_chap.group(1)), "label": f"الفصل {m_chap.group(1)}", "title": title}
            sec = None; prev = None; i = j; continue

        if m_art:
            n = ordinal(m_art.group(2))
            has_word = bool(m_art.group(1))
            # accept a heading without the word "المادة" only when it is exactly the next expected article
            if n and (has_word or n == expected_art) and not b["text"].startswith(("أ)", "ب)")):
                title = m_art.group(3).strip()
                # article titles occasionally wrap onto a second (short, right-aligned) line
                if b["x0"] <= B.left + 5 and i + 1 < len(body) and not marker_of(body[i + 1]["text"]) \
                        and B.short(body[i + 1]) and len(body[i + 1]["text"].split()) <= 6:
                    title += " " + body[i + 1]["text"]; i += 1
                ordtxt = m_art.group(2)
                new_section("article", number=n, label=f"المادة {ordtxt}", title=title,
                            repaired_heading=not has_word)
                sec["pages"].append(b["page"])
                sec["heading_footnotes"] += foots
                expected_art = n + 1
                prev = None; i += 1; continue

        if m_anx and B.headingish(b) and not tf.endswith("."):
            num = re.sub(r"\s+", "", m_anx.group(1))
            sub = m_anx.group(2)
            title = m_anx.group(3).strip() or None
            j = i + 1
            if not title:
                title, j = take_title(j)
            label = f"الملحق {num}" + (f" ({sub})" if sub else "")
            new_section("annex", number=num + (f"({sub})" if sub else ""), label=label, title=title)
            sec["pages"].append(b["page"])
            sec["heading_footnotes"] += foots
            prev = None; i = j; continue

        # ---- body text
        if sec is None:
            new_section("preamble", number=None, label=(chapter or part or {}).get("label"),
                        title=(chapter or part or {}).get("title"))
        if b["page"] not in sec["pages"]:
            sec["pages"].append(b["page"])
        mk = marker_of(t)
        start_new = False
        if not sec["paragraphs"]:
            start_new = True
        elif mk and (prev is None or B.short(prev) or prev["text"].endswith((".", ":", "؛", "،", "؟"))
                     or (mk["kind"] in ("letter", "ordinal_word") and B.right - b["x1"] < 8)):
            start_new = True
        elif prev is not None and B.short(prev) and prev["text"].endswith((".", ":")):
            start_new = True
        elif B.headingish(b) and b["size"] > B.bs + 0.5 and (prev is None or prev["size"] != b["size"]):
            start_new = True  # sub-heading inside an annex
        if start_new:
            sec["paragraphs"].append({"marker": mk["marker"] if mk else None,
                                      "marker_kind": mk["kind"] if mk else None,
                                      "is_subheading": bool(b["size"] > B.bs + 0.5),
                                      "text": t, "pages": [b["page"]]})
        else:
            p = sec["paragraphs"][-1]
            p["text"] += " " + t
            if b["page"] not in p["pages"]:
                p["pages"].append(b["page"])
        prev = b
        i += 1

    # annexes placed after the last article are document-level (not part of the last part/chapter)
    last_art = max((k for k, s in enumerate(doc["sections"]) if s["type"] == "article"), default=-1)
    for s in doc["sections"][last_art + 1:]:
        if s["type"] == "annex":
            s["part"] = s["chapter"] = None

    # ---- paragraph levels & paths, footnotes, binding status
    fn = clean["footnotes"]
    for s in doc["sections"]:
        assign_levels(s)
        s["footnotes"] = []
        for p in s["paragraphs"]:
            p["text"] = re.sub(r"\s+", " ", p["text"]).strip()
            p["footnotes"] = FOOT.findall(p["text"])
            for k in p["footnotes"]:
                if k in fn:
                    s["footnotes"].append({"n": int(k), "scope": "paragraph", "paragraph_path": p["path"],
                                           "text": fn[k]["text"]})
                    if "فقرة استرشادية" in fn[k]["text"] or ("استرشادي" in fn[k]["text"] and "إلزامي" not in fn[k]["text"]):
                        p["binding_status"] = "guidance"
                    elif "إلزامية" in fn[k]["text"]:
                        p["binding_status"] = "mandatory"
        for k in s["heading_footnotes"]:
            if k in fn:
                s["footnotes"].append({"n": int(k), "scope": "section", "text": fn[k]["text"]})
        s["binding_status"] = "mandatory"
        for f in s["footnotes"]:
            if f["scope"] == "section":
                if "إلزامية" in f["text"]:
                    s["binding_status"] = "mandatory"
                elif "استرشادي" in f["text"]:
                    s["binding_status"] = "guidance"
        s["pages"] = sorted(set(s["pages"] + [pg for p in s["paragraphs"] for pg in p["pages"]]))
    return doc


def assign_levels(sec):
    """letter markers → level 1, numbers → level 2 (or 1 if the section has no letters), bullets → deeper."""
    kinds = [p["marker_kind"] for p in sec["paragraphs"]]
    has_letters = "letter" in kinds or "ordinal_word" in kinds
    path = {}
    for p in sec["paragraphs"]:
        k = p["marker_kind"]
        if k in ("letter", "ordinal_word"):
            lvl = 1
        elif k == "number":
            lvl = 2 if has_letters else 1
        elif k == "bullet":
            lvl = (2 if has_letters else 1) + 1
        else:
            lvl = 0
        p["level"] = lvl
        if lvl == 0:
            p["path"] = ""
            continue
        path[lvl] = p["marker"]
        for d in list(path):
            if d > lvl:
                del path[d]
        p["path"] = "".join(f"({path[d]})" for d in sorted(path) if path[d] != "•")


# ---------------------------------------------------------------- validation against the TOC
def validate(doc, clean):
    toc_arts = []
    for t in clean["toc"]:
        m = RE_ART.match(t)
        if m and m.group(1):
            toc_arts.append((ordinal(m.group(2)), m.group(3).strip()))
    got = {s["number"]: s["title"] for s in doc["sections"] if s["type"] == "article"}
    missing = [n for n, _ in toc_arts if n not in got]
    mismatched = []
    for n, title in toc_arts:
        if n in got:
            r = SequenceMatcher(None, title, got[n]).ratio()
            if r < 0.85:
                mismatched.append({"article": n, "toc": title, "extracted": got[n], "similarity": round(r, 2)})
    nums = [s["number"] for s in doc["sections"] if s["type"] == "article"]
    seq_ok = nums == list(range(1, len(nums) + 1))
    toc_annex = [t for t in clean["toc"] if RE_ANNEX.match(t)]
    return {"toc_articles": len(toc_arts), "extracted_articles": len(nums),
            "article_sequence_continuous": seq_ok, "missing_from_body": missing,
            "title_mismatches": mismatched,
            "toc_annexes": len(toc_annex),
            "extracted_annexes": sum(s["type"] == "annex" for s in doc["sections"]),
            "repaired_headings": [s["number"] for s in doc["sections"] if s.get("repaired_heading")]}


def render_md(doc):
    out = [f"# {doc['title']}", ""]
    lp = lc = None
    for s in doc["sections"]:
        if s["part"] and s["part"] != lp:
            out += [f"## {s['part']['label']}: {s['part']['title'] or ''}", ""]; lp = s["part"]; lc = None
        if s["chapter"] and s["chapter"] != lc:
            out += [f"### {s['chapter']['label']}: {s['chapter']['title'] or ''}", ""]; lc = s["chapter"]
        tag = " — *استرشادية*" if s["binding_status"] == "guidance" else ""
        if s["type"] != "preamble":
            out += [f"#### {s['label']}: {s['title'] or ''}{tag}  _(ص {s['pages'][0]}–{s['pages'][-1]})_", ""]
        for p in s["paragraphs"]:
            ind = "  " * max(0, p["level"] - 1)
            g = " *(استرشادية)*" if p.get("binding_status") == "guidance" else ""
            out.append(f"{ind}- {p['text']}{g}" if p["level"] else p["text"])
        for f in s["footnotes"]:
            out.append(f"> [^{f['n']}] {f['text']}")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for f in sorted(IN.glob("*.json")):
        if f.name == "noise_report.json":
            continue
        clean = json.loads(f.read_text(encoding="utf-8"))
        doc = build(clean)
        (OUT / f.name).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        (OUT / (f.stem + ".md")).write_text(render_md(doc), encoding="utf-8")
        v = validate(doc, clean)
        report[doc["doc_id"]] = v
        print(doc["doc_id"], json.dumps(v, ensure_ascii=False)[:600])
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
