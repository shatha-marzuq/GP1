"""
Step 3 — Extract regulatory text from CMA Arabic PDFs.

Why a custom extractor?
  The CMA PDFs use Arabic fonts whose ligature glyphs (لم، لا، في، يج ...) are mapped
  to multi-character strings in *logical* order, while the page itself is drawn in
  *visual* (left-to-right) order. Standard tools (pdftotext, PyMuPDF) reverse each line
  to get RTL text and in doing so also reverse the ligature strings, producing
  corrupted words such as "املالية" (instead of "المالية") or "جملس" (instead of "مجلس").

  Here we work at glyph level with pdfminer:
    1. group glyphs into visual lines by baseline,
    2. sort each line left→right and reverse the *glyph sequence* (not the characters
       inside a ligature), which yields correct logical Arabic,
    3. re-reverse embedded LTR runs (numbers, dates, Latin, URLs).

Output: output/01_extracted/<doc_id>.json  — pages → lines with layout metadata.
"""
import json, re, sys, unicodedata
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar, LAParams

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
OUT = ROOT / "output" / "01_extracted"

# LTR runs inside an RTL line (approximation of the Unicode bidi algorithm):
#  - Latin runs may contain spaces/punctuation between Latin words (neutral between L and L → L)
#  - digit runs bind only through CS/ES separators (. , : / - + %) — spaces and en-dashes between
#    numbers resolve to RTL, so "100,001–300,000" keeps RTL order between the numbers.
LTR_RUN = re.compile(r"[A-Za-z](?:[A-Za-z0-9./\-:_@%,+&' ]*[A-Za-z0-9])?|[0-9](?:[0-9./\-:,+%]*[0-9%])?")
ARABIC = re.compile(r"[؀-ۿ]")


def walk(o):
    if isinstance(o, LTChar):
        yield o
    elif hasattr(o, "__iter__"):
        for x in o:
            yield from walk(x)


def is_mark(t):
    return all(unicodedata.category(c) == "Mn" for c in t)


class G:
    """Lightweight glyph record."""
    __slots__ = ("text", "x0", "x1", "y0", "y1", "size", "font", "idx", "sup")

    def __init__(self, c, i):
        self.text, self.x0, self.x1, self.y0, self.y1 = c.get_text(), c.x0, c.x1, c.y0, c.y1
        self.size, self.font, self.idx, self.sup = c.size, c.fontname, i, False

    @property
    def yc(self):
        return (self.y0 + self.y1) / 2


def cluster(glyphs):
    glyphs = sorted(glyphs, key=lambda g: -g.yc)
    lines = []
    for g in glyphs:
        tol = max(2.0, 0.35 * g.size)
        if lines and abs(lines[-1]["yc"] - g.yc) <= tol:
            L = lines[-1]
            L["g"].append(g)
            L["yc"] = (L["yc"] * (len(L["g"]) - 1) + g.yc) / len(L["g"])
        else:
            lines.append({"yc": g.yc, "g": [g]})
    for L in lines:
        L["y0"] = min(g.y0 for g in L["g"]); L["y1"] = max(g.y1 for g in L["g"])
        sz = [g.size for g in L["g"] if g.text.strip()]
        L["size"] = max(set(sz), key=sz.count) if sz else 0
    return lines


def build_lines(chars):
    """Cluster glyphs into lines by vertical position (top → bottom).
    Superscript footnote markers (small digits raised above a line) are re-attached to
    their host line and rendered as [^n]."""
    glyphs = [G(c, i) for i, c in enumerate(chars) if c.get_text() and c.size >= 4]
    lines = cluster(glyphs)
    # detect lines made only of small digits that sit on/just above a larger line
    final = []
    for k, L in enumerate(lines):
        txt = "".join(g.text for g in L["g"]).strip()
        if txt.isdigit() and len(txt) <= 3:
            host = None
            for M in lines:
                if M is L or not M["size"]:
                    continue
                if L["size"] < 0.8 * M["size"] and M["y0"] - 1 <= L["yc"] <= M["y1"] + 0.4 * M["size"]:
                    host = M
                    break
            if host is not None:
                for g in L["g"]:
                    g.sup = True
                host["g"].extend(L["g"])
                continue
        final.append(L)
    out = []
    for L in final:
        # in-line superscripts: small, raised digits inside a normal line
        body = [g for g in L["g"] if g.text.strip() and not g.sup and g.size >= 0.8 * L["size"]]
        if body:
            base = sorted(g.y0 for g in body)[len(body) // 2]
            for g in L["g"]:
                if (not g.sup and g.text.strip().isdigit() and g.size < 0.8 * L["size"]
                        and g.y0 > base + 0.15 * L["size"]):
                    g.sup = True
        L["g"] = [g for g in L["g"] if not (g.sup and not g.text.strip())]
        # stacked "vertical" table-header letters are drawn as tiny glyphs (< 7.8pt): drop them
        L["g"] = [g for g in L["g"] if g.sup or g.size >= 7.8]
        if not L["g"]:
            continue
        gs = sorted(L["g"], key=lambda g: (round(g.x0, 1), g.idx))
        out.append(line_to_text(gs))
    return [l for l in out if l and l["text"].strip()]


def line_to_text(cs):
    # merge consecutive superscript digits into one token "[^n]"
    merged = []
    for c in cs:
        if c.sup and merged and merged[-1].sup:
            merged[-1].text += c.text; merged[-1].x1 = c.x1
        else:
            merged.append(c)
    cs = merged
    for c in cs:
        if c.sup:
            c.text = " [^" + c.text.strip() + "] "
    # visual (left→right) glyph sequence, inserting spaces at visual gaps
    seq = []
    prev = None
    for c in cs:
        t = c.text
        if prev is not None and not is_mark(t):
            gap = c.x0 - prev.x1
            if gap > 0.22 * c.size and t != " " and prev.text != " ":
                seq.append(" ")
        seq.append(t)
        if not is_mark(t):
            prev = c
    rtl = any(ARABIC.search(s) for s in seq)
    if rtl:
        # reverse glyph order (keeps multi-char ligature strings intact); combining marks
        # precede their base glyph in the visual stream, so swap mark/base after reversal
        rev = []
        for t in reversed(seq):
            if rev and is_mark(rev[-1]) and not is_mark(t):
                m = rev.pop()
                rev.append(t)
                rev.append(m)
            else:
                rev.append(t)
        # protect footnote tokens from LTR-run reversal
        rev = [("\x00" + t.strip()[2:-1] + "\x01") if t.startswith(" [^") else t for t in rev]
        text = "".join(rev)
        text = LTR_RUN.sub(lambda m: m.group(0)[::-1], text)
        text = re.sub("\x00(\\d+)\x01", lambda m: " [^" + m.group(1)[::-1] + "] ", text)
    else:
        text = "".join(seq)
    body = [c for c in cs if c.text.strip() and not c.sup]
    sizes = [round(c.size, 1) for c in body]
    fonts = [c.font for c in body]
    size = max(set(sizes), key=sizes.count) if sizes else 0
    bold = sum("Bold" in f for f in fonts) > len(fonts) / 2 if fonts else False
    return {
        "text": re.sub(r"[ \t]+", " ", text).strip(),
        "x0": round(min(c.x0 for c in cs), 1),
        "x1": round(max(c.x1 for c in cs), 1),
        "y": round(max(c.y1 for c in cs), 1),
        "size": size,
        "bold": bold,
    }


def extract(pdf_path: Path):
    pages = []
    for pno, page in enumerate(extract_pages(str(pdf_path), laparams=LAParams()), start=1):
        lines = build_lines(list(walk(page)))
        pages.append({"page": pno, "width": page.width, "height": page.height, "lines": lines})
    return pages


DOCS = {
    "investment_funds_regulations": "لائحة صناديق الاستثمار",
    "capital_market_institutions_regulations": "لائحة مؤسسات السوق المالية",
    "corporate_governance_regulations": "لائحة حوكمة الشركات",
    "simplified_investment_funds_instructions": "تعليمات صناديق الاستثمار المبسطة",
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] or list(DOCS)
    for doc_id in targets:
        pages = extract(RAW / f"{doc_id}.pdf")
        (OUT / f"{doc_id}.json").write_text(
            json.dumps({"doc_id": doc_id, "title": DOCS[doc_id], "pages": pages}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"{doc_id}: {len(pages)} pages, {sum(len(p['lines']) for p in pages)} lines")
