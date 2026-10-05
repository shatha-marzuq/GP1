"""
Steps 4 + 5 — Clean / normalize the extracted text and remove noise.

Input : output/01_extracted/<doc_id>.json   (lines with layout metadata)
Output: output/02_clean/<doc_id>.json        (body lines + footnotes + TOC, noise removed)
        output/02_clean/<doc_id>.txt         (human-readable clean text)
        output/02_clean/noise_report.json    (what was removed, per doc and category)

Noise removed
  • cover page and the "ملحوظة مهمة" disclaimer
  • table of contents (kept separately as metadata to validate the structure in step 6)
  • page numbers, "Internal - داخلي" classification labels
  • rotated / vertical table-header glyph fragments (tiny or broken glyphs)
  • empty template rows in annex tables ("-1", "-2", "المجموع" …)
  • footnotes are *separated* from the body (not deleted) and linked to their markers

Normalization
  • Unicode NFC, remove bidi/zero-width controls, NBSP → space
  • remove Arabic diacritics (tashkeel) — in the PDFs they are detached glyphs that land at
    the wrong position (e.g. "دائما ً"), and they carry no regulatory meaning here
  • remove decorative tatweel (keep the Hijri abbreviation "هـ" and list markers like "جـ)")
  • whitespace / punctuation spacing, digit-separator spacing ("28/ 8/ 1439" → "28/8/1439")
  • fix reversed numeric ranges produced by LTR bullet cells ("300,000–100,001" → "100,001–300,000")
"""
import json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "output" / "01_extracted"
OUT = ROOT / "output" / "02_clean"

TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
CONTROL = re.compile(r"[​-‏‪-‮⁦-⁩﻿­]")
FOOT_MARK = re.compile(r"\[\^(\d+)\]")


def normalize(t: str) -> str:
    t = unicodedata.normalize("NFC", t)
    t = CONTROL.sub("", t).replace(" ", " ")
    t = TASHKEEL.sub("", t)
    t = re.sub(r"(?<=[ء-ي])ـ+(?=[ء-ي])", "", t)  # decorative tatweel inside words
    t = re.sub(r"هـ{2,}", "هـ", t)
    # spacing
    t = re.sub(r"[ \t]+", " ", t).strip()
    t = re.sub(r"\s+([.,،؛:؟!)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    t = re.sub(r"(?<=\d)\s*/\s*(?=\d)", "/", t)
    t = re.sub(r"\(\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\)", r"(\1-\2-\3)", t)
    t = re.sub(r"\s*\[\^(\d+)\]\s*", r" [^\1] ", t).strip()
    t = re.sub(r"\s+([.,،؛:])", r"\1", t)
    # LTR bullet cells: "300,000–100,001 •" → "• 100,001–300,000"
    if "•" in t:
        def fix(m):
            a, b = m.group(1), m.group(2)
            return f"{b}–{a}" if float(a.replace(",", "")) > float(b.replace(",", "")) else m.group(0)
        t = re.sub(r"([\d,]+)\s*–\s*([\d,]+)", fix, t)
        if t.endswith("•"):
            t = "• " + t[:-1].strip()
    # unmirrored brackets around embedded Latin terms: "التتبع)Tracking Error(" → "التتبع (Tracking Error)"
    t = re.sub(r"\)([A-Za-z][A-Za-z0-9 /.&\-]*)\(", r" (\1) ", t)
    t = re.sub(r"\s+([.,،؛:)])", r"\1", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()


def is_fragment_noise(line, body_size):
    """Rotated table headers come out as tiny, broken glyph sequences."""
    txt = line["text"]
    if line["size"] and line["size"] < 7:
        return True
    toks = txt.split()
    ar = [w for w in toks if re.search(r"[ء-ي]", w)]
    if len(ar) >= 2 and sum(len(w) for w in ar) / len(ar) < 2.2:
        return True
    return False


EMPTY_ROW = re.compile(r"^(-?\d{1,2}-?|المجموع|\(\.\.\.\)|\.{3,})$")
INTERNAL = re.compile(r"Internal\s*-\s*داخ\s*ل?ي?|داخلي\s*-\s*Internal", re.I)
FIRST_HEADING = re.compile(r"^(الباب|الفصل) الأول$")


# --- kashida-justification artifact repair ---------------------------------------------
# A few justified lines in one PDF draw the stretched kashida with glyphs that map to "س",
# giving e.g. "م سسسسستحقة" / "ب شسسسسكل". We repair them against the corpus vocabulary.
ARTIFACT = re.compile(r"س{3,}|شسس|سشس")


def _candidates(w):
    runs = re.split(r"(س+)", w)
    outs = [""]
    for part in runs:
        if part.startswith("س"):
            outs = [o + "س" * k for o in outs for k in (0, 1, 2)]
        else:
            outs = [o + part for o in outs]
    return set(outs)


def repair_kashida(text, vocab):
    if not ARTIFACT.search(text):
        return text, 0
    toks = text.split(" ")
    out, i, fixed = [], 0, 0
    while i < len(toks):
        done = False
        for n in (3, 2, 1):
            win = toks[i:i + n]
            if len(win) < n or not any("سس" in t or t == "س" for t in win):
                continue
            joined = "".join(win)
            m = re.match(r"^([^\u0621-\u064A]*)(.*?)([^\u0621-\u064A]*)$", joined)
            pre, core, post = m.groups()
            if n == 1 and core in vocab:
                continue
            best = max(_candidates(core), key=lambda c: vocab.get(c, 0))
            if vocab.get(best, 0) > 0 and best != core or (n > 1 and vocab.get(best, 0) > 0):
                out.append(pre + best + post); i += n; fixed += 1; done = True
                break
        if not done and i + 1 < len(toks) and re.match(r"^س{3,}", toks[i + 1]):
            # fallback: word split by a stretched kashida not found in vocabulary
            out.append(re.sub(r"س{3,}", "س", toks[i] + toks[i + 1])); i += 2; fixed += 1; done = True
        if not done:
            out.append(toks[i]); i += 1
    return " ".join(out), fixed


def clean_doc(doc):
    pages = doc["pages"]
    sizes = Counter()
    for p in pages:
        for l in p["lines"]:
            sizes[l["size"]] += len(l["text"])
    body_size = sizes.most_common(1)[0][0]
    noise = Counter()

    # --- locate TOC and body start
    toc_start = next(p["page"] for p in pages if any(l["text"].strip() == "المحتويات" for l in p["lines"]))
    body_start = next(p["page"] for p in pages
                      if p["page"] > toc_start and p["lines"]
                      and FIRST_HEADING.match(normalize(p["lines"][0]["text"])))
    toc, body, footnotes = [], [], {}
    for p in pages:
        pno = p["page"]
        lines = [dict(l) for l in p["lines"]]
        # page number = last digit-only line on the page
        for i in range(len(lines) - 1, -1, -1):
            if lines[i]["text"].strip().isdigit():
                lines.pop(i); noise["page_number"] += 1
                break
        kept = []
        for l in lines:
            if INTERNAL.search(l["text"]):
                l["text"] = INTERNAL.sub("", l["text"]).strip()
                noise["internal_label"] += 1
                if not l["text"]:
                    continue
            kept.append(l)
        lines = kept
        if pno < toc_start:
            noise["cover_page_lines"] += len(lines); continue
        if pno < body_start:
            for l in lines:
                t = normalize(l["text"])
                if t and t != "المحتويات" and not t.isdigit():
                    toc.append(t)
                noise["toc_lines"] += 1
            continue
        # --- footnotes: small-font lines at the bottom that start with a marker referenced on this page
        refs = set()
        for l in lines:
            if l["size"] >= body_size - 1:
                refs.update(FOOT_MARK.findall(l["text"]))
        cur = None
        for l in lines:
            t = l["text"].strip()
            small = l["size"] and l["size"] <= body_size - 3
            m = re.match(r"^(?:\[\^(\d+)\]|(\d+))\s+(.*)$", t)
            if small and m and (m.group(1) or m.group(2)) in refs:
                cur = m.group(1) or m.group(2)
                footnotes[cur] = {"n": int(cur), "page": pno, "text": normalize(m.group(3))}
                l["_foot"] = True; noise["footnote_lines_separated"] += 1
                continue
            if cur and small:
                footnotes[cur]["text"] = normalize(footnotes[cur]["text"] + " " + t)
                l["_foot"] = True; noise["footnote_lines_separated"] += 1
                continue
        for l in lines:
            if l.get("_foot"):
                continue
            if is_fragment_noise(l, body_size):
                noise["rotated_table_fragments"] += 1; continue
            t = normalize(l["text"])
            if not t:
                continue
            if EMPTY_ROW.match(t):
                noise["empty_table_rows"] += 1; continue
            body.append({"page": pno, "text": t, "size": l["size"], "x0": l["x0"], "x1": l["x1"],
                         "width": p["width"]})
    return {"doc_id": doc["doc_id"], "title": doc["title"], "body_font_size": body_size,
            "toc": toc, "body": body, "footnotes": footnotes, "noise": dict(noise),
            "body_start_page": body_start, "pages": len(pages)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    cleaned = [clean_doc(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(IN.glob("*.json"))]
    vocab = Counter()
    for c in cleaned:
        for b in c["body"]:
            if not ARTIFACT.search(b["text"]):
                vocab.update(re.findall(r"[\u0621-\u064A]+", b["text"]))
    for c in cleaned:
        for b in c["body"]:
            b["text"], n = repair_kashida(b["text"], vocab)
            if n:
                c["noise"]["kashida_artifacts_repaired"] = c["noise"].get("kashida_artifacts_repaired", 0) + n
        f = IN / (c["doc_id"] + ".json")
        (OUT / f.name).write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(OUT / (f.stem + ".txt"), "w", encoding="utf-8") as o:
            last = None
            for b in c["body"]:
                if b["page"] != last:
                    o.write(f"\n<<صفحة {b['page']}>>\n"); last = b["page"]
                o.write(b["text"] + "\n")
            if c["footnotes"]:
                o.write("\n<<الحواشي>>\n")
                for k, v in sorted(c["footnotes"].items(), key=lambda kv: int(kv[0])):
                    o.write(f"[^{k}] {v['text']}\n")
        report[c["doc_id"]] = {"pages": c["pages"], "body_start_page": c["body_start_page"],
                               "body_lines": len(c["body"]), "toc_entries": len(c["toc"]),
                               "footnotes": len(c["footnotes"]), "removed": c["noise"]}
        print(c["doc_id"], report[c["doc_id"]])
    (OUT / "noise_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
