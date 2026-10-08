"""تشخيص جداول الصفحات الممسوحة: يطبع شبكة الجدول، ولكل خلية فاضية يجرّب عدة طرق قراءة ويطبع نتيجتها.

    python -m mithaq_docproc.tools.debug_scan testset/test6.pdf 1
النتيجة تُطبع وتُكتب في scan_<page>.txt (UTF-8).
"""
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from ..config import CONFIG
from ..extractors import render_page
from ..ocr import get_engine, preprocess
from ..scan_tables import detect_tables

path, page_no = sys.argv[1], int(sys.argv[2])
page = fitz.open(path)[page_no - 1]
img = render_page(page, CONFIG.ocr_dpi, CONFIG.ocr_max_side_px)
eng = get_engine(CONFIG)
out = [f"{path} page {page_no}: image {img.size}, engine={eng.name if eng else None}, "
       f"lang={getattr(eng, 'lang', '?')}, have={sorted(getattr(eng, '_have', []))}"]
tables = detect_tables(img.convert("L"))
out.append(f"tables: {len(tables)}")
for t in tables:
    out.append(f"  bbox={t.bbox} cols={t.cols} rows={len(t.rows)}")
    ty0, ty1 = t.bbox[1], t.bbox[3]
    for ci, (cx0, cx1) in enumerate(t.cols):
        lines = eng.recognize(preprocess(img.crop((cx0 + 4, ty0, cx1 - 4, ty1)), eng.name))
        out.append(f"  --- column {ci} ({cx0}-{cx1}): {len(lines)} lines")
        for ln in lines:
            yc = ty0 + (ln.bbox[1] + ln.bbox[3]) / 2
            row = next((r for r, (a, b) in enumerate(t.rows) if a <= yc < b), None)
            out.append(f"      row={row} y={yc:6.0f} conf={ln.conf:.2f} {ln.text[:70]!r}")
        for r, (ry0, ry1) in enumerate(t.rows):
            has = any(t.rows[r][0] <= ty0 + (ln.bbox[1] + ln.bbox[3]) / 2 < t.rows[r][1] for ln in lines)
            if has:
                continue
            box = (cx0 + 6, ry0 + 4, cx1 - 6, ry1 - 4)
            cell = img.crop(box)
            out.append(f"      EMPTY cell row={r} box={box}:")
            for psm in (6, 7, 11, 3):
                try:
                    res = eng.recognize(preprocess(cell, eng.name), psm=psm)
                except Exception as e:
                    out.append(f"         psm={psm} ERROR {e}")
                    continue
                out.append(f"         psm={psm}: " + " || ".join(f"{l.text[:40]!r}({l.conf:.2f})" for l in res))
            d = eng._pt.image_to_data(cell.convert("L"), lang=eng.lang, config="--oem 1 --psm 6",
                                      output_type=eng._pt.Output.DICT)
            raw = [(d["text"][i], d["conf"][i]) for i in range(len(d["text"])) if (d["text"][i] or "").strip()]
            out.append(f"         raw psm6 words: {raw[:15]}")
res = "\n".join(out)
print(res)
with open(f"scan_{page_no}.txt", "w", encoding="utf-8") as f:
    f.write(res + "\n")
