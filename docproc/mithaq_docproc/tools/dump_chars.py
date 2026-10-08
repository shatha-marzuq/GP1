"""تشخيص: يطبع الحروف الخام من PyMuPDF حول كلمة معينة (لمعرفة كيف يفكك رموز الـ ligature).

    python -m mithaq_docproc.tools.dump_chars testset/test.pdf 9 "ير"
    python -m mithaq_docproc.tools.dump_chars testset/test.pdf 23 "750" 14   (14 = حروف السياق)
النتيجة تنكتب في dump_<page>.txt أيضاً (UTF-8) عشان تقدر ترفعها.
"""
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

path, page_no, needle = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 6          # عدد الحروف قبل/بعد الكلمة
page = fitz.open(path)[page_no - 1]
chars = []
for bi, b in enumerate(page.get_text("rawdict")["blocks"]):
    for li, l in enumerate(b.get("lines", [])):
        for s in l["spans"]:
            for c in s["chars"]:
                chars.append((c["c"], c["bbox"], s.get("font", ""), bi, li))
stream = "".join(c for c, _, _, _, _ in chars)
# نبحث بالاتجاهين لأن الترتيب داخل الملف قد يكون معكوساً
hits = [i for i in range(len(stream)) if stream.startswith(needle, i) or stream.startswith(needle[::-1], i)]
lines = [f"{path} page {page_no}: {len(chars)} chars, {len(hits)} hits for {needle!r}"]
for h in hits[:3]:
    lines.append("-" * 60)
    for c, bb, font, bi, li in chars[max(0, h - ctx): h + len(needle) + ctx]:
        lines.append(f"{' '.join(f'U+{ord(x):04X}' for x in c):14}  x0={bb[0]:8.2f} x1={bb[2]:8.2f} w={bb[2]-bb[0]:6.2f}  "
                     f"y={bb[1]:7.2f}  block={bi:3} line={li:3}  {font}")
out = "\n".join(lines)
print(out)
with open(f"dump_{page_no}.txt", "a", encoding="utf-8") as f:
    f.write(out + "\n\n")
