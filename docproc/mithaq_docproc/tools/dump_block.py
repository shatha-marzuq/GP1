"""تشخيص عميق: يأخذ block كامل (كل حروفه) ويشغّله عبر خط أنابيبنا الحقيقي (bidi.py) ويطبع:
  1) النص الناتج فعلياً من glyphs_to_lines
  2) كل مسافة في الـ block: موقعها، ولماذا انحذفت أو بقيت (حسب كل قاعدة في bidi.py)

    python -m mithaq_docproc.tools.dump_block testset/test4.pdf 19 29
                                                الملف        الصفحة  رقم الـblock (من dump_chars)
النتيجة تُطبع وتُكتب في block_<page>_<block>.txt (UTF-8).
"""
import sys

try:
    import pymupdf as fitz
except ImportError:
    import fitz

from ..bidi import (Glyph, attach_marks, drop_phantom_spaces, glyphs_to_lines,
                    group_lines, merge_zero_width, order_line)

path, page_no, block_no = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
page = fitz.open(path)[page_no - 1]

out = []
blocks = page.get_text("rawdict")["blocks"]
b = blocks[block_no]
glyphs = []
seq = 0
for li, l in enumerate(b.get("lines", [])):
    for s in l["spans"]:
        for c in s["chars"]:
            x0, y0, x1, y1 = c["bbox"]
            glyphs.append(Glyph(c["c"], x0, y0, x1, y1, seq))
            seq += 1
out.append(f"{path} page {page_no} block {block_no}: {len(glyphs)} glyphs, "
           f"{len(b.get('lines', []))} PyMuPDF-lines")
out.append("=" * 70)
out.append("FINAL RESULT (glyphs_to_lines):")
for ln in glyphs_to_lines([Glyph(g.text, g.x0, g.y0, g.x1, g.y1, g.seq) for g in glyphs]):
    out.append("  " + repr(ln))
out.append("=" * 70)

after_merge = merge_zero_width([Glyph(g.text, g.x0, g.y0, g.x1, g.y1, g.seq) for g in glyphs])
out.append(f"after merge_zero_width: {len(after_merge)} glyphs (started with {len(glyphs)})")
lines = group_lines(after_merge)
out.append(f"group_lines() found {len(lines)} geometric line(s)")
for i, ln in enumerate(lines):
    ys = sorted(g.cy for g in ln)
    out.append(f"  line {i}: {len(ln)} glyphs, cy range {ys[0]:.2f}..{ys[-1]:.2f}, "
               f"text={order_line(ln)!r}")

out.append("=" * 70)
out.append("SPACE DIAGNOSTICS (why kept/dropped):")
before = {id(g): g for g in after_merge}
solid = [g for g in after_merge if not g.text.isspace()]
for g in sorted((x for x in after_merge if x.text.isspace()), key=lambda g: g.x0):
    nearby = [b for b in solid if abs(b.cy - g.cy) < max(b.h, g.h) / 2
             and min(b.x1, g.x1 + 5) > max(b.x0, g.x0 - 5)]
    out.append(f"  space x0={g.x0:8.2f} x1={g.x1:8.2f} w={g.x1-g.x0:5.2f} y={g.y0:7.2f}  "
               f"nearby_glyphs={[(round(n.x0,1), n.text) for n in nearby][:6]}")

result = "\n".join(out)
print(result)
with open(f"block_{page_no}_{block_no}.txt", "w", encoding="utf-8") as f:
    f.write(result + "\n")
