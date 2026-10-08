"""CLI

    python -m mithaq_docproc file.pdf --out-dir out/            # يكتب file.json + file.txt + file.md
    python -m mithaq_docproc file.pdf -o out.txt [--markers]     # نص فقط
    python -m mithaq_docproc file.pdf --json -o out.json          # JSON الكامل (schema 1.0)
    خيارات:  --force-ocr   --engine auto|tesseract|paddle|easyocr|none
"""
import argparse
import dataclasses
import json
import sys

from . import CONFIG, DocProcError, process_document
from .output import build_output, write_outputs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mithaq_docproc", description="Document -> Clean Text")
    ap.add_argument("file")
    ap.add_argument("--out-dir", help="مجلد الإخراج: يكتب JSON + TXT + MD")
    ap.add_argument("-o", "--output", help="ملف إخراج واحد (الافتراضي: الشاشة)")
    ap.add_argument("--json", action="store_true", help="إخراج JSON الكامل (schema 1.0)")
    ap.add_argument("--markers", action="store_true", help="إضافة علامات <<PAGE n>>")
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument("--engine", choices=["auto", "surya", "tesseract", "paddle", "easyocr", "none"], default="auto")
    a = ap.parse_args(argv)

    cfg = dataclasses.replace(CONFIG, ocr_engine=a.engine)
    try:
        res = process_document(a.file, cfg=cfg, force_ocr=a.force_ocr,
                               progress=lambda i, n: print(f"  page {i}/{n}", file=sys.stderr))
    except DocProcError as e:
        print(f"[{e.code}] {e.message_ar} {e.detail}", file=sys.stderr)
        return 1

    if a.out_dir:
        for kind, p in write_outputs(res, a.out_dir, a.file).items():
            print(f"{kind}: {p}", file=sys.stderr)
    else:
        out = json.dumps(build_output(res, a.file), ensure_ascii=False, indent=2) if a.json else (
            res.text_with_page_markers() if a.markers else res.text)
        if a.output:
            with open(a.output, "w", encoding="utf-8", newline="\n") as f:
                f.write(out)
        else:
            print(out)
    print(f"\n{res.page_count} pages | {res.stats}", file=sys.stderr)
    if res.needs_review:
        print(f"تحذيرات تحتاج مراجعة: {res.warnings}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
