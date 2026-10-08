
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

from .config import CONFIG
from .errors import DocProcError
from .pipeline import process_document
from .quality import cer, wer

def evaluate_folder(folder, cfg=CONFIG) -> list[dict]:
    rows = []
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in (".pdf", ".docx"):
            continue
        gt = f.with_suffix(".txt")
        if not gt.exists():
            continue
        row = {"file": f.name}
        try:
            res = process_document(f, cfg=cfg)
            ref = gt.read_text(encoding="utf-8")
            row.update(cer=round(cer(ref, res.text), 4), wer=round(wer(ref, res.text), 4),
                       source=res.source_type, ocr_pages=res.stats["ocr_pages"],
                       avg_quality=res.stats["avg_quality"], needs_review=res.needs_review,
                       seconds=res.stats["seconds"])
        except DocProcError as e:
            row.update(error=e.code)
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--engine", default="auto", choices=["auto", "surya", "tesseract", "paddle", "easyocr", "none"])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rows = evaluate_folder(a.folder, dataclasses.replace(CONFIG, ocr_engine=a.engine))
    print(f"{'file':32} {'CER':>7} {'WER':>7} {'OCR pg':>6} {'review':>7}")
    for r in rows:
        if "error" in r:
            print(f"{r['file']:32} ERROR {r['error']}")
        else:
            print(f"{r['file']:32} {r['cer']:7.2%} {r['wer']:7.2%} {r['ocr_pages']:6} {str(r['needs_review']):>7}")
    ok = [r for r in rows if "error" not in r]
    if ok:
        print(f"{'AVERAGE':32} {sum(r['cer'] for r in ok)/len(ok):7.2%} {sum(r['wer'] for r in ok)/len(ok):7.2%}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
