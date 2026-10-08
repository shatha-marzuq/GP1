"""الناتج النهائي للموديول: Document -> Clean Text (JSON + TXT + Markdown).

الـ JSON هو العقد مع باقي الوكلاء (Orchestrator / Segmentation / Retrieval):
    document : معلومات الملف + هل يحتاج مراجعة + التحذيرات (بالعربي)
    pages    : لكل صفحة: الطريقة (native/ocr) + الجودة + النص
    items    : كل عنصر بالترتيب: heading | paragraph | list_item | table
               (id ثابت، رقم صفحة، نص وفيّ للأصل للاقتباس، نص مبسّط للبحث، علامة البند، والجداول كصفوف)
    text     : النص الكامل النظيف
الوصف الكامل في output_schema.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .errors import WARNING_MESSAGES_AR

SCHEMA_VERSION = "1.0"


def build_output(res, source_path: str | Path | None = None) -> dict:
    sha = None
    if source_path and Path(source_path).is_file():
        sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "document": {
            "filename": res.filename,
            "source_type": res.source_type,
            "sha256": sha,
            "page_count": res.page_count,
            "processed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "needs_review": res.needs_review,
            "warnings": [{"code": w, "message_ar": WARNING_MESSAGES_AR.get(w, w)} for w in res.warnings],
        },
        "stats": res.stats,
        "pages": [{
            "page": p.page, "method": p.method, "ocr_confidence": p.confidence, "quality": p.quality,
            "warnings": p.warnings, "item_ids": [it["id"] for it in p.items], "text": p.text,
        } for p in res.pages],
        "items": res.items,
        "ocr_corrections": getattr(res, "corrections", []),
        "text": res.text,
    }


def to_markdown(res) -> str:
    """نسخة للقراءة والمراجعة البشرية: عناوين، قوائم، وجداول Markdown."""
    out = [f"# {res.filename}", ""]
    if res.warnings:
        out += ["> **تحذيرات:** " + "، ".join(WARNING_MESSAGES_AR.get(w, w) for w in res.warnings), ""]
    for p in res.pages:
        out += [f"<!-- PAGE {p.page} | {p.method} -->", ""]
        for it in p.items:
            if it["type"] == "table":
                tb = it["table"]
                n = tb["n_cols"]
                rows = ([tb["header"]] if tb["header"] else [[""] * n]) + tb["rows"]
                rows = [r + [""] * (n - len(r)) for r in rows]
                esc = [[c.replace("|", "\\|").replace("\n", " ") for c in r] for r in rows]
                if tb.get("continues"):
                    out.append(f"*(تكملة الجدول {tb['continues']})*")
                out.append("| " + " | ".join(esc[0]) + " |")
                out.append("|" + "---|" * n)
                out += ["| " + " | ".join(r) + " |" for r in esc[1:]]
            elif it["type"] == "heading":
                level = min((it.get("level") or 1) + 1, 6)
                out.append("#" * level + " " + it["text"])
            elif it["type"] == "list_item":
                out.append("- " + it["text"])
            else:
                out.append(it["text"])
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_outputs(res, out_dir: str | Path, source_path: str | Path | None = None,
                  formats: tuple[str, ...] = ("json", "txt", "md")) -> dict[str, Path]:
    """يكتب name.json / name.txt (مع <<PAGE n>>) / name.md داخل out_dir. يرجع المسارات."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(res.filename).stem or "document"
    paths: dict[str, Path] = {}
    if "json" in formats:
        paths["json"] = out_dir / f"{stem}.json"
        paths["json"].write_text(json.dumps(build_output(res, source_path), ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    if "txt" in formats:
        paths["txt"] = out_dir / f"{stem}.txt"
        with open(paths["txt"], "w", encoding="utf-8", newline="\n") as f:   # \n حتى على Windows
            f.write(res.text_with_page_markers() + "\n")
    if "md" in formats:
        paths["md"] = out_dir / f"{stem}.md"
        with open(paths["md"], "w", encoding="utf-8", newline="\n") as f:
            f.write(to_markdown(res))
    return paths
