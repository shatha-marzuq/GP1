"""تحويل Blocks الصفحة إلى عناصر منظمة (heading / paragraph / list_item / table).

هذا هو "العقد" بيننا وبين وكيل التقسيم (Segmentation) ووكيل الاسترجاع:
كل عنصر له رقم صفحة + نوع + نص وفيّ للأصل + نص مبسّط للبحث + (للجداول) صفوف وأعمدة.
"""
from __future__ import annotations

import re

from .cleaning import clean_text, join_lines, normalize_for_search
from .bidi import has_arabic
from .layout import Block

# علامة البند في بداية الفقرة:  6.2.1  |  11  |  أ.  |  ب)  |  (3)  |  •  |  المادة 5  |  أولاً
_MARKER = re.compile(
    r"""^\s*(
        (?P<num>\d{1,3}(?:\.\d{1,3}){0,4})(?:[\.\)\-:]\s*|\s+)        |
        \((?P<pnum>\d{1,3}|[\u0621-\u064A])\)\s*                      |
        (?P<let>[\u0621-\u064A])\s?[\.\)\-]\s*                         |
        (?P<bul>[\u2022\u25cf\u25aa\u25e6\u25a0\u2023\u2043•●▪◦■❖➢✓\-–])\s*  |
        (?P<art>(?:المادة|مادة|البند|الفقرة|الباب|الفصل)\s+[^\s:]+)\s*[:\-]?\s* |
        (?P<ord>(?:أولاً|أولا|ثانياً|ثانيا|ثالثاً|ثالثا|رابعاً|رابعا|خامساً|خامسا|سادساً|سادسا|سابعاً|سابعا|ثامناً|ثامنا|تاسعاً|تاسعا|عاشراً|عاشرا))\s*[:\-]?\s*
    )""",
    re.X,
)
_TERMINAL = re.compile(r"[.!؟:؛،]\s*$")


def split_marker(text: str) -> tuple[str | None, str]:
    """يرجع (العلامة، النص بدونها). مثال: '6.2.1 منح التمويل' -> ('6.2.1', 'منح التمويل')."""
    m = _MARKER.match(text)
    if not m:
        return None, text
    marker = next((m.group(k) for k in ("num", "pnum", "let", "bul", "art", "ord") if m.group(k)), None)
    rest = text[m.end():].strip()
    if not rest:                                            # رقم وحده بدون نص (غالباً رقم صفحة/جدول)
        return None, text
    return marker, rest


def classify(text: str) -> tuple[str, str | None, int | None]:
    """(النوع، العلامة، المستوى). المستوى = عدد أجزاء الرقم (6.2.1 -> 3)."""
    marker, rest = split_marker(text)
    short = len(text) <= 90 and not re.search(r"[.!؟]\s*$", text)
    if marker and re.fullmatch(r"\d{1,3}(?:\.\d{1,3})*", marker):
        level = marker.count(".") + 1
        sep = text.lstrip()[len(marker):len(marker) + 1]
        if level == 1 and sep in ".)-":                      # "1. حجم الإيرادات" = عنصر قائمة وليس عنوان
            return "list_item", marker, None
        return ("heading" if short else "list_item"), marker, level
    if marker and marker.startswith(("المادة", "مادة", "الباب", "الفصل")):
        return ("heading" if short else "list_item"), marker, 1
    if marker:
        return "list_item", marker, None
    if (len(text) <= 60 and not _TERMINAL.search(text) and not re.search(r"[0-9\u0660-\u0669@/|]", text)
            and len(text.split()) >= 2):
        return "heading", None, None                        # عنوان بدون رقم (مثل: مدير الصندوق)
    return "paragraph", None, None


def _looks_like_header(rows: list[list[str]]) -> bool:
    """الصف الأول عناوين أعمدة؟ (خلايا قصيرة بدون أرقام، والجدول فيه أكثر من صف)."""
    if len(rows) < 2:
        return False
    first = [c for c in rows[0] if c]
    return bool(first) and all(len(c) <= 40 and not re.search(r"[0-9\u0660-\u0669%•●]", c) for c in first)


def _line_h(b: Block) -> float:
    return (b.bbox[3] - b.bbox[1]) / max(1, b.text.count("\n") + 1)


def merge_continuation_blocks(blocks: list[Block]) -> list[Block]:
    """PyMuPDF أحياناً يعطي كل سطر كـ block مستقل، فينفصل السطر الثاني عن بنده.
    ندمج blocks النص المتتالية لو المسافة العمودية بينها مسافة سطر عادية، وآخر سطر في السابق
    ممتد لطرف العمود (فقرة انلفّت). السطر القصير = نهاية فقرة أو عنوان، فلا ندمج بعده.
    (بعدها join_lines يفصل البنود الجديدة حسب علامتها: 6.3.2 ، • ، أ.)"""
    out: list[Block] = []
    last: list[tuple] = []                     # إحداثيات آخر سطر في كل block ناتج
    for b in blocks:
        p = out[-1] if out else None
        if (p is not None and b.kind == "text" and p.kind == "text" and any(b.bbox) and any(p.bbox)):
            h = max(_line_h(p), _line_h(b), 1.0)
            gap = b.bbox[1] - p.bbox[3]
            overlap = min(p.bbox[2], b.bbox[2]) - max(p.bbox[0], b.bbox[0])
            lx0, _, lx1, _ = last[-1]
            col0, col1 = min(p.bbox[0], b.bbox[0]), max(p.bbox[2], b.bbox[2])
            width = max(col1 - col0, 1.0)
            reaches = (lx0 - col0 <= 0.15 * width) if has_arabic(p.text) else (col1 - lx1 <= 0.15 * width)
            if -0.5 * h <= gap <= 0.9 * h and overlap > 0 and reaches:
                out[-1] = Block(p.text + "\n" + b.text,
                                (col0, p.bbox[1], col1, b.bbox[3]),
                                "text", p.conf if p.conf is not None else b.conf)
                last[-1] = b.bbox
                continue
        out.append(b)
        last.append(b.bbox)
    return out


def page_items(blocks: list[Block], page: int | None, method: str) -> list[dict]:
    """Blocks صفحة واحدة -> عناصر منظمة (بدون أرقام تعريف؛ تضاف لاحقاً على مستوى المستند)."""
    items: list[dict] = []
    for b in merge_continuation_blocks(blocks):
        bbox = [round(v, 1) for v in b.bbox] if b.bbox and any(b.bbox) else None
        if b.kind == "table":
            rows = b.rows or [r.split(" | ") for r in b.text.split("\n")]
            rows = [[clean_text(c) for c in r] for r in rows]
            rows = [r for r in rows if any(r)]
            if not rows:
                continue
            text = "\n".join(" | ".join(r) for r in rows)
            has_header = _looks_like_header(rows)
            items.append({
                "type": "table", "page": page, "text": text, "text_norm": normalize_for_search(text),
                "bbox": bbox, "source": method, "confidence": b.conf,
                "table": {"n_rows": len(rows), "n_cols": max(len(r) for r in rows),
                          "header": rows[0] if has_header else None,
                          "rows": rows[1:] if has_header else rows},
            })
            continue
        for para in join_lines(b.text.split("\n")):
            text = clean_text(para)
            if not text:
                continue
            kind, marker, level = classify(text)
            items.append({
                "type": kind, "page": page, "text": text, "text_norm": normalize_for_search(text),
                "marker": marker, "level": level, "bbox": bbox, "source": method,
                "confidence": round(b.conf, 3) if b.conf is not None else None,
            })
    return items


def link_tables_across_pages(items: list[dict]) -> None:
    """جدول يكمل في الصفحة التالية (نفس عدد الأعمدة وأول عنصر في الصفحة) -> نربطه بالجدول السابق."""
    for prev, cur in zip(items, items[1:]):
        if (cur["type"] == "table" and prev["type"] == "table" and cur["page"] and prev["page"]
                and cur["page"] == prev["page"] + 1
                and cur["table"]["n_cols"] == prev["table"]["n_cols"]):
            cur["table"]["continues"] = prev["id"]
