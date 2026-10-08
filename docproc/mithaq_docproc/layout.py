from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .cleaning import join_lines, normalize_for_search


@dataclass
class Block:
    text: str
    bbox: tuple                 # (x0, y0, x1, y1) بوحدات الصفحة
    kind: str = "text"          # text | table
    conf: float | None = None
    rows: list | None = None    # للجداول فقط: [[خلية, خلية], ...] (الصف الأول غالباً العناوين)


@dataclass
class PageData:
    number: int | None
    width: float
    height: float               # 0 = لا توجد إحداثيات (مثل docx المباشر)
    blocks: list[Block] = field(default_factory=list)
    method: str = "native"      # native | ocr | failed
    confidence: float | None = None
    quality: float | None = None
    quality_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def blocks_text(blocks: list[Block]) -> str:
    parts: list[str] = []
    for b in blocks:
        if b.kind == "table":
            parts.append(b.text.strip())
        else:
            parts.extend(join_lines(b.text.split("\n")))
    return "\n\n".join(p for p in parts if p)


def count_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


# ---------------------------------------------------------------------------
_MARGIN = 0.10
_PAGE_NUM_RE = re.compile(
    r"^[\s\-–—]*(?:page|صفحة|ص)?\s*[\d٠-٩]{1,4}(?:\s*(?:of|/|من)\s*[\d٠-٩]{1,4})?[\s\-–—]*$", re.I)


def _in_margin(b: Block, h: float) -> bool:
    if h <= 0:
        return False
    return b.bbox[3] <= h * _MARGIN or b.bbox[1] >= h * (1 - _MARGIN)


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", normalize_for_search(text))).strip().lower()


def remove_headers_footers(pages: list[PageData]) -> int:
    """يحذف أرقام الصفحات + النصوص المتكررة في هوامش الصفحات (ترويسة/تذييل). يرجع عدد المحذوف."""
    removed = 0
    for pg in pages:
        keep = []
        for b in pg.blocks:
            if b.kind == "text" and _in_margin(b, pg.height) and _PAGE_NUM_RE.match(b.text.strip()):
                removed += 1
            else:
                keep.append(b)
        pg.blocks = keep

    n = len(pages)
    if n < 3:
        return removed

    clusters: list[dict] = []
    for pi, pg in enumerate(pages):
        for bi, b in enumerate(pg.blocks):
            if b.kind != "text" or not _in_margin(b, pg.height) or len(b.text) > 200:
                continue
            k = _key(b.text)
            if len(k) < 2:
                continue
            for c in clusters:
                if SequenceMatcher(None, k, c["key"]).ratio() >= 0.88:
                    c["members"].append((pi, bi)); c["pages"].add(pi)
                    break
            else:
                clusters.append({"key": k, "members": [(pi, bi)], "pages": {pi}})

    threshold = max(3, math.ceil(0.4 * n))
    drop = {m for c in clusters if len(c["pages"]) >= threshold for m in c["members"]}
    if drop:
        for pi, pg in enumerate(pages):
            before = len(pg.blocks)
            pg.blocks = [b for bi, b in enumerate(pg.blocks) if (pi, bi) not in drop]
            removed += before - len(pg.blocks)
    return removed
