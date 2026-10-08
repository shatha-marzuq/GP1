
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_AR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_LTR_STRONG = re.compile(r"[A-Za-z0-9\u0660-\u0669\u06F0-\u06F9\u00C0-\u024F]")
_JOINERS = set(".,،:/\\-–+_@#&%٪'")
_EDGE_ATTACH = set("%٪+#$")


@dataclass
class Glyph:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    seq: int = 0            
    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 0.1)


def _cls(text: str) -> str:
    kinds = {unicodedata.bidirectional(c) for c in text}
    if kinds & {"AL", "R"}:
        return "R"
    if kinds & {"L", "EN", "AN"}:
        return "L"
    return "N"


def _is_mark(text: str) -> bool:
    return bool(text) and all(unicodedata.category(c) == "Mn" for c in text)


def _fix_unit(text: str) -> str:
    letters = "".join(c for c in text if unicodedata.category(c) != "Mn")
    if len(letters) >= 2 and letters[-1] == "ل" and letters[0] != "ل" and all(_AR.match(c) for c in letters):
        clusters: list[str] = []
        for c in text:                      # نعكس الحروف ونبقي كل تشكيل مع حرفه
            if clusters and unicodedata.category(c) == "Mn":
                clusters[-1] += c
            else:
                clusters.append(c)
        return "".join(reversed(clusters))
    return text


def attach_marks(glyphs: list[Glyph]) -> list[Glyph]:
    by_seq = sorted(glyphs, key=lambda g: g.seq)
    bases = [g for g in glyphs if not _is_mark(g.text)]

    def near(b: Glyph, m: Glyph) -> bool:
        return (_AR.search(b.text) is not None and b.x0 - 0.6 <= m.cx <= b.x1 + 0.6
                and max(b.y0 - m.cy, 0.0, m.cy - b.y1) < 0.8 * b.h)

    n = len(by_seq)
    prev_of: list = [None] * n
    next_of: list = [None] * n
    last = None
    for i, g in enumerate(by_seq):
        prev_of[i] = last
        if not _is_mark(g.text):
            last = g
    last = None
    for i in range(n - 1, -1, -1):
        next_of[i] = last
        if not _is_mark(by_seq[i].text):
            last = by_seq[i]

    for i, m in enumerate(by_seq):
        if not _is_mark(m.text):
            continue
        prev, nxt = prev_of[i], next_of[i]
        p_ok = prev is not None and near(prev, m)
        n_ok = nxt is not None and near(nxt, m)
        if p_ok and n_ok:
            target = prev if prev.cx > nxt.cx else nxt
        else:
            target = prev if p_ok else (nxt if n_ok else None)
        if target is None:
            def dist(b: Glyph) -> tuple[float, float]:
                dy = max(b.y0 - m.cy, 0.0, m.cy - b.y1)
                dx = max(b.x0 - m.cx, 0.0, m.cx - b.x1)
                return dy + dx, abs(b.cx - m.cx)
            cands = [b for b in bases if _AR.search(b.text) and max(b.y0 - m.cy, 0.0, m.cy - b.y1) < 0.8 * b.h]
            target = min(cands, key=dist) if cands else None
        if target is not None:
            target.text += m.text
    return bases


def drop_phantom_spaces(glyphs: list[Glyph]) -> list[Glyph]:
    import bisect
    solid_all = sorted((g for g in glyphs if not g.text.isspace()), key=lambda g: g.x0)
    xs = [g.x0 for g in solid_all]
    max_w = max((g.x1 - g.x0 for g in solid_all), default=0.0)

    def nearby(g: Glyph, pad: float = 1.0) -> list[Glyph]:
        lo = bisect.bisect_left(xs, g.x0 - max_w - pad)
        hi = bisect.bisect_right(xs, g.x1 + pad)
        return [b for b in solid_all[lo:hi] if abs(b.cy - g.cy) < max(b.h, g.h)]

    def coverage(g: Glyph) -> float:
        w = max(g.x1 - g.x0, 0.1)
        return max(((min(b.x1, g.x1) - max(b.x0, g.x0)) / w for b in nearby(g)
                    if abs(b.cy - g.cy) < max(b.h, g.h) / 2), default=0.0)

    spaces = [g for g in glyphs if g.text.isspace() and g.x1 - g.x0 >= 0.3]
    drop = {id(g) for g in glyphs if g.text.isspace() and g.x1 - g.x0 < 0.3}

    def cover_glyph(g: Glyph):
        same = [b for b in nearby(g) if abs(b.cy - g.cy) < max(b.h, g.h) / 2]
        return max(same, key=lambda b: min(b.x1, g.x1) - max(b.x0, g.x0), default=None)

    cov = {id(g): coverage(g) for g in spaces}
    cg = {id(g): cover_glyph(g) for g in spaces}
    # الأفضلية: أقل تغطية، ثم الأبعد عن وسط الحرف اللي يغطيها
    rank = {id(g): (cov[id(g)], -abs(g.cx - cg[id(g)].cx) if cg[id(g)] else 0.0) for g in spaces}
    def inside_one_glyph(g: Glyph, margin: float = 0.3) -> bool:
        return any(abs(b.cy - g.cy) < max(b.h, g.h) / 2 and b.x0 + margin <= g.x0 and g.x1 <= b.x1 - margin
                   for b in nearby(g))

    sp_sorted = sorted(spaces, key=lambda t: t.cx)
    sp_cx = [t.cx for t in sp_sorted]
    max_sw = max((t.x1 - t.x0 for t in spaces), default=0.0)
    widths = sorted(g.x1 - g.x0 for g in spaces)
    med_w = widths[len(widths) // 2] if widths else 0.0

    def split_final_yaa(g: Glyph) -> bool:
        if not med_w or (g.x1 - g.x0) < 1.7 * med_w:
            return False
        same = [b for b in nearby(g, pad=max_w + 1.0) if abs(b.cy - g.cy) < max(b.h, g.h) / 2]
        left = [b for b in same if abs(b.x1 - g.x0) < 0.5 and b.text in ("ي", "ى")]
        right = [b for b in same if abs(b.x0 - g.x1) < 0.5 and _AR.search(b.text)]
        if not (left and right):
            return False
        y = left[0]                   
        return not any(abs(b.x1 - y.x0) < 0.5 and _AR.search(b.text) for b in same if b is not y)

    def contains_letter(g: Glyph) -> bool:
        return any(abs(b.cy - g.cy) < max(b.h, g.h) / 2 and g.x0 - 0.05 <= b.x0 and b.x1 <= g.x1 + 0.05
                   and (b.x1 - b.x0) > 0.5 for b in nearby(g))

    def inside_number(g: Glyph) -> bool:
        near = [b for b in nearby(g) if abs(b.cy - g.cy) < max(b.h, g.h) / 2]
        left = [b for b in near if b.cx < g.cx and _LTR_STRONG.search(b.text) or b.text in "/.,:-" and b.cx < g.cx]
        right = [b for b in near if b.cx > g.cx and (_LTR_STRONG.search(b.text) or b.text in "/.,:-")]
        if not (left and right) or any(_AR.search(b.text) for b in near
                                       if min(b.x1, g.x1 + 1) > max(b.x0, g.x0 - 1)):
            return False
        return cov[id(g)] >= 0.8

    for g in spaces:
        if inside_one_glyph(g) or split_final_yaa(g) or contains_letter(g) or inside_number(g):
            drop.add(id(g))                  # Sakkal Majalla: "الأ سهم" -> "الأسهم"
            continue
        if cov[id(g)] <= 0.5:
            continue
        w = g.x1 - g.x0
        lo = bisect.bisect_left(sp_cx, g.cx - 2 * max(w, max_sw))
        hi = bisect.bisect_right(sp_cx, g.cx + 2 * max(w, max_sw))
        twins = [t for t in sp_sorted[lo:hi] if t is not g and abs(t.cy - g.cy) < g.h / 2
                 and abs(t.cx - g.cx) < 2 * max(w, t.x1 - t.x0)]
        if any(rank[id(t)] < rank[id(g)] for t in twins):
            drop.add(id(g))
    out = []
    for g in glyphs:
        if id(g) in drop:
            continue
        b = cg.get(id(g)) if g.text.isspace() else None
        if b is not None and cov[id(g)] > 0.5:
            
            edge = "left" if (g.cx - b.x0) <= (b.x1 - g.cx) else "right"
            other_x = b.x1 if edge == "left" else b.x0
            lonely = (len(b.text) == 1 and _AR.search(b.text) and b.text != "و"
                      and any(t is not g and id(t) not in drop and abs(t.cy - g.cy) < g.h / 2
                              and t.x0 - 0.6 <= other_x <= t.x1 + 0.6 for t in spaces))
            if lonely:
                continue
            x = b.x0 - 0.01 if edge == "left" else b.x1 + 0.01
            g = Glyph(g.text, x, g.y0, x, g.y1, g.seq)
        out.append(g)
    return out


def _reverse_clusters(text: str) -> str:
    clusters: list[str] = []
    for c in text:
        if clusters and unicodedata.category(c) == "Mn":
            clusters[-1] += c
        else:
            clusters.append(c)
    return "".join(reversed(clusters))


def merge_zero_width(glyphs: list[Glyph], eps: float = 0.05) -> list[Glyph]:
    
    glyphs = attach_marks([Glyph(g.text, g.x0, g.y0, g.x1, g.y1, g.seq) for g in glyphs])
    glyphs = drop_phantom_spaces(glyphs)
    seq = sorted(glyphs, key=lambda g: g.seq)

    def is_zero(g: Glyph) -> bool:
        return (g.x1 - g.x0) < eps and not g.text.isspace()

    def same_line(a: Glyph, b: Glyph) -> bool:
        return abs(a.cy - b.cy) < max(a.h, b.h)

    out: list[Glyph] = []
    pending: list[Glyph] = []
    for g in seq:
        if is_zero(g):
            pending.append(g)
            continue
        if pending:
            z = pending[0].x0
            if same_line(pending[0], g) and abs(g.x1 - z) < 0.5 and not g.text.isspace():
                text = _reverse_clusters("".join(p.text for p in pending) + g.text)
                out.append(Glyph(text, g.x0, g.y0, g.x1, g.y1, pending[0].seq))
                pending = []
                continue
            _flush_after(out, pending, same_line)
            pending = []
        out.append(g)
    if pending:
        _flush_after(out, pending, same_line)
    return out


def _flush_after(out: list[Glyph], pending: list[Glyph], same_line) -> None:
    prev = next((g for g in reversed(out) if not g.text.isspace()), None)
    if prev is not None and same_line(prev, pending[0]):
        prev.text = _fix_unit(prev.text + "".join(p.text for p in pending))
    else:
        out.extend(pending)


def group_lines(glyphs: list[Glyph]) -> list[list[Glyph]]:
    gs = [g for g in glyphs if g.text]
    if not gs:
        return []
    hs = sorted(g.h for g in gs if not g.text.isspace()) or [10.0]
    tol = 0.5 * hs[len(hs) // 2]
    lines: list[list[Glyph]] = []
    centers: list[float] = []
    for g in sorted(gs, key=lambda g: (g.cy, g.seq)):
        for i, c in enumerate(centers):
            if abs(g.cy - c) <= tol:
                lines[i].append(g)
                break
        else:
            lines.append([g])
            centers.append(g.cy)
    order = sorted(range(len(lines)), key=lambda i: centers[i])
    return [lines[i] for i in order]


def line_direction(glyphs: list[Glyph]) -> str:
    r = sum(1 for g in glyphs if _cls(g.text) == "R")
    l = sum(1 for g in glyphs if any(unicodedata.bidirectional(c) == "L" for c in g.text))
    return "rtl" if r >= l and r > 0 else "ltr"


def _resolve(classes: list[str], base: str) -> list[str]:
    n = len(classes)
    res = classes[:]
    i = 0
    while i < n:
        if res[i] != "N":
            i += 1
            continue
        j = i
        while j < n and classes[j] == "N":
            j += 1
        left = classes[i - 1] if i > 0 else None
        right = classes[j] if j < n else None
        d = left if (left is not None and left == right) else ("R" if base == "rtl" else "L")
        for k in range(i, j):
            res[k] = d
        i = j
    return res


def order_line(glyphs: list[Glyph], base: str | None = None) -> str:
    if not glyphs:
        return ""
    base = base or line_direction(glyphs)
    vis = sorted(glyphs, key=lambda g: (g.cx, g.seq))          # من اليسار لليمين كما يظهر
    texts = [g.text for g in vis]
    cls = [_cls(t) for t in texts]

    def touching(i: int, j: int) -> bool:
        a, b = (vis[i], vis[j]) if i < j else (vis[j], vis[i])
        return b.x0 - a.x1 < 0.6

    for i, t in enumerate(texts):
        if cls[i] == "N" and t in _JOINERS:
            lft = cls[i - 1] if i > 0 else None
            rgt = cls[i + 1] if i + 1 < len(cls) else None
            if (lft == "L" and rgt == "L") or (t in _EDGE_ATTACH and "L" in (lft, rgt)):
                cls[i] = "L"
            elif t in "-–" and ((lft == "L" and touching(i, i - 1)) or (rgt == "L" and touching(i, i + 1))):
                cls[i] = "L"                                    # "BBB-" : الشرطة ملاصقة للكلمة كما تظهر
    res = _resolve(cls, base)

    runs: list[tuple[str, list[str]]] = []
    for t, d in zip(texts, res):
        if runs and runs[-1][0] == d:
            runs[-1][1].append(t)
        else:
            runs.append((d, [t]))

    if base == "rtl":
        parts = []
        for d, ts in reversed(runs):
            parts.append("".join(ts) if d == "L" else "".join(reversed(ts)))
    else:
        parts = ["".join(ts) if d == "L" else "".join(reversed(ts)) for d, ts in runs]
    return _tidy("".join(parts))


_PHONE_REV = re.compile(r"(?<![\d+])((?:\+?\d{1,8} ){2,5})(\+?(?:00)?966)(?![\d])")


def _fix_phone(s: str) -> str:
    def rev(m: re.Match) -> str:
        groups = m.group(1).split() + [m.group(2)]
        plus = any(g.startswith("+") for g in groups)
        groups = [g.lstrip("+") for g in reversed(groups)]
        if plus and not groups[0].startswith("00"):
            groups[0] = "+" + groups[0]
        return " ".join(groups)
    return _PHONE_REV.sub(rev, s)


def _tidy(s: str) -> str:
    s = _fix_phone(unicodedata.normalize("NFC", s))
    return re.sub(r"[ \t\u00a0]+", " ", s).strip()


def glyphs_to_line_records(glyphs: list[Glyph]) -> list[tuple[str, float, float]]:
    glyphs = merge_zero_width(glyphs)
    out = []
    for l in group_lines(glyphs):
        t = order_line(l)
        if t:
            solid = [g for g in l if not g.text.isspace()] or l
            out.append((t, min(g.x0 for g in solid), max(g.x1 for g in solid)))
    return out


def glyphs_to_lines(glyphs: list[Glyph]) -> list[str]:
    glyphs = merge_zero_width(glyphs)
    return [t for t in (order_line(l) for l in group_lines(glyphs)) if t]


def has_arabic(text: str) -> bool:
    return bool(_AR.search(text or ""))
