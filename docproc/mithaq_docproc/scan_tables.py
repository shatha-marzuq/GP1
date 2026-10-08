"""كشف الجداول في الصفحات الممسوحة (OCR) وقراءتها خلية خلية.

المشكلة: OCR يقرأ الجدول كنص عادي فتختلط الأعمدة ("اسم الصندوق بلوم إنفست | صندوق بلوم...").
الحل: نكشف شبكة الجدول من الصورة نفسها:
  - الصفوف: خطوط أفقية طويلة، أو تبادل لون الخلفية (صفوف مظللة/بيضاء)
  - الأعمدة: خطوط عمودية طويلة، أو تغيّر لون الخلفية (عمود مظلل)
ثم نقرأ كل عمود لوحده ونوزع أسطره على الصفوف حسب الارتفاع.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScanTable:
    bbox: tuple                 # (x0, y0, x1, y1) بكسل
    cols: list                  # [(x0, x1)] من اليسار لليمين
    rows: list                  # [(y0, y1)] من الأعلى للأسفل (قد تكون فاضية = نستنتجها من النص)


def _runs(mask_1d) -> int:
    """أطول امتداد متصل True."""
    if not mask_1d.any():
        return 0
    d = np.diff(np.concatenate(([0], mask_1d.astype(np.int8), [0])))
    s, e = np.where(d == 1)[0], np.where(d == -1)[0]
    return int((e - s).max())


def _group(idx, gap=3) -> list:
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([int(i)])
    return out


def _shade_steps(profile, min_step=5, win=3, min_gap=15) -> list:
    """مواقع تغيّر لون الخلفية (الوسيط) بقيمة واضحة."""
    p = np.asarray(profile, dtype=float)
    n = len(p)
    steps = [i for i in range(win, n - win) if abs(p[i + win] - p[i - win]) >= min_step]
    out = []
    for s in steps:
        if not out or s - out[-1] > min_gap:
            out.append(s)
        else:
            out[-1] = (out[-1] + s) // 2
    return out


def detect_tables(gray, min_rows: int = 2, min_cols: int = 2) -> list[ScanTable]:
    """gray: صورة رمادية (PIL أو numpy). يرجع الجداول المكتشفة."""
    a = np.asarray(gray).astype(np.int16)
    H, W = a.shape
    line = a < 238                                   # خطوط الجداول غالباً رمادي فاتح

    # 1) خطوط أفقية: امتداد متصل ≥ 45% من عرض الصفحة وسمك صغير
    hl = [y for y in range(H) if line[y].sum() > 0.45 * W and _runs(line[y]) > 0.45 * W]
    hgroups = [g for g in _group(hl) if len(g) <= 12]
    hlines = [int(np.mean(g)) for g in hgroups]

    # 2) خطوط عمودية: امتداد ≥ 15% من ارتفاع الصفحة
    vcand = [x for x in range(W) if line[:, x].sum() > 0.15 * H and _runs(line[:, x]) > 0.15 * H]
    vgroups = [g for g in _group(vcand) if len(g) <= 12]
    vlines = []
    for g in vgroups:
        x = int(np.mean(g))
        col = line[:, x]
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        s, e = np.where(d == 1)[0], np.where(d == -1)[0]
        k = int(np.argmax(e - s))
        vlines.append((x, int(s[k]), int(e[k])))

    # 3) حدود الجدول العمودية: من الخطوط الأفقية، أو من الخطوط العمودية
    if len(hlines) >= 2:
        ty0, ty1 = hlines[0], hlines[-1]
    elif vlines:
        ty0, ty1 = min(v[1] for v in vlines), max(v[2] for v in vlines)
        if hlines:
            ty0 = min(ty0, hlines[0])
    else:
        return []
    if ty1 - ty0 < 0.08 * H:
        return []
    region = a[ty0 + 4:ty1 - 4]

    # 4) الأعمدة: الخطوط العمودية لو موجودة (أدق)، وإلا تغيّر تظليل الأعمدة
    inner_v = [v[0] for v in vlines if v[1] <= ty0 + 0.3 * (ty1 - ty0) and v[2] >= ty1 - 0.3 * (ty1 - ty0)]
    if inner_v:
        xs = list(inner_v)
        # حدود الجدول الخارجية لو ما فيه خط: أبعد حبر داخل نطاق الجدول
        ink_cols = np.where((region < 160).sum(axis=0) > 0)[0]
        if len(ink_cols):
            if min(xs) - ink_cols[0] > 0.06 * W:
                xs.append(int(ink_cols[0]) - 10)
            if ink_cols[-1] - max(xs) > 0.06 * W:
                xs.append(int(ink_cols[-1]) + 10)
    else:
        xs = _shade_steps(np.median(region, axis=0))
    xs = sorted(set(xs))
    merged = []
    for x in xs:
        if merged and x - merged[-1] < 0.03 * W:
            continue
        merged.append(x)
    if len(merged) < 2:
        return []
    cols = [(merged[i], merged[i + 1]) for i in range(len(merged) - 1) if merged[i + 1] - merged[i] >= 0.06 * W]
    cols = [(x0, x1) for x0, x1 in cols if (a[ty0:ty1, x0 + 5:x1 - 5] < 160).mean() > 0.002]
    if len(cols) < min_cols:
        return []
    tx0, tx1 = cols[0][0], cols[-1][1]

    # 5) الصفوف: خطوط أفقية داخل الجدول، وإلا تبادل تظليل الصفوف مقاساً على شرائط ضيقة
    #    عند أطراف الأعمدة (النص قليل هناك فالخلفية واضحة)
    rows_y = [y for y in hlines if ty0 <= y <= ty1]
    if len(rows_y) < min_rows + 1:
        strips = []
        for x0, x1 in cols:
            strips.append(a[ty0:ty1, x0 + 6:x0 + 26])
            strips.append(a[ty0:ty1, x1 - 26:x1 - 6])
        prof = np.median(np.concatenate(strips, axis=1), axis=1)
        steps = [ty0 + s for s in _shade_steps(prof, min_step=4, win=2, min_gap=int(0.012 * H))]
        rows_y = sorted(set([ty0, ty1] + steps))
    rows = [(rows_y[i], rows_y[i + 1]) for i in range(len(rows_y) - 1) if rows_y[i + 1] - rows_y[i] > 0.012 * H]
    return [ScanTable((tx0, ty0, tx1, ty1), cols, rows)]
