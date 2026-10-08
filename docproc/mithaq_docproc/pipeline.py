"""خط المعالجة الرئيسي:  Document  ->  Clean Text (لكل صفحة + للمستند كامل)."""
from __future__ import annotations

import io
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import extractors as ex
from .config import CONFIG, Config
from .errors import (REVIEW_WARNINGS, CorruptedFileError, NoTextExtractedError,
                     PasswordProtectedError, TooManyPagesError)
from .layout import (Block, PageData, blocks_text, count_chars,
                     remove_headers_footers)
from .ocr import get_engine, lines_to_blocks, preprocess
from .quality import QualityReport, assess_quality, cer
from .cleaning import build_vocab, repair_split_words
from .lexicon import correct_ocr_text
from .structure import link_tables_across_pages, page_items
from .validation import validate_file

log = logging.getLogger("mithaq.docproc")
fitz = ex.fitz


# ----------------------------- النتيجة -------------------------------------
@dataclass
class PageResult:
    page: int | None            # رقم الصفحة (1-based) أو None لو غير متوفر
    text: str
    method: str                 # native | ocr | failed
    confidence: float | None    # ثقة OCR (0..1)
    quality: float | None       # جودة النص (0..1)
    warnings: list[str] = field(default_factory=list)
    items: list[dict] = field(default_factory=list)   # عناصر منظمة: heading / paragraph / list_item / table


@dataclass
class DocumentResult:
    filename: str
    source_type: str            # pdf | docx | docx->pdf
    page_count: int
    pages: list[PageResult]
    text: str                   # النص النظيف الكامل
    stats: dict
    warnings: list[str]
    needs_review: bool
    items: list[dict] = field(default_factory=list)   # كل عناصر المستند بالترتيب (مع id ثابت)
    corrections: list[dict] = field(default_factory=list)   # تصحيحات OCR (للتتبع): page / before / after

    def text_with_page_markers(self) -> str:
        """نص كامل مع علامات <<PAGE n>> (مفيد لاحقاً لربط البنود برقم الصفحة)."""
        parts = []
        for p in self.pages:
            if p.text:
                parts.append(f"<<PAGE {p.page}>>\n{p.text}" if p.page else p.text)
        return "\n\n".join(parts)

    def to_dict(self, include_text: bool = True) -> dict:
        """الصيغة القديمة المختصرة (للتوافق). للصيغة الكاملة استخدم to_json() / output.build_output()."""
        d = {"filename": self.filename, "source_type": self.source_type, "page_count": self.page_count,
             "stats": self.stats, "warnings": self.warnings, "needs_review": self.needs_review,
             "pages": [{"page": p.page, "method": p.method, "confidence": p.confidence, "quality": p.quality,
                        "warnings": p.warnings, **({"text": p.text} if include_text else {"chars": len(p.text)})}
                       for p in self.pages]}
        if include_text:
            d["text"] = self.text
        return d

    def to_json(self) -> dict:
        """الناتج الرسمي للموديول (schema في output_schema.json)."""
        from .output import build_output
        return build_output(self)


class _LazyEngine:
    """نحمّل محرك OCR فقط عند أول صفحة تحتاجه (التحميل بطيء)."""
    def __init__(self, cfg: Config):
        self.cfg, self._eng, self._loaded = cfg, None, False

    def get(self):
        if not self._loaded:
            self._eng, self._loaded = get_engine(self.cfg), True
        return self._eng

    @property
    def name(self):
        return self._eng.name if self._eng else None


# ----------------------------- PDF -----------------------------------------
def _ocr_image_to_page(engine, image, number, cfg: Config) -> tuple[PageData, str]:
    table_blocks, confs = [], []
    work = image
    if cfg.ocr_tables:
        try:
            table_blocks, confs, work = _ocr_tables(engine, image, cfg)
        except Exception as e:                                   # كشف الجداول ما يوقف الصفحة
            log.debug("scan table detection failed: %s", e)
            table_blocks, confs, work = [], [], image
    lines = engine.recognize(preprocess(work, engine.name))
    blocks, conf = lines_to_blocks(lines, image.width, image.height, image.width, image.height)
    if table_blocks:
        blocks = sorted(blocks + table_blocks, key=lambda b: (round(b.bbox[1]), -b.bbox[2]))
        allc = [c for c in confs if c is not None] + ([conf] if conf is not None else [])
        conf = sum(allc) / len(allc) if allc else conf
    pg = PageData(number, image.width, image.height, blocks, "ocr", conf)
    return pg, blocks_text(blocks)


def cell_variants(image, box):
    """صور مجهزة لقراءة خلية: نقصّ النص نفسه (مو الخلية كاملة)، خلفية بيضاء وهامش، وتكبير للنص الصغير.
    Tesseract يضيّع النص الصغير/العريض وسط مساحة فاضية كبيرة. يرجع [(صورة، مقياس، إزاحة x، إزاحة y)]."""
    import numpy as np
    from PIL import Image, ImageOps
    if box[2] - box[0] < 20 or box[3] - box[1] < 15:
        return []
    g = np.asarray(image.crop(box).convert("L"))
    ink = g < 150
    ink[:3, :] = ink[-3:, :] = False
    ink[:, :3] = ink[:, -3:] = False
    ys, xs = np.where(ink)
    if len(ys) < 30:
        return []                                            # الخلية فاضية فعلاً
    pad = 6
    x0, y0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
    x1, y1 = min(g.shape[1], xs.max() + pad), min(g.shape[0], ys.max() + pad)
    tight = Image.fromarray(g[y0:y1, x0:x1])
    out = []
    for binar in (False, True):
        img = ImageOps.autocontrast(tight)
        if binar:
            img = img.point(lambda p: 0 if p < 160 else 255)
        scale = 2 if (y1 - y0) < 70 else 1
        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        img = ImageOps.expand(img, border=30, fill=255)
        out.append((img.convert("RGB"), scale, box[0] + x0, box[1] + y0))
    return out


def _ocr_cell(engine, image, box):
    """قراءة خلية فاضية بأكثر من طريقة؛ أول نتيجة فيها نص بثقة ≥ 0.3. يرجع (أسطر بإحداثيات الصفحة، box)."""
    from .ocr import OCRLine
    for img, scale, ox, oy in cell_variants(image, box):
        for psm in (11, 7, 6):
            try:
                lines = engine.recognize(img, psm=psm)
            except TypeError:                                  # محركات ما تدعم psm
                lines = engine.recognize(img)
            lines = [ln for ln in lines if ln.text.strip() and ln.conf >= 0.3]
            if lines:
                fixed = []
                for ln in lines:
                    bx0, by0, bx1, by1 = ln.bbox
                    fixed.append(OCRLine(ln.text, ((bx0 - 30) / scale + ox - box[0], (by0 - 30) / scale + oy - box[1],
                                                   (bx1 - 30) / scale + ox - box[0], (by1 - 30) / scale + oy - box[1]),
                                         ln.conf))
                return fixed, box
    return [], box


def _ocr_tables(engine, image, cfg: Config):
    """جداول الصفحة الممسوحة: نكشف الشبكة، نقرأ كل عمود لوحده، ونوزع الأسطر على الصفوف.
    يرجع (blocks الجداول، ثقات، صورة الصفحة بعد تبييض الجداول لقراءة الباقي)."""
    from PIL import ImageDraw
    from .scan_tables import detect_tables
    tables = detect_tables(image.convert("L"))
    if not tables:
        return [], [], image
    work = image.copy()
    draw = ImageDraw.Draw(work)
    blocks, confs = [], []
    for t in tables:
        tx0, ty0, tx1, ty1 = t.bbox
        col_lines = []                                      # لكل عمود: [(نص، y_مركز، ثقة)]
        for (cx0, cx1) in t.cols:
            box = (cx0 + 4, ty0, cx1 - 4, ty1)
            crop = image.crop(box)
            lines = engine.recognize(preprocess(crop, engine.name))
            # (نص، y_مركز، y0، ثقة، x0، x1، ارتفاع) بإحداثيات الصفحة
            col_lines.append([(ln.text, box[1] + (ln.bbox[1] + ln.bbox[3]) / 2, ln.bbox[1] + box[1], ln.conf,
                               box[0] + ln.bbox[0], box[0] + ln.bbox[2], max(1, ln.bbox[3] - ln.bbox[1]))
                              for ln in lines if ln.text.strip()])
            confs += [ln.conf for ln in lines if ln.text.strip()]
        rows_y = t.rows or [(ty0, ty1)]
        rtl_guess = True

        def cell_text(lines) -> str:
            """أسطر الخلية: نجمع اللي على نفس الارتفاع في سطر واحد (من اليمين لليسار للعربي)."""
            groups = []
            for ln in sorted(lines, key=lambda ln: ln[1]):
                if groups and abs(ln[1] - groups[-1][-1][1]) < 0.5 * max(ln[6], groups[-1][-1][6]):
                    groups[-1].append(ln)
                else:
                    groups.append([ln])
            return " ".join(" ".join(x[0] for x in sorted(g, key=lambda x: -x[4] if rtl_guess else x[4]))
                            for g in groups)

        grid = []
        for (ry0, ry1) in rows_y:
            grid.append([[ln for ln in lines if ry0 <= ln[1] < ry1] for lines in col_lines])
        # خلية فاضية بين خلايا فيها نص: نص متفرق ما التقطه العمود كامل -> نقرأ الخلية لوحدها
        for r, (ry0, ry1) in enumerate(rows_y):
            for c, (cx0, cx1) in enumerate(t.cols):
                if grid[r][c] or not any(grid[r]):
                    continue
                lines, box = _ocr_cell(engine, image, (cx0 + 6, ry0 + 4, cx1 - 6, ry1 - 4))
                grid[r][c] = [(ln.text, box[1] + (ln.bbox[1] + ln.bbox[3]) / 2, box[1] + ln.bbox[1], ln.conf,
                               box[0] + ln.bbox[0], box[0] + ln.bbox[2], max(1, ln.bbox[3] - ln.bbox[1]))
                              for ln in lines]
                confs += [ln.conf for ln in lines]
        rows = []
        for r in grid:
            cells = [cell_text(lines) for lines in r]
            if any(c.strip() for c in cells):
                rows.append(cells)
        if not rows:
            continue
        text_all = " ".join(" ".join(r) for r in rows)
        if len(re.findall(r"[\u0600-\u06FF]", text_all)) > len(re.findall(r"[A-Za-z]", text_all)):
            rows = [r[::-1] for r in rows]                     # جدول عربي: العمود الأول على اليمين
        blocks.append(Block("\n".join(" | ".join(r) for r in rows), (tx0, ty0, tx1, ty1), "table",
                            sum(confs) / len(confs) if confs else None, rows=rows))
        draw.rectangle((tx0, ty0, tx1, ty1), fill="white")     # باقي الصفحة يُقرأ بدون الجدول
    return blocks, confs, work


def _process_pdf_page(page, number: int, cfg: Config, eng: _LazyEngine, force_ocr: bool) -> PageData:
    w, h = page.rect.width, page.rect.height
    native = [] if force_ocr else ex.native_blocks(page)
    native_text = blocks_text(native)
    n_chars = count_chars(native_text)
    q = assess_quality(native_text) if n_chars else QualityReport(0.0, ["EMPTY"])

    pg = PageData(number, w, h, native, "native")
    need_ocr = force_ocr or n_chars < cfg.min_native_chars or q.score < cfg.min_native_quality

    if need_ocr:
        engine = eng.get()
        if engine is None:
            if n_chars < cfg.min_native_chars:
                pg.warnings.append("OCR_UNAVAILABLE")
            else:
                pg.warnings.append("LOW_NATIVE_QUALITY_NO_OCR")
        else:
            img = ex.render_page(page, cfg.ocr_dpi, cfg.ocr_max_side_px)
            ocr_pg, ocr_text = _ocr_image_to_page(engine, img, number, cfg)
            # نرجع الإحداثيات لوحدات الصفحة (للترويسة/التذييل)
            sx, sy = w / img.width, h / img.height
            ocr_pg.blocks = [Block(b.text, (b.bbox[0] * sx, b.bbox[1] * sy, b.bbox[2] * sx, b.bbox[3] * sy),
                                   b.kind, b.conf, b.rows) for b in ocr_pg.blocks]
            ocr_pg.width, ocr_pg.height = w, h
            oq = assess_quality(ocr_text)
            ocr_chars = count_chars(ocr_text)
            use_ocr = force_ocr or (ocr_chars > n_chars if n_chars < cfg.min_native_chars else oq.score > q.score)
            if use_ocr:
                pg, q = ocr_pg, oq
            if pg.method == "ocr" and pg.confidence is not None and pg.confidence < cfg.ocr_low_confidence:
                pg.warnings.append("LOW_OCR_CONFIDENCE")

    # تحقق اختياري: صفحة عربية نصها الأصلي "سليم" ظاهرياً، نقارنه بـ OCR لكشف مشاكل ترتيب الأقواس/الأرقام
    if (cfg.verify_native_with_ocr and not need_ocr and pg.method == "native"
            and ex._arabic_dominant(native_text)):
        engine = eng.get()
        if engine is not None:
            img = ex.render_page(page, cfg.ocr_dpi, cfg.ocr_max_side_px)
            _, ocr_text = _ocr_image_to_page(engine, img, number, cfg)
            if count_chars(ocr_text) and cer(ocr_text, native_text) > cfg.verify_max_cer:
                pg.warnings.append("NATIVE_OCR_MISMATCH")

    # قراءة مختلطة: نص مرسوم كأشكال (بدون طبقة نص) داخل صفحة نصية -> OCR لهذه المناطق فقط
    if cfg.ocr_uncovered_regions and pg.method == "native" and not force_ocr:
        try:
            regions = ex.uncovered_ink_regions(page, ex.page_glyph_boxes(page))
        except Exception as e:                                 # لا نوقف الصفحة بسبب هذه الخطوة
            log.debug("uncovered regions skipped: %s", e)
            regions = []
        if regions:
            engine = eng.get()
            if engine is None:
                pg.warnings.append("UNREAD_DRAWN_TEXT")
            else:
                extra, confs = _ocr_regions(engine, page, regions, cfg)
                if extra:
                    pg.blocks = ex.native_blocks(page, extra)
                    pg.warnings.append("PARTIAL_OCR")
                    if any(re.search(r"\d", g.text) for g in extra):
                        pg.warnings.append("OCR_NUMBERS_UNVERIFIED")
                    if confs and sum(confs) / len(confs) < cfg.ocr_low_confidence:
                        pg.warnings.append("LOW_OCR_CONFIDENCE")

    # صفحة كاملة بـ OCR فيها أرقام: OCR يضيّع/يشوّه الأرقام وسط النص العربي حتى مع ثقة عالية
    if pg.method == "ocr" and any(re.search(r"\d", b.text) for b in pg.blocks):
        pg.warnings.append("OCR_NUMBERS_UNVERIFIED")

    pg.quality, pg.quality_flags = q.score, q.flags
    for f in ("REVERSED_ARABIC", "LEGACY_ARABIC_ENCODING"):
        if f in q.flags:
            pg.warnings.append(f)
    return pg


def _ocr_regions(engine, page, regions, cfg: Config):
    """OCR لكل منطقة لوحدها -> حروف وهمية (كل سطر OCR = Glyph واحد بإحداثيات الصفحة)."""
    from .bidi import Glyph
    img = ex.render_page(page, cfg.ocr_dpi, cfg.ocr_max_side_px)
    sx, sy = img.width / page.rect.width, img.height / page.rect.height
    from .cleaning import normalize_for_search
    native = [g for gs in ex._page_glyphs(page) for g in gs]
    out, confs = [], []
    for k, (x0, y0, x1, y1) in enumerate(regions):
        pad = 3
        box = (max(0, int((x0 - pad) * sx)), max(0, int((y0 - pad) * sy)),
               min(img.width, int((x1 + pad) * sx)), min(img.height, int((y1 + pad) * sy)))
        crop = img.crop(box)
        try:
            lines = engine.recognize(preprocess(crop, engine.name))
        except Exception as e:
            log.debug("region OCR failed: %s", e)
            continue
        for ln in lines:
            t = ln.text.strip()
            if not t or ln.conf < 0.5 or not re.search(r"[\w]", t):
                continue                                      # شعارات وزخارف تطلع نص ضعيف الثقة
            # حماية من التكرار: لو كلمات OCR موجودة أصلاً في النص المخزن حول المنطقة، ما نضيفها
            near = "".join(g.text for g in native if x0 - 30 <= g.cx <= x1 + 30 and y0 - 15 <= g.cy <= y1 + 15)
            near_n = normalize_for_search(near).replace(" ", "")
            toks = [w for w in normalize_for_search(t).split() if len(w) >= 2]
            if toks and sum(w in near_n or w[::-1] in near_n for w in toks) >= 0.5 * len(toks):
                continue
            bx0, by0, bx1, by1 = ln.bbox
            gx0, gy0 = (box[0] + bx0) / sx, (box[1] + by0) / sy
            gx1, gy1 = (box[0] + bx1) / sx, (box[1] + by1) / sy
            out.append(Glyph(f" {t} ", gx0, gy0, gx1, gy1, 10 ** 7 + k * 1000 + len(out)))   # مسافة تفصله عن جيرانه
            confs.append(ln.conf)
    return out, confs


def _pdf_pages(path, cfg: Config, eng: _LazyEngine, force_ocr: bool, progress) -> list[PageData]:
    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise CorruptedFileError(f"تعذر فتح PDF: {e}") from e
    with doc:
        if doc.needs_pass:
            raise PasswordProtectedError()
        n = doc.page_count
        if n == 0:
            raise CorruptedFileError("ملف PDF لا يحتوي على صفحات.")
        if n > cfg.max_pages:
            raise TooManyPagesError(f"عدد الصفحات {n}، والحد الأقصى {cfg.max_pages}.")
        pages = []
        for i in range(n):
            try:
                pg = _process_pdf_page(doc[i], i + 1, cfg, eng, force_ocr)
            except Exception as e:                           # صفحة فاشلة لا توقف المستند كله
                log.exception("page %d failed", i + 1)
                r = doc[i].rect
                pg = PageData(i + 1, r.width, r.height, [], "failed", warnings=[f"PAGE_FAILED"])
                pg.quality_flags = [f"ERROR: {str(e)[:120]}"]
            pages.append(pg)
            if progress:
                progress(i + 1, n)
        return pages


# ----------------------------- DOCX ----------------------------------------
def _docx_pages(vf, cfg: Config, eng: _LazyEngine, force_ocr: bool, progress, tmpdir: str):
    blocks, info = ex.docx_direct_blocks(vf.path)           # النص من XML مباشرة = Unicode دقيق
    extra = ["AUTO_NUMBERING_MAY_BE_MISSING"] if info["numbered_paragraphs"] else []

    if blocks and cfg.docx_page_numbers:                    # أرقام الصفحات من نسخة PDF (تقريبية)
        pdf = ex.convert_docx_to_pdf(vf.path, tmpdir, cfg.libreoffice_timeout_s)
        if pdf:
            try:
                nums = ex.map_blocks_to_pages(blocks, pdf)
                by_page: dict[int, list] = {}
                for b, n in zip(blocks, nums):
                    by_page.setdefault(n, []).append(b)
                pages = [PageData(n, 0, 0, bl, "native", warnings=["PAGE_NUMBERS_APPROXIMATE"] + extra)
                         for n, bl in sorted(by_page.items())]
                return pages, "docx"
            except Exception as e:
                log.warning("docx page mapping failed: %s", e)

    pg = PageData(None, 0, 0, blocks, "native")
    pg.warnings.append("NO_PAGE_NUMBERS")
    pg.warnings += extra
    pages = [pg]

    if count_chars(blocks_text(blocks)) < cfg.min_native_chars:       # Word ممسوح: صور فقط
        engine = eng.get()
        images = ex.docx_images(vf.path)
        if engine and images:
            from PIL import Image
            ocr_pages = []
            for i, blob in enumerate(images, 1):
                img = Image.open(io.BytesIO(blob)).convert("RGB")
                if min(img.size) < 150:
                    continue
                p, _ = _ocr_image_to_page(engine, img, i, cfg)
                if (p.confidence or 1) < cfg.ocr_low_confidence:
                    p.warnings.append("LOW_OCR_CONFIDENCE")
                ocr_pages.append(p)
                if progress:
                    progress(i, len(images))
            if ocr_pages:
                pages = ocr_pages
        elif images:
            pg.warnings.append("OCR_UNAVAILABLE")
    return pages, "docx"


# ----------------------------- الواجهة الرئيسية ----------------------------
def process_document(path, original_name: str | None = None, cfg: Config = CONFIG,
                     force_ocr: bool = False,
                     progress: Callable[[int, int], None] | None = None) -> DocumentResult:
    """Document -> Clean Text.  يرفع DocProcError (برسالة عربية) لو الملف غير صالح."""
    t0 = time.time()
    vf = validate_file(path, original_name, cfg)
    eng = _LazyEngine(cfg)

    with tempfile.TemporaryDirectory(prefix="mithaq_") as tmp:
        if vf.kind == "pdf":
            pages, source = _pdf_pages(vf.path, cfg, eng, force_ocr, progress), "pdf"
        else:
            pages, source = _docx_pages(vf, cfg, eng, force_ocr, progress, tmp)

    removed = remove_headers_footers(pages) if cfg.remove_headers_footers else 0

    # كلمات انقسمت بمسافة وهمية ("يؤث ر"): نصلحها بمفردات المستند نفسه
    split_fixes: list[str] = []
    if cfg.repair_split_words:
        all_texts = [b.text for pg in pages for b in pg.blocks]
        vocab = build_vocab(all_texts)
        for pg in pages:
            for b in pg.blocks:
                b.text, f = repair_split_words(b.text, vocab)
                split_fixes += f
                if b.rows:
                    new_rows = []
                    for r in b.rows:
                        cells = []
                        for c in r:
                            c2, f2 = repair_split_words(c, vocab)
                            split_fixes += f2
                            cells.append(c2)
                        new_rows.append(cells)
                    b.rows = new_rows

    # تصحيح أخطاء OCR الشائعة (ي/ب داخل الكلمة) — صفحات OCR فقط، بقاموس + كلمات صفحات المستند النصية
    ocr_corrections: list[dict] = []
    if cfg.ocr_lexicon_correction and any(pg.method == "ocr" for pg in pages):
        from .lexicon import load_lexicon, words_of
        known = load_lexicon(tuple(cfg.lexicon_dirs)).copy()
        for pg in pages:
            if pg.method == "native":
                known.update(words_of(" ".join(b.text for b in pg.blocks)))
        for pg in pages:
            if pg.method != "ocr":
                continue
            for b in pg.blocks:
                b.text, f = correct_ocr_text(b.text, known)
                if not b.rows:                     # الجداول نعدّ تصحيحاتها من الخلايا (بدون تكرار)
                    ocr_corrections += [{"page": pg.number, "before": x, "after": y} for x, y in f]
                if b.rows:
                    new_rows = []
                    for r in b.rows:
                        cells = []
                        for cell in r:
                            c2, f2 = correct_ocr_text(cell, known)
                            ocr_corrections += [{"page": pg.number, "before": x, "after": y} for x, y in f2]
                            cells.append(c2)
                        new_rows.append(cells)
                    b.rows = new_rows

    results: list[PageResult] = []
    for pg in pages:
        items = page_items(pg.blocks, pg.number, pg.method)
        text = "\n\n".join(it["text"] for it in items)
        if not text and pg.method != "failed" and "EMPTY_PAGE" not in pg.warnings and not any(
                w in pg.warnings for w in ("OCR_UNAVAILABLE",)):
            pg.warnings.append("EMPTY_PAGE")
        results.append(PageResult(pg.number, text, pg.method,
                                  round(pg.confidence, 3) if pg.confidence is not None else None,
                                  pg.quality, list(dict.fromkeys(pg.warnings)), items))

    all_items: list[dict] = []
    for r in results:                                         # id ثابت لكل عنصر: p{page}-{n}
        for i, it in enumerate(r.items, 1):
            it["id"] = f"p{r.page or 0}-{i}"
            all_items.append(it)
    link_tables_across_pages(all_items)

    full = "\n\n".join(r.text for r in results if r.text)
    if not full.strip():
        hint = "" if eng.get() else " (محرك OCR غير متوفر على الخادم)"
        raise NoTextExtractedError(f"لا يوجد نص قابل للاستخراج{hint}.")

    all_warn = sorted({w for r in results for w in r.warnings})
    confs = [r.confidence for r in results if r.confidence is not None]
    quals = [r.quality for r in results if r.quality is not None]
    stats = {
        "chars": len(full), "words": len(full.split()),
        "native_pages": sum(r.method == "native" for r in results),
        "ocr_pages": sum(r.method == "ocr" for r in results),
        "failed_pages": sum(r.method == "failed" for r in results),
        "empty_pages": sum("EMPTY_PAGE" in r.warnings for r in results),
        "removed_header_footer_blocks": removed,
        "avg_quality": round(sum(quals) / len(quals), 3) if quals else None,
        "avg_ocr_confidence": round(sum(confs) / len(confs), 3) if confs else None,
        "ocr_engine": eng.name, "seconds": round(time.time() - t0, 2),
    }
    stats["tables"] = sum(it["type"] == "table" for it in all_items)
    stats["split_words_repaired"] = len(split_fixes)
    stats["ocr_words_corrected"] = len(ocr_corrections)
    stats["items"] = len(all_items)
    return DocumentResult(vf.display_name, source, len(results), results, full, stats, all_warn,
                          needs_review=bool(set(all_warn) & REVIEW_WARNINGS), items=all_items,
                          corrections=ocr_corrections)
