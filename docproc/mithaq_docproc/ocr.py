"""محركات OCR: Tesseract (الأساسي بعد التجربة: أدق على ملفاتنا العربية) و PaddleOCR / EasyOCR (بدائل).

كل محرك يرجع أسطراً (نص + مربع + ثقة). بعدها نرتب الأسطر ونجمّعها في فقرات حسب الهندسة.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .cleaning import starts_new_block
from .config import Config
from .errors import OCRUnavailableError
from .layout import Block

log = logging.getLogger("mithaq.docproc")


@dataclass
class OCRLine:
    text: str
    bbox: tuple          # بكسل: (x0, y0, x1, y1)
    conf: float          # 0..1


class OCREngine(ABC):
    name = "base"

    @abstractmethod
    def recognize(self, image) -> list[OCRLine]: ...


# ---------------------------------------------------------------------------
class PaddleEngine(OCREngine):
    name = "paddleocr"

    def __init__(self, lang: str = "ar"):
        try:
            from paddleocr import PaddleOCR
        except Exception as e:  
            raise OCRUnavailableError(f"paddleocr غير مثبت: {e}") from e

       
        attempts = [
            dict(lang=lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                 use_textline_orientation=True, text_det_limit_side_len=2500, text_det_limit_type="max"),  # 3.x
            dict(lang=lang, use_angle_cls=True, show_log=False,
                 det_limit_side_len=2500, det_limit_type="max"),                                            # 2.x
        ]
        errors = []
        self._ocr = None
        for kw in attempts:
            try:
                self._ocr = PaddleOCR(**kw)
                break
            except Exception as e:
                errors.append(str(e)[:200])
        if self._ocr is None:
            raise OCRUnavailableError("تعذر تشغيل PaddleOCR: " + " | ".join(errors))

    def recognize(self, image) -> list[OCRLine]:
        import numpy as np
        arr = np.asarray(image.convert("RGB"))
        out: list[OCRLine] = []

        def add(text, score, poly):
            text = (text or "").strip()
            if not text:
                return
            p = np.asarray(poly, dtype=float).reshape(-1, 2)
            out.append(OCRLine(text, (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()), float(score)))

        if hasattr(self._ocr, "predict"):                       # 3.x
            for res in self._ocr.predict(arr):
                d = dict(res)
                d = d.get("res", d)
                texts = d.get("rec_texts", [])
                scores = d.get("rec_scores", [1.0] * len(texts))
                polys = d.get("rec_polys")
                if polys is None or len(polys) == 0:
                    polys = d.get("dt_polys", [])
                for t, s, p in zip(texts, scores, polys):
                    add(t, s, p)
        else:                                                   # 2.x
            result = self._ocr.ocr(arr, cls=True)
            for line in (result[0] if result and result[0] else []):
                poly, (t, s) = line
                add(t, s, poly)
        return out


class SuryaEngine(OCREngine):
    name = "surya"

    def __init__(self, langs=("ar", "en")):
        self.langs = list(langs)
        errors = []
        self._mode = None
        try:                                                    # الإصدارات الحديثة (0.14+)
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            self._det = DetectionPredictor()
            self._rec = RecognitionPredictor(FoundationPredictor())
            self._mode = "foundation"
        except Exception as e:
            errors.append(f"foundation: {str(e)[:120]}")
        if self._mode is None:
            try:                                                # الإصدارات 0.7 - 0.13
                from surya.detection import DetectionPredictor
                from surya.recognition import RecognitionPredictor
                self._det = DetectionPredictor()
                self._rec = RecognitionPredictor()
                self._mode = "predictor"
            except Exception as e:
                errors.append(f"predictor: {str(e)[:120]}")
        if self._mode is None:
            try:                                                # الإصدارات القديمة (0.4 - 0.6)
                from surya.model.detection.model import load_model as ldm, load_processor as ldp
                from surya.model.recognition.model import load_model as lrm
                from surya.model.recognition.processor import load_processor as lrp
                from surya.ocr import run_ocr
                self._old = (run_ocr, ldm(), ldp(), lrm(), lrp())
                self._mode = "old"
            except Exception as e:
                errors.append(f"old: {str(e)[:120]}")
        if self._mode is None:
            raise OCRUnavailableError("Surya غير متوفر (pip install surya-ocr): " + " | ".join(errors))

    def _predict(self, image):
        img = image.convert("RGB")
        if self._mode == "old":
            run_ocr, dm, dp, rm, rp = self._old
            return run_ocr([img], [self.langs], dm, dp, rm, rp)[0]
        if self._mode == "foundation":
            return self._rec([img], det_predictor=self._det)[0]
        try:
            return self._rec([img], [self.langs], self._det)[0]       # 0.7 - 0.9
        except TypeError:
            return self._rec([img], det_predictor=self._det)[0]      # 0.10 - 0.13

    def recognize(self, image, psm: int = 3) -> list[OCRLine]:
        res = self._predict(image)
        out = []
        for ln in getattr(res, "text_lines", []) or []:
            text = (getattr(ln, "text", "") or "").strip()
            text = re.sub(r"</?[a-z]+>", "", text).strip()           # بعض الإصدارات ترجع وسوم تنسيق <b>
            if not text:
                continue
            bb = getattr(ln, "bbox", None)
            if bb is None and getattr(ln, "polygon", None):
                xs = [p[0] for p in ln.polygon]
                ys = [p[1] for p in ln.polygon]
                bb = (min(xs), min(ys), max(xs), max(ys))
            conf = getattr(ln, "confidence", None)
            out.append(OCRLine(text, tuple(float(v) for v in bb), float(conf) if conf is not None else 0.9))
        return out


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self, langs=("ar", "en")):
        try:
            import easyocr
            self._reader = easyocr.Reader(list(langs), gpu=False, verbose=False)
        except Exception as e:
            raise OCRUnavailableError(f"EasyOCR غير متوفر: {e}") from e

    def recognize(self, image) -> list[OCRLine]:
        import numpy as np
        from PIL import Image
        # الصفحات الكبيرة تستهلك ذاكرة هائلة مع EasyOCR (جربت 300DPI فانقتلت العملية)، فنصغّر داخلياً
        k = min(1.0, 2000 / max(image.size))
        img = image.convert("RGB").resize((int(image.width * k), int(image.height * k)), Image.LANCZOS) if k < 1 else image.convert("RGB")
        out = []
        for poly, text, conf in self._reader.readtext(np.asarray(img), paragraph=False):
            text = (text or "").strip()
            if text:
                p = np.asarray(poly, dtype=float) / k
                out.append(OCRLine(text, (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()), float(conf)))
        return out


def _find_tesseract(cmd: str | None = None) -> str | None:
    import os
    import shutil
    for c in (cmd, os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
              r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
              os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe")):
        if c and os.path.isfile(c):
            return c
    return None


class TesseractEngine(OCREngine):
    name = "tesseract"

    def __init__(self, lang: str = "ara+eng", cmd: str | None = None):
        try:
            import pytesseract
            exe = _find_tesseract(cmd)
            if exe:
                pytesseract.pytesseract.tesseract_cmd = exe
            pytesseract.get_tesseract_version()
            have = set(pytesseract.get_languages())
        except Exception as e:
            raise OCRUnavailableError(f"Tesseract غير متوفر: {e}") from e
        missing = [l for l in lang.split("+") if l not in have]
        if missing:
            raise OCRUnavailableError(f"حزم لغة Tesseract ناقصة: {missing}")
        self._pt, self.lang = pytesseract, lang
        self._have = have
        self.fix_numbers = True              # إعادة قراءة الأرقام/اللاتيني داخل الأسطر العربية

    def recognize(self, image, psm: int = 3) -> list[OCRLine]:
        d = self._pt.image_to_data(image, lang=self.lang, config=f"--oem 1 --psm {psm}",
                                   output_type=self._pt.Output.DICT)
        lines: dict[tuple, dict] = {}
        for i, txt in enumerate(d["text"]):
            txt = (txt or "").strip()
            if not txt:
                continue
            conf = float(d["conf"][i])
            key = (d["block_num"][i], d["par_num"][i], d["line_num"][i])
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            L = lines.setdefault(key, {"words": [], "c": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h})
            L["words"].append([txt, (x, y, x + w, y + h), conf])
            if conf >= 0:
                L["c"].append(conf)
            L["x0"], L["y0"] = min(L["x0"], x), min(L["y0"], y)
            L["x1"], L["y1"] = max(L["x1"], x + w), max(L["y1"], y + h)
        out = []
        for L in lines.values():
            if self.fix_numbers and "eng" in self._have:
                try:
                    fix_line_words(self, image, L)
                except Exception as e:                       # التحسين ما يوقف الصفحة
                    log.debug("number re-read failed: %s", e)
            out.append(OCRLine(" ".join(w[0] for w in L["words"]), (L["x0"], L["y0"], L["x1"], L["y1"]),
                               (sum(L["c"]) / len(L["c"]) / 100.0) if L["c"] else 0.0))
        return out

    def read(self, image, lang: str, psm: int) -> list[tuple]:
        """قراءة صورة صغيرة (سطر/كلمة) -> [(نص، (x0,y0,x1,y1)، ثقة)]."""
        d = self._pt.image_to_data(image, lang=lang, config=f"--oem 1 --psm {psm}",
                                   output_type=self._pt.Output.DICT)
        return [((d["text"][i] or "").strip(), (d["left"][i], d["top"][i], d["left"][i] + d["width"][i],
                 d["top"][i] + d["height"][i]), float(d["conf"][i]))
                for i in range(len(d["text"])) if (d["text"][i] or "").strip()]


# ---------------------------------------------------------------------------
_LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9\u0660-\u0669]")
_AR_LETTER = re.compile(r"[\u0621-\u064A]")
_STRONG_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.,/%:\-+()]*$")
_EDGE_PUNCT = "/,.:;|()[]{}«»\"'`’‘°©_~"


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(c in it for c in small)


def _clean_eng_token(t: str) -> str:
    m = re.search(r"(www\.[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[a-z]{2,4})", t)
    if m:
        return m.group(1)                                  # "powww.blominvest.sa" -> "www.blominvest.sa"
    edge = _EDGE_PUNCT.replace("(", "").replace(")", "")
    t = t.strip(edge)
    # الأقواس: نحذفها من الطرف بس لو غير متوازنة ("(503" -> "503") ونبقيها في "011(4949555)"
    while t and t[0] in "([" and t.count("(") + t.count("[") > t.count(")") + t.count("]"):
        t = t[1:].strip(edge)
    while t and t[-1] in ")]" and t.count(")") + t.count("]") > t.count("(") + t.count("["):
        t = t[:-1].strip(edge)
    if t.startswith("%") and t[1:].replace(".", "").replace(",", "").isdigit():
        t = t[1:] + "%"                                   # "%100" (كما يظهر بصرياً) -> "100%"
    return t


def _overlap(a, b) -> float:
    """نسبة تداخل أفقي من عرض a."""
    inter = min(a[2], b[2]) - max(a[0], b[0])
    return max(0.0, inter) / max(1.0, a[2] - a[0])


def merge_number_tokens(ara_words: list, eng_tokens: list, rtl: bool = True) -> list:
    words = [list(w) for w in ara_words]
    for etext, ebox, econf in eng_tokens:
        t = _clean_eng_token(etext)
        if not t or not _STRONG_TOKEN.match(t) or _AR_LETTER.search(t):
            continue
        is_url = bool(re.search(r"(www\.|https?:|@)[A-Za-z0-9]|\.[a-z]{2,4}$", t)) and "." in t
        if not (_digits(t) or re.search(r"[A-Z]{2,}|[A-Za-z]\d", t) or is_url):
            continue                                    

        def suspicious(t: str) -> bool:
            return bool(_LATIN_OR_DIGIT.search(t)) or len(_AR_LETTER.findall(t)) < 2
        hits = [i for i, w in enumerate(words)
                if (suspicious(w[0]) and (_overlap(w[1], ebox) > 0.3 or _overlap(ebox, w[1]) > 0.3))
                or _overlap(w[1], ebox) > 0.6]
        if not hits:
            continue
        old = " ".join(words[i][0] for i in hits)
        if any(not suspicious(words[i][0]) for i in hits) and not _digits(old):
            continue                       
        ad, ed = _digits(old), _digits(t)
        latin_t = bool(re.search(r"[A-Za-z]", t))
        latin_old = bool(re.search(r"[A-Za-z]", old))
       
        if ed and not latin_t and sorted(ad) == sorted(ed):
            continue
        if ed and not latin_t and len(ad) >= 3 and ad in ed and len(ed) - len(ad) <= 1:
            continue                              
        recovers = bool(ad) and len(ad) <= 2 and _is_subsequence(ad, ed) and len(ed) > len(ad)
        
        is_date = bool(re.fullmatch(r"\d{1,4}[/\-]\d{1,2}[/\-]\d{1,4}", t))
        recovers = recovers or (is_date and bool(ad) and _is_subsequence(ad, ed) and len(ed) >= len(ad) + 2)
        is_code = bool(re.fullmatch(r"[A-Z][A-Za-z]{0,3}\d?[+\-]?", t))     # BBB- ، Baa3 ، AA+ ، A1
        old_no_ar = not _AR_LETTER.search(old)
        latin_fix = latin_t and (bool(ad) or latin_old) and (econf >= 80 or (is_code and old_no_ar and econf >= 50))
        url_fix = False
        if is_url and econf >= 40:
            sus = [i for i in hits if suspicious(words[i][0])]   # نستبدل بس الأرقام/الرموز، والكلمات العربية تبقى
            if sus:
                hits, url_fix = sus, True
        if recovers and econf >= 30 or latin_fix or url_fix:
            num_words = [words[i][0] for i in hits if _digits(words[i][0])]
            suffix = ""
            if len(num_words) == 1:
                m = re.search(r"[\u0621-\u064A]{1,2}$", num_words[0])
                suffix = m.group(0) if m else ""
            # "هـ" بجانب التاريخ تنقرى "2" بالإنجليزي (على يسار الرقم بصرياً): "21429/01/12" -> "1429/01/12"
            if suffix and re.fullmatch(r"\d\d{4}/\d{1,2}/\d{1,2}", t):
                t = t[1:]
            words[hits[0]][0] = t + suffix
            words[hits[0]][2] = max(words[hits[0]][2], econf)
            for i in reversed(hits[1:]):
                del words[i]
    return words


def fix_line_words(engine, image, L: dict) -> None:
    words = L["words"]
    text = " ".join(w[0] for w in words)
    if not _AR_LETTER.search(text):
        return                                             # سطر إنجليزي: ما نحتاج
    suspicious = [w for w in words if _LATIN_OR_DIGIT.search(w[0])]
    if not suspicious:
        return
    from PIL import Image, ImageOps
    x0, y0, x1, y1 = L["x0"], L["y0"], L["x1"], L["y1"]
    h = max(1, y1 - y0)
    scale = 2 if h < 40 else 1                             # النص الصغير يحتاج تكبير

    def read_with_pad(vpad: float) -> list:
        box = (max(0, x0 - int(0.35 * h)), max(0, y0 - int(vpad * h)),
               min(image.width, x1 + int(0.35 * h)), min(image.height, y1 + int(vpad * h)))
        crop = ImageOps.autocontrast(image.crop(box).convert("L"))
        if scale > 1:
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        return [(t, (box[0] + b[0] / scale, box[1] + b[1] / scale, box[0] + b[2] / scale, box[1] + b[3] / scale), c)
                for t, b, c in engine.read(crop, "eng", 7)]

    toks = []
    for t in read_with_pad(0.12) + read_with_pad(0.3):
        clash = [k for k, u in enumerate(toks) if _overlap(u[1], t[1]) > 0.5 or _overlap(t[1], u[1]) > 0.5]
        if not clash:
            toks.append(t)
        elif all(t[2] > toks[k][2] for k in clash):
            for k in reversed(clash):
                del toks[k]
            toks.append(t)
    L["words"] = merge_number_tokens(words, toks)

    ws = L["words"]
    for k, w in enumerate(ws):
        t = w[0]
        if _AR_LETTER.search(t) or _digits(t) or not re.search(r"[A-Za-z]", t):
            continue
        core = t.strip(_EDGE_PUNCT)
        if re.search(r"[./@:\\]", core) or len(core) > 6:
            continue                                       # رابط/إيميل/كلمة إنجليزية طويلة (www.blominvest.sa)
        neighbours = [ws[j][0] for j in (k - 1, k + 1) if 0 <= j < len(ws)]
        if any(re.fullmatch(r"[A-Za-z][A-Za-z.\-]*", n.strip(_EDGE_PUNCT) or "-") for n in neighbours):
            continue                                       # عبارة إنجليزية ("Blominvest IPO Fund")
        if any(_overlap(w[1], b) > 0.5 and c >= 85 and len(tt.strip(_EDGE_PUNCT)) >= 4 for tt, b, c in toks
               if re.fullmatch(r"[A-Za-z][A-Za-z.\-]+", tt.strip(_EDGE_PUNCT) or "-")):
            continue                                       # كلمة إنجليزية حقيقية بثقة عالية (Blominvest)
        wx0, wy0, wx1, wy1 = w[1]
        wh = max(1, wy1 - wy0)
        wb = (max(0, wx0 - int(0.4 * wh)), max(0, wy0 - int(0.3 * wh)),
              min(image.width, wx1 + int(0.4 * wh)), min(image.height, wy1 + int(0.3 * wh)))
        wc = ImageOps.autocontrast(image.crop(wb).convert("L"))
        if wh < 40:
            wc = wc.resize((wc.width * 2, wc.height * 2), Image.LANCZOS)
        res = [(tt, c) for tt, _, c in engine.read(wc, "ara", 8) if _AR_LETTER.search(tt)]
        if res and max(c for _, c in res) >= 50:
            w[0] = " ".join(tt for tt, _ in res)


# ---------------------------------------------------------------------------
_cache: dict = {}
_failed: set = set()


def get_engine(cfg: Config) -> OCREngine | None:
    order = {"auto": ["surya", "tesseract", "paddle", "easyocr"], "surya": ["surya"], "paddle": ["paddle"],
             "easyocr": ["easyocr"], "tesseract": ["tesseract"], "none": []}.get(cfg.ocr_engine, [])
    for name in order:
        key = (name, cfg.paddle_lang, cfg.tesseract_lang)
        if key in _cache:
            return _cache[key]
        if key in _failed:
            continue
        try:
            eng = (PaddleEngine(cfg.paddle_lang) if name == "paddle" else
                   EasyOCREngine() if name == "easyocr" else
                   SuryaEngine() if name == "surya" else TesseractEngine(cfg.tesseract_lang, cfg.tesseract_cmd))
            _cache[key] = eng
            log.info("OCR engine ready: %s", eng.name)
            return eng
        except OCRUnavailableError as e:
            _failed.add(key)
            log.warning("OCR engine '%s' unavailable: %s", name, e)
    return None


def preprocess(image, engine_name: str):
    from PIL import ImageOps
    if engine_name in ("tesseract", "easyocr"):
        return ImageOps.autocontrast(ImageOps.grayscale(image), cutoff=0)
    return ImageOps.autocontrast(image.convert("RGB"), cutoff=0)


# ---------------------------------------------------------------------------
def _is_rtl(lines: list[OCRLine]) -> bool:
    text = " ".join(l.text for l in lines)
    ar = len(re.findall(r"[\u0600-\u06FF]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    return ar > la


def lines_to_blocks(lines: list[OCRLine], img_w: float, img_h: float,
                    page_w: float, page_h: float) -> tuple[list[Block], float | None]:
    lines = [l for l in lines if l.text.strip()]
    if not lines:
        return [], None
    sx, sy = page_w / img_w, page_h / img_h
    items = [(l.bbox[0] * sx, l.bbox[1] * sy, l.bbox[2] * sx, l.bbox[3] * sy, l.text.strip(), l.conf) for l in lines]
    hs = sorted(i[3] - i[1] for i in items)
    med = max(hs[len(hs) // 2], 1.0)
    rtl = _is_rtl(lines)

    items.sort(key=lambda i: (i[1] + i[3]) / 2)
    rows: list[list] = []
    centers: list[float] = []
    for it in items:
        cy = (it[1] + it[3]) / 2
        if rows and abs(cy - centers[-1]) <= 0.6 * med:
            rows[-1].append(it)
            centers[-1] = sum((x[1] + x[3]) / 2 for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([it]); centers.append(cy)

    row_objs = []
    for r in rows:
        r.sort(key=lambda i: i[0], reverse=rtl)
        text = r[0][4]
        for prev, cur in zip(r, r[1:]):
            gap = (prev[0] - cur[2]) if rtl else (cur[0] - prev[2])
            text += (" | " if gap > 2.5 * med else " ") + cur[4]
        row_objs.append({"text": text, "x0": min(i[0] for i in r), "y0": min(i[1] for i in r),
                         "x1": max(i[2] for i in r), "y1": max(i[3] for i in r),
                         "w": sum(len(i[4]) * i[5] for i in r), "n": sum(len(i[4]) for i in r),
                         "table": " | " in text})

    paras: list[dict] = []
    for ro in row_objs:
        if paras:
            p = paras[-1]
            if ro["table"] and p["table"] and (ro["y0"] - p["y1"]) < 2.5 * med:   # صفوف جدول متتالية = جدول واحد
                p["text"] += "\n" + ro["text"]
                p["x0"], p["y0"] = min(p["x0"], ro["x0"]), min(p["y0"], ro["y0"])
                p["x1"], p["y1"] = max(p["x1"], ro["x1"]), max(p["y1"], ro["y1"])
                p["w"] += ro["w"]; p["n"] += ro["n"]
                continue
            new = (ro["table"] or p["table"] or starts_new_block(ro["text"])
                   or (ro["y0"] - p["y1"]) > 0.7 * med)
        else:
            new = True
        if new:
            paras.append(dict(ro, text=ro["text"]))
        else:
            p = paras[-1]
            p["text"] += " " + ro["text"]
            p["x0"], p["y0"] = min(p["x0"], ro["x0"]), min(p["y0"], ro["y0"])
            p["x1"], p["y1"] = max(p["x1"], ro["x1"]), max(p["y1"], ro["y1"])
            p["w"] += ro["w"]; p["n"] += ro["n"]

    blocks = []
    for p in paras:
        bb = (p["x0"], p["y0"], p["x1"], p["y1"])
        conf = p["w"] / max(1, p["n"])
        if p["table"] and "\n" in p["text"]:                  # سطرين أو أكثر بأعمدة = جدول
            rows = [r.split(" | ") for r in p["text"].split("\n")]
            blocks.append(Block(p["text"], bb, "table", conf, rows=rows))
        else:
            blocks.append(Block(p["text"], bb, "text", conf))
    tot = sum(i[5] * len(i[4]) for i in items)
    cnt = sum(len(i[4]) for i in items)
    return blocks, (tot / cnt if cnt else None)
