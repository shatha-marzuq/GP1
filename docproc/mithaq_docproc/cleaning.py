"""تنظيف النص.

مبدأ مهم: نسختان من النص
  1) clean_text()            -> نسخة وفيّة للأصل (تُعرض للمستخدم وتُستخدم في الاقتباس كدليل مخالفة).
                               ما نحذف تشكيل ولا نوحّد حروف، عشان الاقتباس يطابق الوثيقة.
  2) normalize_for_search()  -> نسخة مبسّطة للبحث/الـ embeddings/القواعد (بدون تشكيل، ألف موحدة، أرقام لاتينية).
"""
from __future__ import annotations

import re
import unicodedata

_INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff\u00ad\u061c]")
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_SPACES = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+")

_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")})
_DIGIT_MAP.update({ord("٫"): ".", ord("٬"): ","})


_LONE_YAA = re.compile(r"([\u0621-\u064A]+) ([يى])(?=[\s.،؛:!؟)\]]|$)")


_PERSIAN = str.maketrans({"\u06CC": "ي", "\u06A9": "ك", "\u06C0": "ة", "\u06BE": "ه", "\u06C1": "ه", "\u06D2": "ي"})


def clean_text(text: str) -> str:
    """تنظيف وفيّ: توحيد الأشكال، حذف الأحرف الخفية، ترتيب المسافات والأسطر."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n").replace("\u2029", "\n\n")
    t = _INVISIBLE.sub("", t)
    t = _CONTROL.sub("", t)
    t = t.replace("\u0640", "")                      # التطويل
    t = t.translate(_PERSIAN)                        # OCR أحياناً يطلّع حروف فارسية بنفس الشكل ("سیاسات ترکز")
    t = re.sub(r"(?:\s*[.…]){4,}\s*", " ... ", t)     # نقاط الفهرس ........ 
    t = _SPACES.sub(" ", t)
    t = re.sub(r" +([،؛؟])", r"\1", t)               # مسافة قبل علامة الترقيم العربية
    t = _LONE_YAA.sub(r"\1\2", t)                    # "الرئيس ي" -> "الرئيسي" (مسافة من Word داخل الكلمة)
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def normalize_for_search(text: str) -> str:
    """نسخة للبحث والمقارنة (ليست للعرض)."""
    t = clean_text(text)
    t = _DIACRITICS.sub("", t)
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي")
    return t.translate(_DIGIT_MAP)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
_MARKER_RE = re.compile(
    r"""^(?:
        \d{1,3}(?:\.\d{1,3})+\s+(?=\S)                       |  # 6.3.2 وسيكون  (رقم متعدد المستويات + نص)
        \d{1,3}(?:\.\d{1,3})*(?:[\)\.\-–:]\s+|\s+(?=[A-Z])|\s*$)  |  # 1.  1.2  1)  | 1 Introduction | رقم وحده بسطر
        \(\d{1,3}\)\s+                                         |  # (3)
        [\u2022\u25cf\u25aa\u25e6•●▪◦*\-–—]\s+                 |  # bullets
        \(?[A-Za-z]\)\s+ | [a-z]\.\s+                           |  # (a)  a.
        (?:article|section|chapter|clause|part|appendix)\s+\w+  |
        (?:المادة|مادة|الباب|الفصل|القسم|البند|الفقرة|الملحق)\s  |
        (?:اولا|ثانيا|ثالثا|رابعا|خامسا|سادسا|سابعا|ثامنا|تاسعا|عاشرا)\b |
        \(?[ا-ي][\)\.\-]\s
    )""",
    re.X | re.I,
)


def _for_match(line: str) -> str:
    t = unicodedata.normalize("NFKC", line)
    t = _DIACRITICS.sub("", t)
    return re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").translate(_DIGIT_MAP).strip()


def starts_new_block(line: str) -> bool:
    return bool(_MARKER_RE.match(_for_match(line)))


def join_lines(lines: list[str]) -> list[str]:
    paras: list[str] = []
    cur = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            if cur:
                paras.append(cur)
                cur = ""
            continue
        if not cur:
            cur = line
        elif starts_new_block(line):
            paras.append(cur)
            cur = line
        elif re.search(r"[A-Za-z]-$", cur):          # كلمة مركبة مكسورة عند الشرطة: نبقي الشرطة
            cur += line
        else:
            cur += " " + line
    if cur:
        paras.append(cur)
    return paras


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
_AR_WORD = re.compile(r"[\u0621-\u064A\u064B-\u0652\u0670]+")
_SPLIT_PAIR = re.compile(r"(?<![\u0621-\u064A\u064B-\u0652])([\u0621-\u064A\u064B-\u0652\u0670]+) ([\u0621-\u064A\u064B-\u0652\u0670]+)(?![\u0621-\u064A\u064B-\u0652])")
_STANDALONE_OK = {"و"}      
_SHORT_WORDS = {
    "في", "من", "عن", "علي", "الي", "او", "ان", "لا", "ما", "قد", "لم", "لن", "هو", "هي", "هم", "مع", "ثم",
    "كل", "اي", "بل", "لو", "اذ", "اذا", "تم", "اما", "الا", "اول", "غير", "بين", "عند", "بعد", "قبل", "حتي",
    "ذلك", "تلك", "هذا", "هذه", "التي", "الذي", "كان", "له", "لها", "لهم", "به", "بها", "منه", "منها", "فيه",
    "فيها", "عليه", "حال", "يوم", "عام", "سنة", "مدة", "حق", "حد", "نص", "رقم", "اسم", "عدد", "نوع", "اذن",
}


def _core(w: str) -> str:
    return _for_match(w)


_TOKEN_SPLIT = re.compile(r"(\s+)")


def _is_ar(tok: str) -> bool:
    return bool(tok) and bool(re.fullmatch(r"[\u0621-\u064A\u064B-\u0652\u0670]+", tok))


def build_vocab(texts) -> tuple:
    from collections import Counter
    words, pairs = Counter(), Counter()
    for t in texts:
        words.update(_core(w) for w in _AR_WORD.findall(t))
        parts = _TOKEN_SPLIT.split(t)
        for i in range(0, len(parts) - 2, 2):
            a, sep, b = parts[i], parts[i + 1], parts[i + 2]
            if sep == " " and _is_ar(a) and _is_ar(_strip_punct(b)):
                pairs[(_core(a), _core(_strip_punct(b)))] += 1
    return words, pairs


def _strip_punct(tok: str) -> str:
    return re.sub(r"[.,،:;؛!؟)\]»\"”']+$", "", tok)


_PREFIXES = ("وبال", "وال", "فال", "بال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك")


def _known(word: str, words, split_at: int | None = None) -> int:
    
    n = words.get(word, 0)
    if n:
        return n
    for p in _PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 3 and (split_at is None or split_at > len(p)):
            n = words.get(word[len(p):], 0)
            if n:
                return n
    return 0


def repair_split_words(text: str, vocab: tuple) -> tuple[str, list]:
    
    words, pairs = vocab
    fixes = []
    parts = _TOKEN_SPLIT.split(text)           
    i = 0
    while i + 2 < len(parts):
        a, sep, b_full = parts[i], parts[i + 1], parts[i + 2]
        b = _strip_punct(b_full)
        if sep == " " and _is_ar(a) and _is_ar(b):
            ca, cb = _core(a), _core(b)
            joined = ca + cb
            n_pair = pairs.get((ca, cb), 0)
        
            def fake(c: str) -> bool:
                if c in _STANDALONE_OK or c in _SHORT_WORDS:
                    return False
                return len(c) == 1 or (len(c) <= 3 and words.get(c, 0) <= n_pair)
            a_fake, b_fake = fake(ca), fake(cb)
           
            if words.get(ca + cb, 0) == 0:
                a_fake = a_fake and len(ca) <= 2
                b_fake = b_fake and len(cb) <= 2
            
            if _known(joined, words, len(ca)) >= max(1, n_pair) and (a_fake or b_fake):
                fixes.append(f"{a} {b} -> {a}{b}")
                parts[i:i + 3] = [a + b_full]          
                continue
            if b == b_full and i + 4 < len(parts) and parts[i + 3] == " ":
                c_full = parts[i + 4]
                c = _strip_punct(c_full)
                if _is_ar(c):
                    cc = _core(c)
                    if _known(ca + cb + cc, words, len(ca)) >= max(1, n_pair) and (
                            a_fake or b_fake or (len(cc) == 1 and cc not in _STANDALONE_OK)):
                        fixes.append(f"{a} {b} {c} -> {a}{b}{c}")
                        parts[i:i + 5] = [a + b + c_full]
                        continue
        i += 2
    return "".join(parts), fixes
