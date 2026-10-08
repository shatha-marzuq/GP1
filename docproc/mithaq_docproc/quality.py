"""تقييم جودة النص المستخرج (بدون الحاجة لنص مرجعي).

يكشف: نص فارغ، رموز تالفة، نص عربي معكوس، حروف مبعثرة (علامة OCR سيئ).
تُستخدم لاتخاذ قرار: هل نكتفي بالنص المستخرج أم نشغّل OCR؟ وهل الصفحة تحتاج مراجعة؟
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_DIAC = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_AR_TOKEN = re.compile(r"[\u0600-\u06FF]+")

# كلمات شائعة في اللوائح (بدون كلمات تتحول لكلمة صحيحة لو انعكست)
_STOP = {"في", "من", "على", "هذا", "التي", "الذي", "عن", "بين", "ذلك", "المادة",
         "الصندوق", "الاستثمار", "الهيئة", "المستثمر", "الى", "الصناديق", "المالية"}
_REV = {w[::-1] for w in _STOP}


@dataclass
class QualityReport:
    score: float
    flags: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def _norm_token(tok: str) -> str:
    tok = _DIAC.sub("", tok)
    return re.sub("[إأآٱ]", "ا", tok).replace("ى", "ي")


def assess_quality(text: str) -> QualityReport:
    raw = text or ""
    t = unicodedata.normalize("NFKC", raw)
    compact = re.sub(r"\s+", "", t)
    n = len(compact)
    if n == 0:
        return QualityReport(0.0, ["EMPTY"], {"chars": 0})

    flags: list[str] = []
    letters_digits = sum(1 for c in compact if unicodedata.category(c)[0] in "LN")
    garbage = sum(1 for c in compact
                  if c == "\ufffd" or unicodedata.category(c) in ("Cc", "Co", "Cs", "Cn"))
    pres = sum(1 for c in re.sub(r"\s+", "", raw)
               if "\ufb50" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff")
    garbage_ratio = garbage / n
    letter_ratio = letters_digits / n
    pres_ratio = pres / max(1, len(re.sub(r"\s+", "", raw)))

    tokens = t.split()
    singles = [x for x in tokens if len(x) == 1 and unicodedata.category(x)[0] == "L"
               and x not in "وأإاAaIi"]
    single_ratio = len(singles) / len(tokens) if len(tokens) >= 10 else 0.0

    ar_tokens = [_norm_token(x) for x in _AR_TOKEN.findall(t)]
    normal_hits = sum(1 for x in ar_tokens if x in _STOP)
    rev_hits = sum(1 for x in ar_tokens if x in _REV)
    reversed_ar = rev_hits >= 3 and rev_hits > normal_hits

    score = 1.0
    score -= min(0.6, garbage_ratio * 6)
    if garbage_ratio > 0.02:
        flags.append("GARBAGE_CHARS")
    if reversed_ar:
        score -= 0.5
        flags.append("REVERSED_ARABIC")
    if single_ratio > 0.10:
        score -= min(0.4, (single_ratio - 0.10) * 1.5)
        if single_ratio > 0.25:
            flags.append("FRAGMENTED_TOKENS")
    if n >= 30 and letter_ratio < 0.55:
        score -= 0.3
        flags.append("LOW_LETTER_RATIO")
    if pres_ratio > 0.3 and reversed_ar:
        # أشكال العرض العربية بحد ذاتها ليست مشكلة: الترتيب الهندسي (bidi.py) يحل الترتيب البصري
        # والتنظيف (NFKC) يحوّلها لحروف عادية. المشكلة فقط لو النص طلع معكوساً رغم ذلك.
        score -= 0.45
        flags.append("LEGACY_ARABIC_ENCODING")

    return QualityReport(
        round(max(0.0, min(1.0, score)), 3), flags,
        {"chars": n, "letter_ratio": round(letter_ratio, 3), "garbage_ratio": round(garbage_ratio, 4),
         "single_char_ratio": round(single_ratio, 3), "reversed_hits": rev_hits, "normal_hits": normal_hits},
    )


# ---------------------------------------------------------------------------
# مقارنة نصين (CER / WER): تُستخدم في evaluate وفي التحقق Native-vs-OCR
# ---------------------------------------------------------------------------
try:
    from rapidfuzz.distance import Levenshtein as _Lev

    def _dist(a, b):
        return _Lev.distance(a, b)
except ImportError:                                           # بديل بطيء لكن يشتغل
    def _dist(a, b):
        prev = list(range(len(b) + 1))
        for i, x in enumerate(a, 1):
            cur = [i]
            for j, y in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
            prev = cur
        return prev[-1]


def _norm_cmp(t: str) -> str:
    from .cleaning import normalize_for_search
    t = normalize_for_search(t).replace("|", " ")             # فواصل الجداول
    return re.sub(r"\s+", " ", t).strip()


def cer(reference: str, hypothesis: str) -> float:
    """نسبة خطأ الحروف (0 = مطابق). المقارنة بعد تبسيط النص."""
    r, h = _norm_cmp(reference), _norm_cmp(hypothesis)
    return _dist(r, h) / max(1, len(r))


def wer(reference: str, hypothesis: str) -> float:
    r, h = _norm_cmp(reference).split(), _norm_cmp(hypothesis).split()
    vocab = {w: chr(0x4E00 + i) for i, w in enumerate(set(r) | set(h))}
    return _dist("".join(vocab[w] for w in r), "".join(vocab[w] for w in h)) / max(1, len(r))
