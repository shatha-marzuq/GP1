
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .cleaning import normalize_for_search

_WORD = re.compile(r"[\u0621-\u064A]{2,}")
_CONFUSABLE = ["بي"]
_SWAP: dict[str, str] = {}
for grp in _CONFUSABLE:
    for ch in grp:
        _SWAP[ch] = grp.replace(ch, "")


def words_of(text: str) -> Counter:
    return Counter(_WORD.findall(normalize_for_search(text)))


@lru_cache(maxsize=8)
def load_lexicon(dirs: tuple = ()) -> Counter:
    lex: Counter = Counter()
    seed = Path(__file__).with_name("data") / "lexicon_seed.txt"
    if seed.exists():
        for line in seed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "\t" in line:
                w, n = line.split("\t", 1)
                lex[w] += int(n)
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            continue
        for f in list(p.rglob("*.txt")) + list(p.rglob("*.md")):
            try:
                lex.update(words_of(f.read_text(encoding="utf-8", errors="ignore")))
            except Exception:
                continue
    return lex


_VERB_PREFIX = set("يتنا")


def _candidates(core: str) -> set:
    out = set()
    for i, ch in enumerate(core):
        for alt in _SWAP.get(ch, ""):
            
            if ch in _VERB_PREFIX and alt in _VERB_PREFIX and (i == 0 or (i == 1 and core[0] in "وفسل")):
                continue
            if i == 0 or (i == 1 and core[0] in "وفلكس"):
                continue
            out.add(core[:i] + alt + core[i + 1:])
    return out


def correct_ocr_text(text: str, known: Counter, min_count: int = 3) -> tuple[str, list]:
    fixes = []

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        core = normalize_for_search(tok)
        if len(core) < 3 or known.get(core, 0) >= 1:
            return tok
        cands = [c for c in _candidates(core) if known.get(c, 0) >= min_count]
        if len(cands) != 1:
            return tok                         # لا بديل أو أكثر من بديل: ما نخمّن
        target = cands[0]
        diff = [i for i, (a, b) in enumerate(zip(core, target)) if a != b]
        if len(diff) != 1 or len(tok) != len(core):
            return tok                         # الكلمة فيها تشكيل/همزات تغيّر الطول: نتركها بأمان
        i = diff[0]
        fixed = tok[:i] + target[i] + tok[i + 1:]
        fixes.append((tok, fixed))
        return fixed

    return re.sub(r"[\u0621-\u064A]{3,}", repl, text), fixes
