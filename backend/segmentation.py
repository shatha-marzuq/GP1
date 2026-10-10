import re

def _item_text(it, key="text"):
    if it.get("type") == "table" and it.get("table"):
        return "\n".join(" | ".join(r) for r in it["table"]["rows"])
    return (it.get(key) or it.get("text") or "").strip()

def _words(it):
    return len(_item_text(it).split())

def _split_group(g, max_words):
    parts, cur, n = [], [], 0
    for it in g:
        w = _words(it)
        sub_start = bool(it.get("marker")) or it.get("type") == "heading"
        if cur and sub_start and n + w > max_words:
            parts.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += w
    if cur:
        parts.append(cur)
    return parts

def segment_clauses(items, max_level=3, max_words=300):
    groups, cur = [], None

    def only_headings(g):
        return all(i.get("type") == "heading" for i in g)

    def is_sibling(it, g):
        last = next((i["level"] for i in reversed(g) if i.get("level")), None)
        return bool(last and it.get("level") and it["level"] <= last)

    for it in items:
        if not _item_text(it):
            continue
        starts = it.get("type") == "heading" or (
            it.get("marker") and it.get("level") and it["level"] <= max_level)
        if cur is None or (starts and (not only_headings(cur) or is_sibling(it, cur))):
            cur = []
            groups.append(cur)
        cur.append(it)

    pieces = []  
    for g in groups:
        total = sum(_words(i) for i in g)
        section_title = _item_text(g[0]).split("\n")[0][:120]
        if max_words and total > max_words:
            parts = _split_group(g, max_words)
            for k, p in enumerate(parts, 1):
                pieces.append((p, section_title, k, len(parts)))
        else:
            pieces.append((g, section_title, 1, 1))

    out = []
    for n, (g, section_title, k, K) in enumerate(pieces, 1):
        pages = [i["page"] for i in g if i.get("page")]
        words = sum(_words(i) for i in g)
        out.append({
            "clause_id": f"C{n:03d}",
            "marker": next((i["marker"] for i in g if i.get("marker")), None),
            "title": _item_text(g[0]).split("\n")[0][:120],
            "section_title": section_title,
            "part": k,
            "parts": K,
            "text": "\n\n".join(_item_text(i) for i in g),
            "text_norm": "\n\n".join(_item_text(i, "text_norm") for i in g),
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "item_ids": [i["id"] for i in g],
            "word_count": words,
            "oversize": bool(max_words and words > max_words),
            "has_table": any(i.get("type") == "table" for i in g),
            "from_ocr": any(i.get("source") == "ocr" for i in g),
            "is_front_matter": False,
        })

    first = next((i for i, c in enumerate(out)
                  if c["marker"] and re.fullmatch(r"\d+(\.\d+)*", c["marker"])), None)
    if first:
        for c in out[:first]:
            c["is_front_matter"] = True
    return out
