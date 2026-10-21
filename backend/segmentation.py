import os
import re
import sys

this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(this_folder, "..", "pipeline"))
# use the same normalize function as the regulations database
from chunk import search_normalize

MAX_WORDS = 300

ABJAD = ["أ", "ب", "ج", "د", "ه", "و", "ز", "ح", "ط", "ي", "ك", "ل", "م", "ن",
         "س", "ع", "ف", "ص", "ق", "ر", "ش", "ت", "ث", "خ", "ذ", "ض", "ظ", "غ"]

ARABIC_LETTERS = ["أ", "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س", "ش",
                  "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"]

# the ways a list number can be written
LETTER = "[" + "".join(ABJAD) + "]"
NUMBER_OR_LETTER = r"(\d{1,2}|" + LETTER + r")"
MARKER_STYLES = [
    ("paren", r"^\(\s*" + NUMBER_OR_LETTER + r"\s*\)"),
    ("paren", r"^\)\s*" + NUMBER_OR_LETTER + r"\s*\("),
    ("close", r"^" + NUMBER_OR_LETTER + r"\s*[)(]"),
    ("dot", r"^" + NUMBER_OR_LETTER + r"\s*\.(?!\d)"),
    ("dash", r"^" + NUMBER_OR_LETTER + r"\s*[-–]"),
    ("dash", r"^[-–]\s*" + NUMBER_OR_LETTER + r"(?=\s)"),
    ("plain", r"^(\d{1,2})\s+(?=[^\d\s.)])"),
]


# get the text of an item, if it is a table it becomes rows of text
def get_item_text(item, key="text"):
    if item.get("type") == "table" and item.get("table"):
        lines = []
        for row in item["table"]["rows"]:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    text = item.get(key)
    if not text:
        text = item.get("text")
    if not text:
        text = ""
    return text.strip()


def count_words(text):
    return len(text.split())


def first_line(text):
    return text.split("\n")[0][:120]


# read the list number at the start of a text ex: "ب- عنوان" -> ("letter", "dash", 2)
def read_marker(text):
    for style, pattern in MARKER_STYLES:
        match = re.match(pattern, text)
        if match:
            value = match.group(1)
            if value.isdigit():
                return ("number", style, int(value))
            return ("letter", style, ABJAD.index(value) + 1)
    return None


# lines from the table of contents
def is_contents_line(text):
    return bool(re.search(r"\.{4,}|-{4,}|…", text) or re.search(r"\s\d{1,3}$", text))


# clause title does not end with a full stop, a list item usually does
def is_sentence(text):
    return text.rstrip().endswith((".", "؛"))


# find the main clauses of the document, we look for a list of titles written in the same way that covers most of the file
def find_main_clauses(items):
    candidates = []
    for index in range(len(items)):
        item = items[index]
        text = get_item_text(item)
        if item.get("type") == "table" or not text:
            continue
        if is_contents_line(text):
            continue
        marker = read_marker(text)
        if marker:
            # a short line that does not end like a sentence looks like a clause title
            # a sentence ex: "1) التغيير المهم في أهداف الصندوق." is usually a point inside a list
            is_title = count_words(text) <= 15 and not is_sentence(text)
            candidates.append((index, marker, is_title))

    best_chain = []
    best_style = None
    best_score = 0
    for a in range(len(candidates)):
        start_index, start_marker, start_is_title = candidates[a]
        if start_marker[2] != 1 or not start_is_title:
            continue

        # after 1 we look for 2 (we allow up to 2 missing numbers)
        chain = [(start_index, 1)]
        expected = 2
        position = a + 1
        while position < len(candidates):
            # a title with exactly the next number
            # or the first of a long line / sentence with exactly the next number
            # or a title that skips up to 2 missing numbers
            title_exact = None
            title_skip = None
            not_title_exact = None
            for b in range(position, len(candidates)):
                index, marker, is_title = candidates[b]
                if marker[0] != start_marker[0] or marker[1] != start_marker[1]:
                    continue
                if is_title and marker[2] == expected and title_exact is None:
                    title_exact = b
                if is_title and expected < marker[2] <= expected + 2 and title_skip is None:
                    title_skip = b
                if not is_title and marker[2] == expected and not_title_exact is None:
                    not_title_exact = b

            # a title with the next number wins, unless a title with a bigger number comes before it
            if title_exact is not None and (title_skip is None or title_exact < title_skip):
                found = title_exact
            elif title_skip is not None and not_title_exact is not None:
                found = min(title_skip, not_title_exact)
            elif title_skip is not None:
                found = title_skip
            elif not_title_exact is not None:
                found = not_title_exact
            else:
                break
            index, marker, is_title = candidates[found]
            chain.append((index, marker[2]))
            expected = marker[2] + 1
            position = found + 1

        score = len(chain) * (chain[-1][0] - chain[0][0])
        if len(chain) >= 5 and score > best_score:
            best_score = score
            best_chain = chain
            best_style = start_marker[:2]

    if len(best_chain) < 5:
        return {}, None

    main = {}
    for index, number in best_chain:
        main[index] = number
    return main, best_style


def is_annex_heading(item):
    text = get_item_text(item)
    return item.get("type") == "heading" and (text.startswith("الملحق") or text.startswith("ملحق"))


# used to find glued titles
def label_pattern(style, number):
    kind, how = style
    if kind == "letter":
        value = ABJAD[number - 1] if number <= len(ABJAD) else "?"
    else:
        value = str(number)
    if how == "plain":
        return value + r"\s+"
    if how == "dot":
        return value + r"\s*\.\s*"
    if how == "close":
        return value + r"\s*[)(]\s*"
    if how == "paren":
        return r"\(\s*" + value + r"\s*\)\s*"
    return r"(?:-\s*" + value + r"|" + value + r"\s*-)\s*"


# fix a clause title that is glued to the end of the paragraph before it
# after_title=True: the item is itself a clause title, and the next title is glued to it
def split_glued_title(item, style, next_number, after_title=False):
    text = get_item_text(item)
    # the title can end with ":"
    title_part = r"(\S[^.:؛]{1,60}(?::\s*)?)$"
    if after_title:
        pattern = r"^(.*\S)\s+(" + label_pattern(style, next_number) + r")" + title_part
    else:
        pattern = r"^(.*[.:؛)])\s*(" + label_pattern(style, next_number) + r")" + title_part
    match = re.match(pattern, text, re.DOTALL)
    if not match:
        return None

    title = match.group(3).strip()
    if count_words(title) > 8:
        return None
    if after_title and count_words(match.group(1)) > 15:
        return None

    before = dict(item)
    before["text"] = match.group(1).strip()
    before["text_norm"] = before["text"]

    title_item = dict(item)
    title_item["id"] = item["id"] + "-t"
    title_item["type"] = "heading"
    title_item["marker"] = match.group(2).strip()
    title_item["level"] = 1
    title_item["text"] = match.group(2).strip() + " " + title
    title_item["text_norm"] = title_item["text"]

    return before, title_item


# sometimes the OCR puts a clause title inside a table, as a row with only one filled cell, we take it out of the table
def split_table_title(item, style, next_number):
    if item.get("type") != "table" or not item.get("table"):
        return None
    rows = item["table"]["rows"]
    pattern = r"^(" + label_pattern(style, next_number) + r")(\S.*)$"

    for r in range(len(rows)):
        cells = [cell.strip() for cell in rows[r] if cell.strip()]
        if len(cells) != 1:
            continue
        match = re.match(pattern, cells[0])
        if not match or count_words(cells[0]) > 15 or is_sentence(cells[0]):
            continue

        title_item = dict(item)
        title_item["id"] = item["id"] + "-t"
        title_item["type"] = "heading"
        title_item["table"] = None
        title_item["marker"] = match.group(1).strip()
        title_item["level"] = 1
        title_item["text"] = cells[0]
        title_item["text_norm"] = cells[0]

        # the rows before the title stay with the clause before, the rows after go with the new clause
        before = None
        if r > 0:
            before = dict(item)
            before["table"] = dict(item["table"])
            before["table"]["rows"] = rows[:r]
        after = None
        if r < len(rows) - 1:
            after = dict(item)
            after["id"] = item["id"] + "-b"
            after["table"] = dict(item["table"])
            after["table"]["rows"] = rows[r + 1:]
        return before, title_item, after
    return None


def new_section(kind, number=None):
    return {"kind": kind, "number": number, "items": []}


# sub clause starts a new unit (used when we split long clauses)
def starts_unit(item):
    if item.get("type") == "heading":
        return True
    marker = item.get("marker") or ""
    if re.fullmatch(r"\d+\.\d+", marker):
        return True
    if len(marker) == 1 and marker in ARABIC_LETTERS:
        return True
    text = get_item_text(item)
    if marker.isdigit() and re.match(r"^\d+\s*\.", text):
        return True
    found = read_marker(text)
    if found and found[0] == "letter":
        return True
    return False


def units_of(items):
    units = []
    for item in items:
        if len(units) == 0 or starts_unit(item):
            units.append([item])
        else:
            units[-1].append(item)
    return units


def unit_words(unit):
    total = 0
    for item in unit:
        total += count_words(get_item_text(item))
    return total


def split_sentences(text, max_words):
    sentences = re.split(r"(?<=[.؛])\s+", text)
    pieces = []
    current = ""
    for s in sentences:
        if current and count_words(current) + count_words(s) > max_words:
            pieces.append(current)
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        pieces.append(current)

    # a piece with no end of sentence can still be too long, then we cut it by words
    final = []
    for piece in pieces:
        words = piece.split()
        if len(words) <= max_words:
            final.append(piece)
        else:
            for k in range(0, len(words), max_words):
                final.append(" ".join(words[k:k + max_words]))
    return final


# cut a unit that is too long
def break_big_unit(unit, max_words):
    small_units = []
    for item in unit:
        text = get_item_text(item)
        if count_words(text) <= max_words:
            small_units.append([item])
        elif item.get("type") == "table":
            rows = text.split("\n")

            # one row can also be too long, then we cut it by sentences
            short_rows = []
            for row in rows:
                if count_words(row) > max_words:
                    short_rows += split_sentences(row, max_words)
                else:
                    short_rows.append(row)
            rows = short_rows

            current_rows = []
            for row in rows:
                if current_rows and count_words("\n".join(current_rows + [row])) > max_words:
                    part = dict(item)
                    part["type"] = "table_part"
                    part["text"] = "\n".join(current_rows)
                    part["text_norm"] = part["text"]
                    small_units.append([part])
                    current_rows = []
                current_rows.append(row)
            if current_rows:
                part = dict(item)
                part["type"] = "table_part"
                part["text"] = "\n".join(current_rows)
                part["text_norm"] = part["text"]
                small_units.append([part])
        else:
            for piece in split_sentences(text, max_words):
                part = dict(item)
                part["text"] = piece
                part["text_norm"] = piece
                small_units.append([part])
    return small_units


# put units together into parts of the same size, each <= max_words
def pack_units(units, max_words):
    total = 0
    for unit in units:
        total += unit_words(unit)

    number_of_parts = -(-total // max_words)
    if number_of_parts > 1:
        limit = min(max_words, int(total / number_of_parts * 1.2) + 1)
    else:
        limit = max_words

    ready_units = []
    for unit in units:
        if unit_words(unit) > limit:
            ready_units += break_big_unit(unit, limit)
        else:
            ready_units.append(unit)

    parts = []
    current = []
    current_words = 0
    for unit in ready_units:
        words = unit_words(unit)
        if len(current) > 0 and current_words + words > limit:
            parts.append(current)
            current = []
            current_words = 0
        current += unit
        current_words += words
    if len(current) > 0:
        parts.append(current)
    return parts


# main function: items from docproc -> list of clauses
def segment_clauses(items, max_words=MAX_WORDS, doc_title="الشروط والأحكام", document=None, main_starts=None):
    items = [item for item in items if get_item_text(item)]

    # info about the uploaded file (from docproc)
    if document is None:
        document = {}
    source_file = document.get("filename")
    doc_id = os.path.splitext(source_file)[0] if source_file else None

    # find how this file writes its main clauses
    main, style = find_main_clauses(items)

    # the segmentation agent can give the clause starts itself
    if main_starts is not None:
        main = dict(main_starts)
        if style is None:
            for index in sorted(main):
                found = read_marker(get_item_text(items[index]))
                if found:
                    style = found[:2]
                    break

    # group the items by main clause
    sections = []
    current = new_section("front")
    sections.append(current)
    current_number = 0
    in_annex = False

    for index in range(len(items)):
        item = items[index]

        if index in main and not in_annex:
            current_number = main[index]
            current = new_section("clause", current_number)
            sections.append(current)

            # the next clause title can be glued to this title
            glued = split_glued_title(item, style, current_number + 1, after_title=True)
            if glued and (current_number + 1) not in main.values():
                before, title_item = glued
                current["items"].append(before)
                current_number += 1
                current = new_section("clause", current_number)
                sections.append(current)
                current["items"].append(title_item)
            else:
                current["items"].append(item)

        elif is_annex_heading(item) and current_number > 0:
            in_annex = True
            current = new_section("annex")
            sections.append(current)
            current["items"].append(item)

        else:
            glued = None
            in_table = None
            # only when the next clause title was not already found by find_main_clauses
            if style and not in_annex and current_number > 0 and (current_number + 1) not in main.values():
                glued = split_glued_title(item, style, current_number + 1)
                in_table = split_table_title(item, style, current_number + 1)

            if in_table:
                before, title_item, after = in_table
                if before:
                    current["items"].append(before)
                current_number += 1
                current = new_section("clause", current_number)
                sections.append(current)
                current["items"].append(title_item)
                if after:
                    current["items"].append(after)

            elif glued:
                before, title_item = glued
                current["items"].append(before)
                current_number += 1
                current = new_section("clause", current_number)
                sections.append(current)
                current["items"].append(title_item)
            else:
                current["items"].append(item)

    # split long clauses into parts
    result = []
    for section in sections:
        if len(section["items"]) == 0:
            continue

        section_title = first_line(get_item_text(section["items"][0]))
        parts = pack_units(units_of(section["items"]), max_words)

        part_number = 1
        for part in parts:
            header = "[" + doc_title + " › " + section_title
            if len(parts) > 1:
                header += " (الجزء " + str(part_number) + " من " + str(len(parts)) + ")"
            header += "]"

            texts = []
            norm_texts = []
            item_ids = []
            pages = []
            marker = None
            has_table = False
            from_ocr = False
            for item in part:
                texts.append(get_item_text(item))
                norm_texts.append(get_item_text(item, "text_norm"))
                item_ids.append(item["id"])
                if item.get("page") and item["page"] not in pages:
                    pages.append(item["page"])
                if marker is None and item.get("marker"):
                    marker = item["marker"]
                if item.get("type") in ("table", "table_part"):
                    has_table = True
                if item.get("source") == "ocr":
                    from_ocr = True

            content = "\n".join(texts)
            text = header + "\n" + content
            words = count_words(content)

            clause = {
                "clause_id": "C" + str(len(result) + 1).zfill(3),
                "section_type": section["kind"],
                "clause_number": section["number"],
                "marker": marker,
                "title": first_line(texts[0]),
                "section_title": section_title,
                "part": part_number,
                "parts": len(parts),
                "content": content,
                "text": text,
                "text_norm": "\n".join(norm_texts),
                "text_search": search_normalize(text),
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "pages": sorted(pages),
                "item_ids": item_ids,
                "word_count": words,
                "oversize": words > max_words,
                "has_table": has_table,
                "from_ocr": from_ocr,
                "is_front_matter": section["kind"] == "front",
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_type": "شروط وأحكام",
                "source_file": source_file,
                "citation": doc_title + "، " + section_title,
                "breadcrumb": header[1:-1],
                "char_count": len(text),
            }
            result.append(clause)
            part_number += 1

    return result
