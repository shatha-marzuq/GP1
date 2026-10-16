import re

# get the text of an item
# if the item is a table turn each row into one line
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


# count how many words are in an item
def count_words(item):
    text = get_item_text(item)
    return len(text.split())


# take the first line of the text to use as a title
def get_first_line(item):
    text = get_item_text(item)
    first_line = text.split("\n")[0]
    return first_line[:120]


# return True if every item in the group is a heading
def only_headings(group):
    for item in group:
        if item.get("type") != "heading":
            return False
    return True


# check if the item is on the same or higher level as the last item in the group
def is_sibling(item, group):
    # find the level of the last item that has a level
    last_level = None
    for i in reversed(group):
        if i.get("level"):
            last_level = i["level"]
            break

    if not last_level:
        return False
    if not item.get("level"):
        return False
    return item["level"] <= last_level


# split a long clause into smaller parts (only cut before heading or a numbered item)
def split_group(group, max_words):
    parts = []
    current_part = []
    word_count = 0

    for item in group:
        words = count_words(item)

        # cheack if item start a sub clause or no
        new_sub_clause = False
        if item.get("marker"):
            new_sub_clause = True
        if item.get("type") == "heading":
            new_sub_clause = True

        # if adding item makes the part too long close the part and start a new one
        if len(current_part) > 0 and new_sub_clause and word_count + words > max_words:
            parts.append(current_part)
            current_part = []
            word_count = 0

        current_part.append(item)
        word_count += words

    if len(current_part) > 0:
        parts.append(current_part)

    return parts


# main function: takes the OCR items and returns a list of clauses
# max_level: deepest numbered level that can start a new clause
# max_words: if a clause is longer than this we split it
def segment_clauses(items, max_level=3, max_words=300):

    # group the items each group is one clause
    groups = []
    current_group = None

    for item in items:
        # skip empty items
        if not get_item_text(item):
            continue

        # heading or numbered item can start a new clause
        starts_new_clause = False
        if item.get("type") == "heading":
            starts_new_clause = True
        elif item.get("marker") and item.get("level") and item["level"] <= max_level:
            starts_new_clause = True

        if current_group is None:
            current_group = []
            groups.append(current_group)

        # start a new group unless the group only has headings so the heading satys with its content except if it's a sibling

        elif starts_new_clause:
            if not only_headings(current_group) or is_sibling(item, current_group):
                current_group = []
                groups.append(current_group)

        current_group.append(item)

    # split long clauses into parts
    pieces = []
    for group in groups:
        total_words = 0
        for item in group:
            total_words += count_words(item)

        # all parts keep the title of the original clause
        section_title = get_first_line(group[0])

        if max_words and total_words > max_words:
            parts = split_group(group, max_words)
            part_number = 1
            for part in parts:
                pieces.append((part, section_title, part_number, len(parts)))
                part_number += 1
        else:
            pieces.append((group, section_title, 1, 1))

    # build the JSON object for each clause
    result = []
    clause_number = 1

    for piece in pieces:
        group = piece[0]
        section_title = piece[1]
        part_number = piece[2]
        total_parts = piece[3]

        # page numbers 
        pages = []
        for item in group:
            if item.get("page"):
                pages.append(item["page"])

        # clause number is the first marker we find
        marker = None
        for item in group:
            if item.get("marker"):
                marker = item["marker"]
                break

        # collect texts, ids, word count, and check for tables and OCR
        texts = []
        norm_texts = []
        item_ids = []
        words = 0
        has_table = False
        from_ocr = False
        for item in group:
            texts.append(get_item_text(item))
            norm_texts.append(get_item_text(item, "text_norm"))
            item_ids.append(item["id"])
            words += count_words(item)
            if item.get("type") == "table":
                has_table = True
            if item.get("source") == "ocr":
                from_ocr = True

        # first and last page of the clause
        if len(pages) > 0:
            page_start = min(pages)
            page_end = max(pages)
        else:
            page_start = None
            page_end = None

        # is it still too long even after splitting
        oversize = False
        if max_words and words > max_words:
            oversize = True

        clause = {
            "clause_id": "C" + str(clause_number).zfill(3),
            "marker": marker,
            "title": get_first_line(group[0]),
            "section_title": section_title,
            "part": part_number,
            "parts": total_parts,
            "text": "\n\n".join(texts),
            "text_norm": "\n\n".join(norm_texts),
            "page_start": page_start,
            "page_end": page_end,
            "item_ids": item_ids,
            "word_count": words,
            "oversize": oversize,
            "has_table": has_table,
            "from_ocr": from_ocr,
            "is_front_matter": False,
        }
        result.append(clause)
        clause_number += 1

    # clauses before the first numbered clause are the intro
    first_numbered = None
    for index in range(len(result)):
        marker = result[index]["marker"]
        if marker and re.fullmatch(r"\d+(\.\d+)*", marker):
            first_numbered = index
            break

    if first_numbered is not None:
        for index in range(first_numbered):
            result[index]["is_front_matter"] = True

    return result
