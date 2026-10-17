import json
import statistics
import sys
from collections import Counter

from segmentation import segment_clauses

# how to run:  python test_on_docproc.py ../docproc/test_items.json
path = sys.argv[1] if len(sys.argv) > 1 else "../docproc/test_items.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)
items = data["items"]
print("items:", len(items), dict(Counter(i["type"] for i in items)))

clauses = segment_clauses(items)
words = [c["word_count"] for c in clauses]
print("clauses:", len(clauses),
      "| intro:", sum(c["section_type"] == "front" for c in clauses),
      "| annex:", sum(c["section_type"] == "annex" for c in clauses),
      "| median words:", statistics.median(words),
      "| max words:", max(words),
      "| oversize:", sum(c["oversize"] for c in clauses))
print()

# ---------- automatic checks ----------

# 1) no item is lost and the order is the same
#    (ids ending with "-t" are titles we cut from a glued paragraph,
#     and a long item can be in two parts, so we remove repeats)
ids = []
for c in clauses:
    for i in c["item_ids"]:
        if i.endswith("-t"):
            continue
        if len(ids) == 0 or ids[-1] != i:
            ids.append(i)
source_ids = [i["id"] for i in items if (i.get("text") or "").strip() or i.get("table")]
print("no item lost, same order:", ids == source_ids)

# 2) main clause numbers come in order 1, 2, 3, ... with nothing missing
numbers = []
for c in clauses:
    if c["clause_number"] is not None and c["clause_number"] not in numbers:
        numbers.append(c["clause_number"])
expected = list(range(1, len(numbers) + 1))
print("main clauses found:", len(numbers), "| in order with none missing:", numbers == expected)
if numbers != expected:
    print("   found:", numbers)

# 3) pages
if any(c["page_start"] for c in clauses):
    print("every clause has a page:", all(c["page_start"] for c in clauses))
    print("pages in order:", all(c["page_start"] <= c["page_end"] for c in clauses if c["page_start"]))
else:
    print("no page numbers in this file (Word file without LibreOffice)")

# 4) titles that were glued to the paragraph before them (docproc problem)
glued = [i for c in clauses for i in c["item_ids"] if i.endswith("-t")]
if glued:
    print("titles fixed (glued to the paragraph before):", glued)

print()
for c in clauses:
    if c["section_type"] == "clause" and c["part"] == 1:
        print(c["clause_number"], "|", c["parts"], "parts | pages", c["page_start"], "-", c["page_end"],
              "|", c["section_title"][:45])
