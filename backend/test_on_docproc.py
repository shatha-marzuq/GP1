import json, statistics, sys
from collections import Counter
from segmentation import segment_clauses

path = sys.argv[1] if len(sys.argv) > 1 else "../docproc/out/test.json"
data = json.load(open(path, encoding="utf-8"))
items = data["items"]
print("items:", len(items), dict(Counter(i["type"] for i in items)))
print("with marker:", sum(1 for i in items if i.get("marker")))

clauses = segment_clauses(items)
words = [c["word_count"] for c in clauses]
print("clauses:", len(clauses),
      "| front matter:", sum(c["is_front_matter"] for c in clauses),
      "| median words:", statistics.median(words),
      "| max words:", max(words),
      "| oversize:", sum(c["oversize"] for c in clauses))

# فحوصات تلقائية
ids = [i for c in clauses for i in c["item_ids"]]
src = [i["id"] for i in items if (i.get("text") or "").strip() or i.get("table")]
print("ما ضاع ولا تكرر أي عنصر:", ids == src)
print("كل بند له صفحة:", all(c["page_start"] for c in clauses))
print("الصفحات مرتبة:", all(c["page_start"] <= c["page_end"] for c in clauses))

print()
for c in clauses[:20]:
    tag = "مقدمة" if c["is_front_matter"] else ""
    print(c["clause_id"], c["marker"], f'ص{c["page_start"]}-{c["page_end"]}',
          f'{c["word_count"]}كلمة', f'جزء {c["part"]}/{c["parts"]}', tag, "|", c["title"][:40])
