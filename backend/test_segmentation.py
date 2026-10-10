import json
from segmentation import segment_clauses

data = json.load(open("items.json", encoding="utf-8"))
clauses = segment_clauses(data["items"])
print(len(clauses))