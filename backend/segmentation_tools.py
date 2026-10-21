# the tools that the segmentation agent can use
import os
import re
import sys

this_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, this_folder)
from segmentation import (MAX_WORDS, count_words, find_main_clauses, get_item_text,
                          is_contents_line, segment_clauses)

MIN_CLAUSES = 5         # less than this = the rules did not find the clauses
LONG_TITLE_WORDS = 15   # a clause title longer than this looks wrong
BIG_FRONT_SHARE = 0.4   # front matter bigger than 40% of the file looks wrong
BIG_CLAUSE_SHARE = 0.35 # one clause bigger than 35% of the file looks wrong
MAX_VIEW_LINES = 40
MAX_SEARCH_RESULTS = 20


def short(text, size=200):
    text = text.replace("\n", " ")
    if len(text) > size:
        return text[:size] + " …"
    return text


# the title part of a line
def title_part(text):
    line = text.split("\n")[0]
    if ":" in line:
        line = line[:line.index(":") + 1]
    return line


class SegmentationSession:
    def __init__(self, items, document=None, max_words=MAX_WORDS):
        # same as segment_clauses so the item numbers are the same everywhere
        self.items = [item for item in items if get_item_text(item)]
        self.document = document or {}
        self.max_words = max_words

        # the clause starts the rules found: {item index: clause number}
        main, style = find_main_clauses(self.items)
        self.first_starts = dict(main)
        self.starts = dict(main)

        self.changes = {}  # clause number
        self.steps = []    # what the agent did (for the report)
        self.finished = None
        self.first_problems = []  # the problems before the agent fixed anything
        self.clauses = []
        self.run_segmentation()


    # --------helpers--------

    def run_segmentation(self):
        self.clauses = segment_clauses(self.items, max_words=self.max_words,
                                       document=self.document, main_starts=self.starts)

    def index_of(self, item_id):
        for index in range(len(self.items)):
            if self.items[index]["id"] == item_id:
                return index
        # a split item (glued title) has an id
        for index in range(len(self.items)):
            if item_id.startswith(self.items[index]["id"]):
                return index
        return None

    def outline(self):
        sections = []
        for clause in self.clauses:
            if clause["part"] != 1:
                sections[-1]["words"] += clause["word_count"]
                continue
            sections.append({
                "type": clause["section_type"],
                "number": clause["clause_number"],
                "title": short(clause["section_title"], 120),
                "first_line": self.index_of(clause["item_ids"][0]),
                "pages": [clause["page_start"], clause["page_end"]],
                "parts": clause["parts"],
                "words": clause["word_count"],
            })
        return sections

    # --------tools--------

    def segment_document(self):
        outline = self.outline()
        numbers = [s["number"] for s in outline if s["type"] == "clause"]
        return {
            "total_lines": len(self.items),
            "main_clauses": len(numbers),
            "outline": outline,
        }

    def check_segmentation(self):
        problems = []
        outline = self.outline()
        clauses = [s for s in outline if s["type"] == "clause"]
        total_words = sum(s["words"] for s in outline) or 1

        if len(clauses) < MIN_CLAUSES:
            problems.append({"problem": "NO_CLAUSES",
                             "detail": "the rules found only " + str(len(clauses)) + " main clauses"})

        numbers = [s["number"] for s in clauses]
        if numbers:
            missing = [n for n in range(1, max(numbers) + 1) if n not in numbers]
            if missing:
                problems.append({"problem": "MISSING_NUMBERS", "numbers": missing,
                                 "detail": "these clause numbers were not found"})
            repeated = sorted(set(n for n in numbers if numbers.count(n) > 1))
            if repeated:
                problems.append({"problem": "REPEATED_NUMBERS", "numbers": repeated})

        for s in clauses:
            if count_words(title_part(s["title"])) > LONG_TITLE_WORDS:
                problems.append({"problem": "SUSPICIOUS_TITLE", "clause": s["number"],
                                 "line": s["first_line"], "title": s["title"],
                                 "detail": "the clause starts with a long line, maybe it is not the real title"})
            if s["words"] > BIG_CLAUSE_SHARE * total_words and len(clauses) >= MIN_CLAUSES:
                problems.append({"problem": "HUGE_CLAUSE", "clause": s["number"],
                                 "detail": "this clause has " + str(round(100 * s["words"] / total_words))
                                           + "% of the file, maybe other clauses are inside it"})

        front_words = sum(s["words"] for s in outline if s["type"] == "front")
        if front_words > BIG_FRONT_SHARE * total_words:
            problems.append({"problem": "HUGE_FRONT_MATTER",
                             "detail": "the part before clause 1 has " + str(round(100 * front_words / total_words))
                                       + "% of the file, maybe clause 1 starts earlier"})

        oversize = [c["clause_id"] for c in self.clauses if c["oversize"]]
        if oversize:
            problems.append({"problem": "OVERSIZE_PARTS", "parts": oversize})

        # problems from the OCR cannot be fixed here, we only report them
        notes = [w.get("code") for w in self.document.get("warnings", [])]
        return {"problems": problems, "ocr_warnings": notes, "ok": len(problems) == 0}

    def view_lines(self, start, end):
        start = max(0, int(start))
        end = min(len(self.items) - 1, int(end), start + MAX_VIEW_LINES - 1)
        lines = []
        for index in range(start, end + 1):
            item = self.items[index]
            clause_number = None
            for s_index, number in self.starts.items():
                if s_index == index:
                    clause_number = number
            lines.append({"line": index, "page": item.get("page"), "type": item.get("type"),
                          "starts_clause": clause_number, "text": short(get_item_text(item))})
        return {"lines": lines}

    def search_lines(self, text):
        found = []
        for index in range(len(self.items)):
            if text in get_item_text(self.items[index]):
                found.append({"line": index, "page": self.items[index].get("page"),
                              "text": short(get_item_text(self.items[index]), 150)})
                if len(found) >= MAX_SEARCH_RESULTS:
                    break
        return {"results": found}

    def get_table_of_contents(self):
        lines = []
        for index in range(len(self.items)):
            text = get_item_text(self.items[index])
            if self.items[index].get("type") != "table" and is_contents_line(text):
                lines.append({"line": index, "text": short(text, 600)})
                if len(lines) >= 60:
                    break
        return {"contents_lines": lines}

    def set_clause_start(self, number, line, reason):
        number = int(number)
        line = int(line)
        if line < 0 or line >= len(self.items):
            return {"error": "no line " + str(line)}

        # one start per number and one number per line
        for index in list(self.starts):
            if self.starts[index] == number or index == line:
                del self.starts[index]
        self.starts[line] = number
        self.changes[number] = reason
        self.run_segmentation()
        return {"done": True, "clause": number, "now_starts_at_line": line,
                "line_text": short(get_item_text(self.items[line]))}

    # the OCR can put a clause title inside another line
    def split_line(self, line, title_text, number, reason):
        line = int(line)
        number = int(number)
        if line < 0 or line >= len(self.items):
            return {"error": "no line " + str(line)}
        item = self.items[line]
        text = get_item_text(item)
        if item.get("type") == "table":
            return {"error": "this line is a table"}
        if title_text not in text:
            return {"error": "the title text is not in this line"}

        # the clause number written at the start of the line belongs to the title
        number_pattern = r"^\s*[-(]?\s*" + str(number) + r"\s*[).\-]?\s*"
        marker_match = re.match(number_pattern, text)
        marker = marker_match.group(0).strip() if marker_match else str(number)

        before_text = text.replace(title_text, " ", 1)
        if marker_match:
            before_text = before_text[marker_match.end():]
        before_text = before_text.strip()

        title_item = dict(item)
        title_item["id"] = item["id"] + "-a"
        title_item["type"] = "heading"
        title_item["marker"] = marker
        title_item["text"] = marker + " " + title_text.strip()
        title_item["text_norm"] = title_item["text"]

        new_items = []
        if before_text:
            before = dict(item)
            before["text"] = before_text
            before["text_norm"] = before_text
            new_items.append(before)
        new_items.append(title_item)

        # put the new lines in place of the old one and move the clause starts after it
        self.items = self.items[:line] + new_items + self.items[line + 1:]
        shift = len(new_items) - 1
        new_starts = {}
        for index, n in self.starts.items():
            if index == line:
                if n != number:
                    new_starts[index] = n
            elif index > line:
                new_starts[index + shift] = n
            else:
                new_starts[index] = n
        for index in list(new_starts):
            if new_starts[index] == number:
                del new_starts[index]
        title_line = line + len(new_items) - 1
        new_starts[title_line] = number
        self.starts = new_starts
        self.changes[number] = reason
        self.run_segmentation()
        return {"done": True, "clause": number, "title_line": title_line,
                "title": title_item["text"], "text_left_before": short(before_text, 120)}

    def remove_clause_start(self, number, reason):
        number = int(number)
        removed = False
        for index in list(self.starts):
            if self.starts[index] == number:
                del self.starts[index]
                removed = True
        if not removed:
            return {"error": "clause " + str(number) + " has no start"}
        self.changes[number] = reason
        self.run_segmentation()
        return {"done": True, "removed_clause": number}

    def finish(self, status, summary):
        if status not in ("ok", "fixed", "needs_review"):
            return {"error": "status must be ok, fixed or needs_review"}
        self.finished = {"status": status, "summary": summary}
        return {"done": True}

    # ----------final output----------

    def result(self):
        changed_numbers = set(self.changes)
        for clause in self.clauses:
            number = clause["clause_number"]
            fixed = clause["section_type"] == "clause" and number in changed_numbers
            clause["fixed_by_agent"] = fixed
            clause["agent_note"] = self.changes.get(number) if fixed else None

        finished = self.finished or {"status": "needs_review",
                                     "summary": "the agent stopped before it finished"}
        report = {
            "status": finished["status"],
            "summary": finished["summary"],
            "problems_found": self.first_problems,
            "changes": [{"clause": n, "reason": r} for n, r in sorted(self.changes.items())],
            "steps": self.steps,
        }
        return self.clauses, report


# what the LLM sees about every tool (name, what it does, its inputs)
TOOLS = [
    {
        "name": "segment_document",
        "description": "Split the document into clauses with the rule based segmentation and return an outline: "
                       "every main clause with its number, title, first line, pages, parts and words.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "check_segmentation",
        "description": "Check the current segmentation and return the problems found "
                       "(missing clause numbers, suspicious titles, a clause that is too big, ...). "
                       "ok=true means no problems.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "view_lines",
        "description": "Show the document lines from start to end (max 40 lines) with their page and type, "
                       "and which line starts a clause.",
        "parameters": {"type": "object", "properties": {
            "start": {"type": "integer", "description": "first line number"},
            "end": {"type": "integer", "description": "last line number"},
        }, "required": ["start", "end"]},
    },
    {
        "name": "search_lines",
        "description": "Find the lines that contain a text (for example a clause title from the table of contents).",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the exact Arabic text to look for"},
        }, "required": ["text"]},
    },
    {
        "name": "get_table_of_contents",
        "description": "Return the table of contents lines of the document, to compare with the clauses found.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "set_clause_start",
        "description": "Make a main clause start at a given line, then segment again. "
                       "Use it to fix a clause that starts at the wrong line or was not found.",
        "parameters": {"type": "object", "properties": {
            "number": {"type": "integer", "description": "the clause number"},
            "line": {"type": "integer", "description": "the line where the clause title is"},
            "reason": {"type": "string", "description": "short reason in Arabic, shown in the report"},
        }, "required": ["number", "line", "reason"]},
    },
    {
        "name": "split_line",
        "description": "Use it when a clause title is inside another line (the OCR glued it to text). "
                       "The title text is taken out of the line and becomes the start of the clause; "
                       "the rest of the line stays with the clause before.",
        "parameters": {"type": "object", "properties": {
            "line": {"type": "integer", "description": "the line that has the title inside it"},
            "title_text": {"type": "string", "description": "the title words exactly as written in the line, without the number"},
            "number": {"type": "integer", "description": "the clause number"},
            "reason": {"type": "string", "description": "short reason in Arabic, shown in the report"},
        }, "required": ["line", "title_text", "number", "reason"]},
    },
    {
        "name": "remove_clause_start",
        "description": "Remove a wrong main clause start (for example a list item that is not a clause), "
                       "then segment again.",
        "parameters": {"type": "object", "properties": {
            "number": {"type": "integer", "description": "the clause number to remove"},
            "reason": {"type": "string", "description": "short reason in Arabic, shown in the report"},
        }, "required": ["number", "reason"]},
    },
    {
        "name": "finish",
        "description": "Finish the work. status: ok (no problems), fixed (problems were fixed) "
                       "or needs_review (some problems could not be fixed and a person must check).",
        "parameters": {"type": "object", "properties": {
            "status": {"type": "string", "enum": ["ok", "fixed", "needs_review"]},
            "summary": {"type": "string", "description": "short summary in Arabic of what was found and done"},
        }, "required": ["status", "summary"]},
    },
]


# run tool by its name (the agent calls this)
def run_tool(session, name, arguments):
    tools = {
        "segment_document": session.segment_document,
        "check_segmentation": session.check_segmentation,
        "view_lines": session.view_lines,
        "search_lines": session.search_lines,
        "get_table_of_contents": session.get_table_of_contents,
        "set_clause_start": session.set_clause_start,
        "split_line": session.split_line,
        "remove_clause_start": session.remove_clause_start,
        "finish": session.finish,
    }
    if name not in tools:
        return {"error": "unknown tool " + name}
    try:
        output = tools[name](**arguments)
    except Exception as e:
        output = {"error": str(e)}
    session.steps.append({"tool": name, "arguments": arguments,
                          "result": "error: " + output["error"] if "error" in output else "ok"})
    return output
