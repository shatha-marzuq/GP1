# Segmentation Agent
# the LLM gets the goal (split the Terms and Conditions file into its real clauses) and a set of tools.
# it decides by itself which tool to use. it runs the rule based segmentation, checks it,
# looks at the lines when something looks wrong, fixes it, checks again, then finishes.

import json
import os
import time

from segmentation_tools import TOOLS, SegmentationSession, run_tool

# ---------- the model (loaded at the start) ----------
# the API key is read from the .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = os.getenv("SEGMENTATION_MODEL", "gpt-5")
# required forces the model to always use a tool
TOOL_CHOICE = os.getenv("SEGMENTATION_TOOL_CHOICE", "required")
MAX_STEPS = 25  # the agent must finish in this number of steps

client = None


def get_client():
    global client
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    return client


SYSTEM_PROMPT = """You are the Segmentation Agent of Mithaq, a system that checks the Terms and Conditions
(الشروط والأحكام) of Saudi private investment funds against the Capital Market Authority regulations.

Your job: make sure the document is split into its real main clauses (البنود الرئيسية), one section per clause,
in the right order, so every clause can later be checked against the regulations.

The document comes from OCR as numbered lines. A rule based segmentation already exists as a tool.
How to work:
1. Call segment_document to see the outline, then check_segmentation to see the problems.
2. If there are no problems, look quickly at the outline. If it looks right, call finish with status "ok".
3. If there are problems, look at the lines with view_lines / search_lines. The table of contents
   (get_table_of_contents) tells you the real clause titles and their order.
4. Fix with set_clause_start, split_line or remove_clause_start, then call check_segmentation again.
5. Call finish with "fixed" when all problems are fixed, or "needs_review" when a problem cannot be fixed
   (for example text that the OCR could not read).

Rules:
- Never invent clauses or titles that are not in the document.
- A main clause title is short (like "11) التعاملات" or "-8 الاشتراك:"). Points inside a list
  (like "1) التغيير المهم في أهداف الصندوق.") are not main clauses.
- OCR warnings cannot be fixed by you; mention them in the summary if they matter.
- Write the reason and the summary in Arabic, short and clear.
"""


def ask_model(messages):
    for attempt in range(6):
        try:
            return get_client().chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools_for_openai(),
                tool_choice=TOOL_CHOICE,
            )
        except Exception as e:
            if getattr(e, "status_code", None) == 429 and attempt < 5:
                print("  agent: too many requests, waiting 20 seconds ...")
                time.sleep(20)
            else:
                raise


def tools_for_openai():
    return [{"type": "function", "function": tool} for tool in TOOLS]


# ----------agent loop----------
def run_segmentation_agent(items, document=None, use_llm=True, verbose=True):
    session = SegmentationSession(items, document)
    session.first_problems = session.check_segmentation()["problems"]

    if not use_llm:
        # without the LLM. only the rules and the check
        status = "ok" if not session.first_problems else "needs_review"
        session.finish(status, "تم التقسيم بالقواعد فقط بدون الـ LLM")
        return session.result()

    if verbose:
        print("  agent model:", MODEL, "| server:", os.getenv("OPENAI_BASE_URL") or "OpenAI")

    filename = (document or {}).get("filename", "")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Segment this Terms and Conditions file: " + filename
                                    + " (" + str(len(session.items)) + " lines). Start now."},
    ]

    for step in range(MAX_STEPS):
        response = ask_model(messages)
        # if the URL or the model name is wrong, some servers send back a web page instead of JSON
        if isinstance(response, str):
            raise RuntimeError("the LLM server did not answer with JSON. check OPENAI_BASE_URL and "
                               "SEGMENTATION_MODEL in .env. the server said: " + response[:300])
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        assistant_message = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            assistant_message["tool_calls"] = [call.model_dump() for call in tool_calls]
        messages.append(assistant_message)

        for call in tool_calls:
            name = call.function.name
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            output = run_tool(session, name, arguments)
            if verbose:
                print("  agent ->", name, json.dumps(arguments, ensure_ascii=False))
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(output, ensure_ascii=False),
            })

        # with tool_choice "auto" the model can answer with text only, then we remind it to use the tools
        if not tool_calls:
            messages.append({"role": "user", "content": "Use the tools. When you are done, call finish."})

        if session.finished:
            break

    return session.result()
