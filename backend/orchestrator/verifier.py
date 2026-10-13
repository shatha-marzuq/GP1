"""Compliance Verification Agent: one clause + its retrieved regulations -> a structured verdict.

The LLM only sees the regulations we retrieved and must quote both sides word for word.
`check_verdict` then verifies those quotes in code, so a hallucinated quote can never reach the report.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["Compliant", "Violation", "Needs Review"]


class Verdict(BaseModel):
    status: Status
    regulation_ids: list[str] = Field(default_factory=list,
                                      description="Labels of the regulations the verdict relies on, e.g. ['REG-2']")
    clause_quote: str = Field("", description="Exact words copied from the clause")
    regulation_quote: str = Field("", description="Exact words copied from the first cited regulation")
    reasoning_ar: str = Field(description="Short justification in Arabic for the compliance officer")
    needs_more_context: bool = Field(False, description="True if the given regulations are not enough to decide")
    suggested_query: str | None = Field(None, description="If needs_more_context: what to search for instead")


SYSTEM_PROMPT = """You are a compliance reviewer for the Saudi Capital Market Authority (CMA).
You check one clause from the Terms and Conditions of a PRIVATE investment fund against the CMA regulations given to you.

Rules:
- Judge ONLY against the regulations provided below. Never use outside or remembered regulations.
- Compliant: the clause satisfies the regulations that apply to it.
- Violation: the clause clearly contradicts or omits something a provided regulation requires.
- Needs Review: the wording is ambiguous, allows more than one legal reading, or the provided regulations do not settle it.
- If none of the provided regulations govern this clause, set needs_more_context=true and write a better Arabic search query in suggested_query.
- clause_quote and regulation_quote must be copied EXACTLY, character for character, from the texts below. Do not paraphrase.
- Write reasoning_ar in clear Arabic, at most 3 sentences."""


def label(n: int) -> str:
    return f"REG-{n}"


def build_prompt(clause: dict, hits: list[dict]) -> str:
    regs = "\n\n".join(f"[{label(n)}] {h['citation']}\n{h['text']}" for n, h in enumerate(hits, 1))
    pages = clause.get("page_start")
    if pages and clause.get("page_end") != pages:
        pages = f"{pages}-{clause['page_end']}"
    return (f"## CMA regulations\n\n{regs}\n\n"
            f"## Clause {clause['clause_id']} (page {pages or '?'}): {clause.get('title', '')}\n\n{clause['text']}\n\n"
            "Return your verdict.")


class LLMVerifier:
    """Wraps any LangChain chat model that supports structured output."""

    def __init__(self, llm):
        self.chain = llm.with_structured_output(Verdict)

    def __call__(self, clause: dict, hits: list[dict]) -> Verdict:
        return self.chain.invoke([("system", SYSTEM_PROMPT), ("human", build_prompt(clause, hits))])

    @classmethod
    def from_env(cls) -> "LLMVerifier":
        """MITHAQ_LLM picks the model, e.g. "azure_openai:gpt-4o" or "openai:gpt-4o" (provider keys from env)."""
        from langchain.chat_models import init_chat_model
        return cls(init_chat_model(os.environ.get("MITHAQ_LLM", "azure_openai:gpt-4o"), temperature=0))


def _norm(s: str) -> str:
    return " ".join(s.split())


def check_verdict(v: Verdict, clause: dict, hits: list[dict]) -> list[str]:
    """Problems that make the verdict untrustworthy. Empty list = evidence checks out."""
    problems = []
    labels = {label(n): h for n, h in enumerate(hits, 1)}
    unknown = [r for r in v.regulation_ids if r not in labels]
    if unknown:
        problems.append(f"cited regulations that were not provided: {unknown}")
    if v.clause_quote and _norm(v.clause_quote) not in _norm(clause["text"]):
        problems.append("clause_quote is not in the clause text")
    cited = [labels[r] for r in v.regulation_ids if r in labels]
    if v.regulation_quote and not any(_norm(v.regulation_quote) in _norm(h["text"]) for h in cited):
        problems.append("regulation_quote is not in any cited regulation")
    if v.status == "Violation" and not (v.clause_quote and v.regulation_quote and cited):
        problems.append("a Violation needs a clause quote, a regulation quote and a citation")
    return problems
