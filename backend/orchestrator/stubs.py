"""Stand-in agents so the graph runs without the embedding model or an LLM key (demos and tests)."""
from __future__ import annotations

from .verifier import Verdict


class StubRetriever:
    """Returns the same fake regulation for every query."""

    def __init__(self, text: str = "يجب أن تتضمن شروط وأحكام الصندوق الخاص بيان الرسوم ومقابل الخدمات."):
        self.text = text
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5) -> list[dict]:
        self.queries.append(query)
        return [{"id": "IFR-anx11-c03", "score": 1.0, "citation": "لائحة صناديق الاستثمار، الملحق 11",
                 "text": self.text}]


def stub_verifier(clause: dict, hits: list[dict]) -> Verdict:
    """Quotes the first words of the clause and marks everything Needs Review."""
    words = clause["text"].split()
    return Verdict(status="Needs Review", regulation_ids=["REG-1"], clause_quote=" ".join(words[:6]),
                   regulation_quote=hits[0]["text"].split("،")[0],
                   reasoning_ar="نتيجة تجريبية: لم يُستدعَ النموذج اللغوي.")
