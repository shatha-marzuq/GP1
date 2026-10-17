"""Stand-in retriever so the graph runs without the BGE-M3 model or the Chroma store (demos and tests)."""
from __future__ import annotations


class StubRetriever:
    """Returns the same fake regulation for every query."""

    def __init__(self, text: str = "يجب أن تتضمن شروط وأحكام الصندوق الخاص بيان الرسوم ومقابل الخدمات."):
        self.text = text
        self.queries: list[str] = []

    def search(self, query: str, k: int = 5, **kwargs) -> list[dict]:
        self.queries.append(query)
        return [{"id": "IFR-anx11-c03", "score": 1.0, "citation": "لائحة صناديق الاستثمار، الملحق 11",
                 "pdf_pages": [120], "text": self.text}]
