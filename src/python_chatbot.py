"""A small retrieval-based Python Q&A chatbot implemented with Python's stdlib.

The project intentionally implements tokenization, TF-IDF and cosine similarity
in this file rather than calling a ready-made NLP or machine-learning package.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# Common conversational words that usually add little topic information.
# Programming words such as "is", "in", "if", "for", "not", and "class"
# are intentionally retained because they can have meaning in Python questions.
STOP_WORDS = {
    "what", "how", "do", "does", "did", "i", "you", "we", "they", "he", "she",
    "a", "an", "the", "please", "tell", "me", "about", "explain", "describe", "use",
    "can", "could", "would", "should", "want", "need", "to", "of", "for", "with",
    "and", "or", "but", "between", "difference", "why", "when", "where", "which",
    "who", "whom",
}

TOKEN_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+(?:\.\d+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into simple word/identifier tokens.

    This deliberately small tokenizer is transparent and easy to explain. It
    preserves Python identifiers such as `__init__` and `list`.
    """
    if not isinstance(text, str):
        raise TypeError("tokenize() expects a string")
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if token not in STOP_WORDS]


def _normalise_entry(entry: dict[str, Any]) -> dict[str, Any]:
    required = {"id", "topic", "question", "answer"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"Knowledge-base entry is missing fields: {sorted(missing)}")
    if not isinstance(entry["answer"], str) or not entry["answer"].strip():
        raise ValueError(f"Knowledge-base entry {entry.get('id')} has an empty answer")
    entry = dict(entry)
    entry["alternate_questions"] = list(entry.get("alternate_questions", []))
    entry["keywords"] = list(entry.get("keywords", []))
    return entry


def load_knowledge_base(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a JSON knowledge base created for this project."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Knowledge base must be a non-empty JSON list")
    entries = [_normalise_entry(item) for item in raw]
    ids = [item["id"] for item in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("Knowledge-base entry IDs must be unique")
    return entries


def _document_text(entry: dict[str, Any]) -> str:
    """Build a searchable document, giving authored question/keywords emphasis."""
    parts: list[str] = []
    # Repetition is a simple, visible way to give headings and keywords more weight.
    parts.extend([entry["question"]] * 2)
    parts.extend(entry.get("alternate_questions", []))
    parts.extend([" ".join(entry.get("keywords", []))] * 2)
    parts.append(entry["topic"])
    parts.append(entry["answer"])
    return " ".join(parts)


def build_tfidf(documents: Iterable[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Build L2-normalised TF-IDF vectors without an external ML library.

    TF is raw token count. IDF = log((N + 1) / (df + 1)) + 1.
    Each document vector is divided by its Euclidean length.
    """
    documents = list(documents)
    if not documents:
        raise ValueError("At least one document is required")
    token_lists = [tokenize(document) for document in documents]
    counters = [Counter(tokens) for tokens in token_lists]

    document_frequency: Counter[str] = Counter()
    for counter in counters:
        document_frequency.update(counter.keys())

    total_documents = len(documents)
    idf = {
        term: math.log((total_documents + 1) / (frequency + 1)) + 1.0
        for term, frequency in document_frequency.items()
    }

    vectors: list[dict[str, float]] = []
    for counter in counters:
        vector = {term: float(count) * idf[term] for term, count in counter.items()}
        length = math.sqrt(sum(value * value for value in vector.values()))
        if length:
            vector = {term: value / length for term, value in vector.items()}
        vectors.append(vector)
    return vectors, idf


def vectorize_query(query: str, idf: dict[str, float]) -> dict[str, float]:
    """Represent a user query in the vocabulary/IDF space of the knowledge base."""
    counts = Counter(tokenize(query))
    vector = {term: count * idf[term] for term, count in counts.items() if term in idf}
    length = math.sqrt(sum(value * value for value in vector.values()))
    if length:
        vector = {term: value / length for term, value in vector.items()}
    return vector


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    """Calculate cosine similarity for two sparse dictionaries."""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def token_coverage(query: str, searchable_text: str) -> float:
    """Fraction of meaningful query tokens found in a candidate entry."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    document_terms = set(tokenize(searchable_text))
    return len(query_terms & document_terms) / len(query_terms)


@dataclass
class ChatAnswer:
    answer: str
    matched_topic: str | None
    confidence: float
    matched_question: str | None
    is_fallback: bool
    entry_id: str | None


class PythonChatbot:
    """Retrieval chatbot that returns authored answers from a local knowledge base."""

    def __init__(self, knowledge_base: list[dict[str, Any]], threshold: float = 0.18):
        if not knowledge_base:
            raise ValueError("knowledge_base cannot be empty")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.entries = [_normalise_entry(item) for item in knowledge_base]
        ids = [item["id"] for item in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Knowledge-base entry IDs must be unique")
        self.threshold = threshold
        self.search_documents = [_document_text(item) for item in self.entries]
        self.document_vectors, self.idf = build_tfidf(self.search_documents)

    @classmethod
    def from_json(cls, path: str | Path, threshold: float = 0.18) -> "PythonChatbot":
        return cls(load_knowledge_base(path), threshold=threshold)

    def answer(self, question: str) -> ChatAnswer:
        """Find the closest knowledge-base answer or return a safe fallback."""
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        if not question.strip():
            return ChatAnswer(
                "Please enter a Python question so I can look for a relevant answer.",
                None, 0.0, None, True, None,
            )

        query_vector = vectorize_query(question, self.idf)
        if not query_vector:
            return ChatAnswer(
                "I could not find enough Python-related terms in that question. "
                "Try mentioning a Python concept such as lists, functions, loops, or exceptions.",
                None, 0.0, None, True, None,
            )

        ranked: list[tuple[float, int]] = []
        for index, (entry, document, doc_vector) in enumerate(
            zip(self.entries, self.search_documents, self.document_vectors)
        ):
            similarity = cosine_similarity(query_vector, doc_vector)
            coverage = token_coverage(question, document)
            # Cosine similarity captures term importance; coverage rewards direct matches.
            score = 0.80 * similarity + 0.20 * coverage
            ranked.append((score, index))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_index = ranked[0]
        best = self.entries[best_index]
        if best_score < self.threshold:
            return ChatAnswer(
                "I do not have a reliable answer for that question in my current Python "
                "knowledge base. Try rephrasing it or add the topic and its explanation "
                "to data/knowledge_base.json.",
                None, round(best_score, 4), None, True, None,
            )

        return ChatAnswer(
            answer=best["answer"],
            matched_topic=best["topic"],
            confidence=round(min(best_score, 1.0), 4),
            matched_question=best["question"],
            is_fallback=False,
            entry_id=best["id"],
        )

    def suggest_topics(self, limit: int = 8) -> list[str]:
        """Return a compact list of topics the chatbot can discuss."""
        if limit < 1:
            return []
        return [entry["topic"] for entry in self.entries[:limit]]


def default_knowledge_base_path() -> Path:
    """Resolve the included knowledge-base file relative to the project root."""
    return Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json"


def create_default_chatbot(threshold: float = 0.18) -> PythonChatbot:
    return PythonChatbot.from_json(default_knowledge_base_path(), threshold=threshold)
