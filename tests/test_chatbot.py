"""Unit tests for the self-written tokenizer, vectorizer and retrieval chatbot."""
import math
import unittest
from pathlib import Path

from src.python_chatbot import (
    PythonChatbot,
    build_tfidf,
    cosine_similarity,
    create_default_chatbot,
    load_knowledge_base,
    tokenize,
    vectorize_query,
)


class TokenizerTests(unittest.TestCase):
    def test_tokenizer_lowercases_and_keeps_identifiers(self):
        self.assertEqual(tokenize("Use __init__ with Student42"), ["__init__", "student42"])

    def test_tokenizer_rejects_non_string(self):
        with self.assertRaises(TypeError):
            tokenize(42)


class MathTests(unittest.TestCase):
    def test_cosine_same_vector_is_one(self):
        self.assertAlmostEqual(cosine_similarity({"a": 1.0}, {"a": 1.0}), 1.0)

    def test_cosine_orthogonal_vectors_is_zero(self):
        self.assertAlmostEqual(cosine_similarity({"a": 1.0}, {"b": 1.0}), 0.0)

    def test_tfidf_vectors_are_normalised(self):
        vectors, idf = build_tfidf(["red red blue", "blue green"])
        for vector in vectors:
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in vector.values())), 1.0)
        self.assertIn("red", idf)

    def test_query_vector_uses_existing_vocabulary(self):
        _, idf = build_tfidf(["python function", "python list"])
        vector = vectorize_query("python unknown", idf)
        self.assertEqual(set(vector), {"python"})


class ChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = create_default_chatbot()

    def test_knowledge_base_has_forty_entries(self):
        self.assertEqual(len(self.bot.entries), 40)

    def test_list_tuple_question_maps_to_correct_topic(self):
        result = self.bot.answer("How is a tuple different from a list?")
        self.assertFalse(result.is_fallback)
        self.assertEqual(result.entry_id, "PY007")

    def test_lambda_question_maps_to_lambda(self):
        result = self.bot.answer("Explain an anonymous function in Python")
        self.assertEqual(result.entry_id, "PY016")

    def test_exception_question_maps_to_exception_handling(self):
        result = self.bot.answer("How do I catch runtime errors?")
        self.assertEqual(result.entry_id, "PY023")

    def test_unknown_question_returns_fallback(self):
        result = self.bot.answer("Who won the football world cup in 1998?")
        self.assertTrue(result.is_fallback)
        self.assertIsNone(result.matched_topic)

    def test_empty_question_returns_prompt(self):
        result = self.bot.answer("   ")
        self.assertTrue(result.is_fallback)
        self.assertIn("enter a Python question", result.answer)

    def test_topics_list_is_bounded(self):
        self.assertEqual(len(self.bot.suggest_topics(3)), 3)
        self.assertEqual(self.bot.suggest_topics(0), [])

    def test_duplicate_ids_are_rejected(self):
        entries = load_knowledge_base(Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json")
        duplicate = [entries[0], dict(entries[0])]
        with self.assertRaises(ValueError):
            PythonChatbot(duplicate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
