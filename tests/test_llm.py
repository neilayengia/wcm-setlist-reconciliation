"""
Tests for LLM matching — all OpenAI calls are mocked.

Covers response parsing, medley handling, cover detection,
retry behavior, and the in-memory cache.
"""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_matching import (
    llm_fuzzy_match,
    _parse_llm_response,
    clear_cache,
)


# ── Sample catalog ────────────────────────────────────────

SAMPLE_CATALOG = [
    {"catalog_id": "CAT-001", "title": "Neon Dreams", "writers": "Alex Park"},
    {"catalog_id": "CAT-002", "title": "Midnight in Tokyo", "writers": "Alex Park; Jane Miller"},
    {"catalog_id": "CAT-004", "title": "Desert Rain", "writers": "Leyla Ademi"},
    {"catalog_id": "CAT-005", "title": "Ocean Avenue", "writers": "Leyla Ademi"},
]


def _mock_client(response_json: str) -> MagicMock:
    """Create a mock OpenAI client that returns the given JSON string."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_json
    client.chat.completions.create.return_value = MagicMock(
        choices=[choice]
    )
    return client


# ── _parse_llm_response() tests ──────────────────────────

class TestParseLLMResponse(unittest.TestCase):

    def test_standard_matches_key(self):
        raw = json.dumps({"matches": [{"catalog_id": "CAT-001", "confidence": "High"}]})
        result = _parse_llm_response(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["catalog_id"], "CAT-001")

    def test_results_key(self):
        raw = json.dumps({"results": [{"catalog_id": "CAT-002", "confidence": "High"}]})
        result = _parse_llm_response(raw)
        self.assertEqual(len(result), 1)

    def test_bare_list(self):
        raw = json.dumps([{"catalog_id": "CAT-001", "confidence": "High"}])
        result = _parse_llm_response(raw)
        self.assertEqual(len(result), 1)

    def test_single_dict_with_catalog_id(self):
        raw = json.dumps({"catalog_id": "CAT-001", "confidence": "High"})
        result = _parse_llm_response(raw)
        self.assertEqual(len(result), 1)

    def test_empty_object_returns_empty(self):
        raw = json.dumps({"foo": "bar"})
        result = _parse_llm_response(raw)
        self.assertEqual(len(result), 0)


# ── llm_fuzzy_match() tests (mocked) ────────────────────

class TestLLMFuzzyMatch(unittest.TestCase):

    def setUp(self):
        clear_cache()

    def test_single_match(self):
        response = json.dumps({"matches": [
            {"catalog_id": "CAT-002", "confidence": "High", "reasoning": "abbreviation"}
        ]})
        client = _mock_client(response)
        result = llm_fuzzy_match("Tokyo (Acoustic)", SAMPLE_CATALOG, client, max_retries=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["catalog_id"], "CAT-002")
        self.assertEqual(result[0]["confidence"], "High")

    def test_medley_split(self):
        response = json.dumps({"matches": [
            {"catalog_id": "CAT-004", "confidence": "High", "reasoning": "exact"},
            {"catalog_id": "CAT-005", "confidence": "High", "reasoning": "exact"},
        ]})
        client = _mock_client(response)
        result = llm_fuzzy_match("Desert Rain / Ocean Avenue", SAMPLE_CATALOG, client, max_retries=0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["catalog_id"], "CAT-004")
        self.assertEqual(result[1]["catalog_id"], "CAT-005")

    def test_cover_detection(self):
        response = json.dumps({"matches": [
            {"catalog_id": "None", "confidence": "None", "reasoning": "Oasis cover"}
        ]})
        client = _mock_client(response)
        result = llm_fuzzy_match("Wonderwall", SAMPLE_CATALOG, client, max_retries=0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["catalog_id"], "None")

    @patch("llm_matching._backoff_sleep")  # Don't actually sleep in tests
    def test_retry_on_bad_json(self, mock_sleep):
        client = MagicMock()
        # First call returns invalid JSON, second call succeeds
        bad_choice = MagicMock()
        bad_choice.message.content = "not json at all"
        good_choice = MagicMock()
        good_choice.message.content = json.dumps(
            {"matches": [{"catalog_id": "CAT-001", "confidence": "High"}]}
        )
        client.chat.completions.create.side_effect = [
            MagicMock(choices=[bad_choice]),
            MagicMock(choices=[good_choice]),
        ]
        result = llm_fuzzy_match("Neon Dreams", SAMPLE_CATALOG, client, max_retries=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["catalog_id"], "CAT-001")

    def test_cache_hit_skips_api_call(self):
        response = json.dumps({"matches": [
            {"catalog_id": "CAT-001", "confidence": "High", "reasoning": "exact"}
        ]})
        client = _mock_client(response)

        # First call — makes API call
        result1 = llm_fuzzy_match("Neon Dreams", SAMPLE_CATALOG, client, max_retries=0)
        self.assertEqual(client.chat.completions.create.call_count, 1)

        # Second call — should use cache, no new API call
        result2 = llm_fuzzy_match("Neon Dreams", SAMPLE_CATALOG, client, max_retries=0)
        self.assertEqual(client.chat.completions.create.call_count, 1)  # Still 1
        self.assertEqual(result1, result2)

    @patch("llm_matching._backoff_sleep")
    def test_all_retries_exhausted_returns_fallback(self, mock_sleep):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API down")
        result = llm_fuzzy_match("Some Track", SAMPLE_CATALOG, client, max_retries=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["catalog_id"], "None")
        self.assertEqual(result[0]["confidence"], "None")


if __name__ == "__main__":
    unittest.main()
