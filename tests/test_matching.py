"""
Unit tests for the deterministic matching module.

Covers normalize(), deterministic_match(), and validate_match().
"""

import sys
import os
import unittest

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from matching import normalize, deterministic_match, validate_match


# ── Sample catalog for testing ────────────────────────────

SAMPLE_CATALOG = [
    {"catalog_id": "CAT-001", "title": "Neon Dreams", "writers": "Alex Park"},
    {"catalog_id": "CAT-002", "title": "Midnight in Tokyo", "writers": "Alex Park; Jane Miller"},
    {"catalog_id": "CAT-003", "title": "Shattered Glass", "writers": "Jane Miller"},
    {"catalog_id": "CAT-007", "title": "Velocity", "writers": "Paul Walker"},
    {"catalog_id": "CAT-006", "title": "Golden Gate", "writers": "Rachel Davis"},
]

CATALOG_IDS = {s["catalog_id"] for s in SAMPLE_CATALOG}


# ── normalize() tests ────────────────────────────────────

class TestNormalize(unittest.TestCase):

    def test_strips_parenthetical_suffix(self):
        self.assertEqual(normalize("Tokyo (Acoustic)"), "tokyo")

    def test_strips_multiple_parentheticals(self):
        self.assertEqual(normalize("Song (Live) (Remix)"), "song")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize("  hello   world  "), "hello world")

    def test_lowercases(self):
        self.assertEqual(normalize("NEON DREAMS"), "neon dreams")

    def test_strips_extended_jam(self):
        self.assertEqual(normalize("Velocity (Extended Jam)"), "velocity")

    def test_plain_string_unchanged(self):
        self.assertEqual(normalize("Golden Gate"), "golden gate")

    def test_empty_string(self):
        self.assertEqual(normalize(""), "")


# ── deterministic_match() tests ──────────────────────────

class TestDeterministicMatch(unittest.TestCase):

    def test_exact_match_case_insensitive(self):
        cat_id, conf = deterministic_match("neon dreams", SAMPLE_CATALOG)
        self.assertEqual(cat_id, "CAT-001")
        self.assertEqual(conf, "Exact")

    def test_exact_match_with_different_case(self):
        cat_id, conf = deterministic_match("MIDNIGHT IN TOKYO", SAMPLE_CATALOG)
        self.assertEqual(cat_id, "CAT-002")
        self.assertEqual(conf, "Exact")

    def test_normalized_match_strips_suffix(self):
        cat_id, conf = deterministic_match("Velocity (Extended Jam)", SAMPLE_CATALOG)
        self.assertEqual(cat_id, "CAT-007")
        self.assertEqual(conf, "Exact")

    def test_normalized_match_acoustic(self):
        # "Tokyo (Acoustic)" normalizes to "tokyo" — but catalog has
        # "Midnight in Tokyo" which normalizes to "midnight in tokyo".
        # These should NOT match because the normalized forms differ.
        cat_id, conf = deterministic_match("Tokyo (Acoustic)", SAMPLE_CATALOG)
        self.assertIsNone(cat_id)
        self.assertIsNone(conf)

    def test_no_match_for_unknown_track(self):
        cat_id, conf = deterministic_match("Wonderwall", SAMPLE_CATALOG)
        self.assertIsNone(cat_id)
        self.assertIsNone(conf)

    def test_no_match_for_partial_title(self):
        cat_id, conf = deterministic_match("Neon", SAMPLE_CATALOG)
        self.assertIsNone(cat_id)
        self.assertIsNone(conf)

    def test_empty_catalog(self):
        cat_id, conf = deterministic_match("Neon Dreams", [])
        self.assertIsNone(cat_id)
        self.assertIsNone(conf)


# ── validate_match() tests ───────────────────────────────

class TestValidateMatch(unittest.TestCase):

    def test_valid_match_passes_through(self):
        result = validate_match(
            {"catalog_id": "CAT-001", "confidence": "High"},
            CATALOG_IDS,
        )
        self.assertEqual(result["catalog_id"], "CAT-001")
        self.assertEqual(result["confidence"], "High")

    def test_none_catalog_id_passes(self):
        result = validate_match(
            {"catalog_id": "None", "confidence": "None"},
            CATALOG_IDS,
        )
        self.assertEqual(result["catalog_id"], "None")
        self.assertEqual(result["confidence"], "None")

    def test_invalid_catalog_id_corrected(self):
        result = validate_match(
            {"catalog_id": "CAT-999", "confidence": "High"},
            CATALOG_IDS,
        )
        self.assertEqual(result["catalog_id"], "None")
        self.assertEqual(result["confidence"], "None")

    def test_invalid_confidence_corrected(self):
        result = validate_match(
            {"catalog_id": "CAT-001", "confidence": "Maybe"},
            CATALOG_IDS,
        )
        self.assertEqual(result["catalog_id"], "CAT-001")
        self.assertEqual(result["confidence"], "Review")

    def test_missing_fields_default_to_none(self):
        result = validate_match({}, CATALOG_IDS)
        self.assertEqual(result["catalog_id"], "None")
        self.assertEqual(result["confidence"], "None")

    def test_whitespace_in_values_stripped(self):
        result = validate_match(
            {"catalog_id": " CAT-001 ", "confidence": " High "},
            CATALOG_IDS,
        )
        self.assertEqual(result["catalog_id"], "CAT-001")
        self.assertEqual(result["confidence"], "High")


if __name__ == "__main__":
    unittest.main()
