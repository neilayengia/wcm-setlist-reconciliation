"""
End-to-end pipeline test using local data and a mocked LLM client.

Verifies that the full pipeline produces the expected CSV output.
"""

import csv
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import setup_logging
from ingestion import fetch_tour_data, load_catalog, flatten_setlists
from reconciler import reconcile
from output import write_csv, FIELDNAMES
from llm_matching import clear_cache


# Quiet logs during tests
setup_logging()


def _mock_llm_response(track_name: str) -> str:
    """Return a realistic LLM response based on the track name."""
    if "Desert Rain / Ocean Avenue" in track_name:
        return json.dumps({"matches": [
            {"catalog_id": "CAT-004", "confidence": "High", "reasoning": "exact match"},
            {"catalog_id": "CAT-005", "confidence": "High", "reasoning": "exact match"},
        ]})
    elif "Tokyo" in track_name.lower() or "tokyo" in track_name.lower():
        return json.dumps({"matches": [
            {"catalog_id": "CAT-002", "confidence": "High", "reasoning": "abbreviation of Midnight in Tokyo"},
        ]})
    elif "Wonderwall" in track_name:
        return json.dumps({"matches": [
            {"catalog_id": "None", "confidence": "None", "reasoning": "Oasis cover"},
        ]})
    elif "Smsls" in track_name:
        return json.dumps({"matches": [
            {"catalog_id": "None", "confidence": "None", "reasoning": "Nirvana cover"},
        ]})
    else:
        return json.dumps({"matches": [
            {"catalog_id": "None", "confidence": "None", "reasoning": "no match"},
        ]})


class TestPipeline(unittest.TestCase):

    def setUp(self):
        clear_cache()

    @patch("ingestion.http_requests.get")
    def test_end_to_end_produces_correct_output(self, mock_get):
        # Force local file fallback
        mock_get.side_effect = Exception("Connection refused")

        # Load real data
        tour_data = fetch_tour_data()
        catalog = load_catalog()
        tracks = flatten_setlists(tour_data)

        # Create mock OpenAI client
        client = MagicMock()

        def create_side_effect(**kwargs):
            messages = kwargs.get("messages", [])
            user_msg = messages[-1]["content"] if messages else ""
            # Extract track name from the prompt
            for line in user_msg.split("\n"):
                if line.startswith('SETLIST TRACK:'):
                    track_name = line.split('"')[1]
                    break
            else:
                track_name = ""

            choice = MagicMock()
            choice.message.content = _mock_llm_response(track_name)
            return MagicMock(choices=[choice])

        client.chat.completions.create.side_effect = create_side_effect

        # Run pipeline
        results = reconcile(tracks, catalog, client=client)

        # Verify row count: 10 tracks, medley expands to 2, so 11 rows
        self.assertEqual(len(results), 11)

        # Verify deterministic matches
        neon_matches = [r for r in results if r["setlist_track_name"] == "Neon Dreams"]
        self.assertTrue(all(r["matched_catalog_id"] == "CAT-001" for r in neon_matches))

        # Verify all results have required fields
        for r in results:
            for field in FIELDNAMES:
                self.assertIn(field, r)
            # Verify matched_catalog_title is populated for catalog matches
            if r["matched_catalog_id"] != "None":
                self.assertTrue(len(r.get("matched_catalog_title", "")) > 0)

    @patch("ingestion.http_requests.get")
    def test_csv_output_has_correct_columns(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")

        tour_data = fetch_tour_data()
        catalog = load_catalog()
        tracks = flatten_setlists(tour_data)

        # Run without LLM (deterministic only)
        results = reconcile(tracks, catalog, client=None)

        # Write to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("output.OUTPUT_DIR",
                        __import__("pathlib").Path(tmpdir)):
                output_path = write_csv(results)

                with open(output_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertEqual(set(reader.fieldnames), set(FIELDNAMES))
                    self.assertTrue(len(rows) > 0)


if __name__ == "__main__":
    unittest.main()
