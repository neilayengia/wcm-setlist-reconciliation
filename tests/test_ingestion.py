"""
Tests for data ingestion: catalog loading, setlist flattening,
and API fallback behavior.
"""

import json
import sys
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion import load_catalog, flatten_setlists, fetch_tour_data, _validate_tour_data


# ── Sample data ───────────────────────────────────────────

SAMPLE_TOUR_DATA = {
    "status": "success",
    "data": {
        "artist": "The Neon Lights",
        "tour": "The Midnight Tour 2024",
        "shows": [
            {
                "date": "2024-11-15",
                "venue": "The Echo Lounge",
                "city": "Los Angeles",
                "setlist": ["Neon Dreams", "Tokyo (Acoustic)", "Shattered Glass"],
            },
            {
                "date": "2024-11-16",
                "venue": "Fillmore",
                "city": "San Francisco",
                "setlist": ["Neon Dreams", "Velocity (Extended Jam)"],
            },
        ],
    },
}


# ── load_catalog() tests ─────────────────────────────────

class TestLoadCatalog(unittest.TestCase):

    def test_loads_correct_count(self):
        catalog = load_catalog()
        self.assertEqual(len(catalog), 15)

    def test_entries_have_required_fields(self):
        catalog = load_catalog()
        for entry in catalog:
            self.assertIn("catalog_id", entry)
            self.assertIn("title", entry)
            self.assertIn("writers", entry)
            self.assertIn("controlled_percentage", entry)

    def test_first_entry_correct(self):
        catalog = load_catalog()
        self.assertEqual(catalog[0]["catalog_id"], "CAT-001")
        self.assertEqual(catalog[0]["title"], "Neon Dreams")

    def test_catalog_ids_unique(self):
        catalog = load_catalog()
        ids = [e["catalog_id"] for e in catalog]
        self.assertEqual(len(ids), len(set(ids)))


# ── flatten_setlists() tests ─────────────────────────────

class TestFlattenSetlists(unittest.TestCase):

    def test_correct_row_count(self):
        tracks = flatten_setlists(SAMPLE_TOUR_DATA)
        # 3 tracks (show 1) + 2 tracks (show 2) = 5
        self.assertEqual(len(tracks), 5)

    def test_rows_have_required_fields(self):
        tracks = flatten_setlists(SAMPLE_TOUR_DATA)
        for t in tracks:
            self.assertIn("show_date", t)
            self.assertIn("venue_name", t)
            self.assertIn("city", t)
            self.assertIn("setlist_track_name", t)

    def test_show_date_carried(self):
        tracks = flatten_setlists(SAMPLE_TOUR_DATA)
        self.assertEqual(tracks[0]["show_date"], "2024-11-15")
        self.assertEqual(tracks[3]["show_date"], "2024-11-16")

    def test_track_names_preserved(self):
        tracks = flatten_setlists(SAMPLE_TOUR_DATA)
        names = [t["setlist_track_name"] for t in tracks]
        self.assertIn("Tokyo (Acoustic)", names)

    def test_empty_setlist(self):
        data = {
            "data": {
                "shows": [
                    {"date": "2024-01-01", "venue": "V", "city": "C", "setlist": []}
                ]
            }
        }
        tracks = flatten_setlists(data)
        self.assertEqual(len(tracks), 0)


# ── _validate_tour_data() tests ──────────────────────────

class TestValidateTourData(unittest.TestCase):

    def test_valid_data_passes(self):
        # Should not raise
        _validate_tour_data(SAMPLE_TOUR_DATA)

    def test_missing_data_key(self):
        with self.assertRaises(ValueError):
            _validate_tour_data({"status": "ok"})

    def test_missing_shows_key(self):
        with self.assertRaises(ValueError):
            _validate_tour_data({"data": {}})

    def test_empty_shows_list(self):
        with self.assertRaises(ValueError):
            _validate_tour_data({"data": {"shows": []}})

    def test_show_missing_setlist(self):
        with self.assertRaises(ValueError):
            _validate_tour_data({
                "data": {"shows": [{"date": "2024-01-01", "venue": "V", "city": "C"}]}
            })

    def test_not_a_dict(self):
        with self.assertRaises(ValueError):
            _validate_tour_data("hello")


# ── fetch_tour_data() fallback test ──────────────────────

class TestFetchTourDataFallback(unittest.TestCase):

    @patch("ingestion.http_requests.get")
    def test_falls_back_to_local_on_http_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        # Should still succeed via local file fallback
        data = fetch_tour_data()
        self.assertIn("data", data)
        self.assertIn("shows", data["data"])


if __name__ == "__main__":
    unittest.main()
