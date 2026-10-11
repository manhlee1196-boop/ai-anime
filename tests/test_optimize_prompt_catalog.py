"""Tests for the deterministic prompt-catalog filter."""

import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.optimize_prompt_catalog import (
    PROMPT_CATEGORIES,
    optimize_catalog,
    prompt_row,
)


class PromptCatalogFilterTests(unittest.TestCase):
    def test_only_prompt_categories_and_valid_rows_are_kept(self):
        self.assertTrue(prompt_row(["solo", "0", "10", "alone", "Một nhân vật"]))
        self.assertTrue(prompt_row(["artist", "8", "2", "", "Họa sĩ"]))
        self.assertFalse(prompt_row(["highres", "5", "100", "", "Độ phân giải cao"]))
        self.assertFalse(prompt_row(["metadata", "14", "100", "", "Metadata"]))
        self.assertFalse(prompt_row(["invalid", "99", "100", "", ""]))
        self.assertFalse(prompt_row(["bad_count", "0", "x", "", ""]))
        self.assertEqual(PROMPT_CATEGORIES & {"5", "14"}, set())

    def test_filter_preserves_surviving_lines_and_is_deterministic(self):
        source = (
            "solo,0,10,alone,Một nhân vật\n"
            "meta,5,9,,Metadata\n"
            "artist,8,8,,Họa sĩ\n"
            "bad,99,7,,Không hợp lệ\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_path = root / "source.csv"
            first_path = root / "first.csv"
            second_path = root / "second.csv"
            source_path.write_text(source, encoding="utf-8")
            first = optimize_catalog(source_path, first_path)
            second = optimize_catalog(source_path, second_path)
            self.assertEqual(first, second)
            self.assertEqual(first[:2], (4, 2))
            self.assertEqual(
                first[2], hashlib.sha256(first_path.read_bytes()).hexdigest()
            )
            self.assertEqual(
                list(csv.reader(first_path.read_text(encoding="utf-8").splitlines())),
                [
                    ["solo", "0", "10", "alone", "Một nhân vật"],
                    ["artist", "8", "8", "", "Họa sĩ"],
                ],
            )

    def test_shipped_catalog_has_no_metadata_categories(self):
        path = Path(__file__).resolve().parents[1] / (
            "danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv"
        )
        with path.open(encoding="utf-8", newline="") as source:
            categories = {row[1] for row in csv.reader(source)}
        self.assertNotIn("5", categories)
        self.assertNotIn("14", categories)
        self.assertTrue(categories <= PROMPT_CATEGORIES)

    def test_studio_prompt_tags_removed_as_metadata_are_still_recognized(self):
        from colab import studio

        rows = studio.parse_tag_csv("solo,0,10,,Một nhân vật\\n")
        _, results = studio.validate_prompt_tags(
            "highres, absurdres, cel shading, traditional media", rows
        )
        self.assertTrue(results)
        self.assertTrue(all(status == "known" for _, status, _, _ in results))
