"""Tests for the selected Vietnamese prompt-catalog files."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_prompt_catalog_vi import SELECTED_GROUPS, build_group_translations


class VietnamesePromptCatalogTests(unittest.TestCase):
    def test_build_writes_selected_groups_and_machine_fallbacks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            groups = root / "groups"
            groups.mkdir()
            (groups / "01_chu_the.csv").write_text(
                "solo,0,10,,Một nhân vật\nunknown,0,1,,\n", encoding="utf-8"
            )
            (groups / "04_loai_lore.csv").write_text(
                "canid,12,9,,Họ chó\n", encoding="utf-8"
            )
            (groups / "99_khac.csv").write_text(
                "misc,0,8,,Khác\nmissing_tag,0,7,,\n", encoding="utf-8"
            )
            (groups / "02_hoa_si.csv").write_text(
                "artist,1,8,,Họa sĩ\n", encoding="utf-8"
            )
            for group_id in SELECTED_GROUPS:
                (groups / f"{group_id}.csv").touch(exist_ok=True)
            translation = root / "translations.csv"
            translation.write_text(
                "solo,0,Một nhân vật\n"
                "canid,12,Họ chó\n"
                "misc,0,Khác\n"
                "artist,1,Họa sĩ\n"
                "not_in_group,0,Không được lấy\n",
                encoding="utf-8",
            )
            output = root / "vi"
            manifest = build_group_translations(groups, translation, output)
            self.assertEqual(manifest["translated_rows"], 4)
            self.assertEqual(manifest["real_translated_rows"], 3)
            self.assertEqual(manifest["machine_translated_rows"], 1)
            self.assertEqual(set(manifest["selected_groups"]), set(SELECTED_GROUPS))
            with (output / "01_chu_the.csv").open(encoding="utf-8") as source:
                self.assertEqual(list(csv.reader(source)), [["solo", "0", "Một nhân vật"]])
            with (output / "04_loai_lore.csv").open(encoding="utf-8") as source:
                self.assertEqual(list(csv.reader(source)), [["canid", "12", "Họ chó"]])
            with (output / "99_khac.csv").open(encoding="utf-8") as source:
                rows = list(csv.reader(source))
            self.assertEqual(rows[0], ["misc", "0", "Khác"])
            self.assertEqual(rows[1], ["missing_tag", "0", "Nhãn thiếu"])
            self.assertNotIn("_", rows[1][2])
            self.assertFalse((output / "02_hoa_si.csv").exists())

    def test_committed_manifest_contains_the_requested_groups(self):
        root = Path(__file__).resolve().parents[1] / "prompt_catalog_vi_vn"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["selected_groups"], list(SELECTED_GROUPS))
        self.assertEqual(manifest["translated_rows"], 64684)
        self.assertEqual(manifest["real_translated_rows"], 29436)
        self.assertEqual(manifest["machine_translated_rows"], 35248)
        self.assertEqual(len(manifest["groups"]), 12)
        self.assertIn("99_khac", manifest["selected_groups"])
        for group in manifest["groups"]:
            path = root / group["file"]
            self.assertTrue(path.is_file())
            with path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.reader(source))
            self.assertEqual(len(rows), group["rows"])
            self.assertTrue(all(len(row) == 3 and "," not in row[2] for row in rows))
            if group["id"] == "99_khac":
                self.assertEqual(group["rows"], 43364)
                self.assertEqual(group["real_translated_rows"], 8116)
                self.assertEqual(group["machine_translated_rows"], 35248)
                self.assertTrue(all("_" not in row[2] for row in rows))
