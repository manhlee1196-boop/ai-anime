"""Tests for the exclusive prompt-catalog groups."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.split_prompt_catalog import GROUPS, classify_row, split_catalog


class PromptCatalogSplitTests(unittest.TestCase):
    def test_examples_are_assigned_to_the_expected_groups(self):
        def row(name, category="0"):
            return [name, category, "1", "", ""]

        self.assertEqual(classify_row(row("school_uniform")), "05_trang_phuc_quan_ao")
        self.assertEqual(classify_row(row("red_ribbon")), "06_phu_kien")
        self.assertEqual(classify_row(row("long_hair")), "07_ngoai_hinh")
        self.assertEqual(classify_row(row("smiling")), "08_tu_the_bieu_cam")
        self.assertEqual(classify_row(row("forest_background")), "09_boi_canh")
        self.assertEqual(classify_row(row("watercolor")), "10_phong_cach")
        self.assertEqual(classify_row(row("soft_lighting")), "11_anh_sang_mau_sac")
        self.assertEqual(classify_row(row("close_up")), "12_bo_cuc_ky_thuat")
        self.assertEqual(classify_row(row("sword")), "13_vat_the")
        self.assertEqual(classify_row(row("artist", "1")), "02_hoa_si")
        self.assertEqual(classify_row(row("character", "4")), "03_tac_pham_nhan_vat")
        self.assertEqual(classify_row(row("canid", "12")), "04_loai_lore")
        self.assertEqual(classify_row(row("unclassified_custom")), "99_khac")

    def test_split_is_lossless_and_exclusive_for_a_fixture(self):
        source = (
            "school_uniform,0,10,,Đồng phục\n"
            "ribbon,0,9,,Ruy băng\n"
            "forest,0,8,,Rừng\n"
            "mystery,0,7,,\n"
            "artist_name,1,6,,Họa sĩ\n"
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "catalog.csv"
            output = root / "groups"
            input_path.write_text(source, encoding="utf-8")
            manifest = split_catalog(input_path, output)
            self.assertEqual(manifest["source_rows"], 5)
            self.assertEqual(manifest["group_rows"], 5)
            self.assertEqual(len(manifest["groups"]), len(GROUPS))
            names = []
            for group in manifest["groups"]:
                with (output / group["file"]).open(encoding="utf-8", newline="") as handle:
                    names.extend(row[0] for row in csv.reader(handle))
            self.assertEqual(sorted(names), ["artist_name", "forest", "mystery", "ribbon", "school_uniform"])
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(json.loads((output / "manifest.json").read_text())["group_rows"], 5)

    def test_committed_groups_cover_the_shipped_catalog_once(self):
        root = Path(__file__).resolve().parents[1]
        catalog = root / "danbooru_e621_merged_2026-10-01_pt20-ia-dd-ed-spc.csv"
        grouped = root / "prompt_catalog"
        manifest = json.loads((grouped / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_rows"], 348716)
        self.assertEqual(manifest["group_rows"], 348716)
        source_names = set()
        with catalog.open(encoding="utf-8", newline="") as handle:
            source_names = {row[0] for row in csv.reader(handle)}
        grouped_names = []
        for group in manifest["groups"]:
            with (grouped / group["file"]).open(encoding="utf-8", newline="") as handle:
                grouped_names.extend(row[0] for row in csv.reader(handle))
        self.assertEqual(len(grouped_names), 348716)
        self.assertEqual(len(grouped_names), len(set(grouped_names)))
        self.assertEqual(set(grouped_names), source_names)
