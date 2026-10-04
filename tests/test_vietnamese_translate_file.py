"""Tests for the standalone Vietnamese translate file (SAA-compatible format)."""

import contextlib
import hashlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from colab import studio
from scripts.build_vietnamese_translate_file import (
    TRANSLATE_CSV_NAME,
    build_translate_file,
    render_translate_file,
    sanitize_translation,
    translate_rows,
)

ROOT = Path(__file__).resolve().parents[1]
TRANSLATE_FILE = ROOT / TRANSLATE_CSV_NAME

SAMPLE_CSV = (
    "long_hair,0,900,\"longhair\",Tóc dài\r\n"
    "solo,0,800,,Một nhân vật\r\n"
    "unknown_tag_xyz,0,700,,Chưa có bản dịch\r\n"
    "touhou,3,600,,Tác phẩm\r\n"
    "example_artist,1,500,,Họa sĩ\r\n"
    "e621_artist,8,400,,Họa sĩ\r\n"
    "hatsune_miku,4,300,,Nhân vật\r\n"
    "mammal,12,200,,Động vật có vú\r\n"
    "highres,5,100,,Độ phân giải cao\r\n"
    "long_hair,0,90,dup,Tóc dài\r\n"
    'messy_tag,0,90,,"Cái, cái này"\r\n'
    'flat_tag,0,80,,"Dòng một\nDòng hai"\r\n'
)


def js_split(line, limit):
    """Mô phỏng ``String.prototype.split(sep, limit)`` của JavaScript: phần dư bị bỏ."""
    return line.split(",")[:limit]


class TranslateRowsTests(unittest.TestCase):
    def setUp(self):
        self.rows = studio.parse_tag_csv(SAMPLE_CSV)

    def test_keeps_only_real_translations_and_skips_artist_groups(self):
        result = translate_rows(self.rows)
        self.assertEqual(
            [(name, category, translation) for name, category, translation, _ in result],
            [
                ("long_hair", "0", "Tóc dài"),
                ("solo", "0", "Một nhân vật"),
                ("mammal", "12", "Động vật có vú"),
                ("highres", "5", "Độ phân giải cao"),
                ("messy_tag", "0", "Cái; cái này"),
                ("flat_tag", "0", "Dòng một Dòng hai"),
            ],
        )

    def test_rows_are_unique_and_keep_the_first_heat_entry(self):
        names = [name for name, _category, _translation, _count in translate_rows(self.rows)]
        self.assertEqual(names.count("long_hair"), 1)
        self.assertEqual(names[0], "long_hair")

    def test_category_fallbacks_and_untranslated_tags_are_dropped(self):
        names = {name for name, _c, _t, _n in translate_rows(self.rows)}
        for missing in ("unknown_tag_xyz", "touhou", "example_artist", "e621_artist", "hatsune_miku"):
            self.assertNotIn(missing, names)

    def test_sanitize_translation_removes_commas_and_line_breaks(self):
        self.assertEqual(sanitize_translation("a, b"), "a; b")
        self.assertEqual(sanitize_translation('nói "sau"'), "nói 'sau'")
        self.assertEqual(sanitize_translation("  nhiều \n khoảng \t trắng "), "nhiều khoảng trắng")
        self.assertEqual(sanitize_translation(None), "")


class RenderFileTests(unittest.TestCase):
    def setUp(self):
        self.rows = translate_rows(studio.parse_tag_csv(SAMPLE_CSV))

    def test_render_is_three_fields_lf_and_utf8_without_bom(self):
        payload = render_translate_file(self.rows)
        self.assertFalse(payload.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", payload)
        self.assertTrue(payload.endswith(b"\n"))
        lines = payload.decode("utf-8").splitlines()
        self.assertEqual(len(lines), len(self.rows))
        for line in lines:
            self.assertEqual(len(line.split(",")), 3)

    def test_loader_style_split_never_truncates_a_translation(self):
        for line in render_translate_file(self.rows).decode("utf-8").splitlines():
            parts = js_split(line, 3)
            self.assertEqual(len(line.split(",")), 3)
            self.assertIn(parts[2], line)
            self.assertTrue(parts[1].isdigit())

    def test_bom_option_only_prepends_the_signature(self):
        plain = render_translate_file(self.rows)
        with_bom = render_translate_file(self.rows, bom=True)
        self.assertTrue(with_bom.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(with_bom[3:], plain)

    def test_empty_catalog_renders_empty_file(self):
        self.assertEqual(render_translate_file([]), b"")
        self.assertEqual(render_translate_file([], bom=True), b"")


class CommittedFileTests(unittest.TestCase):
    def test_file_exists_and_is_well_formed(self):
        raw = TRANSLATE_FILE.read_bytes()
        self.assertGreater(len(raw), 1000)
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        self.assertNotIn(b"\r", raw)
        text = raw.decode("utf-8")
        self.assertTrue(text.endswith("\n"))
        lines = text.splitlines()
        tags = []
        for line in lines:
            parts = line.split(",")
            self.assertEqual(len(parts), 3, line)
            tag, category, translation = parts
            self.assertTrue(tag and tag == tag.strip())
            self.assertNotIn('"', line)
            self.assertIn(category, studio.TAG_CATEGORIES)
            self.assertNotIn(category, ("1", "8"))
            self.assertTrue(translation and translation == translation.strip())
            self.assertNotEqual(translation.replace("_", " ").lower(), tag.lower())
            self.assertNotEqual(translation, studio._TAG_VI_CATEGORY_FALLBACKS.get(category))
            self.assertNotEqual(translation, studio.TAG_VI_TRANSLATION_FALLBACK)
            tags.append(tag)
        self.assertEqual(len(tags), len(set(tags)), "mỗi thẻ phải xuất hiện đúng một lần")
        self.assertGreater(len(tags), 1000)

    def test_every_row_matches_the_studio_label(self):
        rows = {name: (category, label) for name, category, _n, _i, _t, label in studio.load_csv_tags()}
        for line in TRANSLATE_FILE.read_text(encoding="utf-8").splitlines():
            tag, category, translation = line.split(",")
            studio_category, studio_label = rows[tag]
            self.assertEqual(studio_category, category)
            self.assertEqual(translation, sanitize_translation(studio_label))

    def test_regeneration_is_byte_identical(self):
        committed = TRANSLATE_FILE.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / TRANSLATE_CSV_NAME
            with contextlib.redirect_stdout(io.StringIO()):
                build_translate_file(input_path=ROOT / studio.TAG_CSV_NAME, output_path=destination)
            self.assertEqual(destination.read_bytes(), committed)
            digest = hashlib.sha256(committed).hexdigest()
            self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), digest)

    def test_check_mode_accepts_the_committed_file_without_writing(self):
        before = TRANSLATE_FILE.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            stats = build_translate_file(
                input_path=ROOT / studio.TAG_CSV_NAME,
                output_path=TRANSLATE_FILE,
                check=True,
            )
        self.assertGreater(stats["translated"], 1000)
        self.assertEqual(stats["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(TRANSLATE_FILE.read_bytes(), before)


class CommandLineTests(unittest.TestCase):
    def test_cli_check_and_write_to_temporary_path(self):
        script = ROOT / "scripts" / "build_vietnamese_translate_file.py"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / TRANSLATE_CSV_NAME
            write = subprocess.run(
                [sys.executable, str(script), "--output", str(destination)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(write.returncode, 0, write.stderr)
            self.assertIn("Đã viết", write.stdout)
            self.assertTrue(destination.is_file())
            self.assertEqual(destination.read_bytes(), TRANSLATE_FILE.read_bytes())

            destination.write_text("solo,0,old\n", encoding="utf-8")
            stale = subprocess.run(
                [sys.executable, str(script), "--output", str(destination), "--check"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(stale.returncode, 1)
            self.assertIn("--check lệch", stale.stdout)
            self.assertEqual(destination.read_text(encoding="utf-8"), "solo,0,old\n")

    def test_parser_does_not_import_gradio(self):
        script = ROOT / "scripts" / "build_vietnamese_translate_file.py"
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import scripts.build_vietnamese_translate_file as m;"
             " sys.exit(1 if 'gradio' in sys.modules else 0)"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
