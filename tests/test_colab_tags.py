"""CSV tag browser tests without GPU, Gradio or network."""
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from colab import studio
from scripts.add_vietnamese_tag_captions import enrich_csv


class TagCatalogTests(unittest.TestCase):
    def setUp(self):
        self.rows = studio.parse_tag_csv(
            'long_hair,0,100,"longhair,hair_long"\r\n'
            'blue_eyes,0,50,\r\n'
            '1girl,0,80,,Một nhân vật nữ\r\n'
            'red_dress,0,40,,Váy đỏ\r\n'
            'bow,0,100,no,Nơ\r\n'
            'hair_bow,0,80,bow in hair,Nơ cài tóc\r\n'
            'bowtie,0,70,,Nơ cổ\r\n'
            'no_humans,0,500,,Không có người\r\n'
            'nockers,0,400,,Chưa có bản dịch\r\n'
            'highres,5,1000,do phan giai cao,Độ phân giải cao\r\n'
            'blush,0,900,do mat,Đỏ mặt\r\n'
            'red_hair,0,200,,Tóc đỏ\r\n'
            'multicolored_hair,0,800,black and red hair,Tóc nhiều màu\r\n'
            'city,7,30,town\r\n'
            'example_artist,1,20,'
        )

    def test_search_filter_sort_and_page(self):
        rows, total, page, pages = studio.search_csv_tags(
            self.rows, "LONG HAIR", "0", "Ngoại hình"
        )
        self.assertEqual(rows[0][0], "long_hair")
        self.assertEqual((total, page, pages), (1, 1, 1))
        self.assertEqual(studio.search_csv_tags(self.rows, "longhair")[1], 1)
        self.assertEqual(studio.search_csv_tags(self.rows, category="1")[1], 1)
        self.assertEqual(
            studio.search_csv_tags(self.rows, theme="Chưa phân nhóm")[1],
            sum(not row[4] for row in self.rows),
        )
        self.assertEqual(
            studio.search_csv_tags(self.rows, sort="Tên A–Z")[0][0][0], "1girl"
        )
        self.assertEqual(
            studio.search_csv_tags(self.rows, query="missing", page=999)[1:],
            (0, 1, 1),
        )
        with patch.object(studio, "TAG_PAGE_SIZE", 2):
            self.assertEqual(len(studio.search_csv_tags(self.rows, page=2)[0]), 2)

    def test_vietnamese_english_multiterm_and_accentless_search(self):
        self.assertEqual(studio.search_csv_tags(self.rows, "tóc dài")[1], 1)
        self.assertEqual(studio.search_csv_tags(self.rows, "TOC DAI")[1], 1)
        self.assertEqual(studio.search_csv_tags(self.rows, "long hair")[1], 1)
        found, total, _, _ = studio.search_csv_tags(
            self.rows, "tóc dài; blue eyes\ncity"
        )
        self.assertEqual(total, 3)
        self.assertEqual({row[0] for row in found}, {"long_hair", "blue_eyes", "city"})
        natural, _, _, _ = studio.search_csv_tags(
            self.rows, "một cô gái tóc dài, mắt xanh, váy đỏ"
        )
        natural_names = {row[0] for row in natural}
        self.assertTrue({"1girl", "long_hair", "blue_eyes", "red_dress"}.issubset(natural_names))

    def test_fifth_column_is_searchable_but_prompt_gets_canonical_tag(self):
        rows = studio.parse_tag_csv("custom_tag,0,10,,Nhãn tiếng Việt riêng\n")
        self.assertEqual(rows[0][5], "Nhãn tiếng Việt riêng")
        found, total, _, _ = studio.search_csv_tags(rows, "nhan tieng viet rieng")
        self.assertEqual(total, 1)
        with patch.object(studio, "load_csv_tags", return_value=rows):
            positive, negative, _ = studio.apply_csv_tags("", "", [found[0][0]], "Prompt")
        self.assertEqual(positive, "custom_tag")
        self.assertEqual(negative, "")

    def test_insert_only_selected_canonical_tags_and_normalize_duplicates(self):
        with patch.object(studio, "load_csv_tags", return_value=self.rows):
            positive, negative, note = studio.apply_csv_tags(
                "long hair", "bad quality", ["long_hair", "city", "city", "fake"], "Prompt"
            )
            self.assertEqual(positive, "long hair, city")
            self.assertEqual(negative, "bad quality")
            self.assertIn("1 thẻ", note)
            positive, negative, _ = studio.apply_csv_tags(
                positive, negative, ["blue_eyes"], "Negative prompt"
            )
            self.assertEqual(negative, "bad quality, blue_eyes")
            self.assertEqual(
                studio.apply_csv_tags(positive, negative, [], "Prompt")[:2],
                (positive, negative),
            )

    def test_real_resource_hash_schema_and_full_catalog_search(self):
        path = Path(__file__).resolve().parents[1] / studio.TAG_CSV_NAME
        data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), studio.TAG_CSV_SHA256)
        first_row = next(csv.reader(data.decode("utf-8").splitlines()))
        self.assertEqual(len(first_row), 5)
        self.assertEqual(first_row[4], "Một nhân vật")
        rows = studio.parse_tag_csv(data.decode("utf-8"))
        self.assertEqual(len(rows), 349714)
        found, total, _, _ = studio.search_csv_tags(rows, "tóc dài, blue eyes")
        names = {row[0] for row in found}
        self.assertGreaterEqual(total, 2)
        self.assertIn("long_hair", names)
        self.assertIn("blue_eyes", names)
        natural, _, _, _ = studio.search_csv_tags(
            rows, "một cô gái tóc dài, mắt xanh dương"
        )
        natural_names = {row[0] for row in natural}
        self.assertTrue({"1girl", "long_hair", "blue_eyes"}.issubset(natural_names))
        english, _, _, _ = studio.search_csv_tags(
            rows, "a girl with long hair, blue eyes"
        )
        english_names = {row[0] for row in english}
        self.assertTrue({"1girl", "long_hair", "blue_eyes"}.issubset(english_names))

    def test_vietnamese_captions_and_canonical_prompt(self):
        self.assertEqual(studio.csv_tag_caption("long_hair", "0"), "Tóc dài — long_hair")
        self.assertEqual(studio.csv_tag_caption("blue_eyes", "0"), "Mắt xanh dương — blue_eyes")
        self.assertEqual(studio.csv_tag_caption("example_artist", "1"), "Họa sĩ — example_artist")
        self.assertEqual(
            studio.csv_tag_caption("untranslated", "0"),
            "Chưa có bản dịch — untranslated",
        )
        found, total, _, _ = studio.search_csv_tags(self.rows, "tóc dài")
        self.assertEqual(total, 1)
        self.assertEqual(found[0][0], "long_hair")
        with patch.object(studio, "load_csv_tags", return_value=self.rows):
            positive, _, _ = studio.apply_csv_tags("", "", [found[0][0]], "Prompt")
        self.assertEqual(positive, "long_hair")

    def test_inline_suggestions_search_the_final_vietnamese_or_english_phrase(self):
        choices, note = studio.get_prompt_tag_suggestions(
            "1girl, tóc dài", rows=self.rows
        )
        self.assertIn(("Tóc dài — long_hair", "long_hair"), choices)
        self.assertIn("gợi ý", note)
        self.assertNotIn("1girl", [tag for _, tag in choices])

        english, _ = studio.get_prompt_tag_suggestions(
            "1girl, red dress", rows=self.rows
        )
        self.assertIn(("Váy đỏ — red_dress", "red_dress"), english)

        accentless, _ = studio.get_prompt_tag_suggestions(
            "1girl, TOC DAI", rows=self.rows
        )
        self.assertIn("long_hair", [tag for _, tag in accentless])

        natural, _ = studio.get_prompt_tag_suggestions(
            "1girl, một cô gái tóc dài", rows=self.rows
        )
        self.assertTrue({"1girl", "long_hair"}.issubset({tag for _, tag in natural}))

    def test_inline_vietnamese_suggestions_disambiguate_short_words_and_rank_exact_tags(self):
        bows, _ = studio.get_prompt_tag_suggestions("1girl, nơ", rows=self.rows)
        self.assertEqual(
            [tag for _, tag in bows], ["bow", "hair_bow", "bowtie"]
        )
        self.assertFalse({"no_humans", "nockers"} & {tag for _, tag in bows})
        selected_prompt, _ = studio.apply_prompt_tag_suggestion(
            "1girl, nơ", bows[0][1], rows=self.rows
        )
        self.assertEqual(selected_prompt, "1girl, nơ, bow")

        reds, _ = studio.get_prompt_tag_suggestions("1girl, đỏ", rows=self.rows)
        red_tags = {tag for _, tag in reds}
        self.assertIn("blush", red_tags)
        self.assertIn("red_hair", red_tags)
        self.assertNotIn("highres", red_tags)

        english, _ = studio.get_prompt_tag_suggestions("1girl, red hair", rows=self.rows)
        self.assertEqual(english[0], ("Tóc đỏ — red_hair", "red_hair"))

        catalog, _, _, _ = studio.search_csv_tags(self.rows, "nơ")
        self.assertEqual({row[0] for row in catalog}, {"bow", "hair_bow", "bowtie"})

    def test_inline_suggestion_appends_canonical_tag_without_losing_text_or_duplicates(self):
        with patch.object(studio, "load_csv_tags", return_value=self.rows):
            updated, note = studio.apply_prompt_tag_suggestion(
                "1girl, tóc dài ", "long_hair"
            )
            self.assertEqual(updated, "1girl, tóc dài, long_hair")
            self.assertIn("tag tiếng Anh gốc", note)
            self.assertIn("giữ nguyên", note)

            original = "1girl, long_hair, tóc dài"
            duplicated, note = studio.apply_prompt_tag_suggestion(
                original, "long_hair"
            )
            self.assertEqual(duplicated, original)
            self.assertIn("tránh trùng", note)
            semicolon_duplicate = "1girl; long_hair; tóc dài"
            self.assertEqual(
                studio.apply_prompt_tag_suggestion(
                    semicolon_duplicate, "long_hair"
                )[0],
                semicolon_duplicate,
            )

            unchanged, note = studio.apply_prompt_tag_suggestion(
                "1girl, tóc dài", "not_a_catalog_tag"
            )
            self.assertEqual(unchanged, "1girl, tóc dài")
            self.assertIn("không còn trong kho", note)

    def test_inline_suggestions_skip_short_input_without_loading_catalog(self):
        with patch.object(studio, "load_csv_tags", side_effect=AssertionError("too early")):
            choices, note = studio.get_prompt_tag_suggestions("1girl, t")
        self.assertEqual(choices, [])
        self.assertIn("ít nhất 2 ký tự", note)

    def test_caption_builder_is_repeatable_and_keeps_the_source_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tags.csv"
            path.write_text(
                'long_hair,0,100,"longhair,hair_long"\nexample_artist,1,20,\n',
                encoding="utf-8",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                enrich_csv(path)
            first = path.read_bytes()
            self.assertEqual(
                list(csv.reader(first.decode("utf-8").splitlines())),
                [
                    ["long_hair", "0", "100", "longhair,hair_long", "Tóc dài"],
                    ["example_artist", "1", "20", "", "Họa sĩ"],
                ],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                enrich_csv(path)
            self.assertEqual(path.read_bytes(), first)

    def test_reject_invalid_csv(self):
        with self.assertRaises(ValueError):
            studio.parse_tag_csv("<html>Not Found</html>")

    def test_local_catalog_is_cached_without_network(self):
        with (
            patch.object(studio, "_TAG_ROWS", None),
            patch("urllib.request.urlopen", side_effect=AssertionError("unexpected network")),
        ):
            rows = studio.load_csv_tags()
            self.assertEqual(len(rows), 349714)
            self.assertIs(rows, studio.load_csv_tags())
