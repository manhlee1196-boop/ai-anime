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
            'blue_eyes,0,50,,Mắt xanh dương\r\n'
            'light_blue_eyes,0,35,,Mắt xanh dương nhạt\r\n'
            'green_eyes,0,30,,Mắt xanh lá\r\n'
            'red_eyes,0,25,,Mắt đỏ\r\n'
            'heterochromia,0,20,,Mắt hai màu\r\n'
            'eyes,0,10,,Mắt\r\n'
            'face,0,15,,Mặt\r\n'
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
            'example_artist,1,20,\r\n'
            'e621_artist,8,10,'
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
        self.assertEqual(total, 4)
        self.assertEqual(
            {row[0] for row in found},
            {"long_hair", "blue_eyes", "light_blue_eyes", "city"},
        )
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

        eye_choices, note = studio.get_keyword_tag_suggestions("1girl, mắt", rows=rows)
        eye_values = {value for _, value in eye_choices}
        self.assertIn("blue_eyes", eye_values)
        self.assertIn("[G] blue eyes", {label for label, _ in eye_choices})
        self.assertEqual(len(eye_choices), studio.PROMPT_TAG_SUGGESTION_LIMIT)
        self.assertIn("tag từ CSV", note)
        english_choices, _ = studio.get_keyword_tag_suggestions("1girl, eyes", rows=rows)
        self.assertIn("blue_eyes", {value for _, value in english_choices})
        self.assertTrue(all(value in {row[0] for row in rows} for _, value in eye_choices))

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

    def test_app_startup_preloads_catalog_and_recovers_gracefully_on_failure(self):
        output = io.StringIO()
        with (
            patch.object(studio, "load_csv_tags", return_value=self.rows) as load,
            contextlib.redirect_stdout(output),
        ):
            status = studio.prime_prompt_tag_catalog()
        load.assert_called_once_with()
        self.assertIn(f"{len(self.rows)} tag", status)
        self.assertIn("SHA-256", status)
        self.assertIn("nạp và xác minh", output.getvalue())

        with (
            patch.object(
                studio, "load_csv_tags", side_effect=[OSError("offline"), self.rows]
            ) as load,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = studio.prime_prompt_tag_catalog()
            choices, _ = studio.get_keyword_tag_suggestions("1girl, eyes")
        self.assertEqual(load.call_count, 2)
        self.assertIn("OSError", status)
        self.assertIn("thử lại khi bạn gõ", status)
        self.assertIn("vẫn hoạt động", status)
        self.assertIn("blue_eyes", {value for _, value in choices})

    def test_keyword_suggestions_search_vietnamese_and_english_in_catalog(self):
        eyes, note = studio.get_keyword_tag_suggestions("1girl, mắt", rows=self.rows)
        english, _ = studio.get_keyword_tag_suggestions("1girl, eyes", rows=self.rows)
        blue, _ = studio.get_keyword_tag_suggestions("1girl, mắt xanh", rows=self.rows)
        accentless, _ = studio.get_keyword_tag_suggestions("1girl, mat", rows=self.rows)

        eye_values = {tag for _, tag in eyes}
        english_values = {tag for _, tag in english}
        self.assertIn("blue_eyes", eye_values)
        self.assertIn("heterochromia", eye_values)
        self.assertIn("[G] blue eyes", {label for label, _ in eyes})
        self.assertIn("tag từ CSV", note)
        self.assertIn("blue_eyes", english_values)
        self.assertIn("light_blue_eyes", english_values)
        self.assertTrue({
            "blue_eyes", "light_blue_eyes", "green_eyes"
        }.issubset({tag for _, tag in blue}))
        self.assertIn("eyes", {tag for _, tag in accentless})
        self.assertEqual(
            {tag for _, tag in studio.get_keyword_tag_suggestions("1girl, mặt", rows=self.rows)[0]},
            {"blush", "face"},
        )

    def test_keyword_suggestions_use_any_catalog_topic_and_limit_choices(self):
        hair, _ = studio.get_keyword_tag_suggestions("1girl, tóc", rows=self.rows)
        bows, _ = studio.get_keyword_tag_suggestions("1girl, nơ", rows=self.rows)
        dresses, _ = studio.get_keyword_tag_suggestions("1girl, váy", rows=self.rows)
        self.assertIn("long_hair", {tag for _, tag in hair})
        self.assertIn("bow", {tag for _, tag in bows})
        self.assertIn("hair_bow", {tag for _, tag in bows})
        self.assertIn("red_dress", {tag for _, tag in dresses})
        limited, _ = studio.get_keyword_tag_suggestions(
            "1girl, eyes", rows=self.rows, limit=2
        )
        self.assertEqual(len(limited), 2)
        self.assertEqual(
            studio.get_keyword_tag_suggestions("1girl, a", rows=self.rows)[0], []
        )

    def test_semi_auto_tag_patterns_filter_artists_and_search_suffix_or_middle(self):
        artists, artist_note = studio.get_keyword_tag_suggestions(
            "1girl, @example", rows=self.rows
        )
        self.assertIn("example_artist", {tag for _, tag in artists})
        self.assertIn("[A] example artist", {label for label, _ in artists})
        self.assertIn("`@` chỉ lọc họa sĩ", artist_note)

        e621_artists, _ = studio.get_keyword_tag_suggestions(
            "1girl, @e621", rows=self.rows
        )
        self.assertIn("e621_artist", {tag for _, tag in e621_artists})
        not_artist, _ = studio.get_keyword_tag_suggestions("1girl, @city", rows=self.rows)
        self.assertEqual(not_artist, [])

        suffix, suffix_note = studio.get_keyword_tag_suggestions(
            "1girl, *hair", rows=self.rows
        )
        self.assertTrue({"long_hair", "red_hair", "multicolored_hair"}.issubset(
            {tag for _, tag in suffix}
        ))
        self.assertIn("kết thúc", suffix_note)
        middle, middle_note = studio.get_keyword_tag_suggestions(
            "1girl, *hair*", rows=self.rows
        )
        middle_values = {tag for _, tag in middle}
        self.assertTrue({"long_hair", "hair_bow", "red_hair"}.issubset(middle_values))
        self.assertIn("chứa", middle_note)

    def test_keyword_selection_replaces_final_phrase_with_canonical_catalog_tag(self):
        updated, note = studio.apply_keyword_tag_suggestion(
            "1girl, long hair, mắt", "blue_eyes", rows=self.rows
        )
        self.assertEqual(updated, "1girl, long hair, blue_eyes")
        self.assertIn("thay `mắt`", note)
        self.assertIn("từ CSV", note)

        updated, _ = studio.apply_keyword_tag_suggestion(
            "1girl;  mắt xanh  ", "light_blue_eyes", rows=self.rows
        )
        self.assertEqual(updated, "1girl;  light_blue_eyes  ")

        unchanged, note = studio.apply_keyword_tag_suggestion(
            "1girl, mắt", "red_dress", rows=self.rows
        )
        self.assertEqual(unchanged, "1girl, mắt")
        self.assertIn("không còn trong CSV", note)
        unchanged, _ = studio.apply_keyword_tag_suggestion(
            "1girl, portrait", "blue_eyes", rows=self.rows
        )
        self.assertEqual(unchanged, "1girl, portrait")

    def test_semi_auto_weight_controls_adjust_last_visible_tag_in_point_one_steps(self):
        boosted, note = studio.adjust_prompt_tag_weight("1girl, blue_eyes", 1)
        self.assertEqual(boosted, "1girl, (blue_eyes:1.1)")
        self.assertIn("tăng trọng số", note)

        normal, _ = studio.adjust_prompt_tag_weight(boosted, -1)
        self.assertEqual(normal, "1girl, blue_eyes")
        lowered, _ = studio.adjust_prompt_tag_weight("1girl; (blue_eyes:1.2)  ", -1)
        self.assertEqual(lowered, "1girl; (blue_eyes:1.1)  ")
        at_limit, note = studio.adjust_prompt_tag_weight("1girl, (blue_eyes:2.0)", 1)
        self.assertEqual(at_limit, "1girl, (blue_eyes:2.0)")
        self.assertIn("giới hạn", note)

        selected = (
            "1girl, " + studio.PROMPT_TAG_WEIGHT_SELECTION_START + "blue_eyes"
            + studio.PROMPT_TAG_WEIGHT_SELECTION_END + ", long_hair"
        )
        adjusted, note = studio.adjust_prompt_tag_weight(selected, 1)
        self.assertEqual(adjusted, "1girl, (blue_eyes:1.1), long_hair")
        self.assertIn("đang chọn/con trỏ", note)

        selected_weight, _ = studio.adjust_prompt_tag_weight(
            "1girl, " + studio.PROMPT_TAG_WEIGHT_SELECTION_START
            + "(blue_eyes:1.2)" + studio.PROMPT_TAG_WEIGHT_SELECTION_END
            + ", long_hair",
            -1,
        )
        self.assertEqual(selected_weight, "1girl, (blue_eyes:1.1), long_hair")

        # Một đoạn bôi đen nhiều tag được bọc thành nhóm, giống ComfyUI/WebUI,
        # và lần bấm sau chỉnh đúng trọng số của nhóm đó.
        marked = studio.PROMPT_TAG_WEIGHT_SELECTION_START + "long_hair, blue_eyes"
        marked += studio.PROMPT_TAG_WEIGHT_SELECTION_END
        grouped, _ = studio.adjust_prompt_tag_weight("1girl, " + marked, 1)
        self.assertEqual(grouped, "1girl, (long_hair, blue_eyes:1.1)")
        regrouped, _ = studio.adjust_prompt_tag_weight(grouped, 1)
        self.assertEqual(regrouped, "1girl, (long_hair, blue_eyes:1.2)")

        # Ký tự đánh dấu không bao giờ sót lại, kể cả khi không chỉnh được gì.
        for prompt in (
            "",
            studio.PROMPT_TAG_WEIGHT_SELECTION_START
            + studio.PROMPT_TAG_WEIGHT_SELECTION_END,
            "1girl, " + studio.PROMPT_TAG_WEIGHT_SELECTION_START
            + "(blue_eyes:2.0)" + studio.PROMPT_TAG_WEIGHT_SELECTION_END,
        ):
            kept, _ = studio.adjust_prompt_tag_weight(prompt, 1)
            with self.subTest(prompt=prompt):
                self.assertNotIn(studio.PROMPT_TAG_WEIGHT_SELECTION_START, kept)
                self.assertNotIn(studio.PROMPT_TAG_WEIGHT_SELECTION_END, kept)
        message = studio.adjust_prompt_tag_weight(
            studio.PROMPT_TAG_WEIGHT_SELECTION_START
            + studio.PROMPT_TAG_WEIGHT_SELECTION_END,
            1,
        )[1]
        self.assertTrue(message.startswith("Đặt con trỏ"), message)

    def test_keyword_status_explains_minimum_length_and_no_results(self):
        choices, note = studio.get_keyword_tag_suggestions("1girl, a", rows=self.rows)
        self.assertEqual(choices, [])
        self.assertIn("ít nhất", note)
        choices, note = studio.get_keyword_tag_suggestions(
            "1girl, nonexistent topic", rows=self.rows
        )
        self.assertEqual(choices, [])
        self.assertIn("CSV chưa tìm thấy", note)

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


class VietnameseGlossaryTests(unittest.TestCase):
    """Từ điển mở rộng phải thắng cột chú giải cũ và không sinh nhãn vô nghĩa."""

    def test_new_glossary_label_wins_over_stale_csv_fallback(self):
        rows = studio.parse_tag_csv("long_hair,0,100,,Chưa có bản dịch\r\n")
        self.assertEqual(rows[0][5], "Tóc dài")

    def test_glossary_never_invents_a_translation_for_proper_names(self):
        rows = studio.parse_tag_csv("hatsune_miku,4,900,,Nhân vật\r\nsome_artist,1,800,,Họa sĩ\r\n")
        self.assertEqual([row[0] for row in rows], ["hatsune_miku", "some_artist"])
        self.assertEqual(rows[0][5], "Nhân vật")  # nhãn loại, không phải bản dịch
        self.assertFalse(studio._is_translated_tag_label(rows[0][5], rows[0][1]))

    def test_csv_only_label_is_kept_when_the_glossary_is_silent(self):
        rows = studio.parse_tag_csv("zzz_custom_tag,0,10,,Nhãn tự viết trong CSV\r\n")
        self.assertEqual(rows[0][5], "Nhãn tự viết trong CSV")

    def test_composition_uses_vietnamese_word_order(self):
        self.assertEqual(studio.vietnamese_tag_label("rabbit_ear_hat", "0"), "Mũ tai thỏ")
        self.assertEqual(studio.vietnamese_tag_label("cat_eye_glasses", "0"), "Kính mắt mèo")
        self.assertEqual(studio.vietnamese_tag_label("striped_skirt", "0"), "Chân váy kẻ sọc")
        self.assertEqual(studio.vietnamese_tag_label("leather_jacket", "0"), "Áo khoác da")
        self.assertEqual(studio.vietnamese_tag_label("black_bra", "0"), "Áo ngực màu đen")
        # Không có trong từ điển/quy tắc → giữ nguyên nhãn loại, không bịa dịch.
        self.assertEqual(studio.vietnamese_tag_label("qlty_mpd000x0v0024", "4"), "Nhân vật")

    def test_glossary_labels_are_clean_single_phrases(self):
        self.assertGreater(len(studio.TAG_VI_LABELS), 1200)
        for tag, label in studio.TAG_VI_LABELS.items():
            self.assertTrue(label and label == label.strip(), tag)
            self.assertNotIn(",", label)
            self.assertNotIn("\n", label)
            self.assertNotIn('"', label)
            self.assertNotEqual(label, tag.replace("_", " "))
            self.assertNotEqual(label, studio.TAG_VI_TRANSLATION_FALLBACK)

    def test_glossary_contains_no_foreign_scripts(self):
        foreign = {
            tag: label
            for tag, label in studio.TAG_VI_LABELS.items()
            if any("\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9fff" or "\ub800" <= ch <= "\ub8ff"
                   for ch in label)
        }
        self.assertEqual(foreign, {})

    def test_translate_file_covers_the_common_tags(self):
        rows = studio.load_csv_tags()
        names = {row[0] for row in rows if studio._is_translated_tag_label(row[5], row[1])}
        top = {row[0] for row in rows[:500]}
        self.assertGreater(len(top & names) / 500, 0.9)
