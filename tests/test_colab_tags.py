"""CSV tag browser tests without GPU, Gradio or network."""
import contextlib
import csv
import hashlib
import io
import re
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

    def test_shaped_modifiers_read_naturally(self):
        self.assertTrue(
            studio.vietnamese_tag_label("heart-shaped_pupils", "0").startswith("Đồng tử hình"),
            studio.vietnamese_tag_label("heart-shaped_pupils", "0"),
        )
        self.assertEqual(studio.vietnamese_tag_label("star-shaped_background", "0"), "Nền hình ngôi sao")

    def test_vocabulary_literals_have_no_duplicate_keys(self):
        # Key trùng trong cùng một dict literal bị Python lặng lẽ ghi đè — đã xảy ra khi
        # bổ sung từ điển theo đợt, nên chốt lại bằng test.
        import ast

        source = (Path(__file__).resolve().parents[1] / "colab" / "studio.py").read_text(encoding="utf-8")
        targets = {
            "_TAG_VI_WORDS",
            "_TAG_VI_COLORS",
            "_TAG_VI_COMPOSITE_HEADS",
            "TAG_VI_LABELS",
            "TAG_VI_SEARCH_SYNONYMS",
        }
        tree = ast.parse(source)
        found = 0
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                literal = node.value if isinstance(node.value, ast.Dict) else None
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                func = call.func
                names = [func.value.id] if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else []
                literal = call.args[0] if call.args and isinstance(call.args[0], ast.Dict) else None
            else:
                continue
            if not set(names) & targets or literal is None:
                continue
            found += 1
            seen = set()
            for key in literal.keys:
                if isinstance(key, ast.Constant):
                    self.assertNotIn(key.value, seen, f"trùng khóa {key.value!r} trong {names}")
                    seen.add(key.value)
        self.assertGreater(found, 4)

    def test_adjective_modifiers_are_all_declared_in_words(self):
        missing = studio._TAG_VI_ADJECTIVE_MODIFIERS - set(studio._TAG_VI_WORDS)
        self.assertEqual(missing, set())

    # --- Đợt 3: ưu tiên thẻ chi tiết nhân vật (ngoại hình, trang phục, biểu cảm) ---
    def test_character_detail_families_are_translated(self):
        for tag, expected in (
            ("holding_sword", "Cầm kiếm"),
            ("holding_bottle", "Cầm cái chai"),
            ("wearing_hat", "Đội mũ"),
            ("wearing_dress", "Mặc váy liền"),
            ("wearing_boots", "Đi ủng"),
            ("wearing_glasses", "Đeo kính mắt"),
            ("adjusting_glasses", "Đang chỉnh kính mắt"),
            ("no_gloves", "Không có găng tay"),
            ("bandaid_on_face", "Băng cá nhân trên khuôn mặt"),
            ("flower_in_hair", "Hoa trong tóc"),
            ("bikini_under_clothes", "Đồ bơi bikini bên dưới quần áo"),
            ("hat_with_ribbon", "Mũ kèm ruy băng"),
            ("hairless", "Không có tóc"),
            ("gym_uniform", "Đồng phục thể dục"),
            ("monotone_hair", "Tóc một tông"),
            ("police_hat", "Mũ cảnh sát"),
            ("hair_beads", "Chuỗi hạt cài tóc"),
        ):
            self.assertEqual(studio.vietnamese_tag_label(tag, "0"), expected, tag)

    def test_state_suffixes_read_as_vietnamese_postmodifers(self):
        # "<x>_less"/"tied_"/"untied_" phải cho cụm đúng ngữ pháp tiếng Việt, không dịch máy.
        self.assertEqual(studio.vietnamese_tag_label("tied_shirt", "0"), "Áo sơ mi được buộc")
        self.assertEqual(studio.vietnamese_tag_label("unbuttoned_shirt", "0"), "Áo sơ mi mở khuy")
        self.assertEqual(studio.vietnamese_tag_label("untied_panties", "0"), "Quần lót cởi dây")
        self.assertEqual(studio.vietnamese_tag_label("tied_to_chair", "0"), "Trói vào ghế")
        self.assertEqual(studio.vietnamese_tag_label("holding_with_tail", "0"), "Giữ bằng đuôi")
        # Không có nhãn nào bắt đầu bằng phân từ trần trụi.
        for name, label in studio.TAG_VI_LABELS.items():
            self.assertFalse(label.startswith(("được buộc ", "cởi dây ")), name)

    def test_detail_rules_never_guess_unknown_pieces(self):
        self.assertEqual(studio.vietnamese_tag_label("no_zzz_unknown", "0"), "Chưa có bản dịch")
        self.assertEqual(studio.vietnamese_tag_label("holding_zzz_unknown", "0"), "Chưa có bản dịch")
        self.assertEqual(studio.vietnamese_tag_label("zzz_on_face", "0"), "Chưa có bản dịch")
        # Đuôi "-less" chỉ hợp lệ với bộ phận/danh từ đã biết, không phải tính từ tiếng Anh.
        for word in ("fearless", "restless", "countless", "wireless", "relentless"):
            self.assertFalse(studio._is_translated_tag_label(studio.vietnamese_tag_label(word, "0"), "0"), word)
        self.assertEqual(studio.vietnamese_tag_label("shirtless", "0"), "Không có áo sơ mi")

    def test_function_words_stay_out_of_the_noun_vocabulary(self):
        # Nếu giới từ/đại từ lọt vào WORDS hoặc HEADS sẽ sinh cụm kiểu "X trên Y" vô nghĩa.
        for token in ("on", "in", "under", "over", "with", "no", "without", "holding", "wearing"):
            self.assertNotIn(token, studio._TAG_VI_WORDS, token)
            self.assertNotIn(token, studio._TAG_VI_COMPOSITE_HEADS, token)
        # "less" chỉ được dùng làm danh từ phủ định ở cuối cụm, không phải từ bổ nghĩa tùy ý.
        self.assertNotIn("less", studio._TAG_VI_WORDS)
        self.assertEqual(studio._TAG_VI_COMPOSITE_HEADS["less"], "Không có")

    def test_pose_and_only_rules_read_as_vietnamese(self):
        # Tiếng Việt đặt động từ trước nên "<x>_only" và "<bộ phận>_<hướng>" cần mẫu riêng.
        self.assertEqual(studio.vietnamese_tag_label("hat_only", "0"), "Chỉ đội mũ")
        self.assertEqual(studio.vietnamese_tag_label("gloves_only", "0"), "Chỉ đeo găng tay")
        self.assertEqual(studio.vietnamese_tag_label("choker_only", "0"), "Chỉ đeo vòng cổ choker")
        self.assertEqual(studio.vietnamese_tag_label("skirt_down", "0"), "Kéo chân váy xuống")
        self.assertEqual(studio.vietnamese_tag_label("tail_up", "0"), "Đuôi dựng lên")
        self.assertEqual(studio.vietnamese_tag_label("ears_down", "0"), "Tai cụp xuống")
        self.assertEqual(studio.vietnamese_tag_label("eyebrows_hidden", "0"), "Lông mày bị che")
        self.assertEqual(studio.vietnamese_tag_label("shirt_tucked", "0"), "Áo sơ mi giắt vào trong")
        # Động từ nhìn/chỉ: liên từ "at" của tiếng Anh phải bị lược, không dịch thành "tại".
        self.assertEqual(studio.vietnamese_tag_label("looking_at_phone", "0"), "Nhìn điện thoại")
        self.assertEqual(studio.vietnamese_tag_label("leaning_against_wall", "0"), "Tựa vào bức tường")
        self.assertEqual(studio.vietnamese_tag_label("pointing_at_gun", "0"), "Chỉ vào súng")

    def test_body_state_and_species_heads_compose(self):
        # Danh từ chính ở CUỐI cụm (loài, nội thất) + trạng thái đứng sau.
        self.assertEqual(studio.vietnamese_tag_label("hairless_cat", "0"), "Mèo không lông")
        self.assertEqual(studio.vietnamese_tag_label("canine_ears", "0"), "Tai chó")
        self.assertEqual(studio.vietnamese_tag_label("equine_tail", "0"), "Đuôi ngựa")
        self.assertEqual(studio.vietnamese_tag_label("office_chair", "0"), "Ghế văn phòng")
        self.assertEqual(studio.vietnamese_tag_label("tail_raised", "0"), "Đuôi dựng lên")
        self.assertEqual(studio.vietnamese_tag_label("eyelids_visible", "0"), "Mí mắt nhìn thấy được")
        self.assertEqual(studio.vietnamese_tag_label("arms_crossed", "0"), "Cánh tay bắt chéo")
        # "see_through_<x>" và "<verb>_<hướng>_<tân ngữ>".
        self.assertEqual(studio.vietnamese_tag_label("see_through_shirt", "0"), "Áo sơ mi xuyên thấu")
        self.assertEqual(studio.vietnamese_tag_label("looking_down_at_viewer", "0"), "Nhìn xuống người xem")
        self.assertEqual(studio.vietnamese_tag_label("throwing_ball", "0"), "Đang ném quả bóng")
        # Tên riêng chỉ được dịch phần khung, bản thân tên giữ nguyên.
        self.assertEqual(studio.vietnamese_tag_label("mitakihara_school_uniform", "0"), "Đồng phục trường Mitakihara")
        self.assertFalse(studio._is_translated_tag_label(studio.vietnamese_tag_label("guilty_gear", "3"), "3"))

    def test_rule_only_words_do_not_leak_into_noun_vocabulary(self):
        # Các từ này chỉ hợp lệ trong quy tắc cụm; vào WORDS/HEADS sẽ sinh "X dưới Y" vô nghĩa.
        for token in ("down", "up", "aside", "away", "only", "forward", "sideways", "out"):
            self.assertNotIn(token, studio._TAG_VI_WORDS, token)
            self.assertNotIn(token, studio._TAG_VI_COMPOSITE_HEADS, token)

    def test_character_detail_coverage_is_prioritized(self):
        rows = studio.load_csv_tags()
        themes = [(name, re.compile(pattern)) for name, pattern in studio.TAG_THEMES.items()]
        detail = {"Ngoại hình", "Trang phục & phụ kiện", "Biểu cảm & tư thế"}
        for top_n, minimum in (
            (1000, 0.95), (3000, 0.95), (5000, 0.92), (10000, 0.88), (20000, 0.75)
        ):
            hit = total = 0
            for row in rows[:top_n]:
                name, category = row[0], row[1]
                if category in {"1", "8"}:
                    continue
                if not any(theme in detail for theme, pattern in themes if pattern.search(name)):
                    continue
                total += 1
                hit += studio._is_translated_tag_label(row[5], category)
            self.assertGreater(hit / total, minimum, f"top-{top_n}: {hit}/{total}")

    def test_hyphen_parenthetical_and_plural_are_resolved(self):
        # Gạch nối chỉ là ngăn cách từ: key curate có gạch nối vẫn phải thấy.
        self.assertEqual(studio.vietnamese_tag_label("see-through_dress", "0"), "Váy liền xuyên thấu")
        self.assertEqual(studio.vietnamese_tag_label("cross-eyed", "0"), "Mắt lé")
        self.assertEqual(studio.vietnamese_tag_label("t-shirt_only", "0"), "Chỉ có áo phông")
        self.assertEqual(studio.vietnamese_tag_label("text_on_t-shirt", "0"), "Chữ trong ảnh trên áo phông")
        # "(giải nghĩa)" ở cuối tên thẻ không được chặn bản dịch của thẻ gốc.
        self.assertEqual(studio.vietnamese_tag_label("pearl_(gem)", "0"), "Ngọc trai")
        self.assertEqual(studio.vietnamese_tag_label("twintails_(hairstyle)", "0"), "Tóc buộc hai bên")
        # Số nhiều quy tắc/bất quy tắc suy ra từ số ít đã có trong từ điển.
        self.assertEqual(studio.vietnamese_tag_label("scarves", "0"), studio.vietnamese_tag_label("scarf", "0"))
        # Bẫy số nhiều: "shorts" là quần, không phải tính từ "short".
        self.assertNotEqual(studio.vietnamese_tag_label("black_shorts", "0"), "Đen ngắn")

    def test_quantity_and_verb_word_order(self):
        self.assertEqual(studio.vietnamese_tag_label("three_tails", "0"), "Ba cái đuôi")
        self.assertEqual(studio.vietnamese_tag_label("multiple_arms", "0"), "Nhiều cánh tay")
        self.assertEqual(studio.vietnamese_tag_label("single_horn", "0"), "Một cái sừng")
        self.assertEqual(studio.vietnamese_tag_label("extra_tails", "0"), "Thêm đuôi")
        self.assertEqual(studio.vietnamese_tag_label("one_eye_visible", "0"), "Một con mắt nhìn thấy được")
        # Động từ tiếng Việt đặt trước danh từ dù tiếng Anh để sau: "dress_pull".
        self.assertEqual(studio.vietnamese_tag_label("dress_pull", "0"), "Kéo váy")
        self.assertEqual(studio.vietnamese_tag_label("underwear_pull", "0"), "Kéo đồ lót")
        # "of" được lược bỏ vì tiếng Việt nói "Xô sữa", không "Xô của sữa".
        self.assertEqual(studio.vietnamese_tag_label("bucket_of_milk", "0"), "Xô sữa")
        self.assertEqual(studio.vietnamese_tag_label("curved_horns", "0"), "Sừng cong")
        self.assertEqual(studio.vietnamese_tag_label("tail_lick", "0"), "Liếm đuôi")
        # Nội động từ thì ngược lại: "melting_tail" → "Đuôi đang tan chảy".
        self.assertEqual(studio.vietnamese_tag_label("melting_tail", "0"), "Đuôi đang tan chảy")
        self.assertEqual(studio.vietnamese_tag_label("ear_wiggle", "0"), "Tai lắc lư")
        # Không được đảo nhầm động từleading thành vị ngữ sau danh từ.
        self.assertEqual(studio.vietnamese_tag_label("covering_nipples", "0"), "Che núm vú")
        self.assertEqual(studio.vietnamese_tag_label("sitting_on_box", "0"), "Đang ngồi trên cái hộp")
        self.assertEqual(studio.vietnamese_tag_label("leaning_against_wall", "0"), "Tựa vào bức tường")

    def test_vocabulary_values_are_usable_single_phrases(self):
        for name, vocab in (
            ("WORDS", studio._TAG_VI_WORDS),
            ("HEADS", studio._TAG_VI_COMPOSITE_HEADS),
            ("COLORS", studio._TAG_VI_COLORS),
        ):
            for key, value in vocab.items():
                self.assertTrue(value and value == value.strip(), f"{name}:{key}")
                self.assertNotIn(",", value, f"{name}:{key}")
                self.assertNotIn("\n", value, f"{name}:{key}")
                self.assertTrue(key == key.strip() and " " not in key.strip("()"), f"{name}:{key}")


class ClothingVocabularyRound7Tests(unittest.TestCase):
    """Đợt 7: khung đồng phục theo tên riêng, sở hữu cách, động từ mặc/cởi."""

    def test_uniform_frame_keeps_proper_name_but_translates_the_frame(self):
        # Dòng đồng phục đặt theo tên trường/nhóm: chỉ dịch phần khung, không bịa tên riêng.
        self.assertEqual(
            studio.vietnamese_tag_label("tokiwadai_school_uniform", "0"),
            "Đồng phục trường Tokiwadai",
        )
        self.assertEqual(
            studio.vietnamese_tag_label("gekkoukan_high_school_uniform", "0"),
            "Đồng phục trường trung học phổ thông Gekkoukan",
        )
        self.assertEqual(
            studio.vietnamese_tag_label("garreg_mach_monastery_uniform", "0"),
            "Đồng phục tu viện Garreg Mach",
        )
        self.assertEqual(
            studio.vietnamese_tag_label("tokyo-3_middle_school_uniform", "0"),
            "Đồng phục trường trung học cơ sở Tokyo-3",
        )

    def test_uniform_frame_uppercases_abbreviation_and_stays_out_of_the_way(self):
        self.assertEqual(studio.vietnamese_tag_label("u.a._school_uniform", "0"), "Đồng phục trường U.A.")
        # Prefix toàn từ đã có bản dịch thì quy tắc ghép thắng, không coi là tên riêng.
        self.assertIsNone(studio._tag_vi_uniform_frame("red_school_uniform"))
        self.assertEqual(studio.vietnamese_tag_label("red_school_uniform", "0"), "Đồng phục học đường màu đỏ")
        self.assertEqual(studio.vietnamese_tag_label("national_soccer_team_uniform", "0"),
                         "Đồng phục đội tuyển bóng đá quốc gia")

    def test_possessive_apostrophe_reads_as_cua(self):
        self.assertEqual(studio.vietnamese_tag_label("fool's_hat", "0"), "Mũ của chú hề")
        self.assertEqual(
            studio.vietnamese_tag_label("lifting_another's_clothes", "0"),
            "Đang nhấc quần áo người khác lên",
        )
        self.assertEqual(
            studio.vietnamese_tag_label("grabbing_another's_shirt", "0"),
            "Đang nắm áo sơ mi người khác",
        )
        self.assertEqual(studio.vietnamese_tag_label("undressing_another", "0"), "Đang cởi đồ người khác")

    def test_curated_label_beats_composition_for_hyphenated_names(self):
        # "high-waist_panties": ghép từ vẫn chạy được sau khi thêm "waist" nhưng bản curate phải thắng.
        self.assertEqual(studio.vietnamese_tag_label("high-waist_panties", "0"), "Quần lót cạp cao")

    def test_verb_at_the_end_only_reorders_when_it_is_actually_a_verb(self):
        self.assertEqual(studio.vietnamese_tag_label("pseudo_skirt_lift", "0"), "Nhấc chân váy giả")
        # "piercing"/"grab" ở đây là danh từ, không được đảo thành cụm động từ.
        self.assertEqual(studio.vietnamese_tag_label("nose_piercing", "0"), "Khuyên mũi")
        self.assertEqual(studio.vietnamese_tag_label("torn_pantyhose", "0"), "Quần tất bị rách")

    def test_clothing_vocabulary_stays_clean(self):
        for key, value in studio._TAG_VI_UNIFORM_FRAMES.items():
            self.assertTrue(key and value == value.strip(), key)
            self.assertNotIn(",", value, key)
        for key, value in studio._TAG_VI_POSSESSIVES.items():
            self.assertTrue(key.endswith("'s") and value, key)


class CompositionAndBackgroundRound9Tests(unittest.TestCase):
    """Đợt 9: bố cục/kỹ thuật + bối cảnh, và khối động từ tương tác với người xem."""

    def test_interaction_with_viewer_drops_the_english_preposition(self):
        # "to/at" không dịch sang "tới/tại": tiếng Việt nói "Nói chuyện với người xem".
        self.assertEqual(
            studio.vietnamese_tag_label("talking_to_viewer", "0"), "Nói chuyện với người xem"
        )
        self.assertEqual(studio.vietnamese_tag_label("aiming_at_viewer", "0"), "Chĩa về phía người xem")
        self.assertEqual(studio.vietnamese_tag_label("feeding_viewer", "0"), "Đút cho người xem")
        self.assertEqual(studio.vietnamese_tag_label("waving_at_viewer", "0"), "Vẫy tay với người xem")

    def test_composition_frames_compose(self):
        self.assertEqual(studio.vietnamese_tag_label("duo_focus", "0"), "Trọng tâm hai nhân vật")
        self.assertEqual(studio.vietnamese_tag_label("bust_portrait", "0"), "Ảnh chân dung bán thân")
        self.assertEqual(studio.vietnamese_tag_label("high-angle_view", "0"), "Góc nhìn từ trên cao")
        self.assertEqual(studio.vietnamese_tag_label("resolution_mismatch", "0"), "Độ phân giải không khớp")

    def test_background_and_scenery_modifiers(self):
        self.assertEqual(studio.vietnamese_tag_label("blurred_background", "0"), "Nền mờ")
        self.assertEqual(studio.vietnamese_tag_label("grid_background", "0"), "Nền lưới")
        self.assertEqual(studio.vietnamese_tag_label("palm_tree", "0"), "Cây cọ")
        self.assertEqual(studio.vietnamese_tag_label("train_interior", "0"), "Bên trong tàu hỏa")

    def test_light_is_not_translated_as_a_colour_in_every_position(self):
        # "light" một mình là màu nhạt, nhưng "lights" là hệ thống đèn.
        self.assertEqual(studio.vietnamese_tag_label("lights", "0"), "Đèn")
        self.assertEqual(studio.vietnamese_tag_label("floating_lights", "0"), "Đèn lơ lửng")

    def test_uniform_frame_only_yields_to_colour_prefixes(self):
        # Thêm "dream"/"paradise" vào WORDS không được biến tên học viện thành mô tả chung.
        self.assertEqual(
            studio.vietnamese_tag_label("dream_academy_school_uniform", "0"), "Đồng phục học viện Dream"
        )
        self.assertEqual(
            studio.vietnamese_tag_label("paradise_private_academy_school_uniform", "0"),
            "Đồng phục học viện tư thục Paradise",
        )
        self.assertEqual(
            studio.vietnamese_tag_label("red_school_uniform", "0"), "Đồng phục học đường màu đỏ"
        )

    def test_color_vocabulary_stays_grammatical(self):
        self.assertEqual(studio.vietnamese_tag_label("warm_colors", "0"), "Màu ấm")
        self.assertEqual(studio.vietnamese_tag_label("muted_colors", "0"), "Màu trầm")
        self.assertEqual(studio.vietnamese_tag_label("colored_sketch", "0"), "Bản phác thảo có màu")
