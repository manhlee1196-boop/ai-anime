"""Thứ tự sắp xếp prompt: chủ thể → chi tiết nhân vật → trang phục → tư thế → … → chất lượng → độ nét."""
import unittest

from colab import studio


class PromptOrderTests(unittest.TestCase):
    def test_sections_follow_the_documented_order(self):
        self.assertEqual(
            studio.PROMPT_SECTIONS,
            (
                "subject",
                "rating",
                "appearance",
                "outfit",
                "pose",
                "composition",
                "background",
                "lighting",
                "style",
                "extra",
                "quality",
                "tail",
            ),
        )
        self.assertEqual(set(studio.PROMPT_SECTIONS), set(studio.SECTION_LABELS))
        # Chủ thể dẫn đầu; chất lượng áp chót và độ nét luôn chốt cuối.
        self.assertEqual(studio.PROMPT_SECTIONS[0], "subject")
        self.assertEqual(studio.PROMPT_SECTIONS[-2:], ("quality", "tail"))
        for section in ("appearance", "outfit", "pose"):
            self.assertLess(
                studio.PROMPT_SECTIONS.index(section), studio.PROMPT_SECTIONS.index("background"), section
            )

    def test_character_details_come_before_background_and_quality(self):
        tags = studio.split_tags(
            "classroom, masterpiece, best quality, absurdres, 1girl, long hair, blue eyes, sitting, black dress"
        )
        ordered, _, _ = studio.sort_prompt_tags(tags)
        pos = {tag: index for index, tag in enumerate(ordered)}
        self.assertLess(pos["1girl"], pos["long hair"])
        self.assertLess(pos["blue eyes"], pos["black dress"])
        self.assertLess(pos["black dress"], pos["sitting"])
        self.assertLess(pos["sitting"], pos["classroom"])
        self.assertLess(pos["classroom"], pos["masterpiece"])
        self.assertLess(pos["masterpiece"], pos["absurdres"])
        self.assertEqual(ordered[0], "1girl")
        self.assertEqual(ordered[-1], "absurdres")

    def test_underscored_tags_are_classified_like_their_spaced_form(self):
        # Prompt Danbooru dùng `blue_eyes`; `\beyes\b` không khớp vì `_` là ký tự từ.
        for spaced, underscored in (
            ("blue eyes", "blue_eyes"),
            ("long hair", "long_hair"),
            ("from behind", "from_behind"),
            ("close up", "close_up"),
            ("soft light", "soft_light"),
        ):
            self.assertEqual(studio.classify_tag(spaced), studio.classify_tag(underscored), underscored)
        self.assertEqual(studio.classify_tag("blue_eyes"), "appearance")
        self.assertEqual(studio.classify_tag("long_hair"), "appearance")
        # Luật chứa dấu gạch dưới trong regex (rating_\w+) vẫn phải chạy trên bản gốc.
        self.assertEqual(studio.classify_tag("rating_sensitive"), "rating")

    def test_character_detail_anatomy_is_not_left_as_extra(self):
        for tag in ("big_breasts", "forked_tail", "single_horn", "dragon_wings", "muzzle", "navel"):
            self.assertEqual(studio.classify_tag(tag), "appearance", tag)

    def test_sample_prompt_is_already_in_standard_order(self):
        self.assertEqual(
            studio.sort_prompt_tags(studio.split_tags(studio.DEFAULT_PROMPT))[0],
            studio.split_tags(studio.DEFAULT_PROMPT),
        )

    def test_structure_prompt_reports_the_new_order(self):
        prompt, note = studio.structure_prompt("classroom, 1girl, blue eyes, best quality", "character")
        self.assertTrue(prompt.startswith("1girl"), prompt)
        self.assertIn("chủ thể → nhãn phân loại → ngoại hình", note)
        self.assertLess(prompt.index("blue eyes"), prompt.index("classroom"))
        self.assertLess(prompt.index("best quality"), prompt.index("absurdres"))
        # Thẻ neo chỉ thêm khi nhóm trống, không nhồi thêm thẻ chất lượng.
        self.assertEqual(prompt.count("quality"), 1)


if __name__ == "__main__":
    unittest.main()
