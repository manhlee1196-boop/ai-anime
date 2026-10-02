"""CSV tag browser tests without GPU, Gradio or network."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from colab import studio


class TagCatalogTests(unittest.TestCase):
    def setUp(self):
        self.rows = studio.parse_tag_csv(
            'long_hair,0,100,"longhair,hair_long"\r\n'
            'blue_eyes,0,50,\r\ncity,7,30,town\r\nexample_artist,1,20,')

    def test_search_filter_sort_and_page(self):
        rows, total, page, pages = studio.search_csv_tags(self.rows, 'LONG HAIR', '0', 'Ngoại hình')
        self.assertEqual(rows[0][0], 'long_hair')
        self.assertEqual((total, page, pages), (1, 1, 1))
        self.assertEqual(studio.search_csv_tags(self.rows, 'longhair')[1], 1)
        self.assertEqual(studio.search_csv_tags(self.rows, category='1')[1], 1)
        self.assertEqual(studio.search_csv_tags(self.rows, theme='Chưa phân nhóm')[1], 1)
        self.assertEqual(studio.search_csv_tags(self.rows, sort='Tên A–Z')[0][0][0], 'blue_eyes')
        self.assertEqual(studio.search_csv_tags(self.rows, query='missing', page=999)[1:], (0, 1, 1))
        with patch.object(studio, 'TAG_PAGE_SIZE', 2):
            self.assertEqual(len(studio.search_csv_tags(self.rows, page=2)[0]), 2)

    def test_insert_only_selected_canonical_tags_and_normalize_duplicates(self):
        with patch.object(studio, 'load_csv_tags', return_value=self.rows):
            positive, negative, note = studio.apply_csv_tags('long hair', 'bad quality', ['long_hair', 'city', 'city', 'fake'], 'Prompt')
            self.assertEqual(positive, 'long hair, city')
            self.assertEqual(negative, 'bad quality')
            self.assertIn('1 thẻ', note)
            positive, negative, _ = studio.apply_csv_tags(positive, negative, ['blue_eyes'], 'Negative prompt')
            self.assertEqual(negative, 'bad quality, blue_eyes')
            self.assertEqual(studio.apply_csv_tags(positive, negative, [], 'Prompt')[:2], (positive, negative))

    def test_real_resource_hash_and_count(self):
        path = Path(__file__).resolve().parents[1] / studio.TAG_CSV_NAME
        data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), studio.TAG_CSV_SHA256)
        self.assertEqual(len(studio.parse_tag_csv(data.decode('utf-8'))), 349714)

    def test_vietnamese_captions_search_and_canonical_prompt(self):
        self.assertEqual(studio.csv_tag_caption("long_hair", "0"), "Tóc dài — long_hair")
        self.assertEqual(studio.csv_tag_caption("blue_eyes", "0"), "Mắt xanh dương — blue_eyes")
        self.assertEqual(studio.csv_tag_caption("example_artist", "1"), "Họa sĩ — example_artist")
        self.assertEqual(studio.csv_tag_caption("untranslated", "0"), "Chưa có bản dịch — untranslated")
        found, total, _, _ = studio.search_csv_tags(self.rows, "tóc dài")
        self.assertEqual(total, 1)
        self.assertEqual(found[0][0], "long_hair")
        with patch.object(studio, "load_csv_tags", return_value=self.rows):
            positive, _, _ = studio.apply_csv_tags("", "", [found[0][0]], "Prompt")
        self.assertEqual(positive, "long_hair")

    def test_reject_invalid_csv(self):
        with self.assertRaises(ValueError):
            studio.parse_tag_csv('<html>Not Found</html>')

    def test_local_catalog_is_cached_without_network(self):
        with patch.object(studio, '_TAG_ROWS', None), patch('urllib.request.urlopen', side_effect=AssertionError('unexpected network')):
            rows = studio.load_csv_tags()
            self.assertEqual(len(rows), 349714)
            self.assertIs(rows, studio.load_csv_tags())
