import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraping.rss_scraper import _strip_html, _make_news_id


class TestRssScraper(unittest.TestCase):
    def test_strip_html_removes_tags(self):
        raw = "<p>Notisia importante kona ba <a href='#'>governu</a>.</p>"
        result = _strip_html(raw)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<a", result)
        self.assertIn("Notisia importante", result)

    def test_strip_html_handles_empty(self):
        self.assertEqual(_strip_html(""), "")
        self.assertEqual(_strip_html(None), "")

    def test_make_news_id_deterministic(self):
        id1 = _make_news_id("https://tatoli.tl/2026/01/01/berita-a", "Berita A")
        id2 = _make_news_id("https://tatoli.tl/2026/01/01/berita-a", "Berita A")
        self.assertEqual(id1, id2)

    def test_make_news_id_unique_for_different_links(self):
        id1 = _make_news_id("https://tatoli.tl/berita-a", "Berita A")
        id2 = _make_news_id("https://tatoli.tl/berita-b", "Berita B")
        self.assertNotEqual(id1, id2)

    def test_make_news_id_fallback_to_title(self):
        id1 = _make_news_id("", "Judul Tanpa Link")
        self.assertTrue(len(id1) == 16)


if __name__ == "__main__":
    unittest.main()
