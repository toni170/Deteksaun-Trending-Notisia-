import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bs4 import BeautifulSoup
from src.scraping.web_scraper import (
    _extract_content, _get_text_by_selectors, _get_attr_by_selectors, DEFAULT_SELECTORS,
)

SAMPLE_HTML = """
<html>
<head>
  <meta property="og:title" content="Governu Anunsia Polítika Foun" />
  <meta property="og:image" content="https://tatoli.tl/wp-content/uploads/2026/08/foto.jpg" />
  <meta property="article:published_time" content="2026-08-30T09:00:00+09:00" />
</head>
<body>
  <div class="cat-links">
    <span class="cat-links"><a href="#">Nasional</a></span>
  </div>
  <article>
    <div class="entry-content">
      <p>Governu Timor-Leste anunsia polítika foun ba setor saúde iha semana ne'e.</p>
      <p>Polítika ne'e sei fó impaktu diak ba komunidade sira iha area rural.</p>
      <div class="sharedaddy">Bagikan artikel ini di media sosial</div>
    </div>
  </article>
  <time datetime="2026-08-30T09:00:00+09:00">30 Agustus 2026</time>
</body>
</html>
"""


class TestWebScraperExtraction(unittest.TestCase):
    def setUp(self):
        self.soup = BeautifulSoup(SAMPLE_HTML, "lxml")

    def test_extract_content_picks_entry_content(self):
        content = _extract_content(self.soup, DEFAULT_SELECTORS["content"])
        self.assertIn("Governu Timor-Leste anunsia", content)
        self.assertIn("impaktu diak", content)

    def test_extract_content_removes_junk(self):
        content = _extract_content(self.soup, DEFAULT_SELECTORS["content"])
        self.assertNotIn("Bagikan artikel", content)

    def test_get_text_by_selectors_category(self):
        category = _get_text_by_selectors(self.soup, DEFAULT_SELECTORS["category"])
        self.assertEqual(category, "Nasional")

    def test_get_attr_by_selectors_image(self):
        image_url = _get_attr_by_selectors(self.soup, DEFAULT_SELECTORS["image_meta"], "content")
        self.assertTrue(image_url.startswith("https://tatoli.tl/"))

    def test_get_attr_by_selectors_date_meta(self):
        date_val = _get_attr_by_selectors(self.soup, DEFAULT_SELECTORS["date_meta"], "content")
        self.assertEqual(date_val, "2026-08-30T09:00:00+09:00")

    def test_get_attr_by_selectors_missing_returns_empty(self):
        result = _get_attr_by_selectors(self.soup, ["meta[property='nonexistent']"], "content")
        self.assertEqual(result, "")

    def test_extract_content_no_match_returns_empty(self):
        empty_soup = BeautifulSoup("<html><body><p>no article here</p></body></html>", "lxml")
        content = _extract_content(empty_soup, ["div.entry-content"])
        self.assertEqual(content, "")


if __name__ == "__main__":
    unittest.main()
