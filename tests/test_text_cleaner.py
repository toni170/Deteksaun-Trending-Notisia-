import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.text_cleaner import clean_text, detect_language, preprocess_news


class TestTextCleaner(unittest.TestCase):
    def test_clean_text_removes_html_and_lowercase(self):
        result = clean_text("<p>Berita PENTING!!!</p> https://tatoli.tl/x")
        self.assertNotIn("<p>", result)
        self.assertNotIn("http", result)
        self.assertEqual(result, result.lower())

    def test_clean_text_empty(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")

    def test_detect_language_indonesian(self):
        lang = detect_language("pemerintah akan membangun jalan baru di daerah ini dengan anggaran besar")
        self.assertEqual(lang, "id")

    def test_detect_language_english(self):
        lang = detect_language("the government announced a new policy for this year and next")
        self.assertEqual(lang, "en")

    def test_detect_language_tetun(self):
        lang = detect_language("governu ne'e sei halo servisu foun iha area ida ne'e ba povu")
        self.assertEqual(lang, "tet")

    def test_preprocess_news_pipeline(self):
        out = preprocess_news("Berita Penting", "Pemerintah mengumumkan kebijakan baru <p>hari ini</p>")
        self.assertIn("language", out)
        self.assertIn("text_final", out)
        self.assertNotIn("<p>", out["text_final"])
        self.assertGreater(out["token_count"], 0)


if __name__ == "__main__":
    unittest.main()
