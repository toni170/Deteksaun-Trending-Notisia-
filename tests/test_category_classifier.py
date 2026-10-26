import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.category_classifier import (
    classify_category,
    classify_dataframe_categories,
    CATEGORIES,
    FALLBACK_CATEGORY,
)


class TestCategoryClassifier(unittest.TestCase):
    def test_education_keywords_detected(self):
        cat = classify_category(
            "Estudante universidade simu bolsa estudu foun",
            "Ministériu Edukasaun fó eskola no dosente foun ba estudante sira.",
        )
        self.assertEqual(cat, "Edukasaun")

    def test_politics_keywords_detected(self):
        cat = classify_category(
            "Primeiru-Ministru enkontru ho parlamentu kona-ba lei foun",
            "Deputadu husi CNRT no Fretilin diskute konstituisaun iha gabinete.",
        )
        self.assertEqual(cat, "Politika")

    def test_health_keywords_detected(self):
        cat = classify_category(
            "Ospital foun loke iha Dili",
            "Mediku no enfermeira prepara vasina ba pasiente iha klinika.",
        )
        self.assertEqual(cat, "Saúde")

    def test_economy_keywords_detected(self):
        cat = classify_category(
            "Presu simente sa'e iha merkadu",
            "Investimentu no orsamentu governu ba negosiu banku aumenta.",
        )
        self.assertEqual(cat, "Ekonomia")

    def test_junk_scraped_category_ignored(self):
        # Kategoria "Home" (breadcrumb sampah) tenke ignora, la konsidera
        # diretamente hanesan kategoria válidu.
        cat = classify_category(
            "Ospital foun loke iha Dili",
            "Mediku fó tratamentu ba pasiente.",
            raw_category="Home",
        )
        self.assertEqual(cat, "Saúde")

    def test_known_scraped_category_used_directly(self):
        cat = classify_category("Titulu neutru", "Konteudu neutru", raw_category="Health")
        self.assertEqual(cat, "Saúde")

    def test_no_match_returns_fallback(self):
        cat = classify_category("Festa kultura tradisionál", "Dansa no muzika tradisionál iha aldeia")
        self.assertEqual(cat, FALLBACK_CATEGORY)

    def test_all_outputs_within_known_set(self):
        allowed = set(CATEGORIES) | {FALLBACK_CATEGORY}
        samples = [
            ("Governu aprova orsamentu", "Ministru finansa fó relatóriu"),
            ("Estudante hahu tinan letivu foun", "Eskola prepara sala aula"),
            ("", ""),
        ]
        for title, content in samples:
            self.assertIn(classify_category(title, content), allowed)

    def test_dataframe_classification(self):
        df = pd.DataFrame([
            {"title": "Eskola foun loke", "content": "Estudante no dosente kontente", "category": ""},
            {"title": "Ministru diskute lei", "content": "Parlamentu aprova votu", "category": "Home"},
        ])
        result = classify_dataframe_categories(df)
        self.assertEqual(result.iloc[0], "Edukasaun")
        self.assertEqual(result.iloc[1], "Politika")


if __name__ == "__main__":
    unittest.main()
