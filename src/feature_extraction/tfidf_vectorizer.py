"""
Ekstraksi fitur teks berita menjadi representasi TF-IDF menggunakan
sklearn.feature_extraction.text.TfidfVectorizer.
"""
import os
import logging

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("tfidf_vectorizer")


class TfidfFeatureExtractor:
    """Wrapper tipis di atas TfidfVectorizer + utilitas save/load model."""

    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = None

    def fit_transform(self, texts: list):
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            raise ValueError("Tidak ada teks bersih untuk diekstrak fiturnya (semua kosong).")

        # min_df otomatis menyesuaikan jika jumlah dokumen sedikit
        effective_min_df = min(self.min_df, max(1, len(texts) // 20 or 1))

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=effective_min_df,
        )
        X = self.vectorizer.fit_transform(texts)
        logger.info(
            "TF-IDF dibangun: %s dokumen x %s fitur (min_df=%s)",
            X.shape[0], X.shape[1], effective_min_df,
        )
        return X

    def transform(self, texts: list):
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer belum di-fit. Panggil fit_transform() dulu atau load model.")
        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    # ------------------------------------------------------------------
    def save(self, models_dir: str, name: str = "tatoli") -> str:
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"{name}_tfidf_vectorizer.joblib")
        joblib.dump(self.vectorizer, path)
        logger.info("TF-IDF vectorizer disimpan ke %s", path)
        return path

    @staticmethod
    def load(models_dir: str, name: str = "tatoli") -> "TfidfFeatureExtractor":
        path = os.path.join(models_dir, f"{name}_tfidf_vectorizer.joblib")
        extractor = TfidfFeatureExtractor()
        extractor.vectorizer = joblib.load(path)
        return extractor


def run_feature_extraction(texts: list, models_dir: str, name: str = "tatoli",
                            max_features=5000, ngram_range=(1, 2), min_df=2):
    """High-level: fit TF-IDF dari daftar teks bersih, simpan model, return (X, extractor)."""
    extractor = TfidfFeatureExtractor(
        max_features=max_features, ngram_range=ngram_range, min_df=min_df,
    )
    X = extractor.fit_transform(texts)
    extractor.save(models_dir, name=name)
    return X, extractor
