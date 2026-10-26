"""
Deteksi topik berita menggunakan clustering K-Means di atas representasi TF-IDF.
Jumlah cluster optimal dicari otomatis lewat silhouette score, dengan skor
elbow (inertia) turut disimpan sebagai referensi tambahan.
"""
import os
import logging

import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger("kmeans_model")


class TopicKMeans:
    def __init__(self, min_k=2, max_k=10, random_state=42, silhouette_sample_size=3000):
        # Silhouette butuh matriks jarak n x n -> pada puluhan ribu berita memori habis.
        # Dengan sampel acak, skor tetap representatif tapi memori kecil.
        self.silhouette_sample_size = silhouette_sample_size
        self.min_k = min_k
        self.max_k = max_k
        self.random_state = random_state

        self.kmeans = None
        self.best_k = None
        self.silhouette_scores_ = {}  # k -> silhouette score
        self.inertia_scores_ = {}     # k -> inertia (untuk elbow method)

    def fit(self, X):
        """
        Cari k optimal dalam rentang [min_k, max_k] berdasarkan silhouette
        score tertinggi (inertia tiap k turut dicatat untuk elbow method),
        lalu latih model KMeans final dengan k terbaik.
        """
        n_samples = X.shape[0]
        upper_k = min(self.max_k, n_samples - 1)
        upper_k = max(upper_k, self.min_k)

        best_k, best_score, best_model = None, -1, None

        for k in range(self.min_k, upper_k + 1):
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)

            self.inertia_scores_[k] = float(model.inertia_)

            if len(set(labels)) < 2:
                continue

            sample = self.silhouette_sample_size
            score = silhouette_score(
                X, labels,
                sample_size=sample if sample and n_samples > sample else None,
                random_state=self.random_state,
            )
            self.silhouette_scores_[k] = float(score)
            logger.info("k=%s -> silhouette=%.4f | inertia=%.2f", k, score, model.inertia_)

            if score > best_score:
                best_k, best_score, best_model = k, score, model

        if best_model is None:
            best_k = self.min_k
            best_model = KMeans(n_clusters=best_k, random_state=self.random_state, n_init=10).fit(X)

        self.kmeans = best_model
        self.best_k = best_k

        logger.info("K optimal terpilih: %s (silhouette=%.4f)", best_k, best_score)
        return self.kmeans.labels_

    def predict(self, X):
        return self.kmeans.predict(X)

    # ------------------------------------------------------------------
    def save(self, models_dir: str, name: str = "tatoli") -> str:
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"{name}_kmeans.joblib")
        joblib.dump(self.kmeans, path)
        logger.info("Model KMeans disimpan ke %s", path)
        return path

    @staticmethod
    def load(models_dir: str, name: str = "tatoli") -> "TopicKMeans":
        path = os.path.join(models_dir, f"{name}_kmeans.joblib")
        wrapper = TopicKMeans()
        wrapper.kmeans = joblib.load(path)
        wrapper.best_k = wrapper.kmeans.n_clusters
        return wrapper


def run_clustering(X, models_dir: str, name: str = "tatoli", min_k=2, max_k=10,
                   silhouette_sample_size=3000):
    """High-level: fit KMeans di atas matriks TF-IDF, simpan model, return (labels, model)."""
    topic_model = TopicKMeans(min_k=min_k, max_k=max_k,
                              silhouette_sample_size=silhouette_sample_size)
    labels = topic_model.fit(X)
    topic_model.save(models_dir, name=name)
    return labels, topic_model
