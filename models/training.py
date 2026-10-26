"""
Training model topik (TF-IDF + K-Means) dan penyimpanan ke models/model.pkl.

`model.pkl` adalah satu paket (dict) berisi:
    vectorizer : TfidfVectorizer yang sudah di-fit
    kmeans     : KMeans terbaik (k dipilih lewat silhouette score)
    topics     : {cluster_id: [kata kunci teratas]}
    labels     : {cluster_id: "kata1 / kata2 / kata3"}
    metadata   : info training (tanggal, jumlah dokumen, k, skor, versi sklearn, ...)

PERINGATAN: .pkl adalah format pickle — jangan pernah memuat file .pkl dari
sumber yang tidak dipercaya.

Cara pakai (dari root proyek):
    python -m models.training                      # latih dari data/cleaned/tatoli_cleaned_master.csv
    python -m models.training --source db          # latih dari tabel `articles` di SQLite
    python -m models.training --source legacy      # bungkus model lama data/models/*.joblib jadi model.pkl
    python -m models.training --min-k 3 --max-k 8

Memakai model untuk berita baru:
    from models.training import predict_topic
    predict_topic("Ministru Edukasaun hasa'e osan ba eskola ...")
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.clustering.kmeans_model import TopicKMeans  # noqa: E402
from src.feature_extraction.tfidf_vectorizer import TfidfFeatureExtractor  # noqa: E402
from src.keywords.keyword_extractor import (  # noqa: E402
    build_topic_labels, extract_top_terms_per_cluster,
)

logger = logging.getLogger("training")

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model.pkl")
CLEANED_MASTER = os.path.join(PROJECT_ROOT, "data", "cleaned", "tatoli_cleaned_master.csv")
LEGACY_MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")
BUNDLE_VERSION = 1


# ----------------------------------------------------------------------
# Bundle: bangun / simpan / muat
# ----------------------------------------------------------------------
def build_bundle(vectorizer, kmeans, topics: dict, labels: dict, metadata: Optional[dict] = None) -> dict:
    import sklearn
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "n_features": int(len(vectorizer.get_feature_names_out())),
        "best_k": int(kmeans.n_clusters),
    }
    meta.update(metadata or {})
    return {
        "bundle_version": BUNDLE_VERSION,
        "vectorizer": vectorizer,
        "kmeans": kmeans,
        "topics": {int(k): list(v) for k, v in topics.items()},
        "labels": {int(k): v for k, v in labels.items()},
        "metadata": meta,
    }


def save_model(bundle: dict, path: str = MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(bundle, path)
    logger.info("model.pkl disimpan ke %s", path)
    return path


def load_model(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} belum ada. Jalankan `python -m models.training` atau `python main.py` dulu."
        )
    bundle = joblib.load(path)
    saved = bundle.get("metadata", {}).get("sklearn_version")
    import sklearn
    if saved and saved != sklearn.__version__:
        logger.warning(
            "model.pkl dilatih dengan scikit-learn %s, sekarang terpasang %s. "
            "Latih ulang jika muncul error/hasil aneh.", saved, sklearn.__version__,
        )
    return bundle


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train(texts: List[str], min_k: int = 2, max_k: int = 10, max_features: int = 5000,
          ngram_range=(1, 2), min_df: int = 2, silhouette_sample_size: int = 3000,
          top_n: int = 10, random_state: int = 42):
    """
    Latih TF-IDF + K-Means dari daftar teks yang SUDAH dibersihkan (kolom `text_final`).
    Mengembalikan (bundle, labels_per_dokumen).
    """
    extractor = TfidfFeatureExtractor(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    X = extractor.fit_transform(texts)

    topic_model = TopicKMeans(min_k=min_k, max_k=max_k, random_state=random_state,
                              silhouette_sample_size=silhouette_sample_size)
    doc_labels = topic_model.fit(X)

    topics = extract_top_terms_per_cluster(extractor.vectorizer, topic_model.kmeans, top_n=top_n)
    labels = build_topic_labels(topics)

    bundle = build_bundle(
        extractor.vectorizer, topic_model.kmeans, topics, labels,
        metadata={
            "n_documents": int(X.shape[0]),
            "silhouette": topic_model.silhouette_scores_.get(topic_model.best_k),
            "inertia": topic_model.inertia_scores_.get(topic_model.best_k),
            "silhouette_scores": topic_model.silhouette_scores_,
            "inertia_scores": topic_model.inertia_scores_,
            "params": {
                "min_k": min_k, "max_k": max_k, "max_features": max_features,
                "ngram_range": list(ngram_range), "min_df": min_df,
                "random_state": random_state, "top_n": top_n,
            },
        },
    )
    return bundle, doc_labels


def bundle_from_legacy(models_dir: str = LEGACY_MODELS_DIR, name: str = "tatoli", top_n: int = 10) -> dict:
    """Bungkus model lama (data/models/<name>_*.joblib) menjadi bundle tanpa melatih ulang."""
    vectorizer = TfidfFeatureExtractor.load(models_dir, name).vectorizer
    kmeans = TopicKMeans.load(models_dir, name).kmeans
    topics = extract_top_terms_per_cluster(vectorizer, kmeans, top_n=top_n)
    return build_bundle(vectorizer, kmeans, topics, build_topic_labels(topics),
                        metadata={"built_from": f"{models_dir}/{name}_*.joblib"})


# ----------------------------------------------------------------------
# Prediksi
# ----------------------------------------------------------------------
def predict_topic(text: str, title: str = "", bundle: Optional[dict] = None) -> dict:
    """Prediksi topik untuk satu berita baru (teks mentah; preprocessing dilakukan otomatis)."""
    from src.preprocessing.text_cleaner import preprocess_news

    bundle = bundle or load_model()
    processed = preprocess_news(title, text)
    final = processed["text_final"]
    if not final.strip():
        raise ValueError("Teks kosong setelah preprocessing — tidak bisa diprediksi.")

    X = bundle["vectorizer"].transform([final])
    cid = int(bundle["kmeans"].predict(X)[0])
    return {
        "cluster_id": cid,
        "label": bundle["labels"].get(cid, ""),
        "top_terms": bundle["topics"].get(cid, []),
        "language": processed["language"],
    }


# ----------------------------------------------------------------------
# Sumber data
# ----------------------------------------------------------------------
def load_texts(source: str, csv_path: str = CLEANED_MASTER):
    """Ambil (DataFrame, daftar text_final) dari CSV cleaned atau dari SQLite."""
    if source == "csv":
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} belum ada. Jalankan `python main.py` dulu.")
        df = pd.read_csv(csv_path)
    elif source == "db":
        from database.database import articles_to_dataframe, get_session, init_db
        init_db()
        with get_session() as session:
            df = articles_to_dataframe(session)
    else:
        raise ValueError(f"source tidak dikenal: {source}")

    df = df[df["text_final"].fillna("").str.strip() != ""]
    return df, df["text_final"].tolist()


def main():
    parser = argparse.ArgumentParser(description="Latih model topik dan simpan ke models/model.pkl")
    parser.add_argument("--source", choices=["csv", "db", "legacy"], default="csv",
                        help="csv = data/cleaned master | db = tabel articles | legacy = bungkus model lama")
    parser.add_argument("--csv", default=CLEANED_MASTER, help="Path CSV cleaned (untuk --source csv)")
    parser.add_argument("--out", default=MODEL_PATH, help="Lokasi output model.pkl")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--no-db", action="store_true", help="Jangan catat training run ke database")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if args.source == "legacy":
        bundle = bundle_from_legacy()
        save_model(bundle, args.out)
        print(f"model.pkl dibuat dari model lama: k={bundle['metadata']['best_k']} -> {args.out}")
        return

    df, texts = load_texts(args.source, args.csv)
    bundle, doc_labels = train(texts, min_k=args.min_k, max_k=args.max_k, max_features=args.max_features)
    save_model(bundle, args.out)

    if not args.no_db:
        try:
            from database.database import get_session, init_db, record_training_run
            init_db()
            sizes = pd.Series(doc_labels).value_counts().to_dict()
            meta = bundle["metadata"]
            with get_session() as session:
                record_training_run(
                    session, bundle["topics"], bundle["labels"], cluster_sizes=sizes,
                    n_documents=meta["n_documents"], n_features=meta["n_features"],
                    best_k=meta["best_k"], silhouette=meta.get("silhouette"),
                    inertia=meta.get("inertia"), params=meta.get("params"), model_path=args.out,
                )
        except Exception:
            logger.exception("Gagal mencatat training run ke database (model.pkl tetap tersimpan).")

    meta = bundle["metadata"]
    print(f"\nSelesai: {meta['n_documents']} dokumen, k={meta['best_k']}, "
          f"silhouette={meta.get('silhouette'):.4f} -> {args.out}")
    for cid, label in bundle["labels"].items():
        print(f"  [Cluster {cid}] {label}")


if __name__ == "__main__":
    main()
