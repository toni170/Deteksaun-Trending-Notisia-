"""
Ekstraksi kata kunci utama (top-N terms) untuk tiap cluster/topik hasil K-Means,
lalu simpan daftar topik terdeteksi ke data/keywords.
"""
import os
import json
import logging
from datetime import datetime

import numpy as np

logger = logging.getLogger("keyword_extractor")


def extract_top_terms_per_cluster(vectorizer, kmeans, top_n=10) -> dict:
    """
    Ambil top-N term dengan bobot centroid tertinggi untuk tiap cluster.
    Mengembalikan dict: {cluster_id: [term1, term2, ...]}
    """
    terms = np.array(vectorizer.get_feature_names_out())
    centroids = kmeans.cluster_centers_

    topics = {}
    for cluster_id, centroid in enumerate(centroids):
        top_indices = centroid.argsort()[::-1][:top_n]
        topics[cluster_id] = [terms[i] for i in top_indices]

    return topics


def build_topic_labels(topics: dict) -> dict:
    """Buat label topik ringkas (3 kata kunci teratas) untuk tiap cluster."""
    return {
        cluster_id: " / ".join(words[:3])
        for cluster_id, words in topics.items()
    }


def save_keywords(topics: dict, labels: dict, keywords_dir: str,
                   cluster_sizes: dict = None, name: str = "tatoli") -> str:
    os.makedirs(keywords_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(keywords_dir, f"{name}_keywords_{stamp}.json")

    payload = {
        "source": name,
        "generated_at": datetime.now().isoformat(),
        "topics": [
            {
                "cluster_id": int(cid),
                "label": labels.get(cid, ""),
                "top_terms": words,
                "news_count": int(cluster_sizes.get(cid, 0)) if cluster_sizes else None,
            }
            for cid, words in topics.items()
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Simpan juga sebagai "latest" agar dashboard mudah memuat hasil terbaru
    latest_path = os.path.join(keywords_dir, f"{name}_keywords_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Keyword/topik disimpan ke %s (dan %s)", out_path, latest_path)
    return out_path


def run_keyword_extraction(vectorizer, kmeans, df_with_clusters, keywords_dir,
                            top_n=10, name: str = "tatoli"):
    """High-level: ekstrak keywords per cluster, hitung ukuran cluster, simpan JSON."""
    topics = extract_top_terms_per_cluster(vectorizer, kmeans, top_n=top_n)
    labels = build_topic_labels(topics)

    cluster_sizes = df_with_clusters["cluster"].value_counts().to_dict()

    out_path = save_keywords(topics, labels, keywords_dir, cluster_sizes, name=name)
    return topics, labels, out_path


def build_candidate_themes(topics: dict, labels: dict, df_with_clusters, X=None, kmeans=None,
                           n_examples: int = 3) -> list:
    """
    Tema KANDIDATU ba revizaun umana: kada cluster -> label, top terms, kuantidade,
    persentajen no títulu exemplu. Se `X` (matriks TF-IDF) no `kmeans` fornese,
    títulu exemplu mak notísia ne'ebé besik liu ba sentroide cluster; se lae,
    foti títulu dahuluk iha cluster.
    Retorna lista ordena tuir kuantidade notísia (boot -> ki'ik).
    """
    total = len(df_with_clusters)
    dist = kmeans.transform(X) if (X is not None and kmeans is not None) else None
    themes = []
    for cid, words in topics.items():
        mask = (df_with_clusters["cluster"] == cid).to_numpy()
        rows = np.flatnonzero(mask)
        if dist is not None and len(rows):
            rows = rows[np.argsort(dist[rows, cid])]
        titles = df_with_clusters.iloc[rows[:n_examples]]["title"].astype(str).tolist() \
            if "title" in df_with_clusters.columns else []
        count = int(mask.sum())
        themes.append({
            "cluster_id": int(cid),
            "label": labels.get(cid, ""),
            "top_terms": [str(w) for w in words],
            "news_count": count,
            "share_pct": round(100 * count / total, 1) if total else 0.0,
            "example_titles": titles,
            "status": "kandidatu",
        })
    return sorted(themes, key=lambda t: t["news_count"], reverse=True)
