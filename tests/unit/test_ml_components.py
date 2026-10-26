"""Teste #11-#14: TF-IDF, K-Means, cluster atribuídu, tema kandidatu."""
import json
import os

import numpy as np
import pytest

from src.clustering.kmeans_model import TopicKMeans, run_clustering
from src.feature_extraction.tfidf_vectorizer import TfidfFeatureExtractor, run_feature_extraction
from src.keywords.keyword_extractor import (
    build_candidate_themes, build_topic_labels, extract_top_terms_per_cluster, save_keywords,
)
from src.preprocessing.pipeline import preprocess_dataframe

pytestmark = pytest.mark.unit


@pytest.fixture
def processed(news_df):
    return preprocess_dataframe(news_df)


@pytest.fixture
def fitted(processed, tmp_path):
    X, extractor = run_feature_extraction(processed["text_final"].tolist(), str(tmp_path), min_df=2)
    labels, model = run_clustering(X, str(tmp_path), min_k=2, max_k=6)
    return processed.assign(cluster=labels), X, extractor, model


def _purity(df):
    """Fraksaun notísia ne'ebé iha cluster ho tópiku majoritáriu hanesan (1.0 = perfeitu)."""
    return sum(g["cluster"].value_counts().iloc[0] for _, g in df.groupby("_topic")) / len(df)


# ---- #11 TF-IDF ------------------------------------------------------------
def test_11_tfidf_shape_normalisation_and_vocabulary(processed):
    ext = TfidfFeatureExtractor(min_df=2)
    X = ext.fit_transform(processed["text_final"].tolist())
    assert X.shape[0] == len(processed) and X.shape[1] > 10
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(norms, 1.0)
    vocab = set(ext.get_feature_names())
    assert {"eskola", "ospital", "banku"} & vocab
    assert any(" " in term for term in vocab)  # bigram (ngram_range=(1,2))


def test_11_tfidf_save_load_roundtrip_gives_same_vectors(processed, tmp_path):
    X, ext = run_feature_extraction(processed["text_final"].tolist(), str(tmp_path))
    loaded = TfidfFeatureExtractor.load(str(tmp_path))
    texts = processed["text_final"].tolist()[:5]
    assert (ext.transform(texts) != loaded.transform(texts)).nnz == 0


def test_11_tfidf_errors_and_small_corpus_min_df():
    with pytest.raises(RuntimeError):
        TfidfFeatureExtractor().transform(["x"])
    with pytest.raises(ValueError):
        TfidfFeatureExtractor().fit_transform(["", "   ", None])
    X = TfidfFeatureExtractor(min_df=5).fit_transform(["eskola estudante", "ospital mediku"])
    assert X.shape[0] == 2 and X.shape[1] > 0  # min_df otomátiku ajusta ba korpus ki'ik


# ---- #12 K-Means -----------------------------------------------------------
def test_12_kmeans_selects_k_and_records_scores(fitted):
    df, X, _, model = fitted
    assert model.best_k == 3  # korpus iha tópiku 3 ne'ebé klaru
    assert set(model.silhouette_scores_) == set(model.inertia_scores_) == set(range(2, 7))
    assert model.silhouette_scores_[3] == max(model.silhouette_scores_.values())
    assert _purity(df) >= 0.95


def test_12_kmeans_save_load_predict_roundtrip(fitted, tmp_path):
    df, X, _, model = fitted
    loaded = TopicKMeans.load(str(tmp_path))
    assert loaded.best_k == model.best_k
    assert (loaded.predict(X) == model.predict(X)).all()


def test_12_kmeans_needs_at_least_three_documents():
    ext = TfidfFeatureExtractor(min_df=1)
    X = ext.fit_transform(["eskola estudante", "ospital mediku"])
    with pytest.raises(ValueError):
        TopicKMeans().fit(X)  # karakteriza: modulu la proteje n<3 — servisu treinu mak proteje


# ---- #13 Cluster atribuídu -------------------------------------------------
def test_13_every_document_gets_valid_cluster(fitted):
    df, _, _, model = fitted
    assert df["cluster"].notna().all()
    assert set(df["cluster"]) == set(range(model.best_k))


def test_13_new_text_is_assigned_to_matching_topic_cluster(fitted):
    df, _, extractor, model = fitted
    majority = {t: g["cluster"].value_counts().idxmax() for t, g in df.groupby("_topic")}
    probes = {"edukasaun": "eskola estudante prufesor", "saude": "ospital mediku vasina", "ekonomia": "banku investimentu merkadu"}
    for topic, text in probes.items():
        assert int(model.predict(extractor.transform([text]))[0]) == majority[topic]


# ---- #14 Tema kandidatu ----------------------------------------------------
def test_14_candidate_themes_structure_and_content(fitted):
    df, X, extractor, model = fitted
    topics = extract_top_terms_per_cluster(extractor.vectorizer, model.kmeans, top_n=8)
    labels = build_topic_labels(topics)
    themes = build_candidate_themes(topics, labels, df, X=X, kmeans=model.kmeans)

    assert len(themes) == model.best_k
    assert sum(t["news_count"] for t in themes) == len(df)
    assert abs(sum(t["share_pct"] for t in themes) - 100) < 0.5
    assert [t["news_count"] for t in themes] == sorted((t["news_count"] for t in themes), reverse=True)
    for t in themes:
        assert t["status"] == "kandidatu" and len(t["top_terms"]) == 8
        assert t["label"] == " / ".join(t["top_terms"][:3])
        assert 1 <= len(t["example_titles"]) <= 3
        in_cluster = set(df.loc[df["cluster"] == t["cluster_id"], "title"])
        assert set(t["example_titles"]) <= in_cluster
    # Kada tema mapa ba tópiku sintétiku ida de'it (stemmer bele tesi liafuan: "vasina" -> "vasi")
    from tests.conftest import TOPICS_A

    def overlap(terms, vocab):
        return sum(any(v.startswith(t) for v in vocab) for t in terms)

    mapped = []
    for t in themes:
        scores = {name: overlap(t["top_terms"], vocab) for name, vocab in TOPICS_A.items()}
        best = max(scores, key=scores.get)
        assert scores[best] >= 7, f"tema {t['cluster_id']} la koerente: {scores}"
        mapped.append(best)
    assert sorted(mapped) == sorted(TOPICS_A)  # tópiku 3 -> tema 3 diferente


def test_14_save_keywords_writes_json(fitted, tmp_path):
    df, _, extractor, model = fitted
    topics = extract_top_terms_per_cluster(extractor.vectorizer, model.kmeans, top_n=5)
    out = save_keywords(topics, build_topic_labels(topics), str(tmp_path / "kw"),
                        df["cluster"].value_counts().to_dict())
    payload = json.load(open(os.path.join(tmp_path, "kw", "tatoli_keywords_latest.json"), encoding="utf-8"))
    assert os.path.exists(out) and len(payload["topics"]) == model.best_k
