"""
Servisu treinu: dataset notísia (formatu RAW_FIELDS) -> preprocessing -> TF-IDF
-> K-Means -> liafuan-xave + tema kandidatu -> rai/promove iha `ModelRegistry`.

Treinu halo iha staging; `CURRENT` muda de'it se hotu-hotu susesu (modelu antigo
la lakon bainhira falla). `progress_cb(fraksaun 0..1, mensajen_tetun)` atu dashboard
hatudu progresu.
"""
import json
import os
import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.clustering.kmeans_model import run_clustering
from src.feature_extraction.tfidf_vectorizer import run_feature_extraction
from src.keywords.keyword_extractor import (
    build_candidate_themes, build_topic_labels, extract_top_terms_per_cluster, save_keywords,
)
from src.preprocessing.pipeline import preprocess_dataframe
from src.training.model_registry import (
    CLUSTERED_CSV, KEYWORDS_JSON, META_JSON, MODEL_NAME, ModelRegistry,
)

logger = logging.getLogger("training_service")


class TrainingError(RuntimeError):
    """Treinu falla. `code` estável; mensajen iha Tetun."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class TrainingResult:
    version_id: str
    df: pd.DataFrame
    topics: dict
    labels: dict
    themes: list
    topics_payload: dict
    meta: dict


def _noop(fraction: float, message: str) -> None:
    pass


def train_and_register(raw_df: pd.DataFrame, registry: ModelRegistry, *, min_k: int = 2,
                       max_k: int = 10, top_n: int = 10, source_name: str = "import",
                       max_features: int = 5000, ngram_range=(1, 2), min_df: int = 2,
                       progress_cb=None) -> TrainingResult:
    progress = progress_cb or _noop
    if min_k < 2 or max_k < min_k:
        raise TrainingError("invalid_params", f"Parámetru cluster la válidu (min_k={min_k}, max_k={max_k}).")

    version_id, staging = registry.create_staging()
    promoted = False
    try:
        progress(0.05, "Hamoos no prepara tékstu (case folding, tokenizasaun, stopword, stemming)...")
        df = preprocess_dataframe(raw_df)
        n_docs = len(df)
        required = max(3, min_k + 1)
        if n_docs < required:
            raise TrainingError(
                "too_few_documents",
                f"Dadus la to'o: iha {n_docs} notísia ne'ebé bele uza, presiza at'e {required}.",
            )

        progress(0.30, "Konstrui TF-IDF...")
        X, extractor = run_feature_extraction(
            df["text_final"].tolist(), staging, name=MODEL_NAME,
            max_features=max_features, ngram_range=ngram_range, min_df=min_df,
        )
        if X.shape[0] != n_docs:
            raise TrainingError("misaligned", "Númeru notísia la hanesan ho matriks TF-IDF.")

        progress(0.55, "Treina K-Means no hili k ne'ebé di'ak liu...")
        labels_arr, topic_model = run_clustering(X, staging, name=MODEL_NAME, min_k=min_k, max_k=max_k)
        df["cluster"] = [int(c) for c in labels_arr]

        progress(0.75, "Gera liafuan-xave no tema kandidatu...")
        topics = extract_top_terms_per_cluster(extractor.vectorizer, topic_model.kmeans, top_n=top_n)
        labels = build_topic_labels(topics)
        sizes = df["cluster"].value_counts().to_dict()
        save_keywords(topics, labels, staging, sizes, name=MODEL_NAME)
        themes = build_candidate_themes(topics, labels, df, X=X, kmeans=topic_model.kmeans)

        progress(0.90, "Rai modelu foun...")
        df.to_csv(os.path.join(staging, CLUSTERED_CSV), index=False, encoding="utf-8-sig")
        with open(os.path.join(staging, KEYWORDS_JSON), "r", encoding="utf-8") as f:
            topics_payload = json.load(f)
        meta = {
            "version_id": version_id,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "source_name": source_name,
            "n_docs_input": int(len(raw_df)),
            "n_docs_used": int(n_docs),
            "best_k": int(topic_model.best_k),
            "silhouette": topic_model.silhouette_scores_.get(topic_model.best_k),
            "params": {"min_k": min_k, "max_k": max_k, "max_features": max_features,
                       "ngram_range": list(ngram_range), "min_df": min_df},
            "themes": themes,
        }
        with open(os.path.join(staging, META_JSON), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        registry.promote(version_id)  # troka CURRENT -> moment ida-ne'e de'it
        promoted = True
        progress(1.0, "Remata: modelu foun ativu ona.")
        return TrainingResult(version_id, df, topics, labels, themes, topics_payload, meta)

    except TrainingError:
        raise
    except Exception as e:
        logger.exception("Treinu falla")
        raise TrainingError("training_failed", f"Treinu falla: {e}") from e
    finally:
        if not promoted:
            registry.discard(version_id)
