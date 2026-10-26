"""
Registu modelu ho versaun (TF-IDF + K-Means + dadus ho cluster + tema).

Prinsípiu: modelu foun TREINA iha área staging, no sei PROMOVE (troka ponteiru
`CURRENT` atomikamente) bainhira tudu remata ho susesu. Se treinu falla, `CURRENT`
la muda -> modelu antigo kontinua disponivel. Hotu-hotu rai iha disku, entaun
restart aplikasaun bele karrega fali.

Layout:
    models_dir/CURRENT                 # ID versaun ativu (ficheiru tékstu)
    models_dir/versions/<id>/          # tatoli_tfidf_vectorizer.joblib, tatoli_kmeans.joblib,
                                       # news_clustered.csv, tatoli_keywords_latest.json, meta.json
    models_dir/.staging/<id>/          # servisu temporáriu (hamoos bainhira falla)
"""
import json
import os
import shutil
import logging
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.clustering.kmeans_model import TopicKMeans
from src.feature_extraction.tfidf_vectorizer import TfidfFeatureExtractor

logger = logging.getLogger("model_registry")

MODEL_NAME = "tatoli"
CLUSTERED_CSV = "news_clustered.csv"
KEYWORDS_JSON = f"{MODEL_NAME}_keywords_latest.json"
META_JSON = "meta.json"


class ModelLoadError(RuntimeError):
    """Versaun modelu iha disku, maibé la bele karrega (korruptu/falta ficheiru)."""


@dataclass
class ModelBundle:
    version_id: str
    path: str
    extractor: TfidfFeatureExtractor
    topic_model: TopicKMeans
    df: pd.DataFrame
    topics_payload: dict
    meta: dict
    themes: list = field(default_factory=list)


class ModelRegistry:
    def __init__(self, models_dir: str, keep_versions: int = 5):
        self.models_dir = models_dir
        self.keep_versions = keep_versions
        self.versions_dir = os.path.join(models_dir, "versions")
        self.staging_dir = os.path.join(models_dir, ".staging")
        self._current_file = os.path.join(models_dir, "CURRENT")

    # -- staging / promote -------------------------------------------------
    def create_staging(self) -> tuple:
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.staging_dir, version_id)
        os.makedirs(path, exist_ok=True)
        return version_id, path

    def discard(self, version_id: str) -> None:
        shutil.rmtree(os.path.join(self.staging_dir, version_id), ignore_errors=True)

    def promote(self, version_id: str) -> str:
        """Muda staging -> versions no troka `CURRENT` atomikamente (os.replace)."""
        src = os.path.join(self.staging_dir, version_id)
        if not os.path.isfile(os.path.join(src, META_JSON)):
            raise ModelLoadError(f"Staging '{version_id}' la kompletu (falta {META_JSON}).")
        os.makedirs(self.versions_dir, exist_ok=True)
        dst = os.path.join(self.versions_dir, version_id)
        os.replace(src, dst)

        tmp = self._current_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(version_id)
        os.replace(tmp, self._current_file)
        self._prune()
        logger.info("Modelu versaun %s promove sai ativu.", version_id)
        return dst

    def _prune(self) -> None:
        current = self.current_version_id()
        for vid in self.list_versions()[self.keep_versions:]:
            if vid != current:
                shutil.rmtree(os.path.join(self.versions_dir, vid), ignore_errors=True)

    # -- consulta ----------------------------------------------------------
    def list_versions(self) -> list:
        """ID versaun (foun -> tuan)."""
        if not os.path.isdir(self.versions_dir):
            return []
        return sorted(os.listdir(self.versions_dir), reverse=True)

    def current_version_id(self):
        if not os.path.isfile(self._current_file):
            return None
        with open(self._current_file, "r", encoding="utf-8") as f:
            vid = f.read().strip()
        return vid if vid and os.path.isdir(os.path.join(self.versions_dir, vid)) else None

    def load_version(self, version_id: str) -> ModelBundle:
        path = os.path.join(self.versions_dir, version_id)
        try:
            extractor = TfidfFeatureExtractor.load(path, name=MODEL_NAME)
            topic_model = TopicKMeans.load(path, name=MODEL_NAME)
            df = pd.read_csv(os.path.join(path, CLUSTERED_CSV))
            with open(os.path.join(path, KEYWORDS_JSON), "r", encoding="utf-8") as f:
                payload = json.load(f)
            with open(os.path.join(path, META_JSON), "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            raise ModelLoadError(f"La bele karrega versaun '{version_id}': {e}") from e
        return ModelBundle(version_id, path, extractor, topic_model, df, payload, meta,
                           themes=meta.get("themes", []))

    def load_current(self):
        """Modelu ativu, ka None se seidauk iha. Hasa'e ModelLoadError se korruptu."""
        vid = self.current_version_id()
        return self.load_version(vid) if vid else None

    def load_best_available(self):
        """Hodi tama restart: tenta CURRENT, se korruptu tenta versaun tuan liu ne'ebé válidu."""
        vid = self.current_version_id()
        candidates = ([vid] if vid else []) + [v for v in self.list_versions() if v != vid]
        for cand in candidates:
            try:
                return self.load_version(cand)
            except ModelLoadError as e:
                logger.warning("%s — tenta versaun anterior.", e)
        return None
