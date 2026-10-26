"""
Treinu modelu TF-IDF + K-Means husi dadus bersih (ba painel), ho avaliasaun.

Etapa: lee dadus -> fahe treinu/teste -> vetorizasaun TF-IDF -> K-Means (k otomátiku ho
silhouette ka k fixu) -> identifikasaun tópiku (liafuan-xave, label, tema kandidatu) ->
avaliasaun (akurasia, presizaun, recall, F1-score) -> lista liafuan -> rai modelu.

Modelu rai iha `ModelRegistry` (versaun, troka atomikamente) ho formatu hanesan ho painel antigu,
maibé `meta.json` iha mós `evaluation`, `k_search`, `wordlist` no `timings`.
"""
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from src.clustering.kmeans_model import TopicKMeans
from src.feature_extraction.tfidf_vectorizer import TfidfFeatureExtractor
from src.keywords.keyword_extractor import (
    build_candidate_themes, build_topic_labels, extract_top_terms_per_cluster, save_keywords,
)
from src.storage.multi_writer import MultiFormatWriter
from src.training.evaluation import evaluate_clusters
from src.training.model_registry import CLUSTERED_CSV, KEYWORDS_JSON, META_JSON, MODEL_NAME, ModelRegistry

logger = logging.getLogger("dashboard_training")

TRAIN_STAGES = [
    ("load", "Lee dadus bersih"),
    ("tfidf", "Vetorizasaun TF-IDF"),
    ("kmeans", "Klaster K-Means"),
    ("topics", "Identifikasaun tópiku"),
    ("evaluation", "Avaliasaun (akurasia, recall, F1-score)"),
    ("save", "Rai modelu no rezultadu"),
]
NEEDED = ["news_id", "source", "title", "link", "published_at", "category", "image_url", "language",
          "text_final", "token_count", "summary"]
CLUSTER_COLUMNS = NEEDED + ["cluster", "split"]


class TrainingError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class _Stopped(Exception):
    pass


def load_clean_frame(path: str) -> pd.DataFrame:
    """Lee dadus bersih (CSV, JSON ka SQLite) ho kolona presiza de'it."""
    ext = os.path.splitext(path)[1].lower()
    if not os.path.isfile(path):
        raise TrainingError("not_found", f"Ficheiru '{os.path.basename(path)}' la hetan.")
    try:
        if ext == ".csv":
            head = pd.read_csv(path, nrows=0).columns
            use = [c for c in NEEDED if c in head]
            df = pd.read_csv(path, usecols=use, dtype=str, keep_default_na=False)
        elif ext == ".json":
            df = pd.read_json(path, dtype=False).fillna("").astype(str)
        elif ext in (".db", ".sqlite"):
            import sqlite3
            conn = sqlite3.connect(path)
            try:
                tbl = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone()[0]
                df = pd.read_sql_query(f'SELECT * FROM "{tbl}"', conn).fillna("").astype(str)
            finally:
                conn.close()
        else:
            raise TrainingError("unsupported_format", f"Formatu '{ext}' la suporta. Uza .csv, .json ka .db.")
    except TrainingError:
        raise
    except Exception as e:
        raise TrainingError("unreadable", f"La bele lee ficheiru dadus bersih: {e}")
    if "text_final" not in df.columns:
        raise TrainingError(
            "missing_text_final",
            "Ficheiru la iha kolona `text_final`. Uza dadus ne'ebé sai husi tab Preprosesamentu.")
    for col in NEEDED:
        if col not in df.columns:
            df[col] = ""
    df = df[df["text_final"].str.strip() != ""].reset_index(drop=True)
    return df


def _wordlist(vectorizer, kmeans, X, top_global=100, top_topic=30) -> dict:
    terms = np.array(vectorizer.get_feature_names_out())
    score = np.asarray(X.sum(axis=0)).ravel()
    docfreq = np.asarray((X > 0).sum(axis=0)).ravel()
    order = np.argsort(score)[::-1][:top_global]
    per_topic = {}
    for cid, centroid in enumerate(kmeans.cluster_centers_):
        idx = np.argsort(centroid)[::-1][:top_topic]
        per_topic[str(cid)] = [{"liafuan": str(terms[i]), "pezu": round(float(centroid[i]), 5)} for i in idx]
    return {
        "global": [{"liafuan": str(terms[i]), "skor": round(float(score[i]), 3), "dokumentu": int(docfreq[i])}
                   for i in order],
        "per_topic": per_topic,
    }


def train_from_clean_file(path: str, registry: ModelRegistry, *, max_features=5000, ngram_range=(1, 2),
                          min_df=2, test_fraction=0.2, k_mode="auto", fixed_k=5, min_k=2, max_k=12,
                          exclude_generic=False, search_sample=20000, random_state=42, top_n=10,
                          progress=None, stop_event=None) -> dict:
    progress = progress or (lambda *a, **k: None)
    started = time.time()
    board = {k: {"key": k, "label": lb, "status": "hein", "seconds": 0.0, "detail": ""} for k, lb in TRAIN_STAGES}

    def emit(frac, msg, current=None):
        if stop_event is not None and stop_event.is_set():
            raise _Stopped()
        for k, v in board.items():
            if v["status"] == "hala'o" and k != current:
                v["status"] = "remata"
        if current:
            board[current]["status"] = "hala'o"
        progress(frac, msg, stages=[dict(v) for v in board.values()], current_stage=current)

    def finish(key, t0, detail=""):
        board[key]["seconds"] = round(time.perf_counter() - t0, 1)
        board[key]["status"] = "remata"
        board[key]["detail"] = detail

    if not 0.05 <= test_fraction <= 0.5:
        raise TrainingError("invalid_params", "Persentajen teste tenke entre 5% no 50%.")
    if k_mode == "auto" and (min_k < 2 or max_k < min_k):
        raise TrainingError("invalid_params", f"Parámetru k la válidu (k mínimu={min_k}, k máximu={max_k}).")

    version_id, staging = registry.create_staging()
    promoted = False
    try:
        # 1) Lee dadus
        emit(0.02, "Lee dadus bersih...", "load")
        t0 = time.perf_counter()
        df = load_clean_frame(path)
        n = len(df)
        if n < 20:
            raise TrainingError("too_few_documents", f"Dadus la to'o: iha {n} notísia, presiza at'e 20.")
        cats = df["category"].replace("", "Jerál")
        stratify = cats if cats.value_counts().min() >= 2 and cats.nunique() > 1 else None
        idx = np.arange(n)
        train_idx, test_idx = train_test_split(idx, test_size=test_fraction, random_state=random_state,
                                               stratify=stratify)
        df["split"] = "teste"
        df.loc[train_idx, "split"] = "treinu"
        finish("load", t0, f"{n:,} notísia · treinu {len(train_idx):,} · teste {len(test_idx):,}")

        # 2) TF-IDF (fit iha dadus treinu de'it, atu teste la kontamina)
        emit(0.10, "Vetorizasaun TF-IDF (fit iha dadus treinu)...", "tfidf")
        t0 = time.perf_counter()
        texts = df["text_final"].tolist()
        eff_min_df = min(min_df, max(1, len(train_idx) // 20))
        vec = TfidfVectorizer(max_features=max_features, ngram_range=tuple(ngram_range), min_df=eff_min_df)
        vec.fit([texts[i] for i in train_idx])
        X_all = vec.transform(texts)
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        finish("tfidf", t0, f"{X_all.shape[1]:,} fitur · matriks {X_all.shape[0]:,} × {X_all.shape[1]:,}")

        # 3) K-Means
        t0 = time.perf_counter()
        rng = np.random.RandomState(random_state)
        k_search = []
        if k_mode == "auto":
            upper = min(max_k, len(train_idx) - 1)
            ks = list(range(min_k, max(upper, min_k) + 1))
            sample = rng.choice(len(train_idx), size=min(search_sample, len(train_idx)), replace=False)
            Xs = X_train[sample]
            for j, k in enumerate(ks):
                emit(0.20 + 0.30 * j / max(len(ks), 1), f"K-Means: koko k={k} ({j + 1}/{len(ks)})...", "kmeans")
                km = KMeans(n_clusters=k, n_init=3, random_state=random_state).fit(Xs)
                if len(set(km.labels_)) < 2:
                    continue
                sil = silhouette_score(Xs, km.labels_, sample_size=min(3000, Xs.shape[0]),
                                       random_state=random_state)
                k_search.append({"k": k, "silhouette": float(sil), "inertia": float(km.inertia_)})
            best_k = max(k_search, key=lambda r: r["silhouette"])["k"] if k_search else min_k
        else:
            best_k = int(fixed_k)
            if best_k < 2 or best_k >= len(train_idx):
                raise TrainingError("invalid_params", f"Valór k={best_k} la válidu ba dadus ne'e.")
        emit(0.55, f"K-Means final ho k={best_k} (dadus treinu tomak)...", "kmeans")
        kmeans = KMeans(n_clusters=best_k, n_init=5, random_state=random_state).fit(X_train)
        sil_final = None
        if len(set(kmeans.labels_)) > 1:
            sil_final = float(silhouette_score(X_train, kmeans.labels_, sample_size=min(5000, X_train.shape[0]),
                                               random_state=random_state))
        finish("kmeans", t0, f"k = {best_k}" + (f" · silhouette {sil_final:.3f}" if sil_final is not None else ""))

        # 4) Identifikasaun tópiku
        emit(0.70, "Identifika tópiku (liafuan-xave no label)...", "topics")
        t0 = time.perf_counter()
        df["cluster"] = kmeans.predict(X_all).astype(int)
        topics = extract_top_terms_per_cluster(vec, kmeans, top_n=top_n)
        labels = build_topic_labels(topics)
        sizes = df["cluster"].value_counts().to_dict()
        save_keywords(topics, labels, staging, sizes, name=MODEL_NAME)
        themes = build_candidate_themes(topics, labels, df, X=X_all, kmeans=kmeans)
        wordlist = _wordlist(vec, kmeans, X_all)
        finish("topics", t0, f"{len(topics)} tópiku identifikadu")

        # 5) Avaliasaun
        emit(0.82, "Avaliasaun ho dadus teste...", "evaluation")
        t0 = time.perf_counter()
        evaluation = evaluate_clusters(
            df.loc[train_idx, "cluster"].to_numpy(), cats.iloc[train_idx].to_numpy(),
            df.loc[test_idx, "cluster"].to_numpy(), cats.iloc[test_idx].to_numpy(),
            exclude_labels=("Jerál",) if exclude_generic else ())
        finish("evaluation", t0, f"akurasia {evaluation['accuracy']:.1%} · F1 {evaluation['f1_macro']:.1%}")

        # 6) Rai
        emit(0.90, "Rai modelu, dadus ho klaster no rezultadu...", "save")
        t0 = time.perf_counter()
        extractor = TfidfFeatureExtractor(max_features=max_features, ngram_range=tuple(ngram_range), min_df=eff_min_df)
        extractor.vectorizer = vec
        extractor.save(staging, name=MODEL_NAME)
        topic_model = TopicKMeans(min_k=min_k, max_k=max_k, random_state=random_state)
        topic_model.kmeans, topic_model.best_k = kmeans, best_k
        topic_model.silhouette_scores_ = {r["k"]: r["silhouette"] for r in k_search}
        topic_model.inertia_scores_ = {r["k"]: r["inertia"] for r in k_search}
        topic_model.save(staging, name=MODEL_NAME)

        with MultiFormatWriter(staging, os.path.splitext(CLUSTERED_CSV)[0], CLUSTER_COLUMNS,
                               formats=("csv", "sqlite"), key_column="news_id",
                               int_columns=("token_count", "cluster")) as w:
            out = df[CLUSTER_COLUMNS].copy()
            out["summary"] = out["summary"].str.slice(0, 600)
            for i in range(0, len(out), 5000):
                w.write_rows(out.iloc[i:i + 5000].to_dict("records"))
        with open(os.path.join(staging, KEYWORDS_JSON), "r", encoding="utf-8") as f:
            topics_payload = json.load(f)

        meta = {
            "version_id": version_id, "trained_at": datetime.now().isoformat(timespec="seconds"),
            "source_name": os.path.basename(os.path.dirname(path)) or os.path.basename(path),
            "dataset_path": path, "n_docs_input": int(n), "n_docs_used": int(n),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "n_features": int(X_all.shape[1]),
            "best_k": int(best_k), "silhouette": sil_final,
            "params": {"min_k": min_k, "max_k": max_k, "k_mode": k_mode, "fixed_k": fixed_k,
                       "max_features": max_features, "ngram_range": list(ngram_range), "min_df": eff_min_df,
                       "test_fraction": test_fraction, "exclude_generic": exclude_generic},
            "themes": themes, "evaluation": evaluation, "k_search": k_search, "wordlist": wordlist,
            "stages": [dict(v, status="remata") for v in board.values()],
            "seconds": round(time.time() - started, 1),
        }
        with open(os.path.join(staging, META_JSON), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        registry.promote(version_id)
        promoted = True
        finish("save", t0, f"versaun {version_id}")
        meta["stages"] = [dict(v) for v in board.values()]
        progress(1.0, f"Treinu remata: modelu versaun {version_id} ativu ona.", stages=meta["stages"],
                 current_stage=None)
        return {"status": "remata", "version_id": version_id, "n_docs": n, "best_k": best_k,
                "evaluation": evaluation, "silhouette": sil_final, "seconds": meta["seconds"],
                "topics_payload": topics_payload}
    except _Stopped:
        progress(None, "Treinu para. Modelu antigu kontinua ativu.")
        return {"status": "para", "message": "Utilizadór para treinu."}
    except TrainingError:
        raise
    except MemoryError:
        raise TrainingError("memory", "Memória la to'o atu treina dadus ne'e. Hamenus max_features ka uza dadus ki'ik liu.")
    except Exception as e:
        logger.exception("Treinu falla")
        raise TrainingError("training_failed", f"Treinu falla: {e}") from e
    finally:
        if not promoted:
            registry.discard(version_id)
