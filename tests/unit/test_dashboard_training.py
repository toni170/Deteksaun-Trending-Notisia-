"""Teste treinu TF-IDF + K-Means ho avaliasaun (akurasia, presizaun, recall, F1) no lista liafuan."""
import json
import os
import sqlite3
import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest

if "feedparser" not in sys.modules:
    try:
        import feedparser  # noqa: F401
    except ImportError:
        sys.modules["feedparser"] = types.ModuleType("feedparser")

from src.preprocessing.staged_pipeline import run_staged_preprocessing  # noqa: E402
from src.training.dashboard_training import TrainingError, train_from_clean_file  # noqa: E402
from src.training.evaluation import evaluate_clusters, majority_mapping  # noqa: E402
from src.training.model_registry import ModelRegistry  # noqa: E402
from tests.unit.test_staged_pipeline import make_raw, write_csv  # noqa: E402


def clean_dataset(tmp_path, n_per=25):
    out = run_staged_preprocessing(write_csv(make_raw(n_per), tmp_path), str(tmp_path / "bersih"))
    return out["files"]["csv"]


def test_evaluation_perfect_and_mapping():
    tr_c, tr_l = [0, 0, 1, 1, 2, 2], ["A", "A", "B", "B", "C", "C"]
    te_c, te_l = [0, 1, 2, 2], ["A", "B", "C", "C"]
    ev = evaluate_clusters(tr_c, tr_l, te_c, te_l)
    assert ev["accuracy"] == 1.0 and ev["f1_macro"] == 1.0 and ev["recall_macro"] == 1.0
    assert ev["mapping"] == {"0": "A", "1": "B", "2": "C"}
    assert np.array(ev["confusion"]).trace() == 4


def test_evaluation_values_and_baseline():
    tr_c, tr_l = [0] * 6 + [1] * 2, ["G"] * 6 + ["P"] * 2   # klaster 0 -> G, 1 -> P
    te_c, te_l = [0, 0, 0, 1, 1, 0], ["G", "G", "G", "P", "G", "P"]
    ev = evaluate_clusters(tr_c, tr_l, te_c, te_l)
    assert abs(ev["accuracy"] - 4 / 6) < 1e-9
    per = {r["kategoria"]: r for r in ev["per_class"]}
    assert abs(per["P"]["presizaun"] - 0.5) < 1e-9 and abs(per["P"]["recall"] - 0.5) < 1e-9
    assert ev["baseline_label"] == "G" and abs(ev["baseline_accuracy"] - 4 / 6) < 1e-9


def test_evaluation_excludes_generic_and_rejects_empty():
    ev = evaluate_clusters([0, 1, 0, 1], ["A", "B", "Jerál", "Jerál"], [0, 1, 0], ["A", "B", "Jerál"],
                           exclude_labels=("Jerál",))
    assert ev["classes"] == ["A", "B"] and ev["n_test"] == 2
    with pytest.raises(ValueError):
        evaluate_clusters([0], ["Jerál"], [0], ["Jerál"], exclude_labels=("Jerál",))
    assert majority_mapping([0, 0, 1], ["x", "x", "y"]) == {0: "x", 1: "y"}


def test_train_end_to_end_saves_model_metrics_wordlist(tmp_path):
    path = clean_dataset(tmp_path)
    reg = ModelRegistry(str(tmp_path / "models"))
    events = []
    res = train_from_clean_file(path, reg, k_mode="auto", min_k=2, max_k=5, min_df=1,
                                progress=lambda f, m, **e: events.append((f, m, e)))
    assert res["status"] == "remata" and reg.current_version_id() == res["version_id"]
    ev = res["evaluation"]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        assert 0.0 <= ev[key] <= 1.0
    assert ev["accuracy"] > 0.8  # 3 tópiku ho liafuan-xave klaru

    bundle = reg.load_current()
    meta = bundle.meta
    assert meta["n_train"] + meta["n_test"] == 75 and meta["evaluation"]["n_test"] == meta["n_test"]
    assert meta["k_search"] and all("silhouette" in r for r in meta["k_search"])
    assert len(meta["wordlist"]["global"]) > 0 and str(0) in meta["wordlist"]["per_topic"]
    assert set(bundle.df["split"]) == {"treinu", "teste"} and "cluster" in bundle.df.columns
    assert len(bundle.df) == 75 and bundle.themes
    assert [s["key"] for s in meta["stages"]] == ["load", "tfidf", "kmeans", "topics", "evaluation", "save"]
    # CSV + SQLite ho klaster
    vdir = bundle.path
    assert os.path.isfile(os.path.join(vdir, "news_clustered.csv")) and os.path.isfile(os.path.join(vdir, "news_clustered.db"))
    assert sqlite3.connect(os.path.join(vdir, "news_clustered.db")).execute("select count(*) from articles").fetchone()[0] == 75
    fr = [e[0] for e in events if e[0] is not None]
    assert fr == sorted(fr) and fr[-1] == 1.0


def test_fixed_k_and_exclude_generic(tmp_path):
    path = clean_dataset(tmp_path)
    reg = ModelRegistry(str(tmp_path / "models"))
    res = train_from_clean_file(path, reg, k_mode="fixed", fixed_k=3, min_df=1, exclude_generic=True)
    assert res["best_k"] == 3 and "Jerál" not in res["evaluation"]["classes"]


def test_errors_keep_old_model(tmp_path):
    path = clean_dataset(tmp_path)
    reg = ModelRegistry(str(tmp_path / "models"))
    v1 = train_from_clean_file(path, reg, k_mode="fixed", fixed_k=3, min_df=1)["version_id"]

    bad = tmp_path / "ruim.csv"
    bad.write_text("title,text\nA,b\n", encoding="utf-8")
    with pytest.raises(TrainingError) as e:
        train_from_clean_file(str(bad), reg)
    assert "text_final" in e.value.message
    small = tmp_path / "ki.csv"
    pd.DataFrame({"text_final": ["eskola estudante"] * 5}).to_csv(small, index=False)
    with pytest.raises(TrainingError) as e:
        train_from_clean_file(str(small), reg)
    assert "Dadus la to'o" in e.value.message
    with pytest.raises(TrainingError) as e:
        train_from_clean_file(path, reg, test_fraction=0.9)
    assert "5%" in e.value.message
    assert reg.current_version_id() == v1 and not os.listdir(reg.staging_dir)


def test_stop_keeps_old_model(tmp_path):
    path = clean_dataset(tmp_path)
    reg = ModelRegistry(str(tmp_path / "models"))
    v1 = train_from_clean_file(path, reg, k_mode="fixed", fixed_k=3, min_df=1)["version_id"]
    stop = threading.Event()
    stop.set()
    res = train_from_clean_file(path, reg, stop_event=stop)
    assert res["status"] == "para" and reg.current_version_id() == v1
