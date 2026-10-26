"""Registu modelu + servisu treinu: progresu (#19), falla ba modelu antigu (#16), persisténsia (#17)."""
import os

import pytest

from src.ingestion.dataset_importer import import_dataset
from src.training import training_service as ts
from src.training.model_registry import ModelLoadError, ModelRegistry
from src.training.training_service import TrainingError, train_and_register
import io

pytestmark = pytest.mark.unit


@pytest.fixture
def raw(csv_bytes):
    return import_dataset(io.BytesIO(csv_bytes), "a.csv").df


@pytest.fixture
def raw_b(csv_bytes_b):
    return import_dataset(io.BytesIO(csv_bytes_b), "b.csv").df


def _staging_empty(registry):
    return not os.path.isdir(registry.staging_dir) or os.listdir(registry.staging_dir) == []


def test_empty_registry_has_no_current(registry):
    assert registry.current_version_id() is None and registry.load_current() is None
    assert registry.load_best_available() is None and registry.list_versions() == []


def test_train_registers_and_promotes_version(registry, raw):
    result = train_and_register(raw, registry, min_k=2, max_k=6, source_name="a.csv")
    assert registry.current_version_id() == result.version_id
    assert result.meta["best_k"] == 3 and result.meta["n_docs_used"] == len(raw)
    assert result.df["cluster"].between(0, 2).all()
    assert len(result.themes) == 3 and _staging_empty(registry)
    for fname in ("tatoli_tfidf_vectorizer.joblib", "tatoli_kmeans.joblib", "news_clustered.csv",
                  "tatoli_keywords_latest.json", "meta.json"):
        assert os.path.isfile(os.path.join(registry.versions_dir, result.version_id, fname))


# ---- #19 progresu ----------------------------------------------------------
def test_19_progress_callback_is_monotonic_tetun_and_ends_at_one(registry, raw):
    events = []
    train_and_register(raw, registry, progress_cb=lambda f, m: events.append((f, m)))
    fractions = [f for f, _ in events]
    assert fractions == sorted(fractions) and fractions[0] < 0.2 and fractions[-1] == 1.0
    assert len(set(fractions)) >= 5 and all(m.strip() for _, m in events)
    assert "TF-IDF" in " ".join(m for _, m in events) and "K-Means" in " ".join(m for _, m in events)


# ---- Validasaun ----------------------------------------------------------
def test_too_few_documents_raises_and_leaves_no_state(registry, raw):
    with pytest.raises(TrainingError) as exc:
        train_and_register(raw.head(2), registry)
    assert exc.value.code == "too_few_documents" and "Dadus la to'o" in exc.value.message
    assert registry.current_version_id() is None and _staging_empty(registry)


@pytest.mark.parametrize("min_k,max_k", [(1, 5), (5, 3)])
def test_invalid_cluster_params(registry, raw, min_k, max_k):
    with pytest.raises(TrainingError) as exc:
        train_and_register(raw, registry, min_k=min_k, max_k=max_k)
    assert exc.value.code == "invalid_params"


# ---- #16 modelu antigu salva bainhira falla ---------------------------------
def test_16_old_model_remains_when_new_training_fails_midway(registry, raw, raw_b, monkeypatch):
    v1 = train_and_register(raw, registry).version_id

    def boom(*a, **k):
        raise RuntimeError("K-Means rebenta")
    monkeypatch.setattr(ts, "run_clustering", boom)  # falla DEPOIS TF-IDF ho rai ona iha staging

    with pytest.raises(TrainingError) as exc:
        train_and_register(raw_b, registry)
    assert exc.value.code == "training_failed" and "K-Means rebenta" in exc.value.message

    assert registry.current_version_id() == v1
    assert registry.list_versions() == [v1] and _staging_empty(registry)
    bundle = registry.load_current()
    assert bundle.version_id == v1 and len(bundle.df) == len(raw)


# ---- #15 modelu foun troka modelu ativu ------------------------------------
def test_15_new_dataset_creates_new_active_version_and_keeps_old_on_disk(registry, raw, raw_b):
    v1 = train_and_register(raw, registry).version_id
    v2 = train_and_register(raw_b, registry).version_id
    assert v1 != v2 and registry.current_version_id() == v2
    assert registry.list_versions() == [v2, v1]
    b2 = registry.load_current()
    assert set(b2.df["source"]) == {"b"} and b2.meta["source_name"] == "import"
    vocab_new = set(b2.extractor.get_feature_names())
    vocab_old = set(registry.load_version(v1).extractor.get_feature_names())
    assert "futebol" in " ".join(vocab_new) and "futebol" not in " ".join(vocab_old)


# ---- #17 persisténsia (instánsia registu foun = restart) -------------------
def test_17_fresh_registry_instance_reloads_saved_model(models_dir, raw):
    result = train_and_register(raw, ModelRegistry(models_dir))
    reloaded = ModelRegistry(models_dir).load_current()
    assert reloaded.version_id == result.version_id
    assert reloaded.df["cluster"].tolist() == result.df["cluster"].tolist()
    assert reloaded.topics_payload["topics"] and reloaded.themes == result.themes
    X = reloaded.extractor.transform(["ospital mediku vasina"])
    assert int(reloaded.topic_model.predict(X)[0]) in range(reloaded.topic_model.best_k)


def test_17_corrupt_current_falls_back_to_previous_valid_version(registry, raw, raw_b):
    v1 = train_and_register(raw, registry).version_id
    v2 = train_and_register(raw_b, registry).version_id
    os.remove(os.path.join(registry.versions_dir, v2, "tatoli_kmeans.joblib"))
    with pytest.raises(ModelLoadError):
        registry.load_current()
    assert registry.load_best_available().version_id == v1


def test_prune_keeps_only_recent_versions_and_never_current(models_dir, raw):
    reg = ModelRegistry(models_dir, keep_versions=2)
    ids = [train_and_register(raw, reg).version_id for _ in range(3)]
    assert reg.list_versions() == [ids[2], ids[1]] and reg.current_version_id() == ids[2]
