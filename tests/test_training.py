import pytest

from models.training import build_bundle, load_model, predict_topic, save_model, train

TEXTS = (
    ["eskola ministru edukasaun estudante profesór aula"] * 12
    + ["ekonomia merkadu osan baku produtu empreza"] * 12
    + ["saúde ospitál doutór moras vasina pasiente"] * 12
)


def test_train_save_load_roundtrip(tmp_path):
    bundle, labels = train(TEXTS, min_k=2, max_k=5, min_df=1, silhouette_sample_size=None)
    assert len(labels) == len(TEXTS)
    assert bundle["metadata"]["best_k"] == bundle["kmeans"].n_clusters
    assert set(bundle["topics"]) == set(range(bundle["kmeans"].n_clusters))

    path = save_model(bundle, str(tmp_path / "model.pkl"))
    loaded = load_model(path)
    assert loaded["labels"] == bundle["labels"]


def test_predict_topic_same_cluster_as_training(tmp_path):
    bundle, labels = train(TEXTS, min_k=2, max_k=5, min_df=1, silhouette_sample_size=None)
    res = predict_topic("ministru edukasaun eskola estudante", bundle=bundle)
    assert res["cluster_id"] == int(labels[0])
    assert res["top_terms"]


def test_load_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "nope.pkl"))
