"""Integrasaun (la iha UI): ficheiru -> importa -> preprocessing -> TF-IDF -> K-Means -> registu -> restart."""
import io
import json
import os
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from src.ingestion.dataset_importer import DatasetImportError, import_dataset
from src.training.model_registry import ModelRegistry
from src.training.training_service import train_and_register

pytestmark = pytest.mark.integration
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_csv_and_excel_give_equivalent_trained_models(registry, models_dir, csv_bytes, xlsx_bytes, tmp_path):
    csv_df = import_dataset(io.BytesIO(csv_bytes), "a.csv").df
    xlsx_df = import_dataset(io.BytesIO(xlsx_bytes), "a.xlsx").df
    r_csv = train_and_register(csv_df, registry)
    r_xlsx = train_and_register(xlsx_df, ModelRegistry(str(tmp_path / "m2")))
    assert r_csv.meta["best_k"] == r_xlsx.meta["best_k"] == 3
    assert sorted(t["news_count"] for t in r_csv.themes) == sorted(t["news_count"] for t in r_xlsx.themes)


def test_dirty_dataset_is_cleaned_then_trained(registry, news_df):
    base = news_df.drop(columns=["_topic"])
    extra_empty = pd.DataFrame([{"title": "", "content": "", "link": ""}])
    dirty = pd.concat([base, base.iloc[[0, 1]], extra_empty], ignore_index=True)  # 2 duplikadu + 1 mamuk
    result = import_dataset(io.BytesIO(dirty.to_csv(index=False).encode()), "dirty.csv")
    assert (result.report.rows_removed_duplicate, result.report.rows_removed_empty) == (2, 1)
    trained = train_and_register(result.df, registry)
    assert trained.meta["n_docs_used"] == len(news_df)


def test_failed_import_never_touches_active_model(registry, csv_bytes, csv_bytes_b):
    v1 = train_and_register(import_dataset(io.BytesIO(csv_bytes), "a.csv").df, registry).version_id
    for name, data in [("rusak.xlsx", b"junk"), ("tanpa_isi.csv", b"title\nx\n"), ("kosong.csv", b"")]:
        with pytest.raises(DatasetImportError):
            import_dataset(io.BytesIO(data), name)
    assert registry.current_version_id() == v1 and registry.list_versions() == [v1]
    v2 = train_and_register(import_dataset(io.BytesIO(csv_bytes_b), "b.csv").df, registry).version_id
    assert registry.current_version_id() == v2 != v1


def test_17_model_reloads_in_a_separate_python_process(models_dir, csv_bytes):
    """Restart loloos: prosesu Python foun karrega modelu husi disku no predict hanesan."""
    trained = train_and_register(import_dataset(io.BytesIO(csv_bytes), "a.csv").df, ModelRegistry(models_dir))
    script = textwrap.dedent(f"""
        import json
        from src.training.model_registry import ModelRegistry
        b = ModelRegistry({models_dir!r}).load_current()
        pred = int(b.topic_model.predict(b.extractor.transform(["ospital mediku vasina"]))[0])
        print(json.dumps({{"version": b.version_id, "n": len(b.df), "k": b.topic_model.best_k,
                          "pred": pred, "themes": len(b.themes)}}))
    """)
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": ROOT}, timeout=120)
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["version"] == trained.version_id and out["n"] == len(trained.df)
    assert out["k"] == 3 and out["themes"] == 3 and 0 <= out["pred"] < 3
