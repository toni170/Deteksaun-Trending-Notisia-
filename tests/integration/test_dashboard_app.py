"""
Integrasaun dashboard Streamlit (AppTest, la presiza browser): #3 #15 #16 #17 #18 #19 #20 iha nível UI.
`main.py` (run_pipeline) la iha iha zip -> stub; teste ne'e la kobre koleta scraping.
"""
import os
import sys
import types

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import src.utils
from src.dashboard.import_view import MSG
from src.training.model_registry import ModelRegistry
from tests.conftest import to_csv_bytes

pytestmark = pytest.mark.integration
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "src", "dashboard", "trend_dashboard.py")


@pytest.fixture
def config(tmp_path, monkeypatch):
    paths = {k: str(tmp_path / k) for k in ("logs", "cleaned", "keywords", "models", "raw")}
    for p in paths.values():
        os.makedirs(p)
    cfg = {"paths": paths, "clustering": {"min_k": 2, "max_k": 6}, "sources": [],
           "scraping": {"full_content": False}, "history": {"start_date": "2016-01-01"}}
    monkeypatch.setattr(src.utils, "load_config", lambda *a, **k: cfg)
    stub = types.ModuleType("main")

    def _no_scraping(*a, **k):
        raise AssertionError("scraping la tenke xama iha teste ne'e")
    stub.run_pipeline = _no_scraping
    monkeypatch.setitem(sys.modules, "main", stub)
    st.cache_data.clear()
    st.cache_resource.clear()
    return cfg


def new_app() -> AppTest:
    """Sesaun foun (session_state mamuk) = utilizadór loke dashboard."""
    st.cache_data.clear()
    st.cache_resource.clear()
    return AppTest.from_file(APP, default_timeout=180).run()


def upload(at, name, data):
    at.sidebar.file_uploader(key="dataset_upload").set_value((name, data, "application/octet-stream"))
    at.run()  # botaun ativa depois widget simu valór
    return at


def train(at):
    at.sidebar.button(key="train_btn").click().run()
    return at


def texts(at) -> str:
    kinds = ["title", "header", "subheader", "markdown", "caption", "text", "info", "success", "warning",
             "error", "button", "tab", "metric", "expander", "status", "file_uploader"]
    out = []
    for k in kinds:
        for el in at.get(k):
            for attr in ("value", "label", "body"):
                v = getattr(el, attr, None)
                if isinstance(v, str):
                    out.append(v)
    return "\n".join(out)


def theme_expanders(at):
    return [e for e in at.expander if e.label.startswith("Tema ")]


# ---- #18 dashboard iha Tetun ----------------------------------------------
REQUIRED_TETUN = ["Importa Dataset & Treina Modelu", "Hili ficheiru dataset", "Importa no Treina Modelu",
                  "Seidauk hili ficheiru", "Halo pedidu manuál agora", "Seidauk iha modelu ne'ebé rai ona"]
FORBIDDEN_ENGLISH = ["Upload", "Train model", "Training", "Please", "Error:", "Success", "Loading", "Import dataset",
                     "No data", "Select file", "Failed"]


def test_18_empty_dashboard_is_in_tetun(config):
    at = new_app()
    assert not at.exception
    page = texts(at)
    for needle in REQUIRED_TETUN:
        assert needle in page, f"falta tékstu Tetun: {needle!r}"
    for bad in FORBIDDEN_ENGLISH:
        assert bad not in page, f"tékstu Inglés la tuir: {bad!r}"


def test_18_trained_dashboard_is_in_tetun(config, csv_bytes):
    at = train(upload(new_app(), "a.csv", csv_bytes))
    assert not at.exception
    page = texts(at)
    for needle in ["Total Notísia", "Kuantidade Tópiku", "Tema Kandidatu", "Liafuan Xave", "Istória",
                   "Modelu foun treina ona", "Liafuan xave", "Títulu exemplu"]:
        assert needle in page, f"falta tékstu Tetun: {needle!r}"
    for bad in FORBIDDEN_ENGLISH:
        assert bad not in page, f"tékstu Inglés la tuir: {bad!r}"


# ---- #19 progresu / loading / error / success --------------------------------
def test_19_loading_state_status_container_is_created_running(config, csv_bytes, monkeypatch):
    calls = []
    orig = st.status

    def spy(label, *a, **k):
        calls.append({"label": label, "state": k.get("state", "running"), "expanded": k.get("expanded")})
        return orig(label, *a, **k)
    monkeypatch.setattr(st, "status", spy)
    train(upload(new_app(), "a.csv", csv_bytes))
    assert calls and calls[0] == {"label": MSG["running"], "state": "running", "expanded": True}


def test_19_success_state_shows_progress_status_and_message(config, csv_bytes):
    at = train(upload(new_app(), "a.csv", csv_bytes))
    assert not at.exception and not at.error
    assert at.progress[0].value == 100
    assert [s.state for s in at.status] == ["complete"]
    assert len(at.success) == 1 and "Modelu foun treina ona" in at.success[0].value


@pytest.mark.parametrize("name,data,fragment", [
    ("rusak.xlsx", b"bukan excel", "La bele lee ficheiru"),
    ("nota.txt", b"hello", "la suporta"),
    ("tanpa_isi.csv", b"title,date\nA,2024-01-01\n", "konteúdu"),
    ("kosong.csv", b"", "Ficheiru mamuk"),
])
def test_19_error_state_is_shown_in_tetun_for_invalid_files(config, name, data, fragment):
    at = train(upload(new_app(), name, data))
    assert not at.exception  # erru trata, la iha traceback
    assert [s.state for s in at.status] == ["error"]
    assert len(at.error) == 1 and "Erru:" in at.error[0].value and fragment in at.error[0].value
    assert not at.success


def test_19_too_few_rows_shows_error(config):
    at = train(upload(new_app(), "kiik.csv", b"title,content\nEskola,estudante eskola ensinu\nOspital,mediku ospital vasina\n"))
    assert len(at.error) == 1 and "Dadus la to'o" in at.error[0].value


# ---- #3 / #16 modelu antigu kontinua bainhira import falla ------------------
def test_16_old_model_stays_active_after_failed_import(config, csv_bytes):
    at = train(upload(new_app(), "a.csv", csv_bytes))
    v1 = at.session_state["news_result"]["model_meta"]["version_id"]
    total_before = [m.value for m in at.metric if "Total" in m.label]

    at = train(upload(at, "rusak.xlsx", b"bukan excel"))
    assert not at.exception and len(at.error) == 1
    assert any(f"versaun {v1}" in i.value and "kontinua" in i.value for i in at.info)
    assert at.session_state["news_result"]["model_meta"]["version_id"] == v1
    assert [m.value for m in at.metric if "Total" in m.label] == total_before
    assert ModelRegistry(config["paths"]["models"]).current_version_id() == v1


def test_16_old_model_stays_active_after_training_crash(config, csv_bytes, csv_bytes_b, monkeypatch):
    at = train(upload(new_app(), "a.csv", csv_bytes))
    v1 = at.session_state["news_result"]["model_meta"]["version_id"]
    from src.training import training_service as ts
    monkeypatch.setattr(ts, "run_clustering", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    at = train(upload(at, "b.csv", csv_bytes_b))
    assert not at.exception and "crash" in at.error[0].value
    assert ModelRegistry(config["paths"]["models"]).current_version_id() == v1


# ---- #15 dataset foun -> modelu foun ----------------------------------------
def test_15_importing_second_dataset_trains_new_model(config, csv_bytes, csv_bytes_b):
    at = train(upload(new_app(), "a.csv", csv_bytes))
    v1 = at.session_state["news_result"]["model_meta"]["version_id"]
    at = train(upload(at, "b.csv", csv_bytes_b))
    assert not at.exception and not at.error
    meta = at.session_state["news_result"]["model_meta"]
    assert meta["version_id"] != v1 and meta["source_name"] == "b.csv"
    assert ModelRegistry(config["paths"]["models"]).current_version_id() == meta["version_id"]
    assert set(at.session_state["news_result"]["df"]["source"]) == {"b"}


# ---- #17 restart -> karrega modelu rai ona ----------------------------------
def test_17_restart_reloads_saved_model_without_any_action(config, csv_bytes):
    first = train(upload(new_app(), "a.csv", csv_bytes))
    meta = first.session_state["news_result"]["model_meta"]
    total = [m.value for m in first.metric if "Total" in m.label]

    restarted = new_app()  # sesaun foun + cache hamoos
    assert not restarted.exception
    assert restarted.session_state["news_result"]["model_meta"]["version_id"] == meta["version_id"]
    assert [m.value for m in restarted.metric if "Total" in m.label] == total
    assert len(theme_expanders(restarted)) == 3
    assert f"versaun {meta['version_id']}" in texts(restarted)


def test_17_newer_scraped_master_wins_over_older_saved_model(config, csv_bytes, news_df):
    train(upload(new_app(), "a.csv", csv_bytes))
    from src.preprocessing.pipeline import preprocess_dataframe
    master = preprocess_dataframe(news_df.head(9))
    master["cluster"] = [i % 3 for i in range(len(master))]
    path = os.path.join(config["paths"]["cleaned"], "tatoli_cleaned_master.csv")
    master.to_csv(path, index=False, encoding="utf-8-sig")
    os.utime(path, (4_000_000_000, 4_000_000_000))  # master foun liu modelu
    at = new_app()
    assert not at.exception and "model_meta" not in at.session_state["news_result"]
    assert len(at.session_state["news_result"]["df"]) == 9


def test_17_corrupt_saved_model_does_not_crash_dashboard(config, csv_bytes):
    train(upload(new_app(), "a.csv", csv_bytes))
    reg = ModelRegistry(config["paths"]["models"])
    os.remove(os.path.join(reg.versions_dir, reg.current_version_id(), "tatoli_kmeans.joblib"))
    at = new_app()
    assert not at.exception
    assert at.session_state["news_result"] is None  # la iha versaun válidu -> estadu mamuk, la crash


# ---- #20 workflow kompletu husi dashboard de'it -----------------------------
def test_20_full_workflow_from_dashboard_only(config, csv_bytes, csv_bytes_b):
    # 1. Loke dashboard mamuk
    at = new_app()
    assert not at.exception and at.session_state["news_result"] is None
    assert not at.metric and not theme_expanders(at)

    # 2. Hili ficheiru invalidu -> erru, la iha modelu
    at = train(upload(at, "rusak.xlsx", b"bukan excel"))
    assert len(at.error) == 1 and ModelRegistry(config["paths"]["models"]).current_version_id() is None

    # 3. Hili ficheiru válidu -> treina -> dashboard hatudu rezultadu
    at = train(upload(at, "a.csv", csv_bytes))
    assert not at.exception and not at.error and len(at.success) == 1
    assert [m.value for m in at.metric if "Total" in m.label] == ["24"]
    assert [m.value for m in at.metric if "Tópiku" in m.label] == ["3"]
    assert len(theme_expanders(at)) == 3
    v1 = at.session_state["news_result"]["model_meta"]["version_id"]

    # 4. Importa dataset foun -> modelu foun
    at = train(upload(at, "b.csv", csv_bytes_b))
    v2 = at.session_state["news_result"]["model_meta"]["version_id"]
    assert not at.exception and v2 != v1 and len(theme_expanders(at)) == 3

    # 5. Restart -> modelu ikus karrega fali
    again = new_app()
    assert not again.exception
    assert again.session_state["news_result"]["model_meta"]["version_id"] == v2
    assert len(theme_expanders(again)) == 3


# ---- MemoryError bainhira lee master CSV boot --------------------------------
def _write_master(config, news_df, rows=9):
    from src.preprocessing.pipeline import preprocess_dataframe
    master = preprocess_dataframe(news_df.head(rows))
    master["cluster"] = [i % 3 for i in range(len(master))]
    path = os.path.join(config["paths"]["cleaned"], "tatoli_cleaned_master.csv")
    master.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_memory_error_falls_back_to_recent_rows_and_warns_in_tetun(config, news_df, monkeypatch):
    from src.dashboard import data_loader as dl
    _write_master(config, news_df, rows=20)
    real = dl.read_cleaned_light

    def oom_unless_limited(path, chunksize=2000, max_rows=None):
        if max_rows is None:
            raise MemoryError
        return real(path, chunksize=5, max_rows=8)
    monkeypatch.setattr(dl, "read_cleaned_light", oom_unless_limited)

    at = new_app()
    assert not at.exception
    assert len(at.session_state["news_result"]["df"]) == 8
    assert any("Memória la to'o" in w.value and "8" in w.value and "20" in w.value for w in at.warning)


def test_memory_error_everywhere_shows_tetun_error_not_traceback(config, news_df, monkeypatch):
    from src.dashboard import data_loader as dl
    _write_master(config, news_df)

    def always_oom(*a, **k):
        raise MemoryError
    monkeypatch.setattr(dl, "read_cleaned_light", always_oom)

    at = new_app()
    assert not at.exception  # la crash hanesan iha relatóriu utilizadór
    assert at.session_state["news_result"] is None
    assert any("Memória la to'o" in e.value for e in at.error)


# ---- Fluxu servisu foun: tab pedidu, preprosesamentu, treinu --------------------
def test_new_workflow_tabs_exist_in_tetun_and_no_old_collect_button(config):
    at = new_app()
    assert not at.exception
    page = texts(at)
    for needle in ["1. Pedidu Dadus", "2. Preprosesamentu", "3. Treinu Modelu", "4. Analiza Notísia",
                   "Halo pedidu manuál agora", "Ativa pedidu automátiku", "Hala'o preprosesamentu hotu-hotu"]:
        assert needle in page, f"falta tékstu Tetun: {needle!r}"
    assert "Kolete Dadus Agora" not in page and "Auto-refresh" not in page
