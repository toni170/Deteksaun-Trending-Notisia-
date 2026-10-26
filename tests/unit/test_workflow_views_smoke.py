"""
Teste smoke ba tab fluxu servisu. Se Streamlit/Plotly la instala, uza versaun falsu (MagicMock)
atu verifika kódigu la iha erru (naran sala, argumentu sala) no tékstu mosu iha Tetun.
"""
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

if "feedparser" not in sys.modules:
    try:
        import feedparser  # noqa: F401
    except ImportError:
        sys.modules["feedparser"] = types.ModuleType("feedparser")

REAL_ST = True
try:
    import streamlit  # noqa: F401
    import plotly  # noqa: F401
except ImportError:
    REAL_ST = False


class Col:
    """Kolona/tab falsu: metodu hotu delega ba Streamlit falsu (atu rejista tékstu)."""

    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __getattr__(self, name):
        return getattr(self._fake, name)


class FakeStreamlit(types.ModuleType):
    """Streamlit falsu: widget retorna valór padraun; rejista tékstu hotu ne'ebé hatudu."""

    def __init__(self, with_fragment=True):
        super().__init__("streamlit")
        self.session_state = {}
        self.texts = []
        self.calls = []
        if with_fragment:
            self.fragment = self._fragment

    def _fragment(self, func=None, *, run_every=None):
        if func is not None:
            return func
        return lambda f: f

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        rec = self

        def generic(*a, **k):
            rec.calls.append((name, a, k))
            for x in a[:1]:
                if isinstance(x, str):
                    rec.texts.append(x)
            if "text" in k and isinstance(k["text"], str):
                rec.texts.append(k["text"])
            return MagicMock()
        return generic

    def cache_data(self, *a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

    def columns(self, n, **k):
        n = n if isinstance(n, int) else len(n)
        return [Col(self) for _ in range(n)]

    def tabs(self, labels):
        return [Col(self) for _ in labels]

    def button(self, *a, **k):
        self.texts.append(a[0])
        return False

    def toggle(self, label, value=False, **k):
        self.texts.append(label)
        return value

    def number_input(self, label, min_value=None, max_value=None, value=None, **k):
        self.texts.append(label)
        return value if value is not None else min_value

    def selectbox(self, label, options, index=0, **k):
        self.texts.append(label)
        return list(options)[index] if len(list(options)) else None

    def radio(self, label, options, index=0, **k):
        self.texts.append(label)
        return list(options)[index]

    def slider(self, label, a=0, b=100, value=None, **k):
        self.texts.append(label)
        return value if value is not None else a

    def checkbox(self, label, **k):
        self.texts.append(label)
        return False

    def file_uploader(self, label, **k):
        self.texts.append(label)
        return None

    def metric(self, label, value, **k):
        self.calls.append(("metric", (label, value), k))
        self.texts.append(label)

    def rerun(self):
        raise RuntimeError("rerun")


def install_fakes(with_fragment=True):
    fake = FakeStreamlit(with_fragment)
    sys.modules["streamlit"] = fake
    px = MagicMock()
    plotly = types.ModuleType("plotly")
    plotly.express = px
    sys.modules["plotly"], sys.modules["plotly.express"] = plotly, px
    for m in [m for m in sys.modules if m.startswith("src.dashboard.workflow_views")]:
        del sys.modules[m]
    return fake


@pytest.fixture
def cfg(tmp_path):
    paths = {k: str(tmp_path / k) for k in ("cleaned", "models", "batches", "uploads")}
    return {"paths": paths, "batch": {"auto_enabled": False}, "training": {}}


def _trained_model(cfg, tmp_path):
    from src.preprocessing.staged_pipeline import run_staged_preprocessing
    from src.training.dashboard_training import train_from_clean_file
    from src.training.model_registry import ModelRegistry
    from tests.unit.test_staged_pipeline import make_raw, write_csv
    out = run_staged_preprocessing(write_csv(make_raw(20), tmp_path), cfg["paths"]["cleaned"])
    train_from_clean_file(out["files"]["csv"], ModelRegistry(cfg["paths"]["models"]), k_mode="fixed", fixed_k=3,
                          min_df=1)


def _all_text(fake):
    return "\n".join(str(t) for t in fake.texts)


def test_empty_state_renders_without_metrics_and_in_tetun(cfg):
    if REAL_ST:
        return
    fake = install_fakes()
    import importlib
    wv = importlib.import_module("src.dashboard.workflow_views")
    src = [{"name": "Tatoli", "feed_url": "https://t.test/feed/", "enabled": True}]
    wv.render_request_tab(cfg, src)
    wv.render_preprocessing_workflow(cfg)
    wv.render_training_workflow(cfg)
    assert not [c for c in fake.calls if c[0] == "metric"]
    page = _all_text(fake)
    for needle in ["Halo pedidu manuál agora", "Ativa pedidu automátiku", "Hala'o preprosesamentu hotu-hotu",
                   "Hahú treinu TF-IDF + K-Means"]:
        assert needle in page
    for bad in ["Upload", "Training", "Loading", "Success", "Failed", "Please", "Error:", "Download"]:
        assert bad not in page


@pytest.mark.parametrize("with_fragment", [True, False])
def test_trained_model_shows_metrics_wordlist_and_trending(cfg, tmp_path, with_fragment):
    if REAL_ST:
        return
    fake = install_fakes(with_fragment)
    _trained_model(cfg, tmp_path)
    import importlib
    wv = importlib.import_module("src.dashboard.workflow_views")
    wv.render_training_workflow(cfg)
    labels = [c[1][0] for c in fake.calls if c[0] == "metric"]
    assert labels == ["Akurasia", "Presizaun (makro)", "Recall (makro)", "F1-score (makro)"]
    page = _all_text(fake)
    for needle in ["Tópiku populár", "Lista liafuan", "Tópiku ne'ebé identifika"]:
        assert any(needle in str(c) for c in fake.calls), needle


def test_collection_monitor_shows_running_job_and_stop(cfg):
    if REAL_ST:
        return
    fake = install_fakes()
    import importlib
    wv = importlib.import_module("src.dashboard.workflow_views")
    from src.jobs.job_manager import JobManager
    import threading
    mgr = JobManager()
    gate = threading.Event()

    def job(progress, stop):
        progress(0.4, "Tatoli: pájina 3", rows=120, mb=80.5, target_mb=200, per_source={"Tatoli": 120})
        gate.wait(3)
    mgr.start("koleta", job)
    import time
    time.sleep(0.2)
    wv.get_job_manager = lambda: mgr
    fake.session_state["_scheduler"] = None
    wv._collection_monitor("koleta", True)
    labels = [c[1][0] for c in fake.calls if c[0] == "metric"]
    assert labels == ["Notísia foti", "Tamañu CSV (MB)", "Meta (MB)"]
    gate.set()
