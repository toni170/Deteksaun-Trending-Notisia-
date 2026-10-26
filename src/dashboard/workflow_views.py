"""
Tab fluxu servisu iha painel: 1) Pedidu dadus  2) Preprosesamentu  3) Treinu modelu.

Servisu naruk hala'o iha thread background (`src.jobs`). Painel de'it hatudu progresu liuhusi
fragmentu ne'ebé atualiza an-an kada 2 segundu (la presiza hamoos pájina tomak).
Tékstu hotu iha Tetun.
"""
import json
import os
import shutil
from datetime import datetime

import pandas as pd
import streamlit as st

from src.collection.batch_collector import DEFAULT_TARGET_MB, list_batches
from src.jobs.job_manager import DONE, FAILED, RUNNING, STOPPED, get_job_manager
from src.jobs.requests_service import COLLECTION_JOB, get_scheduler, start_collection_request
from src.paths import workflow_dirs
from src.preprocessing.staged_pipeline import (
    SUPPORTED_INPUT, list_clean_datasets, run_staged_preprocessing,
)
from src.training.analysis import topic_time_series, trending_topics
from src.training.dashboard_training import TRAIN_STAGES, train_from_clean_file
from src.training.model_registry import ModelRegistry

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # pakote opsionál
    st_autorefresh = None

PREP_JOB, TRAIN_JOB, HEAVY_GROUP = "preprosesamentu", "treinu", "todan"
STATUS_ICON = {"hein": "⏳ Hein", "hala'o": "🔄 Hala'o", "remata": "✅ Remata"}
JOB_STATUS_TETUN = {RUNNING: "hala'o hela", DONE: "remata", STOPPED: "para", FAILED: "falla"}


# ----------------------------------------------------------------------
# Ajuda jerál
# ----------------------------------------------------------------------
def _mb(n_bytes) -> str:
    return f"{(n_bytes or 0) / 1024 / 1024:,.1f} MB"


def _dir_size(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _fmt_seconds(sec) -> str:
    sec = int(sec or 0)
    return f"{sec // 60} minutu {sec % 60} segundu" if sec >= 60 else f"{sec} segundu"


def _save_upload(uploaded, uploads_dir: str) -> str:
    """Rai ficheiru ne'ebé karga ba disku (lote 8 MB) no retorna fatin."""
    os.makedirs(uploads_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in uploaded.name)
    path = os.path.join(uploads_dir, f"{datetime.now():%Y%m%d_%H%M%S}_{safe}")
    uploaded.seek(0)
    with open(path, "wb") as out:
        shutil.copyfileobj(uploaded, out, 8 * 1024 * 1024)
    return path


def _make_fragment(func, run_every):
    """Fragmentu Streamlit (≥1.37). Se la iha, uza funsaun normál."""
    frag = getattr(st, "fragment", None)
    if frag is None:
        return func

    def wrapper(*a, **k):
        return func(*a, **k)
    wrapper.__name__ = wrapper.__qualname__ = f"{func.__name__}_{'live' if run_every else 'quieto'}"
    wrapper.__module__ = func.__module__
    return frag(run_every=run_every)(wrapper) if run_every else frag(wrapper)


def _render_monitor(name: str, pair: tuple, idle_poll: bool = False) -> None:
    """Hili variante fragmentu: "live" bainhira servisu hala'o hela, se lae "quieto"."""
    running = get_job_manager().is_running(name)
    live, quiet = pair
    if getattr(st, "fragment", None) is None:
        if running and st_autorefresh is not None:
            st_autorefresh(interval=3000, key=f"auto_{name}")
        live(name, False)
        return
    (live if running else quiet)(name, running)


def _sync_variant(name: str, was_running: bool) -> None:
    """Troka variante fragmentu (live <-> quieto) bainhira estadu servisu muda."""
    if getattr(st, "fragment", None) is not None and get_job_manager().is_running(name) != was_running:
        st.rerun()


def _stage_table(stages: list, current=None) -> None:
    if not stages:
        return
    rows = []
    for s in stages:
        status = STATUS_ICON.get(s.get("status"), s.get("status", ""))
        if current and s.get("key") == current:
            status = "▶️ Agora"
        rows.append({"Etapa": s["label"], "Estadu": status,
                     "Notísia prosesa": f"{s['rows']:,}" if "rows" in s and s["rows"] else "",
                     "Tempu (segundu)": round(s.get("seconds", 0.0), 1), "Detallu": s.get("detail", "")})
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def _job_banner(snap: dict, ok_text: str) -> None:
    status = snap["status"]
    if status == DONE:
        st.success(ok_text)
    elif status == STOPPED:
        st.warning("Servisu para husi utilizadór. " + (snap["message"] or ""))
    elif status == FAILED:
        st.error(f"❌ Erru: {snap['error']}")


def _log_expander(snap: dict) -> None:
    if snap["log"]:
        with st.expander("📜 Rejistu servisu (liña ikus)"):
            st.code("\n".join(snap["log"][-15:]), language=None)


# ----------------------------------------------------------------------
# TAB 1 — Pedidu dadus
# ----------------------------------------------------------------------
def _collection_monitor(name: str, was_running: bool) -> None:
    _sync_variant(name, was_running)
    manager = get_job_manager()
    snap = manager.snapshot(name)
    sched = st.session_state.get("_scheduler")
    if snap["status"] == RUNNING:
        extra = snap["extra"]
        st.progress(snap["fraction"], text=(snap["message"] or "Hala'o...")[:180])
        c1, c2, c3 = st.columns(3)
        c1.metric("Notísia foti", f"{extra.get('rows', 0):,}")
        c2.metric("Tamañu CSV (MB)", f"{extra.get('mb', 0):,.1f}")
        c3.metric("Meta (MB)", f"{extra.get('target_mb', 0):,.0f}")
        if extra.get("per_source"):
            st.caption("Notísia kada orijen: " + " · ".join(f"{k}: {v:,}" for k, v in extra["per_source"].items()))
    elif snap["status"] in (DONE, STOPPED, FAILED):
        res = snap["result"] or {}
        if snap["status"] == DONE and res.get("status") == "mamuk":
            st.info("ℹ️ " + res.get("message", "La iha notísia foun."))
        else:
            _job_banner(snap, f"✅ Pedidu remata: {res.get('rows', 0):,} notísia, "
                              f"{res.get('csv_mb', 0):,.1f} MB. {res.get('message', '')}")
        if res.get("per_source"):
            st.caption("Orijen: " + " · ".join(f"{k}: {v['status']} ({v['rows']:,})" for k, v in res["per_source"].items()))
    if sched is not None:
        wait = sched.seconds_until_next()
        if wait is None:
            st.caption("🔁 Pedidu automátiku: dezativa.")
        elif manager.is_running(name):
            st.caption("🔁 Pedidu automátiku: sei hein pedidu ne'ebé hala'o remata.")
        else:
            st.caption(f"🔁 Pedidu automátiku tuir mai: iha {_fmt_seconds(wait)}.")
        last = sched.state.get("last_auto_message")
        if last:
            st.caption(f"Pedidu automátiku ikus ({sched.state.get('last_auto_at', '')}): {last}")
    _log_expander(snap)


_COLLECTION_FRAGS = (_make_fragment(_collection_monitor, 2), _make_fragment(_collection_monitor, 10))


def render_request_tab(config: dict, sources: list) -> None:
    dirs = workflow_dirs(config)
    manager = get_job_manager()
    bcfg = config.get("batch", {})
    sched = get_scheduler(manager, dirs["batches"], {
        "enabled": bool(bcfg.get("auto_enabled", False)),
        "interval_minutes": int(bcfg.get("auto_interval_minutes", 30)),
        "target_mb": float(bcfg.get("target_mb", DEFAULT_TARGET_MB)),
        "delay_seconds": float(bcfg.get("delay_seconds", 1.0)),
        "per_page": int(bcfg.get("per_page", 100)),
        "sources": [dict(s) for s in sources],
    })
    st.session_state["_scheduler"] = sched
    if sched.state.get("sources") != sources:
        sched.configure(sources=[dict(s) for s in sources])

    st.subheader("📥 Pedidu Dadus (koleta notísia)")
    st.caption(
        "Pedidu ida foti notísia husi orijen aktivu sira to'o ficheiru CSV to'o meta tamañu (padraun 200 MB). "
        "Kada pedidu rai iha folder rasik (lote) ho formatu **CSV, JSON no SQLite**, la hamutuk ho lote seluk "
        "no la repete notísia ne'ebé rai ona. Se orijen la iha notísia tuan tan, lote sei ki'ik liu meta.")

    running = manager.is_running(COLLECTION_JOB)
    c1, c2 = st.columns(2)
    target = c1.number_input("Meta tamañu CSV kada pedidu (MB)", min_value=1, max_value=1000,
                             value=int(sched.state.get("target_mb", DEFAULT_TARGET_MB)), key="req_target")
    delay = c2.number_input("Intervalu entre pájina (segundu)", min_value=0.0, max_value=10.0, step=0.5,
                            value=float(sched.state.get("delay_seconds", 1.0)), key="req_delay")
    if target != sched.state.get("target_mb") or delay != sched.state.get("delay_seconds"):
        sched.configure(target_mb=float(target), delay_seconds=float(delay))

    b1, b2 = st.columns(2)
    start = b1.button("▶️ Halo pedidu manuál agora", type="primary", key="req_manual", disabled=running)
    stop = b2.button("⏹️ Para pedidu", key="req_stop", disabled=not running)
    if start:
        ok, msg = start_collection_request(manager, dirs["batches"], sources, kind="manuál",
                                           target_mb=float(target), delay_seconds=float(delay),
                                           per_page=int(sched.state.get("per_page", 100)), auto_state=sched.state)
        if not ok:
            st.warning(msg)
    if stop:
        manager.stop(COLLECTION_JOB)

    st.markdown("#### 🔁 Pedidu automátiku")
    a1, a2 = st.columns(2)
    auto_on = a1.toggle("Ativa pedidu automátiku", value=bool(sched.state.get("enabled")), key="auto_on")
    interval = a2.number_input("Intervalu (minutu)", min_value=1, max_value=1440, key="auto_interval",
                               value=int(sched.state.get("interval_minutes", 30)))
    if auto_on != bool(sched.state.get("enabled")) or int(interval) != int(sched.state.get("interval_minutes", 30)):
        sched.configure(enabled=bool(auto_on), interval_minutes=int(interval))
    st.caption("Pedidu automátiku hahú depois pedidu anterior remata no liu intervalu ne'e. Hala'o hela maski tab "
               "browser taka, kuandu servisu painel (Streamlit) sei moris.")

    st.markdown("#### 📡 Progresu pedidu")
    _render_monitor(COLLECTION_JOB, _COLLECTION_FRAGS)

    st.markdown("#### 🗂️ Lote ne'ebé rai ona")
    batches = list_batches(dirs["batches"])
    try:
        free = shutil.disk_usage(dirs["batches"]).free
        st.caption(f"Fatin: `{dirs['batches']}` · uza {_mb(_dir_size(dirs['batches']))} · disku livre {_mb(free)}. "
                   "Lote ida uza besik 3 dala tamañu CSV (CSV + JSON + SQLite).")
    except OSError:
        pass
    if not batches:
        st.info("Seidauk iha lote. Klik **Halo pedidu manuál agora** ka hein pedidu automátiku.")
        return
    st.dataframe(pd.DataFrame([{
        "Lote": b["batch_id"], "Tipu": b.get("kind", ""), "Hahú": b.get("started_at", ""),
        "Notísia": b.get("rows", 0), "CSV (MB)": b.get("csv_mb", 0), "Meta (MB)": b.get("target_mb", 0),
        "Meta to'o": "Sim" if b.get("reached_target") else "Lae", "Estadu": b.get("status", ""),
    } for b in batches]), hide_index=True)


# ----------------------------------------------------------------------
# TAB 2 — Preprosesamentu
# ----------------------------------------------------------------------
def _prep_job(progress, stop_event, *, input_path, cleaned_dir):
    return run_staged_preprocessing(input_path, cleaned_dir, progress=progress, stop_event=stop_event)


def _prep_monitor(name: str, was_running: bool) -> None:
    _sync_variant(name, was_running)
    snap = get_job_manager().snapshot(name)
    if snap["status"] == RUNNING:
        extra = snap["extra"]
        done, total = extra.get("rows_done", 0), extra.get("rows_total", 0)
        st.progress(snap["fraction"], text=f"{snap['message']} · {done:,} / {total:,} liña")
        _stage_table(extra.get("stages", []), extra.get("current_stage"))
    elif snap["status"] in (DONE, STOPPED, FAILED):
        res = snap["result"] or {}
        _job_banner(snap, f"✅ Preprosesamentu remata: {res.get('rows_saved', 0):,} notísia bersih rai "
                          f"({_fmt_seconds(res.get('seconds'))}).")
        if snap["status"] == DONE and res.get("stages"):
            _stage_table(res["stages"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Liña lee", f"{res['rows_read']:,}")
            m2.metric("Notísia bersih", f"{res['rows_saved']:,}")
            m3.metric("Token média", f"{res['avg_tokens_final']}")
            st.caption("Ficheiru: " + " · ".join(f"{k.upper()} ({_mb(v)})" for k, v in res.get("sizes", {}).items()))
            st.caption(f"Fatin: `{res['directory']}`")
            if res.get("samples"):
                with st.expander("🔎 Ezemplu notísia ida iha etapa ida-idak"):
                    s = res["samples"]
                    for title, key in [("Tékstu orijinál (títulu + isi)", "raw"), ("Depois hamoos", "cleaning"),
                                       ("Depois case folding", "case_folding"), ("Depois tokenizasaun", "tokenization"),
                                       (f"Depois hasai stopword (lian: {s.get('language')})", "stopwords"),
                                       ("Depois stemming (rezultadu final)", "stemming")]:
                        st.markdown(f"**{title}**")
                        st.code(s.get(key, ""), language=None)
    _log_expander(snap)


_PREP_FRAGS = (_make_fragment(_prep_monitor, 2), _make_fragment(_prep_monitor, None))


def render_preprocessing_workflow(config: dict) -> None:
    dirs = workflow_dirs(config)
    manager = get_job_manager()
    st.subheader("🧪 Preprosesamentu Dadus")
    st.caption(
        "Hili dadus husi lote pedidu ka karga ficheiru rasik. Etapa hotu hala'o iha painel ne'e: importa → hamoos → "
        "case folding → tokenizasaun → hasai stopword → stemming → kategoria referénsia → rai dadus bersih. "
        "Prosesa ba lote ki'ik (5.000 notísia kada dala) atu memória la nakonu.")

    mode = st.radio("Fonte dadus", ["Lote husi pedidu", "Karga ficheiru husi komputadór"], horizontal=True,
                    key="prep_src")
    input_path, upload = None, None
    if mode == "Lote husi pedidu":
        batches = list_batches(dirs["batches"])
        if not batches:
            st.info("Seidauk iha lote. Halo pedidu dadus uluk iha tab **📥 1. Pedidu Dadus**.")
        else:
            i = st.selectbox("Hili lote", list(range(len(batches))), key="prep_batch",
                             format_func=lambda i: f"{batches[i]['batch_id']} · {batches[i]['rows']:,} notísia · "
                                                   f"{batches[i].get('csv_mb', 0):,.1f} MB")
            files = {k: os.path.join(batches[i]["directory"], os.path.basename(p))
                     for k, p in batches[i].get("files", {}).items()}
            files = {k: p for k, p in files.items() if os.path.isfile(p)}
            if files:
                fmt = st.radio("Formatu ficheiru", list(files), horizontal=True, key="prep_fmt",
                               format_func=lambda k: {"csv": "CSV", "json": "JSON", "sqlite": "SQLite"}.get(k, k))
                input_path = files[fmt]
            else:
                st.warning("Ficheiru lote ne'e la hetan iha disku.")
    else:
        upload = st.file_uploader("Hili ficheiru (.csv, .json, .db)", type=[e.lstrip(".") for e in SUPPORTED_INPUT],
                                  key="prep_upload")
        st.caption("Presiza kolona títulu (title) no konteúdu (content ka summary). Limite karga: tuir "
                   "`.streamlit/config.toml` (padraun 1 GB).")

    running = manager.is_running(PREP_JOB)
    go = st.button("▶️ Hala'o preprosesamentu hotu-hotu", type="primary", key="prep_go",
                   disabled=running or (input_path is None and upload is None))
    stop = st.button("⏹️ Para preprosesamentu", key="prep_stop", disabled=not running)
    if go:
        path = input_path if input_path else _save_upload(upload, dirs["uploads"])
        ok, msg = manager.start(PREP_JOB, _prep_job, kwargs=dict(input_path=path, cleaned_dir=dirs["cleaned"]),
                                group=HEAVY_GROUP)
        if not ok:
            st.warning(msg)
    if stop:
        manager.stop(PREP_JOB)

    _render_monitor(PREP_JOB, _PREP_FRAGS)

    st.markdown("#### 🗂️ Dadus bersih ne'ebé rai ona")
    sets = list_clean_datasets(dirs["cleaned"])
    if not sets:
        st.info("Seidauk iha dadus bersih.")
        return
    st.dataframe(pd.DataFrame([{
        "Dadus bersih": d["dataset_id"], "Notísia": d["rows_saved"], "Remata iha": d.get("finished_at", ""),
        "CSV (MB)": round(d.get("sizes", {}).get("csv", 0) / 1024 / 1024, 1),
        "Lian": ", ".join(f"{k}:{v}" for k, v in d.get("languages", {}).items()),
    } for d in sets]), hide_index=True)


# ----------------------------------------------------------------------
# TAB 3 — Treinu modelu
# ----------------------------------------------------------------------
def _train_job(progress, stop_event, *, path, models_dir, params):
    return train_from_clean_file(path, ModelRegistry(models_dir), progress=progress, stop_event=stop_event,
                                 **params)


def _train_monitor(name: str, was_running: bool) -> None:
    _sync_variant(name, was_running)
    snap = get_job_manager().snapshot(name)
    if snap["status"] == RUNNING:
        st.progress(snap["fraction"], text=(snap["message"] or "Hala'o...")[:180])
        _stage_table(snap["extra"].get("stages", []), snap["extra"].get("current_stage"))
    elif snap["status"] in (DONE, STOPPED, FAILED):
        res = snap["result"] or {}
        if snap["status"] == DONE and res.get("status") == "remata":
            st.success(f"✅ Treinu remata: modelu versaun {res['version_id']} ativu ona "
                       f"({_fmt_seconds(res.get('seconds'))}). Haree rezultadu iha okos.")
        else:
            _job_banner(snap, "")
    _log_expander(snap)


_TRAIN_FRAGS = (_make_fragment(_train_monitor, 2), _make_fragment(_train_monitor, None))


def _current_meta(registry: ModelRegistry):
    vid = registry.current_version_id()
    if not vid:
        return None
    try:
        with open(os.path.join(registry.versions_dir, vid, "meta.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def _load_cluster_frame(path: str, mtime: float) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns
    use = [c for c in ("published_at", "cluster", "category", "split") if c in head]
    return pd.read_csv(path, usecols=use)


def render_training_workflow(config: dict) -> None:
    dirs = workflow_dirs(config)
    manager = get_job_manager()
    registry = ModelRegistry(dirs["models"])
    bcfg = config.get("training", {})

    st.subheader("🧠 Treinu Modelu TF-IDF + K-Means")
    st.caption(
        "Hili dadus bersih (rezultadu preprosesamentu) ka karga ficheiru bersih. Treinu hala'o iha painel ne'e: "
        "vetorizasaun TF-IDF → klaster K-Means → identifikasaun tópiku → avaliasaun ho dadus teste.")

    mode = st.radio("Fonte dadus bersih", ["Dadus bersih ne'ebé rai ona", "Karga ficheiru bersih husi komputadór"],
                    horizontal=True, key="train_src")
    path, upload = None, None
    if mode == "Dadus bersih ne'ebé rai ona":
        sets = list_clean_datasets(dirs["cleaned"])
        if not sets:
            st.info("Seidauk iha dadus bersih. Hala'o preprosesamentu uluk iha tab **🧪 2. Preprosesamentu**.")
        else:
            i = st.selectbox("Hili dadus bersih", list(range(len(sets))), key="train_set",
                             format_func=lambda i: f"{sets[i]['dataset_id']} · {sets[i]['rows_saved']:,} notísia")
            files = {k: p for k, p in {
                "csv": os.path.join(sets[i]["directory"], "notisia_bersih.csv"),
                "json": os.path.join(sets[i]["directory"], "notisia_bersih.json"),
                "sqlite": os.path.join(sets[i]["directory"], "notisia_bersih.db")}.items() if os.path.isfile(p)}
            fmt = st.radio("Formatu ficheiru", list(files), horizontal=True, key="train_fmt",
                           format_func=lambda k: {"csv": "CSV", "json": "JSON", "sqlite": "SQLite"}[k]) if files else None
            path = files.get(fmt)
    else:
        upload = st.file_uploader("Hili ficheiru bersih (.csv, .json, .db)", type=["csv", "json", "db", "sqlite"],
                                  key="train_upload")
        st.caption("Ficheiru tenke iha kolona `text_final` (rezultadu tab Preprosesamentu).")

    with st.expander("⚙️ Parámetru treinu", expanded=False):
        p1, p2, p3 = st.columns(3)
        max_features = p1.number_input("Liafuan máximu TF-IDF", 500, 50000, int(bcfg.get("max_features", 5000)),
                                       step=500, key="tr_maxf")
        ngram = p2.selectbox("Par liafuan (n-gram)", [1, 2], index=int(bcfg.get("ngram_max", 2)) - 1, key="tr_ngram",
                             format_func=lambda n: "Liafuan ida (1)" if n == 1 else "Liafuan ida no par (1-2)")
        min_df = p3.number_input("Frekuénsia mínima (notísia)", 1, 50, int(bcfg.get("min_df", 2)), key="tr_mindf")
        test_pct = st.slider("Persentajen dadus teste", 5, 50, int(float(bcfg.get("test_fraction", 0.2)) * 100),
                             key="tr_test")
        k_mode = st.radio("Númeru klaster (k)", ["Otomátiku (silhouette)", "Hili rasik"], horizontal=True,
                          key="tr_kmode")
        if k_mode.startswith("Otomátiku"):
            k1, k2 = st.columns(2)
            min_k = k1.number_input("k mínimu", 2, 30, int(bcfg.get("min_k", 2)), key="tr_mink")
            max_k = k2.number_input("k máximu", 2, 40, int(bcfg.get("max_k", 12)), key="tr_maxk")
            fixed_k = 5
        else:
            fixed_k = st.number_input("Valór k", 2, 40, 5, key="tr_fixk")
            min_k, max_k = 2, 12
        exclude = st.checkbox("Hasai kategoria 'Jerál' husi avaliasaun", key="tr_excl",
                              help="Kategoria 'Jerál' mak kategoria sobra; bele halo akurasia boot liu.")

    running = manager.is_running(TRAIN_JOB)
    go = st.button("🧠 Hahú treinu TF-IDF + K-Means", type="primary", key="train_go",
                   disabled=running or (path is None and upload is None))
    stop = st.button("⏹️ Para treinu", key="train_stop", disabled=not running)
    if go:
        use_path = path if path else _save_upload(upload, dirs["uploads"])
        params = dict(max_features=int(max_features), ngram_range=(1, int(ngram)), min_df=int(min_df),
                      test_fraction=test_pct / 100, k_mode="auto" if k_mode.startswith("Otomátiku") else "fixed",
                      fixed_k=int(fixed_k), min_k=int(min_k), max_k=int(max_k), exclude_generic=bool(exclude))
        ok, msg = manager.start(TRAIN_JOB, _train_job, kwargs=dict(path=use_path, models_dir=dirs["models"],
                                                                   params=params), group=HEAVY_GROUP)
        if not ok:
            st.warning(msg)
    if stop:
        manager.stop(TRAIN_JOB)

    _render_monitor(TRAIN_JOB, _TRAIN_FRAGS)
    st.divider()
    render_training_results(registry)


# ----------------------------------------------------------------------
# Rezultadu treinu: métrika, vizualizasaun, tópiku populár, lista liafuan
# ----------------------------------------------------------------------
def render_training_results(registry: ModelRegistry) -> None:
    import plotly.express as px

    meta = _current_meta(registry)
    st.subheader("📊 Rezultadu Treinu")
    if not meta:
        st.info("Seidauk iha modelu treinadu. Hala'o treinu iha leten.")
        return
    st.caption(f"Modelu ativu: versaun {meta.get('version_id')} · treinu iha {meta.get('trained_at')} · "
               f"{meta.get('n_docs_used', '?'):,} notísia · {meta.get('best_k')} klaster.")
    ev = meta.get("evaluation")
    if not ev:
        st.info("Modelu ne'e treina ho painel antigu, la iha avaliasaun. Hala'o treinu foun atu hetan akurasia, recall no F1-score.")
        return

    st.markdown("#### 🎯 Avaliasaun ho dadus teste")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Akurasia", f"{ev['accuracy']:.1%}")
    m2.metric("Presizaun (makro)", f"{ev['precision_macro']:.1%}")
    m3.metric("Recall (makro)", f"{ev['recall_macro']:.1%}")
    m4.metric("F1-score (makro)", f"{ev['f1_macro']:.1%}")
    sil = meta.get("silhouette")
    st.caption(
        f"Dadus treinu {meta.get('n_train', 0):,} · dadus teste {ev['n_test']:,} · "
        f"akurasia baze (prediz hotu '{ev['baseline_label']}'): {ev['baseline_accuracy']:.1%}"
        + (f" · silhouette: {sil:.3f}" if sil is not None else "") +
        f" · média ponderadu: presizaun {ev['precision_weighted']:.1%}, recall {ev['recall_weighted']:.1%}, "
        f"F1 {ev['f1_weighted']:.1%}.")
    st.info("ℹ️ K-Means la iha label. Kada klaster simu kategoria dominante husi dadus treinu, depois kompara ho "
            "kategoria referénsia (klasifikadór liafuan-xave) iha dadus teste. Entaun métrika sukat konkordánsia "
            "ho kategoria referénsia, la'ós lorloos absolutu. Kompara akurasia ho akurasia baze.")

    left, right = st.columns(2)
    per = pd.DataFrame(ev["per_class"])
    with left:
        melt = per.melt(id_vars=["kategoria"], value_vars=["presizaun", "recall", "f1"], var_name="métrika",
                        value_name="valór")
        st.plotly_chart(px.bar(melt, x="kategoria", y="valór", color="métrika", barmode="group",
                               title="Presizaun, recall no F1 kada kategoria"))
    with right:
        cm = ev["confusion"]
        st.plotly_chart(px.imshow(cm, x=ev["classes"], y=ev["classes"], text_auto=True, aspect="auto",
                                  labels=dict(x="Kategoria prediz", y="Kategoria referénsia", color="Notísia"),
                                  title="Matriz konfuzaun (dadus teste)"))
    st.dataframe(per.rename(columns={"suporta": "notísia teste"}), hide_index=True)
    st.caption("Mapa klaster → kategoria: " + ", ".join(f"klaster {k} → {v}" for k, v in ev["mapping"].items()))

    ks = meta.get("k_search") or []
    if ks:
        kdf = pd.DataFrame(ks)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(kdf, x="k", y="silhouette", markers=True, title="Silhouette kada valór k"))
        c2.plotly_chart(px.line(kdf, x="k", y="inertia", markers=True, title="Inertia kada valór k (métodu siku)"))

    st.markdown("#### 🧩 Tópiku ne'ebé identifika")
    themes = meta.get("themes") or []
    if themes:
        tdf = pd.DataFrame([{"Klaster": t["cluster_id"], "Tópiku": t["label"], "Notísia": t["news_count"],
                             "Persentajen": t["share_pct"], "Liafuan-xave": ", ".join(t["top_terms"])} for t in themes])
        st.dataframe(tdf, hide_index=True)
        st.plotly_chart(px.bar(tdf, x="Tópiku", y="Notísia", color="Tópiku", title="Notísia kada tópiku"))
    labels = {t["cluster_id"]: t["label"] for t in themes}

    st.markdown("#### 🔥 Tópiku populár")
    cpath = os.path.join(registry.versions_dir, str(meta.get("version_id")), "news_clustered.csv")
    if os.path.isfile(cpath):
        cdf = _load_cluster_frame(cpath, os.path.getmtime(cpath))
        win = st.radio("Janela tempu", [3, 7, 14, 30], index=1, horizontal=True, key="trend_win",
                       format_func=lambda d: f"{d} loron ikus")
        tr = trending_topics(cdf, labels, window_days=win)
        if tr.empty:
            st.info("Seidauk iha dadus data sufisiente atu hatudu tópiku populár.")
        else:
            st.dataframe(tr.rename(columns={
                "klaster": "Klaster", "topiku": "Tópiku", "notisia_foun": "Notísia foun",
                "notisia_molok": "Notísia molok", "aumentu_pct": "Aumentu (%)", "pontuasaun": "Pontuasaun"}),
                hide_index=True)
            st.plotly_chart(px.bar(tr.head(8), x="topiku", y="notisia_foun", color="topiku",
                                   labels={"topiku": "Tópiku", "notisia_foun": "Notísia foun"},
                                   title=f"Tópiku populár iha {win} loron ikus (to'o data notísia foun liu)"))
            st.caption("Pontuasaun = notísia foun + aumentu kompara ho janela molok. Referénsia tempu mak data "
                       "notísia foun liu iha dadus.")
        freq = st.radio("Tendénsia tuir", ["D", "W", "M"], index=1, horizontal=True, key="trend_freq",
                        format_func=lambda f: {"D": "Loron", "W": "Semana", "M": "Fulan"}[f])
        ts = topic_time_series(cdf, labels, freq)
        if not ts.empty:
            st.plotly_chart(px.line(ts, x="periodu", y="kuantidade", color="topiku",
                                    labels={"periodu": "Períodu", "kuantidade": "Notísia", "topiku": "Tópiku"},
                                    title="Tendénsia tópiku iha tempu"))
    else:
        st.info("Ficheiru klaster la hetan iha disku.")

    st.markdown("#### 🔤 Lista liafuan")
    wl = meta.get("wordlist") or {}
    glob = pd.DataFrame(wl.get("global", []))
    if glob.empty:
        st.info("Seidauk iha lista liafuan.")
        return
    n_words = st.slider("Kuantidade liafuan atu hatudu", 10, min(100, len(glob)), min(30, len(glob)), key="wl_n")
    top = glob.head(n_words)
    st.plotly_chart(px.bar(top.iloc[::-1], x="skor", y="liafuan", orientation="h",
                           labels={"skor": "Pontuasaun TF-IDF", "liafuan": "Liafuan"},
                           title="Liafuan prinsipál iha dadus tomak"))
    st.dataframe(top.rename(columns={"liafuan": "Liafuan", "skor": "Pontuasaun TF-IDF", "dokumentu": "Notísia"}),
                 hide_index=True)
    pt = wl.get("per_topic", {})
    if pt:
        cid = st.selectbox("Lista liafuan kada tópiku", list(pt), key="wl_topic",
                           format_func=lambda c: f"Klaster {c} — {labels.get(int(c), '')}")
        tdf = pd.DataFrame(pt[cid]).head(n_words)
        st.plotly_chart(px.bar(tdf.iloc[::-1], x="pezu", y="liafuan", orientation="h",
                               labels={"pezu": "Pezu iha sentroide", "liafuan": "Liafuan"},
                               title=f"Liafuan prinsipál klaster {cid}"))
    rows = [{"tópiku": "hotu-hotu", **r} for r in wl["global"]]
    for c, items in pt.items():
        rows += [{"tópiku": labels.get(int(c), f"Klaster {c}"), "liafuan": r["liafuan"], "skor": r["pezu"]}
                 for r in items]
    st.download_button("⬇️ Foti lista liafuan (CSV)", data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                       file_name="lista_liafuan.csv", mime="text/csv", key="wl_download")
