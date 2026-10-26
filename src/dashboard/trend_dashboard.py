"""
Painel Streamlit — Analiza Notísia Online no Deteksaun Topiku.

Halao ho:
    streamlit run src/dashboard/trend_dashboard.py

Tab sira:
1. Pedidu Dadus: pedidu manuál no automátiku (kada 30 minutu) to'o lote CSV 200 MB (CSV, JSON, SQLite).
2. Preprosesamentu: importa, hamoos, case folding, tokenizasaun, hasai stopword, stemming.
3. Treinu Modelu: TF-IDF + K-Means, akurasia, recall, F1-score, tópiku populár, lista liafuan.
4. Analiza Notísia: tópiku, tendénsia, kalohan liafuan, galeria, istória.
"""
import os
import sys
import json
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

from wordcloud import WordCloud

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_config, setup_logging  # noqa: E402
from src.dashboard.data_loader import read_cleaned_with_fallback  # noqa: E402
from src.dashboard.workflow_views import (  # noqa: E402
    render_preprocessing_workflow, render_request_tab, render_training_workflow,
)
from src.collection.batch_collector import list_batches  # noqa: E402
from src.paths import workflow_dirs  # noqa: E402
from src.training.model_registry import ModelRegistry  # noqa: E402
from src.dashboard.preprocessing_view import PREP_MSG, render_preprocessing  # noqa: E402
from src.dashboard.history_views import render_history, export_csv_bytes  # noqa: E402
from src.dashboard.import_view import (  # noqa: E402
    MSG as IMPORT_MSG, load_saved_model_state, render_candidate_themes, render_import_panel,
)

LANG_LABELS = {"id": "Indonézia", "en": "Inglés", "tet": "Tetun", "unknown": "Seluk"}
HISTORY_START = date(2016, 1, 1)  # filtru "Husi 2016 to agora"
DILI_OFFSET = timedelta(hours=9)  # published_at rai iha UTC; Timor-Leste = UTC+9


def _to_dili(series: pd.Series) -> pd.Series:
    """published_at (UTC) -> oras lokál Dili, atu loron notísia sai loos."""
    return pd.to_datetime(series, errors="coerce") + DILI_OFFSET
LAST_FETCH_FILE = os.path.join(PROJECT_ROOT, "logs", "last_fetch.txt")

st.set_page_config(
    page_title="Analiza Notísia Online no Deteksaun Topiku",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Estilu
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 1300px; }

    .tatoli-header {
        background: linear-gradient(135deg, #b3151a 0%, #7a0f13 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .tatoli-header h1 { margin: 0; font-size: 1.7rem; }
    .tatoli-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.18);
        font-size: 0.8rem;
        margin-top: 0.6rem;
    }

    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-left: 4px solid #b3151a;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    .news-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }
    .news-card h4 { margin: 0 0 0.3rem 0; font-size: 1.0rem; }
    .news-card .meta { font-size: 0.8rem; opacity: 0.7; margin-bottom: 0.3rem; }

    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
</style>
"""


# ----------------------------------------------------------------------
# Ajuda: timestamp "kolete ikus" ne'ebé persiste (uza ba auto-refresh)
# ----------------------------------------------------------------------
def _read_last_fetch_time(config=None):
    """Tempu pedidu ikus: husi lote ikus; se la iha, husi ficheiru antigu."""
    if config is not None:
        batches = list_batches(workflow_dirs(config)["batches"])
        if batches:
            try:
                return datetime.fromisoformat(batches[0]["finished_at"])
            except (KeyError, ValueError):
                pass
    if os.path.exists(LAST_FETCH_FILE):
        try:
            with open(LAST_FETCH_FILE, "r") as f:
                return datetime.fromisoformat(f.read().strip())
        except Exception:
            return None
    return None


def _write_last_fetch_time(ts: datetime):
    os.makedirs(os.path.dirname(LAST_FETCH_FILE), exist_ok=True)
    with open(LAST_FETCH_FILE, "w") as f:
        f.write(ts.isoformat())


def _format_elapsed(ts: datetime) -> str:
    if ts is None:
        return "seidauk iha"
    delta = datetime.now() - ts
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "foin daudaun"
    if minutes < 60:
        return f"{minutes} minutu liu ba"
    hours = minutes // 60
    return f"{hours} oras liu ba"


@st.cache_resource
def get_config():
    config = load_config()
    setup_logging(config["paths"]["logs"])
    return config


@st.cache_data(show_spinner="Lee dadus...")
def _read_cleaned(path: str, mtime: float) -> pd.DataFrame:
    """Cache: lee CSV boot dala ida de'it; refresh automátiku bainhira ficheiru muda (mtime)."""
    return read_cleaned_with_fallback(path)


def load_latest_from_disk(config):
    cleaned_master = os.path.join(config["paths"]["cleaned"], "tatoli_cleaned_master.csv")
    keywords_latest = os.path.join(config["paths"]["keywords"], "tatoli_keywords_latest.json")

    saved = load_saved_model_state(config)  # modelu ne'ebé rai ona (restart-safe)
    master_exists = os.path.exists(cleaned_master)
    if saved is not None:
        saved_ts = datetime.fromisoformat(saved["model_meta"]["trained_at"]).timestamp()
        if not master_exists or saved_ts >= os.path.getmtime(cleaned_master):
            return saved
    if not master_exists:
        return None

    try:
        df = _read_cleaned(cleaned_master, os.path.getmtime(cleaned_master))  # cache_data sudah mengembalikan salinan
    except MemoryError:
        st.session_state["load_error"] = (
            "❌ Memória la to'o atu lee ficheiru dadus (`tatoli_cleaned_master.csv`). "
            "Arkiva ka hamenus ficheiru ne'e, depois loke fali dashboard."
        )
        return None
    st.session_state.pop("load_error", None)
    topics_payload = None
    if os.path.exists(keywords_latest):
        with open(keywords_latest, "r", encoding="utf-8") as f:
            topics_payload = json.load(f)

    result = {"df": df, "topics_payload": topics_payload}
    if df.attrs.get("rows_total"):
        result["memory_notice"] = {"kept": len(df), "total": int(df.attrs["rows_total"])}
    return result


# ----------------------------------------------------------------------
# Hatudu sesaun sira
# ----------------------------------------------------------------------
def render_header(last_fetch):
    st.markdown(
        f"""
        <div class="tatoli-header">
            <h1>📰 Analiza Notísia Online no Deteksaun Topiku</h1>
            <p>Análize tópiku notísia automátika husi RSS &amp; pájina artigu Timor-Leste
            (Tatoli, Timor Post, Timor News, GMN TV, no seluk tan — tuir orijen ne'ebé ativa iha sidebar)
            — koleta, limpeza tékstu, deteksaun tópiku TF-IDF + K-Means, no tendénsia interativu.</p>
            <span class="status-pill">🕓 Dadus atualiza ikus: {_format_elapsed(last_fetch)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stats(df: pd.DataFrame):
    total_news = len(df)
    total_topics = df["cluster"].nunique() if "cluster" in df.columns else 0
    langs = df["language"].value_counts().to_dict() if "language" in df.columns else {}
    n_categories = df["category"].nunique() if "category" in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Total Notísia", f"{total_news:,}")
    c2.metric("🧩 Kuantidade Tópiku", total_topics)
    c3.metric("🗂️ Kategoria Deteta", n_categories if n_categories else "-")
    c4.metric("🌐 Lian Deteta", ", ".join(sorted(langs.keys())) if langs else "-")


def render_overview(df: pd.DataFrame, labels: dict):
    st.subheader("🔥 Destake Ohin Loron")
    dist = df["cluster"].value_counts().reset_index()
    dist.columns = ["cluster", "kuantidade"]
    dist["topik"] = dist["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Klaster {c}")))
    top3 = dist.sort_values("kuantidade", ascending=False).head(3)

    cols = st.columns(len(top3)) if len(top3) else [st]
    for col, (_, row) in zip(cols, top3.iterrows()):
        with col:
            st.markdown(
                f"""<div class="news-card">
                    <div class="meta">Tópiku populár</div>
                    <h4>🏷️ {row['topik']}</h4>
                    <div class="meta">{int(row['kuantidade'])} notísia</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.subheader("🆕 Notísia Foun")
    recent = df.copy()
    if "published_at" in recent.columns:
        recent["published_at"] = pd.to_datetime(recent["published_at"], errors="coerce")
        recent = recent.sort_values("published_at", ascending=False)
    recent = recent.head(5)

    for _, row in recent.iterrows():
        topik = labels.get(row.get("cluster"), labels.get(str(row.get("cluster")), ""))
        pub = row.get("published_at")
        pub_str = pub.strftime("%d %b %Y, %H:%M") if pd.notna(pub) else "-"
        link = row.get("link", "")
        title = row.get("title", "(sem títulu)")
        title_html = f'<a href="{link}" target="_blank" style="text-decoration:none;">{title}</a>' if link else title
        st.markdown(
            f"""<div class="news-card">
                <div class="meta">{pub_str} • {row.get('category', '') or 'Jerál'} • {topik}</div>
                <h4>{title_html}</h4>
            </div>""",
            unsafe_allow_html=True,
        )


def render_topic_distribution(df: pd.DataFrame, labels: dict):
    st.subheader("📊 Distribuisaun Notísia kada Klaster")
    dist = df["cluster"].value_counts().reset_index()
    dist.columns = ["cluster", "kuantidade_notisia"]
    dist["topik"] = dist["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Klaster {c}")))
    dist = dist.sort_values("kuantidade_notisia", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        fig_bar = px.bar(dist, x="topik", y="kuantidade_notisia", color="topik",
                          title="Kuantidade Notísia kada Tópiku")
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        fig_pie = px.pie(dist, names="topik", values="kuantidade_notisia",
                          title="Proporsaun Tópiku", hole=0.35)
        st.plotly_chart(fig_pie, use_container_width=True)


def render_topic_trend(df: pd.DataFrame, labels: dict):
    st.subheader("📈 Tendénsia Tópiku iha Tempu")
    tdf = df.copy()
    tdf["published_at"] = pd.to_datetime(tdf["published_at"], errors="coerce")
    tdf = tdf.dropna(subset=["published_at"])
    if tdf.empty:
        st.info("Seidauk iha dadus data sufisiente atu hatudu tendénsia. Husik pedidu automátiku halao "
                 "dala balu (ka halao koleta manual iha loron seluk) atu tendénsia forma.")
        return

    tdf["topik"] = tdf["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Klaster {c}")))

    span = (tdf["published_at"].max() - tdf["published_at"].min()).days
    auto = "Fulan" if span > 400 else "Semana" if span > 60 else "Loron"
    choice = st.radio("Agrupa tuir:", ["Otomátiku", "Loron", "Semana", "Fulan", "Tinan"],
                      horizontal=True, key="trend_gran")
    gran = auto if choice == "Otomátiku" else choice
    freq = {"Loron": "D", "Semana": "W", "Fulan": "MS", "Tinan": "YS"}[gran]
    tdf["date"] = tdf["published_at"].dt.to_period({"D": "D", "W": "W", "MS": "M", "YS": "Y"}[freq]).dt.start_time

    trend = tdf.groupby(["date", "topik"]).size().reset_index(name="kuantidade")
    fig = px.line(trend, x="date", y="kuantidade", color="topik", markers=(gran in ("Tinan", "Fulan") and span < 1500),
                  title=f"Tendénsia Kuantidade Notísia kada Tópiku ({gran.lower()})")
    st.plotly_chart(fig, use_container_width=True)


def _cluster_word_freq(texts: pd.Series, top_n: int = 300) -> dict:
    """
    Hitung frekuensi kata per berita (hemat memori) dan ambil top_n kata.
    Menggantikan " ".join(semua_teks) yang bisa menyebabkan MemoryError pada
    cluster dengan puluhan ribu berita.
    """
    counter = Counter()
    for t in texts.dropna():
        counter.update(w for w in t.split() if len(w) > 1)
    return dict(counter.most_common(top_n))


def render_wordclouds(df: pd.DataFrame, labels: dict):
    st.subheader("☁️ Kalohan Liafuan kada Klaster")
    clusters = sorted(df["cluster"].dropna().unique())
    if not clusters:
        st.info("Seidauk iha klaster atu hatudu.")
        return

    cols = st.columns(min(3, len(clusters)))
    for i, cid in enumerate(clusters):
        freqs = _cluster_word_freq(df.loc[df["cluster"] == cid, "text_final"])
        if not freqs:
            continue
        # generate_from_frequencies: tidak ada string raksasa / regex di atas seluruh teks
        wc = WordCloud(width=500, height=350, background_color="white",
                       colormap="Reds", max_words=150).generate_from_frequencies(freqs)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        label = labels.get(cid, labels.get(str(cid), f"Klaster {cid}"))
        ax.set_title(f"Klaster {cid} — {label}", fontsize=10)

        with cols[i % len(cols)]:
            st.pyplot(fig)
            plt.close(fig)


def render_keywords_list(topics_payload):
    st.subheader("🔑 Liafuan Xave Prinsipál kada Tópiku")
    if not topics_payload:
        st.info("Dadus liafuan xave seidauk disponivel.")
        return
    for t in topics_payload.get("topics", []):
        with st.expander(f"Klaster {t['cluster_id']} — {t['label']} ({t.get('news_count', 0)} notísia)"):
            st.write(", ".join(t["top_terms"]))


def _txt(value) -> str:
    """Valór CSV ne'ebé mamuk (NaN) -> "" (la'ós "nan")."""
    return "" if value is None or pd.isna(value) else str(value).strip()


def render_news_table(df: pd.DataFrame, labels: dict, search_query: str):
    st.subheader("📰 Lista Notísia")
    show_df = df.copy()
    show_df["topik"] = show_df["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Klaster {c}")))

    if search_query:
        q = search_query.lower()
        clean_col = show_df["text_clean"].fillna("") if "text_clean" in show_df.columns else ""
        text_pool = show_df["title"].fillna("") + " " + clean_col
        show_df = show_df[text_pool.str.lower().str.contains(q)]

    dili_time = _to_dili(show_df["published_at"]) if "published_at" in show_df.columns else None
    sort_option = st.radio("Organiza:", ["Foun liu", "Tuan liu", "Relevante Liu (tópiku)"], horizontal=True)
    if dili_time is not None and sort_option in ("Foun liu", "Tuan liu"):
        show_df = show_df.assign(_t=dili_time).sort_values("_t", ascending=(sort_option == "Tuan liu"))
        dili_time = show_df.pop("_t")
    else:
        show_df = show_df.sort_values("cluster")
        dili_time = _to_dili(show_df["published_at"]) if "published_at" in show_df.columns else None

    if dili_time is not None:
        show_df["tempu (Dili)"] = dili_time.dt.strftime("%d/%m/%Y %H:%M")

    cols = ["title", "topik", "category", "source", "language", "tempu (Dili)", "link"]
    cols = [c for c in cols if c in show_df.columns]
    shown = show_df.head(300)
    st.caption(f"Hatudu {len(shown)} husi {len(show_df)} notísia.")
    st.dataframe(
        shown[cols], use_container_width=True, height=420,
        column_config={"link": st.column_config.LinkColumn("link")} if "link" in cols else None,
    )

    st.download_button(
        "⬇️ Foti lista ne'e (CSV, loke iha Excel)",
        data=export_csv_bytes(show_df, labels), file_name="lista_notisia.csv",
        mime="text/csv", key="news_download",
    )

    # Lee notísia ida (detallu)
    if not shown.empty:
        st.markdown("##### 📖 Lee notísia")
        idx = st.selectbox(
            "Hili notísia", options=list(range(len(shown))),
            format_func=lambda i: f"{shown.iloc[i].get('tempu (Dili)', '')} · {str(shown.iloc[i].get('title', ''))[:90]}",
            key="news_detail_pick",
        )
        row = shown.iloc[idx]
        st.markdown(f"**{_txt(row.get('title'))}**")
        st.caption(" · ".join(x for x in [_txt(row.get("source")), _txt(row.get("tempu (Dili)")),
                                          _txt(row.get("category")), f"Tópiku: {_txt(row.get('topik'))}"] if x))
        img = _txt(row.get("image_url"))
        if img.startswith("http"):
            try:
                st.image(img, width=360)
            except Exception:
                st.caption("(imajen la bele loke)")
        body = _txt(row.get("content")) or _txt(row.get("summary"))
        st.write(body[:4000] + (" ..." if len(body) > 4000 else ""))
        if _txt(row.get("link")):
            st.markdown(f"[🔗 Loke notísia orijinál]({_txt(row.get('link'))})")


def render_gallery(df: pd.DataFrame, labels: dict, n: int = 12):
    st.subheader("🖼️ Galeria Notísia Foun")
    if "image_url" not in df.columns:
        st.info("Seidauk iha dadus imajen. Ativa modu scraping kompletu (`scraping.full_content: true` "
                 "iha config/settings.yaml) atu halibur thumbnail artigu.")
        return

    gdf = df.copy()
    gdf["image_url"] = gdf["image_url"].fillna("")
    gdf = gdf[gdf["image_url"].str.strip() != ""]
    if "published_at" in gdf.columns:
        gdf["published_at"] = pd.to_datetime(gdf["published_at"], errors="coerce")
        gdf = gdf.sort_values("published_at", ascending=False)

    gdf = gdf.head(n)
    if gdf.empty:
        st.info("Seidauk iha notísia ho imajen thumbnail iha dadus agora.")
        return

    cols = st.columns(4)
    for i, (_, row) in enumerate(gdf.iterrows()):
        with cols[i % 4]:
            try:
                st.image(row["image_url"], use_container_width=True)
            except Exception:
                st.caption("(imajen labele karga)")
            topik = labels.get(row.get("cluster"), labels.get(str(row.get("cluster")), ""))
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row.get('category', '')} • {topik}")
            if row.get("link"):
                st.markdown(f"[Lee kompletu]({row['link']})")


def render_language_filter(df: pd.DataFrame):
    langs_present = [l for l in df["language"].dropna().unique() if l in LANG_LABELS]
    options = ["Hotu-hotu"] + [LANG_LABELS.get(l, l) for l in langs_present]
    choice = st.selectbox("🌐 Lian", options)
    if choice == "Hotu-hotu":
        return df
    inv_map = {v: k for k, v in LANG_LABELS.items()}
    return df[df["language"] == inv_map.get(choice, choice)]


def render_category_filter(df: pd.DataFrame):
    if "category" not in df.columns:
        return df
    categories_present = sorted(
        c for c in df["category"].fillna("").unique() if isinstance(c, str) and c.strip()
    )
    if not categories_present:
        return df
    options = ["Hotu-hotu"] + categories_present
    choice = st.selectbox("🗂️ Kategoria", options)
    if choice == "Hotu-hotu":
        return df
    return df[df["category"] == choice]


def render_date_filter(df: pd.DataFrame):
    """Filtru tempu: hotu / 2016 to agora / loron ida / 7-30 loron ikus / intervalu."""
    if "published_at" not in df.columns:
        return df

    dcol = _to_dili(df["published_at"])  # oras Dili
    valid_dates = dcol.dropna()
    if valid_dates.empty:
        st.selectbox("🗓️ Tempu", ["Tempu hotu-hotu"], disabled=True)
        return df

    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    lo = min(min_date, HISTORY_START)
    hi = max(max_date, date.today())

    preset = st.selectbox(
        "🗓️ Tempu",
        ["Tempu hotu-hotu", "Loron ida (hili data)", "Intervalu kustomizadu",
         "Husi 2016 to agora", "7 loron ikus", "30 loron ikus"],
    )

    if preset == "Tempu hotu-hotu":
        return df

    # Dadus ne'ebé koleta de'it bele filtra. Avizu se dadus seidauk to'o 2016.
    if min_date > HISTORY_START and preset in ("Husi 2016 to agora", "Intervalu kustomizadu",
                                               "Loron ida (hili data)"):
        st.caption(
            f"⚠️ Notísia ne'ebé rai hahú de'it husi {min_date:%d/%m/%Y}. Atu foti husi 2016, "
            "klik **📚 Kolete istória** iha sidebar."
        )

    if preset == "Loron ida (hili data)":
        day = st.date_input("Hili loron", value=max_date, min_value=lo, max_value=hi,
                            key="filter_single_day")
        start = end = day
    elif preset == "Husi 2016 to agora":
        start, end = HISTORY_START, hi
    elif preset == "7 loron ikus":
        start, end = max_date - timedelta(days=7), max_date
    elif preset == "30 loron ikus":
        start, end = max_date - timedelta(days=30), max_date
    else:
        picked = st.date_input(
            "Hili intervalu data (data hahú no data remata)",
            value=(lo, max_date), min_value=lo, max_value=hi, key="filter_range",
        )
        if isinstance(picked, (list, tuple)):
            if len(picked) == 2:
                start, end = picked
            elif len(picked) == 1:  # hili data ida de'it = loron ne'e de'it
                start = end = picked[0]
            else:
                return df
        else:
            start = end = picked

    mask = (dcol.dt.date >= start) & (dcol.dt.date <= end)
    result = df[mask]

    label = f"{start:%d/%m/%Y}" if start == end else f"{start:%d/%m/%Y} to {end:%d/%m/%Y}"
    if result.empty:
        st.warning(
            f"La iha notísia iha {label}. Dadus ne'ebé iha: {min_date:%d/%m/%Y} to {max_date:%d/%m/%Y}."
        )
    else:
        st.caption(f"📅 {label}: **{len(result)}** notísia.")
    return result


# ----------------------------------------------------------------------
# Trending: tópiku/notísia ne'ebé sa'e ohin loron (aumenta volume foun)
# ----------------------------------------------------------------------
def compute_trending_topics(df: pd.DataFrame, labels: dict, window_days: int = 3, min_articles: int = 2) -> pd.DataFrame:
    """
    Skór trending simples ba kada tópiku: kombinasaun entre VOLUME notísia foun
    (iha loron `window_days` ikus) no RASIU foun (kuantu boot
    parte notísia husi tópiku ne'e ne'ebé mosu foun liu kompara ho istória nia).
    Tópiku ho notísia foun barak NO parte foun aas = trending liu.
    """
    tdf = df.copy()
    if "published_at" not in tdf.columns or "cluster" not in tdf.columns:
        return pd.DataFrame()
    tdf["published_at"] = pd.to_datetime(tdf["published_at"], errors="coerce")
    tdf = tdf.dropna(subset=["published_at"])
    if tdf.empty:
        return pd.DataFrame()

    max_date = tdf["published_at"].max()
    window_start = max_date - pd.Timedelta(days=window_days)

    recent = tdf[tdf["published_at"] >= window_start]
    recent_counts = recent["cluster"].value_counts()
    total_counts = tdf["cluster"].value_counts()

    rows = []
    for cid, recent_count in recent_counts.items():
        if recent_count < min_articles:
            continue
        total_count = total_counts.get(cid, recent_count)
        recency_ratio = recent_count / total_count if total_count else 0
        rows.append({
            "cluster": cid,
            "topik": labels.get(cid, labels.get(str(cid), f"Klaster {cid}")),
            "recent_count": int(recent_count),
            "total_count": int(total_count),
            "recency_ratio": recency_ratio,
        })

    trending_df = pd.DataFrame(rows)
    if trending_df.empty:
        return trending_df

    trending_df["trend_score"] = trending_df["recent_count"] * (0.5 + trending_df["recency_ratio"])
    return trending_df.sort_values("trend_score", ascending=False).reset_index(drop=True)


# Kategoria prinsipál ne'ebé monitor ba tendénsia (bele muda/aumenta tuir presiza)
TRACKED_CATEGORIES = ["Edukasaun", "Politika", "Saúde", "Ekonomia"]


def compute_trending_categories(
    df: pd.DataFrame,
    window_days: int = 3,
    min_articles: int = 1,
    categories: list | None = None,
) -> pd.DataFrame:
    """
    Skór trending ba kada KATEGORIA (Edukasaun, Politika, Saúde, Ekonomia, etc.):
    kombinasaun entre volume notísia foun iha `window_days` ikus no rasiu foun
    kompara ho total históriku husi kategoria ne'e. Mesmu lójika ho
    `compute_trending_topics`, maibé baze ba kolona `category` la'ós `cluster`.
    """
    if "published_at" not in df.columns or "category" not in df.columns:
        return pd.DataFrame()

    cdf = df.copy()
    cdf["published_at"] = pd.to_datetime(cdf["published_at"], errors="coerce")
    cdf = cdf.dropna(subset=["published_at"])
    cdf["category"] = cdf["category"].fillna("").astype(str).str.strip()
    cdf = cdf[cdf["category"] != ""]

    if categories:
        # Normaliza atu kompara sem preokupa ho maiúskula/minúskula
        wanted = {c.lower(): c for c in categories}
        cdf = cdf[cdf["category"].str.lower().isin(wanted.keys())]
        cdf["category"] = cdf["category"].str.lower().map(wanted)

    if cdf.empty:
        return pd.DataFrame()

    max_date = cdf["published_at"].max()
    window_start = max_date - pd.Timedelta(days=window_days)

    recent = cdf[cdf["published_at"] >= window_start]
    recent_counts = recent["category"].value_counts()
    total_counts = cdf["category"].value_counts()

    rows = []
    base_list = categories if categories else total_counts.index.tolist()
    for cat in base_list:
        recent_count = int(recent_counts.get(cat, 0))
        total_count = int(total_counts.get(cat, 0))
        if recent_count < min_articles:
            continue
        recency_ratio = recent_count / total_count if total_count else 0
        rows.append({
            "category": cat,
            "recent_count": recent_count,
            "total_count": total_count,
            "recency_ratio": recency_ratio,
        })

    trending_df = pd.DataFrame(rows)
    if trending_df.empty:
        return trending_df

    trending_df["trend_score"] = trending_df["recent_count"] * (0.5 + trending_df["recency_ratio"])
    return trending_df.sort_values("trend_score", ascending=False).reset_index(drop=True)


def render_category_trending(df: pd.DataFrame, window_days: int):
    st.subheader("🗂️ Kategoria Aumenta (Populár)")
    st.caption(
        "Monitor kategoria: " + ", ".join(TRACKED_CATEGORIES) +
        " — bazeia ba kuantidade notísia foun no rasiu aumentu kompara ho históriku."
    )

    cat_trending = compute_trending_categories(df, window_days=window_days, categories=TRACKED_CATEGORIES)
    if cat_trending.empty:
        st.info(
            "Seidauk iha dadus sufisiente ba kategoria sira ne'e (" + ", ".join(TRACKED_CATEGORIES) +
            ") iha periudu ne'e. Hein koleta dadus liu ka hili janela tempu boot liu."
        )
        return

    cols = st.columns(len(cat_trending))
    for col, (_, row) in zip(cols, cat_trending.iterrows()):
        pct = row["recency_ratio"] * 100
        with col:
            st.metric(
                label=f"🗂️ {row['category']}",
                value=f"{int(row['recent_count'])} notísia foun",
                delta=f"{pct:.0f}% husi total",
            )

    fig_cat = px.bar(
        cat_trending, x="category", y="recent_count", color="category",
        title=f"Kategoria Aumenta iha {window_days} loron ikus",
        labels={"category": "Kategoria", "recent_count": "Notísia Foun"},
    )
    fig_cat.update_layout(showlegend=False)
    st.plotly_chart(fig_cat, use_container_width=True)


def render_trending(df: pd.DataFrame, labels: dict):
    st.subheader("🔥 Tópiku & Notísia Populár")

    window_label = st.radio(
        "Kalkula tendénsia bazeia ba atividade:", ["loron 3 ikus", "loron 7 ikus", "loron 14 ikus"],
        horizontal=True,
    )
    window_days = {"loron 3 ikus": 3, "loron 7 ikus": 7, "loron 14 ikus": 14}[window_label]

    render_category_trending(df, window_days)
    st.divider()

    trending_df = compute_trending_topics(df, labels, window_days=window_days)
    if trending_df.empty:
        st.info(
            "Seidauk iha dadus ho data sufisiente atu kalkula tendénsia. Husik pedidu automátiku halao "
            "iha loron balu (ka halao koleta manual iha loron seluk) atu tendénsia hahú forma."
        )
        return

    top_trending = trending_df.head(5)
    cols = st.columns(len(top_trending))
    for rank, (col, (_, row)) in enumerate(zip(cols, top_trending.iterrows()), start=1):
        with col:
            st.markdown(
                f"""<div class="news-card">
                    <div class="meta">🔥 #{rank} Populár</div>
                    <h4>{row['topik']}</h4>
                    <div class="meta">{row['recent_count']} notísia foun • {row['total_count']} total</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("**📰 Notísia foun husi tópiku populár sira:**")

    tdf = df.copy()
    tdf["published_at"] = pd.to_datetime(tdf["published_at"], errors="coerce")
    trending_cluster_ids = top_trending["cluster"].tolist()
    trending_news = (
        tdf[tdf["cluster"].isin(trending_cluster_ids)]
        .sort_values("published_at", ascending=False)
        .head(15)
    )

    if trending_news.empty:
        st.info("Seidauk iha notísia espesífiku ba tópiku populár agora.")
        return

    for _, row in trending_news.iterrows():
        topik = labels.get(row.get("cluster"), labels.get(str(row.get("cluster")), ""))
        pub = row.get("published_at")
        pub_str = pub.strftime("%d %b %Y, %H:%M") if pd.notna(pub) else "-"
        link = row.get("link", "")
        title = row.get("title", "(sem títulu)")
        title_html = (
            f'<a href="{link}" target="_blank" style="text-decoration:none;">{title}</a>' if link else title
        )
        st.markdown(
            f"""<div class="news-card">
                <div class="meta">🔥 {pub_str} • {row.get('category', '') or 'Jerál'} • {topik}</div>
                <h4>{title_html}</h4>
            </div>""",
            unsafe_allow_html=True,
        )


def render_source_config(config: dict) -> list:
    """
    Sidebar orijen dadus (versaun hamoos):
    - Orijen ho URL: checkbox ho naran; kampu URL mosu bainhira ativu.
    - Orijen seidauk iha URL: tau hamutuk iha expander ida.
    - Bele aumenta orijen foun.
    Retorna lista `sources` atual (uza ba `_execute_pipeline`).
    """
    if "sources_config" not in st.session_state:
        # Kopia inisiál husi config/settings.yaml — muda sidebar la altera fixu
        st.session_state["sources_config"] = [dict(s) for s in config.get("sources", [])]

    sources = st.session_state["sources_config"]

    def _source_row(i: int, src: dict):
        name = src.get("name", f"Orijen {i+1}")
        src["enabled"] = st.checkbox(
            name, value=src.get("enabled", True), key=f"src_enabled_{i}"
        )
        if src["enabled"]:
            src["feed_url"] = st.text_input(
                f"URL RSS {name}",
                value=src.get("feed_url", ""),
                placeholder="https://.../feed/",
                key=f"src_url_{i}",
                label_visibility="collapsed",
            )
            if not (src.get("feed_url") or "").strip():
                st.caption("⚠️ Seidauk iha URL RSS — sei salta automátiku.")

    with_url = [(i, s) for i, s in enumerate(sources) if (s.get("feed_url") or "").strip()]
    without_url = [(i, s) for i, s in enumerate(sources) if not (s.get("feed_url") or "").strip()]

    st.caption("Ativa/dezativa kada orijen no hatama/hadia URL RSS feed nian.")
    for i, src in with_url:
        _source_row(i, src)

    if without_url:
        with st.expander(f"Orijen seidauk konfigura ({len(without_url)})", expanded=False):
            st.caption("Klik checkbox no hatama URL RSS atu ativa.")
            for i, src in without_url:
                _source_row(i, src)

    with st.expander("➕ Aumenta orijen foun", expanded=False):
        new_name = st.text_input("Naran orijen", key="new_src_name", placeholder="ex: Timor Post")
        new_url = st.text_input("URL RSS feed", key="new_src_url", placeholder="https://.../feed/")
        if st.button("Aumenta", key="add_source_btn"):
            if new_name.strip():
                sources.append({"name": new_name.strip(), "feed_url": new_url.strip(), "enabled": True})
                st.rerun()
            else:
                st.warning("Hatama naran orijen molok aumenta.")

    return sources


# ----------------------------------------------------------------------
# Prinsipál
# ----------------------------------------------------------------------
def render_analysis(data, filtered_search):
    """Tab analiza: tópiku, tendénsia, nuvem liafuan, notísia (baze ba modelu ativu)."""
    if not data:
        st.info("Seidauk iha dadus atu analiza. Hala'o pedidu dadus, preprosesamentu no treinu iha tab 1 to'o 3, "
                "ka importa dataset iha sidebar.")
        return
    notice = data.get("memory_notice")
    if notice:
        st.warning(
            f"⚠️ Memória la to'o atu lee ficheiru master tomak: painel hatudu de'it {notice['kept']:,} "
            f"notísia foun liu husi {notice['total']:,} iha ficheiru. Atu hatudu hotu, aumenta RAM ka arkiva "
            "ficheiru `tatoli_cleaned_master.csv` ne'ebé tuan."
        )

    df = data["df"]
    topics_payload = data["topics_payload"]
    labels = {t["cluster_id"]: t["label"] for t in topics_payload.get("topics", [])} if topics_payload else {}

    if "cluster" not in df.columns:
        st.warning("Dadus seidauk iha rezultadu klaster. Hala'o treinu iha tab **🧠 3. Treinu Modelu**.")
        return

    render_stats(df)
    st.divider()

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        filtered_df = render_language_filter(df)
    with fcol2:
        filtered_df = render_category_filter(filtered_df)
    with fcol3:
        filtered_df = render_date_filter(filtered_df)

    tabs = st.tabs([
        "🏠 Uma", "🔥 Tópiku Populár", "📊 Distribuisaun Tópiku", "📈 Tendénsia Tópiku",
        "☁️ Kalohan Liafuan", "🔑 Liafuan Xave", "📰 Notísia", "🖼️ Galeria", "📅 Istória",
        IMPORT_MSG["themes_title"], PREP_MSG["title"],
    ])
    with tabs[0]:
        render_overview(filtered_df, labels)
    with tabs[1]:
        render_trending(filtered_df, labels)
    with tabs[2]:
        render_topic_distribution(filtered_df, labels)
    with tabs[3]:
        render_topic_trend(filtered_df, labels)
    with tabs[4]:
        render_wordclouds(filtered_df, labels)
    with tabs[5]:
        render_keywords_list(topics_payload)
    with tabs[6]:
        render_news_table(filtered_df, labels, filtered_search)
    with tabs[7]:
        render_gallery(filtered_df, labels)
    with tabs[8]:
        render_history(filtered_df, labels)
    with tabs[9]:
        render_candidate_themes(data.get("themes"), data.get("model_meta"))
    with tabs[10]:
        render_preprocessing(filtered_df)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    config = get_config()
    registry = ModelRegistry(workflow_dirs(config)["models"])

    # Karrega dadus/modelu ne'ebé rai ona; karrega fali bainhira iha modelu foun (treinu iha painel)
    current_version = registry.current_version_id()
    if "news_result" not in st.session_state:
        st.session_state["news_result"] = load_latest_from_disk(config)
        st.session_state["loaded_version"] = current_version
    elif current_version and st.session_state.get("loaded_version") != current_version:
        saved = load_saved_model_state(config)
        if saved is not None:
            st.session_state["news_result"] = saved
        st.session_state["loaded_version"] = current_version

    with st.sidebar:
        st.header("⚙️ Konfigurasaun")

        with st.expander("🔌 Orijen Dadus", expanded=False):
            sources = render_source_config(config)

        with st.expander("🧩 Klaster", expanded=False):
            min_k = st.slider("Kuantidade klaster mínimu", 2, 10, config["clustering"]["min_k"])
            max_k = st.slider("Kuantidade klaster máximu", min_k, 15, max(min_k, config["clustering"]["max_k"]))

        with st.expander(IMPORT_MSG["panel_title"], expanded=False):
            render_import_panel(config, min_k, max_k)

        st.divider()
        search_query = st.text_input("🔍 Buka notísia (títulu/konteúdu)", value="")
        st.caption(
            "Pedidu dadus, preprosesamentu no treinu hala'o iha tab 1 to'o 3. Pedidu automátiku hala'o kada "
            "intervalu ne'ebé hili iha tab 1."
        )

    render_header(_read_last_fetch_time(config))

    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])

    t1, t2, t3, t4 = st.tabs([
        "📥 1. Pedidu Dadus", "🧪 2. Preprosesamentu", "🧠 3. Treinu Modelu", "📊 4. Analiza Notísia",
    ])
    with t1:
        render_request_tab(config, sources)
    with t2:
        render_preprocessing_workflow(config)
    with t3:
        render_training_workflow(config)
    with t4:
        render_analysis(st.session_state.get("news_result"), search_query)


if __name__ == "__main__":
    main()
