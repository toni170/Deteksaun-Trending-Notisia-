"""
Vista istóriku: evolusaun notísia tinan-tinan / fulan-fulan, tópiku kada tinan,
liafuan-xave kada tinan, no export. Funsaun "puru" (la depende Streamlit) atu fasil teste.
Oras hotu konvérte ba oras Dili (UTC+9) — hanesan filtru tempu iha dashboard.
"""
from collections import Counter
from datetime import timedelta

import pandas as pd

DILI_OFFSET = timedelta(hours=9)


def with_dili_time(df: pd.DataFrame) -> pd.DataFrame:
    """Fó kolumna `_t` (oras Dili), `year` no `month`; hasai liña ne'ebé la iha data."""
    out = df.copy()
    out["_t"] = pd.to_datetime(out["published_at"], errors="coerce") + DILI_OFFSET
    out = out.dropna(subset=["_t"])
    out["year"] = out["_t"].dt.year
    out["month"] = out["_t"].dt.to_period("M").dt.to_timestamp()
    return out


def pick_granularity(df: pd.DataFrame) -> str:
    """Hili otomátiku: tinan > 2 -> fulan, > 2 fulan -> semana, la'e -> loron."""
    if df.empty:
        return "Loron"
    span = (df["_t"].max() - df["_t"].min()).days
    return "Fulan" if span > 400 else "Semana" if span > 60 else "Loron"


def articles_per_period(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """Kuantidade notísia kada períodu. granularity: Loron | Semana | Fulan | Tinan."""
    freq = {"Loron": "D", "Semana": "W", "Fulan": "MS", "Tinan": "YS"}[granularity]
    s = df.set_index("_t").resample(freq).size()
    return s.rename("kuantidade").rename_axis("periodu").reset_index()


def topic_by_year(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Tinan x tópiku -> kuantidade (formatu long)."""
    d = df.copy()
    d["topik"] = d["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Klaster {c}")))
    return d.groupby(["year", "topik"]).size().reset_index(name="kuantidade")


def top_terms(df: pd.DataFrame, n: int = 20, min_len: int = 3) -> pd.DataFrame:
    """Liafuan dala barak liu (husi `text_final`, ne'ebé limpu ona)."""
    counter = Counter()
    for text in df["text_final"].dropna().astype(str):
        counter.update(w for w in text.split() if len(w) >= min_len)
    return pd.DataFrame(counter.most_common(n), columns=["liafuan", "kuantidade"])


def yearly_summary(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """Tabela rezumu kada tinan: total, média/loron, tópiku no kategoria prinsipál, orijen."""
    rows = []
    for year, g in df.groupby("year"):
        days = max((g["_t"].max().normalize() - g["_t"].min().normalize()).days + 1, 1)
        top_c = g["cluster"].value_counts().idxmax() if "cluster" in g and not g.empty else None
        top_cat = ""
        if "category" in g.columns:
            cats = g["category"].fillna("").astype(str)
            cats = cats[cats.str.strip() != ""]
            top_cat = cats.value_counts().idxmax() if not cats.empty else ""
        rows.append({
            "tinan": int(year),
            "total notísia": len(g),
            "média/loron": round(len(g) / days, 1),
            "tópiku prinsipál": labels.get(top_c, labels.get(str(top_c), f"Klaster {top_c}")) if top_c is not None else "",
            "kategoria prinsipál": top_cat,
            "orijen": g["source"].nunique() if "source" in g.columns else 1,
        })
    return pd.DataFrame(rows)


def export_csv_bytes(df: pd.DataFrame, labels: dict) -> bytes:
    """CSV (UTF-8 ho BOM, loke diretamente iha Excel) husi notísia ne'ebé filtra."""
    out = df.copy()
    out["topik"] = out["cluster"].map(lambda c: labels.get(c, labels.get(str(c), f"Klaster {c}")))
    out["tempu_dili"] = (pd.to_datetime(out["published_at"], errors="coerce") + DILI_OFFSET).dt.strftime("%Y-%m-%d %H:%M")
    cols = [c for c in ["tempu_dili", "source", "title", "topik", "category", "language", "summary", "link"]
            if c in out.columns]
    return out[cols].sort_values("tempu_dili").to_csv(index=False).encode("utf-8-sig")


# ----------------------------------------------------------------------
# Streamlit
# ----------------------------------------------------------------------
def render_history(df: pd.DataFrame, labels: dict):
    import plotly.express as px
    import streamlit as st

    st.subheader("📅 Istória: Evolusaun Notísia")
    d = with_dili_time(df)
    if d.empty:
        st.info("La iha dadus data atu hatudu istória.")
        return

    gran_options = ["Otomátiku", "Loron", "Semana", "Fulan", "Tinan"]
    choice = st.radio("Agrupa tuir:", gran_options, horizontal=True, key="hist_gran")
    gran = pick_granularity(d) if choice == "Otomátiku" else choice
    per = articles_per_period(d, gran)
    fig = px.area(per, x="periodu", y="kuantidade", title=f"Notísia kada {gran.lower()}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Rezumu kada tinan")
    st.dataframe(yearly_summary(d, labels), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        ty = topic_by_year(d, labels)
        fig2 = px.bar(ty, x="year", y="kuantidade", color="topik", title="Tópiku kada tinan")
        fig2.update_xaxes(type="category")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        years = sorted(d["year"].unique(), reverse=True)
        year = st.selectbox("Liafuan dala barak liu iha tinan:", years, key="hist_year")
        terms = top_terms(d[d["year"] == year], n=20)
        if terms.empty:
            st.info("La iha liafuan atu hatudu.")
        else:
            fig3 = px.bar(terms.iloc[::-1], x="kuantidade", y="liafuan", orientation="h",
                          title=f"Liafuan prinsipál {year}")
            st.plotly_chart(fig3, use_container_width=True)

    st.download_button(
        "⬇️ Foti notísia ne'ebé filtra (CSV)",
        data=export_csv_bytes(df, labels),
        file_name="notisia_filtradu.csv", mime="text/csv", key="hist_download",
    )
