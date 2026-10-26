"""Análize rezultadu treinu (la depende Streamlit): tópiku populár no tendénsia tempu."""
import pandas as pd


def _label(labels: dict, cid) -> str:
    return labels.get(cid, labels.get(str(cid), f"Klaster {cid}"))


def trending_topics(df: pd.DataFrame, labels: dict, window_days: int = 7, min_articles: int = 1) -> pd.DataFrame:
    """
    Tópiku populár: kompara notísia iha janela ikus (`window_days`) ho janela molok nia.
    Referénsia tempu = data notísia foun liu iha dadus (la'ós oras agora), tanba dadus bele tuan.
    pontuasaun = notísia foun + aumentu kompara ho janela molok (de'it se pozitivu).
    """
    cols = ["klaster", "topiku", "notisia_foun", "notisia_molok", "aumentu_pct", "pontuasaun"]
    if df is None or df.empty or "published_at" not in df.columns or "cluster" not in df.columns:
        return pd.DataFrame(columns=cols)
    t = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    d = df.assign(_t=t).dropna(subset=["_t"])
    if d.empty:
        return pd.DataFrame(columns=cols)
    end = d["_t"].max()
    recent_start = end - pd.Timedelta(days=window_days)
    prev_start = recent_start - pd.Timedelta(days=window_days)
    recent = d[d["_t"] > recent_start]["cluster"].value_counts()
    prev = d[(d["_t"] > prev_start) & (d["_t"] <= recent_start)]["cluster"].value_counts()
    rows = []
    for cid, n_recent in recent.items():
        if n_recent < min_articles:
            continue
        n_prev = int(prev.get(cid, 0))
        growth = (n_recent - n_prev) / max(n_prev, 1) * 100
        rows.append({"klaster": cid, "topiku": _label(labels, cid), "notisia_foun": int(n_recent),
                     "notisia_molok": n_prev, "aumentu_pct": round(growth, 1),
                     "pontuasaun": int(n_recent) + max(int(n_recent) - n_prev, 0)})
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(["pontuasaun", "notisia_foun"], ascending=False).reset_index(drop=True)


def topic_time_series(df: pd.DataFrame, labels: dict, freq: str = "W") -> pd.DataFrame:
    """Kuantidade notísia kada períodu no tópiku (formatu long: periodu, topiku, kuantidade)."""
    if df is None or df.empty or "published_at" not in df.columns:
        return pd.DataFrame(columns=["periodu", "topiku", "kuantidade"])
    t = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.tz_localize(None)
    d = df.assign(_t=t).dropna(subset=["_t"])
    d["topiku"] = d["cluster"].map(lambda c: _label(labels, c))
    d["periodu"] = d["_t"].dt.to_period({"D": "D", "W": "W", "M": "M"}[freq]).dt.start_time
    return d.groupby(["periodu", "topiku"]).size().reset_index(name="kuantidade")
