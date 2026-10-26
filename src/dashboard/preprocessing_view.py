"""
Tab dashboard "Preprosesamentu": hatudu etapa ida-idak (cleaning, case folding, tokenization,
stopword removal, stemming) ba notísia husi dataset ka tékstu ne'ebé utilizadór hakerek rasik.
Lójika iha `src.preprocessing.text_cleaner.explain_preprocessing` (hanesan ho pipeline loloos).
"""
import pandas as pd
import streamlit as st

from src.preprocessing.text_cleaner import explain_preprocessing

PREP_MSG = {
    "title": "🧪 Preprosesamentu",
    "intro": "Etapa preprosesamentu ba kada notísia (tuir orden iha pipeline) molok TF-IDF no K-Means:",
    "stages": [
        "**1. Cleaning** — hasai tag HTML, URL, tanda baca, karakter repete, no espasu sobra",
        "**2. Case folding** — hadi'a hotu ba letra ki'ik",
        "**3. Tokenization** — fahe tékstu ba token (liafuan)",
        "**4. Stopword removal** — hasai liafuan komun (iha, ba, husi, ...) no token badak (≤2 letra)",
        "**5. Stemming** — tesi sufiksu ba liafuan-baze (Tetun: heurístika; Indonézia: Sastrawi; Inglés: Porter)",
    ],
    "summary": "Rezumu dataset",
    "docs": "Notísia analiza",
    "avg_clean": "Média token (depois cleaning)",
    "avg_final": "Média token (final)",
    "languages": "Lian ne'ebé deteta",
    "pick": "Hili notísia atu haree etapa sira",
    "manual": "Ka hakerek tékstu rasik (opsionál; se prense, uza tékstu ne'e)",
    "walk": "Etapa ba etapa",
    "s0": "Tékstu orijinál (títulu + isi)",
    "s1": "Depois cleaning 1/2 — hasai tag HTML",
    "s2": "Depois case folding — letra ki'ik",
    "s3": "Depois cleaning 2/2 — hasai URL, tanda baca, karakter repete",
    "s4": "Depois tokenization — {n} token",
    "s5": "Depois stopword removal — {n} token mantein, {r} hasai",
    "removed": "Token ne'ebé hasai",
    "none": "(la iha)",
    "s6": "Depois stemming — lian: {lang}",
    "stem_off": "Stemming la aplika tanba lian la hatene — token la muda.",
    "stem_none": "La iha liafuan ne'ebé muda husi stemming.",
    "s7": "Rezultadu final (uza ba TF-IDF) — {n} token",
    "empty": "Tékstu mamuk depois preprosesamentu.",
    "no_data": "Seidauk iha dadus atu hatudu preprosesamentu.",
}

_SHOW_LIMIT = 2500


def _short(text: str) -> str:
    return text if len(text) <= _SHOW_LIMIT else text[:_SHOW_LIMIT] + " …"


def _txt(v) -> str:
    return v if isinstance(v, str) else ""


def _body(row) -> str:
    """Pilih isi hanesan ho `pipeline._pick_body_text`: content se iha, se lae summary."""
    content = _txt(row.get("content"))
    return content if content.strip() else _txt(row.get("summary"))


def render_walkthrough(steps: dict) -> None:
    M = PREP_MSG
    st.markdown(f"**{M['s0']}**")
    st.code(_short(steps["raw"]), language=None)
    st.markdown(f"**{M['s1']}**")
    st.code(_short(steps["after_html"]), language=None)
    st.markdown(f"**{M['s2']}**")
    st.code(_short(steps["after_case_folding"]), language=None)
    st.markdown(f"**{M['s3']}**")
    st.code(_short(steps["text_clean"]) or M["empty"], language=None)
    st.markdown(f"**{M['s4'].format(n=len(steps['tokens']))}**")
    st.code(_short(" | ".join(steps["tokens"])) or M["empty"], language=None)
    st.markdown(f"**{M['s5'].format(n=len(steps['tokens_no_stopwords']), r=len(steps['removed_tokens']))}**")
    st.code(_short(" | ".join(steps["tokens_no_stopwords"])) or M["empty"], language=None)
    st.caption(f"{M['removed']}: " + (", ".join(steps["removed_tokens"]) or M["none"]))
    st.markdown(f"**{M['s6'].format(lang=steps['language'])}**")
    if not steps["stemming_applied"]:
        st.caption(M["stem_off"])
    elif steps["stem_changes"]:
        st.code("\n".join(f"{a} → {b}" for a, b in steps["stem_changes"]), language=None)
    else:
        st.caption(M["stem_none"])
    st.markdown(f"**{M['s7'].format(n=steps['token_count'])}**")
    st.code(_short(steps["text_final"]) or M["empty"], language=None)


def render_preprocessing(df: pd.DataFrame, max_choices: int = 100) -> None:
    M = PREP_MSG
    st.subheader(M["title"])
    st.caption(M["intro"])
    for line in M["stages"]:
        st.markdown(f"- {line}")

    if df is None or df.empty:
        st.info(M["no_data"])
        return

    st.markdown(f"#### {M['summary']}")
    c1, c2, c3 = st.columns(3)
    c1.metric(M["docs"], f"{len(df):,}")
    if "text_clean" in df.columns:
        c2.metric(M["avg_clean"], f"{df['text_clean'].fillna('').str.split().str.len().mean():.1f}")
    if "text_final" in df.columns:
        c3.metric(M["avg_final"], f"{df['text_final'].fillna('').str.split().str.len().mean():.1f}")
    if "language" in df.columns:
        st.caption(M["languages"] + ": " + ", ".join(
            f"{lang} ({n})" for lang, n in df["language"].fillna("unknown").value_counts().items()))

    st.markdown(f"#### {M['walk']}")
    manual = st.text_area(M["manual"], key="prep_manual", height=80)
    if manual.strip():
        steps = explain_preprocessing("", manual)
    else:
        sample = df.head(max_choices).reset_index(drop=True)
        idx = st.selectbox(
            M["pick"], options=list(range(len(sample))), key="prep_article",
            format_func=lambda i: (_txt(sample.loc[i].get("title")) or f"#{i}")[:90],
        )
        row = sample.loc[idx]
        steps = explain_preprocessing(_txt(row.get("title")), _body(row))
    render_walkthrough(steps)
