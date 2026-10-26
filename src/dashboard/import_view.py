"""
Painel dashboard: importa dataset (CSV/Excel) -> treina modelu -> hatudu tema kandidatu.
Hotu-hotu iha Tetun. Lójika serbisu iha `src/ingestion` no `src/training`; ficheiru ne'e
de'it UI (Streamlit) no helper ki'ik atu karrega modelu ne'ebé rai ona.
"""
import os

import streamlit as st

from src.ingestion.dataset_importer import DatasetImportError, import_dataset
from src.training.model_registry import ModelLoadError, ModelRegistry
from src.training.training_service import TrainingError, train_and_register

MSG = {
    "panel_title": "📥 Importa Dataset & Treina Modelu",
    "uploader": "Hili ficheiru dataset (CSV ka Excel .xlsx)",
    "formats": "Formatu suporta: .csv, .xlsx. Presiza kolona títulu (title) no konteúdu (content ka summary).",
    "idle": "Seidauk hili ficheiru. Hili ficheiru CSV ka Excel atu hahú.",
    "train_button": "🧠 Importa no Treina Modelu",
    "running": "Prosesa dataset no treina modelu...",
    "importing": "Lee no valida ficheiru...",
    "success": "✅ Modelu foun treina ona no ativu: versaun {version}, {n} notísia, {k} tópiku.",
    "error": "❌ Erru: {msg}",
    "old_kept": "ℹ️ Modelu antigu (versaun {version}) kontinua uza, la muda.",
    "no_old": "Seidauk iha modelu antigu ne'ebé rai ona.",
    "model_active": "Modelu ativu: versaun {version} · treina iha {when} · {n} notísia · {k} tópiku.",
    "model_none": "Seidauk iha modelu ne'ebé rai ona.",
    "themes_title": "🧩 Tema Kandidatu",
    "themes_caption": "Tema kandidatu husi K-Means — presiza revizaun umana molok konsidera tema final.",
    "themes_none": "Seidauk iha tema kandidatu. Importa dataset no treina modelu uluk iha sidebar.",
    "theme_item": "Tema {cid} — {label} ({n} notísia, {pct}%)",
    "terms": "Liafuan xave",
    "examples": "Títulu exemplu",
}


def get_registry(config: dict) -> ModelRegistry:
    models_dir = config["paths"].get("models") or os.path.join(os.getcwd(), "models")
    return ModelRegistry(models_dir)


def load_saved_model_state(config: dict):
    """Karrega modelu ne'ebé rai ona (restart-safe). Retorna dict ba `news_result` ka None."""
    try:
        bundle = get_registry(config).load_best_available()
    except Exception:  # la bele estraga dashboard tanba modelu korruptu
        return None
    if bundle is None:
        return None
    return {"df": bundle.df, "topics_payload": bundle.topics_payload,
            "themes": bundle.themes, "model_meta": bundle.meta}


def render_model_status(meta):
    if not meta:
        st.caption(MSG["model_none"])
        return
    st.caption(MSG["model_active"].format(
        version=meta.get("version_id", "?"), when=meta.get("trained_at", "?"),
        n=meta.get("n_docs_used", "?"), k=meta.get("best_k", "?")))


def render_import_panel(config: dict, min_k: int, max_k: int) -> bool:
    """Hatudu painel iha sidebar. Retorna True se modelu foun treina ho susesu iha run ne'e."""
    registry = get_registry(config)
    trained_now = False

    uploaded = st.file_uploader(MSG["uploader"], key="dataset_upload")
    st.caption(MSG["formats"])
    if uploaded is None:
        st.info(MSG["idle"])

    clicked = st.button(MSG["train_button"], key="train_btn", disabled=uploaded is None,
                        use_container_width=True)

    if clicked and uploaded is not None:
        previous = registry.current_version_id()
        with st.status(MSG["running"], expanded=True) as status:
            bar = st.progress(0.0, text=MSG["importing"])

            def _cb(fraction, message):
                bar.progress(min(max(fraction, 0.0), 1.0), text=message)
                st.write(message)

            try:
                imported = import_dataset(uploaded, filename=uploaded.name)
                st.write(imported.report.summary_tetun())
                result = train_and_register(
                    imported.df, registry, min_k=min_k, max_k=max_k,
                    source_name=uploaded.name, progress_cb=_cb,
                )
            except (DatasetImportError, TrainingError) as e:
                status.update(label=MSG["error"].format(msg=e.message), state="error")
                st.session_state["import_outcome"] = {
                    "state": "error", "message": MSG["error"].format(msg=e.message),
                    "old_version": previous,
                }
            except Exception as e:  # erru la antisipa -> la lakon aplikasaun
                status.update(label=MSG["error"].format(msg=str(e)), state="error")
                st.session_state["import_outcome"] = {
                    "state": "error", "message": MSG["error"].format(msg=str(e)),
                    "old_version": previous,
                }
            else:
                status.update(label=MSG["success"].format(
                    version=result.version_id, n=result.meta["n_docs_used"], k=result.meta["best_k"]),
                    state="complete", expanded=False)
                st.session_state["news_result"] = {
                    "df": result.df, "topics_payload": result.topics_payload,
                    "themes": result.themes, "model_meta": result.meta,
                }
                st.session_state["import_outcome"] = {
                    "state": "success", "message": MSG["success"].format(
                        version=result.version_id, n=result.meta["n_docs_used"], k=result.meta["best_k"]),
                }
                trained_now = True

    outcome = st.session_state.get("import_outcome")
    if outcome:
        if outcome["state"] == "success":
            st.success(outcome["message"])
        else:
            st.error(outcome["message"])
            if outcome.get("old_version"):
                st.info(MSG["old_kept"].format(version=outcome["old_version"]))
            else:
                st.caption(MSG["no_old"])

    data = st.session_state.get("news_result") or {}
    render_model_status(data.get("model_meta"))
    return trained_now


def render_candidate_themes(themes, meta=None):
    st.subheader(MSG["themes_title"])
    if not themes:
        st.info(MSG["themes_none"])
        return
    st.caption(MSG["themes_caption"])
    render_model_status(meta)
    for t in themes:
        title = MSG["theme_item"].format(cid=t["cluster_id"], label=t["label"],
                                         n=t["news_count"], pct=t["share_pct"])
        with st.expander(title):
            st.markdown(f"**{MSG['terms']}:** " + ", ".join(t["top_terms"]))
            if t.get("example_titles"):
                st.markdown(f"**{MSG['examples']}:**")
                for ex in t["example_titles"]:
                    st.markdown(f"- {ex}")
