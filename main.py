#!/usr/bin/env python3
"""
main.py — Pipeline prinsipál ba Mining Notísia Online no Deteksaun Topiku.

Fluxu: RSS Tatoli -> Koleksaun (data/raw) -> Prosesamentu inisiál (data/cleaned)
-> Estrasaun Karakterístika TF-IDF (data/models) -> Deteksaun Topiku K-Means (data/models)
-> Estrasaun Liafuan-xave (data/keywords) -> Rai ba database SQLite (database/database.db)
no model.pkl (models/model.pkl) -> rezumu rezultadu.

Ezemplu uza:
    python main.py
    python main.py --feed-url "https://tatoli.tl/feed/" --min-k 2 --max-k 8

Ba dashboard interativu, hala’o:
    streamlit run src/dashboard/trend_dashboard.py
"""
import argparse
import logging
import sys

from src.utils import load_config, setup_logging
from src.scraping.rss_scraper import collect_rss, collect_multi_rss
from src.scraping.web_scraper import collect_full_articles, collect_full_articles_multi
from src.scraping.history_scraper import collect_history_multi, resolve_since
from src.preprocessing.pipeline import run_preprocessing
from src.feature_extraction.tfidf_vectorizer import run_feature_extraction
from src.clustering.kmeans_model import run_clustering
from src.keywords.keyword_extractor import run_keyword_extraction

import pandas as pd

SOURCE_NAME = "tatoli"


def persist_results(config: dict, cleaned_df: pd.DataFrame, extractor, topic_model,
                    topics: dict, topic_labels: dict) -> dict:
    """
    Langkah 6: simpan hasil ke database SQLite (articles, training_runs, topics)
    dan ke bundle models/model.pkl (TF-IDF + K-Means + label topik).
    Import dibuat lazy supaya pipeline inti tetap jalan walau SQLAlchemy belum terpasang.
    """
    from database.database import get_session, init_db, record_training_run, upsert_articles
    from models.training import build_bundle, save_model

    paths = config["paths"]
    clustering_cfg = config.get("clustering", {})
    k = topic_model.best_k

    bundle = build_bundle(
        extractor.vectorizer, topic_model.kmeans, topics, topic_labels,
        metadata={
            "n_documents": int(len(cleaned_df)),
            "silhouette": topic_model.silhouette_scores_.get(k),
            "inertia": topic_model.inertia_scores_.get(k),
            "silhouette_scores": topic_model.silhouette_scores_,
            "inertia_scores": topic_model.inertia_scores_,
            "params": clustering_cfg,
            "built_from": "main.py pipeline",
        },
    )
    model_path = save_model(bundle, paths.get("model_bundle"))

    init_db(paths.get("database"))
    cluster_sizes = cleaned_df["cluster"].value_counts().to_dict()
    with get_session(paths.get("database")) as session:
        inserted, updated = upsert_articles(session, cleaned_df)
        record_training_run(
            session, topics, topic_labels, cluster_sizes=cluster_sizes,
            n_documents=len(cleaned_df), n_features=bundle["metadata"]["n_features"],
            best_k=k, silhouette=bundle["metadata"]["silhouette"],
            inertia=bundle["metadata"]["inertia"], params=clustering_cfg, model_path=model_path,
        )
    return {"model_path": model_path, "db_path": paths.get("database"),
            "db_inserted": inserted, "db_updated": updated}


def run_pipeline(config: dict, sources: list = None, feed_url: str = None,
                  min_k=None, max_k=None, collection_mode: str = None, since: str = None,
                  save_db: bool = True):
    """
    Hala'o pipeline kompletu.

    `sources`: lista orijen `{"name", "feed_url", "enabled"}` (padraun: husi
    `config["sources"]`, dala barak Tatoli + Timor Post + Timor News + GMN TV).
    `feed_url`: se hatama (ex: liuhusi `--feed-url` CLI), ida ne'e SUBSTITUI
    `sources` tomak ho orijen "Custom" ida de'it — mantein kompatibilidade
    ho versaun antigu (feed úniku).
    `since`: data "YYYY-MM-DD" (ex: "2016-01-01"). Se hatama, koleta notísia
    istóriku husi data ne'e to agora (WordPress REST API) NE'EBÉ substitui koleta RSS normál.
    """
    logger = logging.getLogger("main")

    scraping_cfg = config.get("scraping", {})
    mode = collection_mode or ("full" if scraping_cfg.get("full_content", True) else "rss")
    if since:
        mode = "history"

    if feed_url:
        sources = [{"name": "Custom", "feed_url": feed_url, "enabled": True}]
    elif sources is None:
        sources = config.get("sources", [])

    active_names = [s["name"] for s in sources if s.get("enabled", True) and (s.get("feed_url") or "").strip()]
    logger.info(
        "=== Hahu pipeline Mining Notísia (mode=%s) ba orijen: %s ===",
        mode, ", ".join(active_names) if active_names else "(laiha orijen aktivu!)",
    )

    if not active_names:
        raise ValueError(
            "Laiha orijen dadus aktivu ho feed_url válidu. Verifika `sources` iha "
            "config/settings.yaml (ka lista orijen iha sidebar dashboard)."
        )

    # 1) Koleksaun: simu no rai notísia brutu iha data/raw (husi orijen hotu-hotu aktivu)
    if mode == "history":
        hist_cfg = config.get("history", {})
        since = resolve_since(since, config["paths"]["raw"],
                              default=str(hist_cfg.get("start_date", "2016-01-01")))
        logger.info("[1/6] Koleta notísia istóriku husi %s to agora ...", since)
        raw_paths = collect_history_multi(
            sources, config["paths"]["raw"], since=since,
            per_page=hist_cfg.get("per_page", 100),
            max_pages=hist_cfg.get("max_pages", 500),
            delay_seconds=hist_cfg.get("delay_seconds", 1.0),
        )
        logger.info("Notísia istóriku ne’ebé simu: %s | Total notísia úniku: %s",
                    raw_paths["new_entries"], raw_paths["total_unique"])
    elif mode == "full":
        logger.info("[1/6] Simu RSS (deskoberta) + scraping konteúdu kompletu husi orijen hotu-hotu ...")
        raw_paths = collect_full_articles_multi(
            sources,
            config["paths"]["raw"],
            max_articles=scraping_cfg.get("max_articles_per_run"),
            delay_seconds=scraping_cfg.get("delay_seconds", 1.5),
            default_selectors=scraping_cfg.get("selectors"),
            selectors_by_source=scraping_cfg.get("selectors_by_source"),
        )
        logger.info(
            "Artigu foun ne’ebé scrape kompletu: %s (la konsege: %s) | Total notísia úniku: %s",
            raw_paths.get("scraped_new_articles", 0), raw_paths.get("failed_articles", 0),
            raw_paths["total_unique"],
        )
    else:
        logger.info("[1/6] Simu RSS feed husi orijen hotu-hotu (rezumu deit) ...")
        raw_paths = collect_multi_rss(sources, config["paths"]["raw"])
        logger.info(
            "Notísia foun iha ezekusaun ida-ne’e: %s | Total notísia úniku ne’ebé simu: %s",
            raw_paths.get("new_entries", 0), raw_paths["total_unique"],
        )
    for name, info in raw_paths.get("per_source", {}).items():
        logger.info("  - Orijen '%s': %s", name, info)

    # 2) Prosesamentu inisiál: hamos testu, deteksaun lian, hasai stopword, stemming
    logger.info("[2/6] Halo prosesamentu inisiál ba testu (hamos + stemming) ...")
    cleaned_master_path = run_preprocessing(raw_paths["master_csv"], config["paths"]["cleaned"])

    cleaned_df = pd.read_csv(cleaned_master_path)

    # 3) Estrasaun Karakterístika: TF-IDF
    logger.info("[3/6] Hahu halo reprezentasaun TF-IDF ...")
    clustering_cfg = config.get("clustering", {})
    texts = cleaned_df["text_final"].fillna("").tolist()
    X, extractor = run_feature_extraction(
        texts,
        models_dir=config["paths"]["models"],
        name=SOURCE_NAME,
        max_features=clustering_cfg.get("max_features", 5000),
        ngram_range=(clustering_cfg.get("ngram_min", 1), clustering_cfg.get("ngram_max", 2)),
        min_df=clustering_cfg.get("min_df", 2),
    )

    # 4) Deteksaun Topiku: K-Means (k ótimu husi silhouette score)
    logger.info("[4/6] Halo clustering K-Means ba deteksaun topiku ...")
    labels, topic_model = run_clustering(
        X,
        models_dir=config["paths"]["models"],
        name=SOURCE_NAME,
        min_k=min_k or clustering_cfg.get("min_k", 2),
        max_k=max_k or clustering_cfg.get("max_k", 10),
        silhouette_sample_size=clustering_cfg.get("silhouette_sample_size", 3000),
    )
    cleaned_df["cluster"] = labels
    cleaned_df.to_csv(cleaned_master_path, index=False, encoding="utf-8-sig")

    # 5) Estrasaun Liafuan-xave
    logger.info("[5/6] Estrai liafuan-xave husi topiku ida-idak ...")
    top_n = config.get("keywords", {}).get("top_n_terms", 10)
    topics, topic_labels, keywords_path = run_keyword_extraction(
        extractor.vectorizer, topic_model.kmeans, cleaned_df,
        config["paths"]["keywords"], top_n=top_n, name=SOURCE_NAME,
    )

    # 6) Persistensi: database SQLite + models/model.pkl (la fatál se la konsege)
    persisted = {}
    if save_db:
        logger.info("[6/6] Rai rezultadu ba database SQLite no models/model.pkl ...")
        try:
            persisted = persist_results(config, cleaned_df, extractor, topic_model, topics, topic_labels)
            logger.info("Database: +%s foun, %s atualiza | model.pkl -> %s",
                        persisted["db_inserted"], persisted["db_updated"], persisted["model_path"])
        except Exception as e:
            logger.exception("La konsege rai ba database/model.pkl (pipeline CSV mantein ok): %s", e)
    else:
        logger.info("[6/6] Laloos (--no-db): la rai ba database/model.pkl.")

    logger.info("=== Pipeline remata ===")

    return {
        "persisted": persisted,
        "raw_paths": raw_paths,
        "cleaned_master_path": cleaned_master_path,
        "cleaned_df": cleaned_df,
        "extractor": extractor,
        "topic_model": topic_model,
        "topics": topics,
        "topic_labels": topic_labels,
        "keywords_path": keywords_path,
    }


def print_summary(result: dict):
    df = result["cleaned_df"]
    raw_paths = result["raw_paths"]

    print("\n" + "=" * 70)
    print(f"Total notísia (akumuladu)  : {len(df)}")
    print(f"Númeru topiku ne’ebé deteta : {result['topic_model'].best_k}")
    if "scraped_new_articles" in raw_paths:
        print(f"Artigu ne’ebé scrape kompletu iha ezekusaun ida-ne’e : {raw_paths['scraped_new_articles']}")
        if raw_paths.get("failed_articles"):
            print(f"Artigu la konsege scrape (uza rezumu RSS hodi substitui) : {raw_paths['failed_articles']}")
    print("-" * 70)
    print("Topiku ne’ebé deteta:")
    for cid, words in result["topics"].items():
        size = int((df["cluster"] == cid).sum())
        label = result["topic_labels"].get(cid, "")
        print(f"  [Cluster {cid}] ({size} notísia) {label}")
        print(f"      liafuan-xave: {', '.join(words)}")
    print("-" * 70)
    print(f"Dadus brutu (master)  -> {result['raw_paths']['master_csv']}")
    print(f"Dadus hamos (master)  -> {result['cleaned_master_path']}")
    print(f"Liafuan-xave/topiku   -> {result['keywords_path']}")
    persisted = result.get("persisted") or {}
    if persisted:
        print(f"Database SQLite       -> {persisted['db_path']} (+{persisted['db_inserted']} foun, {persisted['db_updated']} atualiza)")
        print(f"Model (TF-IDF+KMeans) -> {persisted['model_path']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mining Notísia Online no Deteksaun Topiku — pipeline prinsipál"
    )
    parser.add_argument("--feed-url", default=None, help="URL RSS feed (padraun husi config/settings.yaml)")
    parser.add_argument("--config", default=None, help="Dalan ba settings.yaml (opsionál)")
    parser.add_argument("--min-k", type=int, default=None, help="Númeru cluster minimu")
    parser.add_argument("--max-k", type=int, default=None, help="Númeru cluster máximu")
    parser.add_argument(
        "--since", default=None, metavar="YYYY-MM-DD",
        help="Koleta notísia istóriku husi data ne'e to agora (ex: --since 2016-01-01). "
             "Uza '--since auto' atu kontinua husi notísia ikus ne'ebé rai ona.",
    )
    parser.add_argument(
        "--source", action="append", default=None, metavar="NARAN",
        help="Uza de'it orijen ne'e (naran hanesan iha settings.yaml, ex: --source Tatoli). "
             "Bele repete.",
    )
    parser.add_argument(
        "--mode", choices=["full", "rss"], default=None,
        help="'full' = scraping konteúdu kompletu+kategoria+imajen; 'rss' = rezumu RSS deit. "
             "Padraun: tuir config/settings.yaml (scraping.full_content).",
    )
    parser.add_argument(
        "--no-db", action="store_true",
        help="La rai rezultadu ba database SQLite no models/model.pkl (CSV/JSON de'it).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config["paths"]["logs"])

    sources = None
    if args.source:
        wanted = {n.strip().lower() for n in args.source}
        sources = [s for s in config.get("sources", []) if s.get("name", "").lower() in wanted]
        if not sources:
            names = ", ".join(s.get("name", "") for s in config.get("sources", []))
            print(f"\n[SALA] Orijen '{', '.join(args.source)}' la hetan. Iha: {names}\n", file=sys.stderr)
            sys.exit(1)

    try:
        result = run_pipeline(
            config, sources=sources, feed_url=args.feed_url, min_k=args.min_k, max_k=args.max_k,
            collection_mode=args.mode, since=args.since,
            save_db=not args.no_db,
        )
    except Exception as e:
        logging.getLogger("main").exception("Pipeline la konsege: %s", e)
        print(f"\n[SALA] Pipeline la konsege: {e}\n", file=sys.stderr)
        sys.exit(1)

    print_summary(result)


if __name__ == "__main__":
    main()