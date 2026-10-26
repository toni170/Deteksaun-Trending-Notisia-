"""
Koneksi & operasi dasar database SQLite (database/database.db).

Contoh:
    from database.database import init_db, get_session, upsert_articles

    init_db()
    with get_session() as session:
        inserted, updated = upsert_articles(session, cleaned_df)

Lokasi file DB bisa diganti lewat argumen `db_path` atau environment
variable TATOLI_DB_PATH (berguna untuk testing).
"""
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Iterator, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

# Izinkan `python database/database.py` dijalankan langsung dari folder mana pun.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from database.models import Article, Base, Topic, TrainingRun  # noqa: E402

logger = logging.getLogger("database")

DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "database", "database.db")

_engines: Dict[str, Engine] = {}


def resolve_db_path(db_path: Optional[str] = None) -> str:
    return db_path or os.environ.get("TATOLI_DB_PATH") or DEFAULT_DB_PATH


def get_engine(db_path: Optional[str] = None) -> Engine:
    """Engine SQLAlchemy (di-cache per path)."""
    path = resolve_db_path(db_path)
    if path not in _engines:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        engine = create_engine(f"sqlite:///{path}", future=True)

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # aktifkan foreign key (mati default di SQLite)
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        _engines[path] = engine
    return _engines[path]


def init_db(db_path: Optional[str] = None) -> str:
    """Buat semua tabel jika belum ada. Aman dipanggil berkali-kali."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    path = resolve_db_path(db_path)
    logger.info("Database siap: %s", path)
    return path


@contextmanager
def get_session(db_path: Optional[str] = None) -> Iterator[Session]:
    """Session dengan commit otomatis; rollback jika terjadi error."""
    session = sessionmaker(bind=get_engine(db_path), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ----------------------------------------------------------------------
# Artikel
# ----------------------------------------------------------------------
_ARTICLE_COLUMNS = [
    "news_id", "source", "title", "link", "published_at", "summary", "content",
    "category", "image_url", "fetched_at", "language", "text_clean", "text_final",
    "token_count",
]


def _clean_value(v):
    """NaN/NaT -> None, numpy scalar -> tipe Python biasa."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if hasattr(v, "item"):
        return v.item()
    return v


def upsert_articles(session: Session, df: pd.DataFrame, chunk_size: int = 2000) -> Tuple[int, int]:
    """
    Simpan DataFrame berita (kolom seperti data/cleaned master, `cluster` opsional).
    Berita dengan `news_id` yang sudah ada di-UPDATE, sisanya di-INSERT.
    Mengembalikan (jumlah_insert, jumlah_update).
    """
    if df is None or df.empty:
        return 0, 0
    if "news_id" not in df.columns:
        raise ValueError("DataFrame harus punya kolom 'news_id'.")

    df = df.drop_duplicates(subset=["news_id"], keep="last").copy()
    df = df[df["news_id"].notna()]
    df["news_id"] = df["news_id"].astype(str)

    if "published_at" in df.columns:
        # format="mixed": RSS (RFC 822) dan WordPress (ISO 8601) bisa bercampur di satu kolom
        published = pd.to_datetime(df["published_at"], utc=True, errors="coerce", format="mixed")
        df["_published_dt"] = published.dt.tz_localize(None)
    else:
        df["_published_dt"] = pd.NaT

    existing: Dict[str, int] = {
        nid: pk for nid, pk in session.execute(select(Article.news_id, Article.id))
    }

    to_insert, to_update = [], []
    for row in df.to_dict("records"):
        rec = {c: _clean_value(row.get(c)) for c in _ARTICLE_COLUMNS}
        rec["published_dt"] = _clean_value(row.get("_published_dt"))
        cluster = _clean_value(row.get("cluster"))
        rec["cluster_id"] = int(cluster) if cluster is not None else None
        if rec["token_count"] is not None:
            rec["token_count"] = int(rec["token_count"])

        pk = existing.get(rec["news_id"])
        if pk is None:
            to_insert.append(rec)
        else:
            rec["id"] = pk
            to_update.append(rec)

    for i in range(0, len(to_insert), chunk_size):
        session.bulk_insert_mappings(Article, to_insert[i:i + chunk_size])
    for i in range(0, len(to_update), chunk_size):
        session.bulk_update_mappings(Article, to_update[i:i + chunk_size])
    session.flush()

    logger.info("Artikel disimpan ke DB: %s baru, %s diperbarui", len(to_insert), len(to_update))
    return len(to_insert), len(to_update)


def articles_to_dataframe(session: Session) -> pd.DataFrame:
    """Baca seluruh tabel articles menjadi DataFrame."""
    return pd.read_sql(select(Article), session.get_bind())


def count_articles(session: Session) -> int:
    return session.query(Article).count()


# ----------------------------------------------------------------------
# Training run & topik
# ----------------------------------------------------------------------
def record_training_run(
    session: Session,
    topics: Dict[int, list],
    labels: Dict[int, str],
    cluster_sizes: Optional[Dict[int, int]] = None,
    n_documents: int = 0,
    n_features: Optional[int] = None,
    best_k: Optional[int] = None,
    silhouette: Optional[float] = None,
    inertia: Optional[float] = None,
    params: Optional[dict] = None,
    model_path: Optional[str] = None,
) -> TrainingRun:
    """Catat satu training run beserta topik-topiknya."""
    run = TrainingRun(
        trained_at=datetime.now(),
        n_documents=int(n_documents),
        n_features=n_features,
        best_k=best_k,
        silhouette=silhouette,
        inertia=inertia,
        params_json=json.dumps(params or {}, ensure_ascii=False, default=str),
        model_path=model_path,
    )
    for cid, words in topics.items():
        run.topics.append(Topic(
            cluster_id=int(cid),
            label=labels.get(cid, ""),
            top_terms_json=json.dumps([str(w) for w in words], ensure_ascii=False),
            news_count=int(cluster_sizes[cid]) if cluster_sizes and cid in cluster_sizes else None,
        ))
    session.add(run)
    session.flush()
    logger.info("Training run #%s dicatat (k=%s, %s topik)", run.id, best_k, len(run.topics))
    return run


def get_latest_run(session: Session) -> Optional[TrainingRun]:
    return session.execute(
        select(TrainingRun).order_by(TrainingRun.trained_at.desc(), TrainingRun.id.desc()).limit(1)
    ).scalar_one_or_none()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Database dibuat/diverifikasi di:", init_db())
