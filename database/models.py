"""
Model ORM (SQLAlchemy 2.0) untuk database SQLite proyek Mining Notísia.

Tabel:
  - articles       : satu baris per berita (data mentah + hasil preprocessing + cluster)
  - training_runs  : riwayat tiap kali model TF-IDF + K-Means dilatih
  - topics         : topik/cluster hasil satu training run (label + kata kunci)
"""
import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Article(Base):
    """Satu berita. `news_id` sama dengan kolom news_id di CSV data/raw & data/cleaned."""
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    news_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    source: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    title: Mapped[Optional[str]] = mapped_column(Text)
    link: Mapped[Optional[str]] = mapped_column(Text)
    published_at: Mapped[Optional[str]] = mapped_column(String(64))          # teks asli dari sumber
    published_dt: Mapped[Optional[datetime]] = mapped_column(DateTime, index=True)  # hasil parse (UTC, naive)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    content: Mapped[Optional[str]] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    image_url: Mapped[Optional[str]] = mapped_column(Text)
    fetched_at: Mapped[Optional[str]] = mapped_column(String(64))

    # Hasil preprocessing
    language: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    text_clean: Mapped[Optional[str]] = mapped_column(Text)
    text_final: Mapped[Optional[str]] = mapped_column(Text)
    token_count: Mapped[Optional[int]] = mapped_column(Integer)

    # Hasil clustering
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<Article id={self.id} news_id={self.news_id!r} cluster={self.cluster_id}>"


class TrainingRun(Base):
    """Riwayat satu kali training (TF-IDF + K-Means)."""
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    n_documents: Mapped[int] = mapped_column(Integer, default=0)
    n_features: Mapped[Optional[int]] = mapped_column(Integer)
    best_k: Mapped[Optional[int]] = mapped_column(Integer)
    silhouette: Mapped[Optional[float]] = mapped_column(Float)
    inertia: Mapped[Optional[float]] = mapped_column(Float)
    params_json: Mapped[Optional[str]] = mapped_column(Text)   # parameter training (JSON)
    model_path: Mapped[Optional[str]] = mapped_column(Text)    # lokasi model.pkl

    topics: Mapped[List["Topic"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="Topic.cluster_id"
    )

    @property
    def params(self) -> dict:
        return json.loads(self.params_json) if self.params_json else {}

    def __repr__(self) -> str:
        return f"<TrainingRun id={self.id} k={self.best_k} docs={self.n_documents}>"


class Topic(Base):
    """Satu topik (cluster) milik satu training run."""
    __tablename__ = "topics"
    __table_args__ = (UniqueConstraint("run_id", "cluster_id", name="uq_topic_run_cluster"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), index=True)
    cluster_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[Optional[str]] = mapped_column(String(255))
    top_terms_json: Mapped[str] = mapped_column(Text, default="[]")
    news_count: Mapped[Optional[int]] = mapped_column(Integer)

    run: Mapped["TrainingRun"] = relationship(back_populates="topics")

    @property
    def top_terms(self) -> List[str]:
        return json.loads(self.top_terms_json or "[]")

    def __repr__(self) -> str:
        return f"<Topic run={self.run_id} cluster={self.cluster_id} label={self.label!r}>"
