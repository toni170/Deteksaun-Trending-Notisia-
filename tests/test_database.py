import pandas as pd

from database.database import (
    articles_to_dataframe, count_articles, get_latest_run, get_session,
    init_db, record_training_run, upsert_articles,
)


def _df():
    return pd.DataFrame({
        "news_id": ["a1", "a2"],
        "source": ["Tatoli", "Timor Post (Tetun)"],
        "title": ["Judul 1", "Judul 2"],
        "published_at": ["Mon, 12 Oct 2026 08:00:00 +0000", "2026-10-13T09:30:00+09:00"],
        "content": ["isi 1", None],
        "category": ["Politika", "Ekonomia"],
        "language": ["tet", "tet"],
        "text_final": ["isi satu", "isi rua"],
        "token_count": [2, 2],
        "cluster": [0, 1],
    })


def test_upsert_insert_then_update(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    with get_session(db) as s:
        assert upsert_articles(s, _df()) == (2, 0)
    with get_session(db) as s:
        df = _df()
        df.loc[0, "cluster"] = 1
        df.loc[len(df)] = ["a3", "Tatoli", "Judul 3", "2026-10-14", "x", "Jerál", "tet", "tiga", 1, 0]
        assert upsert_articles(s, df) == (1, 2)
    with get_session(db) as s:
        assert count_articles(s) == 3
        out = articles_to_dataframe(s).set_index("news_id")
        assert out.loc["a1", "cluster_id"] == 1
        assert out.loc["a2", "published_dt"].day == 13   # ISO +09:00 -> 00:30 UTC tanggal 13


def test_record_training_run(tmp_path):
    db = str(tmp_path / "t.db")
    init_db(db)
    with get_session(db) as s:
        record_training_run(s, {0: ["a", "b"], 1: ["c"]}, {0: "a / b", 1: "c"},
                            cluster_sizes={0: 5, 1: 3}, n_documents=8, best_k=2, silhouette=0.1)
    with get_session(db) as s:
        run = get_latest_run(s)
        assert run.best_k == 2 and len(run.topics) == 2
        assert run.topics[0].top_terms == ["a", "b"] and run.topics[0].news_count == 5
