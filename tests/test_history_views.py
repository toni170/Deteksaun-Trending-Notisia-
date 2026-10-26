import pandas as pd

from src.dashboard import history_views as hv
from src.scraping.history_scraper import resolve_since

LABELS = {0: "Politika", 1: "Saúde"}


def _df():
    return pd.DataFrame({
        "published_at": ["2016-03-01T10:00:00", "2016-03-01T16:00:00", "2017-07-04T01:00:00", "bad-date"],
        "cluster": [0, 0, 1, 1],
        "source": ["Tatoli"] * 4,
        "category": ["Politika", "Politika", "", None],
        "text_final": ["governu lei governu", "governu parlamentu", "ospitál doutór", "x"],
        "title": list("abcd"), "summary": [""] * 4, "link": ["l"] * 4, "language": ["tet"] * 4,
    })


def test_dili_time_moves_late_utc_to_next_day_and_drops_bad_dates():
    d = hv.with_dili_time(_df())
    assert len(d) == 3                                   # data sala hasai
    assert d["_t"].dt.day.tolist() == [1, 2, 4]          # 16:00 UTC = 01:00 Dili loron tuir mai
    assert d["year"].tolist() == [2016, 2016, 2017]


def test_granularity_and_counts():
    d = hv.with_dili_time(_df())
    assert hv.pick_granularity(d) == "Fulan"
    per = hv.articles_per_period(d, "Tinan")
    assert per["kuantidade"].tolist() == [2, 1]


def test_topic_terms_and_summary():
    d = hv.with_dili_time(_df())
    ty = hv.topic_by_year(d, LABELS)
    assert set(ty["topik"]) == {"Politika", "Saúde"}
    terms = hv.top_terms(d[d["year"] == 2016])
    assert terms.iloc[0]["liafuan"] == "governu" and terms.iloc[0]["kuantidade"] == 3
    s = hv.yearly_summary(d, LABELS)
    assert s["total notísia"].tolist() == [2, 1]
    assert s.loc[0, "tópiku prinsipál"] == "Politika" and s.loc[1, "kategoria prinsipál"] == ""


def test_export_csv_has_bom_and_dili_time():
    data = hv.export_csv_bytes(_df(), LABELS)
    assert data.startswith(b"\xef\xbb\xbf") and b"2016-03-02 01:00" in data


def test_resolve_since(tmp_path):
    assert resolve_since("2016-01-01", str(tmp_path)) == "2016-01-01"
    assert resolve_since("auto", str(tmp_path), default="2016-01-01") == "2016-01-01"   # master seidauk iha
    pd.DataFrame({"published_at": ["2026-09-24T10:00:00", "2020-01-01T00:00:00"]}).to_csv(
        tmp_path / "tatoli_master.csv", index=False)
    assert resolve_since("auto", str(tmp_path)) == "2026-09-22"
