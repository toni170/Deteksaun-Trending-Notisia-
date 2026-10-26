"""save_raw: `new_entries` tenke reprezenta notísia FOUN loloos (la repete), la'ós kuantidade ne'ebé foti."""
import pandas as pd
import pytest

from src.scraping.rss_scraper import RAW_FIELDS, save_raw

pytestmark = pytest.mark.unit


def entry(i: str, title: str = "t") -> dict:
    e = {f: "" for f in RAW_FIELDS}
    e.update(news_id=i, title=f"{title}{i}", link=f"https://x.test/{i}", source="Tatoli")
    return e


def master_ids(raw_dir):
    return pd.read_csv(f"{raw_dir}/tatoli_master.csv", dtype={"news_id": str})["news_id"].tolist()


def test_first_collection_counts_all_as_new(tmp_path):
    out = save_raw([entry("a"), entry("b"), entry("c")], str(tmp_path))
    assert out["new_entries"] == 3 and out["fetched_entries"] == 3 and out["total_unique"] == 3


def test_second_identical_collection_has_zero_new_entries(tmp_path):
    batch = [entry("a"), entry("b"), entry("c")]
    save_raw(batch, str(tmp_path))
    out = save_raw(batch, str(tmp_path))
    assert out["new_entries"] == 0          # <- sasan ne'ebé user husu: "dadus iha ona"
    assert out["fetched_entries"] == 3 and out["total_unique"] == 3
    assert sorted(master_ids(tmp_path)) == ["a", "b", "c"]


def test_partial_overlap_counts_only_really_new(tmp_path):
    save_raw([entry("a"), entry("b")], str(tmp_path))
    out = save_raw([entry("b"), entry("c"), entry("d")], str(tmp_path))
    assert out["new_entries"] == 2 and out["fetched_entries"] == 3 and out["total_unique"] == 4


def test_duplicates_inside_one_batch_count_once(tmp_path):
    out = save_raw([entry("a"), entry("a"), entry("b")], str(tmp_path))
    assert out["new_entries"] == 2 and out["total_unique"] == 2


def test_numeric_looking_ids_are_not_recounted_as_new(tmp_path):
    """news_id hexadesimál bele parece numéru ("0000000000000001"): la bele lakon zero bainhira lee master."""
    batch = [entry("0000000000000001"), entry("0000000000000002")]
    save_raw(batch, str(tmp_path))
    out = save_raw(batch, str(tmp_path))
    assert out["new_entries"] == 0 and out["total_unique"] == 2
    assert sorted(master_ids(tmp_path)) == ["0000000000000001", "0000000000000002"]


def test_empty_fetch_is_zero_new(tmp_path):
    save_raw([entry("a")], str(tmp_path))
    assert save_raw([], str(tmp_path))["new_entries"] == 0
