"""Pipeline preprocessing (DataFrame + CSV) no klasifikadór kategoria — kobre #5/#6 iha nível pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.category_classifier import classify_dataframe_categories
from src.preprocessing.pipeline import preprocess_dataframe, run_preprocessing

pytestmark = pytest.mark.unit


def test_empty_dataframe_returns_expected_columns():
    out = preprocess_dataframe(pd.DataFrame(columns=["title", "summary"]))
    assert out.empty and {"text_clean", "language", "text_final", "token_count"} <= set(out.columns)


def test_rows_that_become_empty_after_cleaning_are_dropped():
    df = pd.DataFrame({"title": ["!!!", "Eskola foun"], "summary": ["???", "estudante iha eskola"]})
    out = preprocess_dataframe(df)
    assert len(out) == 1 and "eskola" in out.loc[0, "text_final"]


def test_all_rows_blank_gives_empty_result_without_crash():
    out = preprocess_dataframe(pd.DataFrame({"title": ["!!!"], "summary": ["???"]}))
    assert out.empty and "category" in out.columns


def test_missing_title_column_is_tolerated():
    out = preprocess_dataframe(pd.DataFrame({"summary": ["estudante eskola ensinu foun"]}))
    assert len(out) == 1


def test_classifier_uses_summary_when_content_is_nan():  # regression
    df = pd.DataFrame({"title": ["x"], "content": [np.nan], "summary": ["eskola estudante ensinu"], "category": [np.nan]})
    assert classify_dataframe_categories(df).tolist() == ["Edukasaun"]


def test_classifier_prefers_known_scraped_category_and_ignores_junk():
    df = pd.DataFrame({"title": ["a", "b"], "content": ["ospital", "ospital mediku vasina"], "category": ["Politics", "Home"]})
    assert classify_dataframe_categories(df).tolist() == ["Politika", "Saúde"]


def test_run_preprocessing_dedups_master_by_news_id(tmp_path):
    raw = pd.DataFrame({
        "news_id": ["a", "a", "b"], "title": ["Eskola foun", "Eskola foun", "Ospital foun"],
        "summary": ["estudante iha eskola", "estudante iha eskola", "mediku iha ospital"],
    })
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)
    master = run_preprocessing(str(raw_path), str(tmp_path / "cleaned"))
    assert sorted(pd.read_csv(master)["news_id"]) == ["a", "b"]


def test_run_preprocessing_without_news_id_column_raises_keyerror(tmp_path):
    """Karakteriza: ficheiru raw la iha `news_id` -> KeyError (la iha mensajen klaru)."""
    p = tmp_path / "raw.csv"
    pd.DataFrame({"title": ["Eskola"], "summary": ["estudante eskola ensinu"]}).to_csv(p, index=False)
    with pytest.raises(KeyError):
        run_preprocessing(str(p), str(tmp_path / "c"))
