"""Pembaca master CSV: limite memória, fallback bainhira MemoryError, tesi tékstu."""
import pandas as pd
import pytest

from src.dashboard import data_loader as dl

pytestmark = pytest.mark.unit


@pytest.fixture
def master(tmp_path):
    n = 5000
    df = pd.DataFrame({
        "news_id": [f"id{i}" for i in range(n)], "title": [f"t{i}" for i in range(n)],
        "content": ["x" * 9000] * n, "text_clean": ["y" * 5000] * n, "text_final": ["z"] * n,
        "summary": ["s" * 3000] * n, "fetched_at": ["2024-01-01"] * n, "token_count": [3] * n,
        "cluster": [i % 3 for i in range(n)],
    })
    path = tmp_path / "master.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_full_read_drops_unused_columns_and_truncates_text(master):
    df = dl.read_cleaned_light(master, chunksize=700)
    assert len(df) == 5000 and "rows_total" not in df.attrs
    assert not ({"fetched_at", "token_count"} & set(df.columns))
    assert df["content"].str.len().max() == 6000 and df["text_clean"].str.len().max() == 3000
    assert df["summary"].str.len().max() == 2000


def test_max_rows_keeps_only_newest_rows_in_file_order(master):
    df = dl.read_cleaned_light(master, chunksize=700, max_rows=1000)
    assert len(df) == 1000 and df.attrs["rows_total"] == 5000
    assert df["title"].iloc[-1] == "t4999" and df["title"].iloc[0] == "t4000"
    assert df["title"].tolist() == [f"t{i}" for i in range(4000, 5000)]


def test_max_rows_larger_than_file_changes_nothing(master):
    df = dl.read_cleaned_light(master, chunksize=700, max_rows=10**6)
    assert len(df) == 5000 and "rows_total" not in df.attrs


def test_empty_csv_with_header_only_returns_empty_frame(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("title,content\n", encoding="utf-8")
    assert dl.read_cleaned_light(str(p)).empty


def test_fallback_returns_everything_when_memory_is_enough(master):
    assert len(dl.read_cleaned_with_fallback(master)) == 5000


def test_fallback_retries_with_smaller_limits_on_memory_error(master, monkeypatch):
    real, calls = dl.read_cleaned_light, []

    def flaky(path, chunksize=2000, max_rows=None):
        calls.append(max_rows)
        if max_rows is None or max_rows >= 50_000:
            raise MemoryError
        return real(path, chunksize=700, max_rows=max_rows)

    monkeypatch.setattr(dl, "read_cleaned_light", flaky)
    monkeypatch.setattr(dl, "FALLBACK_ROW_LIMITS", (100_000, 50_000, 2_000, 1_000))
    df = dl.read_cleaned_with_fallback(master)
    assert calls == [None, 100_000, 50_000, 2_000]
    assert len(df) == 2000 and df.attrs["rows_total"] == 5000


def test_fallback_raises_memory_error_when_nothing_fits(master, monkeypatch):
    def always(*a, **k):
        raise MemoryError
    monkeypatch.setattr(dl, "read_cleaned_light", always)
    with pytest.raises(MemoryError):
        dl.read_cleaned_with_fallback(master)


def test_collected_data_exists(tmp_path):
    assert not dl.collected_data_exists(str(tmp_path / "la_existe.csv"))
    empty = tmp_path / "e.csv"; empty.write_text("", encoding="utf-8")
    header_only = tmp_path / "h.csv"; header_only.write_text("title,content\n", encoding="utf-8")
    rows = tmp_path / "r.csv"; rows.write_text("title,content\nA,b\n", encoding="utf-8-sig")
    assert not dl.collected_data_exists(str(empty)) and not dl.collected_data_exists(str(header_only))
    assert dl.collected_data_exists(str(rows))
