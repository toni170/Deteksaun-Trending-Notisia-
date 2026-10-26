"""Teste #1-#6: importa CSV/Excel, ficheiru inválidu, dataset mamuk, kolona falta, hamoos."""
import io

import pandas as pd
import pytest

from src.ingestion.dataset_importer import DatasetImportError, import_dataset
from src.scraping.rss_scraper import RAW_FIELDS

pytestmark = pytest.mark.unit


def _csv(text: str) -> io.BytesIO:
    return io.BytesIO(text.encode("utf-8"))


# ---- #1 CSV válidu ---------------------------------------------------------
def test_01_valid_csv_from_bytes(csv_bytes, news_df):
    result = import_dataset(io.BytesIO(csv_bytes), "berita.csv")
    df, rep = result.df, result.report
    assert list(df.columns) == RAW_FIELDS
    assert len(df) == len(news_df) == rep.rows_valid == rep.rows_read
    assert df["news_id"].is_unique
    assert (df["source"] == "berita").all()  # default = naran ficheiru
    assert df["published_at"].str.endswith("+00:00").all()
    assert rep.rows_removed_empty == 0 and rep.rows_removed_duplicate == 0


def test_01_valid_csv_from_path(tmp_path, csv_bytes, news_df):
    path = tmp_path / "x.csv"
    path.write_bytes(csv_bytes)
    assert len(import_dataset(str(path)).df) == len(news_df)


def test_01_semicolon_csv_with_tetun_headers_and_accents():
    text = "Títulu;Konteúdu;Data\nEskola foun;Estudante iha eskola;2024-05-01\n"
    result = import_dataset(_csv(text), "t.csv")
    assert result.report.mapped_columns == {"title": "Títulu", "content": "Konteúdu", "published_at": "Data"}
    assert result.df.loc[0, "title"] == "Eskola foun"


def test_01_cp1252_encoded_csv_is_read():
    raw = "title,content\nSaúde,Ospital foun iha Dili\n".encode("cp1252")
    assert import_dataset(io.BytesIO(raw), "latin.csv").df.loc[0, "title"] == "Saúde"


# ---- #2 Excel válidu -------------------------------------------------------
def test_02_valid_xlsx(xlsx_bytes, news_df):
    result = import_dataset(io.BytesIO(xlsx_bytes), "berita.xlsx")
    assert result.report.file_type == "xlsx"
    assert len(result.df) == len(news_df)
    assert list(result.df.columns) == RAW_FIELDS
    assert result.df["published_at"].ne("").all()


def test_02_legacy_xls_is_rejected_with_clear_message():
    with pytest.raises(DatasetImportError) as exc:
        import_dataset(io.BytesIO(b"whatever"), "old.xls")
    assert exc.value.code == "unsupported_format" and ".xlsx" in exc.value.message


# ---- #3 Ficheiru inválidu --------------------------------------------------
@pytest.mark.parametrize("name,data,code", [
    ("nota.txt", b"title,content\na,b\n", "unsupported_format"),
    ("rusak.xlsx", b"ini bukan zip", "unreadable"),
    ("biner.csv", b"\x00\x01\x02\xffbinary", "unreadable"),
    ("kosong.csv", b"", "empty_file"),
    ("spasi.csv", b"   \n\n", "empty_file"),
])
def test_03_invalid_files(name, data, code):
    with pytest.raises(DatasetImportError) as exc:
        import_dataset(io.BytesIO(data), name)
    assert exc.value.code == code
    assert exc.value.message  # mensajen Tetun, la mamuk


def test_03_nonexistent_path():
    with pytest.raises(DatasetImportError) as exc:
        import_dataset("/tmp/la-existe-ida-ne'e.csv")
    assert exc.value.code == "not_found"


# ---- #4 Dataset mamuk ------------------------------------------------------
def test_04_header_only_is_empty_dataset():
    with pytest.raises(DatasetImportError) as exc:
        import_dataset(_csv("title,content\n"), "h.csv")
    assert exc.value.code == "empty_dataset"


def test_04_all_rows_blank_is_no_valid_rows():
    with pytest.raises(DatasetImportError) as exc:
        import_dataset(_csv("title,content\n,\n  ,  \nnan,N/A\n"), "b.csv")
    assert exc.value.code == "no_valid_rows"


# ---- #5 Kolona konteúdu falta ----------------------------------------------
def test_05_missing_content_column():
    with pytest.raises(DatasetImportError) as exc:
        import_dataset(_csv("title,date\nEskola,2024-01-01\n"), "m.csv")
    assert exc.value.code == "missing_columns"
    assert exc.value.missing == ["content"]
    assert "konteúdu" in exc.value.message


def test_05_missing_title_and_content_reports_both():
    with pytest.raises(DatasetImportError) as exc:
        import_dataset(_csv("foo,bar\n1,2\n"), "m.csv")
    assert exc.value.missing == ["title", "content"]


def test_05_summary_column_is_accepted_instead_of_content():
    result = import_dataset(_csv("judul,ringkasan\nSekolah baru,Mahasiswa di kampus\n"), "s.csv")
    assert result.df.loc[0, "summary"] == "Mahasiswa di kampus" and result.df.loc[0, "content"] == ""


# ---- #6 Hamoos duplikadu no valór mamuk ------------------------------------
def test_06_removes_duplicates_and_empty_values():
    text = (
        "title,content,link\n"
        "Eskola foun,Estudante iha eskola,http://a/1\n"
        "Beibeik link,Konteúdu seluk,http://a/1\n"           # link hanesan -> duplikadu
        "Ospital  FOUN,Mediku  iha ospital,\n"
        "ospital foun,mediku iha ospital,\n"                 # títulu+konteúdu hanesan (maiúskula/espasu)
        ",,\n"                                              # mamuk
        "   ,   ,   \n"                                     # espasu de'it
        "nan,N/A,null\n"                                    # token 'mamuk'
        "Banku,nan,\n"                                      # títulu de'it -> mantein
    )
    result = import_dataset(_csv(text), "d.csv")
    rep = result.report
    assert rep.rows_read == 8
    assert rep.rows_removed_empty == 3
    assert rep.rows_removed_duplicate == 2
    assert rep.rows_valid == len(result.df) == 3
    assert result.df["news_id"].is_unique
    assert not result.df[["title", "content"]].isin(["nan", "NaN", "None", "null"]).any().any()
    assert result.df.loc[result.df["title"] == "Banku", "content"].iloc[0] == ""


def test_06_invalid_dates_become_empty_and_are_counted():
    result = import_dataset(_csv("title,content,date\nA,b,2024-01-05\nC,d,bukan-tanggal\nE,f,\n"), "d.csv")
    assert result.report.invalid_dates == 1
    assert result.df["published_at"].tolist()[1:] == ["", ""]
