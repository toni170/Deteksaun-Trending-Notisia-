"""Teste preprosesamentu ho etapa: pariedade ho pipeline antigu, formatu tolu, lote, input CSV/JSON/SQLite."""
import json
import os
import random
import sqlite3
import sys
import threading
import types

import pandas as pd
import pytest

if "feedparser" not in sys.modules:
    try:
        import feedparser  # noqa: F401
    except ImportError:
        sys.modules["feedparser"] = types.ModuleType("feedparser")

from src.ingestion.dataset_importer import DatasetImportError  # noqa: E402
from src.preprocessing.staged_pipeline import (  # noqa: E402
    STAGES, detect_language_tokens, list_clean_datasets, run_staged_preprocessing, tokenize,
)
from src.preprocessing.text_cleaner import preprocess_news  # noqa: E402
from src.scraping.rss_scraper import RAW_FIELDS  # noqa: E402

FILL = ["iha", "ba", "husi", "ne'e", "atu", "nia", "sira"]
VOCAB = {
    "edukasaun": "eskola estudante ensinu prufesor kurrikulu universidade exame matrikula alunu".split(),
    "saude": "ospital mediku vasina moras klinika enfermeira tratamentu pasiente farmasia".split(),
    "ekonomia": "banku orsamentu investimentu merkadu presu petroleu kompania finansa negosiu".split(),
}


def make_raw(n_per=10, seed=1):
    rng = random.Random(seed)
    rows, i = [], 0
    for topic, vocab in VOCAB.items():
        for _ in range(n_per):
            words = rng.sample(vocab, 6) + rng.sample(FILL, 3)
            rng.shuffle(words)
            rows.append({
                "news_id": f"id{i}", "source": "Tatoli", "title": f"{topic.title()} {i}: " + " ".join(words[:3]),
                "link": f"https://t.test/{i}", "published_at": f"2024-0{1 + i % 5}-1{i % 9}T08:00:00",
                "summary": "<p>rezumu &amp; ki'ik</p>",
                "content": "<p>" + " ".join(words) + f" http://x.test/a?b=1 {topic} {topic}</p>",
                "category": "", "image_url": "", "fetched_at": "2024-01-01T00:00:00"})
            i += 1
    return pd.DataFrame(rows, columns=RAW_FIELDS)


def write_csv(df, tmp_path, name="raw.csv"):
    path = str(tmp_path / name)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_tokenize_keeps_accents_and_apostrophes():
    assert tokenize("saúde polítika ne'e 'ha'u' ministériu!!!") == ["saúde", "polítika", "ne'e", "ha'u", "ministériu"]


def test_language_detection_matches_legacy_rule():
    assert detect_language_tokens("iha ba husi eskola".split()) == "tet"
    assert detect_language_tokens("the government and the minister".split()) == "en"
    assert detect_language_tokens("xyz abc".split()) == "unknown"
    assert detect_language_tokens([]) == "unknown"


def test_text_final_matches_legacy_pipeline_on_plain_text(tmp_path):
    df = make_raw(5)
    df["summary"] = ""
    df["content"] = ["iha " + " ".join(VOCAB["edukasaun"][:5]) + f" sira nia {i} ba atu" for i in range(len(df))]
    df["title"] = [f"Eskola estudante {i}" for i in range(len(df))]
    out = run_staged_preprocessing(write_csv(df, tmp_path), str(tmp_path / "bersih"))
    got = pd.read_csv(out["files"]["csv"], dtype={"news_id": str}).set_index("news_id")
    for _, row in df.iterrows():
        legacy = preprocess_news(row["title"], row["content"])
        assert got.loc[row["news_id"], "text_final"] == legacy["text_final"]
        assert got.loc[row["news_id"], "language"] == legacy["language"]


def test_full_run_saves_three_formats_and_reports_every_stage(tmp_path):
    df = make_raw(10)
    df = pd.concat([df, df.head(3), pd.DataFrame([{c: "" for c in RAW_FIELDS}])], ignore_index=True)  # duplikadu + mamuk
    events = []
    out = run_staged_preprocessing(write_csv(df, tmp_path), str(tmp_path / "bersih"), chunk_rows=7,
                                   progress=lambda f, m, **e: events.append((f, m, e)))
    assert out["status"] == "remata" and out["rows_read"] == 34
    assert out["rows_saved"] == 30 and out["removed_duplicate"] == 3 and out["removed_empty"] == 1

    f = out["files"]
    n_csv = len(pd.read_csv(f["csv"]))
    n_json = len(json.load(open(f["json"], encoding="utf-8")))
    n_db = sqlite3.connect(f["sqlite"]).execute("select count(*) from articles").fetchone()[0]
    assert n_csv == n_json == n_db == 30

    cols = pd.read_csv(f["csv"], nrows=1).columns
    for c in ["text_clean", "text_final", "language", "category", "token_count", "published_at"]:
        assert c in cols
    assert [s["key"] for s in out["stages"]] == [k for k, _ in STAGES]
    assert all(s["status"] == "remata" and s["detail"] for s in out["stages"])
    assert out["stages"][2]["rows"] == 30 or out["stages"][2]["rows"] > 0
    assert out["languages"].get("tet", 0) == 30
    # progresu iha eventu: aumenta no iha tabela etapa
    fr = [e[0] for e in events if e[0] is not None]
    assert fr == sorted(fr) and fr[-1] == 1.0
    assert any(e[2].get("stages") for e in events)
    # amostra kada etapa
    for key in ["cleaning", "case_folding", "tokenization", "stopwords", "stemming"]:
        assert out["samples"][key]
    assert "http" not in out["samples"]["cleaning"] and "&amp;" not in out["samples"]["cleaning"]
    assert out["samples"]["case_folding"] == out["samples"]["case_folding"].lower()


def test_html_url_entities_removed_and_clean_text_is_lowercase(tmp_path):
    out = run_staged_preprocessing(write_csv(make_raw(4), tmp_path), str(tmp_path / "b"))
    d = pd.read_csv(out["files"]["csv"])
    assert not d["text_clean"].str.contains("http|<|&amp;|amp", regex=True).any()
    assert (d["text_clean"] == d["text_clean"].str.lower()).all()
    assert (d["text_final"].str.split().str.len() == d["token_count"]).all()


def test_chunk_size_does_not_change_result(tmp_path):
    path = write_csv(make_raw(10), tmp_path)
    a = run_staged_preprocessing(path, str(tmp_path / "a"), chunk_rows=5000)
    b = run_staged_preprocessing(path, str(tmp_path / "b"), chunk_rows=4)
    da, db = pd.read_csv(a["files"]["csv"]), pd.read_csv(b["files"]["csv"])
    pd.testing.assert_frame_equal(da, db)


def test_json_and_sqlite_inputs_give_same_result_as_csv(tmp_path):
    from src.storage.multi_writer import MultiFormatWriter
    df = make_raw(6)
    with MultiFormatWriter(str(tmp_path / "src"), "notisia", RAW_FIELDS, key_column="news_id") as w:
        w.write_rows(df.to_dict("records"))
    res = {}
    for kind in ("csv", "json", "sqlite"):
        res[kind] = run_staged_preprocessing(w.paths[kind], str(tmp_path / f"out_{kind}"))
    base = pd.read_csv(res["csv"]["files"]["csv"])
    for kind in ("json", "sqlite"):
        pd.testing.assert_frame_equal(base, pd.read_csv(res[kind]["files"]["csv"]))


def test_errors_are_tetun_and_leave_no_partial_folder(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("title,date\nA,2024-01-01\n", encoding="utf-8")
    out = str(tmp_path / "out")
    with pytest.raises(DatasetImportError) as e:
        run_staged_preprocessing(str(bad), out)
    assert "konteúdu" in e.value.message
    assert not os.path.exists(out) or not os.listdir(out)

    with pytest.raises(DatasetImportError) as e:
        run_staged_preprocessing(str(tmp_path / "nada.csv"), out)
    assert "la hetan" in e.value.message
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    with pytest.raises(DatasetImportError) as e:
        run_staged_preprocessing(str(txt), out)
    assert "la suporta" in e.value.message


def test_stop_removes_partial_output(tmp_path):
    stop = threading.Event()
    calls = []

    def prog(f, m, **e):
        calls.append(m)
        if len(calls) > 3:
            stop.set()
    out = run_staged_preprocessing(write_csv(make_raw(10), tmp_path), str(tmp_path / "b"), chunk_rows=3,
                                   progress=prog, stop_event=stop)
    assert out["status"] == "para"
    assert list_clean_datasets(str(tmp_path / "b")) == []
    assert not any(os.path.isdir(os.path.join(tmp_path / "b", d)) for d in os.listdir(tmp_path / "b"))


def test_list_clean_datasets_and_same_name_twice(tmp_path):
    path = write_csv(make_raw(3), tmp_path)
    a = run_staged_preprocessing(path, str(tmp_path / "b"))
    b = run_staged_preprocessing(path, str(tmp_path / "b"))
    assert a["directory"] != b["directory"]
    assert len(list_clean_datasets(str(tmp_path / "b"))) == 2
