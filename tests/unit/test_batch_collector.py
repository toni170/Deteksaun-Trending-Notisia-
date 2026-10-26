"""Teste koletór lote: limite tamañu, la repete entre lote, kursór tuan/foun, formatu tolu."""
import json
import os
import sqlite3
import sys
import threading
import types
from datetime import datetime, timedelta

import pandas as pd
import pytest

if "feedparser" not in sys.modules:  # ba ambiente la iha feedparser; koletór la uza iha teste ne'e
    try:
        import feedparser  # noqa: F401
    except ImportError:
        sys.modules["feedparser"] = types.ModuleType("feedparser")

from src.collection.batch_collector import collect_batch, list_batches, load_state  # noqa: E402

BASE = datetime(2024, 1, 1, 8, 0, 0)


def make_posts(n, start=0, same_second_every=0):
    posts = []
    for i in range(start, start + n):
        local = BASE + timedelta(minutes=i)
        if same_second_every and i % same_second_every:
            local = BASE + timedelta(minutes=i - i % same_second_every)
        posts.append({
            "id": i, "link": f"https://t.test/p/{i}",
            "date": local.strftime("%Y-%m-%dT%H:%M:%S"),
            "date_gmt": (local - timedelta(hours=9)).strftime("%Y-%m-%dT%H:%M:%S"),
            "title": {"rendered": f"Notísia {i}"},
            "excerpt": {"rendered": "<p>rezumu</p>"},
            "content": {"rendered": "<p>" + ("konteúdu naruk " * 40) + f" {i}</p>"},
        })
    return posts


class FakeWP:
    """Simula /wp-json/wp/v2/posts ho `after`, `before`, `order`, `page`, `per_page`."""

    def __init__(self, posts):
        self.posts = posts
        self.calls = 0

    def __call__(self, api, params):
        self.calls += 1
        items = list(self.posts)
        if params.get("after"):
            items = [p for p in items if p["date"] > params["after"]]
        if params.get("before"):
            items = [p for p in items if p["date"] < params["before"]]
        items.sort(key=lambda p: (p["date"], p["id"]), reverse=params.get("order") == "desc")
        per, page = int(params["per_page"]), int(params["page"])
        total = max(1, -(-len(items) // per)) if items else 0
        return items[(page - 1) * per: page * per], total


SRC = [{"name": "Tatoli", "feed_url": "https://t.test/feed/", "enabled": True}]


def run(tmp_path, wp, **kw):
    kw.setdefault("target_mb", 100)
    kw.setdefault("per_page", 10)
    kw.setdefault("delay_seconds", 0)
    return collect_batch(SRC, str(tmp_path), fetch_page=wp, **kw)


def ids_of(result):
    return set(pd.read_csv(result.files["csv"])["link"])


def test_batch_stops_at_size_cap_and_never_exceeds_it(tmp_path):
    wp = FakeWP(make_posts(300))
    cap_mb = 0.05  # ~52 KB
    r = run(tmp_path, wp, target_mb=cap_mb)
    assert r.reached_target and r.status == "remata"
    assert 0 < r.rows < 300
    assert os.path.getsize(r.files["csv"]) <= cap_mb * 1024 * 1024
    assert r.csv_bytes == os.path.getsize(r.files["csv"])
    assert r.csv_bytes > 0.9 * cap_mb * 1024 * 1024  # besik meta (liña ikus la kabe)


def test_three_formats_have_same_rows(tmp_path):
    r = run(tmp_path, FakeWP(make_posts(25)))
    n_csv = len(pd.read_csv(r.files["csv"]))
    n_json = len(json.load(open(r.files["json"], encoding="utf-8")))
    n_db = sqlite3.connect(r.files["sqlite"]).execute("select count(*) from articles").fetchone()[0]
    assert n_csv == n_json == n_db == r.rows == 25
    assert list(pd.read_csv(r.files["csv"]).columns)[:3] == ["news_id", "source", "title"]
    assert os.path.isfile(os.path.join(r.directory, "meta.json"))


def test_second_request_continues_without_duplicates_or_gaps(tmp_path):
    wp = FakeWP(make_posts(120))
    r1 = run(tmp_path, wp, target_mb=0.03)
    r2 = run(tmp_path, wp, target_mb=0.03)
    r3 = run(tmp_path, wp, target_mb=0.03)
    a, b, c = ids_of(r1), ids_of(r2), ids_of(r3)
    assert not (a & b) and not (a & c) and not (b & c)  # lote la hamutuk, la repete
    assert r1.directory != r2.directory
    # kontinuasaun: tuan liu iha lote anterior; la iha burako
    allseen = a | b | c
    nums = sorted(int(u.rsplit("/", 1)[1]) for u in allseen)
    assert nums == list(range(120 - len(nums), 120))


def test_exhaustion_then_only_new_posts_are_fetched(tmp_path):
    posts = make_posts(40)
    wp = FakeWP(posts)
    r1 = run(tmp_path, wp)
    assert r1.rows == 40 and not r1.reached_target and r1.status == "remata"
    assert load_state(str(tmp_path))["Tatoli"]["exhausted"] is True

    r2 = run(tmp_path, wp)  # la iha foun -> lote mamuk, folder la sobra
    assert r2.status == "mamuk" and r2.rows == 0 and not os.path.exists(r2.directory)

    wp.posts = posts + make_posts(5, start=40)  # notísia foun mosu
    r3 = run(tmp_path, wp)
    assert r3.rows == 5
    assert all(int(u.rsplit("/", 1)[1]) >= 40 for u in ids_of(r3))


def test_same_timestamp_at_cursor_boundary_loses_nothing(tmp_path):
    wp = FakeWP(make_posts(60, same_second_every=7))
    got = set()
    for _ in range(8):
        r = run(tmp_path, wp, target_mb=0.02)
        if r.rows == 0:
            break
        got |= ids_of(r)
    assert len(got) == 60


def test_stop_event_keeps_collected_rows(tmp_path):
    stop = threading.Event()
    wp = FakeWP(make_posts(100))

    def fetch(api, params):
        out = wp(api, params)
        if wp.calls == 2:
            stop.set()
        return out
    r = collect_batch(SRC, str(tmp_path), fetch_page=fetch, per_page=10, delay_seconds=0, stop_event=stop)
    assert r.status == "para" and 0 < r.rows < 100
    assert os.path.isfile(r.files["csv"])


def test_source_failure_does_not_stop_other_sources(tmp_path):
    good = FakeWP(make_posts(15))

    def fetch(api, params):
        if "bad.test" in api:
            raise ConnectionError("server down")
        return good(api, params)
    sources = [{"name": "Bad", "feed_url": "https://bad.test/feed/"}, SRC[0]]
    r = collect_batch(sources, str(tmp_path), fetch_page=fetch, per_page=10, delay_seconds=0,
                      rss_fetch=lambda *a: (_ for _ in ()).throw(RuntimeError("RSS mós falla")))
    assert r.rows == 15
    assert r.per_source["Bad"]["status"].startswith("falla")
    assert r.per_source["Tatoli"]["status"] == "susesu"


def test_rss_fallback_when_wordpress_api_missing(tmp_path):
    def fetch(api, params):
        raise ConnectionError("404")
    entries = [{"news_id": f"r{i}", "source": "Tatoli", "title": f"RSS {i}", "link": f"https://t.test/r/{i}",
                "published_at": "2024-01-01T00:00:00", "summary": "x", "content": "", "category": "",
                "image_url": "", "fetched_at": ""} for i in range(3)]
    r = collect_batch(SRC, str(tmp_path), fetch_page=fetch, delay_seconds=0, rss_fetch=lambda u, n: entries)
    assert r.rows == 3


def test_no_active_source_returns_tetun_error(tmp_path):
    r = collect_batch([{"name": "X", "feed_url": "", "enabled": True}], str(tmp_path))
    assert r.status == "falla" and "orijen" in r.message.lower()


def test_list_batches_only_completed(tmp_path):
    run(tmp_path, FakeWP(make_posts(12)))
    os.makedirs(tmp_path / "batch_incompleto")
    lst = list_batches(str(tmp_path))
    assert len(lst) == 1 and lst[0]["rows"] == 12
