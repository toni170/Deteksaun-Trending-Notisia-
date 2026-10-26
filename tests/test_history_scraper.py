from unittest.mock import MagicMock

import pandas as pd

from src.scraping import history_scraper as hs


def _post(i, date):
    return {
        "link": f"https://tatoli.tl/post-{i}/",
        "title": {"rendered": f"Notísia &amp; {i}"},
        "date_gmt": date,
        "excerpt": {"rendered": "<p>Rezumu</p>"},
        "content": {"rendered": "<p>Konteúdu <b>kompletu</b></p>"},
        "_embedded": {
            "wp:term": [[{"taxonomy": "category", "name": "Politika"}]],
            "wp:featuredmedia": [{"source_url": "https://tatoli.tl/img.jpg"}],
        },
    }


def _resp(status, body=None, total_pages=2):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body or []
    r.headers = {"X-WP-TotalPages": str(total_pages)}
    r.raise_for_status.side_effect = None if status < 400 else RuntimeError("http error")
    return r


def test_fetch_wp_history_paginates_from_2016(monkeypatch):
    calls = []
    pages = {1: _resp(200, [_post(1, "2026-09-01T00:00:00")]),
             2: _resp(200, [_post(2, "2016-01-05T08:00:00")])}

    def fake_get(url, params=None, **kw):
        calls.append((url, dict(params)))
        return pages[params["page"]]

    monkeypatch.setattr(hs.requests, "get", fake_get)
    monkeypatch.setattr(hs.time, "sleep", lambda s: None)

    entries = hs.fetch_wp_history("https://tatoli.tl/feed/", "Tatoli", "2016-01-01")

    assert [e["published_at"][:4] for e in entries] == ["2026", "2016"]
    assert calls[0][0] == "https://tatoli.tl/wp-json/wp/v2/posts"
    assert calls[0][1]["after"] == "2016-01-01T00:00:00"
    assert calls[0][1]["order"] == "asc"
    e = entries[0]
    assert e["title"] == "Notísia & 1" and e["category"] == "Politika"
    assert e["content"] == "Konteúdu kompletu" and e["image_url"].endswith("img.jpg")


def test_stops_on_400_after_last_page(monkeypatch):
    seq = iter([_resp(200, [_post(1, "2020-01-01T00:00:00")], total_pages=0), _resp(400)])
    monkeypatch.setattr(hs.requests, "get", lambda *a, **k: next(seq))
    monkeypatch.setattr(hs.time, "sleep", lambda s: None)
    assert len(hs.fetch_wp_history("https://x.tl/feed/", "X", "2016-01-01")) == 1


def test_collect_history_multi_keeps_going_when_one_source_fails(monkeypatch, tmp_path):
    def fake_fetch(feed_url, name, *a, **k):
        if name == "Bad":
            raise ConnectionError("down")
        return [hs._post_to_entry(_post(1, "2016-02-02T00:00:00"), name, "now")]

    monkeypatch.setattr(hs, "fetch_wp_history", fake_fetch)
    sources = [{"name": "Bad", "feed_url": "https://bad/feed/", "enabled": True},
               {"name": "Tatoli", "feed_url": "https://tatoli.tl/feed/", "enabled": True}]
    paths = hs.collect_history_multi(sources, str(tmp_path), "2016-01-01")

    assert paths["per_source"]["Bad"]["status"].startswith("falha")
    assert paths["per_source"]["Tatoli"]["total"] == 1
    assert len(pd.read_csv(paths["master_csv"])) == 1


def test_keeps_collected_posts_when_later_page_fails(monkeypatch):
    seq = iter([_resp(200, [_post(1, "2016-01-01T00:00:00")], total_pages=5), ConnectionError("down")])

    def fake_get(*a, **k):
        r = next(seq)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(hs.requests, "get", fake_get)
    monkeypatch.setattr(hs.time, "sleep", lambda s: None)
    assert len(hs.fetch_wp_history("https://tatoli.tl/feed/", "Tatoli", "2016-01-01", retries=1)) == 1
