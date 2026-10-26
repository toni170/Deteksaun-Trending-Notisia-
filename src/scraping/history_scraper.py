"""
Koleta notísia ISTÓRIKU (ex: husi 1 Janeiru 2016 to agora).

RSS feed WordPress la fó de'it notísia 10-20 ikus, entaun la bele uza atu foti
notísia tinan hirak liu ona. Modul ne'e uza WordPress REST API
(`<site>/wp-json/wp/v2/posts?after=<data>`) ne'ebé pajina-pajina fó notísia
kompletu (títulu, konteúdu, kategoria, imajen) husi data ne'ebé hili to agora.

Rezultadu iha formatu ne'ebé hanesan ho `rss_scraper.fetch_feed`, entaun bele
tama direta ba `save_raw` (master file + dedup tuir news_id).
"""
import re
import time
import logging
from html import unescape
from urllib.parse import urlparse

import requests

from src.scraping.rss_scraper import (
    _REQUEST_HEADERS, _REQUEST_TIMEOUT, _make_news_id, _strip_html, save_raw,
)
from datetime import datetime

logger = logging.getLogger("history_scraper")

_WS_RE = re.compile(r"\s+")


def _clean_html(text: str) -> str:
    return _WS_RE.sub(" ", unescape(_strip_html(text or ""))).strip()


def resolve_since(since: str, raw_dir: str, default: str = "2016-01-01") -> str:
    """`since="auto"` -> kontinua husi notísia ikus iha master file (-2 loron atu la lakon; dedup
    hasai duplikadu). Se master seidauk iha, uza `default`. Data seluk fo hotu hanesan."""
    import os
    import pandas as pd
    if (since or "").lower() != "auto":
        return since
    master = os.path.join(raw_dir, "tatoli_master.csv")
    if os.path.exists(master):
        latest = pd.to_datetime(pd.read_csv(master, usecols=["published_at"])["published_at"],
                                errors="coerce").max()
        if pd.notna(latest):
            return (latest - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    return default


def _site_root(feed_url: str) -> str:
    p = urlparse(feed_url)
    return f"{p.scheme}://{p.netloc}"


def _post_to_entry(post: dict, source_name: str, fetched_at: str) -> dict:
    link = (post.get("link") or "").strip()
    title = _clean_html((post.get("title") or {}).get("rendered", ""))

    category, image_url = "", ""
    embedded = post.get("_embedded") or {}
    for group in embedded.get("wp:term") or []:
        for term in group or []:
            if isinstance(term, dict) and term.get("taxonomy") == "category":
                category = unescape(term.get("name", ""))
                break
        if category:
            break
    media = embedded.get("wp:featuredmedia") or []
    if media and isinstance(media[0], dict):
        image_url = media[0].get("source_url", "") or ""

    return {
        "news_id": _make_news_id(link, title),
        "source": source_name,
        "title": title,
        "link": link,
        # date_gmt = UTC, hanesan RSS (feedparser) — atu data la kuaze lao-lao
        "published_at": post.get("date_gmt") or post.get("date") or "",
        "summary": _clean_html((post.get("excerpt") or {}).get("rendered", "")),
        "content": _clean_html((post.get("content") or {}).get("rendered", "")),
        "category": category,
        "image_url": image_url,
        "fetched_at": fetched_at,
    }


def _get_page(url: str, params: dict, retries: int, backoff: float):
    headers = {**_REQUEST_HEADERS, "Accept": "application/json"}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return requests.get(url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning("Tentativa %s/%s falha (%s): %s", attempt, retries, url, e)
            time.sleep(backoff * attempt)
    raise ConnectionError(f"La konsege hetan '{url}' depois tentativa {retries}: {last_error}")


def fetch_wp_history(feed_url: str, source_name: str, since: str, until: str = None,
                     per_page: int = 100, max_pages: int = 500,
                     delay_seconds: float = 1.0, retries: int = 3, backoff: float = 2.0) -> list:
    """
    Foti notísia husi `since` (YYYY-MM-DD) to `until` (opsionál, padraun: agora).
    Hanoin: WordPress fó erru 400 bainhira pájina liu pájina ikus — ne'e normal.
    """
    api = _site_root(feed_url) + "/wp-json/wp/v2/posts"
    params = {
        "after": f"{since}T00:00:00",
        "per_page": min(int(per_page), 100),
        "orderby": "date",
        "order": "asc",  # 2016 uluk: se atinje max_pages, notísia foun (ne'ebé RSS mós hetan) mak tesi
        "_embed": "1",
    }
    if until:
        params["before"] = f"{until}T23:59:59"

    fetched_at = datetime.now().isoformat()
    entries, page = [], 1
    while page <= max_pages:
        params["page"] = page
        try:
            resp = _get_page(api, params, retries, backoff)
            if resp.status_code == 400 and page > 1:
                break  # liu pájina ikus
            resp.raise_for_status()
            posts = resp.json()
        except Exception as e:
            if not entries:
                raise  # pájina 1 falha = problema loos (API taka, bloku, ...)
            logger.warning("[%s] para iha pájina %s tanba erru: %s. Rai ona %s notísia.",
                           source_name, page, e, len(entries))
            break
        if not posts:
            break
        entries.extend(_post_to_entry(p, source_name, fetched_at) for p in posts)
        logger.info("[%s] pájina %s: %s notísia (total %s)", source_name, page, len(posts), len(entries))

        total_pages = int(resp.headers.get("X-WP-TotalPages", 0) or 0)
        if total_pages and page >= total_pages:
            break
        page += 1
        time.sleep(delay_seconds)

    return entries


def collect_history_multi(sources: list, raw_dir: str, since: str, until: str = None,
                          per_page: int = 100, max_pages: int = 500,
                          delay_seconds: float = 1.0) -> dict:
    """Koleta istóriku husi orijen aktivu hotu, depois tau iha master file (dedup)."""
    all_entries, per_source = [], {}
    for src in sources:
        name = (src.get("name") or "").strip()
        feed_url = (src.get("feed_url") or "").strip()
        if not src.get("enabled", True) or not feed_url:
            per_source[name] = {"status": "salta (dezativa ka feed_url vazia)", "total": 0}
            continue
        try:
            entries = fetch_wp_history(
                feed_url, name, since, until, per_page, max_pages, delay_seconds,
            )
            all_entries.extend(entries)
            per_source[name] = {"status": "susesu", "total": len(entries)}
        except Exception as e:  # orijen ida falha la para orijen seluk
            logger.warning("Orijen '%s' falha koleta istóriku: %s", name, e)
            per_source[name] = {"status": f"falha: {e}", "total": 0}

    paths = save_raw(all_entries, raw_dir)
    paths["per_source"] = per_source
    return paths
