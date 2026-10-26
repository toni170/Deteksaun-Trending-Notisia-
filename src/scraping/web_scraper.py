"""
Web scraper artikel LENGKAP Tatoli — melengkapi RSS Collection.

RSS feed Tatoli hanya memberi RINGKASAN singkat per berita. Modul ini memakai
daftar artikel dari RSS (`rss_scraper.fetch_feed`) sebagai sumber DISCOVERY
(judul, link, tanggal awal, ringkasan), lalu mengunjungi setiap halaman
artikel satu per satu untuk menarik:
- Isi artikel LENGKAP (bukan cuma ringkasan)
- KATEGORI berita
- URL gambar THUMBNAIL/cover

...memakai `requests` + `BeautifulSoup`.

Karena struktur HTML persis situs Tatoli bisa berubah sewaktu-waktu (ganti
tema WordPress, dsb), daftar CSS selector kandidat untuk tiap elemen (konten,
kategori, tanggal, gambar) dikonfigurasi di `config/settings.yaml` (bagian
`scraping.selectors`) — bisa disesuaikan tanpa mengubah kode sama sekali.

Etika scraping: modul ini menambahkan jeda (`delay_seconds`) antar request ke
halaman artikel, dan HANYA men-scrape artikel yang belum pernah punya konten
lengkap tersimpan (tidak scrape ulang artikel lama setiap kali dijalankan).
"""
import os
import time
import logging

import requests
import pandas as pd
from bs4 import BeautifulSoup

from src.scraping.rss_scraper import (
    fetch_feed, save_raw, _REQUEST_HEADERS, _REQUEST_TIMEOUT,
)

logger = logging.getLogger("web_scraper")

# Selector kandidat default (dipakai jika config/settings.yaml tidak
# menyediakan override di bagian `scraping.selectors`).
DEFAULT_SELECTORS = {
    "content": [
        "div.entry-content", "div.post-content", "div.td-post-content",
        "div.single-content", "div.article-content", "article .content",
        "article",
    ],
    "category": [
        "span.cat-links a", "div.cat-links a", ".breadcrumb a:last-of-type",
        "a[rel='category tag']", ".post-category a",
    ],
    "date_meta": [
        "meta[property='article:published_time']",
        "meta[name='article:published_time']",
    ],
    "date_tag": ["time[datetime]", "time"],
    "image_meta": ["meta[property='og:image']", "meta[name='twitter:image']"],
}

# Elemen "sampah" umum yang sering ikut ke-scrape bersama konten artikel
# (tombol share, artikel terkait, iklan) — dibuang sebelum ekstraksi teks.
_JUNK_SELECTORS = (
    "script, style, .sharedaddy, .related-posts, .jp-relatedposts, "
    ".advertisement, .ads, nav, footer, .social-share, .comments"
)


def _get_text_by_selectors(soup, selectors, min_len=1):
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(" ", strip=True)
            if len(text) >= min_len:
                return text
    return ""


def _get_attr_by_selectors(soup, selectors, attr):
    for sel in selectors:
        el = soup.select_one(sel)
        if el and el.has_attr(attr):
            return el[attr]
    return ""


def _extract_content(soup, selectors):
    """
    Coba tiap selector kandidat secara berurutan; pilih hasil yang
    menghasilkan teks TERPANJANG (heuristik sederhana untuk memilih
    container konten artikel yang paling lengkap/benar).
    """
    best_text = ""
    for sel in selectors:
        el = soup.select_one(sel)
        if not el:
            continue

        el_copy = BeautifulSoup(str(el), "lxml")
        for junk in el_copy.select(_JUNK_SELECTORS):
            junk.decompose()

        paragraphs = el_copy.find_all("p")
        if paragraphs:
            text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        else:
            text = el_copy.get_text(" ", strip=True)

        if len(text) > len(best_text):
            best_text = text

    return best_text.strip()


def scrape_article(url: str, selectors: dict = None, timeout: int = None) -> dict:
    """
    Ambil isi lengkap satu halaman artikel: konten penuh, kategori, gambar
    thumbnail, dan (fallback) judul/tanggal dari meta tag Open Graph.
    """
    selectors = {**DEFAULT_SELECTORS, **(selectors or {})}
    timeout = timeout or _REQUEST_TIMEOUT

    response = requests.get(url, headers=_REQUEST_HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")

    content = _extract_content(soup, selectors["content"])
    category = _get_text_by_selectors(soup, selectors["category"])
    image_url = _get_attr_by_selectors(soup, selectors["image_meta"], "content")

    published_meta = _get_attr_by_selectors(soup, selectors["date_meta"], "content")
    published_tag = _get_attr_by_selectors(soup, selectors["date_tag"], "datetime")
    published_fallback = published_meta or published_tag

    return {
        "content": content,
        "category": category,
        "image_url": image_url,
        "published_fallback": published_fallback,
    }


def _load_scraped_ids(raw_dir: str) -> set:
    """Cek master raw file; kembalikan set news_id yang SUDAH punya konten lengkap."""
    master_path = os.path.join(raw_dir, "tatoli_master.csv")
    if not os.path.exists(master_path):
        return set()
    try:
        df = pd.read_csv(master_path)
    except Exception:
        return set()
    if "content" not in df.columns:
        return set()
    has_content = df[df["content"].fillna("").astype(str).str.len() > 50]
    return set(has_content["news_id"].tolist())


def collect_full_articles(feed_url: str, raw_dir: str, source_name: str = "Tatoli",
                           max_articles: int = None, delay_seconds: float = 1.5,
                           selectors: dict = None) -> dict:
    """
    Pipeline koleksi lengkap:
    1. Discovery: ambil daftar artikel terbaru dari RSS.
    2. Scraping: untuk artikel yang BELUM punya konten lengkap tersimpan,
       kunjungi halamannya (dengan jeda `delay_seconds` antar request) dan
       tarik isi lengkap + kategori + gambar.
    3. Simpan gabungan (lama + baru) ke data/raw, dedup berdasarkan news_id.
    """
    entries = fetch_feed(feed_url, source_name=source_name)
    already_scraped = _load_scraped_ids(raw_dir)

    to_scrape = [e for e in entries if e["news_id"] not in already_scraped]
    if max_articles:
        to_scrape = to_scrape[:max_articles]

    logger.info(
        "Discovery: %s entri dari RSS, %s di antaranya baru & akan di-scrape penuh.",
        len(entries), len(to_scrape),
    )

    scraped_count, failed_count = 0, 0
    for entry in to_scrape:
        try:
            details = scrape_article(entry["link"], selectors=selectors)
            entry["content"] = details["content"] or entry.get("summary", "")
            entry["category"] = details["category"]
            entry["image_url"] = details["image_url"]
            if not entry.get("published_at") and details["published_fallback"]:
                entry["published_at"] = details["published_fallback"]
            scraped_count += 1
            logger.info(
                "Scraped [%s/%s]: %s (%s karakter konten)",
                scraped_count, len(to_scrape), entry["title"][:60], len(entry["content"]),
            )
        except requests.exceptions.RequestException as e:
            failed_count += 1
            logger.warning(
                "Gagal scraping artikel '%s': %s. Fallback ke ringkasan RSS.",
                entry.get("link", ""), e,
            )
            entry["content"] = entry.get("summary", "")
            entry["category"] = ""
            entry["image_url"] = ""
        time.sleep(delay_seconds)

    # Entri lama (sudah pernah di-scrape) atau yang gagal tidak punya field baru;
    # pastikan semua entri punya kolom konsisten sebelum digabung & disimpan.
    for entry in entries:
        entry.setdefault("content", entry.get("summary", ""))
        entry.setdefault("category", "")
        entry.setdefault("image_url", "")

    paths = save_raw(entries, raw_dir)
    paths["scraped_new_articles"] = scraped_count
    paths["failed_articles"] = failed_count
    return paths


def collect_full_articles_multi(sources: list, raw_dir: str, max_articles: int = None,
                                 delay_seconds: float = 1.5,
                                 default_selectors: dict = None,
                                 selectors_by_source: dict = None) -> dict:
    """
    Versaun multi-orijen husi `collect_full_articles`: hala'o discovery +
    scraping konteúdu kompletu ba KADA orijen aktivu iha `sources`
    (ex: Tatoli, Timor Post, Timor News, GMN TV), depois kombina hotu-hotu
    notísia iha master file ida de'it (data/raw/tatoli_master.csv).

    `selectors_by_source` (opsionál) permite kada orijen (bazeia ba `name`)
    iha ninia seletór CSS próprio (barak vezes presiza, tanba kada site iha
    estrutura HTML diferente) — se orijen ida la iha entrada aki, uza
    `default_selectors` (husi `scraping.selectors` iha settings.yaml).

    Orijen ho `feed_url` vazia, `enabled: false`, ka ne'ebé falha (server
    erru, timeout) sei SALTA/regista falha maibé la para orijen seluk sira.
    """
    selectors_by_source = selectors_by_source or {}
    already_scraped = _load_scraped_ids(raw_dir)

    all_entries = []
    per_source = {}

    for src in sources:
        name = src.get("name", "").strip()
        feed_url = (src.get("feed_url") or "").strip()
        src_selectors = selectors_by_source.get(name) or src.get("selectors") or default_selectors

        if not src.get("enabled", True):
            logger.info("Orijen '%s' dezativa — salta.", name)
            per_source[name] = {"status": "dezativa", "scraped": 0, "failed": 0, "total": 0}
            continue

        if not feed_url:
            logger.warning("Orijen '%s' seidauk iha feed_url — salta.", name)
            per_source[name] = {
                "status": "seidauk konfigura (feed_url vazia)", "scraped": 0, "failed": 0, "total": 0,
            }
            continue

        try:
            entries = fetch_feed(feed_url, source_name=name)
        except Exception as e:
            logger.warning("Orijen '%s' falha discovery husi %s: %s", name, feed_url, e)
            per_source[name] = {"status": f"falha discovery: {e}", "scraped": 0, "failed": 0, "total": 0}
            continue

        to_scrape = [e for e in entries if e["news_id"] not in already_scraped]
        if max_articles:
            to_scrape = to_scrape[:max_articles]

        scraped_count, failed_count = 0, 0
        for entry in to_scrape:
            try:
                details = scrape_article(entry["link"], selectors=src_selectors)
                entry["content"] = details["content"] or entry.get("summary", "")
                entry["category"] = details["category"]
                entry["image_url"] = details["image_url"]
                if not entry.get("published_at") and details["published_fallback"]:
                    entry["published_at"] = details["published_fallback"]
                scraped_count += 1
            except requests.exceptions.RequestException as e:
                failed_count += 1
                logger.warning(
                    "[%s] Gagal scraping artikel '%s': %s. Fallback ke ringkasan RSS.",
                    name, entry.get("link", ""), e,
                )
                entry["content"] = entry.get("summary", "")
                entry["category"] = ""
                entry["image_url"] = ""
            time.sleep(delay_seconds)

        for entry in entries:
            entry.setdefault("content", entry.get("summary", ""))
            entry.setdefault("category", "")
            entry.setdefault("image_url", "")

        all_entries.extend(entries)
        per_source[name] = {
            "status": "susesu", "scraped": scraped_count, "failed": failed_count, "total": len(entries),
        }
        logger.info(
            "Orijen '%s': %s entri total, %s scrape kompletu foun, %s falha.",
            name, len(entries), scraped_count, failed_count,
        )

    paths = save_raw(all_entries, raw_dir)
    paths["scraped_new_articles"] = sum(v.get("scraped", 0) for v in per_source.values())
    paths["failed_articles"] = sum(v.get("failed", 0) for v in per_source.values())
    paths["per_source"] = per_source
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = collect_full_articles(
        "https://tatoli.tl/feed/", "data/raw", max_articles=5, delay_seconds=1.5,
    )
    print(result)
