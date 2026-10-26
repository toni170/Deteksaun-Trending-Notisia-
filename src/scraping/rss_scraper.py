"""
Modul parsing RSS feed Tatoli (https://tatoli.tl/feed/) menggunakan `feedparser`.

Bertanggung jawab untuk:
- Mengambil & parsing entri RSS (judul, link, tanggal publikasi, ringkasan konten).
- Menyimpan snapshot mentah ke data/raw (CSV & JSON, dengan timestamp).
- Menjaga "master file" kumulatif (dedup berdasarkan link/guid) supaya riwayat
  berita dari waktu ke waktu tetap terkumpul untuk analisis tren.
"""
import os
import re
import json
import time
import hashlib
import logging
from datetime import datetime

import feedparser
import requests
import pandas as pd

logger = logging.getLogger("rss_scraper")

RAW_FIELDS = [
    "news_id", "source", "title", "link", "published_at", "summary",
    "content", "category", "image_url", "fetched_at",
]

# Banyak server (termasuk WordPress, yang umum dipakai portal berita) menolak
# atau memutus koneksi dari request tanpa User-Agent yang terlihat "wajar".
# feedparser.parse(url) secara default TIDAK mengirim User-Agent yang aman,
# jadi kita ambil kontennya sendiri lewat `requests` lebih dulu.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://tatoli.tl/",
    "Connection": "keep-alive",
}
_REQUEST_TIMEOUT = 20  # detik

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Hapus tag HTML sederhana dari ringkasan RSS (mis. <p>, <a>)."""
    if not isinstance(text, str):
        return ""
    return _TAG_RE.sub(" ", text).strip()


def _make_news_id(link: str, title: str) -> str:
    """Buat ID unik & stabil dari link (fallback ke title bila link kosong)."""
    basis = link or title or ""
    return hashlib.md5(basis.encode("utf-8")).hexdigest()[:16]


def _parse_published(entry) -> str:
    """Ambil tanggal publikasi dalam format ISO 8601, dengan beberapa fallback."""
    for key in ("published_parsed", "updated_parsed"):
        struct = entry.get(key)
        if struct:
            try:
                return datetime(*struct[:6]).isoformat()
            except Exception:
                continue
    # fallback ke string mentah jika parsing struct_time gagal
    return entry.get("published", entry.get("updated", ""))


def _fetch_feed_content(feed_url: str) -> bytes:
    """
    Ambil konten RSS mentah lewat `requests` (dengan User-Agent browser) agar
    tidak diblokir/diputus oleh server. Hasilnya diserahkan ke feedparser.parse().
    """
    response = requests.get(feed_url, headers=_REQUEST_HEADERS, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content


def fetch_feed(feed_url: str, source_name: str = "Tatoli", retries: int = 3, backoff: float = 2.0) -> list:
    """
    Ambil & parsing RSS feed. Mengembalikan list of dict berisi:
    news_id, source, title, link, published_at, summary, fetched_at.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            content = _fetch_feed_content(feed_url)
        except requests.exceptions.RequestException as e:
            last_error = e
            logger.warning(
                "Percobaan %s/%s gagal mengambil feed %s: %s", attempt, retries, feed_url, e
            )
            time.sleep(backoff * attempt)
            continue

        parsed = feedparser.parse(content)

        # feedparser tidak melempar exception untuk error parsing; cek bozo flag
        if parsed.bozo and not parsed.entries:
            last_error = parsed.get("bozo_exception")
            logger.warning(
                "Percobaan %s/%s gagal parsing feed %s: %s", attempt, retries, feed_url, last_error
            )
            time.sleep(backoff * attempt)
            continue

        entries = []
        fetched_at = datetime.now().isoformat()
        for entry in parsed.entries:
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()
            summary_raw = entry.get("summary", entry.get("description", ""))
            summary = _strip_html(summary_raw)

            entries.append({
                "news_id": _make_news_id(link, title),
                "source": source_name,
                "title": title,
                "link": link,
                "published_at": _parse_published(entry),
                "summary": summary,
                "fetched_at": fetched_at,
            })

        logger.info("Berhasil mengambil %s entri dari feed %s", len(entries), feed_url)
        return entries

    raise ConnectionError(
        f"Gagal mengambil RSS feed dari '{feed_url}' setelah {retries} percobaan: {last_error}"
    )


def save_raw(entries: list, raw_dir: str) -> dict:
    """
    Simpan snapshot mentah (timestamped) + update master file kumulatif
    (dedup berdasarkan news_id/link) ke data/raw.
    """
    os.makedirs(raw_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    snapshot_json = os.path.join(raw_dir, f"tatoli_{stamp}.json")
    snapshot_csv = os.path.join(raw_dir, f"tatoli_{stamp}.csv")

    df_new = pd.DataFrame(entries, columns=RAW_FIELDS)

    with open(snapshot_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    df_new.to_csv(snapshot_csv, index=False, encoding="utf-8-sig")

    # --- update master file kumulatif (untuk analisis tren lintas waktu) ---
    master_path = os.path.join(raw_dir, "tatoli_master.csv")
    df_new_unique = df_new.drop_duplicates(subset=["news_id"], keep="first")
    if os.path.exists(master_path):
        # dtype=str: news_id hexadesimál bele parece numéru (ezemplu "0000000000000001"); se lee nu'udar
        # int, zero iha oin lakon no dedup la loos.
        df_master = pd.read_csv(master_path, dtype={"news_id": str})
        known_ids = set(df_master["news_id"].astype(str))
        # Notísia FOUN loloos = ne'ebé seidauk iha master (la'ós de'it kuantidade ne'ebé foti husi feed)
        new_count = int((~df_new_unique["news_id"].astype(str).isin(known_ids)).sum())
        combined = pd.concat([df_master, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["news_id"], keep="first")
    else:
        new_count = len(df_new_unique)
        combined = df_new_unique

    combined.to_csv(master_path, index=False, encoding="utf-8-sig")

    logger.info(
        "Snapshot disimpan: %s | %s. Foti %s, foun %s. Master file kini berisi %s berita unik.",
        snapshot_csv, snapshot_json, len(df_new), new_count, len(combined),
    )

    return {
        "snapshot_json": snapshot_json,
        "snapshot_csv": snapshot_csv,
        "master_csv": master_path,
        "total_unique": len(combined),
        "new_entries": new_count,          # notísia foun loloos (seidauk iha master)
        "fetched_entries": len(df_new),    # total ne'ebé foti husi feed dala ida-ne'e
    }


def collect_rss(feed_url: str, raw_dir: str, source_name: str = "Tatoli") -> dict:
    """High-level: fetch_feed -> save_raw. Mengembalikan info path & jumlah data."""
    entries = fetch_feed(feed_url, source_name=source_name)
    paths = save_raw(entries, raw_dir)
    return paths


def fetch_multi_feeds(sources: list, retries: int = 3, backoff: float = 2.0) -> dict:
    """
    Simu entrada RSS husi VÁRIU orijen (Tatoli, Timor Post, Timor News, GMN TV,
    dll), kada ida ho ninia `name` + `feed_url` própriu.

    `sources` = lista dict `{"name": ..., "feed_url": ..., "enabled": bool}`
    (mesmu formatu ho `sources:` iha config/settings.yaml).

    Orijen ho `feed_url` vazia ka `enabled: false` SALTA automátikamente.
    Se orijen ida FALHA (server down, URL sala, etc.), ida ne'e regista
    hanesan "falha" maibé LA PARA orijen seluk sira — atu koleta multi-orijen
    kontinua funsiona mesmu se orijen ida ka rua iha problema.

    Retorna: {"entries": [...gabungan husi hotu-hotu...], "per_source": {...}}
    """
    all_entries = []
    per_source = {}

    for src in sources:
        name = src.get("name", "").strip()
        feed_url = (src.get("feed_url") or "").strip()

        if not src.get("enabled", True):
            logger.info("Orijen '%s' dezativa (enabled=false) — salta.", name)
            per_source[name] = {"status": "dezativa", "count": 0}
            continue

        if not feed_url:
            logger.warning(
                "Orijen '%s' seidauk iha URL RSS feed (feed_url vazia) — salta.", name
            )
            per_source[name] = {"status": "seidauk konfigura (feed_url vazia)", "count": 0}
            continue

        try:
            entries = fetch_feed(feed_url, source_name=name, retries=retries, backoff=backoff)
            all_entries.extend(entries)
            per_source[name] = {"status": "susesu", "count": len(entries)}
        except Exception as e:
            logger.warning("Orijen '%s' falha koleta husi %s: %s", name, feed_url, e)
            per_source[name] = {"status": f"falha: {e}", "count": 0}

    return {"entries": all_entries, "per_source": per_source}


def collect_multi_rss(sources: list, raw_dir: str) -> dict:
    """High-level: fetch_multi_feeds -> save_raw (kombina hotu-hotu orijen iha master ida)."""
    result = fetch_multi_feeds(sources)
    paths = save_raw(result["entries"], raw_dir)
    paths["per_source"] = result["per_source"]
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = collect_rss("https://tatoli.tl/feed/", "data/raw")
    print(result)
