"""
Koleta lote (pedidu) notísia to'o tamañu CSV 200 MB.

Kada pedidu halo lote INDEPENDENTE iha folder rasik (`batch_AAAAMMDD_HHMMSS/`) ho ficheiru
CSV, JSON no SQLite. Lote la hamutuk ho lote seluk, atu bele treina kada lote.

Dadus foti husi WordPress REST API (`/wp-json/wp/v2/posts`), pájina 100 notísia, ho kursór
tempu kada orijen (`state.json`):
  - fase "foun": notísia ne'ebé publika depois notísia foun liu ne'ebé rai ona;
  - fase "tuan": hahú husi notísia tuan liu ne'ebé rai ona, fila ba kotuk.
Notísia ne'ebé rai ona iha lote anterior la repete (indeks `index.db`).

Pedidu para bainhira: tamañu CSV to'o meta; ka orijen hotu la iha notísia foun tan; ka
utilizadór para. Se orijen la iha notísia nune'e to'o la bele kompleta 200 MB, lote sei
ki'ik liu meta no marka ho loloos (`reached_target = False`).
"""
import json
import logging
import os
import shutil
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import requests

from src.scraping.history_scraper import _post_to_entry, _site_root
from src.scraping.rss_scraper import RAW_FIELDS, _REQUEST_HEADERS
from src.storage.multi_writer import MultiFormatWriter

logger = logging.getLogger("batch_collector")

DEFAULT_TARGET_MB = 200
STATE_FILE = "state.json"
INDEX_FILE = "index.db"
_WP_TIME = "%Y-%m-%dT%H:%M:%S"
_FETCH_TIMEOUT = 60


# ----------------------------------------------------------------------
# Indeks notísia ne'ebé rai ona (atu la repete entre lote)
# ----------------------------------------------------------------------
class SeenIndex:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS seen (news_id TEXT PRIMARY KEY, batch_id TEXT)")
        self.conn.commit()

    def known(self, ids) -> set:
        ids = list(ids)
        found = set()
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            marks = ",".join("?" for _ in part)
            found.update(r[0] for r in self.conn.execute(
                f"SELECT news_id FROM seen WHERE news_id IN ({marks})", part))
        return found

    def add(self, batch_id: str, ids) -> None:
        self.conn.executemany("INSERT OR IGNORE INTO seen (news_id, batch_id) VALUES (?, ?)",
                              [(i, batch_id) for i in ids])
        self.conn.commit()

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]

    def close(self) -> None:
        self.conn.close()


def load_state(batches_dir: str) -> dict:
    path = os.path.join(batches_dir, STATE_FILE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(batches_dir: str, state: dict) -> None:
    os.makedirs(batches_dir, exist_ok=True)
    tmp = os.path.join(batches_dir, STATE_FILE + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(batches_dir, STATE_FILE))


# ----------------------------------------------------------------------
# Hetan pájina husi WordPress API
# ----------------------------------------------------------------------
def wp_fetch_page(api_url: str, params: dict, retries: int = 3, backoff: float = 2.0):
    """Retorna (posts, total_pájina). Pájina liu limite (HTTP 400) retorna lista mamuk."""
    headers = {**_REQUEST_HEADERS, "Accept": "application/json"}
    last = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(api_url, params=params, headers=headers, timeout=_FETCH_TIMEOUT)
            break
        except requests.exceptions.RequestException as e:
            last = e
            logger.warning("Tentativa %s/%s falla (%s): %s", attempt, retries, api_url, e)
            time.sleep(backoff * attempt)
    else:
        raise ConnectionError(f"La konsege hetan '{api_url}' depois tentativa {retries}: {last}")
    if resp.status_code == 400 and int(params.get("page", 1)) > 1:
        return [], 0
    resp.raise_for_status()
    posts = resp.json()
    if not isinstance(posts, list):
        raise ValueError("Resposta API la tuir formatu WordPress (lista notísia).")
    return posts, int(resp.headers.get("X-WP-TotalPages", 0) or 0)


def _shift(date_text: str, seconds: int) -> str:
    try:
        return (datetime.strptime(date_text[:19], _WP_TIME) + timedelta(seconds=seconds)).strftime(_WP_TIME)
    except (ValueError, TypeError):
        return date_text


# ----------------------------------------------------------------------
# Rezultadu
# ----------------------------------------------------------------------
@dataclass
class BatchResult:
    batch_id: str = ""
    directory: str = ""
    kind: str = "manuál"
    status: str = "remata"          # remata | para | falla | mamuk
    rows: int = 0
    csv_bytes: int = 0
    target_bytes: int = 0
    reached_target: bool = False
    message: str = ""
    started_at: str = ""
    finished_at: str = ""
    per_source: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    sizes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["csv_mb"] = round(self.csv_bytes / 1024 / 1024, 2)
        d["target_mb"] = round(self.target_bytes / 1024 / 1024, 2)
        return d


class _Cursor:
    """Kursór pájina ba orijen ida iha fase ida."""

    def __init__(self, source: str, api: str, mode: str, params: dict):
        self.source, self.api, self.mode, self.params = source, api, mode, params
        self.page = 1
        self.done = False
        self.exhausted = False
        self.error = None


def _free_space_ok(path: str, need_bytes: int) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        return shutil.disk_usage(path).free >= need_bytes
    except OSError:
        return True


def collect_batch(sources: list, batches_dir: str, *, kind: str = "manuál",
                  target_mb: float = DEFAULT_TARGET_MB, per_page: int = 100,
                  delay_seconds: float = 1.0, progress=None, stop_event=None,
                  fetch_page=None, rss_fetch=None) -> BatchResult:
    """
    Hala'o pedidu ida: foti notísia husi `sources` to'o CSV = `target_mb` MB.
    `fetch_page(api_url, params) -> (posts, total_pájina)` bele troka (ba teste).
    `rss_fetch(feed_url, naran)` = fallback se API WordPress la funsiona.
    """
    progress = progress or (lambda *a, **k: None)
    stop_event = stop_event or threading.Event()
    fetch_page = fetch_page or wp_fetch_page
    if rss_fetch is None:
        def rss_fetch(feed_url, name):
            from src.scraping.rss_scraper import fetch_feed
            return fetch_feed(feed_url, source_name=name)

    target_bytes = int(float(target_mb) * 1024 * 1024)
    now = datetime.now()
    batch_id = "batch_" + now.strftime("%Y%m%d_%H%M%S")
    directory = os.path.join(batches_dir, batch_id)
    n = 1
    while os.path.exists(directory):
        n += 1
        directory = os.path.join(batches_dir, f"{batch_id}_{n}")
    batch_id = os.path.basename(directory)

    result = BatchResult(batch_id=batch_id, directory=directory, kind=kind,
                         target_bytes=target_bytes, started_at=now.isoformat(timespec="seconds"))

    active = [s for s in sources
              if s.get("enabled", True) and (s.get("feed_url") or "").strip() and (s.get("name") or "").strip()]
    if not active:
        result.status, result.message = "falla", "Laiha orijen aktivu ho URL feed. Hili orijen iha sidebar."
        result.finished_at = datetime.now().isoformat(timespec="seconds")
        return result
    if not _free_space_ok(batches_dir, int(target_bytes * 3.2) + 200 * 1024 * 1024):
        result.status = "falla"
        result.message = "Fatin iha disku la to'o (presiza besik 3 dala tamañu meta, tanba CSV + JSON + SQLite)."
        result.finished_at = datetime.now().isoformat(timespec="seconds")
        return result

    state = load_state(batches_dir)
    index = SeenIndex(os.path.join(batches_dir, INDEX_FILE))
    writer = MultiFormatWriter(directory, "notisia", RAW_FIELDS, max_csv_bytes=target_bytes,
                               key_column="news_id")
    stats = {s["name"]: {"status": "hein", "rows": 0, "newest": None, "oldest": None,
                         "exhausted": False} for s in active}
    batch_seen = set()
    fetched_at = datetime.now().isoformat()

    def report(text: str):
        frac = min(writer.csv_bytes / target_bytes, 1.0) if target_bytes else 0.0
        progress(frac, text, rows=writer.rows, mb=round(writer.csv_bytes / 1024 / 1024, 2),
                 target_mb=round(target_bytes / 1024 / 1024, 2), per_source={k: v["rows"] for k, v in stats.items()})

    def store(name: str, pairs: list) -> bool:
        """Hakerek pár (entrada, data lokál). Retorna False se meta to'o ona."""
        fresh = []
        for entry, local in pairs:
            nid = entry["news_id"]
            if nid in batch_seen or not (entry.get("title") or entry.get("content") or entry.get("summary")):
                continue
            fresh.append((entry, local))
        if fresh:
            known = index.known(e["news_id"] for e, _ in fresh)
            fresh = [(e, d) for e, d in fresh if e["news_id"] not in known]
        if not fresh:
            return True
        n_written = writer.write_rows(e for e, _ in fresh)
        done = fresh[:n_written]
        ids = [e["news_id"] for e, _ in done]
        index.add(batch_id, ids)
        batch_seen.update(ids)
        st = stats[name]
        st["rows"] += len(done)
        for _, local in done:
            if local:
                st["newest"] = local if st["newest"] is None or local > st["newest"] else st["newest"]
                st["oldest"] = local if st["oldest"] is None or local < st["oldest"] else st["oldest"]
        return n_written == len(fresh)

    def build_cursors():
        fresh_c, back_c = [], []
        for s in active:
            name, api = s["name"].strip(), _site_root(s["feed_url"].strip()) + "/wp-json/wp/v2/posts"
            sst = state.get(name, {})
            base = {"per_page": min(int(per_page), 100), "_embed": "1", "orderby": "date"}
            if sst.get("newest"):
                fresh_c.append(_Cursor(name, api, "foun",
                                       {**base, "order": "asc", "after": _shift(sst["newest"], -1)}))
            if not sst.get("exhausted"):
                params = {**base, "order": "desc"}
                if sst.get("oldest"):
                    params["before"] = _shift(sst["oldest"], 1)
                back_c.append(_Cursor(name, api, "tuan", params))
        return fresh_c, back_c

    def run_phase(cursors: list) -> bool:
        """Retorna False se tenke para tomak (meta ka utilizadór para)."""
        while any(not c.done for c in cursors):
            for cur in cursors:
                if cur.done:
                    continue
                if stop_event.is_set():
                    return False
                try:
                    cur.params["page"] = cur.page
                    posts, total_pages = fetch_page(cur.api, dict(cur.params))
                except Exception as e:
                    cur.done, cur.error = True, str(e)
                    logger.warning("[%s] para fase '%s' iha pájina %s: %s", cur.source, cur.mode, cur.page, e)
                    if cur.page == 1 and cur.mode == "tuan" and not stats[cur.source]["rows"]:
                        _rss_fallback(cur)
                    continue
                if not posts:
                    cur.done = cur.exhausted = True
                    continue
                pairs = [(_post_to_entry(p, cur.source, fetched_at), p.get("date") or "") for p in posts]
                ok = store(cur.source, pairs)
                report(f"{cur.source}: pájina {cur.page} · {writer.rows:,} notísia · "
                       f"{writer.csv_bytes / 1024 / 1024:.1f} / {target_bytes / 1024 / 1024:.0f} MB")
                if not ok:
                    return False
                if total_pages and cur.page >= total_pages:
                    cur.done = cur.exhausted = True
                cur.page += 1
                if not cur.done and delay_seconds:
                    if stop_event.wait(delay_seconds):
                        return False
        return True

    def _rss_fallback(cur: _Cursor):
        src = next(s for s in active if s["name"].strip() == cur.source)
        try:
            entries = rss_fetch(src["feed_url"].strip(), cur.source)
            store(cur.source, [(e, "") for e in entries])
            cur.error = f"API WordPress la disponivel; uza RSS ({len(entries)} notísia). Detallu: {cur.error}"
        except Exception as e:  # RSS mós falla
            cur.error = f"{cur.error} | RSS: {e}"

    cursors_all = []
    try:
        report("Hahú pedidu dadus...")
        fresh_c, back_c = build_cursors()
        cursors_all = fresh_c + back_c
        proceed = run_phase(fresh_c) if fresh_c else True
        if proceed and back_c:
            proceed = run_phase(back_c)
        if stop_event.is_set():
            result.status, result.message = "para", "Utilizadór para pedidu. Dadus ne'ebé foti ona rai hela."
        elif writer.cap_hit:
            result.message = f"Meta {target_bytes / 1024 / 1024:.0f} MB to'o ona."
        else:
            result.message = "Orijen hotu la iha notísia foun tan; lote ki'ik liu meta."
    except Exception as e:
        logger.exception("Pedidu falla")
        result.status, result.message = "falla", f"Erru durante pedidu: {e}"
    finally:
        sizes = writer.close()
        # atualiza kursór orijen kada orijen (de'it ho saida ne'ebé hakerek ona)
        for cur in cursors_all:
            stats[cur.source]["status"] = f"falla: {cur.error}" if cur.error else "susesu"
        for name, st in stats.items():
            cur_state = state.setdefault(name, {})
            if st["newest"] and (not cur_state.get("newest") or st["newest"] > cur_state["newest"]):
                cur_state["newest"] = st["newest"]
            if st["oldest"] and (not cur_state.get("oldest") or st["oldest"] < cur_state["oldest"]):
                cur_state["oldest"] = st["oldest"]
            for cur in cursors_all:
                if cur.source == name and cur.mode == "tuan" and cur.exhausted and not cur.error:
                    cur_state["exhausted"] = True
                    st["exhausted"] = True
        save_state(batches_dir, state)
        index.close()

    result.rows = writer.rows
    result.csv_bytes = writer.csv_bytes
    result.reached_target = bool(writer.cap_hit)
    result.per_source = stats
    result.sizes = sizes
    result.finished_at = datetime.now().isoformat(timespec="seconds")

    if result.rows == 0:
        shutil.rmtree(directory, ignore_errors=True)
        if result.status == "remata":
            result.status = "mamuk"
            result.message = "La iha notísia foun iha orijen sira ba pedidu ne'e."
        result.files = {}
        return result

    result.files = dict(writer.paths)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    progress(min(result.csv_bytes / target_bytes, 1.0), result.message, rows=result.rows,
             mb=round(result.csv_bytes / 1024 / 1024, 2))
    return result


# ----------------------------------------------------------------------
# Lista lote
# ----------------------------------------------------------------------
def list_batches(batches_dir: str) -> list:
    """Lote ne'ebé kompletu (iha meta.json), foun -> tuan."""
    out = []
    if not os.path.isdir(batches_dir):
        return out
    for name in sorted(os.listdir(batches_dir), reverse=True):
        meta = os.path.join(batches_dir, name, "meta.json")
        if os.path.isfile(meta):
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["directory"] = os.path.join(batches_dir, name)
                out.append(data)
            except (OSError, ValueError):
                continue
    return out
