"""
Preprosesamentu ho etapa (ba painel): importa -> hamoos -> case folding -> tokenizasaun ->
hasai stopword -> stemming -> kategoria -> rai (CSV + JSON + SQLite).

Dadus prosesa ba lote ki'ik (padraun 5.000 notísia) atu memória la nakonu bainhira ficheiru
to'o 200 MB. Etapa kada lote hala'o tuir orden, no tempu/kuantidade kada etapa akumula atu
painel bele hatudu progresu kada etapa. Dadus bersih rai iha folder foun, ho CSV, JSON no SQLite.

Diferensa ho `text_cleaner.preprocess_news` (pipeline antigu):
  - karakter ho asentu (saúde, polítika) mantein; la tesi ba "sa de";
  - entidade HTML (&amp;) no URL hasai molok tokenizasaun;
  - apóstrofu iha ninin token hasai (ne'e, ha'u mantein).
"""
import html
import json
import logging
import os
import re
import shutil
import sqlite3
import time
import codecs
from datetime import datetime

import pandas as pd

from src.ingestion.dataset_importer import (
    COLUMN_ALIASES, DatasetImportError, _TEXT_COLUMNS, _clean_cell, _map_columns, _norm_header,
)
from src.preprocessing.category_classifier import classify_category
from src.preprocessing.stopwords_data import STOPWORDS_EN, STOPWORDS_ID, STOPWORDS_TET
from src.preprocessing.text_cleaner import _get_en_stemmer, _get_id_stemmer, _stem_tetun_token
from src.scraping.rss_scraper import _make_news_id
from src.storage.multi_writer import MultiFormatWriter

logger = logging.getLogger("staged_pipeline")

STAGES = [
    ("import", "Importa dadus"),
    ("cleaning", "Hamoos dadus (HTML, URL, entidade)"),
    ("case_folding", "Case folding (letra ki'ik)"),
    ("tokenization", "Tokenizasaun (fahe ba liafuan)"),
    ("stopwords", "Hasai stopword"),
    ("stemming", "Stemming (tesi sufiksu)"),
    ("category", "Kategoria referénsia"),
    ("save", "Rai dadus bersih (CSV, JSON, SQLite)"),
]
CLEAN_COLUMNS = ["news_id", "source", "title", "link", "published_at", "category", "image_url",
                 "language", "text_clean", "text_final", "token_count", "summary"]
SUMMARY_LIMIT = 600
SUPPORTED_INPUT = (".csv", ".json", ".db", ".sqlite")

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{3,}")
_NOISE_RE = re.compile(r"[^\w\s']|_")

_STOP = {"id": {w.lower() for w in STOPWORDS_ID}, "en": {w.lower() for w in STOPWORDS_EN},
         "tet": {w.lower() for w in STOPWORDS_TET}}
_STOP_ALL = _STOP["id"] | _STOP["en"] | _STOP["tet"]
_TET_MARKERS = {"iha", "nia", "sira", "ba", "husi", "ne'e", "atu"}


# ----------------------------------------------------------------------
# Etapa ba lista tékstu
# ----------------------------------------------------------------------
def clean_html_url(text: str) -> str:
    t = _TAG_RE.sub(" ", html.unescape(text))
    t = _URL_RE.sub(" ", t)
    return _SPACE_RE.sub(" ", t).strip()


def tokenize(text: str) -> list:
    t = _REPEAT_RE.sub(r"\1\1\1", text)
    t = _NOISE_RE.sub(" ", t)
    return [x for x in (tok.strip("'") for tok in t.split()) if x]


def detect_language_tokens(tokens: list) -> str:
    """Regra hanesan ho `text_cleaner.detect_language`, maibé husi lista token."""
    if not tokens:
        return "unknown"
    tokset = set(tokens)
    scores = {lang: len(tokset & words) for lang, words in _STOP.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "tet" if tokset & _TET_MARKERS else "unknown"
    return best


def remove_stopwords_tokens(tokens: list, language: str) -> list:
    stop = _STOP.get(language, _STOP_ALL)
    return [t for t in tokens if t not in stop and len(t) > 2]


class Stemmers:
    """Stemming ho memória (kamus) — liafuan hanesan la stem dala barak, lalais liu ba ficheiru boot."""

    def __init__(self):
        self.memo = {"id": {}, "en": {}, "tet": {}}
        self._fn = {}
        self.notes = []

    def _get(self, lang):
        if lang in self._fn:
            return self._fn[lang]
        fn = None
        try:
            if lang == "tet":
                fn = _stem_tetun_token
            elif lang == "id":
                fn = _get_id_stemmer().stem
            elif lang == "en":
                fn = _get_en_stemmer().stem
        except Exception as e:  # pakote la instala: stemming la aplika
            self.notes.append(f"Stemming '{lang}' la aplika: {e}")
        self._fn[lang] = fn
        return fn

    def stem(self, tokens: list, language: str) -> list:
        fn = self._get(language) if language in self.memo else None
        if fn is None:
            return tokens
        memo, out = self.memo[language], []
        for tok in tokens:
            s = memo.get(tok)
            if s is None:
                s = memo[tok] = fn(tok)
            out.append(s)
        return out


# ----------------------------------------------------------------------
# Lee dadus input (CSV / JSON / SQLite) ba lote ki'ik
# ----------------------------------------------------------------------
class InputReader:
    def __init__(self, path: str):
        self.path = path
        self.ext = os.path.splitext(path)[1].lower()
        if self.ext not in SUPPORTED_INPUT:
            raise DatasetImportError("unsupported_format",
                                     f"Formatu '{self.ext or '?'}' la suporta. Uza .csv, .json, .db ka .sqlite.")
        if not os.path.isfile(path):
            raise DatasetImportError("not_found", f"Ficheiru '{os.path.basename(path)}' la hetan.")
        if os.path.getsize(path) == 0:
            raise DatasetImportError("empty_file", "Ficheiru mamuk (la iha konteúdu).")

    def _csv_setup(self):
        with open(self.path, "rb") as f:
            head = f.read(65536)
        enc = "utf-8-sig"
        try:
            text = codecs.getincrementaldecoder("utf-8-sig")().decode(head, final=False)
        except UnicodeDecodeError:
            enc = "cp1252"
            text = head.decode("cp1252", errors="replace")
        first = text.lstrip("\r\n").split("\n", 1)[0]
        return enc, max([",", ";", "\t"], key=first.count)

    def estimate_rows(self) -> int:
        try:
            if self.ext == ".csv":
                n = 0
                with open(self.path, "rb") as f:
                    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        n += block.count(b"\n")
                return max(n - 1, 1)
            if self.ext == ".json":
                return max(len(self._load_json()), 1)
            conn = sqlite3.connect(self.path)
            try:
                return max(conn.execute(f'SELECT COUNT(*) FROM "{self._table(conn)}"').fetchone()[0], 1)
            finally:
                conn.close()
        except Exception:
            return 1

    def _load_json(self):
        with open(self.path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [])
        if not isinstance(data, list):
            raise DatasetImportError("unreadable", "JSON tenke ho lista notísia (array).")
        return data

    @staticmethod
    def _table(conn) -> str:
        names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if not names:
            raise DatasetImportError("empty_file", "SQLite la iha tabela.")
        return "articles" if "articles" in names else names[0]

    @staticmethod
    def _as_text(df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna("").astype(str)

    def chunks(self, chunk_rows: int):
        try:
            if self.ext == ".csv":
                enc, sep = self._csv_setup()
                yield from pd.read_csv(self.path, sep=sep, dtype=str, keep_default_na=False,
                                       chunksize=chunk_rows, encoding=enc)
            elif self.ext == ".json":
                data = self._load_json()
                for i in range(0, len(data), chunk_rows):
                    yield self._as_text(pd.DataFrame(data[i:i + chunk_rows]))
            else:
                conn = sqlite3.connect(self.path)
                try:
                    q = f'SELECT * FROM "{self._table(conn)}"'
                    for part in pd.read_sql_query(q, conn, chunksize=chunk_rows):
                        yield self._as_text(part)
                finally:
                    conn.close()
        except DatasetImportError:
            raise
        except pd.errors.EmptyDataError:
            raise DatasetImportError("empty_file", "Ficheiru mamuk (la iha kolona ka dadus).")
        except (pd.errors.ParserError, ValueError, sqlite3.DatabaseError, UnicodeDecodeError) as e:
            raise DatasetImportError("unreadable", f"La bele lee ficheiru. Verifika se la korruptu ({e}).")


def _normalize_chunk(df: pd.DataFrame, mapping: dict, id_col, seen: set, counters: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in _TEXT_COLUMNS:
        out[col] = _clean_cell(df[mapping[col]]) if col in mapping else ""
    if "published_at" in mapping:
        raw = _clean_cell(df[mapping["published_at"]])
        parsed = pd.to_datetime(raw.mask(raw == ""), errors="coerce", utc=True, format="mixed")
        counters["invalid_dates"] += int((parsed.isna() & (raw != "")).sum())
        out["published_at"] = parsed.dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("")
    else:
        out["published_at"] = ""

    has_text = (out["title"] != "") | (out["content"] != "") | (out["summary"] != "")
    counters["removed_empty"] += int((~has_text).sum())
    out, df = out[has_text], df[has_text]
    if out.empty:
        return out

    ids = _clean_cell(df[id_col]) if id_col else pd.Series("", index=out.index)
    generated = [_make_news_id(l or f"{t}|{(c or s)[:200]}", t)
                 for l, t, c, s in zip(out["link"], out["title"], out["content"], out["summary"])]
    out["news_id"] = [i if i else g for i, g in zip(ids, generated)]

    dup = out["news_id"].duplicated(keep="first") | out["news_id"].isin(seen)
    counters["removed_duplicate"] += int(dup.sum())
    out = out[~dup]
    seen.update(out["news_id"])
    return out


def _check_columns(df: pd.DataFrame) -> dict:
    mapping = _map_columns(df)
    missing = []
    if "title" not in mapping:
        missing.append("títulu (title)")
    if "content" not in mapping and "summary" not in mapping:
        missing.append("konteúdu (content ka summary)")
    if missing:
        raise DatasetImportError(
            "missing_columns",
            "Koluna ne'ebé falta: " + ", ".join(missing) + f". Koluna ne'ebé iha: {', '.join(map(str, df.columns))}.",
            missing=missing)
    return mapping


# ----------------------------------------------------------------------
# Pipeline prinsipál
# ----------------------------------------------------------------------
def _short(text: str, n: int = 320) -> str:
    return text if len(text) <= n else text[:n] + " …"


def list_clean_datasets(cleaned_dir: str) -> list:
    """Dadus bersih ne'ebé kompletu (iha meta.json), foun -> tuan."""
    out = []
    if not os.path.isdir(cleaned_dir):
        return out
    for name in sorted(os.listdir(cleaned_dir), reverse=True):
        meta = os.path.join(cleaned_dir, name, "meta.json")
        if os.path.isfile(meta):
            try:
                with open(meta, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("kind") == "dadus_bersih":
                    data["directory"] = os.path.join(cleaned_dir, name)
                    out.append(data)
            except (OSError, ValueError):
                continue
    return out


def run_staged_preprocessing(input_path: str, cleaned_dir: str, *, chunk_rows: int = 5000,
                             progress=None, stop_event=None) -> dict:
    """
    Hala'o preprosesamentu tomak ba ficheiru `input_path`. `progress(fraksaun, mensajen, **extra)`
    simu `stages` (lista estadu kada etapa) no `rows_done`/`rows_total`.
    Retorna dict rezultadu (mós rai iha `meta.json`). Se para ka falla, folder parsiál hamoos.
    """
    progress = progress or (lambda *a, **k: None)
    started = time.time()
    reader = InputReader(input_path)
    total_rows = reader.estimate_rows()

    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(input_path))[0]).strip("_") or "dadus"
    out_dir = os.path.join(cleaned_dir, f"bersih_{stem}")
    n = 1
    while os.path.exists(out_dir):
        n += 1
        out_dir = os.path.join(cleaned_dir, f"bersih_{stem}_{n}")
    os.makedirs(cleaned_dir, exist_ok=True)

    board = {k: {"key": k, "label": lb, "status": "hein", "rows": 0, "seconds": 0.0, "detail": ""}
             for k, lb in STAGES}
    tok = {"tokenized": 0, "after_stop": 0, "after_stem": 0}
    vocab_before, vocab_after = set(), set()
    langs, cats = {}, {}
    samples = {}
    counters = {"invalid_dates": 0, "removed_empty": 0, "removed_duplicate": 0, "removed_after_clean": 0}
    rows_read = rows_saved = 0
    stemmers = Stemmers()
    seen = set()
    mapping, id_col = None, None

    def emit(message, current=None):
        for k, v in board.items():
            if k == current:
                v["status"] = "hala'o"
        frac = min(rows_read / total_rows, 0.98) if total_rows else 0.0
        progress(frac, message, stages=[dict(v) for v in board.values()], rows_done=rows_read,
                 rows_total=total_rows, current_stage=current)

    def timed(key, fn, *a):
        t0 = time.perf_counter()
        res = fn(*a)
        board[key]["seconds"] += time.perf_counter() - t0
        return res

    writer = None
    ok = False
    try:
        writer = MultiFormatWriter(out_dir, "notisia_bersih", CLEAN_COLUMNS, key_column="news_id",
                                   int_columns=("token_count",))
        emit("Hahú lee dadus...", "import")
        for raw_chunk in reader.chunks(chunk_rows):
            if stop_event is not None and stop_event.is_set():
                break
            rows_read += len(raw_chunk)
            # 1) importa
            emit(f"Importa dadus ({rows_read:,}/{total_rows:,})", "import")
            t0 = time.perf_counter()
            if mapping is None:
                mapping = _check_columns(raw_chunk)
                id_col = next((c for c in raw_chunk.columns if _norm_header(c) == "news_id"), None)
            chunk = _normalize_chunk(raw_chunk, mapping, id_col, seen, counters)
            board["import"]["seconds"] += time.perf_counter() - t0
            board["import"]["rows"] = rows_read
            board["import"]["status"] = "hala'o"
            if chunk.empty:
                continue
            bodies = [c if c else s for c, s in zip(chunk["content"], chunk["summary"])]
            raw_texts = [f"{t} . {b}" for t, b in zip(chunk["title"], bodies)]

            # 2) hamoos
            emit("Hamoos HTML, entidade no URL...", "cleaning")
            cleaned = timed("cleaning", lambda xs: [clean_html_url(x) for x in xs], raw_texts)
            # 3) case folding
            emit("Case folding...", "case_folding")
            folded = timed("case_folding", lambda xs: [x.lower() for x in xs], cleaned)
            # 4) tokenizasaun
            emit("Tokenizasaun...", "tokenization")
            tokens = timed("tokenization", lambda xs: [tokenize(x) for x in xs], folded)
            tok["tokenized"] += sum(len(t) for t in tokens)
            vocab_before.update(w for t in tokens for w in t)
            text_clean = [" ".join(t) for t in tokens]
            # 5) stopword (deteta lian molok)
            emit("Deteta lian no hasai stopword...", "stopwords")
            languages = timed("stopwords", lambda xs: [detect_language_tokens(t) for t in xs], tokens)
            kept = timed("stopwords", lambda ts, ls: [remove_stopwords_tokens(t, l) for t, l in zip(ts, ls)],
                         tokens, languages)
            tok["after_stop"] += sum(len(t) for t in kept)
            for lg in languages:
                langs[lg] = langs.get(lg, 0) + 1
            # 6) stemming
            emit("Stemming...", "stemming")
            stemmed = timed("stemming", lambda ts, ls: [stemmers.stem(t, l) for t, l in zip(ts, ls)], kept, languages)
            tok["after_stem"] += sum(len(t) for t in stemmed)
            vocab_after.update(w for t in stemmed for w in t)
            text_final = [" ".join(t) for t in stemmed]
            # 7) kategoria referénsia
            emit("Klasifika kategoria...", "category")
            categories = timed("category", lambda: [
                classify_category(t, b, c) for t, b, c in zip(chunk["title"], bodies, chunk["category"])])
            for c in categories:
                cats[c] = cats.get(c, 0) + 1

            for key in ("cleaning", "case_folding", "tokenization", "stopwords", "stemming", "category"):
                board[key]["rows"] += len(chunk)
            if not samples and any(text_final):
                i = next(i for i, x in enumerate(text_final) if x)
                samples = {"raw": _short(raw_texts[i]), "cleaning": _short(cleaned[i]),
                           "case_folding": _short(folded[i]), "tokenization": _short(" | ".join(tokens[i])),
                           "stopwords": _short(" | ".join(kept[i])), "stemming": _short(text_final[i]),
                           "language": languages[i]}

            # 8) rai (hasai liña ho text_final mamuk)
            emit("Rai dadus bersih...", "save")
            t0 = time.perf_counter()
            keep_idx = [i for i, x in enumerate(text_final) if x.strip()]
            counters["removed_after_clean"] += len(text_final) - len(keep_idx)
            ids, srcs, ttl = chunk["news_id"].tolist(), chunk["source"].tolist(), chunk["title"].tolist()
            lnk, pub, img = chunk["link"].tolist(), chunk["published_at"].tolist(), chunk["image_url"].tolist()
            summ = chunk["summary"].tolist()
            rows = [{"news_id": ids[i], "source": srcs[i], "title": ttl[i], "link": lnk[i],
                     "published_at": pub[i], "category": categories[i], "image_url": img[i],
                     "language": languages[i], "text_clean": text_clean[i], "text_final": text_final[i],
                     "token_count": len(stemmed[i]), "summary": summ[i][:SUMMARY_LIMIT]} for i in keep_idx]
            rows_saved += writer.write_rows(rows)
            board["save"]["seconds"] += time.perf_counter() - t0
            board["save"]["rows"] = rows_saved

        stopped = stop_event is not None and stop_event.is_set()
        sizes = writer.close()
        writer = None
        if stopped:
            progress(None, "Preprosesamentu para. Rezultadu parsiál la rai.")
            return {"status": "para", "message": "Utilizadór para preprosesamentu."}
        if rows_saved == 0:
            raise DatasetImportError("no_valid_rows",
                                     "La iha liña válidu depois preprosesamentu (tékstu hotu mamuk).")

        for v in board.values():
            v["status"] = "remata"
        n_docs = max(rows_saved, 1)
        board["import"]["detail"] = (f"{rows_read:,} liña lee · {counters['removed_empty']:,} mamuk · "
                                     f"{counters['removed_duplicate']:,} duplikadu hasai")
        board["cleaning"]["detail"] = "Tag HTML, entidade (&amp;) no URL hasai."
        board["case_folding"]["detail"] = "Letra boot hotu sai letra ki'ik."
        board["tokenization"]["detail"] = (f"{tok['tokenized']:,} token · média "
                                           f"{tok['tokenized'] / max(rows_read, 1):.0f} kada notísia")
        removed = tok["tokenized"] - tok["after_stop"]
        board["stopwords"]["detail"] = (f"{removed:,} token hasai ({100 * removed / max(tok['tokenized'], 1):.0f}%) · "
                                        f"lian: " + ", ".join(f"{k} {v:,}" for k, v in sorted(langs.items())))
        board["stemming"]["detail"] = (f"Liafuan úniku {len(vocab_before):,} → {len(vocab_after):,}"
                                       + (" · " + "; ".join(stemmers.notes) if stemmers.notes else ""))
        board["category"]["detail"] = ", ".join(f"{k} {v:,}" for k, v in sorted(cats.items(), key=lambda x: -x[1]))
        board["save"]["detail"] = f"{rows_saved:,} notísia rai"

        result = {
            "kind": "dadus_bersih", "status": "remata",
            "dataset_id": os.path.basename(out_dir), "directory": out_dir, "files": writer_paths(out_dir),
            "sizes": sizes, "source_file": input_path, "rows_read": rows_read, "rows_saved": rows_saved,
            "removed_empty": counters["removed_empty"], "removed_duplicate": counters["removed_duplicate"],
            "removed_after_clean": counters["removed_after_clean"], "invalid_dates": counters["invalid_dates"],
            "languages": langs, "categories": cats, "stages": [dict(v) for v in board.values()],
            "samples": samples, "avg_tokens_final": round(tok["after_stem"] / n_docs, 1),
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1),
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        ok = True
        progress(1.0, f"Preprosesamentu remata: {rows_saved:,} notísia bersih rai.", stages=result["stages"],
                 rows_done=rows_read, rows_total=total_rows)
        return result
    finally:
        if writer is not None:
            writer.close()
        if not ok:
            shutil.rmtree(out_dir, ignore_errors=True)


def writer_paths(out_dir: str) -> dict:
    return {"csv": os.path.join(out_dir, "notisia_bersih.csv"),
            "json": os.path.join(out_dir, "notisia_bersih.json"),
            "sqlite": os.path.join(out_dir, "notisia_bersih.db")}
