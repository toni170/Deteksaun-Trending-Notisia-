"""
Importa dataset notísia husi ficheiru CSV ka Excel (.xlsx) no hamoos nia
antes tama ba pipeline preprocessing.

Responsabilidade:
- Valida formatu ficheiru (.csv / .xlsx) no konteúdu (la mamuk, la korruptu).
- Rekonhese naran kolona (Tetun / Indonézia / Inglés) no mapa ba formatu
  `RAW_FIELDS` hanesan ho `src/scraping/rss_scraper.py`.
- Hamoos liña mamuk no duplikadu, no relata saida ne'ebé hasai.

Erru hotu-hotu sai `DatasetImportError` ho `code` estável (atu teste/dashboard
bele hatudu mensajen Tetun ne'ebé loos) — la iha traceback bruta ba utilizadór.
"""
import io
import os
import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.scraping.rss_scraper import RAW_FIELDS, _make_news_id

logger = logging.getLogger("dataset_importer")

SUPPORTED_EXTENSIONS = (".csv", ".xlsx")

# Naran kolona (depois normaliza: minúskula, la iha asentu, "_" troka espasu)
COLUMN_ALIASES = {
    "title": {"title", "titulu", "titulo", "judul", "headline"},
    "content": {"content", "konteudu", "conteudo", "konten", "isi", "body",
                "text", "teks", "artigu", "artikel"},
    "summary": {"summary", "rezumu", "resumo", "ringkasan", "excerpt",
                "description", "deskrisaun"},
    "published_at": {"published_at", "published", "date", "data", "tanggal",
                     "tempu", "tempo", "datetime", "tgl"},
    "category": {"category", "kategoria", "kategori", "categoria"},
    "source": {"source", "orijen", "sumber", "fonte"},
    "link": {"link", "url", "liga"},
    "image_url": {"image_url", "image", "imajen", "gambar"},
}

# Valór selula ne'ebé reprezenta "mamuk" iha ficheiru eksportadu husi Excel/sistema seluk.
_EMPTY_TOKENS = {"", "nan", "none", "null", "n/a", "na", "#n/a"}

_TEXT_COLUMNS = ["title", "content", "summary", "category", "source", "link", "image_url"]


class DatasetImportError(ValueError):
    """Erru importa dataset ho `code` estável + mensajen Tetun ba utilizadór."""

    def __init__(self, code: str, message: str, missing: list | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.missing = missing or []


@dataclass
class ImportReport:
    filename: str
    file_type: str
    rows_read: int = 0
    rows_removed_empty: int = 0
    rows_removed_duplicate: int = 0
    rows_valid: int = 0
    invalid_dates: int = 0
    mapped_columns: dict = field(default_factory=dict)  # kolona kanónika -> naran orijinál

    def summary_tetun(self) -> str:
        return (
            f"Lee {self.rows_read} liña husi '{self.filename}': "
            f"{self.rows_valid} válidu, {self.rows_removed_empty} mamuk hasai, "
            f"{self.rows_removed_duplicate} duplikadu hasai."
        )


@dataclass
class ImportResult:
    df: pd.DataFrame
    report: ImportReport


# ----------------------------------------------------------------------
# Lee ficheiru
# ----------------------------------------------------------------------
def _norm_header(name) -> str:
    text = unicodedata.normalize("NFD", str(name))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve_source(source, filename):
    """Retorna (bytes, filename) husi path, file-like (Streamlit UploadedFile), ka BytesIO."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise DatasetImportError("not_found", f"Ficheiru '{os.path.basename(path)}' la hetan.")
        except OSError as e:
            raise DatasetImportError("unreadable", f"La bele loke ficheiru: {e}")
        return data, filename or os.path.basename(path)

    name = filename or getattr(source, "name", None) or ""
    if hasattr(source, "getvalue"):
        data = source.getvalue()
    elif hasattr(source, "read"):
        data = source.read()
    else:
        raise DatasetImportError("unreadable", "Fonte ficheiru la válidu.")
    return data, name


def _read_csv(raw: bytes) -> pd.DataFrame:
    text = None
    for enc in ("utf-8-sig", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DatasetImportError("unreadable", "La bele lee kodifikasaun ficheiru. Rai nu'udar CSV UTF-8.")

    first_line = text.lstrip("\r\n").split("\n", 1)[0]
    sep = max([",", ";", "\t"], key=first_line.count)
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def _read_table(raw: bytes, ext: str) -> pd.DataFrame:
    if not raw or not raw.strip():
        raise DatasetImportError("empty_file", "Ficheiru mamuk (la iha konteúdu).")
    if ext == ".csv" and b"\x00" in raw[:4096]:
        raise DatasetImportError("unreadable", "Ficheiru la'ós CSV tékstu válidu (konteúdu bináriu).")
    try:
        if ext == ".csv":
            return _read_csv(raw)
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl", dtype=str, keep_default_na=False)
    except DatasetImportError:
        raise
    except pd.errors.EmptyDataError:
        raise DatasetImportError("empty_file", "Ficheiru mamuk (la iha kolona ka dadus).")
    except Exception as e:  # ParserError, BadZipFile, InvalidFileException, ...
        logger.warning("Falla lee ficheiru %s: %s", ext, e)
        raise DatasetImportError(
            "unreadable",
            f"La bele lee ficheiru {ext}. Verifika se ficheiru la korruptu no formatu loos.",
        )


# ----------------------------------------------------------------------
# Valida, mapa kolona, hamoos
# ----------------------------------------------------------------------
def _map_columns(df: pd.DataFrame) -> dict:
    """Retorna {kolona_kanónika: naran_orijinál}; kolona dahuluk ne'ebé matxa mak manán."""
    mapping = {}
    for original in df.columns:
        norm = _norm_header(original)
        for canonical, aliases in COLUMN_ALIASES.items():
            if norm in aliases and canonical not in mapping:
                mapping[canonical] = original
                break
    return mapping


def _clean_cell(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    return s.mask(s.str.lower().isin(_EMPTY_TOKENS), "")


def import_dataset(source, filename: str | None = None) -> ImportResult:
    """
    Importa no hamoos dataset notísia.

    `source`: path (str/Path) ka file-like (BytesIO, Streamlit UploadedFile).
    Retorna `ImportResult(df, report)` ho df iha formatu `RAW_FIELDS`.
    Hasa'e `DatasetImportError` se ficheiru la válidu.
    """
    raw, name = _resolve_source(source, filename)
    ext = os.path.splitext(name or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetImportError(
            "unsupported_format",
            f"Formatu '{ext or name or '?'}' la suporta. Uza ficheiru .csv ka .xlsx.",
        )

    table = _read_table(raw, ext)
    report = ImportReport(filename=name, file_type=ext.lstrip("."), rows_read=len(table))

    if len(table.columns) == 0 or table.empty:
        raise DatasetImportError("empty_dataset", "Dataset mamuk: la iha liña notísia iha ficheiru.")

    mapping = _map_columns(table)
    report.mapped_columns = {k: str(v) for k, v in mapping.items()}

    missing = []
    if "title" not in mapping:
        missing.append("title")
    if "content" not in mapping and "summary" not in mapping:
        missing.append("content")
    if missing:
        labels = {"title": "títulu (title)", "content": "konteúdu (content ka summary)"}
        raise DatasetImportError(
            "missing_columns",
            "Koluna ne'ebé falta: " + ", ".join(labels[m] for m in missing)
            + f". Koluna ne'ebé iha: {', '.join(map(str, table.columns))}.",
            missing=missing,
        )

    out = pd.DataFrame(index=table.index)
    for col in _TEXT_COLUMNS:
        out[col] = _clean_cell(table[mapping[col]]) if col in mapping else ""

    published_raw = _clean_cell(table[mapping["published_at"]]) if "published_at" in mapping else pd.Series("", index=table.index)
    parsed = pd.to_datetime(published_raw.replace("", None), errors="coerce", utc=True, format="mixed")
    report.invalid_dates = int((parsed.isna() & (published_raw != "")).sum())
    out["published_at"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00").fillna("")

    # 1) Hasai liña mamuk (la iha títulu NO la iha konteúdu/rezumu)
    has_text = (out["title"] != "") | (out["content"] != "") | (out["summary"] != "")
    report.rows_removed_empty = int((~has_text).sum())
    out = out[has_text]
    if out.empty:
        raise DatasetImportError(
            "no_valid_rows", "La iha liña válidu: liña hotu-hotu mamuk (la iha títulu no konteúdu)."
        )

    # 2) news_id estável (link, se la iha: títulu+konteúdu) -> hasai duplikadu
    body = out["content"].where(out["content"] != "", out["summary"])

    def _key(row_link, row_title, row_body):
        if row_link:
            return row_link
        return " ".join(f"{row_title}|{row_body[:200]}".lower().split())

    keys = [_key(l, t, b) for l, t, b in zip(out["link"], out["title"], body)]
    out["news_id"] = [_make_news_id(k, t) for k, t in zip(keys, out["title"])]
    before = len(out)
    out = out.drop_duplicates(subset=["news_id"], keep="first")
    report.rows_removed_duplicate = before - len(out)

    # 3) Kompleta kolona ne'ebé falta ba formatu RAW_FIELDS
    default_source = os.path.splitext(os.path.basename(name))[0] or "Import"
    out["source"] = out["source"].where(out["source"] != "", default_source)
    out["fetched_at"] = datetime.now().isoformat(timespec="seconds")

    out = out[RAW_FIELDS].reset_index(drop=True)
    report.rows_valid = len(out)
    logger.info(report.summary_tetun())
    return ImportResult(df=out, report=report)
