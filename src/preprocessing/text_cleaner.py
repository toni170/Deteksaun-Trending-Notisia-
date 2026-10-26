"""
Pembersihan teks berita & normalisasi bahasa (Tetun, Indonesia, English) untuk
Mining Notísia Online no Deteksaun Topiku.

Termasuk stemming/lemmatization ringan per bahasa:
- Indonesia -> Sastrawi StemmerFactory (kamus bawaan, tidak perlu internet)
- English   -> nltk.stem.porter.PorterStemmer (algoritmik, tidak perlu download corpus)
- Tetun     -> heuristik suffix-stripping manual (tidak ada library stemmer resmi)
"""
import re
import unicodedata
import logging

from src.preprocessing.stopwords_data import STOPWORDS_ID, STOPWORDS_EN, STOPWORDS_TET

logger = logging.getLogger("text_cleaner")

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s']")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")

ALL_STOPWORDS = {
    "id": STOPWORDS_ID,
    "en": STOPWORDS_EN,
    "tet": STOPWORDS_TET,
}

# --- Stemmer Indonesia (Sastrawi) — lazy init, kamus bawaan offline ---
_id_stemmer = None


def _get_id_stemmer():
    global _id_stemmer
    if _id_stemmer is None:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        _id_stemmer = StemmerFactory().create_stemmer()
    return _id_stemmer


# --- Stemmer English (Porter, algoritmik — tidak perlu download nltk corpus) ---
_en_stemmer = None


def _get_en_stemmer():
    global _en_stemmer
    if _en_stemmer is None:
        from nltk.stem.porter import PorterStemmer
        _en_stemmer = PorterStemmer()
    return _en_stemmer


# --- Stemmer Tetun (heuristik manual, tidak ada library resmi) ---
_TET_SUFFIXES = ["nain", "sira", "ida", "na", "n"]


def _stem_tetun_token(token: str) -> str:
    """
    Heuristik ringan pemotongan sufiks umum bahasa Tetun (mis. penanda jamak
    'sira', partikel 'na'). Bukan stemmer linguistik penuh, hanya normalisasi
    kasar agar variasi kata sedikit lebih menyatu untuk TF-IDF.
    """
    if len(token) <= 4:
        return token
    for suf in _TET_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def remove_html(text: str) -> str:
    """Cleaning 1/2: hasai tag HTML ne'ebé sobra husi RSS."""
    return _HTML_TAG_RE.sub(" ", text)


def case_fold(text: str) -> str:
    """Case folding: hadi'a hotu ba letra ki'ik."""
    return text.lower()


def remove_noise(text: str) -> str:
    """
    Cleaning 2/2 (tuir orden): hasai URL, karakter repete (>3x), tanda baca
    (apóstrofu mantein — importante ba Tetun: ne'e), no rapi espasu.
    """
    t = _URL_RE.sub(" ", text)
    t = _REPEATED_CHAR_RE.sub(r"\1\1\1", t)
    t = _NON_ALPHANUM_RE.sub(" ", t)
    return _MULTISPACE_RE.sub(" ", t).strip()


def clean_text(text: str) -> str:
    """
    Membersihkan teks berita: remove_html -> case_fold -> remove_noise.
    (Orden importante: case_fold tenke molok remove_noise tanba regex hasai letra bo'ot.)
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    return remove_noise(case_fold(remove_html(text)))


def detect_language(text_clean: str) -> str:
    """
    Deteksi bahasa sederhana berbasis overlap stopwords (id / en / tet).
    Cukup memadai untuk judul & ringkasan berita Tatoli yang dominan Tetun,
    kadang Indonesia atau Inggris.
    """
    if not text_clean:
        return "unknown"

    tokens = set(text_clean.split())
    if not tokens:
        return "unknown"

    scores = {
        lang: len(tokens & {w.lower() for w in words})
        for lang, words in ALL_STOPWORDS.items()
    }

    best_lang = max(scores, key=scores.get)
    if scores[best_lang] == 0:
        tet_markers = {"iha", "nia", "sira", "ba", "husi", "ne'e", "atu"}
        if tokens & tet_markers:
            return "tet"
        return "unknown"

    return best_lang


def remove_stopwords(tokens: list, language: str) -> list:
    """Hapus stopwords sesuai bahasa terdeteksi. Untuk 'unknown', gabungkan semua."""
    if language in ALL_STOPWORDS:
        stop = ALL_STOPWORDS[language]
    else:
        stop = STOPWORDS_ID | STOPWORDS_EN | STOPWORDS_TET
    return [tok for tok in tokens if tok not in stop and len(tok) > 2]


def stem_tokens(tokens: list, language: str) -> list:
    """Terapkan stemming/lemmatization ringan sesuai bahasa terdeteksi."""
    try:
        if language == "id":
            stemmer = _get_id_stemmer()
            return [stemmer.stem(tok) for tok in tokens]
        if language == "en":
            stemmer = _get_en_stemmer()
            return [stemmer.stem(tok) for tok in tokens]
        if language == "tet":
            return [_stem_tetun_token(tok) for tok in tokens]
    except Exception as e:  # pragma: no cover - fallback aman jika stemmer gagal
        logger.warning("Stemming gagal untuk bahasa %s: %s. Menggunakan token asli.", language, e)
    return tokens


def preprocess_news(raw_title: str, raw_summary: str) -> dict:
    """
    Pipeline lengkap untuk satu berita:
    1. gabungkan judul + ringkasan
    2. clean_text
    3. deteksi bahasa
    4. tokenisasi
    5. stopword removal
    6. stemming/lemmatization sesuai bahasa
    7. gabungkan kembali jadi string bersih (untuk TF-IDF)
    """
    # NaN (float) ka None ho'o "" — se lae, f-string bele halo "nan" sai token.
    raw_title = raw_title if isinstance(raw_title, str) else ""
    raw_summary = raw_summary if isinstance(raw_summary, str) else ""
    combined_raw = f"{raw_title} . {raw_summary}"
    cleaned = clean_text(combined_raw)
    language = detect_language(cleaned)

    tokens = cleaned.split()
    tokens_no_stop = remove_stopwords(tokens, language)
    tokens_stemmed = stem_tokens(tokens_no_stop, language)

    final_text = " ".join(tokens_stemmed)

    return {
        "text_clean": cleaned,
        "language": language,
        "text_final": final_text,
        "token_count": len(tokens_stemmed),
    }


def explain_preprocessing(raw_title, raw_summary) -> dict:
    """
    Versaun "eksplika" husi `preprocess_news`: fó rezultadu ba etapa ida-idak
    (ba dashboard). Rezultadu final (`text_final`) TENKE hanesan ho `preprocess_news`
    (verifika iha teste).
    """
    raw_title = raw_title if isinstance(raw_title, str) else ""
    raw_summary = raw_summary if isinstance(raw_summary, str) else ""
    raw = f"{raw_title} . {raw_summary}"

    after_html = remove_html(raw)
    after_fold = case_fold(after_html)
    cleaned = remove_noise(after_fold)
    language = detect_language(cleaned)

    tokens = cleaned.split()
    kept = remove_stopwords(tokens, language)
    # remove_stopwords filter ho orden: desizaun hanesan ba token hanesan -> bele hatene saida hasai
    removed, j = [], 0
    for tok in tokens:
        if j < len(kept) and tok == kept[j]:
            j += 1
        else:
            removed.append(tok)

    stemmed = stem_tokens(kept, language)
    changes = list(dict.fromkeys((a, b) for a, b in zip(kept, stemmed) if a != b))

    return {
        "raw": raw,
        "after_html": after_html,
        "after_case_folding": after_fold,
        "text_clean": cleaned,
        "language": language,
        "tokens": tokens,
        "tokens_no_stopwords": kept,
        "removed_tokens": removed,
        "tokens_stemmed": stemmed,
        "stemming_applied": language in ("id", "en", "tet"),
        "stem_changes": changes,
        "text_final": " ".join(stemmed),
        "token_count": len(stemmed),
    }
