"""
Klasifikadór kategoria notísia (baze ba liafuan-xave) — Edukasaun, Politika,
Saúde, Ekonomia.

Kontestu: kategoria ne'ebé `src/scraping/web_scraper.py` foti husi HTML
Tatoli (breadcrumb/tag CSS selector) la fiável — barak vezes fila fali
"Home", vazia, ka valór jenériku seluk ne'ebé la koresponde ba kategoria
temátiku ne'ebé ita hakarak analiza (Edukasaun/Politika/Saúde/Ekonomia).

Módulu ida-ne'e substitui/kompleta kategoria ne'e liuhusi klasifikasaun
baze ba liafuan-xave (multilingue: Tetun, Indonézia, Inglés) aplika ba
títulu + konteúdu kada notísia. Abordajen simples (rule-based) — sufisiente
ba dominiu limitadu (4 kategoria) no la presiza treinu modelu adisionál.
"""
import re
import unicodedata

# Kategoria prinsipál ne'ebé ita monitor/analiza (mesmu lista ho
# `TRACKED_CATEGORIES` iha dashboard).
CATEGORIES = ["Edukasaun", "Politika", "Saúde", "Ekonomia"]

# Kategoria fallback bainhira notísia la matxa ho liafuan-xave husi
# kategoria 4 nia leten (ex: notísia desportu, kultura, aksidente, etc.)
FALLBACK_CATEGORY = "Jerál"

# Se kategoria ne'ebé site fornese ona (scraped) matxa ho valór sira ne'e
# (case-insensitive), uza direitamente — la presiza klasifika liafuan-xave.
# Kobre etiketa oioin ne'ebé site WordPress Tatoli uza (Tetun/Indonézia/Inglés).
KNOWN_CATEGORY_MAP = {
    # Edukasaun
    "edukasaun": "Edukasaun", "pendidikan": "Edukasaun", "education": "Edukasaun",
    # Politika
    "politika": "Politika", "politik": "Politika", "politics": "Politika",
    "nasional": "Politika", "national": "Politika",
    # Saúde
    "saude": "Saúde", "saúde": "Saúde", "kesehatan": "Saúde", "health": "Saúde",
    # Ekonomia
    "ekonomia": "Ekonomia", "ekonomi": "Ekonomia", "economy": "Ekonomia",
    "business": "Ekonomia", "negosiu": "Ekonomia", "bisnis": "Ekonomia",
}

# Valór "sampah" konhesidu husi breadcrumb/menu ne'ebé NUNKA reprezenta
# kategoria temátiku real (buat balun sai husi seletór CSS ne'ebé aponta
# ba link menu jerál, la'ós ba tag kategoria artigu).
_JUNK_CATEGORY_VALUES = {"home", "página inicial", "pajina inicial", "tatoli", ""}

# ----------------------------------------------------------------------
# Liafuan-xave multilingue (Tetun / Indonézia / Inglés) kada kategoria.
# Lista ne'e la kompletu 100% — bele aumenta/ajusta tuir presiza bazeia
# ba notísia real ne'ebé sai mal-klasifika.
# ----------------------------------------------------------------------
_KEYWORDS = {
    "Edukasaun": [
        # Tetun
        "eskola", "estudante", "ensinu", "edukasaun", "prufesor", "dosente",
        "kurríkulu", "kurrikulu", "diploma", "graduasaun", "alunu", "mestre",
        "universidade", "instituisaun edukasionál", "bolsa estudu", "exame",
        "matríkula", "matrikula",
        # Indonésia
        "sekolah", "mahasiswa", "guru", "universitas", "kampus", "kurikulum",
        "beasiswa", "pendidikan",
        # Inglés
        "school", "student", "university", "teacher", "lecturer", "curriculum",
        "scholarship", "campus", "education",
    ],
    "Politika": [
        # Tetun
        "governu", "parlamentu", "ministru", "primeiru-ministru", "prezidente",
        "deputadu", "partidu", "eleisaun", "votu", "konstituisaun", "gabinete",
        "diplomasia", "embaixadór", "polítika", "bankada", "parlamentár",
        "konsellu ministru",
        # Partidu polítiku Timor-Leste (sinál forte ba notísia polítika)
        "cnrt", "fretilin", "plp", "khunto", "pd", "udt",
        # Indonésia
        "pemerintah", "menteri", "presiden", "partai", "pemilu", "parlemen",
        "politik",
        # Inglés
        "government", "minister", "president", "parliament", "election",
        "party", "vote", "diplomat", "politics", "policy",
    ],
    "Saúde": [
        # Tetun
        "saude", "saúde", "ospital", "klínika", "klinika", "doente", "mediku",
        "enfermeira", "vasina", "moras", "pandemia", "tratamentu",
        "servisu saúde", "farmásia", "farmasia",
        # Indonésia
        "kesehatan", "rumah sakit", "dokter", "vaksin", "penyakit", "puskesmas",
        # Inglés
        "health", "hospital", "doctor", "nurse", "vaccine", "disease",
        "clinic", "patient", "medical",
    ],
    "Ekonomia": [
        # Tetun
        "ekonomia", "orsamentu", "investimentu", "negosiu", "kompañia",
        "kompania", "merkadu", "presu", "importasaun", "exportasaun", "taxa",
        "banku", "indústria", "industria", "finansa", "petroleu", "petróleu",
        "gas", "orsamentu jerál", "greater sunrise", "timor gap", "dollar",
        # Indonésia
        "ekonomi", "anggaran", "investasi", "bisnis", "pasar", "harga",
        "ekspor", "impor", "bank", "industri",
        # Inglés
        "economy", "budget", "investment", "business", "market", "price",
        "trade", "export", "import", "bank", "finance", "industry", "oil",
        "pipeline",
    ],
}

# Título pesa liu duke konteúdu — liafuan-xave ne'ebé mosu iha título mais
# provável reprezenta tópiku prinsipál artigu ne'e.
_TITLE_WEIGHT = 3
_CONTENT_WEIGHT = 1

_MULTISPACE_RE = re.compile(r"\s+")
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s]")


def _normalize(text: str) -> str:
    """
    Lowercase + hasai asentu (á->a, ú->u, ñ->n, dll) + hasai pontuasaun,
    maibé HALA'O buka liafuan-xave sedauk hasai letra hanesan `clean_text`
    (ne'ebé bele estraga liafuan ho asentu tanba hasai letra la'ós substitui).
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    no_accents = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    t = no_accents.lower()
    t = _NON_ALPHANUM_RE.sub(" ", t)
    return _MULTISPACE_RE.sub(" ", t).strip()


def _normalize_keyword(kw: str) -> str:
    return _normalize(kw)


# Pre-normaliza liafuan-xave dala ida de'it (la iha asentu) atu buka lalais.
_NORMALIZED_KEYWORDS = {
    cat: [_normalize_keyword(kw) for kw in kws] for cat, kws in _KEYWORDS.items()
}


def _score_text(norm_title: str, norm_content: str) -> dict:
    scores = {cat: 0 for cat in CATEGORIES}
    for cat, kws in _NORMALIZED_KEYWORDS.items():
        for kw in kws:
            if not kw:
                continue
            scores[cat] += norm_title.count(kw) * _TITLE_WEIGHT
            scores[cat] += norm_content.count(kw) * _CONTENT_WEIGHT
    return scores


def classify_from_known_category(raw_category: str) -> str | None:
    """
    Se kategoria ne'ebé site fornese ona (scraped) rekonhesivel diretamente
    (ex: 'Politics', 'Kesehatan'), retorna kategoria padronizadu; se
    'sampah' konhesidu (ex: 'Home') ka la rekonhesivel, retorna None
    (sinál atu tenta klasifika liafuan-xave).
    """
    if not isinstance(raw_category, str):
        return None
    key = _normalize(raw_category).strip()
    if key in _JUNK_CATEGORY_VALUES:
        return None
    return KNOWN_CATEGORY_MAP.get(key)


def classify_category(title: str, content: str, raw_category: str = "") -> str:
    """
    Determina kategoria final ba notísia ida:
    1. Se kategoria scraped ona rekonhesivel diretamente -> uza ne'e.
    2. Se lae, klasifika bazeia ba liafuan-xave iha título + konteúdu.
    3. Se la iha matxa forte ida (hotu-hotu skór 0) -> `FALLBACK_CATEGORY`.
    """
    known = classify_from_known_category(raw_category)
    if known:
        return known

    norm_title = _normalize(title)
    norm_content = _normalize(content)
    if not norm_title and not norm_content:
        return FALLBACK_CATEGORY

    scores = _score_text(norm_title, norm_content)
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] <= 0:
        return FALLBACK_CATEGORY
    return best_cat


def classify_dataframe_categories(df):
    """
    Aplika `classify_category` ba kada liña DataFrame notísia (kolona
    `title`, `content` ka `summary` fallback, no `category` scraped orijinál
    se existe). Retorna Series foun ho kategoria padronizadu.
    """
    def _txt(value) -> str:
        # NaN (float) "truthy" — `nan or x` fila nan, la'ós x. Tenke verifika tipu.
        return value if isinstance(value, str) and value.strip() else ""

    def _row_category(row):
        title = _txt(row.get("title", ""))
        content = _txt(row.get("content", "")) or _txt(row.get("summary", ""))
        raw_category = _txt(row.get("category", ""))
        return classify_category(title, content, raw_category)

    return df.apply(_row_category, axis=1)
