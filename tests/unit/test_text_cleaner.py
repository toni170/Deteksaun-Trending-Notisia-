"""Teste #7-#10: case folding, tokenization, stopword removal, stemming Tetun + fallback."""
import logging

import pytest

from src.preprocessing import text_cleaner as tc

pytestmark = pytest.mark.unit


# ---- #7 Case folding -------------------------------------------------------
def test_07_case_folding_makes_variants_identical():
    assert tc.clean_text("ESKOLA Foun") == tc.clean_text("eskola foun") == "eskola foun"


def test_07_case_folding_in_full_pipeline():
    a = tc.preprocess_news("ESKOLA FOUN", "ESTUDANTE iha ESKOLA")
    b = tc.preprocess_news("eskola foun", "estudante iha eskola")
    assert a["text_final"] == b["text_final"]
    assert a["text_final"] == a["text_final"].lower()


@pytest.mark.xfail(strict=True, reason=(
    "DEFEITU ne'ebé hetan: clean_text hasai letra ho asentu (á é í ó ú) -> 'SAÚDE' sai 'sa de', "
    "no stopword Tetun ho asentu ('mós','maibé','sé') nunka matxa. Fix = troka NFD-strip la'ós hasai."))
def test_07_accented_letters_survive_cleaning():
    assert tc.clean_text("SAÚDE mós") == "saúde mós"


# ---- #8 Tokenization -------------------------------------------------------
def test_08_tokenization_removes_html_url_punctuation_keeps_apostrophe():
    cleaned = tc.clean_text("<p>Ne'e notísia, foun!!! Hare https://x.tl/a?b=1 no www.y.com</p>")
    assert "<" not in cleaned and "http" not in cleaned and "www" not in cleaned
    assert "ne'e" in cleaned.split()
    assert all(tok == tok.strip() and tok for tok in cleaned.split())


def test_08_repeated_characters_are_capped_and_spaces_collapsed():
    assert tc.clean_text("Boooooot    sa'e") == "booot sa'e"


def test_08_numbers_are_kept_and_empty_inputs_are_safe():
    assert tc.clean_text("Orsamentu 2024") == "orsamentu 2024"
    for bad in (None, float("nan"), "", "   ", 123):
        assert tc.clean_text(bad) == ""


def test_08_preprocess_news_token_count_matches_final_text():
    out = tc.preprocess_news("Eskola foun iha Dili", "Estudante sira hanesan atu aprende")
    assert out["token_count"] == len(out["text_final"].split()) > 0


def test_08_nan_title_does_not_become_token_nan():  # regression bug 'nan'
    assert "nan" not in tc.preprocess_news(float("nan"), "Estudante eskola ensinu foun")["text_final"].split()


# ---- #9 Stopword removal ---------------------------------------------------
def test_09_tetun_stopwords_removed():
    toks = "eskola foun iha ba husi nia sira atu estudante".split()
    assert tc.remove_stopwords(toks, "tet") == ["eskola", "foun", "estudante"]


def test_09_indonesian_and_english_stopwords_removed():
    assert tc.remove_stopwords(["yang", "sekolah", "dan", "guru"], "id") == ["sekolah", "guru"]
    assert tc.remove_stopwords(["the", "school", "and", "teacher"], "en") == ["school", "teacher"]


def test_09_unknown_language_uses_union_and_short_tokens_dropped():
    assert tc.remove_stopwords(["yang", "the", "iha", "xy", "eskola"], "unknown") == ["eskola"]


def test_09_language_detection_picks_tetun_for_tetun_text():
    assert tc.detect_language(tc.clean_text("estudante iha eskola husi nia sira atu hatene")) == "tet"
    assert tc.detect_language("") == "unknown"


# ---- #10 Stemming Tetun no fallback ----------------------------------------
def test_10_tetun_heuristic_stemmer_strips_suffix_and_respects_limits():
    assert tc.stem_tokens(["eskolasira", "estudantenain"], "tet") == ["eskola", "estudante"]
    assert tc._stem_tetun_token("ida") == "ida"          # token <= 4 la muda
    assert tc._stem_tetun_token("abcsira") == "abc"      # stem minimu 3 letra
    assert tc._stem_tetun_token("absira") == "absira"    # stem <3 -> la muda


def test_10_unknown_language_falls_back_to_original_tokens():
    assert tc.stem_tokens(["eskolasira", "xyz"], "unknown") == ["eskolasira", "xyz"]


def test_10_stemmer_failure_falls_back_to_original_tokens(monkeypatch, caplog):
    def boom():
        raise RuntimeError("kamus la hetan")
    monkeypatch.setattr(tc, "_get_id_stemmer", boom)
    with caplog.at_level(logging.WARNING, logger="text_cleaner"):
        assert tc.stem_tokens(["pemerintahan"], "id") == ["pemerintahan"]
    assert "Stemming gagal" in caplog.text


def test_10_indonesian_and_english_stemmers_work():
    assert tc.stem_tokens(["pemerintahan"], "id") == ["perintah"]
    assert tc.stem_tokens(["running", "schools"], "en") == ["run", "school"]


# ---- Etapa preprosesamentu (ba dashboard): explain_preprocessing -------------
import random
import re
import string

SAMPLES = [
    ("<b>Eskolasira FOUN</b>", "Estudante iha eskola HTTP://X.COM !!!!! ne'e"),
    ("Governu hatete", "Ospital foun iha Dili ba ema sira"),
    ("Pemerintah membangun sekolah baru", "yang dan di untuk berlarian"),
    ("The government building schools", "and running the teachers"),
    ("zzzz qqqq", "xx yy"), ("", ""), (float("nan"), float("nan")), ("!!!", "???"),
    ("Titulu", None), (None, "Isi de'it www.site.tl/a?b=1 <p>HTML</p>"),
]


@pytest.mark.parametrize("title,summary", SAMPLES)
def test_explain_preprocessing_matches_preprocess_news(title, summary):
    exp, ref = tc.explain_preprocessing(title, summary), tc.preprocess_news(title, summary)
    assert exp["text_final"] == ref["text_final"]
    assert exp["text_clean"] == ref["text_clean"] and exp["language"] == ref["language"]
    assert exp["token_count"] == ref["token_count"]


def test_explain_preprocessing_stage_contract():
    r = tc.explain_preprocessing("<b>Eskolasira FOUN</b>", "Estudante iha eskola HTTP://X.COM !!!!! ne'e")
    assert "<b>" in r["raw"] and "<b>" not in r["after_html"] and "FOUN" in r["after_html"]       # cleaning 1
    assert r["after_case_folding"] == r["after_html"].lower()                                       # case folding
    assert "http" not in r["text_clean"] and "!" not in r["text_clean"]                             # cleaning 2
    assert r["tokens"] == r["text_clean"].split()                                                   # tokenization
    assert r["removed_tokens"] == ["iha", "ne'e"]                                                   # stopword
    assert sorted(r["tokens"]) == sorted(r["tokens_no_stopwords"] + r["removed_tokens"])
    assert r["stem_changes"] == [("eskolasira", "eskola")] and r["stemming_applied"]                # stemming
    assert r["text_final"] == "eskola foun estudante eskola"


def test_explain_preprocessing_unknown_language_skips_stemming():
    r = tc.explain_preprocessing("zzzz qqqq", "wwwww")
    assert r["language"] == "unknown" and not r["stemming_applied"] and r["stem_changes"] == []
    assert r["tokens_stemmed"] == r["tokens_no_stopwords"]


def test_clean_text_refactor_is_behavior_identical_to_original_on_random_input():
    H, U, M = re.compile(r"<[^>]+>"), re.compile(r"http\S+|www\.\S+"), re.compile(r"\s+")
    N, R = re.compile(r"[^a-z0-9\s']"), re.compile(r"(.)\1{3,}")

    def original(text):  # implementasaun orijinál husi src.zip
        if not isinstance(text, str) or not text.strip():
            return ""
        t = H.sub(" ", text).lower()
        t = N.sub(" ", R.sub(r"\1\1\1", U.sub(" ", t)))
        return M.sub(" ", t).strip()

    rng, alphabet = random.Random(7), string.ascii_letters + string.digits + " <>/'.,!?:;-_\n\t" + "áéíóúÁ" + "httpwW."
    for _ in range(3000):
        x = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
        assert tc.clean_text(x) == original(x), repr(x)
