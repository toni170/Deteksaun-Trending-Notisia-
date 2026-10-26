"""Fixture bersama: dataset sintétiku Tetun, direktóriu temporáriu, registu modelu."""
import io
import random

import pandas as pd
import pytest

from src.training.model_registry import ModelRegistry

FILLERS = ["iha", "ba", "husi", "ne'e", "atu", "nia", "sira"]  # stopword Tetun -> language = "tet"

TOPICS_A = {
    "edukasaun": "eskola estudante ensinu prufesor kurrikulu universidade exame matrikula alunu bolsa diploma graduasaun".split(),
    "saude": "ospital mediku vasina moras klinika enfermeira tratamentu pasiente farmasia doente pandemia medikamentu".split(),
    "ekonomia": "banku orsamentu investimentu merkadu presu petroleu kompania finansa importasaun exportasaun negosiu industria".split(),
}
TOPICS_B = {
    "desportu": "bola futebol atleta kompetisaun treinador kampeaun estadiu selesaun jogadór torneiu medalla ekipa".split(),
    "agrikultura": "kafe produtor arroz agrikultór irigasaun kolleita tanaman fazenda sementes batar trigu kebun".split(),
    "transporte": "estrada ponte autocarro aeroportu portu viajen kareta trafegu kamiaun motór rota stasaun".split(),
}


def make_news_df(topics: dict, n_per_topic: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, i = [], 0
    for topic, vocab in topics.items():
        for _ in range(n_per_topic):
            w = rng.sample(vocab, 8)
            f = rng.sample(FILLERS, 4)
            rows.append({
                "title": f"{w[0]} {w[1]} {f[0]} {w[2]}",
                "content": f"{w[3]} {f[1]} {w[4]} {w[5]} {f[2]} {w[6]} {f[3]} {w[7]} {w[0]}",
                "published_at": (pd.Timestamp("2024-03-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "link": f"https://example.test/{seed}/{i}",
                "_topic": topic,
            })
            i += 1
    return pd.DataFrame(rows)


@pytest.fixture
def news_df():
    return make_news_df(TOPICS_A, seed=1)


@pytest.fixture
def news_df_b():
    return make_news_df(TOPICS_B, seed=2)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.drop(columns=["_topic"], errors="ignore").to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.drop(columns=["_topic"], errors="ignore").to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


@pytest.fixture
def csv_bytes(news_df):
    return to_csv_bytes(news_df)


@pytest.fixture
def csv_bytes_b(news_df_b):
    return to_csv_bytes(news_df_b)


@pytest.fixture
def xlsx_bytes(news_df):
    return to_xlsx_bytes(news_df)


@pytest.fixture
def models_dir(tmp_path):
    return str(tmp_path / "models")


@pytest.fixture
def registry(models_dir):
    return ModelRegistry(models_dir)
