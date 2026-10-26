"""
Pipeline preprocessing: data/raw (RSS) -> pembersihan teks, deteksi bahasa,
stopword removal, stemming -> data/cleaned.
"""
import os
import logging
from datetime import datetime

import pandas as pd

from src.preprocessing.text_cleaner import preprocess_news
from src.preprocessing.category_classifier import classify_dataframe_categories

logger = logging.getLogger("preprocessing")


def _pick_body_text(row) -> str:
    """
    Pilih teks isi berita yang dipakai untuk preprocessing:
    - Jika hasil web scraping (`content`, isi artikel LENGKAP) tersedia & tidak kosong, pakai itu.
    - Jika tidak (mis. hanya koleksi via RSS biasa, atau scraping gagal), fallback ke `summary`.
    """
    content = row.get("content", "")
    if isinstance(content, str) and content.strip():
        return content
    summary = row.get("summary", "")
    return summary if isinstance(summary, str) else ""


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan pembersihan + stemming ke setiap baris berita (judul + isi/ringkasan)."""
    if df.empty:
        return df.assign(text_clean="", language="", text_final="", token_count=0)

    results = df.apply(
        lambda row: preprocess_news(row.get("title", ""), _pick_body_text(row)),
        axis=1,
    )
    result_df = pd.DataFrame(list(results))

    out = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
    # Buang berita yang setelah dibersihkan jadi kosong
    out = out[out["text_final"].str.strip() != ""].reset_index(drop=True)

    # Kategoria husi web scraping (breadcrumb/tag CSS) la fiável — barak vezes
    # "Home" ka vazia. Substitui/normaliza ho klasifikadór liafuan-xave atu
    # kada notísia hetan kategoria husi lista fixu: Edukasaun, Politika,
    # Saúde, Ekonomia (ka "Jerál" se la matxa ho ida ne'ebé).
    out["category"] = classify_dataframe_categories(out)

    return out


def run_preprocessing(raw_csv_path: str, cleaned_dir: str) -> str:
    """
    Baca CSV mentah (master atau snapshot) dari data/raw, jalankan preprocessing,
    simpan hasil ke data/cleaned (baik snapshot timestamped maupun master kumulatif).
    Mengembalikan path master file hasil (dipakai untuk clustering & dashboard).
    """
    os.makedirs(cleaned_dir, exist_ok=True)

    df = pd.read_csv(raw_csv_path)
    logger.info("Memuat %s berita mentah dari %s", len(df), raw_csv_path)

    cleaned_df = preprocess_dataframe(df)
    logger.info("Preprocessing selesai. %s berita tersisa setelah pembersihan.", len(cleaned_df))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = os.path.join(cleaned_dir, f"tatoli_cleaned_{stamp}.csv")
    cleaned_df.to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    master_path = os.path.join(cleaned_dir, "tatoli_cleaned_master.csv")
    cleaned_df = cleaned_df.drop_duplicates(subset=["news_id"], keep="last")
    cleaned_df.to_csv(master_path, index=False, encoding="utf-8-sig")

    logger.info("Data bersih disimpan ke %s (snapshot) dan %s (master)", snapshot_path, master_path)

    return master_path
