"""
Pembaca master CSV ba dashboard — hemat memória no la depende ba Streamlit (bele teste direta).

Estratéjia:
1. `read_cleaned_light`: lee ba chunk, hasai kolona la uza, tesi tékstu naruk iha kada chunk.
   Se `max_rows` fornese, mantein de'it liña ikus (foun liu, tuir orden ficheiru) -> memória limitadu.
2. `read_cleaned_with_fallback`: tenta lee hotu; se MemoryError, tenta fali ho limite liña
   ki'ik liu (100k -> 50k -> 20k -> 10k). Rezultadu marka `df.attrs["rows_total"]` bainhira ba tiha.
"""
import gc
from collections import deque

import pandas as pd

# Kolom yang tidak dipakai dashboard -> dilewati saat membaca CSV
UNUSED_COLS = {"fetched_at", "token_count"}
# Batas panjang teks per sel (hanya di memori dashboard; CSV di disk tidak diubah)
MAX_CHARS = {"content": 6000, "text_clean": 3000, "text_final": 3000, "summary": 2000}
FALLBACK_ROW_LIMITS = (100_000, 50_000, 20_000, 10_000)
FALLBACK_CHUNKSIZE = 500  # chunk ki'ik -> pik memória durante parsing menus


def read_cleaned_light(path: str, chunksize: int = 2000, max_rows: int | None = None) -> pd.DataFrame:
    # +1: chunk ikus bele la kompletu, entaun presiza chunk extra atu garante >= max_rows liña
    max_chunks = None if not max_rows else -(-max_rows // chunksize) + 1
    parts = deque(maxlen=max_chunks)  # deque ho maxlen -> chunk tuan automatikamente sai
    total = 0
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=lambda c: c not in UNUSED_COLS):
        for col, limit in MAX_CHARS.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].map(lambda v, n=limit: v[:n] if isinstance(v, str) else v)
        total += len(chunk)
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    del parts
    if max_rows and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
    if total > len(df):
        df.attrs["rows_total"] = total
    return df


def read_cleaned_with_fallback(path: str) -> pd.DataFrame:
    """Lee hotu; se memória la to'o, mantein liña foun liu de'it. Hasa'e MemoryError se hotu-hotu falla."""
    try:
        return read_cleaned_light(path)
    except MemoryError:
        gc.collect()
    for limit in FALLBACK_ROW_LIMITS:
        try:
            return read_cleaned_light(path, chunksize=FALLBACK_CHUNKSIZE, max_rows=limit)
        except MemoryError:
            gc.collect()
    raise MemoryError(f"Memória la to'o atu lee '{path}' (to'o ho limite {FALLBACK_ROW_LIMITS[-1]} liña).")


def collected_data_exists(path: str) -> bool:
    """True se ficheiru master iha no iha liña dadus (la'ós header de'it)."""
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            f.readline()  # header
            return bool(f.readline().strip())
    except OSError:
        return False
