"""
Hakerek dadus ba formatu tolu iha tempu hanesan: CSV, JSON no SQLite.

Hakerek lia-lian (streaming): liña sira tama ba ficheiru ho lote ki'ik, la rai hotu iha memória.
Bele hatuur limite tamañu CSV (ezemplu 200 MB): bainhira liña tuir mai sei halo ficheiru liu
limite, hakerek para no `cap_hit` sai True.
"""
import csv
import io
import json
import math
import os
import sqlite3

BOM = "\ufeff".encode("utf-8")


def plain(value):
    """Troka valór numpy/pandas ba tipu Python normál; NaN sai None."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class MultiFormatWriter:
    def __init__(self, directory: str, basename: str, columns: list, *,
                 formats=("csv", "json", "sqlite"), max_csv_bytes=None,
                 int_columns=(), key_column=None, table: str = "articles"):
        os.makedirs(directory, exist_ok=True)
        self.columns = list(columns)
        self.formats = tuple(formats)
        self.max_csv_bytes = max_csv_bytes
        self.table = table
        self.key_column = key_column
        self.rows = 0
        self.csv_bytes = 0
        self.cap_hit = False
        self._closed = False
        self.paths = {}
        self._csv = self._json = self._db = None
        self._json_count = 0

        if "csv" in self.formats:
            self.paths["csv"] = os.path.join(directory, f"{basename}.csv")
            self._csv = open(self.paths["csv"], "wb")
            self._buf = io.StringIO()
            self._writer = csv.writer(self._buf, lineterminator="\n")
            head = BOM + self._line(self.columns)
            self._csv.write(head)
            self.csv_bytes = len(head)
        if "json" in self.formats:
            self.paths["json"] = os.path.join(directory, f"{basename}.json")
            self._json = open(self.paths["json"], "w", encoding="utf-8")
            self._json.write("[\n")
        if "sqlite" in self.formats:
            self.paths["sqlite"] = os.path.join(directory, f"{basename}.db")
            if os.path.exists(self.paths["sqlite"]):
                os.remove(self.paths["sqlite"])
            self._db = sqlite3.connect(self.paths["sqlite"])
            defs = []
            for c in self.columns:
                kind = "INTEGER" if c in int_columns else "TEXT"
                pk = " PRIMARY KEY" if c == key_column else ""
                defs.append(f'"{c}" {kind}{pk}')
            self._db.execute(f'CREATE TABLE "{table}" ({", ".join(defs)})')
            self._db.commit()
            marks = ", ".join("?" for _ in self.columns)
            names = ", ".join(f'"{c}"' for c in self.columns)
            self._insert_sql = f'INSERT OR IGNORE INTO "{table}" ({names}) VALUES ({marks})'

    def _line(self, values) -> bytes:
        self._buf.seek(0)
        self._buf.truncate()
        self._writer.writerow(["" if v is None else v for v in values])
        return self._buf.getvalue().encode("utf-8")

    def write_rows(self, rows) -> int:
        """Hakerek liña sira tuir orden. Retorna kuantidade liña ne'ebé hakerek ona."""
        json_parts, db_rows, written = [], [], 0
        for raw in rows:
            row = {c: plain(raw.get(c)) for c in self.columns}
            if self._csv is not None:
                line = self._line([row[c] for c in self.columns])
                if self.max_csv_bytes is not None and self.csv_bytes + len(line) > self.max_csv_bytes:
                    self.cap_hit = True
                    break
                self._csv.write(line)
                self.csv_bytes += len(line)
            if self._json is not None:
                sep = ",\n" if (self._json_count + len(json_parts)) else ""
                json_parts.append(sep + json.dumps(row, ensure_ascii=False))
            if self._db is not None:
                db_rows.append(tuple(row[c] for c in self.columns))
            written += 1
        if json_parts:
            self._json.write("".join(json_parts))
            self._json_count += len(json_parts)
        if db_rows:
            self._db.executemany(self._insert_sql, db_rows)
            self._db.commit()
        if self._csv is not None:
            self._csv.flush()
        self.rows += written
        return written

    def close(self) -> dict:
        """Taka ficheiru hotu no retorna tamañu (bytes) kada formatu."""
        if not self._closed:
            self._closed = True
            if self._csv is not None:
                self._csv.close()
            if self._json is not None:
                self._json.write("\n]\n")
                self._json.close()
            if self._db is not None:
                self._db.commit()
                self._db.close()
        return {k: os.path.getsize(p) for k, p in self.paths.items() if os.path.exists(p)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
