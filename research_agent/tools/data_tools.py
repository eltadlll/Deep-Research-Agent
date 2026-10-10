"""Loading, cleaning and profiling tabular datasets."""
from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MAX_ROWS = 500_000
TIME_NAME = re.compile(r"(^|_|\b)(year|yr|date|time|period|month|day|quarter)(\b|_|$)", re.I)
ID_NAME = re.compile(r"(^|_)(id|code|iso\d?|uuid|key|index)(_|$)", re.I)


def _read_csv(path: str) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                sample = f.read(20000)
            try:
                delim = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except Exception:
                delim = ","
            return pd.read_csv(path, sep=delim, encoding=enc, nrows=MAX_ROWS, on_bad_lines="skip", low_memory=False)
        except Exception as e:  # noqa: BLE001
            last = e
    raise ValueError(f"cannot parse CSV: {last}")


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith((".csv", ".xlsx", ".json"))]
            if not names:
                raise ValueError("zip has no tabular files")
            out = p.parent / Path(names[0]).name
            out.write_bytes(z.read(names[0]))
            return load_table(str(out))
    if ext in (".csv", ".tsv", ".txt"):
        return _read_csv(path)
    if ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None)
        if not sheets:
            raise ValueError("empty workbook")
        return max(sheets.values(), key=lambda d: d.shape[0] * d.shape[1])
    if ext == ".json":
        try:
            return pd.read_json(path)
        except Exception:
            data = json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        data = v
                        break
            return pd.json_normalize(data)
    if ext == ".parquet":
        return pd.read_parquet(path)
    return _read_csv(path)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = df.loc[:, [not str(c).lower().startswith("unnamed") or df[c].notna().mean() > 0.5 for c in df.columns]]
    for c in list(df.columns):
        s = df[c]
        if s.dtype == object or str(s.dtype).startswith(("string", "str")):
            sample = s.dropna().astype(str).head(2000)
            if sample.empty:
                continue
            stripped = sample.str.replace(r"[,\s$%€£]", "", regex=True)
            num = pd.to_numeric(stripped, errors="coerce")
            if num.notna().mean() >= 0.85 and not ID_NAME.search(c):
                df[c] = pd.to_numeric(s.astype(str).str.replace(r"[,\s$%€£]", "", regex=True), errors="coerce")
            elif TIME_NAME.search(c):
                try:
                    parsed = pd.to_datetime(s, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        df[c] = parsed
                except Exception:
                    pass
    return df.drop_duplicates().reset_index(drop=True)


def col_kind(df: pd.DataFrame, c: str) -> str:
    s = df[c]
    if pd.api.types.is_datetime64_any_dtype(s):
        return "time"
    if pd.api.types.is_numeric_dtype(s):
        name_is_year = bool(re.search(r"(^|_|\b)(year|yr)(\b|_|$)", c, re.I))
        if name_is_year and s.dropna().between(1000, 2200).all():
            return "time"
        if ID_NAME.search(c) and s.nunique() > 0.9 * max(1, len(s)):
            return "id"
        return "numeric"
    nun = s.nunique(dropna=True)
    if nun <= max(50, 0.05 * len(s)):
        return "categorical"
    return "text"


def profile_df(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        k = col_kind(df, c)
        s = df[c]
        info = {"name": c, "kind": k, "dtype": str(s.dtype), "null_pct": round(float(s.isna().mean() * 100), 1),
                "unique": int(s.nunique(dropna=True))}
        if k in ("numeric", "time") and not pd.api.types.is_datetime64_any_dtype(s):
            info.update(min=float(s.min()), max=float(s.max()), mean=float(s.mean()))
        elif k == "time":
            info.update(min=str(s.min())[:10], max=str(s.max())[:10])
        elif k == "categorical":
            info["top"] = [str(v) for v in s.value_counts().head(5).index]
        cols.append(info)
    head = df.head(4).astype(str).to_dict(orient="records")
    return {"rows": int(len(df)), "cols": int(df.shape[1]), "columns": cols, "head": head}


def fmt_num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return "-"
    a = abs(f)
    if a >= 1e12:
        return f"{f/1e12:.2f}T"
    if a >= 1e9:
        return f"{f/1e9:.2f}B"
    if a >= 1e6:
        return f"{f/1e6:.2f}M"
    if f == int(f) and a < 1e6:
        return f"{int(f):,}"
    if a >= 100:
        return f"{f:,.1f}"
    return f"{f:.3g}"
