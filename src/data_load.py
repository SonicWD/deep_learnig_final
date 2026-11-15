# -*- coding: utf-8 -*-
"""Carga y resumen del dataset ULB/Kaggle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import CSV_PATH, TARGET_COL


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or CSV_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Descargue creditcard.csv en data/raw/ "
            "(ver README)."
        )
    return pd.read_csv(csv_path)


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    n_fraud = int(df[TARGET_COL].sum())
    return {
        "n_rows": n,
        "n_cols": int(df.shape[1]),
        "n_fraud": n_fraud,
        "n_legit": n - n_fraud,
        "fraud_rate": n_fraud / n if n else 0.0,
        "nulls": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


if __name__ == "__main__":
    frame = load_raw()
    info = summarize(frame)
    print(info)
