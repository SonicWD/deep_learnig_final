# -*- coding: utf-8 -*-
"""Genera un CSV sintético con el mismo esquema ULB para pruebas locales.

No sustituye al dataset real de Kaggle en la entrega final; sirve para
validar el pipeline cuando creditcard.csv aún no está descargado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CSV_PATH, FEATURE_COLS, SEED, TARGET_COL


def main(n: int = 20000, fraud_rate: float = 0.002):
    rng = np.random.default_rng(SEED)
    n_fraud = max(int(n * fraud_rate), 20)
    rows = []
    for i in range(n):
        is_fraud = i < n_fraud
        row = {f"V{j}": float(rng.normal(2.0 if is_fraud else 0.0, 1.2)) for j in range(1, 29)}
        row["Time"] = float(i)
        row["Amount"] = float(rng.uniform(50, 800) if is_fraud else rng.uniform(1, 200))
        row[TARGET_COL] = int(is_fraud)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # orden de columnas como Kaggle
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", TARGET_COL]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(CSV_PATH, index=False)
    print(f"Escrito {CSV_PATH} filas={len(df)} fraude={df[TARGET_COL].sum()}")


if __name__ == "__main__":
    main()
