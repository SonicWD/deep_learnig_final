# -*- coding: utf-8 -*-
"""Pruebas unitarias del preprocesamiento (guía del gestor S13)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURE_COLS, SEED, TARGET_COL
from src.preprocess import build_splits, class_weight_ratio


def _toy_frame(n: int = 1000, fraud_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n_fraud = max(int(n * fraud_rate), 2)
    data = {c: rng.normal(size=n) for c in FEATURE_COLS}
    data["Amount"] = rng.uniform(0, 500, size=n)
    y = np.zeros(n, dtype=int)
    y[:n_fraud] = 1
    rng.shuffle(y)
    data[TARGET_COL] = y
    return pd.DataFrame(data)


def test_split_stratified_proportions():
    df = _toy_frame()
    bundle = build_splits(df)
    rate = df[TARGET_COL].mean()
    for key in ("y_train", "y_val", "y_test"):
        assert abs(bundle[key].mean() - rate) < 0.02


def test_scaler_zero_mean_train():
    df = _toy_frame()
    bundle = build_splits(df)
    means = bundle["X_train"][bundle["scale_cols"]].mean().abs()
    assert (means < 1e-6).all()


def test_class_weight_positive():
    y = pd.Series([0] * 98 + [1] * 2)
    assert class_weight_ratio(y) == 49.0
