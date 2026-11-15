# -*- coding: utf-8 -*-
"""Preprocesamiento: escala, split estratificado y persistencia."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    ARTIFACTS,
    DATA_PROCESSED,
    FEATURE_COLS,
    OPTIONAL_TIME,
    SCALE_COLS,
    SEED,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
)
from src.data_load import load_raw, summarize


def resolve_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = list(FEATURE_COLS)
    scale = list(SCALE_COLS)
    if OPTIONAL_TIME in df.columns:
        features = [OPTIONAL_TIME] + features
        scale = [OPTIONAL_TIME] + scale
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return features, scale


def build_splits(df: pd.DataFrame):
    features, scale_cols = resolve_features(df)
    x = df[features].copy()
    y = df[TARGET_COL].astype(int).copy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=TEST_SIZE + VAL_SIZE, stratify=y, random_state=SEED,
    )
    relative_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=1 - relative_val, stratify=y_temp, random_state=SEED,
    )

    scaler = StandardScaler()
    x_train.loc[:, scale_cols] = scaler.fit_transform(x_train[scale_cols])
    x_val.loc[:, scale_cols] = scaler.transform(x_val[scale_cols])
    x_test.loc[:, scale_cols] = scaler.transform(x_test[scale_cols])

    return {
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "scaler": scaler,
        "feature_cols": features,
        "scale_cols": scale_cols,
    }


def class_weight_ratio(y: pd.Series) -> float:
    n_pos = max(int(y.sum()), 1)
    n_neg = len(y) - n_pos
    return n_neg / n_pos


def save_processed(bundle: dict, meta: dict, out_dir: Path | None = None) -> Path:
    out = out_dir or DATA_PROCESSED
    out.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    for key in ("X_train", "X_val", "X_test"):
        bundle[key].to_csv(out / f"{key}.csv", index=False)
    for key in ("y_train", "y_val", "y_test"):
        bundle[key].to_csv(out / f"{key}.csv", index=False, header=True)

    joblib.dump(bundle["scaler"], ARTIFACTS / "scaler.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def run() -> dict:
    df = load_raw()
    summary = summarize(df)
    bundle = build_splits(df)
    ratio = class_weight_ratio(bundle["y_train"])
    meta = {
        **summary,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "scale_pos_weight": ratio,
        "n_train": int(len(bundle["y_train"])),
        "n_val": int(len(bundle["y_val"])),
        "n_test": int(len(bundle["y_test"])),
        "feature_cols": bundle["feature_cols"],
        "scale_cols": bundle["scale_cols"],
    }
    save_processed(bundle, meta)
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    run()
