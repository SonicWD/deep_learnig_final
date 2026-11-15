# -*- coding: utf-8 -*-
"""Configuración central del proyecto de fraude."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

CSV_NAME = "creditcard.csv"
CSV_PATH = DATA_RAW / CSV_NAME

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # respecto del total; train = 1 - test - val

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLS = ["Time", "Amount"]
TARGET_COL = "Class"

# MLP
MLP_HIDDEN = (128, 64)
MLP_DROPOUT = 0.3
MLP_LR = 1e-3
MLP_EPOCHS = 20
MLP_BATCH = 2048
MLP_PATIENCE = 5

# XGBoost
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": SEED,
    "n_jobs": -1,
}

WANDB_PROJECT = "fraude-dl"
WANDB_ENABLED = False  # activar en S13 tras wandb.login()
