# -*- coding: utf-8 -*-
"""Entrenamiento MLP + XGBoost y guardado de artefactos."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    ARTIFACTS,
    DATA_PROCESSED,
    MLP_BATCH,
    MLP_EPOCHS,
    MLP_LR,
    MLP_PATIENCE,
    SEED,
    WANDB_ENABLED,
    WANDB_PROJECT,
)
from src.evaluate import binary_metrics, save_metrics
from src.models.mlp import FraudMLP
from src.models.xgboost_model import build_xgboost
from src.preprocess import class_weight_ratio, run as preprocess_run


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_processed():
    meta_path = DATA_PROCESSED / "meta.json"
    if not meta_path.exists():
        preprocess_run()
    x_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    x_val = pd.read_csv(DATA_PROCESSED / "X_val.csv")
    x_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0]
    y_val = pd.read_csv(DATA_PROCESSED / "y_val.csv").iloc[:, 0]
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").iloc[:, 0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return x_train, x_val, x_test, y_train, y_val, y_test, meta


def train_mlp(x_train, y_train, x_val, y_val, pos_weight: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FraudMLP(n_features=x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    train_ds = TensorDataset(
        torch.tensor(x_train.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_x = torch.tensor(x_val.values, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val.values, dtype=torch.float32).to(device)
    loader = DataLoader(train_ds, batch_size=MLP_BATCH, shuffle=True)

    best_state = None
    best_val = float("inf")
    wait = 0
    history = []

    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = criterion(val_logits, val_y).item()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= MLP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict_proba(frame: pd.DataFrame) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(frame.values, dtype=torch.float32).to(device)
            logits = model(xt)
            return torch.sigmoid(logits).detach().cpu().numpy()

    return model, predict_proba, history


def train_xgb(x_train, y_train, x_val, y_val, scale_pos_weight: float):
    model = build_xgboost(scale_pos_weight)
    model.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    return model


def maybe_wandb(metrics: dict) -> None:
    if not WANDB_ENABLED:
        return
    import wandb
    wandb.init(project=WANDB_PROJECT, config={"seed": SEED})
    wandb.log(metrics)
    wandb.finish()


def run():
    _set_seed()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    x_train, x_val, x_test, y_train, y_val, y_test, meta = _load_processed()
    pos_weight = float(meta.get("scale_pos_weight") or class_weight_ratio(y_train))

    mlp, mlp_predict, history = train_mlp(x_train, y_train, x_val, y_val, pos_weight)
    torch.save(mlp.state_dict(), ARTIFACTS / "mlp.pt")
    (ARTIFACTS / "mlp_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    xgb = train_xgb(x_train, y_train, x_val, y_val, pos_weight)
    joblib.dump(xgb, ARTIFACTS / "xgboost.joblib")

    results = {}
    for name, proba in (
        ("mlp", mlp_predict(x_test)),
        ("xgboost", xgb.predict_proba(x_test)[:, 1]),
    ):
        metrics = binary_metrics(y_test, proba)
        save_metrics(name, metrics)
        results[name] = {k: metrics[k] for k in ("precision", "recall", "f1", "roc_auc", "auc_pr")}
        print(name, results[name])
        print(metrics["report"])

    (ARTIFACTS / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    maybe_wandb({"comparison": results})
    return results


if __name__ == "__main__":
    run()
