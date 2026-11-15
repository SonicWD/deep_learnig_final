# -*- coding: utf-8 -*-
"""Front end: puntuación de fraude sobre CSV o fila manual."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS, FEATURE_COLS, SCALE_COLS
from src.models.mlp import FraudMLP

st.set_page_config(page_title="Detección de fraude", layout="centered")
st.title("Detección de fraude en tarjetas")
st.caption("Proyecto final · MLP tabular vs XGBoost · UCundinamarca")


@st.cache_resource
def load_artifacts():
    scaler_path = ARTIFACTS / "scaler.joblib"
    xgb_path = ARTIFACTS / "xgboost.joblib"
    mlp_path = ARTIFACTS / "mlp.pt"
    if not scaler_path.exists():
        return None
    scaler = joblib.load(scaler_path)
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None
    mlp = None
    if mlp_path.exists():
        mlp = FraudMLP(n_features=len(FEATURE_COLS))
        mlp.load_state_dict(torch.load(mlp_path, map_location="cpu"))
        mlp.eval()
    return {"scaler": scaler, "xgb": xgb, "mlp": mlp}


def prepare(df: pd.DataFrame, scaler) -> pd.DataFrame:
    frame = df[FEATURE_COLS].copy()
    frame.loc[:, SCALE_COLS] = scaler.transform(frame[SCALE_COLS])
    return frame


arts = load_artifacts()
if arts is None:
    st.warning(
        "Aún no hay artefactos en `artifacts/`. "
        "Ejecute `python -m src.preprocess` y `python -m src.train` con el CSV en data/raw/."
    )
    st.stop()

modelo = st.selectbox("Modelo", ["XGBoost", "MLP"])
modo = st.radio("Entrada", ["Subir CSV", "Ejemplo de una fila"])

if modo == "Subir CSV":
    file = st.file_uploader("CSV con columnas Time, V1–V28, Amount", type=["csv"])
    if file is not None:
        raw = pd.read_csv(file)
        missing = [c for c in FEATURE_COLS if c not in raw.columns]
        if missing:
            st.error(f"Faltan columnas: {missing[:5]}...")
        else:
            x = prepare(raw, arts["scaler"])
            if modelo == "XGBoost" and arts["xgb"] is not None:
                probs = arts["xgb"].predict_proba(x)[:, 1]
            elif arts["mlp"] is not None:
                with torch.no_grad():
                    probs = torch.sigmoid(
                        arts["mlp"](torch.tensor(x.values, dtype=torch.float32))
                    ).numpy()
            else:
                st.error("Modelo no disponible.")
                st.stop()
            out = raw.copy()
            out["fraud_score"] = probs
            out["pred_fraude"] = (probs >= 0.5).astype(int)
            st.dataframe(out.head(50))
            st.download_button(
                "Descargar scores",
                out.to_csv(index=False).encode("utf-8"),
                "scores_fraude.csv",
                "text/csv",
            )
else:
    st.write("Ingrese Amount y Time; el resto se rellena en 0 solo para demo de UX.")
    amount = st.number_input("Amount", min_value=0.0, value=100.0)
    time_v = st.number_input("Time", min_value=0.0, value=0.0)
    if st.button("Calcular score"):
        row = {c: 0.0 for c in FEATURE_COLS}
        row["Amount"] = amount
        row["Time"] = time_v
        x = prepare(pd.DataFrame([row]), arts["scaler"])
        if modelo == "XGBoost" and arts["xgb"] is not None:
            prob = float(arts["xgb"].predict_proba(x)[:, 1][0])
        else:
            with torch.no_grad():
                prob = float(torch.sigmoid(
                    arts["mlp"](torch.tensor(x.values, dtype=torch.float32))
                ).numpy()[0])
        st.metric("Probabilidad de fraude", f"{prob:.4f}")
        st.write("Predicción:", "Fraude" if prob >= 0.5 else "Legítima")
