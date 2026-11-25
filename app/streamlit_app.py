# -*- coding: utf-8 -*-
"""Front end intuitivo: detección de fraude en tarjetas (MLP vs XGBoost)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS, DATA_PROCESSED, FEATURE_COLS
from src.models.mlp import FraudMLP

THRESHOLD = 0.5

st.set_page_config(
    page_title="Detector de fraude en tarjetas",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    scaler_path = ARTIFACTS / "scaler.joblib"
    xgb_path = ARTIFACTS / "xgboost.joblib"
    mlp_path = ARTIFACTS / "mlp.pt"
    meta_path = DATA_PROCESSED / "meta.json"
    cmp_path = ARTIFACTS / "comparison.json"
    if not scaler_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    comparison = json.loads(cmp_path.read_text(encoding="utf-8")) if cmp_path.exists() else {}
    features = meta.get("feature_cols", FEATURE_COLS)
    scale_cols = meta.get("scale_cols", ["Amount"])
    scaler = joblib.load(scaler_path)
    xgb = joblib.load(xgb_path) if xgb_path.exists() else None
    mlp = None
    if mlp_path.exists():
        mlp = FraudMLP(n_features=len(features))
        try:
            state = torch.load(mlp_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(mlp_path, map_location="cpu")
        mlp.load_state_dict(state)
        mlp.eval()
    return {
        "scaler": scaler,
        "xgb": xgb,
        "mlp": mlp,
        "features": features,
        "scale_cols": scale_cols,
        "meta": meta,
        "comparison": comparison,
    }


def prepare(df: pd.DataFrame, scaler, features, scale_cols) -> pd.DataFrame:
    frame = df[features].copy()
    frame.loc[:, scale_cols] = scaler.transform(frame[scale_cols])
    return frame


def predict_proba(modelo: str, x: pd.DataFrame, arts: dict):
    if modelo == "XGBoost" and arts["xgb"] is not None:
        return arts["xgb"].predict_proba(x)[:, 1]
    if arts["mlp"] is not None:
        with torch.no_grad():
            return torch.sigmoid(
                arts["mlp"](torch.tensor(x.values, dtype=torch.float32))
            ).numpy()
    return None


def interpretar(prob: float) -> tuple[str, str, str]:
    """Devuelve etiqueta, color y explicación en lenguaje claro."""
    pct = 100 * prob
    if prob >= 0.8:
        return (
            "FRAUDE (alta sospecha)",
            "🔴",
            f"El modelo asignó {pct:.1f} % de probabilidad de fraude. "
            "Conviene bloquear o revisar la transacción con prioridad.",
        )
    if prob >= THRESHOLD:
        return (
            "FRAUDE (sospecha)",
            "🟠",
            f"El modelo asignó {pct:.1f} % de probabilidad de fraude "
            f"(por encima del umbral {THRESHOLD:.0%}). Revisar antes de aprobar.",
        )
    if prob >= 0.25:
        return (
            "LEGÍTIMA (con cautela)",
            "🟡",
            f"El modelo asignó {pct:.1f} % de probabilidad de fraude "
            f"(por debajo del umbral {THRESHOLD:.0%}). Suele aprobarse, "
            "pero si el monto o el contexto son raros, conviene una revisión manual.",
        )
    return (
        "LEGÍTIMA",
        "🟢",
        f"El modelo asignó solo {pct:.1f} % de probabilidad de fraude. "
        "La transacción se ve como comportamiento normal.",
    )


def explicar_metricas(nombre: str, m: dict) -> str:
    return (
        f"**{nombre}** en el conjunto de prueba del proyecto:\n\n"
        f"- **Precision {m['precision']:.1%}**: de las que marcó como fraude, "
        f"¿cuántas sí lo eran?\n"
        f"- **Recall {m['recall']:.1%}**: de todos los fraudes reales, "
        f"¿cuántos detectó?\n"
        f"- **F1 {m['f1']:.1%}**: equilibrio entre Precision y Recall.\n"
        f"- **AUC-PR {m['auc_pr']:.1%}**: calidad en problema desbalanceado "
        f"(más útil que la exactitud global)."
    )


# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------
st.title("💳 Detector de fraude en tarjetas de crédito")
st.markdown(
    """
Esta aplicación ayuda a **decidir si una transacción parece fraudulenta o legítima**.

Fue entrenada con el dataset público **Credit Card Fraud** (ULB / Kaggle):
casi 285 mil transacciones, de las cuales solo ~0,17 % son fraude.
Por eso no basta mirar “accuracy”: aquí importan **Recall** (atrapar fraudes)
y **Precision** (no alarmar de más).
"""
)

with st.expander("¿Qué hace esta app? (léeme primero)", expanded=True):
    st.markdown(
        """
1. **Eliges un modelo** (XGBoost o MLP).
2. **Cargas datos**: un CSV de transacciones o un monto de prueba.
3. La app calcula un **score de fraude** entre 0 y 1:
   - cerca de **0** → parece legítima  
   - cerca de **1** → parece fraude  
4. Si el score es **≥ 0,5**, se etiqueta como fraude; si no, como legítima.
5. Te mostramos **qué significa el resultado** en lenguaje claro.

**Columnas del CSV:** `V1` … `V28` (variables anonimizadas) y `Amount` (monto).
Usa los archivos de ejemplo (`ejemplo_mixto.csv`, etc.) si no tienes datos propios.
"""
    )

arts = load_artifacts()
if arts is None:
    st.error(
        "No se encontraron los modelos entrenados en `artifacts/`. "
        "En local ejecuta `python -m src.preprocess` y `python -m src.train`."
    )
    st.stop()

features = arts["features"]
scale_cols = arts["scale_cols"]
comparison = arts.get("comparison") or {}

# ---------------------------------------------------------------------------
# Sidebar: contexto y modelos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Cómo usar la app")
    st.markdown(
        """
**Paso 1.** Elige el modelo abajo.  
**Paso 2.** Sube un CSV o prueba un monto.  
**Paso 3.** Lee el score y la explicación.
"""
    )
    st.divider()
    st.subheader("¿Qué modelo elijo?")
    st.markdown(
        """
**XGBoost (recomendado para operar)**  
Árboles con boosting. En nuestras pruebas equilibra mejor:
menos falsas alarmas y buen F1.

**MLP (red neuronal)**  
Deep Learning tabular. Detecta más fraudes (alto Recall),
pero genera más falsas alarmas si el umbral es 0,5.
"""
    )
    if comparison:
        st.divider()
        st.subheader("Desempeño en el proyecto")
        if "xgboost" in comparison:
            st.caption("XGBoost")
            st.write(
                f"Precision {comparison['xgboost']['precision']:.1%} · "
                f"Recall {comparison['xgboost']['recall']:.1%} · "
                f"F1 {comparison['xgboost']['f1']:.1%}"
            )
        if "mlp" in comparison:
            st.caption("MLP")
            st.write(
                f"Precision {comparison['mlp']['precision']:.1%} · "
                f"Recall {comparison['mlp']['recall']:.1%} · "
                f"F1 {comparison['mlp']['f1']:.1%}"
            )

    st.divider()
    st.caption("Universidad de Cundinamarca · Especialización en IA")
    st.caption("Proyecto final · Aplicaciones en Deep Learning")

# ---------------------------------------------------------------------------
# Controles principales
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)
with col_a:
    modelo = st.selectbox(
        "1) Modelo a usar",
        ["XGBoost", "MLP"],
        help="XGBoost suele ser más equilibrado; MLP atrapa más fraudes pero alarma más.",
    )
with col_b:
    modo = st.radio(
        "2) Cómo ingresar datos",
        ["Subir CSV de transacciones", "Probar un solo monto (demo)"],
        help="CSV = varias filas reales/ejemplo. Demo = solo Amount para ver la UX.",
    )

if modelo == "XGBoost" and "xgboost" in comparison:
    st.info(explicar_metricas("XGBoost", comparison["xgboost"]))
elif modelo == "MLP" and "mlp" in comparison:
    st.warning(explicar_metricas("MLP", comparison["mlp"]))

st.divider()

# ---------------------------------------------------------------------------
# Modo CSV
# ---------------------------------------------------------------------------
if modo.startswith("Subir CSV"):
    st.subheader("Cargar archivo CSV")
    st.markdown(
        f"""
Sube un archivo con estas columnas: **{", ".join(features[:3])} … {features[-1]}**.

Ejemplos listos: `ejemplo_una_fila.csv`, `ejemplo_legitimas.csv`,
`ejemplo_fraudes.csv`, `ejemplo_mixto.csv`.
"""
    )
    file = st.file_uploader("Selecciona el CSV", type=["csv"])

    if file is not None:
        raw = pd.read_csv(file)
        missing = [c for c in features if c not in raw.columns]
        if missing:
            st.error(
                "Ese CSV no tiene todas las columnas que el modelo necesita. "
                f"Faltan, por ejemplo: {', '.join(missing[:6])}"
                + ("…" if len(missing) > 6 else "")
            )
            st.stop()

        x = prepare(raw, arts["scaler"], features, scale_cols)
        probs = predict_proba(modelo, x, arts)
        if probs is None:
            st.error("El modelo elegido no está disponible.")
            st.stop()

        out = raw.copy()
        out["probabilidad_fraude"] = probs
        out["predicciones"] = [
            "Fraude" if p >= THRESHOLD else "Legítima" for p in probs
        ]
        out["nivel"] = [
            "Alta" if p >= 0.8 else ("Media" if p >= THRESHOLD else "Baja")
            for p in probs
        ]

        n = len(out)
        n_fraud = int((probs >= THRESHOLD).sum())
        n_ok = n - n_fraud

        st.success("Análisis listo. Así se leen los resultados:")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transacciones", n)
        m2.metric("Marcadas como fraude", n_fraud)
        m3.metric("Marcadas como legítimas", n_ok)
        m4.metric("Score promedio", f"{float(probs.mean()):.3f}")

        st.markdown(
            f"""
### ¿Qué significa esto?
- Se revisaron **{n}** transacciones con el modelo **{modelo}**.
- **{n_fraud}** quedaron por encima del umbral {THRESHOLD:.0%} → etiqueta **Fraude**.
- **{n_ok}** quedaron por debajo → etiqueta **Legítima**.
- `probabilidad_fraude` es el score continuo (0 a 1). Mientras más alto, más sospechosa.
- `nivel` resume la urgencia: **Alta** (≥80 %), **Media** (≥50 %), **Baja** (<50 %).
"""
        )

        st.dataframe(out.head(100), use_container_width=True)
        st.download_button(
            "Descargar resultados (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            "resultados_fraude.csv",
            "text/csv",
            help="Guarda el archivo con scores para revisarlo en Excel.",
        )

        # Destacar la más sospechosa
        idx = int(probs.argmax())
        etiqueta, icono, texto = interpretar(float(probs[idx]))
        st.markdown(f"### Transacción más sospechosa de este archivo {icono}")
        st.write(f"**Resultado:** {etiqueta}")
        st.write(texto)
        st.write(f"Monto (Amount): **{float(raw.iloc[idx]['Amount']):.2f}**")

# ---------------------------------------------------------------------------
# Modo demo Amount
# ---------------------------------------------------------------------------
else:
    st.subheader("Probar un solo monto (demo de UX)")
    st.markdown(
        """
Esta demo **no usa una transacción real completa**: solo cambias el **Amount**
y el resto de variables (`V1`–`V28`) se ponen en 0 para mostrar cómo responde la interfaz.

Para un análisis serio, usa **Subir CSV** con los archivos de ejemplo.
"""
    )
    amount = st.number_input(
        "Monto de la transacción (Amount)",
        min_value=0.0,
        value=100.0,
        step=10.0,
        help="Ejemplos: 10 (compra chica), 100, 500, 2000 (monto alto).",
    )
    if st.button("Calcular probabilidad de fraude", type="primary"):
        row = {c: 0.0 for c in features}
        row["Amount"] = amount
        x = prepare(pd.DataFrame([row]), arts["scaler"], features, scale_cols)
        probs = predict_proba(modelo, x, arts)
        if probs is None:
            st.error("El modelo elegido no está disponible.")
            st.stop()
        prob = float(probs[0])
        etiqueta, icono, texto = interpretar(prob)

        st.markdown(f"## Resultado {icono}")
        c1, c2 = st.columns(2)
        c1.metric("Probabilidad de fraude", f"{prob:.1%}")
        c2.metric("Decisión (umbral 50 %)", etiqueta.split("(")[0].strip())
        st.write(texto)
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(
            "Barra = score 0–100 %. La línea mental de decisión está en 50 %."
        )

st.divider()
with st.expander("Glosario rápido"):
    st.markdown(
        """
| Término | Significado simple |
|---------|-------------------|
| **Fraude** | Transacción que el modelo cree ilegítima |
| **Legítima** | Transacción que el modelo cree normal |
| **Score / probabilidad** | Número entre 0 y 1; más alto = más sospechosa |
| **Umbral 0,5** | Si score ≥ 0,5 → se etiqueta fraude |
| **Precision** | De las alarmas, cuántas eran correctas |
| **Recall** | De los fraudes reales, cuántos se atraparon |
| **V1–V28** | Variables anonimizadas del banco (PCA) |
| **Amount** | Monto de la compra/transacción |
"""
    )
