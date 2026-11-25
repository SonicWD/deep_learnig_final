# -*- coding: utf-8 -*-
"""Front end intuitivo: detección de fraude (explicaciones + CSV de ejemplo)."""

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
EJEMPLOS_DIR = Path(__file__).resolve().parent / "ejemplos_csv"

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


def interpretar_una(prob: float) -> dict:
    pct = 100 * prob
    if prob >= 0.8:
        return {
            "decision": "FRAUDE — alta sospecha",
            "accion": "Bloquear o revisar YA con un analista.",
            "color": "error",
            "texto": (
                f"El modelo cree que hay un {pct:.1f} % de chance de que sea fraude. "
                "Eso está muy por encima del umbral del 50 %. En un banco real, "
                "esta operación no debería aprobarse a ciegas."
            ),
        }
    if prob >= THRESHOLD:
        return {
            "decision": "FRAUDE — sospecha",
            "accion": "Revisar antes de aprobar.",
            "color": "warning",
            "texto": (
                f"El modelo dio {pct:.1f} % (umbral = 50 %). Cruzó la línea de "
                "decisión: la etiqueta es fraude, pero no es el caso más extremo. "
                "Conviene validar monto, hora y comportamiento del cliente."
            ),
        }
    if prob >= 0.25:
        return {
            "decision": "LEGÍTIMA — con cautela",
            "accion": "Aprobar con seguimiento si el contexto es raro.",
            "color": "info",
            "texto": (
                f"El modelo dio {pct:.1f} %, por debajo del 50 %, así que la "
                "etiqueta es legítima. Aun así no es un score cercanísimo a 0: "
                "si el monto es atípico, una revisión humana ayuda."
            ),
        }
    return {
        "decision": "LEGÍTIMA",
        "accion": "Aprobar con normalidad.",
        "color": "success",
        "texto": (
            f"El modelo solo asignó {pct:.1f} % de probabilidad de fraude. "
            "Se parece al comportamiento habitual de transacciones normales."
        ),
    }


def mostrar_resultado_claro(prob: float, amount: float | None = None):
    info = interpretar_una(prob)
    st.markdown("### Resultado en palabras (no solo el número)")
    box = {
        "error": st.error,
        "warning": st.warning,
        "info": st.info,
        "success": st.success,
    }[info["color"]]
    box(f"**Decisión:** {info['decision']}\n\n**Qué hacer:** {info['accion']}")
    st.write(info["texto"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Probabilidad de fraude", f"{prob:.1%}")
    c2.metric("Umbral de decisión", "50 %")
    c3.metric(
        "Etiqueta final",
        "Fraude" if prob >= THRESHOLD else "Legítima",
    )
    if amount is not None:
        st.caption(f"Monto analizado (Amount): {amount:,.2f}")

    st.markdown("#### Cómo leer el porcentaje")
    st.markdown(
        f"""
| Rango del score | Significado práctico |
|-----------------|----------------------|
| 0 % – 25 % | Casi seguro legítima |
| 25 % – 49 % | Legítima, pero no “perfecta” |
| **50 % – 79 %** | **Se etiqueta fraude** (sospecha) |
| 80 % – 100 % | Fraude con alta prioridad |

Tu score fue **{prob:.1%}** → cae en la fila correspondiente de la tabla.
"""
    )
    st.progress(min(max(float(prob), 0.0), 1.0), text=f"Score {prob:.1%} (0 % = limpia · 100 % = fraude)")


def listar_ejemplos() -> list[Path]:
    if not EJEMPLOS_DIR.exists():
        return []
    return sorted(EJEMPLOS_DIR.glob("*.csv"))


# ---------------------------------------------------------------------------
# Cabecera: objetivo + entrada
# ---------------------------------------------------------------------------
st.title("💳 Detector de fraude en tarjetas de crédito")

st.markdown(
    """
## ¿Cuál es el objetivo?
Ayudar a un analista (o al banco) a responder:

> **“¿Esta transacción parece fraude o parece normal?”**

El sistema **no reemplaza** al humano: da una **probabilidad** y una **recomendación**
para decidir más rápido y con menos errores.
"""
)

cobj1, cobj2 = st.columns(2)
with cobj1:
    st.markdown(
        """
### ¿Qué entra al modelo? (la entrada)
Aceptamos un CSV (o una demo con monto) con:

- **`Amount`**: cuánto dinero movió la operación  
- **`V1` … `V28`**: variables del banco ya anonimizadas (PCA).  
  No son “nombre” ni “tarjeta”: son números que resumen el patrón de la compra.

**Por qué aceptamos esta entrada:** es el formato del dataset público ULB/Kaggle
con el que se entrenó el modelo. Así cualquiera puede probar con ejemplos reales
sin datos personales sensibles.
"""
    )
with cobj2:
    st.markdown(
        """
### ¿Qué sale? (la salida)
No es solo un número suelto. Verás:

1. **Probabilidad de fraude** (0 % a 100 %)  
2. **Decisión** Legítima / Fraude (umbral 50 %)  
3. **Qué hacer** (aprobar, revisar o bloquear)  
4. **Texto explicando** por qué se tomó esa decisión  

**Importante:** el fraude es raro (~0,17 % de los casos). Por eso no usamos
“accuracy” sola: miramos si atrapamos fraudes (Recall) sin alarmar de más (Precision).
"""
    )

with st.expander("Descarga CSV de ejemplo para probar la app", expanded=True):
    st.markdown(
        """
Elige un archivo, descárgalo y luego súbelo en la sección de abajo:

| Archivo | Qué contiene | Para qué sirve |
|---------|--------------|----------------|
| `ejemplo_una_fila.csv` | 1 transacción normal | Prueba rapidísima |
| `ejemplo_legitimas.csv` | Varias normales | Ver scores bajos |
| `ejemplo_fraudes.csv` | Varias fraudulentas | Ver scores altos / alarmas |
| `ejemplo_mixto.csv` | Normales + fraudes | Caso realista de lote |
"""
    )
    ejemplos = listar_ejemplos()
    if not ejemplos:
        st.warning("No se encontraron CSV de ejemplo en `app/ejemplos_csv/`.")
    else:
        cols = st.columns(2)
        for i, path in enumerate(ejemplos):
            with cols[i % 2]:
                st.download_button(
                    label=f"⬇ Descargar {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/csv",
                    key=f"dl_{path.name}",
                    use_container_width=True,
                )

arts = load_artifacts()
if arts is None:
    st.error(
        "No están los modelos en `artifacts/`. "
        "En local: `python -m src.preprocess` y `python -m src.train`."
    )
    st.stop()

features = arts["features"]
scale_cols = arts["scale_cols"]
comparison = arts.get("comparison") or {}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Guía rápida")
    st.markdown(
        """
1. Descarga un CSV de ejemplo  
2. Elige modelo  
3. Sube el CSV  
4. Lee la **decisión en palabras**
"""
    )
    st.divider()
    st.subheader("Modelos")
    st.markdown(
        """
**XGBoost** — más equilibrado (menos falsas alarmas).  
**MLP** — red neuronal; atrapa más fraudes pero alarma más.
"""
    )
    if comparison:
        st.divider()
        st.markdown("**Resultados del entrenamiento (test)**")
        for key, label in (("xgboost", "XGBoost"), ("mlp", "MLP")):
            if key in comparison:
                m = comparison[key]
                st.caption(label)
                st.write(
                    f"Precision {m['precision']:.0%} · "
                    f"Recall {m['recall']:.0%} · "
                    f"F1 {m['f1']:.0%}"
                )
    st.divider()
    st.caption("UCundinamarca · Esp. IA · Deep Learning")

# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------
st.markdown("## Analizar transacciones")
col_a, col_b = st.columns(2)
with col_a:
    modelo = st.selectbox(
        "1) ¿Con qué modelo quieres analizar?",
        ["XGBoost", "MLP"],
        help="Si no sabes cuál, empieza con XGBoost.",
    )
with col_b:
    modo = st.radio(
        "2) ¿Cómo vas a ingresar los datos?",
        ["Subir CSV (recomendado)", "Probar solo un monto (demo UX)"],
    )

if modelo == "XGBoost" and "xgboost" in comparison:
    m = comparison["xgboost"]
    st.info(
        f"**XGBoost en este proyecto:** de cada 100 alarmas, ~{100*m['precision']:.0f} "
        f"eran fraude de verdad (Precision). Detectó ~{100*m['recall']:.0f} % de los "
        f"fraudes reales (Recall). F1 = {m['f1']:.0%}."
    )
elif modelo == "MLP" and "mlp" in comparison:
    m = comparison["mlp"]
    st.warning(
        f"**MLP en este proyecto:** detectó ~{100*m['recall']:.0f} % de los fraudes "
        f"(Recall alto), pero solo ~{100*m['precision']:.0f} % de sus alarmas eran "
        f"correctas (Precision baja → muchas falsas alarmas). F1 = {m['f1']:.0%}."
    )

st.divider()

# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
if modo.startswith("Subir CSV"):
    st.subheader("3) Sube tu CSV (o uno de los ejemplos)")
    st.markdown(
        f"Columnas obligatorias: `{features[0]}` … `{features[-2]}` y **`{features[-1]}`**."
    )
    file = st.file_uploader("Archivo CSV", type=["csv"])

    if file is not None:
        raw = pd.read_csv(file)
        missing = [c for c in features if c not in raw.columns]
        if missing:
            st.error(
                "Este archivo no sirve todavía: le faltan columnas que el modelo espera. "
                f"Ejemplos de faltantes: {', '.join(missing[:8])}"
            )
            st.stop()

        x = prepare(raw, arts["scaler"], features, scale_cols)
        probs = predict_proba(modelo, x, arts)
        if probs is None:
            st.error("Modelo no disponible.")
            st.stop()

        out = raw.copy()
        out["probabilidad_fraude"] = probs
        out["porcentaje"] = [f"{100*p:.1f} %" for p in probs]
        out["decision"] = [
            "Fraude" if p >= THRESHOLD else "Legítima" for p in probs
        ]
        out["que_hacer"] = [
            ("Bloquear/revisar YA" if p >= 0.8 else "Revisar antes de aprobar")
            if p >= THRESHOLD
            else ("Aprobar con cautela" if p >= 0.25 else "Aprobar")
            for p in probs
        ]

        n = len(out)
        n_fraud = int((probs >= THRESHOLD).sum())
        n_ok = n - n_fraud

        st.markdown("## Resumen del lote (en español)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Transacciones analizadas", n)
        m2.metric("Decisión: Fraude", n_fraud)
        m3.metric("Decisión: Legítima", n_ok)

        st.success(
            f"Con **{modelo}** y umbral **50 %**: "
            f"**{n_fraud}** se marcaron como fraude y **{n_ok}** como legítimas. "
            "Abajo cada fila trae el porcentaje, la decisión y qué hacer."
        )

        st.dataframe(
            out[
                ["Amount", "porcentaje", "decision", "que_hacer"]
                + [c for c in out.columns if c not in ("Amount", "porcentaje", "decision", "que_hacer", "probabilidad_fraude")]
            ].head(100),
            use_container_width=True,
        )

        st.download_button(
            "⬇ Descargar resultados interpretados (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            "resultados_interpretados.csv",
            "text/csv",
        )

        idx = int(probs.argmax())
        st.markdown("---")
        st.markdown("## Caso más sospechoso del archivo")
        mostrar_resultado_claro(float(probs[idx]), float(raw.iloc[idx]["Amount"]))

else:
    st.subheader("3) Demo con un solo monto")
    st.warning(
        "Esta demo **solo cambia Amount** y pone V1–V28 en cero. "
        "Sirve para ver la interfaz. Para resultados serios, usa un CSV de ejemplo."
    )
    amount = st.number_input("Monto (Amount)", min_value=0.0, value=100.0, step=10.0)
    if st.button("Calcular y explicar resultado", type="primary"):
        row = {c: 0.0 for c in features}
        row["Amount"] = amount
        x = prepare(pd.DataFrame([row]), arts["scaler"], features, scale_cols)
        probs = predict_proba(modelo, x, arts)
        if probs is None:
            st.error("Modelo no disponible.")
            st.stop()
        mostrar_resultado_claro(float(probs[0]), amount)

st.divider()
with st.expander("Glosario: cómo interpretar TODO"):
    st.markdown(
        """
| Lo que ves | Qué significa | Cómo actuar |
|------------|---------------|-------------|
| Probabilidad 12 % | Muy poca sospecha | Aprobar |
| Probabilidad 40 % | Aún bajo el umbral | Aprobar; revisar si el monto es raro |
| Probabilidad 55 % | Cruzó el 50 % | Etiqueta fraude → revisar |
| Probabilidad 90 % | Muy sospechosa | Bloquear / investigar |
| Precision alta | Pocas alarmas falsas | Confías más en cada alerta |
| Recall alto | Se escapan pocos fraudes | Bueno para no perder dinero |
| V1–V28 | Huella numérica de la compra | No necesitas “traducirlas” a texto |
| Amount | Dinero de la operación | Parte de la entrada del modelo |
"""
    )
