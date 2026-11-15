# deep_learnig_final

Detección de fraude en tarjetas de crédito (Kaggle / ULB) — **MLP tabular (PyTorch)** vs **XGBoost**.

Proyecto final · Aplicaciones en Deep Learning · Universidad de Cundinamarca · Wilson Alfonso Díaz Capador.

Repositorio: https://github.com/SonicWD/deep_learnig_final

## Objetivo

Clasificar transacciones como legítimas o fraudulentas, comparar una red neuronal densa con Gradient Boosting e interpretar Precision, Recall, F1, AUC-ROC y AUC-PR en un problema con desbalance extremo (~0,17 % fraude).

## Estructura

```
data/raw/          # creditcard.csv (no versionado)
data/processed/    # artefactos locales
src/
  config.py
  data_load.py
  preprocess.py
  models/mlp.py
  models/xgboost_model.py
  train.py
  evaluate.py
notebooks/01_eda.ipynb
docs/pipeline.md
app/streamlit_app.py   # front (S15)
```

## Datos

1. Descargar [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Colocar `creditcard.csv` en `data/raw/`.

Con Kaggle CLI:

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

## Instalación

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
# Preprocesar (requiere data/raw/creditcard.csv)
python -m src.preprocess

# Entrenar MLP y XGBoost
python -m src.train

# Evaluar
python -m src.evaluate

# Interfaz local
streamlit run app/streamlit_app.py
```

## Resultados iniciales (test, umbral 0,5, semilla 42)

| Modelo | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|--------|-----------|--------|-----|---------|--------|
| MLP | 0.052 | 0.878 | 0.098 | 0.971 | 0.765 |
| XGBoost | 0.803 | 0.824 | 0.813 | 0.961 | 0.836 |

XGBoost ofrece mejor equilibrio F1/AUC-PR. El MLP prioriza Recall y motiva rediseño de umbral / TabNet / ensemble.

## Pruebas

```bash
python -m pytest tests/test_preprocess.py -q
```

## MLOps

Weights & Biases opcional: activar `WANDB_ENABLED = True` en `src/config.py` tras `wandb.login()`.

## Licencia de datos

El dataset ULB/Kaggle (o espejo OpenML 1597) se usa solo con fines académicos; no se redistribuye en este repositorio.
