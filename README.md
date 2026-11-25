# deep_learnig_final

Detección de fraude en tarjetas de crédito (Kaggle / ULB) — **MLP tabular (PyTorch)** vs **XGBoost**.

Proyecto final · Aplicaciones en Deep Learning · Universidad de Cundinamarca  
Estudiante: Wilson Alfonso Díaz Capador · Correo: walfonsodiaz@ucundinamarca.edu.co

| Recurso | Link |
|---------|------|
| Repositorio | https://github.com/SonicWD/deep_learnig_final |
| App desplegada | https://deeplearnigfinal-hyoq28uvnazx5hybeusfsw.streamlit.app/ |

## Objetivo

Clasificar transacciones como **legítimas (0)** o **fraudulentas (1)**, comparar una red neuronal densa con Gradient Boosting e interpretar Precision, Recall, F1, AUC-ROC y AUC-PR en un problema con desbalance extremo (~0,17 % fraude). Idea alineada con la guía del gestor (riesgo/fraude · redes tabulares vs boosting · Kaggle Credit Card Fraud).

## Demo rápida (sin instalar)

1. Abre la app: https://deeplearnigfinal-hyoq28uvnazx5hybeusfsw.streamlit.app/
2. Elige **MLP** o **XGBoost**.
3. Usa **Ejemplo de una fila** e ingresa un `Amount`, o sube un CSV con columnas `V1`–`V28` y `Amount`.
4. Revisa `fraud_score` y `pred_fraude` (umbral 0,5).

## Estructura del repositorio

```
deep_learnig_final/
├── README.md
├── requirements.txt
├── .gitignore
├── app/streamlit_app.py          # front end
├── artifacts/                    # scaler + pesos para inferencia en la nube
├── data/raw/                     # creditcard.csv (no versionado)
├── data/processed/meta.json
├── src/
│   ├── config.py
│   ├── data_load.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── models/
│       ├── mlp.py
│       └── xgboost_model.py
├── notebooks/01_eda.ipynb
├── tests/test_preprocess.py
└── docs/
    ├── pipeline.md
    └── resultados_iniciales.md
```

## Datos

1. Descargar [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB).
2. Colocar `creditcard.csv` en `data/raw/`.

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

Alternativa académica: OpenML dataset 1597 (puede no incluir `Time`; el código lo resuelve).

## Instalación local

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

# Interfaz local
streamlit run app/streamlit_app.py

# Pruebas unitarias
python -m pytest tests/test_preprocess.py -q
```

## Resultados (test, umbral 0,5, semilla 42)

| Modelo | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|--------|-----------|--------|-----|---------|--------|
| MLP | 0.052 | 0.878 | 0.098 | 0.971 | 0.765 |
| XGBoost | 0.803 | 0.824 | 0.813 | 0.961 | 0.836 |

**Interpretación:** XGBoost equilibra mejor F1 y AUC-PR (menos falsas alarmas). El MLP prioriza Recall (atrapa más fraudes) a costa de Precision; el rediseño propuesto calibra umbral, TabNet o ensemble.

## Fases del proyecto (S11–S16)

1. **S11** Definición del problema + papers + diseño  
2. **S12** Pipeline, datos, preprocesamiento, repo  
3. **S13** Implementación, pruebas, métricas, rediseño  
4. **S14** % avance, objetivos, retos  
5. **S15** Front end / UX  
6. **S16** Documentación final + nube + socialización  

## MLOps

Weights & Biases opcional: `WANDB_ENABLED = True` en `src/config.py` tras `wandb.login()`.

## Licencia de datos

El dataset ULB/Kaggle se usa solo con fines académicos; no se redistribuye el CSV completo en este repositorio.
