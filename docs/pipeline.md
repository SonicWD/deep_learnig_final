# Pipeline de detección de fraude

1. **Recolección:** `creditcard.csv` (ULB/Kaggle) en `data/raw/`.
2. **Carga / EDA:** `src/data_load.py` — filas, tasa de fraude, nulos.
3. **Preprocess:** `src/preprocess.py` — StandardScaler en Time/Amount, split 70/15/15 estratificado, semilla 42.
4. **Modelos:** MLP (`src/models/mlp.py`) y XGBoost (`src/models/xgboost_model.py`).
5. **Entrenamiento:** `python -m src.train` — class weights / scale_pos_weight.
6. **Evaluación:** Precision, Recall, F1, AUC-ROC, AUC-PR en `artifacts/`.
7. **MLOps:** W&B opcional (`WANDB_ENABLED` en `src/config.py`).
