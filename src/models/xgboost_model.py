# -*- coding: utf-8 -*-
"""Baseline XGBoost para fraude tabular."""

from __future__ import annotations

from xgboost import XGBClassifier

from src.config import XGB_PARAMS


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)
