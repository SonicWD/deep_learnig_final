# -*- coding: utf-8 -*-
"""MLP tabular para detección de fraude."""

from __future__ import annotations

import torch
from torch import nn

from src.config import MLP_DROPOUT, MLP_HIDDEN


class FraudMLP(nn.Module):
    def __init__(self, n_features: int, hidden=MLP_HIDDEN, dropout: float = MLP_DROPOUT):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden:
            layers.extend([
                nn.Linear(prev, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
