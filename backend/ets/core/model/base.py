# -*- coding: utf-8 -*-
"""
Base model protocol defining the contract for all ML model adapters in ETS.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class BaseModel(Protocol):
    """
    Protocol for models integrated into the ETS execution pipelines.
    Ensures interface consistency across diverse frameworks (LightGBM, XGBoost, PyTorch).
    """

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame | None = None, y_val: pd.Series | None = None) -> Any:
        """Fit the model to the training data."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous values or class probabilities."""
        ...

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        ...

    def save(self, path: str) -> None:
        """Serialize and save the model to disk."""
        ...

    def load(self, path: str) -> None:
        """Deserialize and load the model from disk."""
        ...
