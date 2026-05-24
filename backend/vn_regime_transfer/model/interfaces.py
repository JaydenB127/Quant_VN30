# -*- coding: utf-8 -*-
"""
Protocol definitions for models in the ETS platform.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class BaseModel(Protocol):
    """
    Protocol defining standard ML model methods.
    Ensures that all model components implement identical interfaces.
    """

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
    ) -> BaseModel:
        """Fit the model to the training data with optional early stopping on validation data."""
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities or continuous targets."""
        ...

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels (0 or 1)."""
        ...
