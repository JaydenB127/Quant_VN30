# -*- coding: utf-8 -*-
"""
Ensemble: weighted combination of base + adapted model predictions.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import CFG
from .base_lgb import BaseLGBModel
from .transfer_lgb import TransferLGBModel

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Weighted ensemble of base and transfer-learned models.

    score = w_base * base_pred + w_adapted * adapted_pred

    Weights can be static or dynamically adjusted by regime confidence.
    """

    def __init__(
        self,
        base_model: BaseLGBModel,
        transfer_model: Optional[TransferLGBModel] = None,
        w_base: Optional[float] = None,
        w_adapted: Optional[float] = None,
    ):
        self.base_model = base_model
        self.transfer_model = transfer_model
        self.w_base = w_base or CFG.model.ensemble_w_base
        self.w_adapted = w_adapted or CFG.model.ensemble_w_adapted

    def predict(
        self,
        X: pd.DataFrame,
        regime_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Ensemble prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        regime_confidence : float, optional
            HMM posterior probability [0, 1]. If provided, dynamically
            adjusts weights: higher confidence → more weight on transfer model.

        Returns
        -------
        np.ndarray
            Predicted probabilities.
        """
        base_pred = self.base_model.predict(X)

        if self.transfer_model is None or self.transfer_model.model is None:
            return base_pred

        adapted_pred = self.transfer_model.predict(X)

        # Dynamic weighting based on regime confidence
        if regime_confidence is not None:
            # Scale adapted weight by confidence
            w_a = self.w_adapted * regime_confidence
            w_b = 1.0 - w_a
        else:
            w_b = self.w_base
            w_a = self.w_adapted

        ensemble = w_b * base_pred + w_a * adapted_pred

        return ensemble

    def predict_binary(
        self, X: pd.DataFrame, threshold: float = 0.5,
        regime_confidence: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary labels from ensemble."""
        proba = self.predict(X, regime_confidence)
        return (proba >= threshold).astype(int)
