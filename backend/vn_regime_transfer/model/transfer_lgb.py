# -*- coding: utf-8 -*-
"""
Transfer Learning LightGBM: fine-tune from pre-trained base model.

Key features:
  - init_model parameter for warm-start
  - Time-decay sample weighting
  - Regime-conditioned fine-tuning
"""
from __future__ import annotations
import logging
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import CFG
from .base_lgb import BaseLGBModel

logger = logging.getLogger(__name__)


def compute_time_decay_weights(
    dates: pd.Index,
    alpha: float = 0.997,
) -> np.ndarray:
    """
    Exponential time-decay sample weights: w_t = alpha^(T - t).

    Recent samples get higher weight.

    Parameters
    ----------
    dates : pd.Index
        Date index of training samples.
    alpha : float
        Decay factor. Closer to 1.0 = slower decay.

    Returns
    -------
    np.ndarray
        Weights array, same length as dates.
    """
    if isinstance(dates, pd.MultiIndex):
        dates = dates.get_level_values("date")

    unique_dates = dates.unique().sort_values()
    date_to_rank = {d: i for i, d in enumerate(unique_dates)}
    T = len(unique_dates)

    ranks = dates.map(date_to_rank).values
    weights = alpha ** (T - 1 - ranks)

    # Normalize to mean = 1
    weights = weights / weights.mean()
    return weights.astype(np.float64)


class TransferLGBModel:
    """
    Fine-tuned LightGBM via transfer learning.

    Uses ``init_model`` to continue training from a pre-trained base model,
    with time-decay weighting and regime-conditioned data selection.
    """

    def __init__(
        self,
        base_model: BaseLGBModel,
        ft_num_boost_round: Optional[int] = None,
        ft_learning_rate: Optional[float] = None,
        time_decay_alpha: Optional[float] = None,
    ):
        self.base_model = base_model
        self.ft_num_boost_round = ft_num_boost_round or CFG.model.ft_num_boost_round
        self.ft_learning_rate = ft_learning_rate or CFG.model.ft_learning_rate
        self.time_decay_alpha = time_decay_alpha or CFG.model.time_decay_alpha
        self.model: Optional[lgb.Booster] = None

    def finetune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        regime_mask: Optional[pd.Series] = None,
        current_regime: Optional[int] = None,
    ) -> "TransferLGBModel":
        """
        Fine-tune the base model on new/recent data.

        Parameters
        ----------
        X_train, y_train : pd.DataFrame, pd.Series
            Training data.
        X_valid, y_valid : optional
            Validation data.
        regime_mask : pd.Series, optional
            Regime labels aligned to X_train index.
        current_regime : int, optional
            If provided with regime_mask, only fine-tune on samples
            from this regime (regime-conditioned transfer).

        Returns
        -------
        self
        """
        # Optionally filter by regime
        if regime_mask is not None and current_regime is not None:
            mask = regime_mask == current_regime
            if mask.sum() < 50:
                logger.warning(
                    "Only %d samples in regime %d, using all data for fine-tuning",
                    mask.sum(), current_regime,
                )
            else:
                # Use mask.values to avoid IndexingError when indices don't exactly match (e.g. MultiIndex vs DatetimeIndex)
                mask_vals = mask.values if hasattr(mask, "values") else mask
                X_train = X_train.loc[mask_vals]
                y_train = y_train.loc[mask_vals]
                logger.info(
                    "Regime-conditioned fine-tuning: %d samples from regime %d",
                    len(X_train), current_regime,
                )

        # Compute time-decay weights
        weights = compute_time_decay_weights(X_train.index, alpha=self.time_decay_alpha)

        # Update params with lower learning rate for fine-tuning
        ft_params = dict(self.base_model.params)
        ft_params["learning_rate"] = self.ft_learning_rate

        dtrain = lgb.Dataset(
            X_train.values, label=y_train.values,
            weight=weights,
            feature_name=self.base_model.feature_names,
            free_raw_data=False,
        )

        valid_sets = [dtrain]
        valid_names = ["train"]
        if X_valid is not None and y_valid is not None:
            dvalid = lgb.Dataset(
                X_valid.values, label=y_valid.values,
                feature_name=self.base_model.feature_names,
                free_raw_data=False,
            )
            valid_sets.append(dvalid)
            valid_names.append("valid")

        callbacks = [lgb.log_evaluation(period=20)]

        # Fine-tune using init_model (transfer learning)
        self.model = lgb.train(
            ft_params,
            dtrain,
            num_boost_round=self.ft_num_boost_round,
            init_model=self.base_model.get_booster(),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(
            "Fine-tuned model: +%d trees (total %d), lr=%.4f, decay_alpha=%.4f",
            self.ft_num_boost_round,
            self.model.num_trees(),
            self.ft_learning_rate,
            self.time_decay_alpha,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of drop (class 1)."""
        if self.model is None:
            raise RuntimeError("Model not fine-tuned yet!")
        return self.model.predict(X.values)

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        return (self.predict(X) >= threshold).astype(int)
