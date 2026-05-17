# -*- coding: utf-8 -*-
"""
Base LightGBM model for pre-training.

This is trained on the full historical training window and serves
as the foundation for subsequent transfer/fine-tuning.
"""
from __future__ import annotations
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import CFG, MODEL_DIR

logger = logging.getLogger(__name__)


class BaseLGBModel:
    """
    Pre-trained LightGBM binary classifier.

    Parameters
    ----------
    params : dict, optional
        LightGBM parameters. Defaults to config.
    num_boost_round : int, optional
        Max boosting rounds.
    early_stopping : int, optional
        Early stopping patience.
    """

    def __init__(
        self,
        params: Optional[Dict] = None,
        num_boost_round: Optional[int] = None,
        early_stopping: Optional[int] = None,
    ):
        self.params = params or dict(CFG.model.base_params)
        self.num_boost_round = num_boost_round or CFG.model.base_num_boost_round
        self.early_stopping = early_stopping or CFG.model.base_early_stopping
        self.model: Optional[lgb.Booster] = None
        self.feature_names: list = []
        self.evals_result: dict = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[pd.Series] = None,
        sample_weight: Optional[np.ndarray] = None,
        categorical_feature: str = "auto",
    ) -> "BaseLGBModel":
        """
        Train the base model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels (0/1).
        X_valid, y_valid : optional
            Validation set for early stopping.
        sample_weight : np.ndarray, optional
            Sample weights.
        categorical_feature : str
            Categorical feature handling.

        Returns
        -------
        self
        """
        self.feature_names = list(X_train.columns)

        dtrain = lgb.Dataset(
            X_train.values, label=y_train.values,
            weight=sample_weight, feature_name=self.feature_names,
            free_raw_data=False,
        )

        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_valid is not None and y_valid is not None:
            dvalid = lgb.Dataset(
                X_valid.values, label=y_valid.values,
                feature_name=self.feature_names, free_raw_data=False,
            )
            valid_sets.append(dvalid)
            valid_names.append("valid")

        self.evals_result = {}

        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(self.evals_result),
        ]
        if X_valid is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping))

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        logger.info(
            "Base model trained: %d trees, best_iteration=%d",
            self.model.num_trees(),
            self.model.best_iteration if self.model.best_iteration > 0
            else self.model.num_trees(),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for class 1 (drop)."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet!")
        return self.model.predict(X.values, num_iteration=self.model.best_iteration)

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels."""
        proba = self.predict(X)
        return (proba >= threshold).astype(int)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Get feature importance as named Series."""
        if self.model is None:
            raise RuntimeError("Model not fitted!")
        imp = self.model.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model to disk."""
        path = path or MODEL_DIR / "base_lgb_model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseLGBModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Model loaded from %s", path)
        return obj

    def get_booster(self) -> lgb.Booster:
        """Return the underlying LightGBM Booster for transfer learning."""
        if self.model is None:
            raise RuntimeError("Model not fitted!")
        return self.model
