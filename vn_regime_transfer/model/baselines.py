# -*- coding: utf-8 -*-
"""
Baseline models for comparison in ablation study.
"""
from __future__ import annotations
import logging
from typing import Optional, Dict

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import CFG

logger = logging.getLogger(__name__)


class StaticLGBBaseline:
    """Static LightGBM retrained from scratch each fold (no transfer)."""

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or dict(CFG.model.base_params)
        self.model = None
        self.feature_names = []

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        self.feature_names = list(X_train.columns)
        dtrain = lgb.Dataset(X_train.values, label=y_train.values,
                             feature_name=self.feature_names, free_raw_data=False)
        valid_sets = [dtrain]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(period=100)]
        if X_valid is not None:
            dvalid = lgb.Dataset(X_valid.values, label=y_valid.values,
                                 feature_name=self.feature_names, free_raw_data=False)
            valid_sets.append(dvalid)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(50))

        self.model = lgb.train(
            self.params, dtrain,
            num_boost_round=CFG.model.base_num_boost_round,
            valid_sets=valid_sets, valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X):
        return self.model.predict(X.values)

    def predict_binary(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)


class XGBoostBaseline:
    """XGBoost baseline for comparison with LightGBM."""

    def __init__(self, params: Optional[Dict] = None, seed: int = 42):
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 10.0,
            "reg_lambda": 50.0,
            "min_child_weight": 50,
            "random_state": seed,
            "verbosity": 0,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        try:
            import xgboost as xgb
        except ImportError:
            logger.warning("XGBoost not installed — falling back to LGB")
            fallback = StaticLGBBaseline()
            fallback.fit(X_train, y_train, X_valid, y_valid)
            self.model = fallback
            return self

        # XGBoost cannot handle inf values gracefully by default, replace with NaN
        X_train_clean = X_train.replace([np.inf, -np.inf], np.nan)
        dtrain = xgb.DMatrix(X_train_clean.values, label=y_train.values,
                             feature_names=list(X_train_clean.columns))
        evals = [(dtrain, "train")]
        if X_valid is not None:
            X_valid_clean = X_valid.replace([np.inf, -np.inf], np.nan)
            dvalid = xgb.DMatrix(X_valid_clean.values, label=y_valid.values,
                                 feature_names=list(X_valid_clean.columns))
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            self.params, dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100,
        )
        self._feature_names = list(X_train.columns)
        return self

    def predict(self, X):
        if isinstance(self.model, StaticLGBBaseline):
            return self.model.predict(X)
        import xgboost as xgb
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        dtest = xgb.DMatrix(X_clean.values, feature_names=self._feature_names)
        return self.model.predict(dtest)

    def predict_binary(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)


class RandomBaseline:
    """Random prediction baseline for sanity check."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.base_rate = 0.5

    def fit(self, X_train, y_train, **kwargs):
        self.base_rate = y_train.mean()
        return self

    def predict(self, X):
        return self.rng.random(len(X))

    def predict_binary(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)


class BuyAndHoldBaseline:
    """Buy-and-hold benchmark (always predict 'no drop' = 0)."""

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_binary(self, X, threshold=0.5):
        return np.zeros(len(X), dtype=int)
