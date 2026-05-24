# -*- coding: utf-8 -*-
"""Tests for model training and prediction."""
import numpy as np
import pandas as pd
import pytest


def _make_feature_data(n=500, n_features=10, seed=42):
    """Create synthetic feature + label data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["A", "B", "C"]

    rows = []
    for ticker in tickers:
        for date in dates:
            row = {"date": date, "ticker": ticker}
            for j in range(n_features):
                row[f"feat_{j}"] = rng.randn()
            row["label"] = rng.binomial(1, 0.3)
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    features = [f"feat_{j}" for j in range(n_features)]
    return df, features


class TestBaseLGB:
    """Test base LightGBM model."""

    def test_fit_predict(self):
        from vn_regime_transfer.model.base_lgb import BaseLGBModel

        df, feat_cols = _make_feature_data(n=200)
        X = df[feat_cols]
        y = df["label"]

        split = int(len(X) * 0.7)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test = X.iloc[split:]

        model = BaseLGBModel(num_boost_round=50, early_stopping=10)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        assert preds.shape[0] == len(X_test)
        assert (preds >= 0).all() and (preds <= 1).all()

    def test_feature_importance(self):
        from vn_regime_transfer.model.base_lgb import BaseLGBModel

        df, feat_cols = _make_feature_data(n=200)
        model = BaseLGBModel(num_boost_round=30)
        model.fit(df[feat_cols], df["label"])

        imp = model.feature_importance()
        assert len(imp) == len(feat_cols)


class TestTransferLGB:
    """Test transfer learning fine-tuning."""

    def test_finetune_from_base(self):
        from vn_regime_transfer.model.base_lgb import BaseLGBModel
        from vn_regime_transfer.model.transfer_lgb import TransferLGBModel

        df, feat_cols = _make_feature_data(n=300)
        X = df[feat_cols]
        y = df["label"]

        split = int(len(X) * 0.6)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_ft, y_ft = X.iloc[split:], y.iloc[split:]

        base = BaseLGBModel(num_boost_round=50)
        base.fit(X_train, y_train)
        n_trees_base = base.model.num_trees()

        transfer = TransferLGBModel(base, ft_num_boost_round=20)
        transfer.finetune(X_ft, y_ft)

        # Transfer model should have MORE trees than base
        assert transfer.model.num_trees() > n_trees_base

        preds = transfer.predict(X_ft)
        assert preds.shape[0] == len(X_ft)


class TestEnsemble:
    """Test ensemble model."""

    def test_ensemble_prediction(self):
        from vn_regime_transfer.model.base_lgb import BaseLGBModel
        from vn_regime_transfer.model.transfer_lgb import TransferLGBModel
        from vn_regime_transfer.model.ensemble import EnsembleModel

        df, feat_cols = _make_feature_data(n=300)
        X = df[feat_cols]
        y = df["label"]

        split = int(len(X) * 0.6)

        base = BaseLGBModel(num_boost_round=30)
        base.fit(X.iloc[:split], y.iloc[:split])

        transfer = TransferLGBModel(base, ft_num_boost_round=10)
        transfer.finetune(X.iloc[split:], y.iloc[split:])

        ensemble = EnsembleModel(base, transfer, w_base=0.6, w_adapted=0.4)
        preds = ensemble.predict(X.iloc[split:])

        assert preds.shape[0] == len(X.iloc[split:])
        assert (preds >= 0).all() and (preds <= 1).all()
