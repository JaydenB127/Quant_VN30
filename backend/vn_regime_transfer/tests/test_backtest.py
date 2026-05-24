# -*- coding: utf-8 -*-
"""Tests for backtest execution constraints."""
import numpy as np
import pandas as pd
import pytest

from vn_regime_transfer.validation.metrics import portfolio_metrics, classification_metrics
from vn_regime_transfer.validation.statistical_tests import (
    bootstrap_confidence_interval,
    population_stability_index,
    diebold_mariano_test,
    diebold_mariano_returns_test,
)


class TestMetrics:
    def test_portfolio_metrics(self):
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 252))
        m = portfolio_metrics(returns)
        assert "sharpe" in m
        assert "max_drawdown" in m
        assert m["max_drawdown"] <= 0

    def test_classification_metrics(self):
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9, 0.6])
        m = classification_metrics(y_true, y_pred, y_proba)
        assert "auc_roc" in m
        assert 0 <= m["accuracy"] <= 1


class TestStatisticalTests:
    def test_bootstrap_ci(self):
        data = np.random.RandomState(42).normal(0.01, 0.02, 100)
        result = bootstrap_confidence_interval(data, n_bootstrap=1000)
        assert result["ci_lower"] < result["ci_upper"]
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_psi(self):
        ref = np.random.RandomState(42).normal(0, 1, 1000)
        cur = np.random.RandomState(42).normal(0.5, 1, 1000)  # shifted
        psi = population_stability_index(ref, cur)
        assert psi > 0
        # Same distribution should have low PSI
        psi_same = population_stability_index(ref, ref)
        assert psi_same < psi

    def test_diebold_mariano_brier(self):
        """DM test with Brier score for binary classification."""
        rng = np.random.RandomState(42)
        actual = rng.binomial(1, 0.3, 200).astype(float)
        pred1 = np.clip(actual + rng.randn(200) * 0.1, 0, 1)
        pred2 = np.clip(actual + rng.randn(200) * 0.4, 0, 1)
        result = diebold_mariano_test(actual, pred1, pred2, loss="brier")
        assert "dm_statistic" in result
        assert "p_value" in result
        assert result["loss_used"] == "brier"
        assert result["mean_loss1"] < result["mean_loss2"]  # pred1 is closer

    def test_diebold_mariano_logloss(self):
        """DM test with log loss for calibration-sensitive comparison."""
        rng = np.random.RandomState(42)
        actual = rng.binomial(1, 0.3, 200).astype(float)
        pred1 = np.clip(actual + rng.randn(200) * 0.1, 0.01, 0.99)
        pred2 = np.clip(actual + rng.randn(200) * 0.4, 0.01, 0.99)
        result = diebold_mariano_test(actual, pred1, pred2, loss="logloss")
        assert result["loss_used"] == "logloss"
        assert result["mean_loss1"] < result["mean_loss2"]

    def test_diebold_mariano_returns(self):
        """DM test on strategy daily returns (P&L series)."""
        rng = np.random.RandomState(42)
        returns1 = rng.normal(0.001, 0.02, 252)  # better strategy
        returns2 = rng.normal(-0.0005, 0.02, 252)  # worse strategy
        result = diebold_mariano_returns_test(returns1, returns2)
        assert "dm_statistic" in result
        assert "p_value" in result
        assert "sharpe_diff" in result
        assert result["mean_return1"] > result["mean_return2"]

    def test_diebold_mariano_rejects_unknown_loss(self):
        """Unknown loss function should raise ValueError."""
        rng = np.random.RandomState(42)
        actual = rng.randn(50)
        pred1 = rng.randn(50)
        pred2 = rng.randn(50)
        with pytest.raises(ValueError, match="Unknown loss"):
            diebold_mariano_test(actual, pred1, pred2, loss="invalid")

    def test_diebold_mariano_squared_is_brier_alias(self):
        """loss='squared' should be accepted as alias for 'brier'."""
        rng = np.random.RandomState(42)
        actual = rng.binomial(1, 0.3, 100).astype(float)
        pred1 = np.clip(actual + rng.randn(100) * 0.1, 0, 1)
        pred2 = np.clip(actual + rng.randn(100) * 0.3, 0, 1)
        result = diebold_mariano_test(actual, pred1, pred2, loss="squared")
        assert result["loss_used"] == "brier"  # renamed internally
