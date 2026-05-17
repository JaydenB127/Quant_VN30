# -*- coding: utf-8 -*-
"""
Performance metrics for the regime-aware transfer learning strategy.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss,
)
from ..config import CFG


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_proba: np.ndarray = None) -> dict:
    """Compute classification metrics for binary prediction."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc_roc"] = np.nan
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except ValueError:
            metrics["log_loss"] = np.nan
    return metrics


def portfolio_metrics(daily_returns: pd.Series,
                      trading_days: int = 252) -> dict:
    """
    Compute portfolio-level risk/return metrics.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily strategy returns.
    trading_days : int
        Number of trading days per year.

    Returns
    -------
    dict
        Keys: ann_return, ann_vol, sharpe, max_drawdown, calmar, cvar_95, hit_rate.
    """
    if len(daily_returns) == 0:
        return {k: np.nan for k in [
            "ann_return", "ann_vol", "sharpe", "max_drawdown",
            "calmar", "cvar_95", "hit_rate", "total_return"
        ]}

    mean_r = daily_returns.mean()
    std_r = daily_returns.std(ddof=1)
    ann_ret = mean_r * trading_days
    ann_vol = std_r * np.sqrt(trading_days)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # CVaR 95%
    sorted_ret = daily_returns.sort_values()
    n_tail = max(1, int(len(sorted_ret) * 0.05))
    cvar_95 = sorted_ret.iloc[:n_tail].mean()

    hit_rate = (daily_returns > 0).mean()
    total_return = cum.iloc[-1] - 1 if len(cum) > 0 else 0.0

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "cvar_95": cvar_95,
        "hit_rate": hit_rate,
        "total_return": total_return,
    }


def regime_stratified_metrics(
    daily_returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """
    Compute portfolio metrics stratified by regime.

    Returns
    -------
    pd.DataFrame
        Rows = regime labels, columns = metric names.
    """
    results = {}
    for regime in sorted(regime_labels.unique()):
        mask = regime_labels == regime
        regime_rets = daily_returns[mask]
        results[regime] = portfolio_metrics(regime_rets)

    # Add overall
    results["overall"] = portfolio_metrics(daily_returns)

    return pd.DataFrame(results).T
