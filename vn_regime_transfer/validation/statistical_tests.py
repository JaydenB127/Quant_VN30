# -*- coding: utf-8 -*-
"""
Statistical tests: Diebold-Mariano, Bootstrap CI, PSI, KS test.

Changes from previous version:
  - Added explicit ``loss="brier"`` for classification (was misleadingly called "squared")
  - Added ``loss="logloss"`` for calibration-sensitive comparison
  - Added ``diebold_mariano_returns_test()`` for comparing strategy P&L series
  - Clarified docstrings with interpretation guidelines
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Diebold-Mariano Test
# ═══════════════════════════════════════════════════════════════════════

def diebold_mariano_test(
    actual: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    h: int = 1,
    loss: str = "brier",
) -> dict:
    """
    Diebold-Mariano test for comparing two forecasts.

    H0: Both forecasts have equal predictive accuracy.
    H1: pred1 is more accurate than pred2 (one-sided).

    Parameters
    ----------
    actual : np.ndarray
        Actual values (binary labels for classification, continuous for regression).
    pred1 : np.ndarray
        Predictions from model 1 (expected to be better).
    pred2 : np.ndarray
        Predictions from model 2.
    h : int
        Forecast horizon (for HAC variance correction).
    loss : str
        Loss function to use:
        - ``"brier"``   : (actual - pred)² — proper scoring rule for binary classification.
                          Equivalent to MSE. Measures calibration + discrimination.
        - ``"logloss"`` : -[y·log(p) + (1-y)·log(1-p)] — more sensitive to confident
                          wrong predictions. Preferred when calibration matters.
        - ``"squared"``  : Alias for "brier" (kept for backward compatibility).
        - ``"absolute"`` : |actual - pred| — MAE loss for regression tasks.

    Returns
    -------
    dict
        ``dm_statistic`` : float — test statistic (negative = pred1 better)
        ``p_value``      : float — one-sided p-value
        ``loss_used``    : str   — which loss function was applied
        ``mean_loss1``   : float — mean loss for pred1
        ``mean_loss2``   : float — mean loss for pred2

    Interpretation
    --------------
    - p_value < 0.05 → pred1 is significantly better than pred2
    - For classification: use "brier" (default) or "logloss"
    - For strategy comparison: use ``diebold_mariano_returns_test()`` instead
    """
    actual = np.asarray(actual, dtype=np.float64)
    pred1 = np.asarray(pred1, dtype=np.float64)
    pred2 = np.asarray(pred2, dtype=np.float64)

    if loss in ("squared", "brier"):
        e1 = (actual - pred1) ** 2
        e2 = (actual - pred2) ** 2
        loss_name = "brier"
    elif loss == "logloss":
        eps = 1e-15
        p1 = np.clip(pred1, eps, 1 - eps)
        p2 = np.clip(pred2, eps, 1 - eps)
        e1 = -(actual * np.log(p1) + (1 - actual) * np.log(1 - p1))
        e2 = -(actual * np.log(p2) + (1 - actual) * np.log(1 - p2))
        loss_name = "logloss"
    elif loss == "absolute":
        e1 = np.abs(actual - pred1)
        e2 = np.abs(actual - pred2)
        loss_name = "absolute"
    else:
        raise ValueError(
            f"Unknown loss: {loss!r}. Use 'brier', 'logloss', or 'absolute'."
        )

    d = e1 - e2  # loss differential: negative means pred1 is better
    n = len(d)

    if n < 2:
        return {
            "dm_statistic": np.nan, "p_value": np.nan,
            "loss_used": loss_name, "mean_loss1": np.nan, "mean_loss2": np.nan,
        }

    mean_d = d.mean()

    # Autocovariance-adjusted variance (Newey-West style for h-step ahead)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        if len(d[k:]) > 1:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += gamma_k

    var_d = (gamma_0 + 2 * gamma_sum) / n
    if var_d <= 0:
        return {
            "dm_statistic": np.nan, "p_value": np.nan,
            "loss_used": loss_name,
            "mean_loss1": float(e1.mean()), "mean_loss2": float(e2.mean()),
        }

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = float(1 - stats.norm.cdf(dm_stat))  # one-sided

    return {
        "dm_statistic": float(dm_stat),
        "p_value": p_value,
        "loss_used": loss_name,
        "mean_loss1": float(e1.mean()),
        "mean_loss2": float(e2.mean()),
    }


def diebold_mariano_returns_test(
    returns1: np.ndarray,
    returns2: np.ndarray,
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test on strategy daily returns (P&L series).

    Compares whether two trading strategies have significantly different
    risk-adjusted returns. This is the correct test for a trading paper
    because it directly tests the economic value, not just prediction accuracy.

    H0: Both strategies have equal expected daily returns.
    H1: Strategy 1 has higher expected daily returns.

    Parameters
    ----------
    returns1 : np.ndarray
        Daily returns from strategy 1 (expected to be better).
    returns2 : np.ndarray
        Daily returns from strategy 2.
    h : int
        Forecast horizon for HAC correction.

    Returns
    -------
    dict
        ``dm_statistic`` : float — positive means strategy 1 is better
        ``p_value``      : float — one-sided p-value
        ``mean_return1``  : float — mean daily return of strategy 1
        ``mean_return2``  : float — mean daily return of strategy 2
        ``sharpe_diff``   : float — annualized Sharpe difference

    Interpretation
    --------------
    - p_value < 0.05 → strategy 1 generates significantly higher returns
    - This tests the economic hypothesis directly, not just statistical accuracy
    """
    r1 = np.asarray(returns1, dtype=np.float64)
    r2 = np.asarray(returns2, dtype=np.float64)

    # Align lengths
    n = min(len(r1), len(r2))
    r1, r2 = r1[:n], r2[:n]

    if n < 2:
        return {
            "dm_statistic": np.nan, "p_value": np.nan,
            "mean_return1": np.nan, "mean_return2": np.nan,
            "sharpe_diff": np.nan,
        }

    d = r1 - r2  # return differential
    mean_d = d.mean()

    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0.0
    for k in range(1, h):
        if len(d[k:]) > 1:
            gamma_k = np.cov(d[k:], d[:-k])[0, 1]
            gamma_sum += gamma_k

    var_d = (gamma_0 + 2 * gamma_sum) / n
    if var_d <= 0:
        return {
            "dm_statistic": np.nan, "p_value": np.nan,
            "mean_return1": float(r1.mean()), "mean_return2": float(r2.mean()),
            "sharpe_diff": np.nan,
        }

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = float(1 - stats.norm.cdf(dm_stat))

    # Annualized Sharpe difference
    s1 = r1.mean() / r1.std() * np.sqrt(252) if r1.std() > 0 else 0.0
    s2 = r2.mean() / r2.std() * np.sqrt(252) if r2.std() > 0 else 0.0

    return {
        "dm_statistic": float(dm_stat),
        "p_value": p_value,
        "mean_return1": float(r1.mean()),
        "mean_return2": float(r2.mean()),
        "sharpe_diff": float(s1 - s2),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Bootstrap
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn=np.mean,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap.
    statistic_fn : callable
        Function computing the statistic (e.g. np.mean, np.median).
    n_bootstrap : int
        Number of bootstrap samples.
    ci : float
        Confidence level (e.g. 0.95).
    seed : int
        Random seed.

    Returns
    -------
    dict
        {"point_estimate", "ci_lower", "ci_upper", "std_error"}
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        boot_stats[i] = statistic_fn(sample)

    alpha = 1 - ci
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "point_estimate": float(statistic_fn(data)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "std_error": float(boot_stats.std()),
    }


def sharpe_bootstrap_ci(
    daily_returns: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    trading_days: int = 252,
    seed: int = 42,
) -> dict:
    """Bootstrap CI specifically for Sharpe ratio."""
    def sharpe_fn(rets):
        if rets.std() == 0:
            return 0.0
        return rets.mean() / rets.std() * np.sqrt(trading_days)

    return bootstrap_confidence_interval(
        daily_returns, statistic_fn=sharpe_fn,
        n_bootstrap=n_bootstrap, ci=ci, seed=seed,
    )


# ═══════════════════════════════════════════════════════════════════════
#  PSI — Population Stability Index
# ═══════════════════════════════════════════════════════════════════════

def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index for distribution drift detection.

    PSI < 0.10 → no significant shift
    0.10 ≤ PSI < 0.25 → moderate shift
    PSI ≥ 0.25 → significant shift

    Parameters
    ----------
    reference : np.ndarray
        Reference distribution (training period).
    current : np.ndarray
        Current distribution (test period).
    n_bins : int
        Number of bins.

    Returns
    -------
    float
        PSI value.
    """
    eps = 1e-6

    # Create bins from reference
    _, bin_edges = np.histogram(reference, bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current, bins=bin_edges)[0]

    ref_pct = ref_counts / len(reference) + eps
    cur_pct = cur_counts / len(current) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_feature_psi(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.25,
) -> pd.DataFrame:
    """
    Compute PSI for all features.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, psi, drifted]
    """
    results = []
    for col in X_train.columns:
        ref = X_train[col].dropna().values
        cur = X_test[col].dropna().values
        if len(ref) < 10 or len(cur) < 10:
            continue
        psi_val = population_stability_index(ref, cur)
        results.append({
            "feature": col, "psi": psi_val,
            "drifted": psi_val >= threshold,
        })

    df = pd.DataFrame(results).sort_values("psi", ascending=False)
    n_drifted = df["drifted"].sum()
    logger.info("PSI check: %d/%d features drifted (threshold=%.2f)",
                n_drifted, len(df), threshold)
    return df


# ═══════════════════════════════════════════════════════════════════════
#  KS Test
# ═══════════════════════════════════════════════════════════════════════

def ks_test_returns(returns_a: np.ndarray, returns_b: np.ndarray) -> dict:
    """
    KS test: do two return distributions differ?

    Returns
    -------
    dict
        {"ks_statistic": float, "p_value": float}
    """
    stat, p = stats.ks_2samp(returns_a, returns_b)
    return {"ks_statistic": float(stat), "p_value": float(p)}
