# -*- coding: utf-8 -*-
"""
Technical indicator features.

ALL indicators are computed on **lagged** data (shift ≥ 1) to prevent
look-ahead bias.  The ``shift`` parameter in every function guarantees
this — it is applied *before* return so that at time T the model only
sees data up to T-1.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import CFG

logger = logging.getLogger(__name__)


# ── Helper: ensure we work per-ticker ─────────────────────────────────

def _per_ticker(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    """Group by ticker level of a MultiIndex DataFrame."""
    return df.groupby(level="ticker")


# ── Momentum ──────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14, lag: int = 1) -> pd.Series:
    """
    Relative Strength Index — lagged.

    Parameters
    ----------
    close : pd.Series
        MultiIndex (date, ticker) close prices.
    period : int
        RSI look-back window.
    lag : int
        Shift to apply for anti-leakage.

    Returns
    -------
    pd.Series
        RSI values in [0, 100], shifted by ``lag``.
    """
    delta = close.groupby(level="ticker").diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    avg_loss = loss.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.groupby(level="ticker").shift(lag)


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    lag: int = 1,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD line, signal line, histogram — lagged.

    Returns
    -------
    tuple of pd.Series
        (macd_line, signal_line, histogram), all shifted by ``lag``.
    """
    ema_fast = close.groupby(level="ticker").transform(
        lambda x: x.ewm(span=fast, adjust=False).mean()
    )
    ema_slow = close.groupby(level="ticker").transform(
        lambda x: x.ewm(span=slow, adjust=False).mean()
    )
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.groupby(level="ticker").transform(
        lambda x: x.ewm(span=signal, adjust=False).mean()
    )
    histogram = macd_line - signal_line

    shift = lambda s: s.groupby(level="ticker").shift(lag)
    return shift(macd_line), shift(signal_line), shift(histogram)


def rate_of_change(close: pd.Series, period: int = 10, lag: int = 1) -> pd.Series:
    """Rate of Change = (close / close_n_ago - 1), lagged."""
    roc = close.groupby(level="ticker").transform(
        lambda x: x.pct_change(periods=period)
    )
    return roc.groupby(level="ticker").shift(lag)


def stochastic_k(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    lag: int = 1,
) -> pd.Series:
    """Stochastic %K oscillator, lagged."""
    lowest = low.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).min()
    )
    highest = high.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).max()
    )
    k = 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    return k.groupby(level="ticker").shift(lag)


# ── Volatility ────────────────────────────────────────────────────────

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    lag: int = 1,
) -> pd.Series:
    """Average True Range, lagged."""
    prev_close = close.groupby(level="ticker").shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = true_range.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    return atr_val.groupby(level="ticker").shift(lag)


def bollinger_pct_b(
    close: pd.Series,
    period: int = 20,
    n_std: float = 2.0,
    lag: int = 1,
) -> pd.Series:
    """Bollinger %B = (close - lower) / (upper - lower), lagged."""
    ma = close.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    std = close.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).std()
    )
    upper = ma + n_std * std
    lower = ma - n_std * std
    pct_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pct_b.groupby(level="ticker").shift(lag)


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    lag: int = 1,
) -> pd.Series:
    """Garman-Klass volatility estimator, lagged."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

    gk_vol = gk.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    return np.sqrt(gk_vol).groupby(level="ticker").shift(lag)


def realized_vol(
    returns: pd.Series,
    period: int = 20,
    lag: int = 1,
) -> pd.Series:
    """Rolling realized volatility (std of returns), lagged."""
    vol = returns.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).std()
    )
    return vol.groupby(level="ticker").shift(lag)


# ── Volume ────────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series, lag: int = 1) -> pd.Series:
    """On-Balance Volume, normalized and lagged."""
    direction = np.sign(close.groupby(level="ticker").diff())
    signed_vol = direction * volume
    obv_val = signed_vol.groupby(level="ticker").cumsum()
    # Normalize per ticker to [-1, 1] range over rolling window
    obv_norm = obv_val.groupby(level="ticker").transform(
        lambda x: (x - x.rolling(60, min_periods=20).mean())
        / x.rolling(60, min_periods=20).std().replace(0, np.nan)
    )
    return obv_norm.groupby(level="ticker").shift(lag)


def volume_ma_ratio(
    volume: pd.Series,
    period: int = 20,
    lag: int = 1,
) -> pd.Series:
    """Volume / MA(volume), lagged."""
    vol_ma = volume.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    ratio = volume / vol_ma.replace(0, np.nan)
    return ratio.groupby(level="ticker").shift(lag)


def vwap_ratio(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 20,
    lag: int = 1,
) -> pd.Series:
    """Close / VWAP ratio, lagged. Approximation using typical price."""
    typical = (high + low + close) / 3.0
    vwap_num = (typical * volume).groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).sum()
    )
    vwap_den = volume.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).sum()
    )
    vwap = vwap_num / vwap_den.replace(0, np.nan)
    ratio = close / vwap
    return ratio.groupby(level="ticker").shift(lag)


# ── Trend ─────────────────────────────────────────────────────────────

def ema(close: pd.Series, span: int = 20, lag: int = 1) -> pd.Series:
    """EMA of close, lagged."""
    ema_val = close.groupby(level="ticker").transform(
        lambda x: x.ewm(span=span, adjust=False).mean()
    )
    return ema_val.groupby(level="ticker").shift(lag)


def ema_crossover(
    close: pd.Series,
    fast_span: int = 20,
    slow_span: int = 50,
    lag: int = 1,
) -> pd.Series:
    """EMA crossover signal: (fast_ema - slow_ema) / close, lagged."""
    fast = close.groupby(level="ticker").transform(
        lambda x: x.ewm(span=fast_span, adjust=False).mean()
    )
    slow = close.groupby(level="ticker").transform(
        lambda x: x.ewm(span=slow_span, adjust=False).mean()
    )
    cross = (fast - slow) / close.replace(0, np.nan)
    return cross.groupby(level="ticker").shift(lag)


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    lag: int = 1,
) -> pd.Series:
    """Average Directional Index (simplified), lagged."""
    prev_high = high.groupby(level="ticker").shift(1)
    prev_low = low.groupby(level="ticker").shift(1)
    prev_close = close.groupby(level="ticker").shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    # When both are positive, keep only the larger
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0)
    minus_dm = minus_dm.where(~mask, 0)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_val = tr.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    plus_di = 100 * plus_dm.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    ) / atr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    ) / atr_val.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    return adx_val.groupby(level="ticker").shift(lag)


# ── Mean Reversion ────────────────────────────────────────────────────

def zscore_vs_ma(
    close: pd.Series,
    period: int = 20,
    lag: int = 1,
) -> pd.Series:
    """Z-score of close vs its moving average, lagged."""
    ma = close.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).mean()
    )
    std = close.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=period).std()
    )
    z = (close - ma) / std.replace(0, np.nan)
    return z.groupby(level="ticker").shift(lag)


def distance_from_high(
    close: pd.Series,
    period: int = 252,
    lag: int = 1,
) -> pd.Series:
    """(close - 52w_high) / 52w_high, lagged. Always ≤ 0."""
    high_52w = close.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=min(60, period)).max()
    )
    dist = (close - high_52w) / high_52w.replace(0, np.nan)
    return dist.groupby(level="ticker").shift(lag)


def distance_from_low(
    close: pd.Series,
    period: int = 252,
    lag: int = 1,
) -> pd.Series:
    """(close - 52w_low) / 52w_low, lagged. Always ≥ 0."""
    low_52w = close.groupby(level="ticker").transform(
        lambda x: x.rolling(period, min_periods=min(60, period)).min()
    )
    dist = (close - low_52w) / low_52w.replace(0, np.nan)
    return dist.groupby(level="ticker").shift(lag)


# ── Assemble all technical features ───────────────────────────────────

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical features for a MultiIndex (date, ticker) DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Processed OHLCV with columns [open, high, low, close, volume, returns].

    Returns
    -------
    pd.DataFrame
        Original columns + ~40 technical features.
    """
    cfg = CFG.features
    lag = cfg.min_lag

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    vol = df["volume"]
    rets = df["returns"]

    features = {}

    # ── Momentum
    features["rsi_14"] = rsi(close, period=cfg.rsi_period, lag=lag)
    macd_l, macd_s, macd_h = macd(
        close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal, lag=lag
    )
    features["macd_line"] = macd_l
    features["macd_signal"] = macd_s
    features["macd_hist"] = macd_h
    for p in cfg.roc_periods:
        features[f"roc_{p}"] = rate_of_change(close, period=p, lag=lag)
    features["stoch_k"] = stochastic_k(high, low, close, lag=lag)

    # ── Volatility
    features["atr_14"] = atr(high, low, close, period=cfg.atr_period, lag=lag)
    features["boll_pctb"] = bollinger_pct_b(
        close, period=cfg.bollinger_period, n_std=cfg.bollinger_std, lag=lag
    )
    features["gk_vol"] = garman_klass_vol(open_, high, low, close, lag=lag)
    features["realized_vol_20"] = realized_vol(rets, period=20, lag=lag)

    # ── Volume
    features["obv_z"] = obv(close, vol, lag=lag)
    features["vol_ma_ratio"] = volume_ma_ratio(vol, period=cfg.volume_ma_period, lag=lag)
    features["vwap_ratio"] = vwap_ratio(close, high, low, vol, lag=lag)

    # ── Trend
    for span in cfg.ema_periods:
        features[f"ema_{span}"] = ema(close, span=span, lag=lag)
    features["ema_cross_20_50"] = ema_crossover(close, 20, 50, lag=lag)
    features["ema_cross_50_200"] = ema_crossover(close, 50, 200, lag=lag)
    features["adx_14"] = adx(high, low, close, period=cfg.adx_period, lag=lag)

    # ── Mean reversion
    features["zscore_20"] = zscore_vs_ma(close, period=20, lag=lag)
    features["zscore_60"] = zscore_vs_ma(close, period=60, lag=lag)
    features["dist_52w_high"] = distance_from_high(close, period=252, lag=lag)
    features["dist_52w_low"] = distance_from_low(close, period=252, lag=lag)

    # ── Lagged returns (additional momentum signals)
    for d in [1, 2, 3, 5, 10, 20]:
        features[f"ret_lag_{d}"] = rets.groupby(level="ticker").shift(lag + d - 1)

    feat_df = pd.DataFrame(features, index=df.index)

    logger.info("Computed %d technical features", len(features))
    return pd.concat([df, feat_df], axis=1)
